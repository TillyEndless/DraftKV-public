"""Expose the source-locked H3 SGF video seam to the shared HookSession.

The upstream processor still owns projection, QK normalization, Ulysses,
condition/audio attention, output PRoPE, projection and residual math.
"""

import inspect
from contextlib import nullcontext

from .contracts import BackboneError
from .solarwm_geometry import NativeGeometry

def compact_prefix(raw_key, fixed_prefix, layout, geometry, apply_rotary):
    """Keep original prefix positions/cameras when video time becomes compact."""
    prefix = fixed_prefix[0].shape[1]
    key = fixed_prefix[0].clone()
    unshifted = apply_rotary(raw_key[:, :prefix], *geometry.rope(layout.position_ids[:prefix]))
    key[..., :96] = unshifted[..., :96]
    return key.contiguous(), fixed_prefix[1]


def bind(core, session):
    import torch
    from solarwm.backends.minimax_h3 import sgf_attention as native
    from solarwm.backends.minimax_h3 import sgf_rollout as rollout
    from solarwm.backends.minimax_h3.torch_prope import prope_qkv_separate

    original_attention = native.sgf_attention
    original_forward = rollout.h3_student_forward
    if (
        set(inspect.signature(original_attention).parameters)
        != {"q", "k", "v", "control", "layer", "apply_rotary", "attend"}
        or "commit_cache" not in inspect.signature(original_forward).parameters
    ):
        raise BackboneError("SOURCE_ABI_DRIFT", "H3 SGF native seam changed")
    core.session = session
    algorithm = getattr(getattr(session, "registration", None), "algorithm", None)
    plugin_video = algorithm in {"worldtrace-landmark", "draftkv", "infllm", "deep-forcing", "future-forcing"}
    bypass_native_video = algorithm is not None and not getattr(session, "native_observer", False)
    if bypass_native_video:
        from draftkv.algorithms.worldtrace_landmark.solarwm import native_prefix_attention
        prefix_attention = native_prefix_attention(original_attention)

    def attention(q, k, v, *, control, layer, apply_rotary, attend):
        if control.mode != "rollout" or session.active is None:
            raise BackboneError(
                "SOURCE_ABI_DRIFT", "released H3 requires transactional SGF rollout"
            )
        geometry = NativeGeometry(core.target, control)
        core.active_geometry = geometry
        layout = control.layout
        condition = int(layout.condition_indices.numel())
        prefix = condition + int(layout.audio_indices.numel())
        rows, start = geometry.rows, control.chunk_index * 5
        count = 5 * rows
        if q.shape[1] != prefix + count or q.shape[2:] != (
            core.spec.heads // core.plan["sp"],
            128,
        ):
            raise BackboneError(
                "SOURCE_ABI_DRIFT", "H3 native token/head layout changed"
            )
        history = () if plugin_video else control.cache.history(layer, control.chunk_index)
        first = max(0, start - 25)
        video_first = start if plugin_video else first
        ids = torch.arange(video_first * rows, (start + 5) * rows, device=q.device)
        current_ids = ids[-count:]
        raw_k = torch.cat([entry[1] for entry in history] + [k[:, prefix:]], 1)
        raw_v = torch.cat([entry[2] for entry in history] + [v[:, prefix:]], 1)

        def cameras(a, b):
            return (
                control.viewmats[:, 1 + a : 1 + b].repeat_interleave(rows, 1),
                control.intrinsics[:, 1 + a : 1 + b].repeat_interleave(rows, 1),
            )

        qcam, kcam = cameras(start, start + 5), cameras(video_first, start + 5)
        query = apply_rotary(
            q[:, prefix:], *geometry.rope(geometry.positions[current_ids])
        )
        key = apply_rotary(raw_k, *geometry.rope(geometry.positions[ids]))
        query, key, value, output_basis = prope_qkv_separate(
            query.transpose(1, 2),
            key.transpose(1, 2),
            raw_v.transpose(1, 2),
            query_viewmats=qcam[0],
            query_K=qcam[1],
            key_viewmats=kcam[0],
            key_K=kcam[1],
        )
        query, key, value = (
            t.transpose(1, 2).contiguous() for t in (query, key, value)
        )
        raw_query, raw_current_key, _, _ = prope_qkv_separate(
            q[:, prefix:].transpose(1, 2),
            k[:, prefix:].transpose(1, 2),
            v[:, prefix:].transpose(1, 2),
            query_viewmats=qcam[0],
            query_K=qcam[1],
            key_viewmats=qcam[0],
            key_K=qcam[1],
        )
        # A common temporal translation also applies to condition K. Prefix
        # queries are separate native segments and keep their native basis.
        positions = layout.position_ids[:prefix].clone()
        positions[:, 0] += first // 5 * (85.0 / 3.0)
        pk = apply_rotary(k[:, :prefix], *geometry.rope(positions))
        pv = v[:, :prefix]
        view = torch.eye(4, device=q.device).repeat(q.shape[0], prefix, 1, 1)
        K = torch.eye(3, device=q.device).repeat(q.shape[0], prefix, 1, 1)
        view[:, layout.condition_video_indices] = control.viewmats[:, :1]
        K[:, layout.condition_video_indices] = control.intrinsics[:, :1]
        _, pk, pv, _ = prope_qkv_separate(
            query.transpose(1, 2),
            pk.transpose(1, 2),
            pv.transpose(1, 2),
            query_viewmats=qcam[0],
            query_K=qcam[1],
            key_viewmats=view,
            key_K=K,
        )
        fixed_prefix = tuple(t.transpose(1, 2).contiguous() for t in (pk, pv))
        if getattr(getattr(session, 'registration', None), 'algorithm', None) == 'draftkv':
            # Chrono-Compact video slots use the native clock starting at zero.
            # Derive the condition/audio prefix from raw K at its original
            # layout positions, not by inverting the rounded translated K.
            geometry.chrono_compact_prefix = compact_prefix(
                k, fixed_prefix, layout, geometry, apply_rotary)
        calls = 0
        expected_prefix_calls = 1 + int(prefix > condition)

        def consumer(nq, nk, nv):
            nonlocal calls
            calls += 1
            if calls <= expected_prefix_calls:
                return attend(nq, nk, nv)
            if calls != expected_prefix_calls + 1:
                raise BackboneError(
                    "SOURCE_ABI_DRIFT", "unexpected H3 video attention segment"
                )
            core.trace(
                "solarwm_native_geometry",
                layer=layer,
                start_frame=start,
                position_policy=geometry.policy,
                fixed_prefix_tokens=prefix,
                frame_tokens=rows,
                spatial_grid=[24, 42],
                coordinate_translation=first // 5 * (85.0 / 3.0),
            )
            return session.attend(
                layer,
                query,
                key[:, -count:].contiguous(),
                value[:, -count:].contiguous(),
                key,
                value,
                raw_query.transpose(1, 2).contiguous(),
                raw_current_key.transpose(1, 2).contiguous(),
                rows,
                torch.tensor([[5, 24, 42]], device=q.device),
                geometry,
                (lambda *unused: attend(nq, nk, nv))
                if getattr(session, "native_observer", False)
                else attend,
                fixed_prefix=fixed_prefix,
                native_geometry=geometry,
            )

        if bypass_native_video:
            def video():
                output = consumer(query, key, value)
                return output_basis(output.transpose(1, 2)).transpose(1, 2)
            result = prefix_attention(q, k, v, control=control, layer=layer,
                apply_rotary=apply_rotary, attend=consumer, _worldtrace_video=video)
            if not plugin_video and control.commit_cache:
                # Warmup-compatible routes still need exact native raw history.
                # The removed native video loop used to perform this commit.
                control.cache.commit(layer, control.chunk_index, k[:, prefix:], v[:, prefix:])
        else:
            result = original_attention(q, k, v, control=control, layer=layer,
                apply_rotary=apply_rotary, attend=consumer)
        if calls != expected_prefix_calls + 1:
            raise BackboneError(
                "PARTIAL_NATIVE_WIRING", "H3 video consumer was not reached"
            )
        return result

    native.sgf_attention = attention
    bind_transactions(core, session)
    core._solarwm_original_attention = original_attention
    core._solarwm_original_forward = original_forward
    core.trace(
        "solarwm_hook_bound",
        native_attention="sgf_attention",
        transaction="whole_h3_student_forward",
        clean="commit_cache_not_timestep_zero",
        geometry="h3_native_mmrope_camera_prope",
        silent_fallback=False,
        video_history_owner=algorithm if plugin_video else "native_and_plugin",
        redundant_native_video_cache=not plugin_video,
        native_compatibility_cache_device="none" if plugin_video else "cpu",
        discarded_native_video_math=not bypass_native_video,
    )


def bind_transactions(core, session=None):
    """Private raw native cache until the complete model forward succeeds."""
    from solarwm.backends.minimax_h3 import sgf_attention as native
    from solarwm.backends.minimax_h3 import sgf_rollout as rollout

    original_forward = rollout.h3_student_forward
    core._solarwm_original_forward = original_forward
    forward_signature = inspect.signature(original_forward)
    if "commit_cache" not in forward_signature.parameters:
        raise BackboneError("SOURCE_ABI_DRIFT", "H3 forward transaction seam changed")

    def forward(*args, **kwargs):
        params = forward_signature.bind(*args, **kwargs)
        params.apply_defaults()
        values = params.arguments
        if values["student"] is not core.model or values["clean"] is not None:
            raise BackboneError("SOURCE_ABI_DRIFT", "unexpected H3 model/replay path")
        cache = values["cache"]
        layers = {layer: list(entries) for layer, entries in cache.layers.items()}
        if session is not None:
            from .solarwm_host_cache import CpuRawKVCache
            private = CpuRawKVCache(layers, core.device)
        else:
            private = native.H3RawKVCache(layers)
        values["cache"] = private
        # H3 clean cache refresh uses t=1, NOT the Wan t=0 convention.
        clean = bool(values["commit_cache"])
        transaction = (
            session.forward(
                start_frame=values["chunk_index"] * 5,
                clean=clean,
                timestep=values["timestep"],
            )
            if session is not None
            else nullcontext()
        )
        with transaction:
            result = original_forward(*params.args, **params.kwargs)
        if clean:
            cache.layers = private.layers
        return result

    rollout.h3_student_forward = forward
