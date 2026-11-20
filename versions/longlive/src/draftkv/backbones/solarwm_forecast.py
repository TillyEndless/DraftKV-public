"""H3 native normalization/projection callbacks for shared InfiniGen.

Selection, partial weights, dynamic cap and prefetch remain in the canonical
InfiniGen provider. This file only extracts and freezes native model math.
"""

import ast
import copy
import hashlib
import inspect
import textwrap
from types import SimpleNamespace

from .contracts import BackboneError


def normalizer(block):
    original = block.forward.__func__
    source = textwrap.dedent(inspect.getsource(original))
    tree = ast.parse(source)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
    calls = [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.Call) and ast.unparse(n.func) == "self.attn"
    ]
    if len(calls) != 1 or ast.unparse(calls[0].args[0]) != "norm_hidden_states":
        raise BackboneError(
            "SOURCE_ABI_DRIFT", "native H3 AdaLN attention seam changed"
        )
    body = [
        copy.deepcopy(n)
        for n in fn.body
        if n.lineno < calls[0].lineno and not isinstance(n, ast.Expr)
    ]
    generated = ast.FunctionDef(
        name="normalize",
        args=ast.arguments(
            posonlyargs=[],
            args=[
                ast.arg(arg=n)
                for n in ("self", "hidden_states", "temb", "adaln_indices")
            ],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=body + [ast.Return(ast.Name("norm_hidden_states", ast.Load()))],
        decorator_list=[],
    )
    namespace = dict(original.__globals__)
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[generated], type_ignores=[])),
            inspect.getsourcefile(original),
            "exec",
        ),
        namespace,
    )
    return namespace["normalize"], hashlib.sha256(source.encode()).hexdigest()


def freeze(blocks):
    profiles = []
    for block in blocks:
        normalize, sha = normalizer(block)
        profile = SimpleNamespace(
            normalize=normalize,
            source_sha=sha,
            signature=inspect.signature(block.forward),
        )
        for name, source in (
            ("norm1", block.norm1),
            ("adaln_proj", block.adaln_proj),
            ("projection", block.attn.to_q),
            ("norm_q", block.attn.norm_q),
        ):
            setattr(
                profile, name, copy.deepcopy(source).cpu().eval().requires_grad_(False)
            )
        profiles.append(profile)
    return profiles


def install(core, hook):
    import torch
    from solarwm.backends.minimax_h3.distributed import sequence_all_to_all
    from solarwm.backends.minimax_h3.torch_prope import prope_qkv_separate

    profiles = core.query_profiles
    core.trace(
        "infinigen_frozen_partial_weight_bootstrap",
        layers=len(profiles),
        before_fsdp=True,
        native_normalization_source_sha256=[p.source_sha for p in profiles],
    )
    for index, block in enumerate(core.blocks[:-1]):

        def after(module, args, kwargs, result, index=index):
            if hook.active is None:
                return
            start = hook.active["start_frame"]
            cache = getattr(hook.modules[index + 1], "_portable_speculation", None)
            if cache is None or cache.banks[0].get("current_start") != start * 1008:
                return
            params = profiles[index].signature.bind(*args, **kwargs).arguments
            p = profiles[index + 1]
            geometry = core.active_geometry
            control = geometry.control
            residual = result
            names = ("norm1", "adaln_proj", "projection", "norm_q")
            for name in names:
                getattr(p, name).to(residual.device)
            with torch.no_grad():
                normalized = p.normalize(
                    p, residual, params["temb"], params["adaln_indices"]
                )
                query = p.norm_q(p.projection(normalized).unflatten(-1, (56, 128)))
            if core.plan["sp"] > 1:
                query = sequence_all_to_all(query, scatter_dim=2, gather_dim=1)
            query = query[:, -5040:]
            ids = torch.arange(start * 1008, (start + 5) * 1008, device=query.device)
            query = geometry.apply(query, *geometry.rope(geometry.positions[ids]))
            views = control.viewmats[:, 1 + start : 1 + start + 5].repeat_interleave(
                1008, 1
            )
            K = control.intrinsics[:, 1 + start : 1 + start + 5].repeat_interleave(
                1008, 1
            )
            query, _, _, _ = prope_qkv_separate(
                query.transpose(1, 2),
                query.transpose(1, 2),
                query.transpose(1, 2),
                query_viewmats=views,
                query_K=K,
                key_viewmats=views,
                key_K=K,
            )
            cache.prefetch(query.contiguous())
            for name in names:
                getattr(p, name).cpu()
            core.trace(
                "infinigen_previous_layer_forecast",
                layer=index + 1,
                start_frame=start,
                clean=bool(hook.active["clean"]),
                native_normalization_source_sha256=p.source_sha,
                index="original_partial_weight_controller",
                fetch="original_worker_stream",
            )

        if not hasattr(core, '_solarwm_forecast_handles'):
            core._solarwm_forecast_handles = []
        core._solarwm_forecast_handles.append(block.register_forward_hook(after, with_kwargs=True))
