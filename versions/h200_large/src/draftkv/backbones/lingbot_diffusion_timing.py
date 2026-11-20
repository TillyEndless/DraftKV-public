"""Opt-in LingBot causal-fast diffusion boundary, independent of KV algorithm.

Source-locked native causal-fast: image VAE encode -> KV construction ->
four denoising forwards + one clean KV commit per chunk -> VAE decode.
Observe only the outer model's post-forward count and the native VAE encode
return; no per-layer/block events and no changes to the upstream generator.
"""
from contextlib import contextmanager
import inspect

from .contracts import BackboneError
from .gpu_profile import diffusion_start, diffusion_end


def expected_native_forwards(core, request):
    """Derive the actual causal-fast outer DiT call count from native geometry."""
    pipeline = core.pipeline
    if getattr(pipeline, "infer_mode", None) != "causal_fast":
        raise BackboneError("SOURCE_ABI_DRIFT", "LingBot profile requires native causal_fast")
    if not callable(getattr(pipeline, "_generate_causal_fast", None)):
        raise BackboneError("SOURCE_ABI_DRIFT", "missing LingBot causal_fast implementation")
    signature = inspect.signature(pipeline.generate)
    param = signature.parameters.get("timesteps_index")
    indices = param.default if param else None
    if (not isinstance(indices, (tuple, list)) or len(indices) != 4
            or any(type(index) is not int for index in indices)):
        raise BackboneError("SOURCE_ABI_DRIFT", "unexpected native LingBot four-step schedule")

    frames = request.get("native_frames")
    chunk_size = core.spec.chunk
    if (type(frames) is not int or frames < 1 or type(chunk_size) is not int
            or chunk_size < 1):
        raise BackboneError("BLOCKED_INPUT_PROTOCOL", "invalid LingBot frame/chunk geometry")
    latent_frames = (frames - 1) // 4 + 1
    chunks = latent_frames // chunk_size  # Native floors to complete latent chunks.
    if chunks < 1:
        raise BackboneError("BLOCKED_INPUT_PROTOCOL", "LingBot has no full native chunk")
    return chunks * (len(indices) + 1)  # Native final t=0 clean commit per chunk.


@contextmanager
def lingbot_diffusion_timing(core, request):
    """Track image-encode completion through the final clean model Forward.

    The hook sees one callback per outer DiT Forward (not each attention
    layer), so it adds no CUDA events or per-chunk synchronization. It also
    works with the WorldKV camera AST bridge: no generator methods are
    replaced or recompiled. Install only with --gpu-profile enabled.
    """
    if not core.plan.get("gpu_profile", False):
        yield
        return

    expected = expected_native_forwards(core, request)
    pipeline = core.pipeline
    vae, model = pipeline.vae, pipeline.model
    original_encode = vae.encode
    had_instance_encode = "encode" in vars(vae)
    instance_encode = vars(vae).get("encode")
    state = {"encoded": 0, "forwards": 0}

    def encode(*args, **kwargs):
        if state["encoded"]:
            raise BackboneError("SOURCE_ABI_DRIFT", "LingBot encoded image more than once")
        result = original_encode(*args, **kwargs)
        diffusion_start()  # VAE encoding and T5 conditioning have finished.
        state["encoded"] = 1
        return result

    def after_forward(module, args, kwargs, output):
        if state["encoded"] != 1:
            raise BackboneError("SOURCE_ABI_DRIFT", "LingBot DiT Forward before native VAE encode")
        state["forwards"] += 1
        if state["forwards"] == expected:
            diffusion_end()  # All denoising steps and clean KV commits complete.
        elif state["forwards"] > expected:
            raise BackboneError("SOURCE_ABI_DRIFT", "unexpected extra LingBot DiT Forward")

    handle = model.register_forward_hook(after_forward, with_kwargs=True)
    try:
        vae.encode = encode
        try:
            yield
        except BaseException:
            raise
        else:
            if state["encoded"] != 1 or state["forwards"] != expected:
                raise BackboneError(
                    "SOURCE_ABI_DRIFT",
                    f"LingBot profile incomplete: image_encode={state['encoded']}, "
                    f"outer_dit_forwards={state['forwards']}, expected={expected}",
                )
    finally:
        if had_instance_encode:
            vae.encode = instance_encode
        else:
            del vae.encode
        handle.remove()
