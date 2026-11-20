"""Delegate pure T2V to the locked native LongLive pipeline and native VAE.

The author pipeline owns scheduler steps, absolute frame offsets, all Forward
calls and t=0 clean cache updates. No image conditioning or new diffusion loop.
"""

from .contracts import BackboneError
import os


def run_native_t2v(pipe, noise, request):
    """Exact native entry, separately testable without loading a checkpoint."""
    if (request.get("input_mode") != "bench1_t2v" or any(request.get(key) is not None for key in
            ("image", "initial_image", "adapter_image", "clean_latent_prefix", "camera", "camera_dir"))
            or request.get("image_conditioning") not in (None, "none")):
        raise BackboneError("BLOCKED_INPUT_PROTOCOL", "LongLive native T2V accepts text only")
    from draftkv.benchmarks.worldattention import validate_schedule
    validate_schedule(request)
    # Native text_prompts is a batch, not a six-event timeline. The bound
    # wrapper selects each of the six pre-encoded samples by current_start.
    # Source-locked upstream owns every denoising/clean Forward and scheduler.
    return pipe.inference(noise=noise, text_prompts=request["prompts"][:1], return_latents=True,
                          profile=False, low_memory=False, skip_vae_decode=True)


def decode_native_t2v(vae, latents, request=None):
    """Call the author decoder selected by the frozen native output receipt.

    The successful 240-latent native entry uses decode_to_pixel. Its AST is
    identical in the locked wrapper. Older diagnostic requests retain the
    locked default chunk decoder; no temporal policy is silently substituted.
    """
    method = (request or {}).get("vae_decode_method", "decode_to_pixel_chunk")
    if method not in ("decode_to_pixel", "decode_to_pixel_chunk"):
        raise BackboneError("BLOCKED_OUTPUT_SPEC", "unbound native T2V VAE decode method")
    return getattr(vae, method)(latents, use_cache=False)


def generate(core, request):
    import inspect
    import torch
    from pathlib import Path
    from pipeline.causal_inference import CausalInferencePipeline
    from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
    from utils.scheduler import FlowMatchScheduler
    from .models import relocated_asset_init

    if (request.get("input_mode") != "bench1_t2v" or any(request.get(key) is not None for key in
            ("image", "initial_image", "adapter_image", "clean_latent_prefix", "camera", "camera_dir"))
            or request.get("image_conditioning") not in (None, "none")):
        raise BackboneError("BLOCKED_INPUT_PROTOCOL", "LongLive native T2V accepts text only")
    if core.plan["algorithm"] == "draftkv" and not core.plan.get("algorithm_plugin"):
        raise BackboneError("BLOCKED_ALGORITHM_SEMANTICS", "native T2V requires the canonical DraftKV HookRegistry binding")
    parameters = inspect.signature(CausalInferencePipeline.inference).parameters
    required = {"noise", "text_prompts", "return_latents", "profile", "low_memory", "skip_vae_decode"}
    if not required <= set(parameters):
        raise BackboneError("SOURCE_ABI_DRIFT", "locked native T2V inference ABI changed")
    device = next(core.model.parameters()).device
    torch.manual_seed(request["seed"])
    # Reuse the actually loaded base+LoRA model, without constructing a second
    # generator. This is the historical successful native runner's wrapper.
    wrapper = WanDiffusionWrapper.__new__(WanDiffusionWrapper)
    torch.nn.Module.__init__(wrapper)
    wrapper.model, wrapper.uniform_timestep, wrapper.seq_len = core.model, False, 32760
    wrapper.scheduler = FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True)
    wrapper.scheduler.set_timesteps(1000, training=True)
    wrapper.post_init()

    class AttrDict(dict):
        def __getattr__(self, key):
            try:
                return self[key]
            except KeyError as error:
                raise AttributeError(key) from error

    class PreEncodedText(torch.nn.Module):
        def __init__(self, conditioning, prompts):
            super().__init__()
            self.conditioning, self.prompts = conditioning, prompts

        def forward(self, text_prompts):
            if text_prompts != self.prompts:
                raise BackboneError("BLOCKED_INPUT_PROTOCOL", "native pipeline changed the frozen initial prompt")
            return self.conditioning

    config = AttrDict(denoising_step_list=[1000, 750, 500, 250], warp_denoising_step=True,
                     num_frame_per_block=core.spec.chunk, context_noise=0.0,
                     model_kwargs=AttrDict(local_attn_size=core.spec.window, sink_size=core.spec.sink,
                                           memory_size=0, compression_method="avg_pool", recent_exclude=0))
    with torch.no_grad():
        # The original encoder constructs FP32 T5 on CUDA. Stage DiT out
        # during encoding so the two full models do not exhaust a 24GB card.
        # Copies preserve weights/RNG; encoder precision and native calls stay
        # identical. The same wrapper still owns the same model object.
        core.model.cpu()
        torch.cuda.empty_cache()
        core.trace("native_text_memory_stage", dit_device="cpu", text_dtype="float32")
        text_encoder = relocated_asset_init(WanTextEncoder, core.plan["assets_dir"])
        from .longlive_conditioning import stage_prompt_embeddings, bind_native_prompt_schedule
        embeddings = stage_prompt_embeddings(text_encoder, request["prompts"], device=device, dtype=torch.bfloat16)
        conditioning = {"prompt_embeds": [embeddings[0]]}
        text_encoder.cpu()
        torch.cuda.empty_cache()
        core.model.to(device=device)
        core.trace("native_text_memory_stage", dit_device=str(device), text_device="cpu")
        vae = relocated_asset_init(WanVAEWrapper, core.plan["assets_dir"])
        frozen_text = PreEncodedText(conditioning, request["prompts"][:1])
        pipe = CausalInferencePipeline(config, device=device, generator=wrapper, text_encoder=frozen_text, vae=vae)
        if core.plan["algorithm"] == "draftkv":
            from .draftkv_native import install_empty_native_cache
            install_empty_native_cache(pipe, "longlive")
        pipe.frame_seq_length = request["width"] * request["height"] // 256
        schedule = bind_native_prompt_schedule(wrapper, embeddings, pipe.frame_seq_length, core.trace)
        core.trace("text_conditioning_bound", case_id=request["case_id"], input_mode=request["input_mode"],
                   dataset_manifest_sha256=request["dataset_manifest_sha256"], native_frames=request["native_frames"],
                   scheduler_owner="source_locked_CausalInferencePipeline.inference")
        # Native RNG consumption is retained: create full request noise once,
        # then let the original pipeline draw its own scheduler noise.
        noise = torch.randn((1, request["latent_frames"], 16, request["height"] // 8, request["width"] // 8),
                            device=device, dtype=torch.bfloat16)
        from .gpu_profile import diffusion_start, diffusion_end
        diffusion_start()
        _, latents = run_native_t2v(pipe, noise, request)
        diffusion_end()
        schedule.finish()
        if tuple(latents.shape) != tuple(noise.shape):
            raise BackboneError("OUTPUT_SPEC_MISMATCH", "native T2V latent sequence changed")
        if int(os.environ.get("RANK", "0")) == 0:
            torch.save(latents.cpu(), Path(core.plan["run_dir"]) / "native_latents.pt")
        core.model.cpu()
        del noise, conditioning, embeddings, frozen_text, text_encoder, pipe, schedule
        torch.cuda.empty_cache()
        vae.to(device=device, dtype=torch.bfloat16)
        video = decode_native_t2v(vae, latents, request)
        result = video[0].permute(1, 0, 2, 3).contiguous()
        core.trace("decoded_tensor", layout="C,F,H,W", shape=list(result.shape))
        return result
