"""Native SolarWM-H3 model and generation protocol, shared by all methods."""

from pathlib import Path

from .contracts import BackboneError


class SolarwmCore:
    camera_interface = "native_h3_cadence_relative_w2c_mmrope_camera_prope"
    camera_provenance = "original_orbia_c2w_native_h3_cadence"

    def __init__(self, spec, plan, trace):
        self.spec, self.plan, self.trace = spec, plan, trace
        self.session = None

    def attention_modules(self):
        return [block.attn for block in self.blocks]

    def bind_attention(self, session):
        from .solarwm_seam import bind

        bind(self, session)
        if callable(getattr(getattr(session, "hook", None), "set_camera_poses", None)):
            session.hook.set_camera_poses(self.camera_c2w)

    def load(self):
        import gc
        import os
        import torch
        import torch.distributed as dist
        from solarwm.backends.minimax_h3.fsdp import initialize_distributed
        from solarwm.backends.minimax_h3.optional import load_transformer
        from solarwm.backends.minimax_h3.lora import inject_h3_lora
        from solarwm.backends.minimax_h3.runtime import _base_model_load_receipt
        from solarwm.backends.minimax_h3.weights import load_initial_weights
        from draftkv.backbones.budgets import bound_budget

        local = int(os.environ.get("LOCAL_RANK", "0"))
        self.rank, world = initialize_distributed(
            sp_size=self.plan["sp"], local_rank=local
        )
        if world != self.plan["sp"]:
            raise BackboneError(
                "SOURCE_ABI_DRIFT", "H3 SP world differs from selected topology"
            )
        self.device = torch.device("cuda", local)
        self.config = dict(
            self.plan["native_model_config"],
            checkpoint_path=self.plan["assets_dir"],
            transformer_device_map="cpu",
            local_files_only=True,
        )
        self.prepare_condition()
        # Serialize CPU deserialization; all ranks subsequently own an ordinary
        # FULL_SHARD parameter partition. No memory estimate admission gate.
        for loading_rank in range(world):
            if self.rank == loading_rank:
                modules = load_transformer(self.config, device="cpu")
                base = _base_model_load_receipt(self.config, modules.transformer)
                model, lora = inject_h3_lora(
                    modules.transformer, self.config["adapter"], base_identity=base
                )
                identity = load_initial_weights(
                    dict(
                        path=self.plan["checkpoint"],
                        stage="stage2",
                        weight_source="ema",
                    ),
                    lora,
                )
                expected_identity = f"stage2:ema:step={self.config['checkpoint_step']}"
                if identity != expected_identity:
                    raise BackboneError(
                        "BLOCKED_CHECKPOINT", f"H3 release requires {expected_identity}"
                    )
                model.eval().requires_grad_(False)
                self.target = (
                    model.get_base_model()
                    if hasattr(model, "get_base_model")
                    else model
                )
                self.blocks = list(self.target.transformer_blocks)
                if len(self.blocks) != 50 or any(
                    b.attn.heads != 56 for b in self.blocks
                ):
                    raise BackboneError(
                        "SOURCE_ABI_DRIFT", "H3 checkpoint architecture changed"
                    )
                budget = bound_budget(self, self.plan["algorithm"])
                for attn in self.attention_modules():
                    attn.local_attn_size, attn.sink_size = budget.capacity, budget.sink
                if self.plan["algorithm"] == "infinigen":
                    from .solarwm_forecast import freeze

                    self.query_profiles = freeze(self.blocks)
                if world > 1:
                    from torch.distributed.fsdp import (
                        FullyShardedDataParallel as FSDP,
                        ShardingStrategy,
                    )

                    fp32 = {id(unit) for unit in modules.fp32_fsdp_units}

                    def policy(module, recurse, nonwrapped_numel):
                        return (
                            recurse
                            or isinstance(module, modules.transformer_block_cls)
                            or id(module) in fp32
                        )

                    model = FSDP(
                        model,
                        auto_wrap_policy=policy,
                        sharding_strategy=ShardingStrategy.FULL_SHARD,
                        device_id=self.device,
                        use_orig_params=True,
                        sync_module_states=False,
                        limit_all_gathers=True,
                        process_group=dist.group.WORLD,
                    )
                else:
                    model = model.to(self.device)
                self.model = model
                del modules, lora
                gc.collect()
            if world > 1:
                dist.barrier()
        from flash_attn import flash_attn_func

        self.native_fa2 = lambda q, k, v: flash_attn_func(
            q.contiguous(), k.contiguous(), v.contiguous(), dropout_p=0.0, causal=False
        )
        self.trace(
            "weights_loaded",
            checkpoint=self.plan["checkpoint"],
            layers=50,
            heads=56,
            head_dim=128,
            checkpoint_step=self.config["checkpoint_step"],
            rank=self.rank,
        )
        self.trace("distributed_topology_bound", **self.plan["distributed_topology"])
        return self

    def prepare_condition(self):
        import gc
        import numpy as np
        import torch
        import torch.distributed as dist
        from PIL import Image
        from safetensors.torch import save_file, load_file
        from solarwm.backends.minimax_h3.optional import load_conditioners
        from solarwm.backends.minimax_h3.official_codec import (
            encode_joint_prompt_condition,
        )
        from diffusers.modular_pipelines.minimax_h3.encoders import encode_vae_condition
        from .solarwm_geometry import camera_arrays
        from .planning import sha256

        request = self.plan["request"]
        path = Path(self.plan["run_dir"]) / "h3_condition.safetensors"
        cached = False
        expected_condition = self.plan.get('condition_sha256')
        if expected_condition is not None:
            if not path.exists() or sha256(path) != expected_condition:
                raise BackboneError('SOURCE_SHA_MISMATCH', 'cached H3 condition differs')
            cached = True
        if self.rank == 0 and not cached:
            components = load_conditioners(
                self.config,
                device="cpu",
                video_vae=False,
                audio_vae=False,
                schedulers=False,
            )
            from accelerate import cpu_offload

            cpu_offload(
                components.text_encoder.model,
                execution_device=self.device,
                offload_buffers=True,
            )
            image = (
                Image.open(request["image"])
                .convert("RGB")
                .resize((1344, 768), Image.Resampling.LANCZOS)
            )
            with torch.inference_mode():
                prompt, tags = encode_joint_prompt_condition(
                    image,
                    request["prompts"][0],
                    processor=components.processor,
                    tokenizer=components.tokenizer,
                    text_encoder=components.text_encoder,
                    device=self.device,
                )
                del components
                gc.collect()
                torch.cuda.empty_cache()
                components = load_conditioners(
                    self.config,
                    device=self.device,
                    qwen=False,
                    audio_vae=False,
                    schedulers=False,
                )
                components.video_vae.to(dtype=torch.bfloat16)
                pixels = (
                    torch.from_numpy(np.asarray(image).copy())
                    .permute(2, 0, 1)[None, :, None]
                    .to(self.device)
                )
                anchor = encode_vae_condition(
                    components.video_vae,
                    pixels,
                    (0.485, 0.456, 0.406),
                    (0.229, 0.224, 0.225),
                    encode_seed=42,
                )
            save_file(
                {
                    "prompt": prompt.cpu().contiguous(),
                    "tags": tags.cpu().contiguous(),
                    "anchor": anchor.cpu().contiguous(),
                },
                str(path),
            )
            del components, prompt, tags, anchor, pixels
            gc.collect()
            torch.cuda.empty_cache()
        if self.plan["sp"] > 1:
            dist.barrier()
        self.condition = load_file(str(path))
        source = (
            Path(request["dataset_root"]) / "public" / request["case_id"] / "camera.npz"
        )
        with np.load(source, allow_pickle=False) as archive:
            _, _, absolute = camera_arrays(
                archive["poses_c2w"],
                request["original_intrinsics_K"],
                request["source_width"],
                request["source_height"],
                frames=request["frames"],
            )
        self.camera_c2w = torch.from_numpy(absolute).to(self.device)
        self.trace(
            "solarwm_condition_prepared",
            condition_sha256=sha256(path),
            original_file_sha256=request["original_file_sha256"],
            frame_policy=request["frame_policy"],
        )

    def install_query_forecast(self, hook):
        from .solarwm_forecast import install

        return install(self, hook)

    def generate(self):
        import numpy as np
        import torch
        import torch.distributed as dist
        from solarwm.backends.minimax_h3.artifacts import load_silence_latents
        from solarwm.backends.minimax_h3.distributed import broadcast_sp_tensor
        from solarwm.backends.minimax_h3.layout import patchify_video
        from solarwm.backends.minimax_h3.torch_flow import (
            scale_noise,
            sample_shifted_timestep,
        )
        from solarwm.backends.minimax_h3.sgf_rollout import H3SGFInputs, h3_sgf_rollout
        from solarwm.backends.minimax_h3.full_length import full_length_geometry
        from solarwm.backends.minimax_h3.full_inference import decode_write
        from solarwm.backends.minimax_h3.optional import load_conditioners

        if self.session is None:
            from .solarwm_seam import bind_transactions

            bind_transactions(self)
        request = self.plan["request"]
        geometry = full_length_geometry(request["frames"])
        if (
            geometry.rollout_latents,
            geometry.decode_latents,
            geometry.decoded_frames,
        ) != (
            request["latent_frames"],
            request["decode_latents"],
            request["native_frames"],
        ):
            raise BackboneError(
                "SOURCE_ABI_DRIFT",
                "native H3 length protocol differs from frozen request",
            )
        torch.cuda.reset_peak_memory_stats(self.device)
        rng = torch.Generator(device=self.device).manual_seed(request["seed"])

        def random_like(value):
            out = torch.randn(
                value.shape, device=self.device, dtype=torch.float32, generator=rng
            )
            broadcast_sp_tensor(out)
            return out

        values = self.condition
        anchor = values["anchor"].to(self.device).float()
        if anchor.ndim == 4:
            anchor = anchor.unsqueeze(0)
        anchor = scale_noise(anchor, random_like(anchor), 0.999)
        silence, _ = load_silence_latents(self.plan["silence_path"], pixel_frames=158)
        silence = (
            silence.to(self.device)
            .float()
            .unsqueeze(0)
            .permute(0, 1, 3, 2)
            .reshape(1, -1, 32)
            .contiguous()
        )
        audio_t = sample_shifted_timestep(
            1, 3.0, self.device, torch.float32, generator=rng
        ).reshape(())
        broadcast_sp_tensor(audio_t)
        audio = scale_noise(silence, random_like(silence), audio_t)
        camera = Path(request["camera_dir"])
        views, K = [
            torch.from_numpy(np.load(camera / name, allow_pickle=False))
            .unsqueeze(0)
            .to(self.device)
            for name in ("poses.npy", "intrinsics.npy")
        ]
        views, K = [torch.cat((t[:, :1], t), 1) for t in (views, K)]
        prompt = values["prompt"].to(self.device).bfloat16()
        if prompt.ndim == 2:
            prompt = prompt.unsqueeze(0)
        inputs = H3SGFInputs(
            prompt,
            values["tags"].to(self.device),
            patchify_video(anchor),
            audio,
            audio_t,
            views,
            K,
            48,
            84,
            0.999,
            self.plan["sp"] > 1,
        )
        noise = random_like(
            torch.empty(1, 24, geometry.rollout_latents, 48, 84, device=self.device)
        )
        from .gpu_profile import diffusion_start, diffusion_end
        diffusion_start()
        with torch.no_grad():
            result = h3_sgf_rollout(
                student=self.model,
                inputs=inputs,
                noise=noise,
                exit_index=3,
                generator=rng,
                video_shift=12.0,
                inference_full_length=True,
                output_device="cpu",
                progress=lambda chunk: self.trace(
                    "solarwm_chunk_complete", chunk=chunk
                ),
            )
        diffusion_end()
        from safetensors.torch import save_file
        from .planning import sha256
        import json
        import subprocess
        import sys
        run = Path(self.plan['run_dir'])
        latent_path = run/'complete_latents.safetensors'
        if self.rank == 0:
            save_file({'cache_target': result.cache_target.detach().cpu().contiguous()}, str(latent_path))
            receipt = dict(status='COMPLETE', latents_sha256=sha256(latent_path),
                source_commit=self.plan['repo_commit'], request=self.plan['request'],
                peak_allocated_bytes=torch.cuda.max_memory_allocated(self.device),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(self.device))
            (run/'sampling_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
        # Persist all completed samples before dismantling the sampler. A VAE
        # failure can resume decode without another model forward or new RNG.
        del result, inputs, noise, anchor, silence, audio, prompt, views, K, values
        self.release_sampler()
        if self.rank == 0:
            # RuntimeBackbone writes backbone_plan.json; retain legacy plan.json.
            decode_plan = run/'plan.json'
            if not decode_plan.is_file():
                decode_plan = run/'backbone_plan.json'
            subprocess.run([sys.executable, '-m', 'draftkv.backbones.solarwm_decode',
                            '--plan', str(decode_plan), '--latents', str(latent_path)],
                           check=True)
        if self.plan["sp"] > 1:
            dist.barrier()

    def release_sampler(self):
        """Release global seam closures, model, and plugin caches before VAE."""
        import gc
        import torch
        from solarwm.backends.minimax_h3 import sgf_attention as native
        from solarwm.backends.minimax_h3 import sgf_rollout as rollout
        if hasattr(self, '_solarwm_original_attention'):
            native.sgf_attention = self._solarwm_original_attention
        if hasattr(self, '_solarwm_original_forward'):
            rollout.h3_student_forward = self._solarwm_original_forward
        from .solarwm_host_cache import release_provider_registries
        release_provider_registries(self)
        for handle in getattr(self, '_solarwm_forecast_handles', ()):
            handle.remove()
        self._solarwm_forecast_handles = []
        self.query_profiles = []
        self.session = None
        self.active_geometry = None
        self.blocks = []
        self.target = None
        self.model = None
        self.camera_c2w = None
        self.condition = None
        gc.collect()
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        remaining = torch.cuda.memory_allocated(self.device)
        self.trace('solarwm_sampler_released', allocated_bytes=remaining,
                   reserved_bytes=torch.cuda.memory_reserved(self.device),
                   independent_vae_decoder=True)
        if remaining > 4*1024**3:
            raise BackboneError('KV_CONSTRAINT', 'sampling tensors remain before independent VAE decode')
