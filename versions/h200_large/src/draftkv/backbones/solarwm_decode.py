"""Independent native H3 VAE decode from SHA-bound complete CPU latents."""
import argparse
import json
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan', required=True)
    p.add_argument('--latents', required=True)
    a = p.parse_args()
    from .planning import sha256, verify_files
    plan = json.loads(Path(a.plan).read_text())
    verify_files(plan['repo_root'], plan['adapter_source_files'])
    verify_files(plan['source_root'], plan['source_files'])
    run = Path(plan['run_dir'])
    receipt = json.loads((run/'sampling_receipt.json').read_text())
    if sha256(a.latents) != receipt['latents_sha256']:
        raise RuntimeError('SOLARWM_LATENTS_SHA_MISMATCH')
    for root in reversed(plan['import_roots']):
        sys.path.insert(0, root)
    import torch
    from safetensors.torch import load_file
    from solarwm.backends.minimax_h3.full_length import full_length_geometry
    from solarwm.backends.minimax_h3.full_inference import decode_write
    from solarwm.backends.minimax_h3.optional import load_conditioners
    device = torch.device('cuda', int(__import__('os').environ.get('LOCAL_RANK', '0')))
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    config = dict(plan['native_model_config'], checkpoint_path=plan['assets_dir'],
                  transformer_device_map='cpu', local_files_only=True)
    latent = load_file(a.latents, device='cpu')['cache_target']
    geometry = full_length_geometry(plan['request']['frames'])
    def decode():
        with torch.inference_mode():
            vae = load_conditioners(config, device=device, qwen=False,
                                    audio_vae=False, schedulers=False).video_vae
            vae.set_attention_backend('native')
            decode_write(vae, latent, geometry, Path(plan['output']),
                         plan['request']['fps'], device, streaming=True)
    from .latency_build import compilation_watch
    watch = compilation_watch(torch) if plan.get('gpu_profile', False) else None
    if watch is not None:
        with watch:
            watch.attach_runtime_loader(torch)
            decode()
    else:
        decode()
    (run/'decode_receipt.json').write_text(json.dumps(dict(
        status='COMPLETE', latents_sha256=receipt['latents_sha256'],
        video_sha256=sha256(plan['output']), independent_process=True,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(device)), indent=2)+'\n')
    if watch is not None:
        path = run/'decode_receipt.json'
        report = json.loads(path.read_text())
        report['latency_admission'] = watch.report()
        path.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
