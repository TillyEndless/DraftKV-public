# SPDX-License-Identifier: Apache-2.0
"""Independent native H3 VAE decode from SHA-bound complete CPU latents."""
from __future__ import annotations
from draftkv.runtime import profiling as pm
from draftkv.streaming.external import external_symbol, external_path
import argparse
import json
import sys
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan', required=True)
    p.add_argument('--latents', required=True)
    a = p.parse_args()
    from draftkv.backbones.planning import sha256, verify_files
    plan = json.loads(Path(a.plan).read_text())
    verify_files(plan['repo_root'], plan['adapter_source_files'])
    verify_files(plan['source_root'], plan['source_files'])
    run = Path(plan['run_dir'])
    receipt = json.loads((run / 'sampling_receipt.json').read_text())
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
    if pm.enabled(): torch.cuda.reset_peak_memory_stats(device)
    config = dict(plan['native_model_config'], checkpoint_path=plan['assets_dir'], transformer_device_map='cpu', local_files_only=True)
    latent = load_file(a.latents, device='cpu')['cache_target']
    geometry = full_length_geometry(plan['request']['frames'])

    def decode():
        with torch.inference_mode():
            vae = load_conditioners(config, device=device, qwen=False, audio_vae=False, schedulers=False).video_vae
            vae.set_attention_backend('native')
            decode_write(vae, latent, geometry, Path(plan['output']), plan['request']['fps'], device, streaming=True)
    from draftkv.backbones.latency_build import compilation_watch
    watch = compilation_watch(torch) if pm.enabled() else None
    try:
        if watch is not None:
            with watch:
                watch.attach_runtime_loader(torch)
                decode()
        else:
            decode()
    except BaseException as error:
        if pm.enabled():
            try:
                from draftkv.backbones.memory_profile import atomic_json
                atomic_json(run / 'decode_memory_failure.json', dict(status='FAILED', failed_during_phase='vae_decode', error=f'{type(error).__name__}: {error}', peak_allocated_bytes=pm.peak_allocated(torch, device), peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if pm.enabled() else None))
            except BaseException:
                pass
        raise
    (run / 'decode_receipt.json').write_text(json.dumps(dict(status='COMPLETE', latents_sha256=receipt['latents_sha256'], video_sha256=sha256(plan['output']), independent_process=True, peak_allocated_bytes=pm.peak_allocated(torch, device), peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if pm.enabled() else None, latency_admission=watch.report() if watch is not None else None), indent=2) + '\n')
if __name__ == '__main__':
    main()
