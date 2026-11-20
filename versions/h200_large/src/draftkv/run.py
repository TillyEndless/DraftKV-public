"""Explicit mechanism ablation runner; ordinary release runners remain unchanged.

Dry-run never imports torch, probes hardware, reads weights or executes workers.
Real runs need the ordinary frozen production worker plan (--production-plan).
All outputs are exclusive, identity-marked directories. No evaluator is launched.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

from draftkv.ablation.config import AblationError, read_config, resolve, resolve_attention_backend
from draftkv.ablation.planning import manifest


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ablation-config', required=True)
    p.add_argument('--backbone', required=True)
    p.add_argument('--benchmark', required=True, choices=('orbia', 'bench1', 'bench2'))
    p.add_argument('--method', '--algorithm', default='draftkv', choices=('draftkv',))
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--production-plan', help='frozen ordinary worker plan, required for real runs')
    p.add_argument('--device', default='H200', help='explicit architecture, not a GPU probe')
    p.add_argument('--devices', default='0', help='physical device IDs; real runs require SP1 canonical adapter')
    p.add_argument('--history-frames', type=int, default=72, help='dry-run history assumption; real plan history is authoritative')
    p.add_argument('--frame-tokens', type=int, help='dry-run shape assumption; runtime actual QKV is authoritative')
    p.add_argument('--device-memory-gib', type=float)
    p.add_argument('--output-root', default='outputs/ablations')
    p.add_argument('--repo-root')
    return p


def prepare(args):
    from draftkv.backbones.catalog import MODELS
    from draftkv.runtime.config import parse_devices
    selected_devices = parse_devices(args.devices)
    if len(selected_devices) != 1:
        raise AblationError('mechanism runner currently requires explicit SP1; use ordinary production runner for other SPs')
    root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[2]
    config_path = Path(args.ablation_config)
    if not config_path.is_absolute():
        config_path = root/config_path
    config = resolve(read_config(config_path))
    if config is None:
        raise AblationError('disabled file resolves to unchanged production; use ordinary production runner')
    models = {name.lower(): spec for name, spec in MODELS.items()}
    if args.backbone.lower() not in models:
        raise AblationError('unknown backbone')
    spec = models[args.backbone.lower()]
    benchmark = 'bench2' if args.benchmark == 'orbia' else args.benchmark
    if (benchmark == 'bench1') != (spec.family == 'longlive'):
        raise AblationError('backbone/benchmark protocol mismatch')
    plan = None
    if args.production_plan:
        plan = json.loads(Path(args.production_plan).read_text())
        if plan['backbone'] != spec.backbone or plan['algorithm'] != 'draftkv' or plan['benchmark'] != benchmark:
            raise AblationError('production plan identity mismatch')
        if plan.get('sp', 1) != 1:
            raise AblationError('experimental runtime currently requires SP1; no silent SP remapping')
        history_frames = int(plan.get('history_plan', {}).get('capacity_frames', plan['request']['latent_frames']))
        frame_tokens = int(plan.get('history_plan', {}).get('frame_tokens', args.frame_tokens or spec.frame_tokens or 1508))
        settings = plan['algorithm_plugin'].get('config', {})
        if plan['algorithm_plugin']['entry'] != 'src/draftkv/algorithms/draftkv/entry.py':
            raise AblationError('explicit ablation requires canonical entry.py; legacy separate plugin entries do not implement this config ABI')
        if config['profile'] not in ('draftkv_full','fullkv') and settings.get('gpu_device_route'):
            if not settings.get('headgroup_pipeline') or spec.family=='solarwm':
                raise AblationError('this device provider lacks production grouped/native consumer support; use a supported canonical provider')
        if config['profile'] not in ('draftkv_full','fullkv') and settings.get('gpu_prepack') and not settings.get('gpu_device_route'):
            raise AblationError('legacy prepacked-only provider is unsupported; use canonical block or production device/headgroup plan')
    else:
        history_frames, frame_tokens = args.history_frames, args.frame_tokens or spec.frame_tokens or 1508
    sha = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    result = manifest(config, spec, benchmark=benchmark, device=args.device, history_frames=history_frames,
                      frame_tokens=frame_tokens, git_sha=sha, config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
                      production_settings=plan['algorithm_plugin'].get('config', {}) if plan else None,
                      device_memory_gib=args.device_memory_gib)
    if plan and config['retrieval']['selector'] != 'all_history' and plan['algorithm_plugin'].get('config', {}).get('gpu_device_route'):
        result['execution_provider'] = 'production_device_headgroup'
    if config['retrieval']['selector'] == 'all_history':
        result['execution_provider'] = 'full_history_native_FA'
    result['shape_source'] = ('frozen_plan_with_frame_token_assumption' if plan and not plan.get('history_plan', {}).get('frame_tokens') and args.frame_tokens is None and spec.frame_tokens is None else 'frozen_plan_or_explicit_shape') if plan else 'explicit_dry_run_assumptions_not_observed_inputs'
    result['sp'] = 1
    result['selected_devices'] = list(selected_devices)
    result['config_path'] = str(config_path)
    # Actual clean-frontier evolution is recorded by runtime traces, not invented
    # by extrapolating this dry-run static maximum.
    folder = Path(args.output_root)/('abl_'+config['profile'])/uuid.uuid4().hex
    if not folder.is_absolute():
        folder = root/folder
    folder.mkdir(parents=True, exist_ok=False)
    result['output_directory'] = str(folder)
    with (folder/'manifest.json').open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    return root, config, plan, folder, result


def execute_plan(root, config, plan, folder, result, args):
    """Called only explicitly without --dry-run. CPU tests mock subprocess launch."""
    if plan is None:
        raise AblationError('real generation requires --production-plan from the normal frozen assets/request planner')
    if ',' in args.devices:
        raise AblationError('canonical experimental adapter currently requires SP1')
    from draftkv.release.audit import release_source_files
    plan['ablation_production_source_sha'] = plan.get('repo_commit')
    plan.update(repo_commit=result['git_sha'], ablation=config, repo_root=str(root), run_dir=str(folder), output=str(folder/'video.mp4'),
                selected_devices=[int(args.devices)], ablation_manifest=str(folder/'manifest.json'), ablation_expected_backend=result['attention_backend'],
                ablation_source=dict(path=result['config_path'],sha256=result['config_sha256']))
    # Source inventory remains authoritative; ablation does not disable locks.
    plan['adapter_source_files'] = release_source_files(root, require_complete=True)
    plan['algorithm_plugin']['root'] = str(root)
    plan['algorithm_plugin']['files'] = dict(plan['adapter_source_files'])
    # Preserve prompt/seed/case/protocol and frozen request fields exactly.
    # Ablation identity is in plan/manifest/output, never re-labelled as release evidence.
    path = folder/'backbone_plan.json'
    with path.open('x') as f:
        json.dump(plan, f, indent=2)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.devices,
               PYTHONPATH=os.pathsep.join((str(root/'src'), os.environ.get('PYTHONPATH', ''))))
    result['status'] = 'RUNNING'
    (folder/'manifest.json').write_text(json.dumps(result, indent=2)+'\n')
    try:
        subprocess.run([sys.executable, '-m', 'draftkv.backbones.worker', '--plan', str(path)],
                       cwd=root, env=env, check=True)
    except BaseException:
        result['status'] = 'FAILED'
        raise
    else:
        result['status'] = 'GENERATION_RETURNED_REQUIRES_EVALUATION'
    finally:
        from draftkv.ablation.reporting import finalize_manifest
        try:
            finalize_manifest(folder, result)
        except (OSError, ValueError, KeyError) as error:
            result['manifest_accounting_error'] = str(error)
            if result['status'] != 'FAILED':
                result['status'] = 'GENERATION_RETURNED_TRACE_ACCOUNTING_FAILED'
        (folder/'manifest.json').write_text(json.dumps(result, indent=2)+'\n')


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        if not args.dry_run and not args.production_plan:
            raise AblationError('real generation requires --production-plan; use --dry-run tonight')
        root, config, plan, folder, result = prepare(args)
        if not args.dry_run:
            execute_plan(root, config, plan, folder, result, args)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (AblationError, OSError, KeyError, ValueError, subprocess.CalledProcessError) as e:
        print(f'ablation error: {e}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
