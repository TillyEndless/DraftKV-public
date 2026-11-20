"""ORBIA deployment assembly; use the shared Bench1 release and config binder."""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
DRAFTKV_IMPORTANCE_CONFIG = Path('configs/algorithms/draftkv/scorer.json')


def draftkv_importance_binding():
    path = ROOT / DRAFTKV_IMPORTANCE_CONFIG
    if not path.is_file():
        raise ValueError('DraftKV importance config missing from checkout')
    return {
        'importance_config': DRAFTKV_IMPORTANCE_CONFIG.as_posix(),
        'importance_config_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def script(name):
    spec = importlib.util.spec_from_file_location('_draftkv_' + name, ROOT/'scripts'/f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def runtime_name(backbone):
    return 'bench2-solarwm-cu128' if backbone == 'SolarWM33B' else 'bench2-lingbot-cu128'


def runtime_python(cache, backbone=None):
    return Path(cache)/'envs'/runtime_name(backbone)/'bin/python'



def selected_python(args, cache, backbone):
    return (getattr(args, 'solarwm_python', None) or runtime_python(cache, backbone)
            if backbone == 'SolarWM33B' else runtime_python(cache, backbone))


def lingbot_assets(cache, backbone):
    cache=Path(cache)
    if backbone == 'LingBot14B':
        return cache/'assets/lingbot-v2-14b/snapshot'
    required=('models_t5_umt5-xxl-enc-bf16.pth','Wan2.1_VAE.pth','google/umt5-xxl')
    candidates=(cache/'assets/lingbot-v2-14b/snapshot', cache/'assets/lingbot-shared/snapshot')
    for candidate in candidates:
        if all((candidate/name).exists() for name in required):
            return candidate
    raise ValueError('LingBot1.3B requires shared T5/VAE/tokenizer assets; run install.sh all --benchmark bench2')


def command_logs(args, backbone=None, algorithm=None):
    path = Path(getattr(args, 'workspace', None) or ROOT/'runs/bench2/_release')/'release-commands'
    return path/backbone/algorithm if backbone and algorithm else path/'shared'


def dataset_binding(args, cache):
    from draftkv.benchmarks.orbia import ORBIADatasetResolver, ORBIASpec
    from draftkv.release.options import dataset_root_override
    root = Path(dataset_root_override(args) or cache/'orbia/data').expanduser().resolve()
    index = Path(args.orbia_index or cache/'orbia/installed-files.json').expanduser().resolve()
    pin = args.orbia_index_sha256 or hashlib.sha256(index.read_bytes()).hexdigest()
    return ORBIADatasetResolver(ROOT).resolve(ORBIASpec(root, index, pin, 'inference'))


def b01(args, cache, backbone, family, arch, sp, device_uuids):
    from draftkv.backbones.sequence_parallel import local_heads
    from draftkv.streaming.config import select_config, validate_parameters
    shared = script('bench1_release')
    shared._COMMAND_LOG_ROOT = command_logs(args, backbone, 'draftkv')
    from draftkv.backbones.catalog import model_spec
    spec = model_spec(backbone)
    heads = local_heads(spec.heads, spec.family, sp)
    if args.b01_lingbot:
        built = shared.read_json(args.b01_lingbot)
    else:
        cuda_home = shared.resolve_cuda_home(args)
        out = cache/'b01'/runtime_name(backbone)
        built = json.loads(shared.command([sys.executable, ROOT/'scripts/bench1_build_b01.py',
            '--python', selected_python(args, cache, backbone), '--cuda-home', cuda_home,
            '--arch', arch, '--cache-root', out/'cache', '--spec-dir', out/'spec']))
    if family == 'H100':
        # Exactly the H200 source-bound SM90 token Streaming FA2 route.
        # Never open a cached accepted.json: old GPU Autotune/Frontier
        # consumers are not part of the canonical H100 execution path.
        if (args.b01_lingbot_parameters or args.b01_autotune
                or args.b01_lingbot_autotune_result):
            raise ValueError('H100 uses fixed H200-compatible SM90 Token Streaming FA2; remove legacy B01 tuning/parameter flags')
        parameters = select_config('H100', local_heads=heads)['parameters']
    elif family == 'B200':
        override = args.b01_autotune or args.b01_lingbot_autotune_result
        path = (Path(override).expanduser().resolve() if override else
                cache/'autotune'/family/runtime_name(backbone)/backbone/f'sp{sp}'/
                shared.autotune_device_key(device_uuids)/'accepted.json')
        autotune = shared.read_json(path) if path.is_file() else None
        if autotune and (autotune.get('backbone') != backbone or autotune.get('sp') != sp
                         or autotune.get('device_uuids') != device_uuids):
            raise ValueError('Target autotune backbone/SP/physical GPU UUIDs differ from --devices')
        if args.b01_lingbot_parameters:
            parameters = shared.read_json(args.b01_lingbot_parameters)
        elif autotune is not None:
            parameters = autotune.get('parameters')
        elif override:
            raise ValueError(f'explicit target-autotune record missing: {path}')
        else:
            raise ValueError('B200 requires target-autotune B01 parameters')
    else:
        profile = select_config(family, local_heads=heads)
        parameters = profile.get('parameters')
        if parameters is None:
            raise ValueError(f'{family} has no source-bound B01 runtime parameters')
        if args.b01_lingbot_parameters:
            supplied = shared.read_json(args.b01_lingbot_parameters)
            if supplied != parameters:
                raise ValueError('B01 parameters differ from frozen source profile')
    validate_parameters(parameters)
    return {**draftkv_importance_binding(), 'packing_policy':'chronological_frame_id',
        'profile_admission':'runtime',
        'host_pages_policy':getattr(args, 'host_pages_policy', 'quality'), 'b01': {
        'cache_directory':built['cache_directory'], 'build_spec_path':built['build_spec_path'],
        'build_spec_sha256':built.get('build_spec_sha256') or shared.sha(Path(built['build_spec_path'])),
        'parameters':parameters}}


def build_deployment(args, cache, pool):
    from draftkv.registry import Registry
    from draftkv.benchmarks.orbia_requests import prepare
    shared = script('bench1_release')
    shared._COMMAND_LOG_ROOT = command_logs(args)
    groups = Registry().select('bench2', args.backbone, args.algorithm)
    assigned = shared.assignments(groups, pool)
    picked = shared.gpu_inventory(pool)
    family, arch = shared.gpu_identity(pool,picked)
    device_uuids = [row['uuid'] for row in picked]
    dataset = dataset_binding(args, cache)
    evaluation_path = Path(args.evaluation_config or cache/'evaluator/orbia/evaluation.json').expanduser().resolve()
    evaluation = shared.read_json(evaluation_path)
    evaluation = evaluation.get('evaluation_config', evaluation)
    deployment = {'schema_version':1, 'backbones':{}, 'algorithm_configs':{},
                  'backbone_algorithm_configs':{}, 'group_configs':{}, 'evaluation_config':evaluation}
    if getattr(args, 'gpu_profile', False):
        deployment['gpu_profile'] = True
    algorithms = {g.algorithm for g in groups}
    for algorithm in algorithms & shared.ALGO_DEFAULTS.keys():
        deployment['algorithm_configs'][algorithm] = {'plugin_config':shared.ALGO_DEFAULTS[algorithm]}
    if 'worldtrace-landmark' in algorithms:
        deployment['algorithm_configs']['worldtrace-landmark'] = {
            'plugin_config': {'preset': getattr(args, 'worldtrace_preset', None) or 'table2-budget-matched'}}
    for backbone in sorted({g.backbone for g in groups}):
        result = prepare(backbone, dataset, cache/'inputs/bench2'/backbone, root=ROOT)
        index = cache/'inputs/bench2'/backbone/'requests.json'
        model = cache/'assets'/('lingbot-v2-1.3b' if backbone == 'LingBot1.3B' else 'lingbot-v2-14b')/'snapshot'
        assets = lingbot_assets(cache, backbone) if backbone != "SolarWM33B" else None
        deployment['backbones'][backbone] = {
            'dataset':{'path':str(dataset.root/'manifest.json'), 'sha256':dataset.manifest_sha256},
            'dataset_config':{'root':str(dataset.root), 'index':str(Path(args.orbia_index or cache/'orbia/installed-files.json').expanduser().resolve()), 'index_sha256':dataset.index_sha256},
            'checkpoint':{'path':str(model)},
            'requests_index':{'path':str(index), 'sha256':shared.sha(index)},
            'backbone_config':{'source_root':args.lingbot_source or 'src/draftkv/vendor/backbones/lingbot14',
                'source_lock':'configs/backbones/source_locks.json', 'assets_dir':str(assets),
                'python':str(runtime_python(cache, backbone))}}
        if backbone == 'SolarWM33B':
            from draftkv.backbones.solarwm_checkpoint import release_binding
            snapshot = cache/'assets/solarwm-h3-33b/snapshot'
            package, _ = release_binding(ROOT)
            deployment['backbones'][backbone].update(checkpoint={'path': str(getattr(args, 'solarwm_checkpoint', None) or snapshot/package)})
            deployment['backbones'][backbone]['backbone_config'] = {
                'source_root': str(getattr(args, 'solarwm_source', None) or cache/'sources/solarwm'),
                'source_lock': 'configs/backbones/solarwm_source_lock.json',
                'assets_dir': str(getattr(args, 'solarwm_assets', None) or snapshot/'SolarWM-h3-33B-base'),
                'silence_path': str(getattr(args, 'solarwm_silence', None) or snapshot/'support/h3_silence_153_158_170.safetensors'),
                'python': str(getattr(args, 'solarwm_python', None) or runtime_python(cache, backbone))}
        per = {}
        if backbone == 'SolarWM33B':
            from draftkv.backbones.budgets import effective_budget
            from draftkv.backbones.catalog import model_spec
            solar_world_budget = effective_budget(model_spec(backbone), 'worldkv')
            per.update(infllm={'plugin_config': {'block_size': 252, 'repr_topk': 4}},
                       worldkv={'plugin_config': {'keep_ratio': solar_world_budget.keep_ratio,
                                                 'position_policy': 'h3_native_mmrope_camera_prope'}})

        if backbone == 'LingBot14B' and 'infllm' in algorithms:
            per['infllm'] = {'plugin_config': {'block_size': 'frame', 'repr_topk': 4,
                                               'retrieval_geometry': 'native_absolute'}}

        if backbone in ('LingBot1.3B', 'LingBot14B') and 'worldkv' in algorithms:
            from draftkv.backbones.budgets import effective_budget
            from draftkv.backbones.catalog import model_spec
            per['worldkv'] = {'plugin_config': {'keep_ratio':
                effective_budget(model_spec(backbone), 'worldkv').keep_ratio}}

        if 'draftkv' in algorithms:
            draft_config = b01(args,cache,backbone,family,arch,len(pool),device_uuids)
            # CLI variant is bound before hashing the immutable per-group
            # runtime.json. Separate configuration IDs/workspaces prevent
            # Table2 cross-variant generation and evaluator contamination.
            mode = getattr(args, 'draftkv_ablation', 'full')
            if mode != 'full':
                draft_config['retrieval_ablation'] = mode
            per['draftkv'] = {'plugin_config':draft_config}
            deployment['group_configs'][f'bench2/{backbone}/draftkv'] = {'backbone_config':{'history_capacity_frames':110 if backbone == 'SolarWM33B' else 92}}
            if getattr(args, 'draftkv_sp8_profile', False) and backbone == 'LingBot14B':
                from draftkv.backbones.planning import DRAFTKV_SP8_BF16_PROFILE
                deployment['group_configs'][f'bench2/{backbone}/draftkv']['backbone_config']['draftkv_loading_profile'] = DRAFTKV_SP8_BF16_PROFILE
        if 'freekv' in algorithms:
            shared._COMMAND_LOG_ROOT = command_logs(args, backbone, 'freekv')
            python = selected_python(args, cache, backbone)
            if args.freekv_lingbot_extension:
                binding = shared.extension_binding(args.freekv_lingbot_extension,python,'sm'+arch.replace('.',''))
            else:
                built = json.loads(shared.command([sys.executable,ROOT/'scripts/bench1_build_freekv.py',
                    '--python',python,'--cache-root',cache/'freekv'/runtime_name(backbone),'--device',pool[0]],timeout=14400))
                binding = {k:built[k] for k in ('extension','extension_sha256','python_soabi','torch_version','cuda_version','sm')}
            per['freekv'] = {'plugin_config':{'kernel_binding':binding}}
        if per:
            deployment['backbone_algorithm_configs'][backbone] = per
    return deployment, assigned, {'family':family,'arch':arch}
