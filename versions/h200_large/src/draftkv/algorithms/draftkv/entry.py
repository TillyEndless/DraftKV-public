"""File-loaded canonical DraftKV HookRegistry entry. Importing is GPU inert."""
import hashlib
from pathlib import Path


def configure_retrieval(backbone, settings):
    from draftkv.backbones.catalog import BLOCK_DRAFTKV_BACKBONES
    from draftkv.backbones.contracts import BackboneError
    settings = dict(settings)
    from draftkv.algorithms.draftkv.experiment import require_active, archive_page_policy
    override = require_active(settings, backbone)
    archive_page_policy(settings, backbone)
    if backbone in BLOCK_DRAFTKV_BACKBONES:
        from draftkv.algorithms.draftkv.block_policy import configured_protection
        expected = configured_protection(backbone)
        supplied = settings.get('protect_recent_frames', expected)
        if type(supplied) is not int or supplied != expected:
            raise BackboneError('BLOCKED_ALGORITHM_SEMANTICS',
                                f'{backbone} requires Direct Block Top-K P{expected}')
        settings['protect_recent_frames'] = expected
        if settings.get('retrieval_granularity', 'block') != 'block':
            raise BackboneError('BLOCKED_ALGORITHM_SEMANTICS',
                                'this DraftKV group requires block retrieval')
        settings['retrieval_granularity'] = 'block'
    mode = settings.get('retrieval_ablation', 'full')
    if type(mode) is not str or mode not in ('full', 'draft-only', 'camera-only'):
        raise BackboneError('BLOCKED_ALGORITHM_SEMANTICS',
                            'retrieval_ablation must be full, draft-only or camera-only')
    if mode != 'full':
        if backbone not in ('LingBot1.3B', 'LingBot14B', 'SolarWM33B'):
            raise BackboneError('BLOCKED_ALGORITHM_SEMANTICS',
                                'Table2 A2 retrieval ablation requires native camera geometry')
        if settings.get('gpu_prepack', False) or settings.get('gpu_device_route', False):
            raise BackboneError('BLOCKED_ALGORITHM_SEMANTICS',
                                'A2 ablation requires canonical CPU B01 route')
    settings['retrieval_ablation'] = mode
    if mode == 'camera-only':
        from draftkv.backbones.budgets import effective_budget
        from draftkv.backbones.catalog import MODELS
        spec = MODELS[backbone]
        budget = effective_budget(spec, 'draftkv')
        remaining = budget.retrieved - settings['protect_recent_frames']
        if remaining <= 0 or remaining % spec.chunk:
            raise BackboneError('BLOCKED_ALGORITHM_SEMANTICS',
                                'camera-only remote frame budget must fit whole native chunks')
        k = remaining // spec.chunk
        supplied = settings.get('camera_only_top_chunks', k)
        if type(supplied) is not int or supplied != k:
            raise BackboneError('BLOCKED_ALGORITHM_SEMANTICS',
                                f'{backbone} Camera-only must use Top-{k} whole chunks')
        settings['camera_only_top_chunks'] = k
    elif 'camera_only_top_chunks' in settings:
        raise BackboneError('BLOCKED_ALGORITHM_SEMANTICS',
                            'camera_only_top_chunks is only valid for camera-only ablation')
    # Native-conditioned models share the same A2 BFO camera score and
    # no-backfill semantics, but retain their own C-sized chunks and P budgets.
    top_by_backbone = {'LingBot14B': 3, 'LingBot1.3B': 3, 'SolarWM33B': 7}
    if backbone in top_by_backbone:
        top = override['camera_top_chunks'] if override is not None else top_by_backbone[backbone]
        if settings.get('camera_gate_mode', 'hard_top_m_no_backfill') != 'hard_top_m_no_backfill':
            raise BackboneError('BLOCKED_ALGORITHM_SEMANTICS',
                                'A2 Final forbids Early Fix and alternative camera gates')
        supplied = settings.get('camera_top_chunks', top)
        if type(supplied) is not int or supplied != top:
            raise BackboneError('BLOCKED_ALGORITHM_SEMANTICS',
                                f'{backbone} requires A2 Camera Top-{top}')
        settings['camera_gate_mode'] = 'hard_top_m_no_backfill'
        settings['camera_top_chunks'] = top
    # LongLive1.3B/WorldAttention is strict text-only T2V: no real camera
    # C2W exists. It stays on P3 Block Top-K, never a fictitious BFO route.
    if backbone == 'LongLive1.3B' and any(
            k in settings for k in ('camera_top_chunks', 'camera_gate_mode')):
        raise BackboneError('CAMERA_POSES_MISSING',
                            'LongLive text-only T2V cannot use A2 Top-2 without observed camera geometry')
    return settings


def create_hook(core):
    # Only a resolved, explicit ablation file can enter this separate owner.
    if getattr(core, 'plan', {}).get('ablation', {}).get('enabled', False):
        from draftkv.ablation.runtime import create_ablation_hook
        return create_ablation_hook(core)
    return _create_production_hook(core)


def _create_production_hook(core):
    cfg = getattr(core, 'plan', {}).get('algorithm_plugin', {}).get('config', {})
    mode = configure_retrieval(core.spec.backbone, cfg)['retrieval_ablation']
    if mode == 'draft-only':
        # Exact existing DraftKV Direct Block Top-K; no camera score or new selection.
        from draftkv.algorithms.draftkv.block_backbone import BlockDraftKVHook
        return BlockDraftKVHook(core)
    if mode == 'camera-only':
        from draftkv.algorithms.draftkv.camera_frustum_multibackbone import CameraOnlyChunkHook
        return CameraOnlyChunkHook(core)
    if getattr(core,'plan',{}).get('algorithm_plugin',{}).get('config',{}).get('gpu_device_route',False):
        from draftkv.algorithms.draftkv.device_backbone import DeviceRouteBlockHook
        return DeviceRouteBlockHook(core)
    if getattr(core,'plan',{}).get('algorithm_plugin',{}).get('config',{}).get('gpu_prepack',False):
        from draftkv.algorithms.draftkv.prepacked_backbone import GPUPreparedBlockHook
        return GPUPreparedBlockHook(core)
    if core.spec.backbone == 'LingBot14B':
        from draftkv.algorithms.draftkv.experiment import active_override, experimental_14b_hook
        if active_override(core.spec.backbone) is not None:
            return experimental_14b_hook(core)
        # Exact frozen pre-Early-Fix A2 Frustum V2 route.
        from draftkv.algorithms.draftkv.camera_frustum_v2 import CameraGatedBlockDraftHook
        return CameraGatedBlockDraftHook(core)
    if core.spec.backbone in ('LingBot1.3B', 'SolarWM33B'):
        from draftkv.algorithms.draftkv.camera_frustum_multibackbone import CameraGatedMultiBackboneHook
        return CameraGatedMultiBackboneHook(core)
    if core.spec.backbone == 'LongLive1.3B':
        from draftkv.algorithms.draftkv.block_backbone import BlockDraftKVHook
        return BlockDraftKVHook(core)
    from draftkv.algorithms.draftkv.backbone import DraftKVHook
    return DraftKVHook(core)


def register(registry):
    from draftkv.backbones.hooks import HookRegistration
    from draftkv.algorithms.draftkv.experiment import active_override
    scoped_semantics = set()
    for backbone in ('LingBot1.3B', 'LingBot14B', 'SolarWM33B'):
        override = active_override(backbone)
        if override is not None:
            scoped_semantics.add(f"direct_block_topk_p{override['protect_recent_frames']}")
    registry.register(HookRegistration(
        'draftkv', create_hook,
        frozenset({'draftattention2_postrope_spatial8x16_meanmax', 'hierarchical_v1',
                   'direct_block_topk_p2', 'direct_block_topk_p3', 'direct_block_topk_p5', 'all_history_softmax', 'chronological_source_token_packing',
                   'original_b01_producer_plan', 'asynchronous_h2d',
                   'cuda_frontier', 'streaming_fa2', 'clean_commit', 'chrono_compact_temporal', 'chronological_frame_packing'}) | frozenset(scoped_semantics),
        hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'streaming_fa2'))
