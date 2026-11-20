"""File-loaded SM120 experiment entry, isolated from the canonical SM89 entry. Importing is GPU inert."""
import hashlib
from pathlib import Path


def configure_retrieval(backbone, settings):
    from draftkv.backbones.catalog import BLOCK_DRAFTKV_BACKBONES
    from draftkv.backbones.contracts import BackboneError
    settings = dict(settings)
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
    return settings


def create_hook(core):
    if getattr(core,'plan',{}).get('algorithm_plugin',{}).get('config',{}).get('gpu_prepack',False):
        from draftkv.algorithms.draftkv.prepacked_sm120_backbone import GPUPreparedSM120Hook
        return GPUPreparedSM120Hook(core)
    if core.spec.backbone in ('LongLive1.3B', 'LingBot1.3B', 'LingBot14B', 'SolarWM33B'):
        from draftkv.algorithms.draftkv.block_backbone import BlockDraftKVHook
        return BlockDraftKVHook(core)
    from draftkv.algorithms.draftkv.backbone import DraftKVHook
    return DraftKVHook(core)


def register(registry):
    from draftkv.backbones.hooks import HookRegistration
    registry.register(HookRegistration(
        'draftkv', create_hook,
        frozenset({'draftattention2_postrope_spatial8x16_meanmax', 'hierarchical_v1',
                   'direct_block_topk_p2', 'direct_block_topk_p3', 'direct_block_topk_p5', 'all_history_softmax', 'chronological_source_token_packing',
                   'original_b01_producer_plan', 'asynchronous_h2d',
                   'cuda_frontier', 'streaming_fa2', 'clean_commit', 'chrono_compact_temporal', 'chronological_frame_packing'}),
        hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'streaming_fa2'))
