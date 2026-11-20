"""Separate opt-in GPU-route entry; frozen SM89 and SM120 entries stay unchanged."""
import hashlib
from pathlib import Path
from draftkv.algorithms.draftkv.entry_sm120 import configure_retrieval

def create_hook(core):
    import os
    if os.environ.get('CUDA_MODULE_LOADING','').upper()!='EAGER':
        from draftkv.backbones.contracts import BackboneError
        raise BackboneError('GPU_ROUTE_REQUIRES_EAGER_MODULE_LOADING',
            'Set CUDA_MODULE_LOADING=EAGER before worker startup: native model lazy loads can synchronize behind work pending on the producer thread')
    from draftkv.algorithms.draftkv.routed_sm120_backbone import GPURoutedSM120Hook
    return GPURoutedSM120Hook(core)

def register(registry):
    from draftkv.backbones.hooks import HookRegistration
    from draftkv.backbones.catalog import ALGORITHM_REQUIREMENTS
    registry.register(HookRegistration('draftkv',create_hook,
        frozenset(ALGORITHM_REQUIREMENTS['draftkv'])|frozenset({'direct_block_topk_p3','chrono_compact_temporal','chronological_frame_packing','chronological_source_token_packing','all_history_softmax'}),
        hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'streaming_fa2'))
