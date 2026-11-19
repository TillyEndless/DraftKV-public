# SPDX-License-Identifier: Apache-2.0
import hashlib
from pathlib import Path
from draftkv.algorithms.draftkv.config import config, validate_deployment
from .block_backbone import BlockDraftKVHook
from ..backbones.contracts import BackboneError

def configure_retrieval(backbone, settings):
    c = config(backbone)
    validate_deployment(c, {'effective_budget': c['budget'], 'algorithm_plugin': {'entry': 'src/draftkv/algorithms/draftkv/entry.py', 'config': settings}})
    return dict(settings)

def create_hook(core):
    configure_retrieval(core.spec.backbone, core.plan['algorithm_plugin']['config'])
    if core.spec.backbone == 'LingBot14B':
        from .camera_frustum_v2 import CameraGatedBlockDraftHook
        return CameraGatedBlockDraftHook(core)
    if core.spec.backbone in ('LingBot1.3B', 'SolarWM33B'):
        from .camera_frustum_multibackbone import CameraGatedMultiBackboneHook
        return CameraGatedMultiBackboneHook(core)
    raise BackboneError('BLOCKED_CONFIGURATION', 'Wrong historical source profile')
