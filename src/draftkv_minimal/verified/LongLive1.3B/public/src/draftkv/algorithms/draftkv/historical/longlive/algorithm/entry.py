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
    if core.spec.backbone != 'LongLive1.3B':
        raise BackboneError('BLOCKED_CONFIGURATION', 'Wrong historical source profile')
    return BlockDraftKVHook(core)
