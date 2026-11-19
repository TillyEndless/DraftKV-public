# SPDX-License-Identifier: Apache-2.0
"""Benchmark architectures and native loader capabilities."""
from __future__ import annotations
from draftkv.streaming.external import external_symbol, external_path
from dataclasses import asdict, dataclass
from .contracts import BackboneError, InputMode

@dataclass(frozen=True)
class ModelSpec:
    backbone: str
    family: str
    task: str
    layers: int
    heads: int
    head_dim: int
    window: int
    sink: int
    chunk: int
    native_sp: tuple[int, ...]
    draftkv_sp: tuple[int, ...]
    checkpoint_id: str
    frame_tokens: int | None = None
    spatial_grid: tuple[int, int] | None = None

    def sp_status(self, sp: int, algorithm: str) -> str:
        supported = self.draftkv_sp if algorithm == 'draftkv' else self.native_sp
        return 'STATIC_CAPABLE_GPU_UNVERIFIED' if type(sp) is int and sp in supported else 'UNSUPPORTED'
MODELS = {'SolarWM33B': ModelSpec('SolarWM33B', 'solarwm', 'minimax-h3-33b', 50, 56, 128, 30, 0, 5, (1, 2, 4, 8), (1, 2, 4, 8), 'junchaoh-cs/SolarWM-H3-33B', 1008, (24, 42)), 'LongLive1.3B': ModelSpec('LongLive1.3B', 'longlive', 't2v', 30, 12, 128, 12, 3, 3, (1, 2, 4, 8), (1, 2, 4, 8), 'LongLive/base+LoRA'), 'LingBot1.3B': ModelSpec('LingBot1.3B', 'lingbot', 'i2v-1.3B', 30, 12, 128, 18, 6, 4, (1, 2, 4, 8), (1, 2, 4, 8), 'robbyant/lingbot-world-v2-1.3b-causal-fast'), 'LingBot14B': ModelSpec('LingBot14B', 'lingbot', 'i2v-A14B', 40, 40, 128, 18, 6, 4, (1, 2, 4, 8), (1, 2, 4, 8), 'robbyant/lingbot-world-v2-14b-causal-fast')}
ROUTE_MODES = {('bench1', 'LongLive1.3B'): InputMode.BENCH1_T2V, ('bench2', 'LingBot1.3B'): InputMode.ORBIA, ('bench2', 'LingBot14B'): InputMode.ORBIA, ('bench2', 'SolarWM33B'): InputMode.ORBIA}
ALGORITHM_REQUIREMENTS = {'draftkv': ('draftattention2_postrope_spatial8x16_meanmax', 'hierarchical_v1', 'original_b01_producer_plan', 'asynchronous_h2d', 'cuda_frontier', 'streaming_fa2', 'clean_commit', 'chrono_compact_temporal', 'chronological_frame_packing')}
from draftkv.algorithms.draftkv.historical.orbia.algorithm.block_policy import PROTECTED_RECENT_BY_BACKBONE, configured_protection
BLOCK_DRAFTKV_BACKBONES = frozenset(PROTECTED_RECENT_BY_BACKBONE)

def algorithm_requirements(algorithm, backbone):
    requirements = ALGORITHM_REQUIREMENTS[algorithm]
    if algorithm == 'draftkv' and backbone in BLOCK_DRAFTKV_BACKBONES:
        replace = {'hierarchical_v1': f'direct_block_topk_p{configured_protection(backbone)}', 'chronological_frame_packing': 'chronological_source_token_packing'}
        return tuple((replace.get(value, value) for value in requirements)) + ('all_history_softmax',)
    return requirements

def model_spec(backbone: str) -> ModelSpec:
    try:
        return MODELS[backbone]
    except KeyError as exc:
        raise BackboneError('UNSUPPORTED', f'unknown backbone {backbone}') from exc
