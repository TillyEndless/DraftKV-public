# SPDX-License-Identifier: Apache-2.0
"""Single source of truth for the canonical model-specific Direct Block Top-K policy.

P is the number of *complete* recent historical latent frames protected from
block ranking. It is not a different scoring algorithm or a capacity override.
"""
from __future__ import annotations
from draftkv.streaming.external import external_symbol, external_path
PROTECTED_RECENT_BY_BACKBONE = {'LingBot14B': 3, 'LingBot1.3B': 3, 'LongLive1.3B': 2, 'SolarWM33B': 5}

def configured_protection(backbone):
    try:
        return PROTECTED_RECENT_BY_BACKBONE[backbone]
    except KeyError as error:
        raise ValueError(f'unknown Block Top-K backbone: {backbone!r}') from error

def applied_protection(backbone, history_budget):
    if type(history_budget) is not int or history_budget < 1:
        raise ValueError('DRAFTKV_INVALID_HISTORY_BUDGET')
    return min(configured_protection(backbone), history_budget)
