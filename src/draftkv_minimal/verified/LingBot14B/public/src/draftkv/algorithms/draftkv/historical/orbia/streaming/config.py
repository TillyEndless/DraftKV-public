# SPDX-License-Identifier: Apache-2.0
"""CPU contracts; no retrieval, RoPE, SP reshaping or model policy lives here."""
from __future__ import annotations
from draftkv.streaming.external import external_symbol, external_path
import hashlib
import json
import tarfile
from pathlib import Path
FAMILIES = {'H100': '9.0', 'H200': '9.0', 'B200': '10.0', 'RTX4090': '8.9', 'RTX5090': '12.0'}

def consumer_for(algorithm):
    if algorithm != 'draftkv':
        raise ValueError('Only DraftKV Full is supported')
    return 'StreamingFA2'

def frame_frontiers(block_n, sink_tokens, frame_tokens, frames):
    if block_n not in (32, 64, 128) or min(sink_tokens, frame_tokens, frames) <= 0:
        raise ValueError('Unsupported tile size or empty history')
    first = sink_tokens // block_n
    last = (sink_tokens + frame_tokens * frames - 1) // block_n
    updates = [(rank, first if rank == 0 else (sink_tokens + rank * frame_tokens + block_n - 1) // block_n) for rank in reversed(range(frames))]
    return (first, last, updates)

def transfer_schedule(frame_ids, *, heads, head_group, sink_tokens, frame_tokens, block_n):
    """IDs retain caller order; rank denotes packed position, never absolute frame ID."""
    ids = tuple(frame_ids)
    if not ids or any((type(i) is not int or i < 0 for i in ids)) or len(set(ids)) != len(ids):
        raise ValueError('Unique nonnegative Frame IDs required')
    if type(heads) is not int or type(head_group) is not int or min(heads, head_group) < 1 or heads % head_group:
        raise ValueError('Head group must divide actual local heads')
    (first, last, updates) = frame_frontiers(block_n, sink_tokens, frame_tokens, len(ids))
    return {'first': first, 'last': last, 'initial': last + 1, 'transfers': [{'group': g, 'rank': rank, 'frame_id': ids[rank], 'frontier': value} for g in range(heads // head_group) for (rank, value) in updates]}

def validate_parameters(parameters):
    """Reject settings without an exact implementation in the original consumer."""
    consumer = parameters.get('consumer')
    if consumer not in ('original', 'ready', 'frontier_opt', 'autotune'):
        raise ValueError('UnsupportedArchitecture: original consumer implementation unavailable')
    m = parameters.get('block_m', 64)
    n = parameters.get('block_n')
    splits = parameters.get('splits', 1)
    if consumer == 'autotune':
        if type(m) is not int or type(n) is not int or type(splits) is not int or (m not in (64, 128)) or (n not in (32, 64, 128)) or (not 1 <= splits <= 16) or (splits > 1 and m != 64):
            raise ValueError('UnsupportedArchitecture: tile/split not compiled by original Autotune provider')
    elif n != (32 if consumer == 'ready' else 64) or m != 64 or splits != 1 or (consumer == 'frontier_opt' and parameters.get('variant') != 9):
        raise ValueError('UnsupportedArchitecture: parameters do not match the compiled kernel tile/variant')
    group = parameters.get('head_group')
    gate = parameters.get('gate_tiles')
    if type(group) is not int or not 0 < group < 2 ** 31 or (consumer != 'autotune' and group not in (1, 2, 3, 4, 6, 12)) or (type(gate) is not int) or (not 0 < gate < 2 ** 31) or gate & gate - 1:
        raise ValueError('UnsupportedArchitecture: unsupported group/gate parameters')
    if parameters.get('host_pages') not in ('pytorch', 'base4k', 'thp2m', 'thp2m_slab') or parameters.get('dispatch') not in ('all', 'early') or parameters.get('layout', 'grouped') != 'grouped':
        raise ValueError('UnsupportedArchitecture: page/dispatch/split configuration unavailable in this bridge')
    return parameters

def host_pages_policy(parameters, policy='systems'):
    """Select original pinned storage without changing the frozen system recipe.

    Quality changes only host backing. Systems never falls back after a failed
    THP allocation. Both routes retain the canonical native producer/consumer.
    """
    validate_parameters(parameters)
    if policy not in ('quality', 'systems'):
        raise ValueError('host_pages_policy must be quality or systems')
    result = dict(parameters)
    if policy == 'quality':
        result['host_pages'] = 'pytorch'
    return result

def host_pages_provenance(parameters, policy='systems'):
    if policy not in ('quality', 'systems'):
        raise ValueError('host_pages_policy must be quality or systems')
    if policy == 'quality' and parameters['host_pages'] != 'pytorch':
        raise ValueError('quality host storage must use original PyTorch pinned memory')
    return {'host_pages_backend': parameters['host_pages'], 'host_pages_policy': policy, 'host_pages_role': 'correctness_only' if policy == 'quality' else 'performance_profile'}
