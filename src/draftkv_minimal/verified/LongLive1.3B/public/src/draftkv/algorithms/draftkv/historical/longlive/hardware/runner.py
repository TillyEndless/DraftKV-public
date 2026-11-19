# SPDX-License-Identifier: Apache-2.0
"""CPU orchestration around an explicitly supplied original-B01 target adapter.

The adapter owns real checkpoint capture, C++ DMA, Frontier, native FA2, CUDA
events and rank coordination. There is deliberately no Torch/default provider.
All measurement returns are hash-addressed files, never synthetic estimates.
"""
from __future__ import annotations
from draftkv.streaming.external import external_symbol, external_path
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time
from typing import Protocol
from .profiles import ProfileError, check_config, checked, digest, read_json, require, request_family, verify_artifact, verify_binding, portable, seal

@dataclass(frozen=True)
class TunePolicy:
    max_candidates: int = 8
    warmup: int = 2
    screen_repeats: int = 3
    confirm_pairs: int = 9
    replay_pairs: int = 5
    validate_repeats: int = 2
    frontier_timeout_s: float = 10.0
    total_timeout_s: float = 180.0
    min_gain: float = 0.01
    max_p90_regression: float = 0.1

    def __post_init__(self):
        for (key, lo, hi) in (('max_candidates', 1, 8), ('warmup', 1, 5), ('screen_repeats', 3, 9), ('confirm_pairs', 9, 31), ('replay_pairs', 5, 15), ('validate_repeats', 2, 4)):
            require(type(getattr(self, key)) is int and lo <= getattr(self, key) <= hi, 'BLOCKED_PROFILE', 'invalid lightweight tuner budget: ' + key)
        for (key, lo, hi) in (('frontier_timeout_s', 0.1, 30), ('total_timeout_s', 1, 600), ('min_gain', 0.01, 0.5), ('max_p90_regression', 0, 0.1)):
            number(getattr(self, key), lo, hi, key)

def number(value, low, high, label):
    require(type(value) in {int, float} and math.isfinite(value) and (low <= value <= high), 'BLOCKED_PROFILE', 'invalid measured value: ' + label)
    return value

def history_bytes(shape):
    return 2 * 2 * shape['batch'] * shape['history_tokens'] * shape['kv_heads'] * shape['dim']

def observation(record, root):
    checked(record, 'artifact')
    return read_json(verify_artifact(record, root))

def bf16_ulp(value):
    """Spacing of BF16 numbers (8 significant bits) at magnitude ``value``."""
    return 2.0 ** (math.floor(math.log2(value)) - 7) if value > 0 else 0.0

def output_abs_limit(row):
    """Absolute output gate: 2^-5, widened to one BF16 ULP of the largest reference output.

    A BF16 result cannot be closer than half an ULP to an FP32 reference, so for
    outputs >= 8 (e.g. LingBot layer 39, |out| up to ~15) even native FA2 exceeds
    2^-5. Relative (rel_l2, normalized_max) and LSE gates are unchanged.
    """
    magnitude = row.get('dense_output_abs_max')
    if magnitude is None:
        return 0.03125
    number(magnitude, 0, 1000000000.0, 'dense_output_abs_max')
    return max(0.03125, bf16_ulp(magnitude))

def validate_observation(data, request, config):
    """Require real BF16 same-input dense errors plus byte-exact KV and NaN checks."""
    checked(data, 'validation')
    require(data.get('execution') == 'CUDA' and data.get('request_sha256') == digest(request) and (data.get('config_sha256') == digest(config)), 'BLOCKED_PROFILE', 'validation context differs')
    rows = data.get('ranks')
    require(isinstance(rows, list) and len(rows) == len(request['ranks']), 'BLOCKED_PROFILE', 'missing golden ranks')
    for (index, (row, rank)) in enumerate(zip(rows, request['ranks'])):
        require(row.get('rank') == index and row.get('input_kind') == 'REAL_CHECKPOINT_QKV' and (row.get('dtype') == 'bf16'), 'BLOCKED_PROFILE', 'real per-rank BF16 checkpoint QKV required')
        for key in ('q_sha256', 'k_sha256', 'v_sha256', 'dense_output_sha256', 'dense_lse_sha256'):
            checked({'path': 'tensor', 'sha256': row.get(key)}, 'artifact')
        ref = row.get('reference')
        require(ref in ('SAME_INPUT_DENSE', 'SAME_INPUT_NATIVE_FA2_LSE_DENSE_FP32_OUTPUT_V2') and row.get('selected_ids_unchanged') is True, 'BLOCKED_PROFILE', 'Numerical reference or selected KV IDs changed')
        if ref == 'SAME_INPUT_NATIVE_FA2_LSE_DENSE_FP32_OUTPUT_V2':
            checked({'path': 'native-fp32-lse', 'sha256': row.get('native_lse_sha256')}, 'artifact')
            number(row.get('dense_lse_max_abs'), 0, 1000000000.0, 'dense_lse_max_abs')
        else:
            require('native_lse_sha256' not in row and 'dense_lse_max_abs' not in row, 'BLOCKED_PROFILE', 'legacy dense contract must not mix native LSE fields')
        for tensor in ('q', 'k', 'v'):
            require(row.get(tensor + '_after_sha256') == row[tensor + '_sha256'], 'BLOCKED_PROFILE', 'Q/K/V input changed during validation')
        for tensor in ('k', 'v'):
            require(row.get(tensor + '_destination_sha256') == row[tensor + '_sha256'], 'BLOCKED_PROFILE', 'retrieved KV bytes differ from expected logical tensor')
        expected = history_bytes(rank['shape'])
        resident = rank['shape']['history_tokens'] == 0
        require(type(row.get('h2d_bytes')) is int and row['h2d_bytes'] == expected and (type(row.get('nan_poisoned_bytes')) is int) and (row['nan_poisoned_bytes'] == expected) and (row.get('resident_zero_history') is True if resident else 'resident_zero_history' not in row), 'BLOCKED_PROFILE', 'historical KV payload/NaN poison coverage differs; zero-H2D Resident Golden must be explicitly marked')
        if resident:
            require(row['reference'] == 'SAME_INPUT_NATIVE_FA2_LSE_DENSE_FP32_OUTPUT_V2' and row['k_destination_sha256'] == row['k_sha256'] and (row['v_destination_sha256'] == row['v_sha256']), 'BLOCKED_PROFILE', 'Resident must consume unchanged real model KV with Native-FA2 LSE reference')
        require(type(row.get('nan_survivors')) is int and row['nan_survivors'] == 0 and (row.get('output_finite') is True) and (row.get('lse_finite') is True), 'BLOCKED_PROFILE', 'NaN contamination or nonfinite output/LSE')
        require(row.get('frontier_stalled') is False and row.get('final_join') is True, 'GPU_GATE_REQUIRED', 'Frontier stalled or missing producer/consumer join; discard worker')
        for (key, maximum) in (('output_max_abs', output_abs_limit(row)), ('output_rel_l2', 0.002), ('output_normalized_max', 0.5), ('lse_max_abs', 0.0001)):
            number(row.get(key), 0, maximum, key)
    return data

def measurement(data, request, config, *, phase, pair_id, order, comparison_sha256=None):
    checked(data, 'measurement')
    require(data.get('execution') == 'CUDA' and data.get('request_sha256') == digest(request) and (data.get('config_sha256') == digest(config)) and (data.get('phase') == phase) and (data.get('pair_id') == pair_id) and (data.get('order') == order) and (data.get('comparison_sha256') == (comparison_sha256 or digest(config))), 'BLOCKED_PROFILE', 'measurement order/context differs')
    rows = data.get('ranks')
    require(isinstance(rows, list) and len(rows) == len(request['ranks']), 'BLOCKED_PROFILE', 'missing measured ranks')
    walls = []
    for (index, (row, rank)) in enumerate(zip(rows, request['ranks'])):
        require(row.get('frontier_stalled') is False and row.get('final_join') is True, 'GPU_GATE_REQUIRED', 'Frontier stalled or unjoined streams; discard worker')
        require(row.get('rank') == index and row.get('producer') == 'cpp_dma' and (row.get('consumer') == 'original_b01') and (row.get('final_join') is True) and (row.get('frontier_stalled') is False) and (row.get('page_backing_verified') is True) and (row.get('h2d_bytes') == history_bytes(rank['shape'])) and (row.get('timing_scope') == 'DMA_RESIDENT_PACK_FA2_MERGE_JOIN'), 'BLOCKED_PROFILE', 'missing real DMA/FA2/scope/page/Frontier evidence')
        walls.append(number(row.get('wall_ms'), 1e-06, 1000000.0, 'wall_ms'))
        number(row.get('gpu_critical_ms'), 1e-06, row['wall_ms'], 'gpu_critical_ms')
        for key in ('dma_ms', 'fa2_ms', 'resident_pack_ms', 'split_merge_ms'):
            number(row[key], 1e-06 if key in {'dma_ms', 'fa2_ms'} else 0, row['wall_ms'], key)
        s = rank['shape']
        require(row['h2d_copies'] == 2 * (s['history_tokens'] // s['frame_tokens']) * (s['heads'] // config['head_group']), 'BLOCKED_PROFILE', 'group/frame K+V DMA copy count differs')
        require((row['split_merge_ms'] == 0) == (config['splits'] == 1), 'BLOCKED_PROFILE', 'Split merge timing does not match actual configuration')
    return max(walls)

def p90(values):
    ordered = sorted(values)
    return ordered[math.ceil(0.9 * len(ordered)) - 1]

def compare_pairs(pairs, policy):
    old = [p[0] for p in pairs]
    new = [p[1] for p in pairs]
    gains = [(a - b) / a for (a, b) in pairs]
    rng = random.Random(20261026)
    boot = sorted((statistics.mean(rng.choices(gains, k=len(gains))) for _ in range(1000)))
    lower = boot[24]
    gain = 1 - statistics.median(new) / statistics.median(old)
    tail = p90(new) / p90(old) - 1
    return {'old_median_max_rank_wall_ms': statistics.median(old), 'candidate_median_max_rank_wall_ms': statistics.median(new), 'gain': gain, 'paired_bootstrap_95_lower': lower, 'p90_regression': tail, 'accepted': gain >= policy.min_gain and lower > 0 and (tail <= policy.max_p90_regression)}

def verify_tuning_evidence(record, profile, request, root):
    """Recompute A/B decisions from hashed raw samples before GPU-run authorization."""
    result = observation(record, root)
    require(result.get('mocked') is False and result.get('status') == 'CUDA_GOLDEN' and (result.get('validation_level') == 'CUDA_GOLDEN') and (result.get('request_sha256') == digest(request)) and (result.get('winner') == profile['config']), 'GPU_GATE_REQUIRED', 'missing real exact-shape H100/B200 tuning evidence')
    baseline = checked(result.get('baseline'), 'profile')
    require(result.get('baseline_profile_sha256') == baseline['sha256'] and baseline['key'] == request and (baseline['source_files'] == profile['source_files']) and (baseline['kernel'] == profile['kernel']), 'GPU_GATE_REQUIRED', 'old profile A/B binding differs')
    for config in (baseline['config'], profile['config']):
        records = result.get('validations', {}).get(digest(config), [])
        require(len(records) >= 2, 'GPU_GATE_REQUIRED', 'old/candidate independent numerical validations required')
        seeds = []
        for raw in records:
            seeds.append(validate_observation(observation(raw, root), request, config)['poison_seed'])
        require(len(set(seeds)) == len(seeds), 'GPU_GATE_REQUIRED', 'duplicate numerical validation replay')
    if baseline['config'] == profile['config']:
        require(result.get('recommendation', {}).get('changed') is False, 'GPU_GATE_REQUIRED', 'baseline retention mislabeled as a new winner')
        return result
    comparisons = [c for c in result.get('comparisons', []) if c.get('config') == profile['config']]
    require(len(comparisons) == 1, 'GPU_GATE_REQUIRED', 'winner lacks paired A/B comparison')
    for (phase, minimum) in (('confirm', 9), ('paired_replay', 5)):
        samples = [s for s in result.get('samples', []) if s.get('phase') == phase and s.get('comparison_sha256') == digest(profile['config'])]
        require(len(samples) >= minimum * 2 and len(samples) % 2 == 0, 'GPU_GATE_REQUIRED', 'incomplete paired A/B sample set')
        pairs = []
        for pair_id in range(len(samples) // 2):
            rows = [s for s in samples if s.get('pair_id') == pair_id]
            require(len(rows) == 2 and {s.get('order') for s in rows} == {0, 1}, 'GPU_GATE_REQUIRED', 'paired replay order or ranks missing')
            values = {}
            for sample in rows:
                config = baseline['config'] if sample['order'] == pair_id % 2 else profile['config']
                require(sample.get('config_sha256') == digest(config), 'GPU_GATE_REQUIRED', 'A/B order changed')
                values['old' if config == baseline['config'] else 'new'] = measurement(observation(sample['raw'], root), request, config, phase=phase, pair_id=pair_id, order=sample['order'], comparison_sha256=digest(profile['config']))
            pairs.append((values['old'], values['new']))
        recomputed = compare_pairs(pairs, TunePolicy())
        require(recomputed['accepted'] and recomputed == comparisons[0].get(phase), 'GPU_GATE_REQUIRED', 'A/B evidence does not support the reported winner')
    return result
