# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
from draftkv.streaming.external import external_symbol, external_path
from pathlib import Path
import json
from dataclasses import dataclass, asdict
from .contracts import BackboneError
DEFAULT_BINDING = Path(__file__).with_name('budget_data.json')

@dataclass(frozen=True)
class EffectiveBudget:
    capacity: int
    retrieved: int
    sink: int
    local: int
    current: int
    block_size: int | None = None
    keep_ratio: float | None = None
    retrieved_equivalent: bool = False
    retrieval_unit: str = 'frame'
    dynamic_retrieval: bool = False
    retrieved_source_frames: int | None = None
    retrieved_equivalent_frames: int | None = None
    compressed_equivalent_per_chunk: float | None = None
    policy_name: str | None = None
    landmark_slots: int = 0

    def __post_init__(self):
        for name in ('capacity', 'retrieved', 'sink', 'local', 'current', 'landmark_slots'):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise BackboneError('BLOCKED_CONFIGURATION', f'invalid budget {name}')
        if self.current < 1 or self.capacity < 1 or self.retrieved + self.sink + self.local + self.current + self.landmark_slots > self.capacity:
            raise BackboneError('KV_CONSTRAINT', 'effective video budget exceeds capacity')
        if self.block_size is not None and (type(self.block_size) is not int or self.block_size < 1):
            raise BackboneError('BLOCKED_CONFIGURATION', 'invalid block_size')
        if self.keep_ratio is not None and (type(self.keep_ratio) not in (float, int) or not 0 < self.keep_ratio <= 1):
            raise BackboneError('BLOCKED_CONFIGURATION', 'invalid keep_ratio')
        if type(self.retrieved_equivalent) is not bool:
            raise BackboneError('BLOCKED_CONFIGURATION', 'retrieved_equivalent must be boolean')
        if self.retrieval_unit not in {'frame', 'block', 'page', 'equivalent_frame', 'landmark'} or type(self.dynamic_retrieval) is not bool:
            raise BackboneError('BLOCKED_CONFIGURATION', 'invalid retrieval unit/dynamic policy')
        if self.policy_name is not None and (not isinstance(self.policy_name, str) or not self.policy_name):
            raise BackboneError('BLOCKED_CONFIGURATION', 'invalid budget policy name')
        if self.retrieved_source_frames is not None:
            source = self.retrieved_source_frames
            equivalent = self.compressed_equivalent_per_chunk
            if type(source) is not int or source <= 0 or source % self.current or (not self.retrieved_equivalent) or (self.retrieval_unit != 'equivalent_frame') or (self.keep_ratio is None) or (equivalent != 1 + (self.current - 1) * self.keep_ratio) or (self.retrieved_equivalent_frames != self.retrieved) or (source // self.current * equivalent != self.retrieved):
                raise BackboneError('KV_CONSTRAINT', 'raw source chunks and compressed equivalent budget differ')
        elif self.retrieved_equivalent_frames is not None or self.compressed_equivalent_per_chunk is not None:
            raise BackboneError('KV_CONSTRAINT', 'compressed budget requires explicit source frames')

    def to_dict(self):
        result = asdict(self)
        if not self.landmark_slots:
            del result['landmark_slots']
        if self.policy_name is None:
            for key in ('retrieval_unit', 'dynamic_retrieval', 'retrieved_source_frames', 'retrieved_equivalent_frames', 'compressed_equivalent_per_chunk', 'policy_name'):
                del result[key]
        return result

    def tokens(self, frame_tokens):
        if type(frame_tokens) is not int or frame_tokens < 1:
            raise BackboneError('KV_CONSTRAINT', 'positive frame_tokens required')
        result = {name: getattr(self, name) * frame_tokens for name in ('capacity', 'retrieved', 'sink', 'local', 'current')}
        if self.landmark_slots:
            result['landmark_slots'] = self.landmark_slots * frame_tokens
        if self.block_size and any((value % self.block_size for value in result.values())):
            raise BackboneError('KV_CONSTRAINT', 'effective budget must align to whole blocks')
        return result

def _binding(binding=DEFAULT_BINDING):
    data = json.loads(Path(binding).read_text())
    if data.get('schema_version') != 1:
        raise BackboneError('BLOCKED_CONFIGURATION', 'unsupported effective-budget schema')
    return data

def author_budget_policy(spec, algorithm, *, binding=DEFAULT_BINDING, preset=None):
    if algorithm != 'draftkv':
        raise BackboneError('BLOCKED_CONFIGURATION', 'Only Full DraftKV is supported')
    if 'draftkv' != 'draftkv' or preset not in (None, 'author_primary'):
        raise BackboneError('BLOCKED_CONFIGURATION', 'Only canonical DraftKV Full is supported')
    return dict(_binding(binding).get('author_policies', {}).get('draftkv', {}))

def effective_budget(spec, algorithm, *, binding=DEFAULT_BINDING, preset=None):
    if algorithm != 'draftkv':
        raise BackboneError('BLOCKED_CONFIGURATION', 'Only Full DraftKV is supported')
    if 'draftkv' != 'draftkv' or preset not in (None, 'author_primary'):
        raise BackboneError('BLOCKED_CONFIGURATION', 'Only canonical DraftKV Full is supported')
    explicit = _binding(binding)['backbones'].get(spec.backbone)
    if explicit is None:
        raise BackboneError('BLOCKED_CONFIGURATION', 'unbound backbone budget')
    budget = EffectiveBudget(**explicit['draftkv'])
    if (budget.capacity, budget.current) != (spec.window, spec.chunk):
        raise BackboneError('KV_CONSTRAINT', 'budget differs from native window/chunk')
    return budget

def bound_budget(core, algorithm):
    if algorithm != 'draftkv':
        raise BackboneError('BLOCKED_CONFIGURATION', 'Only Full DraftKV is supported')
    budget = effective_budget(core.spec, 'draftkv')
    policy = author_budget_policy(core.spec, 'draftkv')
    settings = (core.plan.get('algorithm_plugin') or {}).get('config', {})
    if 'draftkv' == 'infinigen' and budget.policy_name is not None and any((settings.get(key, policy[key]) != policy[key] for key in ('partial_ratio', 'alpha'))):
        raise BackboneError('BLOCKED_CONFIGURATION', 'InfiniGen partial ratio/alpha differ from release policy')
    if budget.block_size is not None and settings.get('block_size', 128) != budget.block_size:
        raise BackboneError('BLOCKED_CONFIGURATION', 'algorithm block size differs from effective budget')
    if budget.keep_ratio is not None and settings.get('keep_ratio', 0.5) != budget.keep_ratio:
        raise BackboneError('BLOCKED_CONFIGURATION', 'compression ratio differs from effective budget')
    supplied = core.plan.get('effective_budget')
    if supplied is not None and supplied != budget.to_dict():
        raise BackboneError('BLOCKED_CONFIGURATION', 'resolved budget differs from release binding')
    supplied_policy = core.plan.get('author_budget_policy')
    if supplied_policy is not None and supplied_policy != policy:
        raise BackboneError('BLOCKED_CONFIGURATION', 'resolved author budget policy differs from release binding')
    return budget
