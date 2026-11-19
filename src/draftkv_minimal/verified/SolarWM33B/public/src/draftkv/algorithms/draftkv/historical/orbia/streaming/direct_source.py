# SPDX-License-Identifier: Apache-2.0
"""Verified, final-layout CPU source for B01's original grouped ProducerPlan.

The DraftKV block selector fills exactly this pinned Group-major tensor. It is
NOT a prepacked intermediate: the native DMA reads this very allocation.
"""
from __future__ import annotations
from draftkv.streaming.external import external_symbol, external_path
from dataclasses import dataclass

@dataclass(frozen=True)
class DirectGroupMajorSource:
    key: object
    value: object
    frame_ids: tuple
    frame_tokens: int
    history_start: int
    kv_tokens: int
    heads: int
    groups: int
    head_group: int

    def verify(self, spec):
        B01Error = external_symbol('orbia', 'draftkv.vendor.b01.integrity', 'B01Error')
        import torch
        if self.frame_ids != spec.frame_ids or self.frame_tokens != spec.frame_tokens or self.history_start != spec.history_start or (self.kv_tokens != spec.kv_tokens) or (self.heads != spec.heads) or (self.groups != spec.groups) or (self.head_group != (spec.heads if spec.consumer == 'token' else spec.head_group)):
            raise B01Error('LogicError', 'prepacked direct CPU source does not match bound B01 plan')
        shape = (spec.groups, len(spec.frame_ids) * spec.frame_tokens, self.head_group, 128)
        if any((not isinstance(t, torch.Tensor) or tuple(t.shape) != shape or t.dtype != torch.bfloat16 or (t.device.type != 'cpu') or (not t.is_contiguous()) or (not t.is_pinned()) for t in (self.key, self.value))):
            raise B01Error('LogicError', 'direct Group-major K/V must be contiguous pinned BF16 pages')
        if self.key.data_ptr() == self.value.data_ptr():
            raise B01Error('LogicError', 'direct Group-major K/V must not overlap')
        return (self.key, self.value)
