"""Source-verified B01 GPU temporal RoPE producer adapter for 8x16 DraftKV.

The original C++ ProducerPlan still binds and validates every pinned source
page, device destination and Frontier, and initializes the CUDA wait values.
For pages requiring temporal re-positioning, this adapter enqueues pinned H2D
on the native FETCH stream, then applies 8x16-selected-token GPU K-RoPE, THEN
publishes readiness via the native cuStreamWriteValue32 extension.

No GPU K/V history persists after the layer, no second H2D pass, and the
native B01 Streaming FA2 consumer ABI remains unchanged.
"""
from __future__ import annotations

from draftkv.vendor.b01.integrity import B01Error


class RotatingProducer:
    def __init__(self, original, spec, source, destination, frontier, modules,
                 rotation):
        if ('frontier_streamwrite_ext' not in modules or
                not hasattr(modules['frontier_streamwrite_ext'], 'write_frontier')):
            raise B01Error('MissingDependency',
                           'native stream-ordered Frontier writer is required')
        self.original = original
        self.spec = spec
        self.source = source
        self.destination = destination
        self.frontier = frontier
        self.rotator = rotation
        self.publish = modules['frontier_streamwrite_ext'].write_frontier
        self.entries = spec.entries()
        self.bytes_written = 0

    def initialize(self, value):
        # Identical native cuStreamWriteValue32 initialization on fetch stream.
        self.original.initialize(value)

    def enqueue(self, begin, end):
        if not (0 <= begin <= end <= len(self.entries)):
            raise B01Error('LogicError','GPU K-RoPE producer interval invalid')
        ft = self.spec.frame_tokens
        i=begin
        while i<end:
            # Entries are in group-major then descending-rank order. One
            # pinned H2D copy and one fused RoPE kernel per contiguous group
            # span is substantially cheaper than one Python launch per page.
            # The final Frontier value is published only after ALL pages in
            # this span have completed H2D + GPU temporal position transform.
            group, max_rank, _ = self.entries[i]
            j=i+1
            while (j < end and self.entries[j][0] == group
                   and self.entries[j][1] == self.entries[j-1][1]-1):
                j+=1
            last_group, min_rank, ready_value = self.entries[j-1]
            if last_group != group or min_rank < 0:
                raise B01Error('LogicError','noncontiguous producer span')
            lo, hi = min_rank*ft, (max_rank+1)*ft
            offset = self.spec.history_start
            key_dst = self.destination[0][group, offset+lo:offset+hi]
            val_dst = self.destination[1][group, offset+lo:offset+hi]
            key_src = self.source[0][group, lo:hi]
            val_src = self.source[1][group, lo:hi]
            key_dst.copy_(key_src, non_blocking=True)
            val_dst.copy_(val_src, non_blocking=True)
            # Reuse 900d66d's native GPU RoPE kernel on the complete
            # group-history base so its absolute token-offset metadata is
            # valid even for an early/partial Frontier submission.
            self.rotator.rotate_span_(
                self.destination[0][group, offset:], min_rank, max_rank+1)
            self.publish(self.frontier[group:group+1], ready_value)
            self.bytes_written += key_src.numel() * key_src.element_size() * 2
            i=j

    def bytes(self):
        original = self.original.bytes()
        if self.bytes_written != original:
            raise B01Error('LogicError',
                           'GPU K-RoPE source bytes differ from original ProducerPlan')
        return original
