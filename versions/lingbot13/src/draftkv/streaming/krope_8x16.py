"""8x16 DraftKV temporal K relocation using already-vendored GPU RoPE kernels.

Selection is performed entirely by the existing Full A2 Camera→Draft hook;
only the position transform of selected source-ordered K moves from CPU to the
B01 fetch stream, after H2D and before CUDA Frontier publication.

The Wan and H3 mathematical kernels are the existing 900d66d-era implementations
in prepacked.py and temporal_kernels.py, not a new retrieval or RoPE kernel.
"""
from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass

from draftkv.vendor.b01.integrity import B01Error


def _rotate(payload, offsets, slots, phase, count, head_group, family, *, preload):
    if not count:
        return
    if family == "h3":
        from draftkv.streaming.temporal_kernels import rotate_h3_fetched
        return rotate_h3_fetched(payload, offsets, slots, phase, count,
                                 head_group, preload=preload)
    if family == "wan":
        from draftkv.streaming.prepacked import _rotate_fetched
        return _rotate_fetched(payload, offsets, slots, phase, count,
                               head_group, heads=head_group, preload=preload)
    raise B01Error("LogicError", "unrecognized temporal RoPE family")


def phase_table(geometry, family, device):
    """Reuse exact native post-RoPE coefficients and original kernel dtype."""
    import torch
    if family == "solarwm":
        c, s = geometry.coefficients
        # The native H3 operation casts cos/sin to the raw BF16 K dtype
        # BEFORE multiplication. The 900d66d fetched kernel expects FP32
        # coefficients, so pass those BF16-rounded values promoted to FP32.
        return torch.cat((c.to(torch.bfloat16), s.to(torch.bfloat16)), -1).to(
            device=device, dtype=torch.float32).contiguous()
    if family in ("lingbot", "longlive"):
        return torch.view_as_real(geometry.coefficients[0]).flatten(-2).to(
            device=device, dtype=torch.float64).contiguous()
    raise B01Error("LogicError", "unsupported temporal RoPE coefficient family")


@dataclass(frozen=True)
class GpuKRoPEPlan:
    # Source-token offsets in contiguous final DMA history and exact target
    # Chrono-Compact temporal positions. They omit unchanged source tokens.
    rows: tuple
    offsets: object
    slots: object
    phases: object
    frame_tokens: int
    sink_frames: int
    family: str
    head_group: int
    rotated_ranks: tuple

    @classmethod
    def from_flags(cls, flags, *, sink_frames, frame_tokens, family,
                   head_group, geometry, device):
        import torch
        if flags.dtype != torch.uint8 or flags.device.type != "cpu":
            raise B01Error("LogicError", "Chrono-Compact source change flags must be CPU uint8")
        if flags.numel() % frame_tokens:
            raise B01Error("LogicError", "8x16-selected history flags not frame aligned")
        shifted = flags.nonzero().flatten().to(torch.long)
        # Keep the sorted CPU index vector in NumPy rather than materializing
        # tens of thousands of Python ints at every layer / diffusion step.
        rows = shifted.numpy()
        offsets = (shifted * head_group * 128).to(device=device, dtype=torch.int64)
        slots = (sink_frames + shifted // frame_tokens).to(device=device, dtype=torch.int32)
        rotated = tuple(bool(flags[i:i + frame_tokens].any())
                        for i in range(0, flags.numel(), frame_tokens))
        return cls(rows=rows, offsets=offsets, slots=slots,
                   phases=phase_table(geometry, family, device),
                   frame_tokens=frame_tokens, sink_frames=sink_frames,
                   family="h3" if family == "solarwm" else "wan",
                   head_group=head_group, rotated_ranks=rotated)

    def validate(self, source, spec, device):
        import torch
        if (source is None or
                self.frame_tokens != spec.frame_tokens or
                # H3 has an unshifted native condition/audio prefix before
                # Sink. The source history offset includes that prefix, but
                # compact temporal positions must NOT count its tokens.
                self.sink_frames * self.frame_tokens > spec.history_start or
                self.head_group != (spec.heads if spec.consumer == "token" else spec.head_group) or
                len(self.rotated_ranks) != len(spec.frame_ids) or
                self.family not in ("wan", "h3") or
                not isinstance(self.offsets, torch.Tensor) or
                not isinstance(self.slots, torch.Tensor) or
                self.offsets.dtype != torch.int64 or self.slots.dtype != torch.int32 or
                self.offsets.device != device or self.slots.device != device or
                not self.offsets.is_contiguous() or not self.slots.is_contiguous() or
                self.offsets.numel() != self.slots.numel() or
                self.offsets.numel() != len(self.rows) or
                (len(self.rows) > 0 and (self.rows[0] < 0 or
                                self.rows[-1] >= len(spec.frame_ids) * spec.frame_tokens)) or
                not isinstance(self.phases, torch.Tensor) or
                self.phases.device != device or not self.phases.is_contiguous() or
                self.phases.shape[0] < self.sink_frames + len(spec.frame_ids) or
                self.phases.shape[1] != (32 if self.family == "h3" else 44)):
            raise B01Error("LogicError", "B01 GPU K-RoPE metadata does not match bound history")
        return self

    def in_physical_order(self, physical_to_logical):
        """Relocate temporal positions without changing canonical Chrono ranks.

        Packed archives store every historical temporal component raw, so the
        rotation plan must cover all selected tokens. Physical offsets stay
        dense; each offset receives its ORIGINAL logical token's compact slot.
        """
        from dataclasses import replace
        n=len(self.rotated_ranks)*self.frame_tokens
        if (physical_to_logical.numel()!=n or len(self.rows)!=n or
                (n and (self.rows[0]!=0 or self.rows[-1]!=n-1))):
            raise B01Error('LogicError','physical archive rotation requires complete raw temporal coverage')
        return replace(self,slots=self.slots.index_select(0,physical_to_logical).contiguous())

    def _span(self, first_rank, end_rank):
        if not 0 <= first_rank <= end_rank <= len(self.rotated_ranks):
            raise B01Error("LogicError", "invalid K-RoPE compact history span")
        i = bisect_left(self.rows, first_rank * self.frame_tokens)
        j = bisect_left(self.rows, end_rank * self.frame_tokens)
        return i, j

    def rotate_span_(self, group_history, first_rank, end_rank):
        i, j = self._span(first_rank, end_rank)
        if i != j:
            _rotate(group_history, self.offsets[i:j], self.slots[i:j],
                    self.phases, j - i, self.head_group, self.family, preload=False)

    def rotate_tokens_(self, group_history, begin, end, *, preload=False):
        """Existing math on a split's exact token range, including partial frames.

        Offsets stay relative to the complete history allocation. Splits are
        disjoint, so no selected K is rotated twice at a split boundary.
        """
        if not 0 <= begin <= end <= len(self.rotated_ranks)*self.frame_tokens:
            raise B01Error('LogicError', 'invalid K-RoPE token span')
        i, j = bisect_left(self.rows, begin), bisect_left(self.rows, end)
        if i != j:
            _rotate(group_history, self.offsets[i:j], self.slots[i:j],
                    self.phases, j-i, self.head_group, self.family, preload=preload)

    def preload(self, destination, spec):
        """Preload all needed source-bound GPU RoPE specializations BEFORE FA waits."""
        entries = spec.entries()
        # Actual B01 early schedule submits 1 page first, then remaining pages.
        # Also preload the complete span used by serial/full-dispatch mode.
        intervals = ((0, 1), (1, len(entries)), (0, len(entries)))
        seen = set()
        for begin, end in intervals:
            if begin >= end:
                continue
            index = begin
            while index < end:
                group, max_rank, _ = entries[index]
                stop = index + 1
                while (stop < end and entries[stop][0] == group
                       and entries[stop][1] == entries[stop-1][1] - 1):
                    stop += 1
                _, min_rank, _ = entries[stop-1]
                i, j = self._span(min_rank, max_rank + 1)
                n = j - i
                if n and n not in seen:
                    seen.add(n)
                    payload = destination[group, spec.history_start:]
                    _rotate(payload, self.offsets[i:j], self.slots[i:j],
                            self.phases, n, self.head_group, self.family, preload=True)
                index = stop
        return self


def cpu_reference_for_validation(plan, source_key, *, rank):
    """Slow validation-only replay of native Chrono-Compact CPU K rotation."""
    import torch
    key = source_key.clone()
    ft = plan.frame_tokens
    begin, end = plan._span(rank, rank + 1)
    if begin == end:
        return key
    positions = torch.tensor(plan.rows[begin:end], dtype=torch.long) - rank * ft
    c = plan.phases[plan.sink_frames + rank].cpu()
    if plan.family == "h3":
        cos, sin = c[:16].bfloat16(), c[16:].bfloat16()
        a, b = key[positions, :, :16].clone(), key[positions, :, 48:64].clone()
        key[positions, :, :16] = a * cos + (-b) * sin
        key[positions, :, 48:64] = b * cos + a * sin
    else:
        z = torch.view_as_complex(key[positions, :, :44].double().reshape(
            len(positions), key.shape[1], 22, 2).contiguous())
        coeff = torch.view_as_complex(c.reshape(22, 2).contiguous())
        key[positions, :, :44] = torch.view_as_real(z * coeff).flatten(-2).bfloat16()
    return key
