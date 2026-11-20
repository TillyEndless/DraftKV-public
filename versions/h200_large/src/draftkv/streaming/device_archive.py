"""Commit-published pointer directory and fixed-shape GPU archive plans.

The host registers new immutable allocations after D2H completion. A route
never visits the host bank, reads selected IDs, or sizes a buffer from a GPU
scalar. The existing AsyncProducer consumes the resulting DMA descriptors.
"""
from dataclasses import dataclass
from .canonical_prepack import PackedArchiveSource, LogicalArchiveTensor, token_layout


class ArchiveDirectory:
    def __init__(self, h, w, heads, group, device):
        import torch
        self.h, self.w, self.ft = h, w, h*w
        self.heads, self.group, self.device = heads, group, device
        blocks, inverse, starts = token_layout(h, w)
        self.regions = len(blocks)
        self.mapping = torch.tensor(inverse, device=device, dtype=torch.int64)
        self.starts = torch.tensor(starts, device=device, dtype=torch.int64)
        self.sizes = torch.tensor([len(b) for b in blocks], device=device, dtype=torch.int64)
        self.pointers = torch.empty((0, 2), device=device, dtype=torch.int64)
        self.owners = ()
        self.frontier = 0

    def publication(self, rows, descriptors):
        """Build a private successor using ONLY the newly staged chunk owners."""
        import torch
        from copy import copy
        ids = tuple(rows)
        if not ids or ids != tuple(range(ids[0], ids[0]+len(ids))):
            raise ValueError('device archive publication must be a contiguous chunk')
        if ids[0] > self.frontier:
            raise ValueError('device archive publication skips the clean frontier')
        pointers, owners, allocation_bases = [], [], {}
        for fid in ids:
            key, value = rows[fid]
            if (not isinstance(key, LogicalArchiveTensor) or not isinstance(value, LogicalArchiveTensor)
                    or key.frame is not value.frame or key.side != 0 or value.side != 1):
                raise ValueError('device archive requires paired immutable packed owners')
            f = key.frame
            if ((f.h, f.w, f.heads, f.group) != (self.h, self.w, self.heads, self.group)
                    or not f.slab.is_pinned() or f.slab.element_size() != 2):
                raise ValueError('device archive publication geometry/pinning changed')
            desc = descriptors[fid]
            if (desc.shape != (2, 1, self.heads, self.regions, 128)
                    or desc.dtype != torch.float16 or not desc.is_contiguous()
                    or desc.device != self.device):
                raise ValueError('device archive descriptor layout changed')
            # One allocation registration per new frame, never per block/group.
            # Native sinks are never selected; retain their allocation normally.
            identity = id(f.slab)
            if identity not in allocation_bases:
                allocation_bases[identity] = f.slab.data_ptr()
            pointers.append((allocation_bases[identity]+f.frame*2*self.ft*self.heads*128*2,
                             desc.data_ptr()))
            owners.append((f, desc))
        tail = torch.tensor(pointers, device=self.device, dtype=torch.int64)
        new = copy(self)
        end = ids[-1]+1
        if ids[0] == self.frontier:
            new.pointers = torch.cat((self.pointers, tail), 0)
        else:
            if end > self.frontier:
                raise ValueError('replacement publication overlaps the frontier')
            new.pointers = self.pointers.clone()
            new.pointers[ids[0]:end].copy_(tail)
        new.frontier = max(self.frontier, end)
        # Ownership is lifetime bookkeeping; no route traverses this tuple.
        new.owners = self.owners + (tuple(owners),)
        return new


_KERNELS = None


def kernels():
    global _KERNELS, tl
    if _KERNELS is not None:
        return _KERNELS
    import triton
    import triton.language as tl

    @triton.jit
    def gather_descriptors(Pointers, Frames, Out, C:tl.constexpr, H:tl.constexpr,
                           R:tl.constexpr, B:tl.constexpr):
        x = tl.program_id(0)*B+tl.arange(0, B)
        head, side = tl.program_id(1), tl.program_id(2)
        frame = x//(R*128)
        offset = x % (R*128)
        fid = tl.load(Frames+frame, frame<C, 0)
        base = tl.load(Pointers+fid*2+1, frame<C, 0).to(tl.pointer_type(tl.float16))
        v = tl.load(base+(side*H+head)*R*128+offset, frame<C, 0)
        tl.store(Out+(side*H+head)*C*R*128+x, v, frame<C)

    @triton.jit
    def compact_ids(Mask, Rank, Frames, Out, FT:tl.constexpr, N:tl.constexpr,
                    B:tl.constexpr):
        i = tl.program_id(0)*B+tl.arange(0, B)
        keep = tl.load(Mask+i, i<N, 0)
        rank = tl.load(Rank+i, (i<N)&keep, 0)
        fid = tl.load(Frames+i//FT, (i<N)&keep, 0)
        tl.store(Out+rank, fid*FT+i%FT, (i<N)&keep)

    @triton.jit
    def run_starts(Begin, Rank, Starts, N:tl.constexpr, B:tl.constexpr):
        i = tl.program_id(0)*B+tl.arange(0, B)
        begin = tl.load(Begin+i, i<N, 0)
        rank = tl.load(Rank+i, (i<N)&begin, 0)
        tl.store(Starts+rank, i, (i<N)&begin)
    _KERNELS = gather_descriptors, compact_ids, run_starts
    return _KERNELS


def gather_descriptors(directory, frames):
    import torch
    import triton
    c = frames.numel()
    torch._assert_async(((frames>=0)&(frames<directory.frontier)).all(), 'unpublished descriptor frame')
    out = torch.empty((2, 1, directory.heads, c*directory.regions, 128),
                      device=directory.device, dtype=torch.float16)
    kernels()[0][(triton.cdiv(c*directory.regions*128, 256), directory.heads, 2)](
        directory.pointers, frames, out, c, directory.heads, directory.regions, 256)
    return out


@dataclass(frozen=True)
class DeviceDecision:
    device_tokens: object
    candidates: object
    take_by_block: object
    ranked_blocks: object
    scores: object
    history_budget: int
    protected_count: int


def select_device(scores, frames, budget, protected, directory):
    """Exact stable ranking and boundary crop; all tensor shapes are known."""
    import torch
    import triton
    c, r, ft = frames.numel(), directory.regions, directory.ft
    if scores.shape != (c*r,) or not 0 <= protected <= budget <= c or budget == 0:
        raise ValueError('device selection shape/budget/protection mismatch')
    torch._assert_async((torch.isfinite(scores)&(scores>=0)).all(), 'nonfinite/negative Draft score')
    eligible = (c-protected)*r
    rank = torch.argsort(scores[:eligible], descending=True, stable=True)
    counts = directory.sizes[rank % r]
    target = (budget-protected)*ft
    take = torch.minimum(counts, (target-(counts.cumsum(0)-counts)).clamp_min(0))
    takes = torch.zeros(c*r, device=scores.device, dtype=torch.int64)
    takes.scatter_(0, rank, take)
    if protected:
        takes[eligible:] = directory.sizes.repeat(protected)
    mapping = directory.mapping
    mask = (mapping[:, 1][None, :] < takes.view(c, r)[:, mapping[:, 0]]).flatten()
    compact_rank = mask.to(torch.int64).cumsum(0)-1
    torch._assert_async(mask.sum() == budget*ft, 'incomplete selected history')
    ids = torch.empty(budget*ft, device=scores.device, dtype=torch.int64)
    kernels()[1][(triton.cdiv(c*ft, 256),)](mask, compact_rank, frames, ids, ft, c*ft, 256)
    return DeviceDecision(ids, frames, takes, rank, scores, budget, protected)


@dataclass(frozen=True)
class DeviceArchiveSource(PackedArchiveSource):
    directory: object = None

    def verify(self, spec):
        import torch
        for name in ('frame_ids','frame_tokens','history_start','kv_tokens','heads','groups'):
            if getattr(self,name) != getattr(spec,name):
                raise ValueError('device archive differs from bound plan: '+name)
        if self.head_group != spec.head_group or self.directory is None:
            raise ValueError('invalid device archive head group/owner')
        p = self.physical_to_logical
        if (p.shape != (self.history_tokens,) or p.dtype != torch.int64
                or p.device != self.metadata.device or not p.is_contiguous()
                or self.metadata.ndim != 3 or self.metadata.shape[0] != self.groups
                or self.metadata.shape[2] != 4):
            raise ValueError('invalid device archive permutation/spans')


def bind_device_archive(directory, decision, *, history_start, total):
    """GPU source IDs -> physical addresses, permutation, fixed-capacity spans."""
    import torch
    import triton
    ids = decision.device_tokens
    n, ft, r = ids.numel(), directory.ft, directory.regions
    group, heads = directory.group, directory.heads
    groups = heads//group
    b, offset = directory.mapping[ids % ft].unbind(-1)
    order = torch.argsort((ids//ft*r+b)*ft+offset, stable=True)
    ids, b, offset = ids[order], b[order], offset[order]
    base = directory.pointers[ids//ft, 0]
    # Paired block/group layout: slab frame base + block + group/side + token.
    block = base + directory.starts[b]*2*heads*128*2
    g = torch.arange(groups, device=ids.device, dtype=torch.int64)[:, None]
    key = block[None, :] + (g*2*directory.sizes[b][None, :]+offset[None, :])*group*128*2
    ptr = torch.stack((key, key+directory.sizes[b][None, :]*group*128*2), -1)
    contiguous = (ptr[:, 1:] == ptr[:, :-1]+group*128*2).all(-1).all(0)
    begin = torch.cat((torch.ones(1, device=ids.device, dtype=torch.bool), ~contiguous))
    ranks = begin.to(torch.int64).cumsum(0)-1
    capacity = decision.candidates.numel()*r
    torch._assert_async(begin.sum() <= capacity, 'device archive span capacity exceeded')
    starts = torch.full((capacity+1,), n, device=ids.device, dtype=torch.int64)
    kernels()[2][(triton.cdiv(n, 256),)](begin, ranks, starts, n, 256)
    valid = starts[:-1] < n
    at = starts[:-1].clamp_max(n-1)
    metadata = torch.cat((ptr[:, at], starts[:-1][None, :, None].expand(groups, -1, 1),
        (starts[1:]-starts[:-1])[None, :, None].expand(groups, -1, 1)), -1).contiguous()
    return DeviceArchiveSource((), (), tuple(range(decision.history_budget)), ft,
        history_start, total, heads, groups, group, metadata, order, directory)
