"""Canonical clean block/group archive, independent of retrieval policy.

Full-resolution native K is used for descriptors. B archives raw temporal K
in the DMA payload and retains native temporal components in the auxiliary
space formerly occupied by raw K. Thus neither archive size nor scoring
semantics change. Logical CPU views are compatibility/reference adapters;
the native consumer reads the immutable packed allocation directly.
"""
from dataclasses import dataclass
from functools import lru_cache


@lru_cache(None)
def token_layout(h, w, ph=8, pw=16):
    from .prepacked import layout
    blocks, inverse, starts = layout(h, w, ph, pw, 1)
    return blocks, inverse, starts


_PREPARE = None


def prepare_kernel():
    global _PREPARE, tl
    if _PREPARE is not None:
        return _PREPARE
    import triton
    import triton.language as tl

    @triton.jit
    def prepare(K, V, Raw, Out, Index, Counts, Starts,
                T: tl.constexpr, H: tl.constexpr, G: tl.constexpr,
                R: tl.constexpr, C: tl.constexpr, FRAMES: tl.constexpr,
                TD: tl.constexpr, H3: tl.constexpr,
                START, SINK: tl.constexpr,
                KT: tl.constexpr, KH: tl.constexpr, VT: tl.constexpr,
                VH: tl.constexpr, RT: tl.constexpr, RH: tl.constexpr,
                GROUPED_SINK: tl.constexpr = False):
        f = tl.program_id(0); b = tl.program_id(1); head = tl.program_id(2)
        t = tl.arange(0, C); d = tl.arange(0, 128)
        n = tl.load(Counts+b); start = tl.load(Starts+b)
        token = tl.load(Index+b*C+t, t < n, 0)
        key = tl.load(K+(f*T+token[:, None])*KT+head*KH+d[None, :], t[:, None] < n, 0)
        value = tl.load(V+(f*T+token[:, None])*VT+head*VH+d[None, :], t[:, None] < n, 0)
        if TD:
            raw = tl.load(Raw+(f*T+token[:, None])*RT+head*RH+d[None, :], t[:, None] < n, 0)
            temporal = ((d < 16) | ((d >= 48) & (d < 64))) if H3 else d < 44
            compact_d = tl.where(d < 16, d, d-32) if H3 else d
            # Save native rounded temporal values for sink/reference reads.
            aux = FRAMES*2*T*H*128+((f*T+token[:, None])*H+head)*TD+compact_d[None, :]
            tl.store(Out+aux, tl.where(f+START<SINK,raw,key),
                     (t[:, None] < n) & temporal[None, :])
            key = tl.where(temporal[None, :] & (f+START>=SINK), raw, key)
        base = f*2*T*H*128+start*2*H*128+(head//G)*2*n*G*128
        offset = (t[:, None]*G+head % G)*128+d[None, :]
        if f+START<SINK:
            if GROUPED_SINK:
                native=f*2*T*H*128+((head//G)*T*G+token[:,None]*G+head%G)*128+d[None,:]
            else:
                native=f*2*T*H*128+(token[:,None]*H+head)*128+d[None,:]
            tl.store(Out+native,key,t[:,None]<n)
            tl.store(Out+native+T*H*128,value,t[:,None]<n)
        else:
            tl.store(Out+base+offset, key, t[:, None] < n)
            tl.store(Out+base+n*G*128+offset, value, t[:, None] < n)
    _PREPARE = prepare
    return prepare


@dataclass(frozen=True)
class PackedFrame:
    slab: object
    frame: int
    frames: int
    h: int
    w: int
    heads: int
    group: int
    temporal_dims: int
    native_sink: bool = False
    grouped_sink: bool = False

    @property
    def ft(self):
        return self.h*self.w

    def block(self, block, group, side):
        blocks, _, starts = token_layout(self.h, self.w)
        n = len(blocks[block]); g = self.group
        begin = (self.frame*2*self.ft*self.heads+starts[block]*2*self.heads+group*2*n*g)*128
        return self.slab[begin+side*n*g*128:begin+(side+1)*n*g*128].view(n, g, 128)

    def logical(self, side, *, raw=False):
        """Explicit compatibility materialization, never used for native fetch."""
        import torch
        if self.native_sink:
            if raw:
                begin=self.frames*2*self.ft*self.heads*128+self.frame*self.ft*self.heads*self.temporal_dims
                return self.slab[begin:begin+self.ft*self.heads*self.temporal_dims].view(self.ft,self.heads,self.temporal_dims)
            begin=(self.frame*2+side)*self.ft*self.heads*128
            if self.grouped_sink:
                return self.resident(side).permute(1,0,2,3).reshape(self.ft,self.heads,128)
            return self.slab[begin:begin+self.ft*self.heads*128].view(self.ft,self.heads,128)
        shape = (self.ft, self.heads, self.temporal_dims if raw else 128)
        # Reference/CPU-RoPE scratch is not a DMA source; keep it pageable.
        # The native fetch reads slab itself and allocates no selected CPU KV.
        result = torch.empty(shape, dtype=self.slab.dtype, device='cpu')
        blocks, _, _ = token_layout(self.h, self.w)
        for b, tokens in enumerate(blocks):
            index = torch.tensor(tokens, dtype=torch.long)
            for group in range(self.heads//self.group):
                src = self.block(b, group, side)
                if raw:
                    src = (torch.cat((src[..., :16], src[..., 48:64]), -1)
                           if self.temporal_dims == 32 else src[..., :44])
                result[index, group*self.group:(group+1)*self.group] = src
        if side == 0 and self.temporal_dims and not raw:
            begin = self.frames*2*self.ft*self.heads*128+self.frame*self.ft*self.heads*self.temporal_dims
            native = self.slab[begin:begin+self.ft*self.heads*self.temporal_dims].view(self.ft,self.heads,self.temporal_dims)
            if self.temporal_dims == 32:
                result[..., :16].copy_(native[..., :16]); result[..., 48:64].copy_(native[..., 16:])
            else:
                result[..., :44].copy_(native)
        return result

    def resident(self, side):
        """Zero-copy pinned sink groups; no second sink allocation or staging."""
        if not self.native_sink or not self.grouped_sink or side not in (0,1):
            raise ValueError('resident view requires group-major native sink')
        begin=(self.frame*2+side)*self.ft*self.heads*128
        return self.slab[begin:begin+self.ft*self.heads*128].view(
            self.heads//self.group,self.ft,self.group,128)


class LogicalArchiveTensor:
    """Small immutable handle; references materialize explicitly on demand."""
    def __init__(self, frame, side, raw=False):
        self.frame, self.side, self.raw = frame, side, raw

    @property
    def device(self): return self.frame.slab.device
    @property
    def dtype(self): return self.frame.slab.dtype
    @property
    def is_cuda(self): return False
    @property
    def shape(self): return (self.frame.ft,self.frame.heads,self.frame.temporal_dims if self.raw else 128)
    def numel(self):
        import math
        return math.prod(self.shape)
    def element_size(self): return self.frame.slab.element_size()
    def is_pinned(self): return self.frame.slab.is_pinned()
    def tensor(self): return self.frame.logical(self.side,raw=self.raw)
    def unsqueeze(self, dim): return self.tensor().unsqueeze(dim)
    def to(self, *args, **kwargs): return self.tensor().to(*args, **kwargs)
    def index_select(self, dim, index): return self.tensor().index_select(dim,index)
    def __getitem__(self, index): return self.tensor()[index]
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        return func(*(x.tensor() if isinstance(x,cls) else x for x in args),**(kwargs or {}))


def native_tensor(value):
    return value.tensor() if isinstance(value,LogicalArchiveTensor) else value


@dataclass(frozen=True)
class ArchiveRun:
    frame: PackedFrame
    block: int
    source_begin: int
    count: int
    destination: int


@dataclass(frozen=True)
class PackedArchiveSource:
    runs: tuple
    owners: tuple
    frame_ids: tuple
    frame_tokens: int
    history_start: int
    kv_tokens: int
    heads: int
    groups: int
    head_group: int
    metadata: object = None
    physical_to_logical: object = None

    @property
    def history_tokens(self): return len(self.frame_ids)*self.frame_tokens

    def verify(self, spec):
        # Unlike DirectGroupMajorSource, the source is a collection of spans
        # in ORIGINAL clean archives, not another selected-history allocation.
        for name in ('frame_ids','frame_tokens','history_start','kv_tokens','heads','groups'):
            if getattr(self,name) != getattr(spec,name):
                raise ValueError('packed archive differs from canonical bound plan: '+name)
        expected = spec.heads if spec.consumer == 'token' else spec.head_group
        if self.head_group != expected or (self.metadata is None and sum(r.count for r in self.runs) != self.history_tokens):
            raise ValueError('packed archive group/coverage mismatch')
        if self.physical_to_logical is not None:
            import torch
            p=self.physical_to_logical
            if (p.dtype!=torch.int64 or p.ndim!=1 or not p.is_contiguous()
                    or p.numel()!=self.history_tokens or self.metadata is None
                    or p.device!=self.metadata.device):
                raise ValueError('invalid private physical/logical token mapping')
        if any(not frame.slab.is_pinned() for frame in self.owners):
            raise ValueError('packed archive must retain pinned slab owners')


def bind_archive(bank, history, *, head_group, frame_tokens, history_start, total,
                 device=None, device_tokens=None, cache=None):
    """Generate span metadata only. Never read/copy a CPU K or V payload."""
    runs=[]; owners=[]; source_map=[]; cursor=0
    for fid, tokens in history:
        key, value = bank[fid]
        if not isinstance(key,LogicalArchiveTensor) or not isinstance(value,LogicalArchiveTensor):
            return None  # explicit recache/reference banks retain old adapter
        frame=key.frame
        if frame.native_sink:raise ValueError('sink cannot enter retrieved archive spans')
        if frame.group != head_group or frame.ft != frame_tokens:
            raise ValueError('clean/fetch head group or geometry differs')
        owners.append(frame)
        if device is not None:
            source_map.extend(fid*frame_tokens+t for t in tokens);cursor+=len(tokens)
            continue
        _, inverse, _ = token_layout(frame.h, frame.w)
        for token in tokens:
            b, offset = inverse[token]
            if (runs and runs[-1].frame is frame and runs[-1].block == b
                    and runs[-1].source_begin+runs[-1].count == offset
                    and runs[-1].destination+runs[-1].count == cursor):
                old=runs[-1];runs[-1]=ArchiveRun(frame,b,old.source_begin,old.count+1,old.destination)
            else:
                runs.append(ArchiveRun(frame,b,offset,1,cursor))
            source_map.append(fid*frame_tokens+token);cursor+=1
    if not owners or cursor % frame_tokens:
        raise ValueError('packed archive must fill exact historical frame budget')
    heads=owners[0].heads
    metadata=permutation=None
    if device is not None:
        metadata,permutation=device_spans(bank,owners[0],source_map,device_tokens,device,cache)
    return PackedArchiveSource(tuple(runs),tuple(owners),tuple(range(cursor//frame_tokens)),
        frame_tokens,history_start,total,heads,heads//head_group,head_group,metadata,permutation),tuple(source_map)


def device_spans(bank, sample, source_map, device_tokens, device, cache):
    """Resolve dynamic source IDs and coalesce actual contiguous spans on GPU.

    CPU builds only the immutable archive pointer registry once per commit;
    GPU resolves selection, block/local offsets and contiguous DMA runs.
    """
    import torch
    cache={} if cache is None else cache
    entry=cache.get('archive_address_registry')
    blocks,inverse,_=token_layout(sample.h,sample.w)
    groups=sample.heads//sample.group
    if entry is None or entry[0] is not bank:
        rows=[]
        for fid in range(max(bank)+1):
            key=bank[fid][0]
            for b in range(len(blocks)):
                for group in range(groups):
                    rows.append(tuple(key.frame.block(b,group,side).data_ptr() for side in (0,1))
                                if isinstance(key,LogicalArchiveTensor) and not key.frame.native_sink else (0,0))
        addresses=torch.tensor(rows,dtype=torch.int64,device=device).view(-1,len(blocks),groups,2)
        mapping=torch.tensor(inverse,dtype=torch.int64,device=device)
        entry=(bank,addresses,mapping)
        cache['archive_address_registry']=entry
    _,addresses,mapping=entry
    ids=(torch.tensor(source_map,dtype=torch.int64,device=device)
         if device_tokens is None else device_tokens.to(device=device))
    if ids.numel()!=len(source_map):raise ValueError('device route/source-map size mismatch')
    b,offset=mapping[ids%sample.ft].unbind(-1)
    # Preserve the canonical source-token order in the public route/map, but
    # consume PRIVATE physical KV in frame/block/local order as the historical
    # native grouped provider did. RoPE slots follow logical ranks, never this
    # physical order. Sorting scalar GPU indices does not materialize host KV.
    order=torch.argsort((ids//sample.ft*len(blocks)+b)*sample.ft+offset,stable=True)
    ids,b,offset=ids[order],b[order],offset[order]
    ptr=addresses[ids//sample.ft,b].permute(1,0,2)
    ptr=ptr+offset[None,:,None]*sample.group*128*2
    # Physical addresses and chronological destinations must BOTH continue.
    # This also prevents merging across K/V payload boundaries.
    contiguous=(ptr[:,1:]==ptr[:,:-1]+sample.group*128*2).all(-1).all(0)
    begin=torch.cat((torch.ones(1,dtype=torch.bool,device=device),~contiguous)).nonzero().flatten()
    end=torch.cat((begin[1:],torch.tensor([ids.numel()],dtype=torch.int64,device=device)))
    return torch.cat((ptr[:,begin],begin[None,:,None].expand(groups,-1,1),
                      (end-begin)[None,:,None].expand(groups,-1,1)),-1).contiguous(),order
