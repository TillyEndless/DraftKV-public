"""Resident ranges written once into the native consumer's final allocation.

Historical payload is deliberately absent: the archive producer fills that
gap. Current K's temporal correction uses the existing geometry arithmetic.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class PinnedGroupTensor:
    """A native sink view into its single immutable, group-major CPU owner."""
    storage: object

    def __post_init__(self):
        import torch
        if (not isinstance(self.storage,torch.Tensor) or self.storage.ndim!=4
                or self.storage.device.type!='cpu' or self.storage.dtype!=torch.bfloat16
                or self.storage.shape[-1]!=128 or min(self.storage.shape)<1
                or not self.storage.is_contiguous() or not self.storage.is_pinned()):
            raise ValueError('resident groups must reference contiguous pinned CPU BF16 KV')

    @property
    def shape(self):
        groups, tokens, group, dim = self.storage.shape
        return (1, tokens, groups*group, dim)

    @property
    def dtype(self): return self.storage.dtype
    @property
    def device(self): return self.storage.device
    @property
    def ndim(self): return 4

    def copy_into(self, destination, head_begin, token_begin, token_end):
        group = self.storage.shape[2]
        head_end = head_begin + destination.shape[1]
        for g in range(head_begin//group, (head_end-1)//group+1):
            lo, hi = max(head_begin,g*group), min(head_end,(g+1)*group)
            destination[:,lo-head_begin:hi-head_begin].copy_(
                self.storage[g,token_begin:token_end,lo-g*group:hi-g*group], non_blocking=True)


@dataclass(frozen=True)
class ResidentKV:
    total_tokens: int
    history_start: int
    history_tokens: int
    prefix_tokens: int
    segments: tuple
    temporal: tuple
    geometry: object
    prepared: object = None

    def validate(self, q, k, v):
        import torch
        if (not all(isinstance(t, torch.Tensor) for t in (q,k,v))
                or q.ndim != 4 or q.shape[0] != 1 or q.shape[-1] != 128
                or min(q.shape) < 1 or q.dtype != torch.bfloat16 or not q.is_contiguous()
                or k.ndim != 4 or k.shape != v.shape or k.shape[0] != 1
                or min(k.shape) < 1 or k.shape[2:] != q.shape[2:]
                or any(t.dtype != q.dtype or t.device != q.device for t in (k,v))):
            raise ValueError('invalid direct resident Q/current KV')
        if (not all(type(x) is int for x in (self.total_tokens, self.history_start,
                self.history_tokens, self.prefix_tokens))
                or not 0 <= self.prefix_tokens <= self.history_start
                or self.history_tokens <= 0
                or self.history_start + self.history_tokens >= self.total_tokens
                or not self.segments):
            raise ValueError('invalid direct resident geometry')
        cursor = 0
        for begin, key, value in self.segments:
            if cursor == self.history_start:
                cursor += self.history_tokens
            if (type(begin) is not int or begin != cursor
                    or not isinstance(key, (torch.Tensor, PinnedGroupTensor))
                    or not isinstance(value, (torch.Tensor, PinnedGroupTensor))
                    or key.shape != value.shape or key.ndim != 4 or key.shape[0] != 1
                    or key.shape[1] < 1 or key.shape[2:] != q.shape[2:]
                    or key.dtype != q.dtype or value.dtype != q.dtype
                    or any(t.device.type != 'cpu' and t.device != q.device for t in (key, value))):
                raise ValueError('invalid direct resident segment')
            cursor += key.shape[1]
            if cursor > self.total_tokens or begin < self.history_start < cursor:
                raise ValueError('resident segment overlaps historical payload')
        if (cursor != self.total_tokens or self.segments[-1][1] is not k
                or self.segments[-1][2] is not v):
            raise ValueError('resident ranges do not cover the native layout')
        current_begin = self.history_start + self.history_tokens
        last = current_begin
        for begin, value in self.temporal:
            if (type(begin) is not int or not last <= begin < self.total_tokens
                    or not isinstance(value, torch.Tensor) or value.ndim != 4
                    or value.shape[0] != 1 or value.shape[2] != q.shape[2]
                    or value.shape[3] != self.geometry.dimensions
                    or value.shape[1] < 1
                    or begin + value.shape[1] > self.total_tokens
                    or value.device != q.device or value.dtype != q.dtype):
                raise ValueError('invalid resident temporal correction')
            last = begin + value.shape[1]
        return self

    def validate_selection(self, context, sink_ids, budget, k, v, *, query=None):
        ft = context.frame_tokens
        if (self.history_start != self.prefix_tokens + len(sink_ids)*ft
                or self.history_tokens != budget*ft
                or self.total_tokens != self.prefix_tokens + (len(sink_ids)+budget+context.current_frames)*ft
                or k.shape[1] != context.current_frames*ft):
            raise ValueError('resident layout differs from selected source map')
        return self.validate(context.query if query is None else query, k, v)

    def write_unit(self, slot, group_index, begin, end):
        """Fill only this unit's resident ranges; historical DMA owns the gap."""
        if (slot.ndim!=4 or slot.shape[0]!=2 or slot.shape[-1]!=128
                or not 0<=begin<end<=self.total_tokens or end-begin>slot.shape[1]
                or group_index<0 or (group_index+1)*slot.shape[2]>self.segments[-1][1].shape[2]):
            raise ValueError('invalid resident slot range')
        if self.prepared is not None:
            raise ValueError('slot path must not retain a full prepared workspace')
        group=slot.shape[2];hs=slice(group_index*group,(group_index+1)*group)
        for start,key,value in self.segments:
            lo,hi=max(begin,start),min(end,start+key.shape[1])
            if lo<hi:
                for side,tensor in enumerate((key,value)):
                    dst = slot[side,lo-begin:hi-begin]
                    if isinstance(tensor, PinnedGroupTensor):
                        tensor.copy_into(dst, group_index*group, lo-start, hi-start)
                    else:
                        dst.copy_(tensor[0,lo-start:hi-start,hs],non_blocking=True)
        for start,temporal in self.temporal:
            lo,hi=max(begin,start),min(end,start+temporal.shape[1])
            if lo<hi:
                self.geometry.assign(slot[0,lo-begin:hi-begin],temporal[0,lo-start:hi-start,hs])

    def write(self, dense):
        key = self.segments[-1][1]
        if (dense.ndim != 5 or dense.shape[1:3] != (2,self.total_tokens)
                or dense.shape[-1] != 128 or dense.shape[0]*dense.shape[3] != key.shape[2]
                or dense.device != key.device or dense.dtype != key.dtype
                or not dense.is_contiguous()):
            raise ValueError('invalid final resident destination')
        groups, _, _, group, _ = dense.shape
        if self.prepared is not None:
            source=self.prepared
            if (source.ndim != 5 or source.shape[1:3] != (2,self.total_tokens)
                    or source.shape[0]*source.shape[3] != key.shape[2]
                    or source.shape[-1] != 128 or source.device != dense.device
                    or source.dtype != dense.dtype or not source.is_contiguous()):
                raise ValueError('invalid prepared resident workspace')
            if source.data_ptr()==dense.data_ptr() and source.shape==dense.shape:
                return  # Already populated before the unchanged pipeline ready point.
            sg=source.shape[3]
            for g in range(groups):
                lo,hi=g*group,(g+1)*group
                for begin,key,value in self.segments:
                    end=begin+key.shape[1]
                    for src_group in range(lo//sg,(hi-1)//sg+1):
                        a,b=max(lo,src_group*sg),min(hi,(src_group+1)*sg)
                        dense[g,:,begin:end,a-lo:b-lo].copy_(source[src_group,:,begin:end,a-src_group*sg:b-src_group*sg])
            return
        for g in range(groups):
            hs = slice(g*group, (g+1)*group)
            for begin, key, value in self.segments:
                end = begin + key.shape[1]
                for side, tensor in enumerate((key,value)):
                    if isinstance(tensor, PinnedGroupTensor):
                        tensor.copy_into(dense[g,side,begin:end],g*group,0,tensor.shape[1])
                    else:
                        dense[g,side,begin:end].copy_(tensor[0,:,hs])
            for begin, temporal in self.temporal:
                self.geometry.assign(dense[g, 0, begin:begin+temporal.shape[1]], temporal[0, :, hs])


def prepare_resident(context, geometry, sink, current, budget, bank, validate_sink, *, chrono=True, destination=None):
    """Return relocated Q plus references to resident KV, without full KV copies."""
    ft = context.frame_tokens
    prefix = geometry.prefix
    prefix_tokens = prefix[0].shape[1] if prefix is not None else 0
    segments = []
    if prefix_tokens:
        segments.append((0, prefix[0], prefix[1]))
    for i, fid in enumerate(sink):
        if fid not in bank:
            from draftkv.backbones.contracts import BackboneError
            raise BackboneError('DIRTY_HISTORY', 'CPU committed sink missing from this layer')
        pair = bank[fid]
        validate_sink(pair)
        from .canonical_prepack import LogicalArchiveTensor
        if all(isinstance(t, LogicalArchiveTensor) and t.frame.grouped_sink for t in pair):
            key, value = (PinnedGroupTensor(t.frame.resident(t.side)) for t in pair)
        else:
            key, value = (t.unsqueeze(0) for t in pair)
        segments.append((prefix_tokens+i*ft, key, value))
    history_start = prefix_tokens + len(sink)*ft
    current_begin = history_start + budget*ft
    segments.append((current_begin, context.key, context.value))
    layout = ResidentKV(current_begin+context.key.shape[1], history_start, budget*ft,
                        prefix_tokens, tuple(segments), (), geometry)
    if destination is not None:
        layout.write(destination)
    q = context.query.clone()
    temporal = []
    for i, fid in enumerate(current):
        slot, a, b = len(sink)+budget+i, i*ft, (i+1)*ft
        if chrono and slot != fid:
            geometry.assign(q[:, a:b], geometry.rotate(geometry.extract(context.raw_query[:, a:b]), slot, q.dtype))
            correction=geometry.rotate(geometry.extract(context.raw_key[:, a:b]), slot, context.key.dtype)
            if destination is None:
                temporal.append((current_begin+a,correction))
            else:
                group=destination.shape[3]
                for g in range(destination.shape[0]):
                    geometry.assign(destination[g,0,current_begin+a:current_begin+b],
                        correction[0,:,g*group:(g+1)*group])
                del correction
    from dataclasses import replace
    return q.contiguous(), replace(layout,temporal=tuple(temporal),prepared=destination)


def prepare_native_workspace(hook, context, geometry, sink, current, budget):
    """Connect device selection to the existing dense/slots consumer ownership."""
    import torch
    prefix = geometry.prefix
    prefix_tokens = prefix[0].shape[1] if prefix is not None else 0
    total = (len(sink)+budget+len(current))*context.frame_tokens
    bank = hook.committed.get(context.layer, {})
    if budget and getattr(hook,'settings',{}).get('direct_resident_kv',True):
        from .table2_native import reserve_native_dense
        config = hook._native_runtime_config(context)
        dense = None if config.get('kv_slots') else reserve_native_dense(
            context.query,prefix_tokens+total,binding=hook.binding,
            cache=hook._b01_pinned_staging,config=config,family=hook.host.spec.family,
            capacity_tokens=prefix_tokens+context.window_frames*context.frame_tokens)
        q, resident = prepare_resident(context,geometry,sink,current,budget,bank,
            hook._validate_sink,destination=dense)
        return q, context.key, context.value, None, None, resident
    # Before history exists the established resident-only native call is small.
    shape = (1,prefix_tokens+total,*context.query.shape[2:])
    full_k, full_v = (torch.empty(shape,dtype=context.key.dtype,device=context.key.device) for _ in range(2))
    if prefix_tokens:
        full_k[:,:prefix_tokens].copy_(prefix[0]); full_v[:,:prefix_tokens].copy_(prefix[1])
    k,v = full_k[:,prefix_tokens:],full_v[:,prefix_tokens:]
    ft = context.frame_tokens
    for i,fid in enumerate(sink):
        hook._validate_sink(bank[fid])
        for dst,src in zip((k,v),bank[fid]):
            dst[:,i*ft:(i+1)*ft].copy_(src.unsqueeze(0))
    offset = (len(sink)+budget)*ft
    k[:,offset:].copy_(context.key); v[:,offset:].copy_(context.value)
    q = context.query.clone()
    for i,fid in enumerate(current):
        slot,a,b = len(sink)+budget+i,i*ft,(i+1)*ft
        if slot != fid:
            geometry.assign(q[:,a:b],geometry.rotate(geometry.extract(context.raw_query[:,a:b]),slot,q.dtype))
            geometry.assign(k[:,offset+a:offset+b],geometry.rotate(geometry.extract(context.raw_key[:,a:b]),slot,k.dtype))
    return q.contiguous(), k, v, full_k, full_v, None
