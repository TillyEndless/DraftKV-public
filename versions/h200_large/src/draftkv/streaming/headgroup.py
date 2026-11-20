"""Copy-engine / native FlashAttention tasks with independent group/split readiness.

No waiting CTA or replacement attention math: stream waits precede the native
kernel, including its earliest TMA prefetch. Every receive allocation survives
all Q tiles. Logical Chrono slots come unchanged from the device route.
"""
from dataclasses import dataclass
from .device_route import DevicePlan, native, geometry


@dataclass(frozen=True)
class PipelineConfig:
    group: int
    splits: int = 1
    native_splits: int = 1
    schedule: str = 'pipeline'
    reverse: bool = True
    diagnostics: bool = False
    submission: str = 'wave'
    native_attention_version: int = 3

    def validate(self, heads, tokens):
        if self.group < 1 or heads % self.group or not 1 <= self.splits <= tokens:
            raise ValueError('invalid head group or split partition')
        if self.native_splits < 1 or self.schedule not in ('pipeline', 'serial'):
            raise ValueError('invalid native split or submission policy')
        if self.submission not in ('wave', 'unit'):
            raise ValueError('invalid group task submission order')
        if self.native_attention_version not in (2,3):
            raise ValueError('native attention version must be FA2 or FA3')


def validate_native_prefix(prefix, q, prefix_tokens, family, heads):
    """No implicit prefix fallback: preserve exact native [condition,audio] KV."""
    import torch
    if (not isinstance(prefix_tokens,int) or isinstance(prefix_tokens,bool) or
            prefix_tokens<0):
        raise ValueError('invalid prefix_tokens')
    if family!='solarwm':
        if prefix is not None or prefix_tokens:
            raise ValueError('fixed prefix only supported for SolarWM-H3')
        return
    if prefix_tokens==0:
        if prefix is not None:
            raise ValueError('unexpected H3 native fixed prefix')
        return
    if (not isinstance(prefix,(tuple,list)) or len(prefix)!=2 or
            any(not isinstance(t,torch.Tensor) or
                t.ndim!=4 or t.shape!=(1,prefix_tokens,heads,128) or
                t.device!=q.device or t.dtype!=q.dtype or t.dtype!=torch.bfloat16
                for t in prefix)):
        raise ValueError('H3 requires paired native BF16 prefix tensors [1,P,H,128]')


def native_attention(q, k, v, *, out=None, splits=1, version=3):
    """Explicit runtime FA2/FA3 selection on Hopper; no silent FA3 fallback."""
    import torch
    if version not in (2,3):raise ValueError("invalid native FlashAttention version")
    if version==3:
        if torch.cuda.get_device_capability(q.device)[0] != 9:
            raise RuntimeError("FA3 requires Hopper SM90")
        from flash_attn_interface import _flash_attn_forward
        result = _flash_attn_forward(q=q, k=k, v=v, k_new=None, v_new=None,
            qv=None, out=out, cu_seqlens_q=None, cu_seqlens_k=None,
            cu_seqlens_k_new=None, seqused_q=None, seqused_k=None,
            max_seqlen_q=None, max_seqlen_k=None, page_table=None,
            kv_batch_idx=None, leftpad_k=None, rotary_cos=None, rotary_sin=None,
            seqlens_rotary=None, q_descale=None, k_descale=None, v_descale=None,
            softmax_scale=q.shape[-1]**-.5, causal=False, num_splits=splits)
        return result[0], result[1]
    if splits!=1:raise ValueError('FA2 grouped native_splits must be 1')
    from flash_attn import flash_attn_func
    result = flash_attn_func(q, k, v, return_attn_probs=True)
    if out is not None:
        out.copy_(result[0])
    return result[0] if out is None else out, result[1]


_ROTATE = _MERGE = _PACK = None


def _kernels():
    global _ROTATE, _MERGE, _PACK, tl
    if _ROTATE is not None:
        return _ROTATE, _MERGE
    import triton
    import triton.language as tl

    @triton.jit
    def pack(K,V,Raw,Out,Index,Counts,Starts,T:tl.constexpr,H:tl.constexpr,G:tl.constexpr,
             R:tl.constexpr,C:tl.constexpr,CHRONO:tl.constexpr,KT:tl.constexpr,KH:tl.constexpr,
             VT:tl.constexpr,VH:tl.constexpr,RT:tl.constexpr,RH:tl.constexpr,H3:tl.constexpr):
        f=tl.program_id(0)//R;b=tl.program_id(0)%R;head=tl.program_id(1)
        t=tl.program_id(2)*64+tl.arange(0,64);d=tl.arange(0,128)
        count=tl.load(Counts+b);start=tl.load(Starts+b)
        token=tl.load(Index+b*C+t,t<count,0)
        key=tl.load(K+(f*T+token[:,None])*KT+head*KH+d[None,:],t[:,None]<count,0)
        value=tl.load(V+(f*T+token[:,None])*VT+head*VH+d[None,:],t[:,None]<count,0)
        if CHRONO:
            # H3 temporal-only K: [0:16] and [48:64]. All other camera/
            # spatial components remain native post-RoPE, identical to B direct.
            temporal=((d[None,:]<16)|((d[None,:]>=48)&(d[None,:]<64))) if H3 else d[None,:]<44
            raw=tl.load(Raw+(f*T+token[:,None])*RT+head*RH+d[None,:],
                        (t[:,None]<count)&temporal,0)
            key=tl.where(temporal,raw,key)
        base=f*2*T*H*128+start*2*H*128+(head//G)*2*count*G*128
        address=(t[:,None]*G+head%G)*128+d[None,:]
        tl.store(Out+base+address,key,t[:,None]<count)
        tl.store(Out+base+count*G*128+address,value,t[:,None]<count)

    @triton.jit
    def rotate(P, S, F, LO:tl.constexpr, N:tl.constexpr, G:tl.constexpr,
               H3:tl.constexpr):
        t=LO+tl.program_id(0)*32+tl.arange(0,32)
        h=tl.program_id(1);d=tl.arange(0,32)
        valid=(t[:,None]<LO+N)&(d[None,:]<(16 if H3 else 22))
        slot=tl.load(S+t,t<LO+N,0)
        if H3:
            # Match temporal_kernels.py exactly: native H3 uses 16 pairs at
            # offsets 0 and 48, FP32 products rounded to BF16 *before*
            # their summation, and FP fusion disabled.
            address=(t[:,None]*G+h)*128+d[None,:]
            a=tl.load(P+address,valid,0).to(tl.float32)
            b=tl.load(P+address+48,valid,0).to(tl.float32)
            c=tl.load(F+slot[:,None]*32+d[None,:],valid,0)
            sn=tl.load(F+slot[:,None]*32+16+d[None,:],valid,0)
            ac=(a*c).to(tl.bfloat16).to(tl.float32)
            bs=(-b*sn).to(tl.bfloat16).to(tl.float32)
            bc=(b*c).to(tl.bfloat16).to(tl.float32)
            ass=(a*sn).to(tl.bfloat16).to(tl.float32)
            tl.store(P+address,ac+bs,valid)
            tl.store(P+address+48,bc+ass,valid)
        else:
            address=(t[:,None]*G+h)*128+d[None,:]*2
            a=tl.load(P+address,valid,0).to(tl.float64)
            b=tl.load(P+address+1,valid,0).to(tl.float64)
            c=tl.load(F+slot[:,None]*44+d[None,:]*2,valid,0)
            s=tl.load(F+slot[:,None]*44+d[None,:]*2+1,valid,0)
            tl.store(P+address,(a*c-b*s).to(tl.float32).to(tl.bfloat16),valid)
            tl.store(P+address+1,(a*s+b*c).to(tl.float32).to(tl.bfloat16),valid)

    @triton.jit
    def merge(P,L,O,Z,Q:tl.constexpr,H:tl.constexpr,S:tl.constexpr,BS:tl.constexpr):
        q=tl.program_id(0)*16+tl.arange(0,16);h=tl.program_id(1)
        s=tl.arange(0,BS);d=tl.arange(0,128)
        l=tl.load(L+(s[:,None]*H+h)*Q+q[None,:],(s[:,None]<S)&(q[None,:]<Q),float('-inf'))
        mx=tl.max(l,0);w=tl.exp(l-mx[None,:]);den=tl.sum(w,0)
        total=tl.full((16,128),0,tl.float32)
        for i in range(S):
            li=tl.load(L+(i*H+h)*Q+q,q<Q,float('-inf'))
            wi=tl.exp(li-mx)/den
            x=tl.load(P+((i*Q+q[:,None])*H+h)*128+d[None,:],q[:,None]<Q,0).to(tl.float32)
            total+=x*wi[:,None]
        tl.store(O+(q[:,None]*H+h)*128+d[None,:],total,q[:,None]<Q)
        tl.store(Z+h*Q+q,mx+tl.log(den),q<Q)
    _ROTATE, _MERGE, _PACK = rotate, merge, pack
    return rotate, merge


def pack_current(key,value,raw,out,h,w,group,chrono,*,h3=False):
    """Noisy current KV needs a layout, but no clean-history descriptors.

    Tile its independent copies rather than retaining a whole Spatial Block
    and unused mean/max reductions in each CTA. Clean archive stays fused.
    """
    _kernels()
    ph,pw=8,16
    index,counts,starts,_=geometry(h,w,ph,pw,str(key.device));r=counts.numel()
    _PACK[(key.shape[1]//(h*w)*r,key.shape[2],(ph*pw+63)//64)](key,value,raw,out,index,counts,starts,
        h*w,key.shape[2],group,r,ph*pw,chrono,key.stride(1),key.stride(2),
        value.stride(1),value.stride(2),raw.stride(1),raw.stride(2),h3,num_warps=4)


class HeadGroupPlan(DevicePlan):
    def __init__(self, h, w, capacity, device, heads=12, family='wan', config=None, ph=8, pw=16):
        import torch
        if family not in ('wan','lingbot','longlive','solarwm'):
            raise ValueError('unsupported grouped provider backbone family')
        super().__init__(h,w,capacity,device,heads,family,ph,pw)
        self.config=config or PipelineConfig(heads)
        self.prep=torch.cuda.Stream(device=device,priority=-1)
        self.group_records=[]
        self.primed=set()
        self.cold_dma=[]
        self.prefix_tokens=0
        print(f'DRAFTKV_HEADGROUP_NATIVE_FA{self.config.native_attention_version}',flush=True)

    def route(self, score, packed, addresses, sink, start, current, budget, protect,
              chrono, batch, *, submit=True, prefix_tokens=0):
        import torch
        if (not isinstance(prefix_tokens,int) or isinstance(prefix_tokens,bool) or
                prefix_tokens<0 or (prefix_tokens and self.family!='solarwm')):
            raise ValueError('invalid H3 native fixed-prefix token count')
        self.prefix_tokens=prefix_tokens
        events=super().route(score,packed,addresses,sink,start,current,budget,protect,
                             chrono,batch,submit=False,index_only=True)
        c=self.config;c.validate(self.heads,self.total)
        self.dense=self.reserve('group_dense',(self.heads//c.group,2,self.total,c.group,128),torch.bfloat16)
        self.dma_ready=self.reserve('group_dma_ready',(self.heads//c.group*c.splits,),torch.int32)
        self.consume_ready=self.reserve('group_consume_ready',self.dma_ready.shape,torch.int32)
        rows=(self.heads//c.group)*(2*self.capacity*self.r+3*c.splits)
        table=self.reserve('group_table',(8+5*rows,),torch.int64)
        cpu=self.reserve('group_description',table.shape,torch.int64,True)
        workspace=self.reserve('group_unit_descriptions',
            (self.heads//c.group*c.splits,8+5*(2*self.capacity*self.r+1)),torch.int64)
        _,_,starts,_=geometry(self.h,self.w,self.ph,self.pw,str(self.device))
        native().build_groups(addresses,self.meta,starts,packed,self.dense,table,
            self.dma_ready,self.consume_ready,self.slots,start,current,self.r,self.ft,
            self.total,self.heads,c.group,c.splits,c.reverse,workspace,self.block_capacity)
        end=torch.cuda.Event(enable_timing=True);end.record();self.ready=end;self.table=table
        self.description=cpu
        if submit:self.producer.submit(table,cpu,self.dma_ready,self.dense,self.fetch.cuda_stream)
        return events[0],end

    def _units(self):
        c=self.config
        for g in range(self.heads//c.group):
            for s in (range(c.splits-1,-1,-1) if c.reverse else range(c.splits)):
                yield g,s,self.total*s//c.splits,self.total*(s+1)//c.splits

    def prime(self,q,phase):
        """Explicit measured cold setup, before any asynchronous readiness waits.

        Call after a serial first fetch or on valid resident data. No dummy KV,
        hidden per-layer warmup, or warmup excluded from cold-start accounting.
        """
        import torch
        signature=(tuple(q.shape),self.total,self.config.group,self.config.splits,
                   self.config.native_splits,self.chrono,self.family,self.config.native_attention_version)
        if signature in self.primed:return
        self.cold_dma.extend(self.producer.collect())  # before any first-use stream wait
        self.compute.wait_event(self.ready)
        if self.chrono:
            rotate,_=_kernels()
            for g,s,lo,hi in self._units():
                compiled=rotate.warmup(self.dense[g,0],self.slots,phase,lo,hi-lo,
                    self.config.group,self.family=='solarwm',grid=((hi-lo+31)//32,self.config.group),
                    enable_fp_fusion=False)
                _=compiled.run
        # Native FA module's first call is a valid resident attention call;
        # execute(first serial) supplies its real result, not discarded runs.
        self.primed.add(signature)

    def execute(self,q,phase,route_events,prefix=None,*,resident=False,compute=True,prepare_unit=None):
        import torch
        import triton
        validate_native_prefix(prefix,q,self.prefix_tokens,self.family,self.heads)
        c=self.config;m=native();g=c.group;diag=c.diagnostics
        if not resident:self.prime(q,phase)
        out=self.reserve('group_out',q.shape,q.dtype)
        self.lse=self.reserve('group_lse',(1,self.heads,q.shape[1]),torch.float32)
        # Treat native condition/audio KV as a separate resident softmax
        # partition. This avoids a second H2D, leaves fixed prefix untouched
        # by H3 Chrono-RoPE and preserves the original video DMA layout.
        pieces=c.splits+(1 if self.prefix_tokens else 0)
        parts=self.reserve('group_partial',(pieces,*q.shape),q.dtype) if pieces>1 else out.unsqueeze(0)
        lses=self.reserve('group_partial_lse',(pieces,1,self.heads,q.shape[1]),torch.float32)
        rotate,merge=_kernels()
        caller_ready=torch.cuda.Event();caller_ready.record()
        self.compute.wait_event(caller_ready);self.prep.wait_event(self.ready)
        stages=[]
        if c.schedule=='serial' and not resident:
            with torch.cuda.stream(self.prep):
                for unit in range(self.dma_ready.numel()):m.group_ready(self.dma_ready,unit,False)

        def consume(record,group,split,lo,hi):
            with torch.cuda.stream(self.compute):
                waiting,begin,end=[torch.cuda.Event(enable_timing=True) if diag else None for _ in range(3)]
                if diag:waiting.record()
                if not resident:m.group_ready(self.consume_ready,record[0],False)
                if diag:begin.record()
                if compute:
                    hs=slice(group*g,(group+1)*g)
                    _,lse=native_attention(q[:,:,hs],self.dense[group,0,lo:hi].unsqueeze(0),
                        self.dense[group,1,lo:hi].unsqueeze(0),out=parts[split,:,:,hs],
                        splits=c.native_splits,version=c.native_attention_version)
                    lses[split,:,hs].copy_(lse)
                if diag:end.record()
                record.extend([waiting,begin,end])

        def consume_prefix():
            if not self.prefix_tokens or not compute:return
            with torch.cuda.stream(self.compute):
                for group in range(self.heads//g):
                    hs=slice(group*g,(group+1)*g)
                    _,lse=native_attention(q[:,:,hs],prefix[0][:,:,hs],prefix[1][:,:,hs],
                                           out=parts[c.splits,:,:,hs],splits=c.native_splits,
                                           version=c.native_attention_version)
                    lses[c.splits,:,hs].copy_(lse)

        if c.schedule=='pipeline':consume_prefix()
        interleave=c.submission=='unit' and c.schedule=='pipeline'
        # Prepare tasks on a separate stream. Stream waits allocate no CTA and
        # cannot prevent the next group's RoPE kernel from getting SM resources.
        for group,split,lo,hi in self._units():
            unit=group*c.splits+split
            before,ready,rotated=[torch.cuda.Event(enable_timing=True) if diag else None for _ in range(3)]
            with torch.cuda.stream(self.prep):
                if diag:before.record()
                if not resident:m.group_ready(self.dma_ready,unit,False)
                if diag:ready.record()
                if prepare_unit is not None and not resident:
                    prepare_unit(group,split,lo,hi)
                elif self.chrono and not resident:
                    rotate[((hi-lo+31)//32,g)](self.dense[group,0],self.slots,phase,lo,hi-lo,g,
                        self.family=='solarwm',enable_fp_fusion=False)
                if diag:rotated.record()
                if not resident:m.group_ready(self.consume_ready,unit,True)
            stages.append([unit,before,ready,rotated])
            if interleave:consume(stages[-1],group,split,lo,hi)
        prepared=torch.cuda.Event();prepared.record(self.prep)
        if c.schedule=='serial':
            if not resident:self.compute.wait_event(prepared)
            consume_prefix()
        if not interleave:
            for record,(group,split,lo,hi) in zip(stages,self._units()):
                consume(record,group,split,lo,hi)
        with torch.cuda.stream(self.compute):
            merge_begin=torch.cuda.Event(enable_timing=True) if diag else None
            done=torch.cuda.Event(enable_timing=True)
            if diag:merge_begin.record()
            if compute:
                if pieces>1:
                    merge[((q.shape[1]+15)//16,self.heads)](parts,lses,out,self.lse,
                        q.shape[1],self.heads,pieces,triton.next_power_of_2(pieces))
                else:self.lse.copy_(lses[0])
            done.record()
        torch.cuda.current_stream(self.device).wait_event(done)
        self.group_records.append((route_events,stages,merge_begin,done,resident))
        self.end=done
        return out

    def collect(self):
        dma=self.cold_dma+self.producer.collect();self.cold_dma=[]
        if self.group_records:self.group_records[-1][3].synchronize()
        rows=[]
        for (origin,route_end),stages,merge_begin,end,resident in self.group_records:
            rows.append(dict(route_index_ms=origin.elapsed_time(route_end),
                fetch_attention_ms=route_end.elapsed_time(end),
                rotation_wait_ms=sum(a.elapsed_time(b) for _,a,b,*_ in stages) if merge_begin else None,
                rope_ms=sum(b.elapsed_time(c) for _,_,b,c,*_ in stages) if merge_begin else None,
                attention_ms=sum(b.elapsed_time(e) for *_,b,e in stages) if merge_begin else None,
                consumer_wait_ms=sum(a.elapsed_time(b) for _,_,_,_,a,b,_ in stages) if merge_begin else None,
                merge_ms=merge_begin.elapsed_time(end) if merge_begin else None,
                groups=self.heads//self.config.group,splits=self.config.splits,resident=resident))
        self.group_records=[]
        return dma,rows
