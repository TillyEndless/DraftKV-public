"""Experimental SM89/SM90 BF16 paired cache, with no host KV materialization.

Native FA2 is derived from the bundled B01 snapshot, built under a distinct
content hash. Token addresses and logical Chrono slots are independent maps.
"""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import hashlib
import json
import os
import tarfile
import time


@lru_cache(None)
def layout(h, w, ph, pw, group):
    from draftkv.retrieval.draftattention2_importance import spatial_blocks
    blocks = spatial_blocks(h, w, ph, pw)
    inverse = [None] * (h*w)
    prefix = 0
    for b, tokens in enumerate(blocks):
        for j, token in enumerate(tokens):
            inverse[token] = (b, j)
        prefix += len(tokens)
    starts = [0]
    for tokens in blocks:
        starts.append(starts[-1]+len(tokens))
    return blocks, tuple(inverse), tuple(starts)


@lru_cache(None)
def native():
    import torch
    from torch.utils.cpp_extension import load
    capability = torch.cuda.get_device_capability()
    if capability not in ((8, 9), (9, 0)):
        raise RuntimeError('prepacked experiment requires SM89 or SM90')
    architecture = '.'.join(map(str, capability))
    here = Path(__file__).with_name('prepacked_cuda')
    vendor = here.parents[1] / 'vendor/b01/source_snapshot.tar.gz'
    files = sorted(here.glob('*'))
    digest = hashlib.sha256(vendor.read_bytes())
    digest.update(f'{architecture}:{torch.__version__}:{torch.version.cuda}'.encode())
    for p in files:
        digest.update(p.name.encode()); digest.update(p.read_bytes())
    identity = digest.hexdigest()[:20]
    root = Path(os.environ.get('DRAFTKV_PREPACK_BUILD', '/tmp/draftkv-prepacked')) / identity
    root.mkdir(parents=True, exist_ok=True)
    include = root/'snapshot'
    if not (include/'sources/fa2_custom/ready_params.h').exists():
        with tarfile.open(vendor) as archive:
            archive.extractall(include, filter='data')
    fa = include/'sources/fa2_custom'
    flash = fa/'flash-attention-v2.8.3-streaming/csrc/flash_attn'
    cutlass = os.environ.get('CUTLASS_INCLUDE_DIR')
    if not cutlass:
        # The vendored FA2 tree ships its own pinned CUTLASS dependency.
        candidates=list(include.rglob('cute/tensor.hpp'))
        if not candidates:
            raise RuntimeError('Set CUTLASS_INCLUDE_DIR to the verified B01 CUTLASS include')
        cutlass=str(candidates[0].parents[1])
    os.environ['TORCH_CUDA_ARCH_LIST'] = architecture
    os.environ.setdefault('MAX_JOBS', '2')
    module = load(name='draftkv_prepacked_'+identity,
        sources=[str(here/'prepacked_binding.cpp'),str(here/'prepacked_kernels.cu')],
        extra_include_paths=[str(here),str(fa),str(flash),str(flash/'src'),cutlass],
        extra_cflags=['-O3','-std=c++17'],
        extra_cuda_cflags=['-O3','-std=c++17','--expt-relaxed-constexpr','--expt-extended-lambda',
            '-U__CUDA_NO_HALF_OPERATORS__','-U__CUDA_NO_HALF_CONVERSIONS__',
            '-U__CUDA_NO_HALF2_OPERATORS__','-U__CUDA_NO_BFLOAT16_CONVERSIONS__',
            '-DFLASHATTENTION_DISABLE_DROPOUT','-DFLASHATTENTION_DISABLE_ALIBI',
            '-DFLASHATTENTION_DISABLE_SOFTCAP','-DFLASHATTENTION_DISABLE_LOCAL'],
        extra_ldflags=['-lcuda'],build_directory=str(root),verbose=False)
    module.source_identity=identity
    return module


_KERNEL=None

def _kernel():
    # Triton 3.2 resolves JIT annotations from module globals, not closures.
    global _KERNEL, tl
    if _KERNEL is not None: return _KERNEL
    import triton
    import triton.language as tl
    @triton.jit
    def prepare(K,V,Raw,Out,Desc,Index,Counts,Starts,
                F:tl.constexpr,T:tl.constexpr,H:tl.constexpr,G:tl.constexpr,R:tl.constexpr,
                C:tl.constexpr,CHRONO:tl.constexpr,
                KT:tl.constexpr,KH:tl.constexpr,VT:tl.constexpr,VH:tl.constexpr,RT:tl.constexpr,RH:tl.constexpr,
                H3:tl.constexpr=False):
        f=tl.program_id(0); b=tl.program_id(1); head=tl.program_id(2)
        t=tl.arange(0,C); d=tl.arange(0,128)
        count=tl.load(Counts+b); start=tl.load(Starts+b)
        token=tl.load(Index+b*C+t, t<count, 0)
        offset=(f*T+token[:,None])*KT+head*KH+d[None,:]
        key=tl.load(K+offset,t[:,None]<count,0)
        value=tl.load(V+(f*T+token[:,None])*VT+head*VH+d[None,:],t[:,None]<count,0)
        narrowed=key.to(tl.float16).to(tl.float32)
        mean=tl.div_rn(tl.sum(narrowed,axis=0),count.to(tl.float32))
        maximum=tl.max(tl.where(t[:,None]<count,narrowed,float('-inf')),axis=0)
        target=((f*2*H+head)*R+b)*128+d
        tl.store(Desc+target,mean.to(tl.float16))
        tl.store(Desc+target+H*R*128,maximum.to(tl.float16))
        if CHRONO:
            raw=tl.load(Raw+(f*T+token[:,None])*RT+head*RH+d[None,:],t[:,None]<count,0)
            temporal=((d[None,:]<16)|((d[None,:]>=48)&(d[None,:]<64))) if H3 else d[None,:]<44
            key=tl.where(temporal,raw,key)
        base=f*2*T*H*128+start*2*H*128+(head//G)*2*count*G*128
        address=(t[:,None]*G+head%G)*128+d[None,:]
        tl.store(Out+base+address,key,t[:,None]<count)
        tl.store(Out+base+count*G*128+address,value,t[:,None]<count)
    _KERNEL=prepare
    return prepare


@dataclass
class PreparedChunk:
    slab: object
    descriptors: object
    event: object
    frames: int
    h: int
    w: int
    ph: int
    pw: int
    group: int
    chrono: bool
    prepare_start: object
    prepare_end: object
    copy_start: object
    pinned_owner: object = None
    def wait(self):
        self.event.synchronize()
    def times(self):
        self.wait()
        return dict(prepare_ms=self.prepare_start.elapsed_time(self.prepare_end),
                    d2h_ms=self.copy_start.elapsed_time(self.event),
                    total_gpu_ms=self.prepare_start.elapsed_time(self.event),
                    d2h_bytes=self.slab.numel()*self.slab.element_size(),
                    pinned_backend=('cudaHostRegister_'+getattr(self.pinned_owner,'page_policy','base4k')) if self.pinned_owner is not None else ('gpu_resident' if self.slab.is_cuda else 'torch.empty(pin_memory=True)'),pinned=bool(self.slab.is_pinned()))
    def payload(self,frame,block,group):
        blocks,_,starts=layout(self.h,self.w,self.ph,self.pw,self.group)
        n=len(blocks[block]); heads=12; ft=self.h*self.w
        begin=(frame*2*ft*heads+starts[block]*2*heads+group*2*n*self.group)*128
        return self.slab[begin:begin+2*n*self.group*128]


@lru_cache(None)
def _device_geometry(h,w,ph,pw,g,device):
    import torch
    blocks,_,starts=layout(h,w,ph,pw,g); capacity=ph*pw
    index=torch.tensor([list(t)+[0]*(capacity-len(t)) for t in blocks],device=device,dtype=torch.int32)
    counts=torch.tensor([len(t) for t in blocks],device=device,dtype=torch.int32)
    return index,counts,torch.tensor(starts[:-1],device=device,dtype=torch.int32)


def prepare(key,value,raw,h,w,ph,pw,group,chrono, *, copy_stream=None, archive=True, workspace=None, registered=False, archive_view=None):
    import torch
    if key.shape!=value.shape or key.ndim!=4 or key.shape[0]!=1 or key.shape[2:]!=(12,128):
        raise ValueError('prepacked SM89 requires B1 H12 D128')
    if (key.dtype!=torch.bfloat16 or value.dtype!=key.dtype or value.device!=key.device
            or not key.is_cuda or key.stride(-1)!=1 or value.stride(-1)!=1):
        raise ValueError('contiguous CUDA BF16 clean KV required')
    if chrono and (raw is None or raw.dtype!=torch.bfloat16 or raw.device!=key.device or raw.shape!=key.shape or raw.stride(-1)!=1):
        raise ValueError('B requires exact contiguous raw BF16 temporal K')
    frames=key.shape[1]//(h*w)
    if frames*(h*w)!=key.shape[1] or type(group) is not int or group not in (3,4,6,12): raise ValueError('invalid geometry')
    blocks,_,_=layout(h,w,ph,pw,group)
    index,counts,starts=_device_geometry(h,w,ph,pw,group,str(key.device))
    workspace={} if workspace is None else workspace
    if workspace.get('completion') is not None:workspace['completion'].synchronize()
    packed=workspace.get('packed')
    if packed is None or packed.numel()<key.numel()*2:
        packed=torch.empty(key.numel()*2,device=key.device,dtype=key.dtype)
        workspace['packed']=packed
    packed=packed[:key.numel()*2]
    desc=torch.empty((frames,2,12,len(blocks),128),device=key.device,dtype=torch.float16)
    owner=None
    if archive_view is not None:
        slab, owner = archive_view
        if (not archive or slab.device.type!='cpu' or not slab.is_pinned()
                or slab.dtype!=key.dtype or slab.numel()!=packed.numel() or not slab.is_contiguous()):
            raise ValueError('prepacked archive requires exact pinned immutable arena slice')
    elif archive and registered:
        from draftkv.algorithms.draftkv.registered_pages import RegisteredKVSlab
        owner=RegisteredKVSlab(key,value,frames,h*w)
        slab=owner._storage.view(-1)
    else:
        slab=torch.empty(packed.numel(),dtype=key.dtype,pin_memory=True) if archive else packed
    events=[torch.cuda.Event(enable_timing=True) for _ in range(4)]
    events[0].record()
    _kernel()[(frames,len(blocks),12)](key,value,raw if chrono else key,packed,desc,index,counts,starts,
        frames,h*w,12,group,len(blocks),ph*pw,chrono,
        key.stride(1),key.stride(2),value.stride(1),value.stride(2),
        (raw if chrono else key).stride(1),(raw if chrono else key).stride(2),num_warps=8)
    events[1].record()
    if copy_stream is None:
        if 'copy_stream' not in workspace:workspace['copy_stream']=torch.cuda.Stream(device=key.device)
        copy_stream=workspace['copy_stream']
    copy_stream.wait_event(events[1])
    with torch.cuda.stream(copy_stream):
        events[2].record(); 
        if archive: slab.copy_(packed,non_blocking=True)
        events[3].record()
        packed.record_stream(copy_stream)
    workspace['completion']=events[3]
    return PreparedChunk(slab,desc,events[3],frames,h,w,ph,pw,group,chrono,*events[:3],owner)


@dataclass(frozen=True)
class BlockRef:
    chunk: PreparedChunk
    frame: int
    block: int
    tokens: tuple
    # Logical Chrono slots are supplied by source order, never by the DMA order.
    slots: tuple


def execute(q,refs,phase, *, batch_bytes=2**20, warmup=True, cache=None, chrono_placement="consumer"):
    """Direct paired DMA, descending frontier, and mapped BF16 FA2 consumer.

    refs may change every call. Only address/slot metadata is packed by the CPU.
    KV bytes always come directly from the original pinned clean slabs.
    """
    import numpy as np
    import torch
    m=native(); g=refs[0].chunk.group; groups=12//g
    if any(r.chunk.group!=g or r.chunk.chrono!=refs[0].chunk.chrono for r in refs): raise ValueError('mixed cache layout')
    chrono=refs[0].chunk.chrono
    total=sum(len(r.tokens) for r in refs)
    cache={} if cache is None else cache
    payload_size=sum(r.chunk.payload(r.frame,r.block,0).numel() for r in refs)*groups
    def reserve(name,shape,dtype,device):
        key=(name,str(device),dtype)
        count=1
        for n in shape:count*=n
        value=cache.get(key)
        if value is None or value.numel()<count:
            value=torch.empty(count,device=device,dtype=dtype,pin_memory=str(device)=='cpu')
            cache[key]=value
        return value[:count].view(shape)
    cpu_offsets=reserve('offsets',(groups,2,total),torch.int64,'cpu')
    cpu_slots=reserve('slots',(total,),torch.int32,'cpu')
    offsets=cpu_offsets.numpy()
    slots=cpu_slots.numpy()
    slots[:]=[s for r in refs for s in r.slots]
    device_payload=reserve('payload',(payload_size,),torch.bfloat16,q.device)
    sources=[];targets=[];indices=[];values=[]
    cursor=0; token_start=0; parts=[]
    for r in refs:
        block=r.chunk.payload(r.frame,r.block,0)
        count=len(layout(r.chunk.h,r.chunk.w,r.chunk.ph,r.chunk.pw,g)[0][r.block])
        if any(t<0 or t>=count for t in r.tokens): raise ValueError('block token out of range')
        parts.append((token_start,cursor,count,r))
        token_start+=len(r.tokens);cursor+=block.numel()*groups
    # Each group's selected blocks arrive in reverse attention order. Adjacent
    # original slab ranges are coalesced only when both source and target agree.
    history_end=0
    for group in range(groups):
        entries=[]
        for start,base,count,r in parts:
            size=2*count*g*128; offset=base+group*size
            tok=np.asarray(r.tokens,dtype=np.int64)*g*128+offset
            offsets[group,0,start:start+len(tok)]=tok
            offsets[group,1,start:start+len(tok)]=tok+count*g*128
            source=r.chunk.payload(r.frame,r.block,group)
            target=device_payload[offset:offset+size]
            if source.is_cuda:
                target.copy_(source,non_blocking=True)
            else:
                entries.append((source,target,start))
                history_end=max(history_end,start+len(r.tokens))
        pending=[]; nbytes=0
        for j,(source,target,start) in enumerate(reversed(entries)):
            sources.append(source);targets.append(target);indices.append(group)
            nbytes+=source.numel()*2
            publish=nbytes>=batch_bytes or j==len(entries)-1
            values.append(((start+63)//64 if j!=len(entries)-1 else 0) if publish else -1)
            if publish: nbytes=0
    # Coalesce consecutive descriptors in either direction only for true adjacency.
    sk=[];dk=[];ix=[];val=[]
    for source,target,index,value in zip(sources,targets,indices,values):
        merge=False
        if sk and ix[-1]==index and val[-1]<0 and source.untyped_storage().data_ptr()==sk[-1].untyped_storage().data_ptr():
            if source.data_ptr()+source.numel()*2==sk[-1].data_ptr() and target.data_ptr()+target.numel()*2==dk[-1].data_ptr():
                sk[-1]=source.as_strided((source.numel()+sk[-1].numel(),),(1,))
                dk[-1]=target.as_strided((target.numel()+dk[-1].numel(),),(1,));val[-1]=value;merge=True
        if not merge:sk.append(source);dk.append(target);ix.append(index);val.append(value)
    gpu_offsets=reserve('offsets',(groups,2,total),torch.int64,q.device)
    gpu_slots=reserve('slots',(total,),torch.int32,q.device)
    gpu_offsets.copy_(cpu_offsets,non_blocking=True)
    gpu_slots.copy_(cpu_slots,non_blocking=True)
    phase=phase.to(device=q.device,dtype=torch.float64).contiguous()
    frontier=reserve('frontier',(groups,),torch.int32,q.device)
    out=reserve('out',q.shape,torch.bfloat16,q.device)
    lse=reserve('lse',(1,12,q.shape[1]),torch.float32,q.device)
    if 'streams' not in cache:
        cache['streams']=(torch.cuda.Stream(device=q.device),torch.cuda.Stream(device=q.device))
    fetch,compute=cache['streams']
    plan=m.PairedProducer(dk,dk,sk,sk,frontier,ix,val) if sk else None
    events={k:torch.cuda.Event(enable_timing=True) for k in ('origin','fetch_begin','fetch_end','consumer_begin','consumer_end','join')}
    # Warmup once per worker/query geometry; the ungated invocation uses zeroed
    # payload and discarded output and is outside measured repeats only.
    if warmup:
        m.fwd(q,device_payload,gpu_offsets,gpu_slots,phase,frontier,out,lse,g,0,-1,chrono)
        torch.cuda.synchronize()
    events['origin'].record();fetch.wait_event(events['origin']);compute.wait_event(events['origin'])
    with torch.cuda.stream(fetch):
        events['fetch_begin'].record()
        if plan is not None:plan.initialize((history_end+63)//64)
        initialized=torch.cuda.Event();initialized.record(fetch)
        # Submit first batch before the consumer so the suffix can make progress.
        first=next((i+1 for i,v in enumerate(val) if v>=0),len(val))
        if plan is not None:plan.enqueue(0,first)
    fetch_chrono=chrono and chrono_placement=='fetch'
    if chrono_placement not in ('consumer','fetch'):raise ValueError('unknown Chrono placement')
    if fetch_chrono:
        with torch.cuda.stream(fetch):
            if plan is not None:plan.enqueue(first,len(val))
            events['fetch_end'].record()
            _rotate_fetched(device_payload,gpu_offsets,gpu_slots,phase,total,g)
            rotated=torch.cuda.Event();rotated.record(fetch)
        compute.wait_event(rotated)
    else:
        compute.wait_event(initialized)
    with torch.cuda.stream(compute):
        events['consumer_begin'].record()
        m.fwd(q,device_payload,gpu_offsets,gpu_slots,phase,frontier,out,lse,g,0,(history_end-1)//64,chrono and not fetch_chrono)
        events['consumer_end'].record()
    if not fetch_chrono:
        with torch.cuda.stream(fetch):
            if plan is not None:plan.enqueue(first,len(val))
            events['fetch_end'].record()
    current=torch.cuda.current_stream(q.device)
    current.wait_event(events['fetch_end']);current.wait_event(events['consumer_end']);events['join'].record();events['join'].synchronize()
    timings={k:events['origin'].elapsed_time(v) for k,v in events.items() if k!='origin'}
    return out,dict(source_identity=m.source_identity,h2d_bytes=plan.bytes() if plan is not None else 0,dma_count=len(sk),
        source_mode='original_gpu_prepacked_pinned_pairs',chrono_placement=chrono_placement,cuda_ms=timings,
        h2d_ms=timings['fetch_end']-timings['fetch_begin'],gpu_critical_ms=timings['join'],
        metadata_bytes=gpu_offsets.numel()*8+gpu_slots.numel()*4,source_pinned=all(t.is_pinned() for t in sk),
        _lse=lse,_payload=device_payload,_offsets=gpu_offsets)


_ROTATE_KERNEL=None

def _rotate_fetched(payload,offsets,slots,phase,total,group,*,preload=False,heads=12):
    global _ROTATE_KERNEL, tl
    import triton
    import triton.language as tl
    if _ROTATE_KERNEL is None:
        @triton.jit
        def rotate(P,O,S,F,T:tl.constexpr,G:tl.constexpr):
            rows=tl.program_id(0)*32+tl.arange(0,32)
            head=tl.program_id(1)
            pair=tl.arange(0,32)
            base=tl.load(O+(head//G*2)*T+rows,rows<T,0)
            slot=tl.load(S+rows,rows<T,0)
            address=base[:,None]+(head%G)*128+pair[None,:]*2
            a=tl.load(P+address,(rows[:,None]<T)&(pair[None,:]<22),0).to(tl.float64)
            b=tl.load(P+address+1,(rows[:,None]<T)&(pair[None,:]<22),0).to(tl.float64)
            c=tl.load(F+slot[:,None]*44+pair[None,:]*2,(rows[:,None]<T)&(pair[None,:]<22),0)
            d=tl.load(F+slot[:,None]*44+pair[None,:]*2+1,(rows[:,None]<T)&(pair[None,:]<22),0)
            tl.store(P+address,(a*c-b*d).to(tl.float32).to(tl.bfloat16),(rows[:,None]<T)&(pair[None,:]<22))
            tl.store(P+address+1,(a*d+b*c).to(tl.float32).to(tl.bfloat16),(rows[:,None]<T)&(pair[None,:]<22))
        _ROTATE_KERNEL=rotate
    grid=(triton.cdiv(total,32),heads)
    if preload:
        compiled=_ROTATE_KERNEL.warmup(payload,offsets,slots,phase,total,group,grid=grid,enable_fp_fusion=False)
        # Triton warmup compiles but load_binary is lazy until .run is read.
        # Load BEFORE any stream wait on an as-yet-unsubmitted producer job.
        _=compiled.run
        return
    _ROTATE_KERNEL[grid](payload,offsets,slots,phase,total,group,enable_fp_fusion=False)
