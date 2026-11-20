"""Independent SM120 BF16 paired-cache consumer; frozen SM89 implementation is reused read-only for GPU preparation and geometry."""
from functools import lru_cache
from pathlib import Path
import hashlib
import os
import tarfile
from .prepacked import prepare, layout, BlockRef, _rotate_fetched

@lru_cache(None)
def native():
    import torch
    from torch.utils.cpp_extension import load
    if torch.cuda.get_device_capability() != (12,0):raise RuntimeError('prepacked experiment requires SM120')
    here = Path(__file__).with_name('prepacked_sm120_cuda')
    vendor = here.parents[1] / 'vendor/b01/source_snapshot.tar.gz'
    files = sorted(here.glob('*'))
    digest = hashlib.sha256(vendor.read_bytes())
    for p in files:
        digest.update(p.name.encode()); digest.update(p.read_bytes())
    identity = digest.hexdigest()[:20]
    root = Path(os.environ.get('DRAFTKV_PREPACK_SM120_BUILD', '/tmp/draftkv-prepacked-sm120')) / identity
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
    os.environ['TORCH_CUDA_ARCH_LIST']='12.0'
    os.environ.setdefault('MAX_JOBS', '2')
    module = load(name='draftkv_prepacked_sm120_'+identity,
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


def execute(q,refs,phase, *, batch_bytes=2**20, warmup=True, cache=None, chrono_placement="consumer",splits=4):
    """Direct paired DMA, descending frontier, and mapped BF16 FA2 consumer.

    refs may change every call. Only address/slot metadata is packed by the CPU.
    KV bytes always come directly from the original pinned clean slabs.
    """
    import numpy as np
    import torch
    if splits not in (1,2,4):raise ValueError("SM120 requires S1/S2/S4")
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
    partial=reserve('partial',(splits,1,12,q.shape[1],128) if splits>1 else (0,),torch.float32,q.device)
    partial_lse=reserve('partial_lse',(splits,1,12,q.shape[1]) if splits>1 else (0,),torch.float32,q.device)
    fetch,compute=cache['streams']
    plan=m.PairedProducer(dk,dk,sk,sk,frontier,ix,val) if sk else None
    events={k:torch.cuda.Event(enable_timing=True) for k in ('origin','fetch_begin','fetch_end','consumer_begin','consumer_end','join')}
    # Warmup once per worker/query geometry; the ungated invocation uses zeroed
    # payload and discarded output and is outside measured repeats only.
    if warmup:
        m.fwd(q,device_payload,gpu_offsets,gpu_slots,phase,frontier,out,lse,g,0,-1,chrono,splits,partial,partial_lse)
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
        m.fwd(q,device_payload,gpu_offsets,gpu_slots,phase,frontier,out,lse,g,0,(history_end-1)//64,chrono and not fetch_chrono,splits,partial,partial_lse)
        events['consumer_end'].record()
    if not fetch_chrono:
        with torch.cuda.stream(fetch):
            if plan is not None:plan.enqueue(first,len(val))
            events['fetch_end'].record()
    current=torch.cuda.current_stream(q.device)
    current.wait_event(events['fetch_end']);current.wait_event(events['consumer_end']);events['join'].record();events['join'].synchronize()
    timings={k:events['origin'].elapsed_time(v) for k,v in events.items() if k!='origin'}
    return out,dict(source_identity=m.source_identity,consumer_backend='sm120_splitkv',splits=splits,split_workspace_bytes=(partial.numel()+partial_lse.numel())*4,h2d_bytes=plan.bytes() if plan is not None else 0,dma_count=len(sk),
        source_mode='original_gpu_prepacked_pinned_pairs',chrono_placement=chrono_placement,cuda_ms=timings,
        h2d_ms=timings['fetch_end']-timings['fetch_begin'],gpu_critical_ms=timings['join'],
        metadata_bytes=gpu_offsets.numel()*8+gpu_slots.numel()*4,source_pinned=all(t.is_pinned() for t in sk),
        _lse=lse,_payload=device_payload,_offsets=gpu_offsets)
