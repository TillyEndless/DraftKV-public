"""Independent SM120 BF16 paired-cache consumer; frozen SM89 implementation is reused read-only for GPU preparation and geometry."""
from functools import lru_cache
from pathlib import Path
import hashlib
import os
import tarfile

@lru_cache(None)
def native():
    import torch
    from torch.utils.cpp_extension import load
    if torch.cuda.get_device_capability() != (12,0):raise RuntimeError('prepacked experiment requires SM120')
    here = Path(__file__).with_name('routed_sm120_cuda')
    vendor = here.parents[1] / 'vendor/b01/source_snapshot.tar.gz'
    files = sorted(here.glob('*'))
    digest = hashlib.sha256(vendor.read_bytes())
    for p in files:
        digest.update(p.name.encode()); digest.update(p.read_bytes())
    identity = digest.hexdigest()[:20]
    root = Path(os.environ.get('DRAFTKV_ROUTE_SM120_BUILD', '/tmp/draftkv-routed-sm120')) / identity
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
    module = load(name='draftkv_routed_sm120_'+identity,
        sources=[str(here/'prepacked_binding.cpp'),str(here/'prepacked_kernels.cu'),str(here/'route.cu')],
        extra_include_paths=[str(here),str(fa),str(flash),str(flash/'src'),cutlass],
        extra_cflags=['-O3','-std=c++17'],
        extra_cuda_cflags=['-O3','-std=c++17','--expt-relaxed-constexpr','--expt-extended-lambda',
            '-U__CUDA_NO_HALF_OPERATORS__','-U__CUDA_NO_HALF_CONVERSIONS__',
            '-U__CUDA_NO_HALF2_OPERATORS__','-U__CUDA_NO_BFLOAT16_CONVERSIONS__',
            '-DFLASHATTENTION_DISABLE_DROPOUT','-DFLASHATTENTION_DISABLE_ALIBI',
            '-DFLASHATTENTION_DISABLE_SOFTCAP','-DFLASHATTENTION_DISABLE_LOCAL'],
        extra_ldflags=['-lcuda'],build_directory=str(root),verbose=False)
    module.info(1);module.info(4);module.initialize_route_kernels()  # Resolve kernels/attributes before asynchronous submission.
    module.source_identity=identity
    return module


# Only this module opts into the new runtime. The frozen SM89/SM120 paths are untouched.
from dataclasses import dataclass
from .prepacked import PreparedChunk, _device_geometry

def pool_query(q,h,w,ph,pw,cache=None):
    """One gathered load, with the exact eager FP32 sum tree and FP16 narrowing."""
    import torch
    index,counts,_=_device_geometry(h,w,ph,pw,12,str(q.device))
    frames=q.shape[1]//(h*w);r=counts.numel();shape=(2,1,12,frames*r,128)
    key=('q_pool',frames,r)
    out=None if cache is None else cache.get(key)
    if out is None:
        out=torch.empty(shape,dtype=torch.float16,device=q.device)
        if cache is not None:cache[key]=out
    native().pool_query_exact(q,out,index,counts,h*w,ph*pw)
    return out[0],out[1]


def prepare_current(key,value,raw,h,w,ph,pw,chrono,cache):
    """Two GPU slots with device event dependencies; no per-layer host fence."""
    import torch
    from .prepacked import _kernel
    slots=cache.setdefault('slots',[{},{}]);slot=slots[cache.get('next',0)%len(slots)];cache['next']=cache.get('next',0)+1
    if 'release' in slot:torch.cuda.current_stream().wait_event(slot['release'])
    count=2*key.numel();frames=key.shape[1]//(h*w)
    if slot.get('packed') is None or slot['packed'].numel()<count:
        slot['packed']=torch.empty(count,device=key.device,dtype=key.dtype)
    packed=slot['packed'][:count]
    index,counts,starts=_device_geometry(h,w,ph,pw,12,str(key.device));r=counts.numel()
    shape=(frames,2,12,r,128)
    if slot.get('desc') is None or tuple(slot['desc'].shape)!=shape:
        slot['desc']=torch.empty(shape,device=key.device,dtype=torch.float16)
    desc=slot['desc'];begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
    begin.record()
    source=raw if chrono else key
    _kernel()[(frames,r,12)](key,value,source,packed,desc,index,counts,starts,
        frames,h*w,12,12,r,ph*pw,chrono,key.stride(1),key.stride(2),value.stride(1),value.stride(2),source.stride(1),source.stride(2),num_warps=8)
    end.record();slot['release']=end
    chunk=PreparedChunk(packed,desc,end,frames,h,w,ph,pw,12,chrono,begin,end,end)
    chunk._slot=slot
    return chunk


def scores(q,bank,sink,start,h,w,config,cache=None):
    """Same two GEMMs, FP16 logits/probability and FP32 reduction order as fcd65a9."""
    import torch
    qm,qx=pool_query(q,h,w,config.pool_h,config.pool_w,cache)
    km=bank[0,:,sink:start].reshape(1,12,-1,128)
    kx=bank[1,:,sink:start].reshape(1,12,-1,128)
    scale=128**-.5
    prob=(torch.matmul(qm.float(),km.float().transpose(-2,-1))*scale).half().float().softmax(-1)
    weight=float(config.maxpool_weight)
    if weight:
        maximum=(torch.matmul(qx.float(),kx.float().transpose(-2,-1))*scale).half().float().softmax(-1)
        prob=(1-weight)*prob+weight*maximum
    prob=prob.half()
    return (prob.float().sum(2).sum(1)/(prob.shape[1]*prob.shape[2])).flatten()


@dataclass
class GPUPlan:
    q: object
    payload: object
    offsets: object
    slots: object
    sources: object
    table: object
    host_table: object
    workspace: object
    phase: object
    frontier: object
    flags: object
    out: object
    lse: object
    partial: object
    partial_lse: object
    producer: object
    current: object
    history_tokens: int
    total: int
    start: int
    sink: int
    budget: int
    route_events: object


def plan_route(q,current,bank,addresses,phase,h,w,config,start,sink,budget,protect,splits,batch,cache,score_query=None,validation=False):
    import torch
    m=native();ft=h*w;r=bank.shape[-2];total=(sink+budget+current.frames)*ft
    reserve=(sink+budget)*ft+config.capacity
    def get(name,shape,dtype,device=q.device):
        key=(name,dtype,str(device));n=1
        for extent in shape:n*=extent
        t=cache.get(key)
        if t is None or t.numel()<n:
            t=torch.empty(n,dtype=dtype,device=device,pin_memory=str(device)=='cpu');cache[key]=t
        return t[:n].view(shape)
    begin=torch.cuda.Event(enable_timing=True);scored=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);begin.record()
    sc=scores(q if score_query is None else score_query,bank,sink,start,h,w,config,cache) if start>sink else torch.empty(0,dtype=torch.float32,device=q.device)
    scored.record()
    payload=get('payload',(2*(reserve+current.frames*ft)*12*128,),torch.bfloat16)
    payload[2*reserve*12*128:2*(reserve+current.frames*ft)*12*128].copy_(current.slab,non_blocking=True)
    offsets=get('offsets',(1,2,total),torch.int64);slots=get('slots',(total,),torch.int32);sources=get('sources',(total if validation else 0,),torch.int64)
    all_blocks=(start+current.frames)*r;n=(start-sink-protect)*r
    wkey=('workspace',start,current.frames,sink,protect)
    if wkey not in cache:
        # CUB workspace query is metadata-only and never reads a device scalar.
        size=m.route_workspace_bytes(n,all_blocks)
        cache[wkey]=[torch.empty(max(n,1),dtype=torch.float32,device=q.device)]+[torch.empty(max(n,all_blocks,1),dtype=torch.int32,device=q.device) for _ in range(9)]+[torch.empty(size,dtype=torch.uint8,device=q.device)]
    work=cache[wkey]
    table=get('dma_table',(start*r+1,5),torch.int64);host=get('dma_table',(start*r+1,5),torch.int64,'cpu')
    index,counts,starts=_device_geometry(h,w,config.pool_h,config.pool_w,12,str(q.device))
    m.gpu_route(sc,addresses,counts,starts,index,work,offsets,slots,sources,table,payload,sink,start,current.frames,protect,budget,ft,config.capacity,reserve,batch,current.chrono)
    frontier=get('frontier',(1,),torch.int32);frontier.fill_(((sink+budget)*ft+63)//64)
    flags=get('flags',(2,),torch.int32);flags.zero_()
    out=get('out',q.shape,torch.bfloat16);lse=get('lse',(1,12,q.shape[1]),torch.float32)
    partial=get('partial',(splits,1,12,q.shape[1],128) if splits>1 else (0,),torch.float32)
    partial_lse=get('partial_lse',(splits,1,12,q.shape[1]) if splits>1 else (0,),torch.float32)
    if 'producer' not in cache:cache['producer']=m.RouteProducer(q.device.index)
    end.record()
    return GPUPlan(q,payload,offsets,slots,sources,table,host,work,phase,frontier,flags,out,lse,partial,partial_lse,cache['producer'],current,(sink+budget)*ft,total,start,sink,budget,(begin,scored,end))


def execute_plan(plan,splits):
    p=plan
    return p.producer.submit([p.table,p.host_table,p.q,p.payload,p.offsets,p.slots,p.phase,p.frontier,p.out,p.lse,p.partial,p.partial_lse,p.flags],splits,p.history_tokens,p.current.chrono)
