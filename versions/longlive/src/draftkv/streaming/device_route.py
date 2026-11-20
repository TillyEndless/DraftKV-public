"""Device-only selection/indexing with host copy-engine API submission.

No host-visible dynamic scores or per-token lists exist in the production path.
The host descriptor ring contains only DMA spans and frontier publications.
"""
from functools import lru_cache
from pathlib import Path
import hashlib
import os
import tarfile
import bisect
import math
from .prepacked import layout, _device_geometry, _kernel, _rotate_fetched

@lru_cache(None)
def native():
    import torch
    from torch.utils.cpp_extension import load
    capability = torch.cuda.get_device_capability()
    if capability not in ((8, 9), (9, 0), (12, 0)):
        raise RuntimeError('device route requires SM89, SM90 or SM120')
    # SM120 uses the installed native FA2 through HeadGroupPlan. Do not compile
    # or accidentally launch the old SM89 attention consumer on Blackwell.
    native_only = capability == (12, 0)
    architecture = '.'.join(map(str, capability))
    here = Path(__file__).with_name('device_route_cuda')
    shared = here.with_name('prepacked_cuda')
    vendor = here.parents[1] / 'vendor/b01/source_snapshot.tar.gz'
    files = sorted(here.glob('*')) + sorted(shared.glob('*.h'))
    digest = hashlib.sha256(vendor.read_bytes())
    digest.update(f'{architecture}:{torch.__version__}:{torch.version.cuda}'.encode())
    for p in files:
        digest.update(p.name.encode()); digest.update(p.read_bytes())
    identity = digest.hexdigest()[:20]
    root = Path(os.environ.get('DRAFTKV_PREPACK_BUILD', '/tmp/draftkv-device-route')) / identity
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
    module = load(name='draftkv_device_route_'+identity,
        sources=[str(here/'binding.cpp'),str(here/'route.cu')] +
                ([] if native_only else [str(here/'kernels.cu')]),
        extra_include_paths=[str(here),str(shared),str(fa),str(flash),str(flash/'src'),cutlass],
        extra_cflags=['-O3','-std=c++17'] + (['-DDRAFTKV_NATIVE_ONLY'] if native_only else []),
        extra_cuda_cflags=['-O3','-std=c++17','--expt-relaxed-constexpr','--expt-extended-lambda',
            '-U__CUDA_NO_HALF_OPERATORS__','-U__CUDA_NO_HALF_CONVERSIONS__',
            '-U__CUDA_NO_HALF2_OPERATORS__','-U__CUDA_NO_BFLOAT16_CONVERSIONS__',
            '-DFLASHATTENTION_DISABLE_DROPOUT','-DFLASHATTENTION_DISABLE_ALIBI',
            '-DFLASHATTENTION_DISABLE_SOFTCAP','-DFLASHATTENTION_DISABLE_LOCAL'],
        extra_ldflags=['-lcuda'],build_directory=str(root),verbose=False)
    module.source_identity=identity
    return module


@lru_cache(None)
def geometry(h,w,ph,pw,device):
    import torch
    blocks,_,starts=layout(h,w,ph,pw,12)
    cap=ph*pw
    # Static geometry: for each token, count earlier raster tokens per block.
    # This is independent of scores/history, uploaded only once per geometry.
    before=[[[bisect.bisect_left(other,t) for other in blocks] for t in block]+
            [[0]*len(blocks)]*(cap-len(block)) for block in blocks]
    return (*_device_geometry(h,w,ph,pw,12,device),
            torch.tensor(before,dtype=torch.int32,device=device))


_POOL=None

def pool(query,h,w,ph,pw,out=None):
    # Keep lazy imports while supporting Triton 3.2's global-name resolver.
    global _POOL, tl
    import torch
    import triton
    import triton.language as tl
    if _POOL is None:
        @triton.jit
        def fused(Q,O,I,N,T:tl.constexpr,R:tl.constexpr,C:tl.constexpr,F:tl.constexpr,
                  QT:tl.constexpr,QH:tl.constexpr,HH:tl.constexpr):
            f=tl.program_id(0);b=tl.program_id(1);head=tl.program_id(2)
            t=tl.arange(0,C);d=tl.arange(0,128)
            n=tl.load(N+b);idx=tl.load(I+b*C+t,t<n,0)
            x=tl.load(Q+(f*T+idx[:,None])*QT+head*QH+d[None,:],t[:,None]<n,0).to(tl.float16).to(tl.float32)
            mean=tl.div_rn(tl.sum(x,0),n.to(tl.float32)).to(tl.float16)
            maximum=tl.max(tl.where(t[:,None]<n,x,float('-inf')),0).to(tl.float16)
            dst=((head*F+f)*R+b)*128+d
            tl.store(O+dst,mean);tl.store(O+HH*F*R*128+dst,maximum)
        _POOL=fused
    f=query.shape[1]//(h*w);index,counts,_,_=geometry(h,w,ph,pw,str(query.device));r=counts.numel()
    heads=query.shape[2]
    if out is None:out=torch.empty((2,1,heads,f*r,128),device=query.device,dtype=torch.float16)
    _POOL[(f,r,heads)](query,out,index,counts,h*w,r,ph*pw,f,query.stride(1),query.stride(2),heads,num_warps=8)
    return out[0],out[1]


def scores(query,bank,h,w,ph,pw,sink,start):
    import torch
    qm,qx=pool(query,h,w,ph,pw)
    r=((h+ph-1)//ph)*((w+pw-1)//pw)
    # Bank is [2,1,H,capacity*R,D] FP32, narrowed once at clean prepare.
    km,kx=bank[0,:,:,sink*r:start*r],bank[1,:,:,sink*r:start*r]
    scale=128**-.5
    lm=(qm.float()@km.transpose(-2,-1)*scale).to(torch.float16)
    lx=(qx.float()@kx.transpose(-2,-1)*scale).to(torch.float16)
    p=(.96*lm.float().softmax(-1)+.04*lx.float().softmax(-1)).to(torch.float16)
    return (p.float().sum(2).sum(1)/(p.shape[1]*p.shape[2])).flatten()


class DevicePlan:
    def __init__(self,h,w,capacity,device,heads=12,family='wan',ph=8,pw=16):
        import torch
        self.h,self.w,self.capacity,self.device=h,w,capacity,device
        self.heads,self.family=heads,family
        if (ph,pw)!=(8,16):raise ValueError('B device route requires Spatial Block 8x16')
        self.ph,self.pw,self.block_capacity=ph,pw,ph*pw
        self.blocks,self.inverse,self.starts=layout(h,w,ph,pw,12)
        self.r=len(self.blocks);self.ft=h*w
        if capacity*self.r>65535:raise ValueError('device route capacity exceeds supported 65535 blocks')
        native().preload()  # Resolve CUDA lazy function loading before any ready wait.
        self.fetch=torch.cuda.Stream(device=device);self.compute=torch.cuda.Stream(device=device)
        self.producer=native().AsyncProducer();self.buffers={};self.records=[]
        self.workspace_bytes=native().route_workspace(capacity*self.r)
    def reserve(self,name,shape,dtype,cpu=False):
        import torch
        n=math.prod(shape);key=(name,dtype,cpu)
        x=self.buffers.get(key)
        if x is None or x.numel()<n:
            x=torch.empty(n,dtype=dtype,device='cpu' if cpu else self.device,pin_memory=cpu)
            self.buffers[key]=x
        return x[:n].view(shape)
    def route(self,score,packed,addresses,sink,start,current,budget,protect,chrono,batch,*,submit=True,prefix_tokens=0,index_only=False):
        import torch
        m=native();n=(start-sink)*self.r;total=(sink+budget+current)*self.ft
        self.total,self.history,self.chrono=total,(sink+budget)*self.ft,chrono
        reserve=self.reserve
        keys=reserve('keys',(max(1,n),),torch.float32);sortkeys=reserve('sortedkeys',keys.shape,keys.dtype)
        ids=reserve('ids',keys.shape,torch.int32);sortedids=reserve('sortedids',keys.shape,torch.int32)
        workspace=reserve('workspace',(self.workspace_bytes,),torch.uint8)
        self.take=reserve('take',((start+current)*self.r,),torch.int32)
        self.meta=reserve('meta',(self.take.numel(),4),torch.int64)
        self.offset=reserve('offset',(1,2,total),torch.int64);self.slots=reserve('slots',(total,),torch.int32)
        self.prefix_base=(total+max(map(len,self.blocks)))*self.heads*256
        self.payload=reserve('payload',(1 if index_only else self.prefix_base+prefix_tokens*self.heads*256,),torch.bfloat16)
        self.frontier=reserve('frontier',(1,),torch.int32)
        # Fixed capacity small transfer; count and rows share one D2H operation.
        table=reserve('table',(8+5*self.capacity*self.r,),torch.int64)
        cpu=reserve('description',table.shape,torch.int64,True)
        _,counts,starts,before=geometry(self.h,self.w,self.ph,self.pw,str(self.device))
        origin=torch.cuda.Event(enable_timing=True);route_end=torch.cuda.Event(enable_timing=True)
        origin.record()
        m.route(score,keys,sortkeys,ids,sortedids,workspace,self.take,counts,before,starts,
            self.meta,self.offset,self.slots,addresses,table,self.frontier,packed,self.payload,
            sink,start,current,budget,protect,self.ft,self.r,self.block_capacity,batch,chrono,self.heads,index_only,getattr(self,'_ablation_skip_topk',False))
        route_end.record();self.ready=route_end
        self.table=table
        if submit:self.producer.submit(table,cpu,self.frontier,self.payload,self.fetch.cuda_stream)
        return origin,route_end
    def execute(self,q,phase,route_events,prefix=None):
        import torch
        m=native();out=self.reserve('out',q.shape,q.dtype)
        lse=self.reserve('lse',(1,self.heads,q.shape[1]),torch.float32)
        self.compute.wait_event(self.ready)
        a,b,c=[torch.cuda.Event(enable_timing=True) for _ in range(3)]
        with torch.cuda.stream(self.compute):
            a.record()
            if self.chrono:
                m.wait_ready(self.frontier)
                if self.family=='solarwm':
                    from .temporal_kernels import rotate_h3_fetched
                    rotate_h3_fetched(self.payload,self.offset,self.slots,phase,self.total,self.heads)
                else:
                    _rotate_fetched(self.payload,self.offset,self.slots,phase,self.total,self.heads,heads=self.heads)
            b.record()
            offset,slots,last=self.offset,self.slots,(self.history-1)//64
            if prefix is not None:
                # Resident condition/audio K/V retain their unshifted native basis.
                # Their offsets are assembled on GPU; no host token map exists.
                pk,pv=prefix;n=pk.shape[1];width=self.heads*128;base=self.prefix_base
                self.payload[base:base+n*width].copy_(pk.reshape(-1))
                self.payload[base+n*width:base+2*n*width].copy_(pv.reshape(-1))
                offset=self.reserve('prefix_offset',(1,2,n+self.total),torch.int64)
                positions=self.reserve('prefix_positions',(n,),torch.int64)
                torch.arange(n,device=q.device,out=positions)
                positions.mul_(width).add_(base)
                offset[0,0,:n].copy_(positions);offset[0,1,:n].copy_(positions).add_(n*width)
                offset[:,:,n:].copy_(self.offset)
                slots=self.reserve('prefix_slots',(n+self.total,),torch.int32)
                slots[:n].zero_();slots[n:].copy_(self.slots)
                # Chrono already waited for all DMA before rotation. Prefixes
                # change tile numbering, so disable the redundant frontier wait.
                if not self.chrono:m.wait_ready(self.frontier)
                last=-1
            m.fwd(q,self.payload,offset,slots,phase,self.frontier,out,lse,self.heads,0,last,False)
            c.record()
        torch.cuda.current_stream(self.device).wait_event(c)
        self.records.append((*route_events,a,b,c))
        self.end=c;self.lse=lse
        return out
    def collect(self):
        # Explicit bounded reporting boundary. No per-layer metric synchronization.
        rows=self.producer.collect()
        if self.records:self.records[-1][-1].synchronize()
        timings=[dict(route_index_ms=a.elapsed_time(b),fetch_attention_ms=c.elapsed_time(e),
                      rotation_wait_ms=c.elapsed_time(d),attention_ms=d.elapsed_time(e))
                 for a,b,c,d,e in self.records]
        self.records=[]
        return rows,timings


_QUERY_ROTATE=None

def rotate_query(query,raw,phase,ft,start,*,preload=False):
    global _QUERY_ROTATE, tl
    import torch
    import triton
    import triton.language as tl
    if _QUERY_ROTATE is None:
        @triton.jit
        def rotate(Q,R,P,O,N:tl.constexpr,FT:tl.constexpr,START:tl.constexpr,QT:tl.constexpr,QH:tl.constexpr,RT:tl.constexpr,RH:tl.constexpr,HH:tl.constexpr):
            rows=tl.program_id(0)*32+tl.arange(0,32);head=tl.program_id(1);d=tl.arange(0,128)
            native=tl.load(Q+rows[:,None]*QT+head*QH+d[None,:],rows[:,None]<N,0)
            pair=d//2
            a=tl.load(R+rows[:,None]*RT+head*RH+pair[None,:]*2,rows[:,None]<N,0).to(tl.float64)
            b=tl.load(R+rows[:,None]*RT+head*RH+pair[None,:]*2+1,rows[:,None]<N,0).to(tl.float64)
            slot=START+rows//FT
            c=tl.load(P+slot[:,None]*44+pair[None,:]*2,(rows[:,None]<N)&(d[None,:]<44),0)
            s=tl.load(P+slot[:,None]*44+pair[None,:]*2+1,(rows[:,None]<N)&(d[None,:]<44),0)
            rotated=tl.where(d[None,:]%2==0,a*c-b*s,a*s+b*c).to(tl.float32).to(tl.bfloat16)
            tl.store(O+(rows[:,None]*HH+head)*128+d[None,:],tl.where(d[None,:]<44,rotated,native),rows[:,None]<N)
        _QUERY_ROTATE=rotate
    out=torch.empty_like(query,memory_format=torch.contiguous_format)
    grid=(triton.cdiv(query.shape[1],32),query.shape[2])
    args=(query,raw,phase,out,query.shape[1],ft,start,query.stride(1),query.stride(2),raw.stride(1),raw.stride(2),query.shape[2])
    if preload:
        compiled=_QUERY_ROTATE.warmup(*args,grid=grid,enable_fp_fusion=False)
        _=compiled.run
        return
    _QUERY_ROTATE[grid](*args,enable_fp_fusion=False)
    return out
