"""GPU DMA-table generation for canonical immutable archive spans.

The established C++ AsyncProducer submits copy-engine DMA and per-unit
system-scope writes. No mapped-host GPU loads and no CPU KV gathering.
"""
_KERNELS = None


def kernels():
    global _KERNELS, tl
    if _KERNELS is not None: return _KERNELS
    import triton
    import triton.language as tl

    @triton.jit
    def counts(Runs, Counts, R:tl.constexpr, S:tl.constexpr,T:tl.constexpr,
               A:tl.constexpr,N:tl.constexpr,B:tl.constexpr,REVERSE:tl.constexpr):
        unit=tl.program_id(0); s=S-1-unit%S if REVERSE else unit%S; g=unit//S
        i=tl.arange(0,B)
        dst=tl.load(Runs+(g*R+i)*4+2,i<R,0)
        n=tl.load(Runs+(g*R+i)*4+3,i<R,0)
        lo=tl.maximum(dst,tl.maximum(0,T*s//S-A))
        hi=tl.minimum(dst+n,tl.minimum(N,T*(s+1)//S-A))
        tl.store(Counts+unit,2*tl.sum(((i<R)&(hi>lo)).to(tl.int32),0)+1)

    @triton.jit
    def emit(Runs,Counts,Table,DEST,R:tl.constexpr,S:tl.constexpr,
             T:tl.constexpr,A:tl.constexpr,N:tl.constexpr,G:tl.constexpr,
             UNITS:tl.constexpr,B:tl.constexpr,UB:tl.constexpr,REVERSE:tl.constexpr):
        unit=tl.program_id(0); s=S-1-unit%S if REVERSE else unit%S; g=unit//S
        u=tl.arange(0,UB); c=tl.load(Counts+u,u<UNITS,0)
        base=tl.sum(tl.where(u<unit,c,0),0)
        total=tl.sum(c,0)
        if unit==0:
            tl.store(Table,total);tl.store(Table+1,N*(UNITS//S)*G*128*4)
        i=tl.arange(0,B)
        src_k=tl.load(Runs+(g*R+i)*4,i<R,0)
        src_v=tl.load(Runs+(g*R+i)*4+1,i<R,0)
        dst=tl.load(Runs+(g*R+i)*4+2,i<R,0)
        n=tl.load(Runs+(g*R+i)*4+3,i<R,0)
        lo=tl.maximum(dst,tl.maximum(0,T*s//S-A))
        hi=tl.minimum(dst+n,tl.minimum(N,T*(s+1)//S-A))
        valid=(i<R)&(hi>lo)
        # Match historical group_descriptions: later physical blocks first
        # within a unit; K and V of each block remain adjacent descriptors.
        rank=tl.sum(valid.to(tl.int32),0)-tl.cumsum(valid.to(tl.int32),0)
        size=(hi-lo).to(tl.int64)*G*128*2
        src_off=(lo-dst).to(tl.int64)*G*128*2
        target=DEST+((g*2*T+A+lo).to(tl.int64)*G*128*2)
        idx=g*S+s
        for side in tl.static_range(2):
            row=8+(base+2*rank+side)*5
            tl.store(Table+row,tl.where(side==0,src_k,src_v)+src_off,valid)
            tl.store(Table+row+1,target+side*T*G*128*2,valid)
            tl.store(Table+row+2,size,valid)
            tl.store(Table+row+3,idx,valid)
            tl.store(Table+row+4,-1,valid)
        end=8+(base+tl.load(Counts+unit)-1)*5
        tl.store(Table+end,0);tl.store(Table+end+1,0);tl.store(Table+end+2,0)
        tl.store(Table+end+3,idx);tl.store(Table+end+4,1)
    _KERNELS=counts,emit
    return _KERNELS


def submit_archive(plan, source):
    import torch
    metadata=source.metadata
    if metadata is None:
        rows=[]
        for group in range(source.groups):
            for run in source.runs:
                k=run.frame.block(run.block,group,0)
                v=run.frame.block(run.block,group,1)
                offset=run.source_begin*source.head_group*128*2
                rows.append((k.data_ptr()+offset,v.data_ptr()+offset,run.destination,run.count))
        metadata=torch.tensor(rows,dtype=torch.int64,device=plan.device).view(source.groups,-1,4)
    return _submit(plan,source,metadata,source.history_tokens)


def submit_group_major(plan, source):
    """Use the same producer for CPU-RoPE's already prepared pinned source.

    Only one K/V address pair per head group crosses the control plane.
    The source payload is neither read nor repacked here. The existing GPU
    descriptor builder splits these continuous ranges into consumer units.
    """
    import torch
    n=source.key.shape[1]
    rows=[(source.key[g].data_ptr(),source.value[g].data_ptr(),0,n)
          for g in range(source.groups)]
    metadata=torch.tensor(rows,dtype=torch.int64,device=plan.device).view(source.groups,1,4)
    return _submit(plan,source,metadata,n)


def _submit(plan, source, metadata, history_tokens):
    import torch
    import triton
    actual=metadata.shape[1]
    # Specialize on a power-of-two capacity, never dynamic selected-run count
    # or a receive allocation's address. Otherwise each route recompiles.
    r=triton.next_power_of_2(actual)
    padded=plan.reserve('canonical_dma_spans',(source.groups,r,4),torch.int64)
    padded.zero_();padded[:,:actual].copy_(metadata);metadata=padded
    s=plan.config.splits;u=source.groups*s
    count=plan.reserve('canonical_dma_counts',(u,),torch.int32)
    table=plan.reserve('canonical_dma_table',(8+5*u*(2*r+1),),torch.int64)
    host=plan.reserve('canonical_dma_description',table.shape,torch.int64,True)
    c,e=kernels()
    c[(u,)](metadata,count,r,s,plan.total,source.history_start,history_tokens,
             triton.next_power_of_2(r),plan.config.reverse)
    e[(u,)](metadata,count,table,plan.dense.data_ptr(),r,s,plan.total,
        source.history_start,history_tokens,source.head_group,u,
        triton.next_power_of_2(r),triton.next_power_of_2(u),plan.config.reverse)
    plan.producer.submit(table,host,plan.dma_ready,plan.dense,plan.fetch.cuda_stream)
    return history_tokens*source.heads*128*4
