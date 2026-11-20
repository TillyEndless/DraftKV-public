"""H3 temporal-only rotation, retaining native intermediate BF16 rounding."""
_FETCH = _QUERY = None


def _kernels():
    global _FETCH, _QUERY, tl
    import triton
    import triton.language as tl
    if _FETCH is not None:
        return _FETCH, _QUERY

    @triton.jit
    def fetched(P,O,S,F,T:tl.constexpr,H:tl.constexpr):
        rows=tl.program_id(0)*32+tl.arange(0,32);head=tl.program_id(1)
        d=tl.arange(0,16)
        base=tl.load(O+rows,rows<T,0)+head*128
        slot=tl.load(S+rows,rows<T,0)
        a=tl.load(P+base[:,None]+d[None,:],rows[:,None]<T,0).to(tl.float32)
        b=tl.load(P+base[:,None]+48+d[None,:],rows[:,None]<T,0).to(tl.float32)
        c=tl.load(F+slot[:,None]*32+d[None,:],rows[:,None]<T,0)
        s=tl.load(F+slot[:,None]*32+16+d[None,:],rows[:,None]<T,0)
        ac=(a*c).to(tl.bfloat16).to(tl.float32)
        bs=(-b*s).to(tl.bfloat16).to(tl.float32)
        bc=(b*c).to(tl.bfloat16).to(tl.float32)
        ass=(a*s).to(tl.bfloat16).to(tl.float32)
        tl.store(P+base[:,None]+d[None,:],ac+bs,rows[:,None]<T)
        tl.store(P+base[:,None]+48+d[None,:],bc+ass,rows[:,None]<T)

    @triton.jit
    def query(Q,R,F,O,N:tl.constexpr,H:tl.constexpr,FT:tl.constexpr,START:tl.constexpr,
              QT:tl.constexpr,QH:tl.constexpr,RT:tl.constexpr,RH:tl.constexpr):
        rows=tl.program_id(0)*32+tl.arange(0,32);head=tl.program_id(1);d=tl.arange(0,128)
        mask=rows[:,None]<N
        native=tl.load(Q+rows[:,None]*QT+head*QH+d[None,:],mask,0)
        pair=tl.maximum(0,tl.minimum(15,tl.where(d<16,d,d-48)))
        a=tl.load(R+rows[:,None]*RT+head*RH+pair[None,:],mask,0).to(tl.float32)
        b=tl.load(R+rows[:,None]*RT+head*RH+48+pair[None,:],mask,0).to(tl.float32)
        slot=START+rows//FT
        c=tl.load(F+slot[:,None]*32+pair[None,:],mask,0)
        s=tl.load(F+slot[:,None]*32+16+pair[None,:],mask,0)
        ac=(a*c).to(tl.bfloat16).to(tl.float32);bs=(-b*s).to(tl.bfloat16).to(tl.float32)
        bc=(b*c).to(tl.bfloat16).to(tl.float32);ass=(a*s).to(tl.bfloat16).to(tl.float32)
        rotated=tl.where(d[None,:]<16,ac+bs,bc+ass)
        temporal=(d[None,:]<16)|((d[None,:]>=48)&(d[None,:]<64))
        tl.store(O+(rows[:,None]*H+head)*128+d[None,:],tl.where(temporal,rotated,native),mask)
    _FETCH,_QUERY=fetched,query
    return _FETCH,_QUERY


def rotate_h3_fetched(payload,offsets,slots,phase,total,heads,*,preload=False):
    import triton
    kernel,_=_kernels();grid=(triton.cdiv(total,32),heads)
    if preload:
        compiled=kernel.warmup(payload,offsets,slots,phase,total,heads,grid=grid,enable_fp_fusion=False)
        _=compiled.run
    else:
        kernel[grid](payload,offsets,slots,phase,total,heads,enable_fp_fusion=False)


def rotate_h3_query(q,raw,phase,ft,start,*,preload=False):
    import torch
    import triton
    _,kernel=_kernels();out=torch.empty_like(q,memory_format=torch.contiguous_format)
    grid=(triton.cdiv(q.shape[1],32),q.shape[2])
    args=(q,raw,phase,out,q.shape[1],q.shape[2],ft,start,q.stride(1),q.stride(2),raw.stride(1),raw.stride(2))
    if preload:
        compiled=kernel.warmup(*args,grid=grid,enable_fp_fusion=False);_=compiled.run
    else:
        kernel[grid](*args,enable_fp_fusion=False)
    return out
