"""DraftAttention Sec.3.1 Eq.(2)-(3): per-frame spatial 8x16 BF16 Draft Q/K.

Frame importance = sum of draft softmax over blocks belonging to each history
frame, then mean over query regions and local heads (our retrieval adaptation).
Full-resolution attention and KV content are not modified by this module.
"""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F

POOL_H, POOL_W = 8, 16

def spatial2d_regions_per_frame(grid_h: int, grid_w: int, pool_h: int = POOL_H,
                                pool_w: int = POOL_W) -> int:
    if min(grid_h, grid_w, pool_h, pool_w) <= 0:
        raise ValueError("Invalid spatial pooling geometry")
    return math.ceil(grid_h / min(pool_h, grid_h)) * math.ceil(grid_w / min(pool_w, grid_w))

def spatial2d_pool(x: torch.Tensor, grid_h: int, grid_w: int,
                   pool_h: int = POOL_H, pool_w: int = POOL_W) -> torch.Tensor:
    """[B, F*grid_h*grid_w, local_heads, d] -> [B,local_heads,F*regions,d]."""
    if x.ndim != 4 or x.dtype != torch.bfloat16:
        raise ValueError("Draft 2D Pool requires BF16 projected [B,T,H,D]")
    if min(grid_h, grid_w, pool_h, pool_w) <= 0:
        raise ValueError("Invalid spatial pooling size")
    b, tokens, heads, dim = x.shape
    frame_tokens = grid_h * grid_w
    if tokens <= 0 or tokens % frame_tokens:
        raise ValueError("Input must comprise complete frames, no temporal mixing")
    frames = tokens // frame_tokens
    ph, pw = min(pool_h, grid_h), min(pool_w, grid_w)
    grid = x.reshape(b, frames, grid_h, grid_w, heads, dim)
    image = grid.permute(0, 1, 4, 5, 2, 3).reshape(
        b * frames, heads * dim, grid_h, grid_w)
    reduced = F.avg_pool2d(
        image.float(), kernel_size=(ph, pw), stride=(ph, pw), ceil_mode=True)
    out_h, out_w = reduced.shape[-2:]
    result = reduced.reshape(b, frames, heads, dim, out_h, out_w).permute(
        0, 2, 1, 4, 5, 3).reshape(b, heads, frames * out_h * out_w, dim)
    return result.to(torch.bfloat16).contiguous()

def frame_importance(raw_q: torch.Tensor, descriptors: list[torch.Tensor],
                     grid_h: int, grid_w: int, *,
                     return_block_probs: bool = False):
    """Common numeric route: FP32 logits/softmax -> FP16 probs -> sum K regions -> mean Q/heads.
    Multi-rank models must use their existing distributed aggregation on top.
    """
    if not descriptors:
        raise ValueError("No history draft-K descriptors")
    q = spatial2d_pool(raw_q, grid_h, grid_w)
    r = spatial2d_regions_per_frame(grid_h, grid_w)
    for k in descriptors:
        if k.shape != (q.shape[0], q.shape[1], r, q.shape[-1]):
            raise ValueError("History draft-K descriptor geometry mismatch")
        if k.dtype != torch.bfloat16 or k.device != q.device:
            raise ValueError("History draft-K device/dtype mismatch")
    k = torch.cat(descriptors, dim=2)
    logits = torch.matmul(q.float(), k.float().transpose(-2, -1))
    logits.mul_(q.shape[-1] ** (-0.5))
    probs = logits.softmax(-1).to(torch.float16)
    frame_mass = probs.reshape(q.shape[0], q.shape[1], q.shape[2],
                               len(descriptors), r).float().sum(-1)
    score = frame_mass.mean(dim=(1, 2))
    return (score, probs) if return_block_probs else score

def _resolve_grid(expected_tokens=None):
    import os
    # Explicit native-adapter override. Never infer H/W from frame-token product:
    # 29x52 is not equivalent to 52x29 spatial pools.
    value=os.getenv('DRAFTMAP_SPATIAL_GRID')
    if not value:
        raise RuntimeError('NATIVE_SPATIAL_GRID_REQUIRED: supply native HxW from backbone adapter')
    try:
        h,w=(int(x) for x in value.lower().split('x'))
    except Exception as e:
        raise RuntimeError('NATIVE_SPATIAL_GRID_INVALID: expected HxW') from e
    if h<1 or w<1 or (expected_tokens is not None and h*w!=expected_tokens):
        raise RuntimeError('NATIVE_SPATIAL_GRID_TOKEN_MISMATCH')
    return h,w

def source_pool(x):
    h,w=_resolve_grid()
    return spatial2d_pool(x,h,w)

def frame_mass(query, pooled_frames, pool_fn=None):
    if not pooled_frames:raise ValueError('No committed Draft-K')
    pool_fn=pool_fn or source_pool
    q=pool_fn(query)
    b,h,nq,d=q.shape
    n=pooled_frames[0].shape[2]
    if any(z.shape!=(b,h,n,d) or z.dtype!=torch.bfloat16 or z.device!=q.device for z in pooled_frames):
        raise ValueError('Noncanonical pooled K geometry/head/dtype')
    logits=torch.matmul(q.float(),torch.cat(pooled_frames,dim=2).float().transpose(-2,-1))
    logits.mul_(d**(-0.5))
    prob=logits.softmax(-1).to(torch.float16)
    # Explicit FP32 sum for winner arithmetic (not native FP16 reduction).
    return prob.reshape(b,h,nq,len(pooled_frames),n).float().sum(-1)
