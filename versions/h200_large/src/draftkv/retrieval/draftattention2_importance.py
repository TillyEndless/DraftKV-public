"""DraftAttention2-derived frame importance for DraftKV.

Only importance estimation is reused:
post-QKNorm/post-RoPE Q/K -> Spatial2D(kernel=8x16, stride=8x16, ceil edges) -> Avg/Max draft
probabilities -> mean-dominant fusion -> historical frame probability mass.

No Q/K/V quantization and no mixed-precision attention are implemented here.
Full-resolution historical K/V remain exact BF16 pages consumed by the existing
B01 Streaming FA2 path.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import math
from pathlib import Path
from typing import Iterable

SCORER_NAME = "draftattention2_postrope_spatial8x16_meanmax"
SCHEMA_VERSION = 1
POOL_H, POOL_W = 8, 16
DEFAULT_CAPACITY = POOL_H * POOL_W
DEFAULT_MAXPOOL_WEIGHT = 0.04


@dataclass(frozen=True)
class ImportanceConfig:
    capacity: int = DEFAULT_CAPACITY
    maxpool_weight: float = DEFAULT_MAXPOOL_WEIGHT
    pool_h: int = POOL_H
    pool_w: int = POOL_W

    def __post_init__(self) -> None:
        if (type(self.pool_h) is not int or type(self.pool_w) is not int
                or type(self.capacity) is not int or self.capacity != self.pool_h*self.pool_w
                or (self.pool_h,self.pool_w) not in ((8,16),(8,32),(16,16),(16,32),(16,64),(32,32))):
            raise ValueError("DRAFTKV_IMPORTANCE_CAPACITY")
        if (isinstance(self.maxpool_weight, bool)
                or not isinstance(self.maxpool_weight, (int, float))
                or not math.isfinite(float(self.maxpool_weight))
                or not 0.0 <= float(self.maxpool_weight) <= 1.0):
            raise ValueError("DRAFTKV_IMPORTANCE_MAXPOOL_WEIGHT")

    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION if (self.pool_h,self.pool_w)==(8,16) else 2,
            "name": SCORER_NAME if (self.pool_h,self.pool_w)==(8,16) else f"draftattention2_postrope_spatial{self.pool_h}x{self.pool_w}_meanmax",
            "qk_space": "post_qknorm_post_rope",
            "partition": "spatial2d_fixed8x16" if (self.pool_h,self.pool_w)==(8,16) else "spatial2d_configurable",
            "pool_h": self.pool_h,
            "pool_w": self.pool_w,
            "capacity": self.capacity,
            "fusion": "probability_mean_max",
            "maxpool_weight": float(self.maxpool_weight),
            "draft_descriptor_dtype": "fp16",
            "draft_logit_dtype": "fp16",
            "draft_gemm_accumulation": "fp32",
            "softmax_accumulation": "fp32",
            "mixed_precision_attention": False,
        }


def importance_config_from_dict(data: dict) -> ImportanceConfig:
    """Reject stale partitions/numeric contracts in both configs and captures."""
    if not isinstance(data, dict):
        raise ValueError("DRAFTKV_IMPORTANCE_CONFIG_SCHEMA")
    if data.get("schema_version") not in (SCHEMA_VERSION,2):
        raise ValueError("DRAFTKV_IMPORTANCE_CONFIG_SCHEMA")
    expected_name=SCORER_NAME if data.get("schema_version")==1 else f"draftattention2_postrope_spatial{data.get('pool_h')}x{data.get('pool_w')}_meanmax"
    if data.get("name") != expected_name:
        raise ValueError("DRAFTKV_IMPORTANCE_CONFIG_NAME")
    if data.get("qk_space") != "post_qknorm_post_rope":
        raise ValueError("DRAFTKV_IMPORTANCE_QK_SPACE")
    ph,pw=data.get('pool_h'),data.get('pool_w')
    if type(ph) is not int or type(pw) is not int:raise ValueError('DRAFTKV_IMPORTANCE_PARTITION')
    if data.get('schema_version')==1:
        if data.get('partition')!='spatial2d_fixed8x16' or (ph,pw)!=(8,16):
            raise ValueError('DRAFTKV_IMPORTANCE_PARTITION')
    elif data.get('partition')!='spatial2d_configurable':
        raise ValueError('DRAFTKV_IMPORTANCE_PARTITION')
    if data.get("fusion") != "probability_mean_max":
        raise ValueError("DRAFTKV_IMPORTANCE_FUSION")
    numeric = {
        "draft_descriptor_dtype": "fp16",
        "draft_logit_dtype": "fp16",
        "draft_gemm_accumulation": "fp32",
        "softmax_accumulation": "fp32",
    }
    if any(data.get(key) != value for key, value in numeric.items()):
        raise ValueError("DRAFTKV_IMPORTANCE_NUMERIC_SEMANTICS")
    if data.get("mixed_precision_attention") is not False:
        raise ValueError("DRAFTKV_IMPORTANCE_MIXED_PRECISION_FORBIDDEN")
    return ImportanceConfig(
        capacity=data.get("capacity"),
        maxpool_weight=data.get("maxpool_weight"),pool_h=ph,pool_w=pw,
    )


def load_importance_config(path: str | Path) -> ImportanceConfig:
    return importance_config_from_dict(json.loads(Path(path).read_text()))


_DEVICE_INDEX_CACHE: dict[tuple[int, int, int, str], tuple[object, object, object]] = {}


@lru_cache(maxsize=64, typed=True)
def spatial_blocks(grid_h: int, grid_w: int, pool_h: int = POOL_H, pool_w: int = POOL_W) -> tuple[tuple[int, ...], ...]:
    """Raster-ordered 8x16 rectangles, with real-token-only ceil edges."""
    if any(type(n) is not int or n <= 0 for n in (grid_h, grid_w)):
        raise ValueError("DRAFTKV_IMPORTANCE_FRAME_GEOMETRY")
    ph, pw = min(pool_h, grid_h), min(pool_w, grid_w)
    return tuple(
        tuple(y * grid_w + x for y in range(y0, min(y0 + ph, grid_h))
              for x in range(x0, min(x0 + pw, grid_w)))
        for y0 in range(0, grid_h, ph) for x0 in range(0, grid_w, pw)
    )


def _fixed_capacity(capacity, pool_h=POOL_H, pool_w=POOL_W):
    if type(capacity) is not int or capacity != pool_h*pool_w:
        raise ValueError("DRAFTKV_IMPORTANCE_CAPACITY: fixed8x16 requires 128")


def regions_per_frame(grid_h: int, grid_w: int, capacity: int = DEFAULT_CAPACITY, pool_h=POOL_H, pool_w=POOL_W) -> int:
    _fixed_capacity(capacity,pool_h,pool_w)
    return len(spatial_blocks(grid_h, grid_w,pool_h,pool_w))


def _indices(torch, grid_h: int, grid_w: int, capacity: int, device, pool_h=POOL_H, pool_w=POOL_W):
    _fixed_capacity(capacity,pool_h,pool_w)
    blocks = spatial_blocks(grid_h, grid_w,pool_h,pool_w)
    key = (grid_h, grid_w, capacity, pool_h, pool_w, str(device))
    cached = _DEVICE_INDEX_CACHE.get(key)
    if cached is not None:
        return cached
    rows, valid = [], []
    for block in blocks:
        rows.append(list(block) + [0] * (capacity - len(block)))
        valid.append([True] * len(block) + [False] * (capacity - len(block)))
    index = torch.tensor(rows, dtype=torch.long, device=device)
    mask = torch.tensor(valid, dtype=torch.bool, device=device)
    counts = mask.sum(-1).to(torch.float32)
    _DEVICE_INDEX_CACHE[key] = (index, mask, counts)
    return index, mask, counts


def spatial8x16_mean_max(x, grid_h: int, grid_w: int, *, capacity: int = DEFAULT_CAPACITY, pool_h=POOL_H, pool_w=POOL_W):
    """Pool post-RoPE [B,F*H*W,heads,d] into two [B,heads,F*R,d] maps."""
    import torch
    if not isinstance(x, torch.Tensor) or x.ndim != 4:
        raise ValueError("DRAFTKV_IMPORTANCE_QK_SHAPE")
    if x.dtype != torch.bfloat16:
        raise ValueError("DRAFTKV_IMPORTANCE_QK_DTYPE: exact native BF16 required")
    b, tokens, heads, dim = x.shape
    frame_tokens = int(grid_h) * int(grid_w)
    if frame_tokens <= 0 or tokens <= 0 or tokens % frame_tokens:
        raise ValueError("DRAFTKV_IMPORTANCE_FRAME_GEOMETRY")
    frames = tokens // frame_tokens
    index, valid, counts = _indices(torch, grid_h, grid_w, capacity, x.device,pool_h,pool_w)
    regions = index.shape[0]

    grid = x.to(torch.float16).reshape(b, frames, frame_tokens, heads, dim)
    gathered = grid[:, :, index, :, :]
    mask = valid.view(1, 1, regions, capacity, 1, 1)

    mean = (gathered.float() * mask).sum(3)
    mean = mean / counts.view(1, 1, regions, 1, 1)
    mean = mean.to(torch.float16)

    neg_inf = torch.tensor(float("-inf"), dtype=gathered.dtype, device=x.device)
    maximum = torch.where(mask, gathered, neg_inf).amax(3).to(torch.float16)

    def layout(tensor):
        return tensor.permute(0, 3, 1, 2, 4).reshape(
            b, heads, frames * regions, dim
        ).contiguous()

    return layout(mean), layout(maximum)


def descriptor(post_rope_key, grid_h: int, grid_w: int, *, capacity: int = DEFAULT_CAPACITY, pool_h=POOL_H, pool_w=POOL_W):
    """Return [2,B,heads,regions,d] = mean,max for one historical frame."""
    import torch
    frame_tokens = int(grid_h) * int(grid_w)
    if post_rope_key.shape[1] != frame_tokens:
        raise ValueError("DRAFTKV_IMPORTANCE_DESCRIPTOR_ONE_FRAME_REQUIRED")
    mean, maximum = spatial8x16_mean_max(
        post_rope_key, grid_h, grid_w, capacity=capacity,pool_h=pool_h,pool_w=pool_w
    )
    return torch.stack((mean, maximum), dim=0).contiguous()


def _validate_descriptors(torch, descriptors: Iterable, q_mean, regions: int):
    result = list(descriptors)
    if not result:
        raise ValueError("DRAFTKV_IMPORTANCE_NO_HISTORY")
    expected = (2, q_mean.shape[0], q_mean.shape[1], regions, q_mean.shape[-1])
    for item in result:
        if (not isinstance(item, torch.Tensor) or tuple(item.shape) != expected
                or item.dtype != torch.float16 or item.device != q_mean.device):
            raise ValueError("DRAFTKV_IMPORTANCE_DESCRIPTOR_GEOMETRY")
    return result


def frame_mass(post_rope_query, descriptors, grid_h: int, grid_w: int, *,
               config: ImportanceConfig | None = None,
               return_block_probs: bool = False):
    """Return FP32 [B,heads,query_regions,candidate_frames] fused draft mass."""
    import torch
    cfg = config or ImportanceConfig()
    q_mean, q_max = spatial8x16_mean_max(
        post_rope_query, grid_h, grid_w, capacity=cfg.capacity,pool_h=cfg.pool_h,pool_w=cfg.pool_w
    )
    regions = regions_per_frame(grid_h, grid_w, cfg.capacity,cfg.pool_h,cfg.pool_w)
    desc = _validate_descriptors(torch, descriptors, q_mean, regions)
    k_mean = torch.cat([item[0] for item in desc], dim=2)
    k_max = torch.cat([item[1] for item in desc], dim=2)

    scale = q_mean.shape[-1] ** -0.5
    mean_logits = (torch.matmul(q_mean.float(), k_mean.float().transpose(-2, -1)) * scale).to(torch.float16)
    mean_prob = mean_logits.float().softmax(-1)

    weight = float(cfg.maxpool_weight)
    if weight == 0.0:
        fused = mean_prob
    else:
        max_logits = (torch.matmul(q_max.float(), k_max.float().transpose(-2, -1)) * scale).to(torch.float16)
        max_prob = max_logits.float().softmax(-1)
        fused = (1.0 - weight) * mean_prob + weight * max_prob

    probs = fused.to(torch.float16)
    mass = probs.reshape(
        q_mean.shape[0], q_mean.shape[1], q_mean.shape[2], len(desc), regions
    ).float().sum(-1)
    return (mass, probs) if return_block_probs else mass


def frame_importance(post_rope_query, descriptors, grid_h: int, grid_w: int, *,
                     config: ImportanceConfig | None = None):
    mass = frame_mass(
        post_rope_query, descriptors, grid_h, grid_w, config=config
    )
    return mass.mean(dim=(1, 2))


__all__ = [
    "DEFAULT_CAPACITY", "DEFAULT_MAXPOOL_WEIGHT", "ImportanceConfig", "SCORER_NAME",
    "descriptor", "frame_importance", "frame_mass", "load_importance_config",
    "importance_config_from_dict", "spatial_blocks", "POOL_H", "POOL_W",
    "spatial8x16_mean_max", "regions_per_frame",
]
