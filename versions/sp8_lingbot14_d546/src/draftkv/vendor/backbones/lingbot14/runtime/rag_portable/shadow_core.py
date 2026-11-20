import ast
from functools import lru_cache
import math
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn


@lru_cache(maxsize=4)
def reference_method(path, name):
    tree = ast.parse(Path(path).read_text())
    cache_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'ShadowKVCache')
    method = next(node for node in cache_class.body if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = dict(torch=torch, nn=nn, math=math)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


@torch.no_grad()
def factorize(raw_keys, rank, reference_path):
    if raw_keys.ndim != 4 or raw_keys.shape[0] != 1:
        raise ValueError('Expected batch-one [B,S,H,D] keys')
    count, heads, width = raw_keys.shape[1:]
    if not 0 < rank <= min(count, heads * width):
        raise ValueError('Rank exceeds matrix dimensions')
    context = SimpleNamespace(batch_size=1, num_key_value_heads=heads, head_dim=width,
                              num_layers=1, rank=rank, device=raw_keys.device, dtype=raw_keys.dtype)
    reference_method(reference_path, 'get_svd')(context, raw_keys.reshape(1, count, heads * width), 0)
    basis = context.SV[0]
    flattened = basis.transpose(1, 2).reshape(rank, heads * width).float()
    projection = torch.linalg.pinv(flattened).contiguous()
    return context.U[0], basis, projection


@torch.no_grad()
def landmarks(geometry_keys, block_size):
    if geometry_keys.ndim != 4 or geometry_keys.shape[0] != 1:
        raise ValueError('Expected [1,H,S,D] transformed keys')
    heads, tokens, width = geometry_keys.shape[1:]
    if tokens % block_size:
        raise ValueError('Block size must divide the clean chunk')
    blocked = geometry_keys.reshape(1, heads, tokens // block_size, block_size, width)
    centers = blocked.mean(dim=-2)
    similarity = nn.functional.cosine_similarity(centers.unsqueeze(-2).expand_as(blocked), blocked, dim=-1)
    return centers, similarity.min(dim=-1).values


@torch.no_grad()
def select_landmarks(query, centers, global_block_ids, count, query_tile=64):
    if not 0 <= count <= centers.shape[-2]:
        raise ValueError('Invalid sparse budget')
    if not count:
        return global_block_ids[..., :0]
    if count == centers.shape[-2]:
        return global_block_ids
    score = torch.zeros(centers.shape[:-1], dtype=torch.float32, device=query.device)
    for start in range(0, query.shape[-2], query_tile):
        logits = torch.einsum('bhqd,bhcd->bhqc', query[:, :, start:start + query_tile], centers) / math.sqrt(128)
        probabilities = nn.functional.softmax(logits, dim=-1, dtype=torch.float32).to(query.dtype)
        score.add_(probabilities.float().sum(dim=-2))
    selected = score.topk(count, dim=-1).indices
    return global_block_ids.gather(-1, selected)


def expand_blocks(block_ids, block_size):
    offsets = torch.arange(block_size, device=block_ids.device)
    return (block_ids.unsqueeze(-1) * block_size + offsets).flatten(-2)


def reuse_indices(previous_ids, requested_ids):
    if previous_ids is None or previous_ids.shape[-1] == 0:
        return torch.zeros_like(requested_ids), torch.zeros_like(requested_ids, dtype=torch.bool)
    ordered, order = previous_ids.sort(dim=-1)
    locations = torch.searchsorted(ordered.contiguous(), requested_ids.contiguous())
    clamped = locations.clamp(max=ordered.shape[-1] - 1)
    hits = (locations < ordered.shape[-1]) & (ordered.gather(-1, clamped) == requested_ids)
    return order.gather(-1, clamped), hits
