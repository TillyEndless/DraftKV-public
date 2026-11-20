import ast
from functools import lru_cache
import math
from pathlib import Path

import torch
import torch.nn.functional as F


@lru_cache(maxsize=4)
def author_functions(path):
    tree = ast.parse(Path(path).read_text())
    accepted = {'get_corr_torch_compile', 'quest_sel', 'repeat_kv_BLH'}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in accepted]
    for node in nodes:
        node.decorator_list = []
    namespace = dict(torch=torch, F=F, math=math)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


@torch.no_grad()
def correction_mask(query, previous, threshold, reference):
    if previous.shape != query.shape:
        raise ValueError('Speculation across different query positions is invalid')
    batch, heads, positions, width = query.shape
    current_flat = query.permute(0, 2, 1, 3).reshape(batch * positions, 1, heads, width)
    previous_flat = previous.permute(0, 2, 1, 3).reshape(batch * positions, 1, heads, width)
    correct, _ = author_functions(reference)['get_corr_torch_compile'](
        current_flat, previous_flat, heads, threshold)
    return correct.reshape(batch, positions, heads).any(dim=1)


@torch.no_grad()
def page_scores(query, minimum, maximum, query_tile=64):
    if minimum.shape != maximum.shape or query.shape[:2] != minimum.shape[:2]:
        raise ValueError('MHA summary and query head mismatch')
    scores = torch.zeros(minimum.shape[:-1], device=query.device, dtype=torch.float32)
    for start in range(0, query.shape[-2], query_tile):
        tile = query[:, :, start:start + query_tile]
        logits = torch.einsum('bhqd,bhpd->bhqp', tile.clamp_min(0), maximum)
        logits += torch.einsum('bhqd,bhpd->bhqp', tile.clamp_max(0), minimum)
        scores += F.softmax(logits.float(), dim=-1).sum(dim=-2)
    return scores
