"""Protected full frames + direct spatial-block retrieval at an exact token budget.

Keep the historical P3/Top5 experiment's scorer and softmax domain. Reduce
over query regions and heads, NEVER over a frame's key regions. The last
ranked block alone may be truncated to hit the remaining frame-equivalent token budget.
"""
from dataclasses import dataclass
import math


def block_probabilities(query, descriptors, h, w, *, config):
    """The unchanged DA2 probability math, before any frame-mass summation."""
    import torch
    from .draftattention2_importance import spatial8x16_mean_max, regions_per_frame, _validate_descriptors
    qm, qx = spatial8x16_mean_max(query, h, w, capacity=config.capacity)
    regions = regions_per_frame(h, w, config.capacity)
    desc = _validate_descriptors(torch, descriptors, qm, regions)
    km = torch.cat([item[0] for item in desc], dim=2)
    kx = torch.cat([item[1] for item in desc], dim=2)
    scale = qm.shape[-1] ** -0.5
    logits = (torch.matmul(qm.float(), km.float().transpose(-2, -1)) * scale).to(torch.float16)
    prob = logits.float().softmax(-1)
    weight = float(config.maxpool_weight)
    if weight:
        logits = (torch.matmul(qx.float(), kx.float().transpose(-2, -1)) * scale).to(torch.float16)
        prob = (1.0 - weight) * prob + weight * logits.float().softmax(-1)
    return prob.to(torch.float16)


@dataclass(frozen=True)
class BlockDecision:
    protected: tuple
    remote_ids: tuple
    token_indices: tuple  # (absolute frame ID, sorted real within-frame tokens)
    ranked_blocks: tuple # (frame, spatial block, importance, available, taken)
    block_scores: tuple
    history_budget: int
    remote_token_budget: int
    partial_boundary_block: bool


def select_blocks(scores, candidate_ids, budget, *, query_start, grid_h, grid_w, protect_recent=3):
    """Stable score-descending whole-block prefix + at most one boundary crop."""
    from .draftattention2_importance import spatial_blocks
    ids = tuple(candidate_ids)
    blocks = spatial_blocks(grid_h, grid_w)
    if (not ids or tuple(sorted(set(ids))) != ids
            or any(type(f) is not int or f < 0 or f >= query_start for f in ids)):
        raise ValueError('BLOCK_TOPK_CAUSAL_UNIQUE_HISTORY')
    if type(budget) is not int or not 1 <= budget <= len(ids):
        raise ValueError('BLOCK_TOPK_HISTORY_BUDGET')
    values = tuple(float(v) for v in scores)
    if len(values) != len(ids) * len(blocks) or any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError('BLOCK_TOPK_SCORE_SHAPE_OR_VALUE')
    if type(protect_recent) is not int or protect_recent < 0:
        raise ValueError("BLOCK_TOPK_INVALID_RECENT_PROTECTION")
    protected_count = min(protect_recent, budget)
    protected = ids[-protected_count:] if protected_count else ()
    pset = set(protected)
    count = grid_h * grid_w
    target = (budget - len(protected)) * count
    eligible = [(i, fid, b) for i, fid in enumerate(ids) if fid not in pset for b in range(len(blocks))]
    eligible.sort(key=lambda row: (-values[row[0]*len(blocks)+row[2]], row[1], row[2]))
    remaining, picked, rows = target, {}, []
    for i, fid, b in eligible:
        if not remaining:
            break
        n = min(remaining, len(blocks[b]))
        picked.setdefault(fid, []).extend(blocks[b][:n])
        rows.append((fid, b, values[i*len(blocks)+b], len(blocks[b]), n))
        remaining -= n
    if remaining:
        raise ValueError('BLOCK_TOPK_INSUFFICIENT_REAL_TOKENS')
    remote = tuple(sorted(picked))
    tokens = {fid: tuple(sorted(v)) for fid, v in picked.items()}
    tokens.update({fid: tuple(range(count)) for fid in protected})
    ordered = tuple((fid, tokens[fid]) for fid in sorted(tokens))
    flat = tuple(fid*count+t for fid, ts in ordered for t in ts)
    if len(flat) != budget*count or len(flat) != len(set(flat)) or tuple(sorted(flat)) != flat:
        raise ValueError('BLOCK_TOPK_INVALID_EXACT_TOKEN_PACKING')
    return BlockDecision(protected, remote, ordered, tuple(rows), values, budget, target,
                         any(available != taken for _, _, _, available, taken in rows))
