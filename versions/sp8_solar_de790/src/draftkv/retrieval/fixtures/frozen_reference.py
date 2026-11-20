"""DraftKV V1 hierarchical retrieval on preexisting frame-level softmax mass.

No change to ProducerPlan, DMA, CUDA Frontier, FA2 Consumer or Clean Commit.
Expected input: [C] or [1,C] frame mass normalized over shared candidate history.
For original BF16 Draft-K score use the exact existing float32 score, not a
second approximate retrieval model. Output both orderings so each backbone
can preserve its original Native KV packing semantics.

NOT a full-token self-attention probability: score comes from the frozen
Draft-K descriptor frame-importance calculation.
"""
from dataclasses import dataclass
import math
import os

@dataclass(frozen=True)
class Decision:
    ranked: tuple
    chronological: tuple
    local: tuple
    salient: tuple
    protected: tuple
    budget: int
    candidate_count: int
    threshold_z: float

def select_v1(scores, candidate_ids, budget, *,
              protect_recent=None, salient_z=3.0, relative_floor=1.5):
    import torch
    ids=tuple(int(i) for i in candidate_ids)
    if not ids or any(i < 0 for i in ids) or len(ids)!=len(set(ids)) or ids!=tuple(sorted(ids)):
        raise ValueError("HIER_V1_CANDIDATES_NOT_CHRONOLOGICAL")
    if not isinstance(budget,int) or budget < 1 or budget>len(ids):
        raise ValueError("HIER_V1_BUDGET_INVALID")
    v=scores.detach().float()
    if v.ndim==2 and v.shape[0]==1: v=v[0]
    if v.ndim!=1 or v.numel()!=len(ids):
        raise ValueError("HIER_V1_SCORES_SHAPE")
    if not bool(torch.isfinite(v).all()):
        raise ValueError("HIER_V1_NONFINITE_SCORE")
    if bool((v < 0).any()):
        raise ValueError("HIER_V1_SCORE_NOT_ATTENTION_MASS")
    if float(v.sum())<=0:
        raise ValueError("HIER_V1_ZERO_MASS")
    # For an 8-frame history this protects the newest four. B6 -> L3;
    # B15 -> L8; budget stays fixed across branches.
    floor=math.ceil(budget/2) if protect_recent is None else int(protect_recent)
    if floor<1 or floor>budget:raise ValueError("HIER_V1_RECENT_FLOOR")
    protected=tuple(ids[-floor:])
    protected_set=set(protected)
    rel=(v/v.sum().clamp_min(1.e-12))*len(ids)
    logrel=rel.clamp_min(1.e-12).log()
    med=logrel.median()
    mad=(logrel-med).abs().median()
    robust_z=(logrel-med)/(1.4826*mad+1.e-6)
    eligible=torch.tensor([i not in protected_set for i in ids],device=v.device,dtype=torch.bool)
    mask=eligible & (rel>=relative_floor) & (robust_z>=salient_z)
    # Stable tie: ascending absolute frame index (given ascending candidate ids)
    sorted_scores=torch.argsort(v,descending=True,stable=True).tolist()
    salient=tuple(ids[j] for j in sorted_scores
                  if bool(mask[j]))[:budget-floor]
    salient_set=set(salient)
    local=tuple(i for i in reversed(ids) if i not in salient_set)[:budget-len(salient)]
    selected_set=set(local).union(salient_set)
    if (len(selected_set)!=budget or not protected_set.issubset(selected_set)
        or not selected_set.issubset(set(ids))):
        raise RuntimeError("HIER_V1_FIXED_BUDGET_OR_RECENT_FAILURE")
    ranked=tuple(ids[j] for j in sorted_scores if ids[j] in selected_set)
    chronological=tuple(sorted(selected_set))
    return Decision(ranked,chronological,tuple(sorted(local)),
                    tuple(sorted(salient)),protected,budget,len(ids),salient_z)

def enabled():
    value=os.environ.get("DRAFTKV_HIERARCHICAL_V1","1").lower()
    if value not in ("0","1"):raise ValueError("DRAFTKV_HIERARCHICAL_V1_EXPECTS_0_OR_1")
    return value=="1"
