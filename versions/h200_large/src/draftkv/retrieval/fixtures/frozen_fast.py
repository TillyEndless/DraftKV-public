"""DraftKV hierarchical V1 selector optimized for real model forward.

Transfer candidate-level masses ONCE to CPU; all scalar reductions, MAD,
stable ranking and Local/Salient budget filling are host-only.
No extra device synchronizations per candidate after the one transfer.
"""
from dataclasses import dataclass
import math,os

@dataclass(frozen=True)
class Decision:
    ranked:tuple
    chronological:tuple
    local:tuple
    salient:tuple
    protected:tuple
    budget:int
    candidate_count:int
    threshold_z:float

def _lower_median(xs):
    return sorted(xs)[(len(xs)-1)//2]

def select_v1(scores,candidate_ids,budget,*,protect_recent=None,
              salient_z=3.,relative_floor=1.5):
    import torch
    ids=tuple(int(x) for x in candidate_ids)
    if not ids or tuple(sorted(ids))!=ids or len(set(ids))!=len(ids) or ids[0]<0:
        raise ValueError("HIER_V1_CHRONOLOGY")
    if not isinstance(budget,int) or not(1<=budget<=len(ids)):
        raise ValueError("HIER_V1_BUDGET")
    v=scores.detach().float()
    if v.ndim==2 and v.shape[0]==1:v=v[0]
    if v.ndim!=1 or v.numel()!=len(ids):
        raise ValueError("HIER_V1_SCORE_SHAPE")
    # Single [C] D2H synchronization, not one synchronization per score!
    values=v.contiguous().cpu().tolist()
    if any(not math.isfinite(x) or x<0 for x in values):raise ValueError("HIER_V1_NEGATIVE_NONFINITE")
    summation=sum(values)
    if summation<=0:raise ValueError("HIER_V1_ZERO_MASS")
    floor=math.ceil(budget/2) if protect_recent is None else int(protect_recent)
    if not 1<=floor<=budget:raise ValueError("HIER_V1_PROTECT_COUNT")
    protected=ids[-floor:]
    n=len(ids)
    logs=[math.log(max(1.e-12,n*x/summation)) for x in values]
    med=_lower_median(logs)
    mad=_lower_median([abs(x-med) for x in logs])
    scale=1.4826*mad+1.e-6
    protected_set=set(protected)
    rank=sorted(range(n),key=lambda i:(-values[i],ids[i]))
    salient=tuple(ids[i] for i in rank if
                  ids[i] not in protected_set and
                  n*values[i]/summation>=relative_floor and
                  (logs[i]-med)/scale>=salient_z)[:budget-floor]
    salset=set(salient)
    local=tuple(fid for fid in reversed(ids) if fid not in salset)[:budget-len(salient)]
    selected=set(local)|salset
    if len(selected)!=budget or not protected_set.issubset(selected):
        raise RuntimeError("HIER_V1_BUDGET_VIOLATION")
    return Decision(tuple(ids[i] for i in rank if ids[i] in selected),
                    tuple(sorted(selected)),tuple(sorted(local)),tuple(sorted(salient)),
                    tuple(protected),budget,n,salient_z)

def enabled():
    v=os.environ.get("DRAFTKV_HIERARCHICAL_V1","1")
    if v not in ("0","1"):raise ValueError("DRAFTKV_HIERARCHICAL_V1_EXPECTS_0_OR_1")
    return v=="1"
