"""Frozen fast V1 mathematics with explicit causal IDs and typed telemetry.

The authoritative source is fixtures/frozen_fast.py (SHA in provenance.json).
Only input validation and output observability are added to its selection rule.
Pooling, distributed aggregation, KV ownership, and Clean Commit stay with the
backbone. In particular, this module cannot establish that an ID was committed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Integral, Real
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    import torch

SelectedAbsoluteFrameIds = tuple[int, ...]


@dataclass(frozen=True)
class SelectionContext:
    """Optional telemetry labels; never used to reuse a previous decision."""

    layer: int | None = None
    step: int | None = None
    head_shard: int | None = None
    chunk: int | None = None

    def __post_init__(self) -> None:
        for value in (self.layer, self.step, self.head_shard, self.chunk):
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError("HIER_V1_CONTEXT: labels must be nonnegative integers")


@dataclass(frozen=True)
class ScoreStatistics:
    mass_sum: float
    mass_min: float
    mass_max: float
    log_relative_median: float
    log_relative_mad: float
    robust_scale: float
    robust_z_min: float
    robust_z_max: float


@dataclass(frozen=True)
class SelectionTrace:
    schema_version: int
    policy: str
    context: SelectionContext
    query_start: int
    candidate_ids: SelectedAbsoluteFrameIds
    candidate_count: int
    frame_budget: int
    selected_absolute_frame_ids: SelectedAbsoluteFrameIds
    chronological: SelectedAbsoluteFrameIds
    local: SelectedAbsoluteFrameIds
    salient: SelectedAbsoluteFrameIds
    protected: SelectedAbsoluteFrameIds
    eligible_salient_count: int
    salient_capacity: int
    threshold_z: float
    relative_floor: float
    reason: str
    statistics: ScoreStatistics

    def to_dict(self) -> dict:
        """Fresh JSON-compatible values; no tensors, timings, or success claims."""
        result = asdict(self)
        for name in ("candidate_ids", "selected_absolute_frame_ids", "chronological",
                     "local", "salient", "protected"):
            result[name] = list(result[name])
        return result


@dataclass(frozen=True)
class Decision:
    """Both orderings are absolute IDs; ranked is the original B01 handoff."""

    ranked: SelectedAbsoluteFrameIds
    chronological: SelectedAbsoluteFrameIds
    local: SelectedAbsoluteFrameIds
    salient: SelectedAbsoluteFrameIds
    protected: SelectedAbsoluteFrameIds
    budget: int
    candidate_count: int
    threshold_z: float
    trace: SelectionTrace
    # Exact host scalars already materialized for the frozen selector. Reusing
    # them avoids a second GPU-to-CPU transfer only for telemetry.
    frame_scores: tuple[float, ...] = ()

    @property
    def selected_absolute_frame_ids(self) -> SelectedAbsoluteFrameIds:
        return self.ranked


def _ids(values: Sequence[int], label: str) -> tuple[int, ...]:
    ids = tuple(values)
    if any(isinstance(x, bool) or not isinstance(x, Integral) or x < 0 for x in ids):
        raise ValueError(f"HIER_V1_{label}: expected nonnegative integer absolute IDs")
    ids = tuple(int(x) for x in ids)
    if ids != tuple(sorted(ids)) or len(set(ids)) != len(ids):
        raise ValueError(f"HIER_V1_{label}: IDs must be unique and chronological")
    return ids


def _lower_median(values: Sequence[float]) -> float:
    return sorted(values)[(len(values) - 1) // 2]


def select_v1(
    scores: torch.Tensor,
    candidate_ids: Sequence[int],
    budget: int,
    *,
    query_start: int,
    sink_ids: Sequence[int] = (),
    protect_recent: int | None = None,
    salient_z: float = 3.0,
    relative_floor: float = 1.5,
    context: SelectionContext | None = None,
) -> Decision:
    """Select exactly R committed historical frames from [C] or [1,C] mass.

    `query_start` is the first *absolute frame ID* of the current chunk, not a
    token offset. The owner supplies all eligible clean-committed candidate IDs
    in score-column order and excludes its native sink. R=0 and C<R are errors;
    the owner bypasses retrieval or explicitly determines its warmup R first.
    Valid-domain decisions exactly follow frozen fast V1, including host scalar
    arithmetic after one FP32 vector transfer. See docs/hierarchical_v1.md.
    """
    import torch

    ids = _ids(candidate_ids, "CHRONOLOGY")
    sinks = _ids(sink_ids, "SINK_IDS")
    if type(query_start) is not int or query_start < 0:
        raise ValueError("HIER_V1_QUERY_START")
    if any(fid >= query_start for fid in (*ids, *sinks)):
        raise ValueError("HIER_V1_NONCAUSAL: current/future IDs are forbidden")
    if set(ids).intersection(sinks):
        raise ValueError("HIER_V1_SINK_OVERLAP")
    if not ids:
        raise ValueError("HIER_V1_CHRONOLOGY: no candidates")
    if type(budget) is not int or not 1 <= budget <= len(ids):
        raise ValueError("HIER_V1_BUDGET: require 1 <= R <= candidate count")
    floor = math.ceil(budget / 2) if protect_recent is None else protect_recent
    if type(floor) is not int or not 1 <= floor <= budget:
        raise ValueError("HIER_V1_PROTECT_COUNT")
    for name, value, allow_zero in (("SALIENT_Z", salient_z, True),
                                    ("RELATIVE_FLOOR", relative_floor, False)):
        if (isinstance(value, bool) or not isinstance(value, Real)
                or not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero)):
            raise ValueError(f"HIER_V1_{name}")
    if context is not None and not isinstance(context, SelectionContext):
        raise ValueError("HIER_V1_CONTEXT")
    if not isinstance(scores, torch.Tensor) or scores.is_complex():
        raise ValueError("HIER_V1_SCORE_TYPE")
    v = scores.detach().float()
    if v.ndim == 2 and v.shape[0] == 1:
        v = v[0]
    if v.ndim != 1 or v.numel() != len(ids):
        raise ValueError("HIER_V1_SCORE_SHAPE: only [C] or batch-one [1,C]")
    # One candidate vector transfer, exactly as frozen fast V1.
    values = v.contiguous().cpu().tolist()
    if any(not math.isfinite(x) or x < 0 for x in values):
        raise ValueError("HIER_V1_NEGATIVE_NONFINITE")
    summation = sum(values)
    if summation <= 0:
        raise ValueError("HIER_V1_ZERO_MASS")
    n = len(ids)
    logs = [math.log(max(1.e-12, n * x / summation)) for x in values]
    med = _lower_median(logs)
    mad = _lower_median([abs(x - med) for x in logs])
    scale = 1.4826 * mad + 1.e-6
    robust_z = [(x - med) / scale for x in logs]
    protected = ids[-floor:]
    protected_set = set(protected)
    rank = sorted(range(n), key=lambda i: (-values[i], ids[i]))
    eligible = tuple(ids[i] for i in rank if ids[i] not in protected_set
                     and n * values[i] / summation >= relative_floor
                     and robust_z[i] >= salient_z)
    salient = eligible[:budget - floor]
    salient_set = set(salient)
    local = tuple(fid for fid in reversed(ids) if fid not in salient_set)[:budget - len(salient)]
    selected = set(local) | salient_set
    if len(selected) != budget or not protected_set.issubset(selected):
        raise RuntimeError("HIER_V1_BUDGET_VIOLATION")
    ranked = tuple(ids[i] for i in rank if ids[i] in selected)
    chronological = tuple(sorted(selected))
    local, salient = tuple(sorted(local)), tuple(sorted(salient))
    if n == budget:
        reason = "all_candidates_fit"
    elif salient:
        reason = "local_and_salient"
    elif budget == floor:
        reason = "all_local_protected_budget"
    else:
        reason = "all_local_no_salient"
    stats = ScoreStatistics(summation, min(values), max(values), med, mad, scale,
                            min(robust_z), max(robust_z))
    trace = SelectionTrace(
        1, "draftkv_hierarchical_v1_fast", context or SelectionContext(), query_start,
        ids, n, budget, ranked, chronological, local, salient, protected, len(eligible),
        budget - floor, float(salient_z), float(relative_floor), reason, stats,
    )
    return Decision(ranked, chronological, local, salient, protected, budget, n,
                    float(salient_z), trace, tuple(values))


def reduce_frame_mass(frame_mass: torch.Tensor) -> torch.Tensor:
    """Existing [1,local_heads,query_regions,C] FP32 mass -> [1,C].

    This is the owners' mean(dim=(1,2)). SP head-shard all_gather, if needed,
    must happen before this reduction, exactly as in the original owner.
    """
    import torch

    if (not isinstance(frame_mass, torch.Tensor) or frame_mass.ndim != 4
            or frame_mass.shape[0] != 1 or min(frame_mass.shape) <= 0
            or frame_mass.dtype != torch.float32):
        raise ValueError("HIER_V1_FRAME_MASS_SHAPE_DTYPE: require FP32 [1,H,Q,C]")
    return frame_mass.detach().mean(dim=(1, 2))


def draft_frame_mass(pooled_query: torch.Tensor, pooled_keys: torch.Tensor) -> torch.Tensor:
    """Numeric reference from already pooled real BF16 Draft Q/K descriptors.

    Q=[1,H,Qregions,D], K=[1,H,C,Kregions,D]. FP32 logits and softmax,
    FP16 block probabilities, then FP32 K-region sum. No pooling, RoPE,
    temporal remapping, padding, or distributed communication is performed.
    Backbone owners should normally retain their own established score path.
    """
    import torch

    if (not isinstance(pooled_query, torch.Tensor) or not isinstance(pooled_keys, torch.Tensor)
            or pooled_query.ndim != 4 or pooled_keys.ndim != 5
            or pooled_query.shape[0] != 1 or pooled_keys.shape[0] != 1
            or min(pooled_query.shape) <= 0 or min(pooled_keys.shape) <= 0
            or pooled_query.shape[:2] != pooled_keys.shape[:2]
            or pooled_query.shape[-1] != pooled_keys.shape[-1]
            or pooled_query.dtype != torch.bfloat16 or pooled_keys.dtype != torch.bfloat16
            or pooled_query.device != pooled_keys.device):
        raise ValueError("HIER_V1_DRAFT_GEOMETRY: require matching batch-one BF16 Q/K")
    q, k = pooled_query.detach(), pooled_keys.detach()
    b, h, nq, d = q.shape
    c, regions = k.shape[2:4]
    logits = torch.matmul(q.float(), k.reshape(b, h, c * regions, d).float().transpose(-2, -1))
    logits.mul_(d ** (-0.5))
    probs = logits.softmax(-1).to(torch.float16)
    return probs.reshape(b, h, nq, c, regions).float().sum(-1)
