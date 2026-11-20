"""Explicit video budgets, independent of a backbone's native cache layout.

Condition/text/audio prefixes are outside the video budget. These declarations
do not change a selector: they describe its mandatory and retrieved domains.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .contracts import BackboneError

DEFAULT_BINDING = (
    Path(__file__).resolve().parents[3] / "configs/backbones/effective_budgets.json"
)


@dataclass(frozen=True)
class EffectiveBudget:
    capacity: int
    retrieved: int
    sink: int
    local: int
    current: int
    block_size: int | None = None
    keep_ratio: float | None = None
    retrieved_equivalent: bool = False
    retrieval_unit: str = "frame"
    dynamic_retrieval: bool = False
    retrieved_source_frames: int | None = None
    retrieved_equivalent_frames: int | None = None
    compressed_equivalent_per_chunk: float | None = None
    policy_name: str | None = None
    landmark_slots: int = 0

    def __post_init__(self):
        for name in ("capacity", "retrieved", "sink", "local", "current", "landmark_slots"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise BackboneError("BLOCKED_CONFIGURATION", f"invalid budget {name}")
        if (
            self.current < 1
            or self.capacity < 1
            or self.retrieved + self.sink + self.local + self.current + self.landmark_slots > self.capacity
        ):
            raise BackboneError(
                "KV_CONSTRAINT", "effective video budget exceeds capacity"
            )
        if self.block_size is not None and (
            type(self.block_size) is not int or self.block_size < 1
        ):
            raise BackboneError("BLOCKED_CONFIGURATION", "invalid block_size")
        if self.keep_ratio is not None and (
            type(self.keep_ratio) not in (float, int) or not 0 < self.keep_ratio <= 1
        ):
            raise BackboneError("BLOCKED_CONFIGURATION", "invalid keep_ratio")
        if type(self.retrieved_equivalent) is not bool:
            raise BackboneError(
                "BLOCKED_CONFIGURATION", "retrieved_equivalent must be boolean"
            )
        if (
            self.retrieval_unit not in {"frame", "block", "page", "equivalent_frame", "landmark"}
            or type(self.dynamic_retrieval) is not bool
        ):
            raise BackboneError(
                "BLOCKED_CONFIGURATION", "invalid retrieval unit/dynamic policy"
            )
        if self.policy_name is not None and (
            not isinstance(self.policy_name, str) or not self.policy_name
        ):
            raise BackboneError("BLOCKED_CONFIGURATION", "invalid budget policy name")
        if self.retrieved_source_frames is not None:
            source = self.retrieved_source_frames
            equivalent = self.compressed_equivalent_per_chunk
            if (
                type(source) is not int
                or source <= 0
                or source % self.current
                or not self.retrieved_equivalent
                or self.retrieval_unit != "equivalent_frame"
                or self.keep_ratio is None
                or equivalent != 1 + (self.current - 1) * self.keep_ratio
                or self.retrieved_equivalent_frames != self.retrieved
                or source // self.current * equivalent != self.retrieved
            ):
                raise BackboneError(
                    "KV_CONSTRAINT",
                    "raw source chunks and compressed equivalent budget differ",
                )
        elif (
            self.retrieved_equivalent_frames is not None
            or self.compressed_equivalent_per_chunk is not None
        ):
            raise BackboneError(
                "KV_CONSTRAINT", "compressed budget requires explicit source frames"
            )

    def to_dict(self):
        result = asdict(self)
        if not self.landmark_slots:
            del result["landmark_slots"]
        if self.policy_name is None:
            # Keep Native/DraftKV's pre-alignment resolved representation byte compatible.
            for key in (
                "retrieval_unit",
                "dynamic_retrieval",
                "retrieved_source_frames",
                "retrieved_equivalent_frames",
                "compressed_equivalent_per_chunk",
                "policy_name",
            ):
                del result[key]
        return result

    def tokens(self, frame_tokens):
        if type(frame_tokens) is not int or frame_tokens < 1:
            raise BackboneError("KV_CONSTRAINT", "positive frame_tokens required")
        result = {
            name: getattr(self, name) * frame_tokens
            for name in ("capacity", "retrieved", "sink", "local", "current")
        }
        if self.landmark_slots:
            result["landmark_slots"] = self.landmark_slots * frame_tokens
        if self.block_size and any(
            value % self.block_size for value in result.values()
        ):
            raise BackboneError(
                "KV_CONSTRAINT", "effective budget must align to whole blocks"
            )
        return result


def _binding(binding=DEFAULT_BINDING):
    data = json.loads(Path(binding).read_text())
    if data.get("schema_version") != 1:
        raise BackboneError(
            "BLOCKED_CONFIGURATION", "unsupported effective-budget schema"
        )
    return data


def author_budget_policy(spec, algorithm, *, binding=DEFAULT_BINDING, preset=None):
    """Author-defined allocation semantics, separate from rounded frame views."""
    if algorithm in ("deep-forcing", "future-forcing"):
        from draftkv.algorithms.forcing.budget import policy_budget
        return policy_budget(spec.window, spec.chunk, algorithm, preset=preset or "author_primary")
    data = _binding(binding)
    policy = data.get("author_policies", {}).get(algorithm)
    result = dict(policy) if policy is not None else {}
    if algorithm == "worldtrace-landmark":
        result["preset"] = preset or "table2-budget-matched"
    if algorithm == "worldkv" and getattr(spec, "backbone", None) in data["backbones"]:
        budget = data["backbones"][spec.backbone][algorithm]
        result["resolved_allocation"] = {
            "retrieved_source_frames": budget["retrieved_source_frames"],
            "retrieved_equivalent_frames": budget["retrieved_equivalent_frames"],
            "sink_frames": budget["sink"],
            "local_frames": budget["local"],
            "current_frames": budget["current"],
            "keep_ratio": budget["keep_ratio"],
            "compressed_equivalent_per_chunk": budget[
                "compressed_equivalent_per_chunk"
            ],
            "capacity_equivalent_frames": budget["capacity"],
        }
    return result


def effective_budget(spec, algorithm, *, binding=DEFAULT_BINDING, preset=None):
    if algorithm == 'draftkv' and binding == DEFAULT_BINDING and preset is None:
        from draftkv.algorithms.draftkv.experiment import active_override
        override = active_override(getattr(spec, 'backbone', None))
        if override is not None:
            b = override['budget']
            capacity = (sum(b.values()) if override.get('retrieval_ablation') == 'camera-only'
                        else spec.window)
            return EffectiveBudget(capacity, b['retrieved'], b['sink'], b['local'], b['current'])
    if algorithm != "worldtrace-landmark" and preset not in (None, "author_primary"):
        if algorithm != "future-forcing":
            raise BackboneError("BLOCKED_CONFIGURATION", "named capacity presets require Future Forcing")
        p = author_budget_policy(spec, algorithm, preset=preset)
        return EffectiveBudget(p["attention_cap_frames"], p["selected_frames"], 0, 0,
                               spec.chunk, retrieval_unit="equivalent_frame",
                               policy_name="future_forcing_" + preset)
    data = _binding(binding)
    explicit = data["backbones"].get(getattr(spec, "backbone", None))
    if explicit is not None:
        if algorithm not in explicit:
            raise BackboneError(
                "BLOCKED_CONFIGURATION", "missing per-method effective budget"
            )
        budget = EffectiveBudget(**explicit[algorithm])
        if algorithm == "worldtrace-landmark":
            preset = preset or "table2-budget-matched"
            if preset == "author-default":
                from dataclasses import replace
                budget = replace(budget, landmark_slots=4, local=2,
                                 capacity=budget.sink + budget.current + 6)
            elif preset != "table2-budget-matched":
                raise BackboneError("BLOCKED_CONFIGURATION", "unknown WorldTrace budget preset")
            if budget.capacity > spec.window or budget.current != spec.chunk:
                raise BackboneError("KV_CONSTRAINT", "WorldTrace exceeds host native window")
            return budget
        if algorithm in ("deep-forcing", "future-forcing"):
            from draftkv.algorithms.forcing.budget import policy_budget
            p = policy_budget(spec.window, spec.chunk, algorithm)
            expected = (p["attention_cap_frames"], p["selected_frames"], p["sink_frames"], p["recent_history_frames"], spec.chunk)
            if (budget.capacity, budget.retrieved, budget.sink, budget.local, budget.current) != expected:
                raise BackboneError("KV_CONSTRAINT", "forcing budget differs from deterministic author-derived mapping")
            return budget
        if (budget.capacity, budget.current) != (spec.window, spec.chunk):
            raise BackboneError(
                "KV_CONSTRAINT", "budget differs from native window/chunk"
            )
        return budget
    if getattr(spec, "backbone", None) is not None:
        raise BackboneError("BLOCKED_CONFIGURATION", "unbound backbone budget")
    # Small anonymous synthetic fixtures only. Every released ModelSpec above
    # requires an explicit backbone × algorithm declaration; no formal fallback.
    if algorithm == "native":
        return EffectiveBudget(
            spec.window, 0, spec.sink, spec.window - spec.sink - spec.chunk, spec.chunk
        )
    if algorithm not in (
        "draftkv",
        "shadowkv",
        "infinigen",
        "infllm",
        "freekv",
        "worldkv",
    ):
        raise BackboneError("BLOCKED_CONFIGURATION", "unknown algorithm budget")
    local = 0 if algorithm == "draftkv" else spec.chunk
    return EffectiveBudget(
        spec.window,
        spec.window - spec.sink - spec.chunk - local,
        spec.sink,
        local,
        spec.chunk,
    )


def bound_budget(core, algorithm):
    budget = effective_budget(core.spec, algorithm)
    policy = author_budget_policy(core.spec, algorithm)
    settings = (core.plan.get("algorithm_plugin") or {}).get("config", {})
    if (
        algorithm == "infinigen"
        and budget.policy_name is not None
        and any(
            settings.get(key, policy[key]) != policy[key]
            for key in ("partial_ratio", "alpha")
        )
    ):
        raise BackboneError(
            "BLOCKED_CONFIGURATION",
            "InfiniGen partial ratio/alpha differ from release policy",
        )
    if (
        budget.block_size is not None
        and settings.get("block_size", 128) != budget.block_size
    ):
        raise BackboneError(
            "BLOCKED_CONFIGURATION",
            "algorithm block size differs from effective budget",
        )
    if (
        budget.keep_ratio is not None
        and settings.get("keep_ratio", 0.5) != budget.keep_ratio
    ):
        raise BackboneError(
            "BLOCKED_CONFIGURATION", "compression ratio differs from effective budget"
        )
    supplied = core.plan.get("effective_budget")
    if supplied is not None and supplied != budget.to_dict():
        raise BackboneError(
            "BLOCKED_CONFIGURATION", "resolved budget differs from release binding"
        )
    supplied_policy = core.plan.get("author_budget_policy")
    if supplied_policy is not None and supplied_policy != policy:
        raise BackboneError(
            "BLOCKED_CONFIGURATION",
            "resolved author budget policy differs from release binding",
        )
    return budget
