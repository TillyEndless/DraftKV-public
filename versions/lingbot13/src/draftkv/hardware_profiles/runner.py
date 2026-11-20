"""CPU orchestration around an explicitly supplied original-B01 target adapter.

The adapter owns real checkpoint capture, C++ DMA, Frontier, native FA2, CUDA
events and rank coordination. There is deliberately no Torch/default provider.
All measurement returns are hash-addressed files, never synthetic estimates.
"""

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time
from typing import Protocol

from .profiles import (ProfileError, check_config, checked, digest, read_json, require,
                       request_family, verify_artifact, verify_binding, portable, seal)


class B01TargetAdapter(Protocol):
    """Trusted target implementation; return hash-addressed raw JSON artifacts.

    Every method must honor timeout_s and coordinate abort across all SP ranks.
    describe probes physical devices, loaded ABI/kernel and capture tensor shapes.
    validate reuses real QKV, changing only the destination NaN poison seed.
    measure includes native DMA, resident pack, FA2, split merge and final join.
    """

    def describe(self, *, timeout_s: float) -> dict: ...

    def support(self, config: dict, *, timeout_s: float) -> bool: ...

    def validate(self, config: dict, *, poison_seed: int, timeout_s: float) -> dict: ...

    def measure(self, config: dict, *, phase: str, pair_id: int, order: int,
                comparison_sha256: str, timeout_s: float) -> dict: ...


@dataclass(frozen=True)
class TunePolicy:
    max_candidates: int = 8
    warmup: int = 2
    screen_repeats: int = 3
    confirm_pairs: int = 9
    replay_pairs: int = 5
    validate_repeats: int = 2
    frontier_timeout_s: float = 10.0
    total_timeout_s: float = 180.0
    min_gain: float = 0.01
    max_p90_regression: float = 0.10

    def __post_init__(self):
        for key, lo, hi in (("max_candidates", 1, 8), ("warmup", 1, 5), ("screen_repeats", 3, 9),
                            ("confirm_pairs", 9, 31), ("replay_pairs", 5, 15), ("validate_repeats", 2, 4)):
            require(type(getattr(self, key)) is int and lo <= getattr(self, key) <= hi,
                    "BLOCKED_PROFILE", "invalid lightweight tuner budget: " + key)
        for key, lo, hi in (("frontier_timeout_s", .1, 30), ("total_timeout_s", 1, 600),
                            ("min_gain", .01, .5), ("max_p90_regression", 0, .1)):
            number(getattr(self, key), lo, hi, key)


def number(value, low, high, label):
    require(type(value) in {int, float} and math.isfinite(value) and low <= value <= high,
            "BLOCKED_PROFILE", "invalid measured value: " + label)
    return value


def history_bytes(shape):
    return 2 * 2 * shape["batch"] * shape["history_tokens"] * shape["kv_heads"] * shape["dim"]


def observation(record, root):
    checked(record, "artifact")
    return read_json(verify_artifact(record, root))


def validate_observation(data, request, config):
    """Require real BF16 same-input dense errors plus byte-exact KV and NaN checks."""
    checked(data, "validation")
    require(data.get("execution") == "CUDA" and data.get("request_sha256") == digest(request)
            and data.get("config_sha256") == digest(config), "BLOCKED_PROFILE", "validation context differs")
    rows = data.get("ranks")
    require(isinstance(rows, list) and len(rows) == len(request["ranks"]), "BLOCKED_PROFILE", "missing golden ranks")
    for index, (row, rank) in enumerate(zip(rows, request["ranks"])):
        require(row.get("rank") == index and row.get("input_kind") == "REAL_CHECKPOINT_QKV"
                and row.get("dtype") == "bf16", "BLOCKED_PROFILE", "real per-rank BF16 checkpoint QKV required")
        for key in ("q_sha256", "k_sha256", "v_sha256", "dense_output_sha256", "dense_lse_sha256"):
            checked({"path": "tensor", "sha256": row.get(key)}, "artifact")
        ref = row.get("reference")
        require(ref in ("SAME_INPUT_DENSE", "SAME_INPUT_NATIVE_FA2_LSE_DENSE_FP32_OUTPUT_V2")
                and row.get("selected_ids_unchanged") is True,
                "BLOCKED_PROFILE", "Numerical reference or selected KV IDs changed")
        if ref == "SAME_INPUT_NATIVE_FA2_LSE_DENSE_FP32_OUTPUT_V2":
            checked({"path": "native-fp32-lse", "sha256": row.get("native_lse_sha256")}, "artifact")
            number(row.get("dense_lse_max_abs"), 0, 1e9, "dense_lse_max_abs")
        else:
            require("native_lse_sha256" not in row and "dense_lse_max_abs" not in row,
                    "BLOCKED_PROFILE", "legacy dense contract must not mix native LSE fields")
        for tensor in ("q", "k", "v"):
            require(row.get(tensor + "_after_sha256") == row[tensor + "_sha256"],
                    "BLOCKED_PROFILE", "Q/K/V input changed during validation")
        for tensor in ("k", "v"):
            require(row.get(tensor + "_destination_sha256") == row[tensor + "_sha256"],
                    "BLOCKED_PROFILE", "retrieved KV bytes differ from expected logical tensor")
        expected = history_bytes(rank["shape"])
        resident = rank["shape"]["history_tokens"] == 0
        require(type(row.get("h2d_bytes")) is int and row["h2d_bytes"] == expected
                and type(row.get("nan_poisoned_bytes")) is int and row["nan_poisoned_bytes"] == expected
                and (row.get("resident_zero_history") is True if resident
                     else "resident_zero_history" not in row),
                "BLOCKED_PROFILE", "historical KV payload/NaN poison coverage differs; "
                                   "zero-H2D Resident Golden must be explicitly marked")
        if resident:
            require(row["reference"] == "SAME_INPUT_NATIVE_FA2_LSE_DENSE_FP32_OUTPUT_V2"
                    and row["k_destination_sha256"] == row["k_sha256"]
                    and row["v_destination_sha256"] == row["v_sha256"],
                    "BLOCKED_PROFILE", "Resident must consume unchanged real model KV with Native-FA2 LSE reference")
        require(type(row.get("nan_survivors")) is int and row["nan_survivors"] == 0
                and row.get("output_finite") is True and row.get("lse_finite") is True,
                "BLOCKED_PROFILE", "NaN contamination or nonfinite output/LSE")
        require(row.get("frontier_stalled") is False and row.get("final_join") is True,
                "GPU_GATE_REQUIRED", "Frontier stalled or missing producer/consumer join; discard worker")
        for key, maximum in (("output_max_abs", .03125), ("output_rel_l2", .002),
                             ("output_normalized_max", .5), ("lse_max_abs", .0001)):
            number(row.get(key), 0, maximum, key)
    return data


def binding_digest(profile):
    return digest({k: profile[k] for k in ("key", "config", "source_files", "kernel", "provenance")})


def verify_golden(profile, request, root):
    require(profile["golden"] is not None, "GPU_GATE_REQUIRED", "current exact-binding CUDA Golden receipt required")
    receipt = observation(profile["golden"], root)
    checked(receipt, "golden")
    require(receipt.get("validation_level") == "CUDA_GOLDEN" and receipt.get("status") == "VALIDATED"
            and receipt.get("binding_sha256") == binding_digest(profile)
            and receipt.get("request_sha256") == digest(request),
            "GPU_GATE_REQUIRED", "stale/wrong Golden receipt")
    records = receipt.get("validations", [])
    require(isinstance(records, list) and len(records) >= 2
            and len({r.get("sha256") for r in records}) == len(records),
            "GPU_GATE_REQUIRED", "two independent NaN-poison Golden replays required")
    poisons = []
    for record in records:
        data = validate_observation(observation(record, root), request, profile["config"])
        if receipt.get("reference_contract") == "NATIVE_FA2_LSE_PARITY_V2":
            require(all(row["reference"] == "SAME_INPUT_NATIVE_FA2_LSE_DENSE_FP32_OUTPUT_V2"
                        for row in data["ranks"]), "GPU_GATE_REQUIRED", "native parity receipt has mismatched reference")
        poisons.append(data.get("poison_seed"))
    require(all(type(seed) is int for seed in poisons) and len(set(poisons)) == len(poisons),
            "GPU_GATE_REQUIRED", "Golden replay poison seeds must differ")
    return receipt


def candidate_configs(request, *, limit=8):
    """Small deterministic subset of original search.py candidates, never a winner."""
    require(request_family(request) in {"H100", "B200"}, "BLOCKED_PROFILE", "autotune forbidden on frozen devices")
    require(type(limit) is int and 1 <= limit <= 8, "BLOCKED_PROFILE", "lightweight candidate budget must be 1..8")
    groups = [g for g in (request["ranks"][0]["shape"]["heads"], 3, 2, 4, 1)
              if all(r["shape"]["heads"] % g == 0 for r in request["ranks"])]
    choices = []
    # Source SearchSpace compiles M64/M128, N32/64/128; split only on M64.
    # Prioritize two archived search priors, then cover alternate tiles.
    triples = [(64, 64, 1), (64, 64, 4), (64, 32, 1), (128, 64, 1),
               (64, 128, 1), (128, 32, 1), (128, 128, 1)]
    for m, n, split in triples:
        g = 2 if split == 4 and 2 in groups else groups[0]
        config = {"backend": "original_b01", "entrypoint": "autotune", "block_m": m,
                  "block_n": n, "splits": split, "head_group": g, "gate_tiles": 4,
                  "layout": "head_group_major", "host_pages": "thp2m_slab", "dispatch": "early",
                  "producer": "cpp_dma", "cta_order": "native", "cache_frontier": False}
        config["consumer_start"] = "initialized"
        check_config(config, request)
        choices.append(config)
    for g in dict.fromkeys(groups):
        candidate = {**choices[0], "head_group": g, "host_pages": "pytorch"}
        if candidate not in choices:
            choices.append(candidate)
    return choices[:limit]


def measurement(data, request, config, *, phase, pair_id, order, comparison_sha256=None):
    checked(data, "measurement")
    require(data.get("execution") == "CUDA" and data.get("request_sha256") == digest(request)
            and data.get("config_sha256") == digest(config)
            and data.get("phase") == phase and data.get("pair_id") == pair_id and data.get("order") == order
            and data.get("comparison_sha256") == (comparison_sha256 or digest(config)),
            "BLOCKED_PROFILE", "measurement order/context differs")
    rows = data.get("ranks")
    require(isinstance(rows, list) and len(rows) == len(request["ranks"]), "BLOCKED_PROFILE", "missing measured ranks")
    walls = []
    for index, (row, rank) in enumerate(zip(rows, request["ranks"])):
        require(row.get("frontier_stalled") is False and row.get("final_join") is True,
                "GPU_GATE_REQUIRED", "Frontier stalled or unjoined streams; discard worker")
        require(row.get("rank") == index and row.get("producer") == "cpp_dma"
                and row.get("consumer") == "original_b01" and row.get("final_join") is True
                and row.get("frontier_stalled") is False and row.get("page_backing_verified") is True
                and row.get("h2d_bytes") == history_bytes(rank["shape"])
                and row.get("timing_scope") == "DMA_RESIDENT_PACK_FA2_MERGE_JOIN",
                "BLOCKED_PROFILE", "missing real DMA/FA2/scope/page/Frontier evidence")
        walls.append(number(row.get("wall_ms"), .000001, 1e6, "wall_ms"))
        number(row.get("gpu_critical_ms"), .000001, row["wall_ms"], "gpu_critical_ms")
        for key in ("dma_ms", "fa2_ms", "resident_pack_ms", "split_merge_ms"):
            number(row[key], .000001 if key in {"dma_ms", "fa2_ms"} else 0, row["wall_ms"], key)
        s = rank["shape"]
        require(row["h2d_copies"] == 2 * (s["history_tokens"] // s["frame_tokens"]) * (s["heads"] // config["head_group"]),
                "BLOCKED_PROFILE", "group/frame K+V DMA copy count differs")
        require((row["split_merge_ms"] == 0) == (config["splits"] == 1),
                "BLOCKED_PROFILE", "Split merge timing does not match actual configuration")
    # Maximum is taken *within each paired step*, before computing statistics.
    return max(walls)


def p90(values):
    ordered = sorted(values)
    return ordered[math.ceil(.9 * len(ordered)) - 1]


def compare_pairs(pairs, policy):
    old = [p[0] for p in pairs]
    new = [p[1] for p in pairs]
    gains = [(a - b) / a for a, b in pairs]
    rng = random.Random(20261026)
    boot = sorted(statistics.mean(rng.choices(gains, k=len(gains))) for _ in range(1000))
    lower = boot[24]
    gain = 1 - statistics.median(new) / statistics.median(old)
    tail = p90(new) / p90(old) - 1
    return {"old_median_max_rank_wall_ms": statistics.median(old),
            "candidate_median_max_rank_wall_ms": statistics.median(new),
            "gain": gain, "paired_bootstrap_95_lower": lower, "p90_regression": tail,
            "accepted": gain >= policy.min_gain and lower > 0 and tail <= policy.max_p90_regression}


def run_tuner(catalog, request, baseline, root, adapter, *, policy=None, mocked=False, assets=None):
    """Run a bounded A/B experiment through a trusted adapter; mocked runs cannot promote.

    CLI runs this function in a disposable process group with an outer watchdog.
    Adapters MUST implement their own per-call Frontier timeout and rank abort.
    """
    policy = policy or TunePolicy()
    catalog = checked(catalog, "catalog")
    require(request_family(request) in {"H100", "B200"}, "BLOCKED_PROFILE", "frozen GPU: no autotune")
    verify_binding(baseline, request, root, catalog, assets=assets)
    for record in catalog["tuner_sources"]:
        verify_artifact(record, (assets or {}).get(record["asset"], root))
    started = time.monotonic()

    def call(method, *args, **kwargs):
        require(time.monotonic() - started < policy.total_timeout_s,
                "GPU_GATE_REQUIRED", "total autotune budget exhausted; discard worker")
        before = time.monotonic()
        try:
            value = method(*args, **kwargs, timeout_s=policy.frontier_timeout_s)
        except TimeoutError as exc:
            raise ProfileError("GPU_GATE_REQUIRED", "Frontier/adapter timeout; discard worker") from exc
        require(time.monotonic() - before <= policy.frontier_timeout_s,
                "GPU_GATE_REQUIRED", "Frontier/call deadline exceeded; discard worker")
        require(time.monotonic() - started <= policy.total_timeout_s,
                "GPU_GATE_REQUIRED", "total autotune budget exhausted; discard worker")
        return value
    # describe() must inspect physical devices and the *loaded* extension/capture,
    # not echo the requested JSON. Adapter code is part of the trusted target gate.
    description = call(adapter.describe)
    require(description.get("request") == request, "BLOCKED_ABI", "adapter physical/shape/ABI probe differs")
    require(description.get("implementation") == "original_b01"
            and description.get("cpp_dma") is True and description.get("streaming_fa2") is True
            and description.get("real_checkpoint_qkv") is True,
            "BLOCKED_PROFILE", "original B01 native implementation and real QKV required")
    require(mocked or description.get("evidence_kind") == "REAL_TARGET_GPU",
            "GPU_GATE_REQUIRED", "mocked adapter cannot produce CUDA Golden or a winner")
    result = {"status": "GPU_GATE_REQUIRED" if mocked else "CUDA_GOLDEN",
              "validation_level": "STATIC" if mocked else "CUDA_GOLDEN", "mocked": mocked,
              "request_sha256": digest(request), "baseline_profile_sha256": baseline["sha256"],
              "baseline": deepcopy(baseline),
              "winner": None, "comparisons": [], "rejected": [], "validations": {}, "samples": [],
              "real_checkpoint_forward": "NOT_RUN", "video_e2e": "NOT_RUN", "official_eval": "NOT_RUN"}
    old = baseline["config"]

    def validate_config(config):
        records = []
        for seed in range(policy.validate_repeats):
            record = call(adapter.validate, deepcopy(config), poison_seed=20261026 + seed)
            data = validate_observation(observation(record, root), request, config)
            require(data.get("poison_seed") == 20261026 + seed, "BLOCKED_PROFILE", "wrong NaN replay seed")
            records.append(record)
        result["validations"][digest(config)] = records

    def measure(config, phase, pair_id, order, comparison_sha256=None):
        comparison_sha256 = comparison_sha256 or digest(config)
        record = call(adapter.measure, deepcopy(config), phase=phase, pair_id=pair_id, order=order,
                      comparison_sha256=comparison_sha256)
        wall = measurement(observation(record, root), request, config, phase=phase, pair_id=pair_id, order=order,
                           comparison_sha256=comparison_sha256)
        result["samples"].append({"config_sha256": digest(config), "phase": phase,
                                  "comparison_sha256": comparison_sha256,
                                  "pair_id": pair_id, "order": order, "max_rank_wall_ms": wall, "raw": record})
        return wall

    def paired(config, phase, count):
        pairs = []
        for pair_id in range(count):
            order = (old, config) if pair_id % 2 == 0 else (config, old)
            values = [measure(c, phase, pair_id, index, digest(config)) for index, c in enumerate(order)]
            pairs.append(tuple(values if pair_id % 2 == 0 else reversed(values)))
        return pairs

    validate_config(old)  # A broken old profile aborts the entire experiment.
    for i in range(policy.warmup):
        measure(old, "warmup", i, 0)
    screened = []
    configs = candidate_configs(request, limit=policy.max_candidates)
    for config in configs:
        if config == old or digest(config) in catalog["disabled_config_sha256"]:
            continue
        try:
            # support() must only query actual compiled exports/resources; no fallback.
            require(call(adapter.support, deepcopy(config)) is True,
                    "UNSUPPORTED_SHAPE", "candidate not supported by loaded B01 binary")
            validate_config(config)
            for i in range(policy.warmup):
                measure(config, "warmup", i, 0)
            samples = paired(config, "screen", policy.screen_repeats)
            screened.append((statistics.median(p[1] for p in samples), config))
        except (ProfileError, TimeoutError) as exc:
            result["rejected"].append({"config_sha256": digest(config),
                                       "code": getattr(exc, "code", "FRONTIER_TIMEOUT"), "reason": str(exc)})
            # A timeout may leave the CUDA context/frontier unsafe: abort all work.
            if isinstance(exc, TimeoutError) or getattr(exc, "code", None) == "GPU_GATE_REQUIRED":
                raise ProfileError("GPU_GATE_REQUIRED", "Frontier timeout; discard worker and all candidates") from exc
    if not screened:
        raise ProfileError("GPU_GATE_REQUIRED", "no valid original B01 candidates; baseline not substituted")
    best = old
    # Confirm only two finalists; keep full evidence even when promotion fails.
    for _, config in sorted(screened, key=lambda item: item[0])[:2]:
        confirm = compare_pairs(paired(config, "confirm", policy.confirm_pairs), policy)
        replay = compare_pairs(paired(config, "paired_replay", policy.replay_pairs), policy)
        entry = {"config": config, "confirm": confirm, "paired_replay": replay}
        result["comparisons"].append(entry)
        if confirm["accepted"] and replay["accepted"]:
            if best == old or confirm["candidate_median_max_rank_wall_ms"] < best_score:
                best, best_score = config, confirm["candidate_median_max_rank_wall_ms"]
    result["recommendation"] = {"config": best, "changed": best != old,
                                 "promotion": "GPU_GATE_REQUIRED" if mocked else "EXACT_BINDING_ONLY"}
    if not mocked:
        result["winner"] = best
    return result


def verify_tuning_evidence(record, profile, request, root):
    """Recompute A/B decisions from hashed raw samples before GPU-run authorization."""
    result = observation(record, root)
    require(result.get("mocked") is False and result.get("status") == "CUDA_GOLDEN"
            and result.get("validation_level") == "CUDA_GOLDEN"
            and result.get("request_sha256") == digest(request) and result.get("winner") == profile["config"],
            "GPU_GATE_REQUIRED", "missing real exact-shape H100/B200 tuning evidence")
    baseline = checked(result.get("baseline"), "profile")
    require(result.get("baseline_profile_sha256") == baseline["sha256"] and baseline["key"] == request
            and baseline["source_files"] == profile["source_files"] and baseline["kernel"] == profile["kernel"],
            "GPU_GATE_REQUIRED", "old profile A/B binding differs")
    for config in (baseline["config"], profile["config"]):
        records = result.get("validations", {}).get(digest(config), [])
        require(len(records) >= 2, "GPU_GATE_REQUIRED", "old/candidate independent numerical validations required")
        seeds = []
        for raw in records:
            seeds.append(validate_observation(observation(raw, root), request, config)["poison_seed"])
        require(len(set(seeds)) == len(seeds), "GPU_GATE_REQUIRED", "duplicate numerical validation replay")
    if baseline["config"] == profile["config"]:
        # No winning candidate is claimed. The freshly revalidated old profile is retained.
        require(result.get("recommendation", {}).get("changed") is False,
                "GPU_GATE_REQUIRED", "baseline retention mislabeled as a new winner")
        return result
    comparisons = [c for c in result.get("comparisons", []) if c.get("config") == profile["config"]]
    require(len(comparisons) == 1, "GPU_GATE_REQUIRED", "winner lacks paired A/B comparison")
    for phase, minimum in (("confirm", 9), ("paired_replay", 5)):
        samples = [s for s in result.get("samples", []) if s.get("phase") == phase
                   and s.get("comparison_sha256") == digest(profile["config"])]
        require(len(samples) >= minimum * 2 and len(samples) % 2 == 0,
                "GPU_GATE_REQUIRED", "incomplete paired A/B sample set")
        pairs = []
        for pair_id in range(len(samples) // 2):
            rows = [s for s in samples if s.get("pair_id") == pair_id]
            require(len(rows) == 2 and {s.get("order") for s in rows} == {0, 1},
                    "GPU_GATE_REQUIRED", "paired replay order or ranks missing")
            values = {}
            for sample in rows:
                config = baseline["config"] if sample["order"] == pair_id % 2 else profile["config"]
                require(sample.get("config_sha256") == digest(config), "GPU_GATE_REQUIRED", "A/B order changed")
                values["old" if config == baseline["config"] else "new"] = measurement(
                    observation(sample["raw"], root), request, config, phase=phase, pair_id=pair_id,
                    order=sample["order"], comparison_sha256=digest(profile["config"]))
            pairs.append((values["old"], values["new"]))
        recomputed = compare_pairs(pairs, TunePolicy())
        require(recomputed["accepted"] and recomputed == comparisons[0].get(phase),
                "GPU_GATE_REQUIRED", "A/B evidence does not support the reported winner")
    return result


def gpu_run_gate(catalog, request, root, adapter, *, tuning_record=None, assets=None):
    """The execution gate: a fresh physical probe plus saved exact-binding Golden.

    select_profile(gpu=True) alone only checks saved evidence; execution callers
    must use this gate with their trusted native adapter and currently loaded QKV.
    """
    from .profiles import select_profile
    result = select_profile(catalog, request, root, gpu=True, assets=assets)
    description = adapter.describe(timeout_s=10.0)
    require(description.get("evidence_kind") == "REAL_TARGET_GPU" and description.get("request") == request,
            "BLOCKED_ABI", "fresh physical GPU/loaded ABI/QKV probe differs or is mocked")
    producer = "cpp_dma" if result["profile"]["config"]["producer"] == "cpp_dma" else "stream_ordered_dma"
    require(description.get("implementation") == "original_b01" and description.get("streaming_fa2") is True
            and description.get(producer) is True and description.get("real_checkpoint_qkv") is True,
            "BLOCKED_PROFILE", "original B01 native execution/real QKV unavailable")
    require(adapter.support(deepcopy(result["profile"]["config"]), timeout_s=10.0) is True,
            "UNSUPPORTED_SHAPE", "loaded binary does not support the exact selected B01 entrypoint/configuration")
    result["acceptance"]["shape_kernel"]["status"] = "GPU_SUPPORT_PASS"
    if request_family(request) in {"H100", "B200"}:
        require(tuning_record is not None, "GPU_GATE_REQUIRED", "H100/B200 require old/candidate A/B receipt")
        verify_tuning_evidence(tuning_record, result["profile"], request, root)
    return {**result, "status": "GPU_RUN_ALLOWED"}


def profile_from_tuning(tuning_record, root, *, profile_id, golden_path):
    """Materialize an exact target binding from real evidence, without publishing it.

    golden_path is relative to the externally configured artifact root. The new
    profile is returned for the integration owner to review/add to a catalog.
    """
    result = observation(tuning_record, root)
    require(result.get("mocked") is False and result.get("status") == "CUDA_GOLDEN"
            and result.get("winner") is not None, "GPU_GATE_REQUIRED", "real target-hardware result required")
    baseline = checked(result.get("baseline"), "profile")
    require(baseline["family"] in {"H100", "B200"}, "BLOCKED_PROFILE", "frozen hardware cannot be tuned")
    profile = seal({**baseline, "id": profile_id, "config": result.get("winner"), "golden": None,
                    "provenance": {"origin": "target hardware original B01 paired A/B replay",
                                   "scope": "target_hardware", "references": [{"asset": "target", **tuning_record}]}})
    checked(profile, "profile")
    verify_tuning_evidence(tuning_record, profile, profile["key"], root)
    receipt = {"validation_level": "CUDA_GOLDEN", "status": "VALIDATED",
               "binding_sha256": binding_digest(profile), "request_sha256": digest(profile["key"]),
               "validations": result["validations"][digest(profile["config"])]}
    portable(golden_path)
    base = Path(root).resolve()
    path = (base / golden_path).resolve()
    require(path.is_relative_to(base), "BLOCKED_PROFILE", "Golden output escapes artifact root")
    # The artifact directory must already exist. Never overwrite earlier evidence.
    payload = json.dumps(receipt, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    with path.open("xb") as stream:
        stream.write(payload)
    profile["golden"] = {"path": golden_path, "sha256": hashlib.sha256(payload).hexdigest()}
    return seal(profile)
