"""Dependency-free JSON Schema contract (Draft 2020-12 subset used here)."""

import math
import re


def obj(properties, optional=()):
    return {"type": "object", "properties": properties,
            "required": [key for key in properties if key not in optional], "additionalProperties": False}


TEXT = {"type": "string", "minLength": 1}
HASH = {"type": "string", "pattern": "^[0-9a-f]{64}$"}
POS = {"type": "integer", "minimum": 1}
NAT = {"type": "integer", "minimum": 0}
BOOL = {"type": "boolean"}


def enum(*values):
    return {"enum": list(values)}


def array(items, minimum=1):
    return {"type": "array", "items": items, "minItems": minimum}


def ref(name):
    return {"$ref": "#/$defs/" + name}


DEFS = {}
DEFS["artifact"] = obj({"path": TEXT, "sha256": HASH})
DEFS["device"] = obj({
    "name": TEXT, "uuid": TEXT,
    "pci_bus_id": {"type": "string", "pattern": "^[0-9a-fA-F]{4,8}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\\.[0-7]$"},
    "pci_device_id": {"type": "string", "pattern": "^0x[0-9a-fA-F]{4,8}$"},
    "sm": POS, "sm_count": POS, "memory_bytes": POS,
})
DEFS["shape"] = obj({
    "batch": POS, "q": NAT, "kv": NAT, "heads": POS, "kv_heads": POS,
    "dim": POS, "dtype": enum("bf16", "fp16", "fp32"), "causal": BOOL,
    "layout": enum("BSHD", "head_group_major"),
    "q_strides": array(POS), "k_strides": array(POS), "v_strides": array(POS),
    "history_tokens": NAT, "frame_tokens": POS,
})
DEFS["request"] = obj({
    "ranks": array(obj({"device": ref("device"), "shape": ref("shape")})),
    "abi": obj({"driver": TEXT, "toolkit": TEXT, "torch": TEXT, "torch_cuda": TEXT,
                "python_abi": TEXT, "cxx11_abi": BOOL, "platform": TEXT}),
    "b01": obj({"source_sha256": HASH, "kernel_sha256": HASH}),
    "parallel": obj({"sp": POS, "strategy": enum("local", "ring", "ulysses"),
                     "input_placement": enum("attention_local", "sequence_sharded")}),
})
DEFS["config"] = obj({
    "backend": enum("original_b01"),
    "entrypoint": enum("frontier_opt_v9", "ready", "autotune", "split_5090", "h200_stock_frontier"),
    "block_m": enum(64, 128), "block_n": enum(32, 64, 128),
    "splits": {"type": "integer", "minimum": 1, "maximum": 16}, "head_group": POS, "gate_tiles": POS,
    "layout": enum("head_group_major", "BSHD"),
    "host_pages": enum("pytorch", "thp2m", "thp2m_slab"),
    "dispatch": enum("early", "all"), "producer": enum("cpp_dma", "framewise_dma"),
    "consumer_start": enum("initialized", "first_frame_ready"),
    "cta_order": enum("native"), "cache_frontier": enum(False),
})
DEFS["provenance"] = obj({
    "origin": TEXT, "scope": enum("synthetic_only", "historical_real_capture", "historical_video", "target_hardware"),
    "references": array(obj({"asset": TEXT, "path": TEXT, "sha256": HASH})),
})
DEFS["profile"] = obj({
    "schema_version": enum(1), "id": TEXT, "family": enum("RTX4090", "RTX5090", "H200", "H100", "B200"),
    "policy": enum("frozen", "autotuned"), "enabled": BOOL, "disabled_reason": {"type": "string"},
    "key": ref("request"), "config": ref("config"),
    "source_files": array(ref("artifact")), "kernel": ref("artifact"),
    "provenance": ref("provenance"),
    "golden": {"anyOf": [{"type": "null"}, ref("artifact")]},
    "sha256": HASH,
})
DEFS["kernel_constraints"] = obj({
    "provider": enum("legacy_grouped", "autotune", "split_5090", "h200_stock_frontier"),
    # An empty list means arbitrary positive local heads; group divisibility and
    # the provider's implementation limits are checked independently.
    "local_heads": array(POS, 0),
})
DEFS["recipe"] = obj({
    "id": TEXT, "family": enum("RTX4090", "RTX5090", "H200"),
    "shape": obj({"batch": POS, "q": POS, "kv": POS, "heads": POS, "kv_heads": POS,
                  "dim": POS, "dtype": enum("bf16"), "causal": BOOL, "sp": POS,
                  "history_tokens": NAT, "frame_tokens": POS, "layout": enum("BSHD", "head_group_major")}),
    "config": ref("config"), "provenance": ref("provenance"),
    "current_gate": enum("GPU_GATE_REQUIRED"),
    # shape records the historical experiment. A schedule-scoped recipe freezes
    # only scheduling options; a newly bound profile still has an exact key.
    "binding_scope": enum("exact_shape", "schedule"),
    "kernel_constraints": ref("kernel_constraints"),
}, optional=("binding_scope", "kernel_constraints"))
DEFS["catalog"] = obj({
    "schema_version": enum(1), "policies": obj({
        name: obj({"sm": enum(sm), "mode": enum(mode)})
        for name, sm, mode in (("RTX4090", 89, "frozen"), ("RTX5090", 120, "frozen"),
                              ("H200", 90, "frozen"), ("H100", 90, "autotune"), ("B200", 100, "autotune"))}),
    "recipes": array(ref("recipe")), "profiles": array(ref("profile"), 0),
    "disabled_profile_ids": array(TEXT, 0), "disabled_config_sha256": array(HASH, 0),
    "tuner_sources": array(obj({"asset": TEXT, "path": TEXT, "sha256": HASH})),
    "sha256": HASH,
})
DEFS["validation"] = obj({
    "execution": enum("CUDA"), "request_sha256": HASH, "config_sha256": HASH, "poison_seed": NAT,
    "ranks": array(obj({
        "rank": NAT, "input_kind": enum("REAL_CHECKPOINT_QKV"), "dtype": enum("bf16"),
        **{key: HASH for key in ("q_sha256", "k_sha256", "v_sha256", "q_after_sha256",
                                "k_after_sha256", "v_after_sha256", "k_destination_sha256",
                                "v_destination_sha256", "dense_output_sha256", "dense_lse_sha256")},
        "native_lse_sha256": HASH,
        "reference": enum("SAME_INPUT_DENSE", "SAME_INPUT_NATIVE_FA2_LSE_DENSE_FP32_OUTPUT_V2"), "selected_ids_unchanged": BOOL,
        "h2d_bytes": NAT, "nan_poisoned_bytes": NAT, "nan_survivors": NAT,
        "resident_zero_history": BOOL,
        "output_finite": BOOL, "lse_finite": BOOL, "frontier_stalled": BOOL, "final_join": BOOL,
        **{key: {"type": "number", "minimum": 0} for key in ("output_max_abs", "output_rel_l2",
                                                              "output_normalized_max", "lse_max_abs", "dense_lse_max_abs")},
    }, optional=("native_lse_sha256", "dense_lse_max_abs", "resident_zero_history"))),
})
DEFS["measurement"] = obj({
    "execution": enum("CUDA"), "request_sha256": HASH, "config_sha256": HASH,
    "phase": enum("warmup", "screen", "confirm", "paired_replay"), "pair_id": NAT, "order": NAT,
    "comparison_sha256": HASH,
    "ranks": array(obj({
        "rank": NAT, "producer": enum("cpp_dma"), "consumer": enum("original_b01"),
        "final_join": BOOL, "frontier_stalled": BOOL, "page_backing_verified": BOOL, "h2d_bytes": POS,
        "h2d_copies": POS,
        "timing_scope": enum("DMA_RESIDENT_PACK_FA2_MERGE_JOIN"),
        "wall_ms": {"type": "number", "minimum": 0}, "gpu_critical_ms": {"type": "number", "minimum": 0},
        **{key: {"type": "number", "minimum": 0} for key in
           ("dma_ms", "fa2_ms", "resident_pack_ms", "split_merge_ms")},
    })),
})
DEFS["golden"] = obj({
    "validation_level": enum("CUDA_GOLDEN"), "status": enum("VALIDATED"),
    "binding_sha256": HASH, "request_sha256": HASH, "validations": array(ref("artifact"), 2),
    "reference_contract": enum("NATIVE_FA2_LSE_PARITY_V2"),
}, optional=("reference_contract",))
DEFS["acceptance_stage"] = obj({
    "status": enum("NOT_RUN", "STATIC_PASS", "CUDA_GOLDEN_PASS", "GPU_SUPPORT_PASS", "RAW_CANDIDATE_RECORDED"),
    "evidence": {"anyOf": [{"type": "null"}, ref("artifact")]},
})
DEFS["acceptance"] = obj({
    "request_sha256": HASH, "config_sha256": HASH,
    **{stage: ref("acceptance_stage") for stage in
       ("source_abi", "shape_kernel", "synthetic_golden", "real_qkv_golden", "forward", "overlap")},
})
SCHEMA = {"$schema": "https://json-schema.org/draft/2020-12/schema",
          "$ref": "#/$defs/profile", "$defs": DEFS}


def validate(value, kind):
    """Validate exactly the published schema, rejecting unknown fields and bool-as-int."""
    def check(item, spec, path):
        if "$ref" in spec:
            return check(item, DEFS[spec["$ref"].split("/")[-1]], path)
        if "anyOf" in spec:
            for option in spec["anyOf"]:
                try:
                    check(item, option, path)
                    return
                except ValueError:
                    pass
            raise ValueError(f"{path}: no matching type")
        if "enum" in spec and not any(type(item) is type(x) and item == x for x in spec["enum"]):
            raise ValueError(f"{path}: invalid enum")
        types = {"object": dict, "array": list, "integer": int, "string": str, "boolean": bool, "null": type(None)}
        valid_type = "type" not in spec or (type(item) in {int, float} if spec["type"] == "number"
                                            else type(item) is types[spec["type"]])
        if not valid_type:
            raise ValueError(f"{path}: expected {spec['type']}")
        if isinstance(item, dict) and "properties" in spec:
            if not set(spec["required"]) <= set(item) or not set(item) <= set(spec["properties"]):
                raise ValueError(f"{path}: missing or unknown fields")
            for key, child in item.items():
                check(child, spec["properties"][key], path + "." + key)
        if isinstance(item, list) and "items" in spec:
            if len(item) < spec.get("minItems", 0):
                raise ValueError(f"{path}: empty array")
            for index, child in enumerate(item):
                check(child, spec["items"], f"{path}[{index}]")
        if isinstance(item, str):
            if len(item) < spec.get("minLength", 0) or ("pattern" in spec and not re.fullmatch(spec["pattern"], item)):
                raise ValueError(f"{path}: invalid string")
        if type(item) in {int, float} and (not math.isfinite(item) or item < spec.get("minimum", item)
                                         or item > spec.get("maximum", item)):
            raise ValueError(f"{path}: outside numeric range")
    check(value, DEFS[kind], kind)
    return value
