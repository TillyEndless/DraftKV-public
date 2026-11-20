"""CPU-only admission for real Bench2 target tuning; never perform CUDA work."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import re

from draftkv.backbones.catalog import MODELS
from draftkv.backbones.sequence_parallel import local_heads
from draftkv.hardware_profiles.profiles import checked, request_family

ROOT = Path(__file__).resolve().parents[3]
RUNTIME = "bench2-lingbot-cu128"
BACKBONES = ("LingBot1.3B", "LingBot14B", "SolarWM33B")


def runtime_name(backbone):
    return "bench2-solarwm-cu128" if backbone == "SolarWM33B" else RUNTIME
UUID = re.compile(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}")


def device_key(device_uuids):
    if (not isinstance(device_uuids, (list, tuple)) or not device_uuids
            or any(not isinstance(value, str) or not UUID.fullmatch(value) for value in device_uuids)
            or len(set(device_uuids)) != len(device_uuids)):
        raise ValueError("distinct complete physical GPU UUIDs required in rank order")
    return hashlib.sha256("\n".join(device_uuids).encode()).hexdigest()[:16]


def canonical_output(cache, family, backbone, sp, device_uuids):
    if family not in ("H100", "B200") or backbone not in BACKBONES or type(sp) is not int or sp not in (1, 2, 4, 8):
        raise ValueError("unsupported Bench2 target autotune cell")
    if len(device_uuids) != sp:
        raise ValueError("physical GPU UUID count differs from SP")
    return (Path(cache).expanduser().resolve() / "autotune" / family / runtime_name(backbone) /
            backbone / f"sp{sp}" / device_key(device_uuids) / "accepted.json")


def runtime_identity(backbone="LingBot14B"):
    # Wan-family and H3 backbones have separate runtimes; each recipe is
    # source-derived and never described as a measured Bench2 GPU environment.
    config = json.loads((ROOT / "configs/environments/bench2.runtime.json").read_text())
    return config["groups"]["SolarWM33B" if backbone == "SolarWM33B" else "LingBot14B"]["identity"]


def validate_target_request(request, family, backbone, sp, device_uuids=None):
    """Bind native head/layout geometry, complete physical ranks and runtime ABI."""
    checked(request, "request")
    if backbone not in BACKBONES or type(sp) is not int or sp not in (1, 2, 4, 8):
        raise ValueError("unsupported Bench2 target autotune cell")
    if (family not in ("H100", "B200") or request_family(request) != family
            or request["parallel"]["sp"] != sp or len(request["ranks"]) != sp):
        raise ValueError("target tuning family/rank count/SP differs from release cell")
    uuids = [rank["device"]["uuid"] for rank in request["ranks"]]
    device_key(uuids)
    if device_uuids is not None and uuids != list(device_uuids):
        raise ValueError("target tuning physical GPU UUID rank order differs from --devices")
    spec = MODELS[backbone]
    heads = local_heads(spec.heads, spec.family, sp)
    for rank in request["ranks"]:
        shape = rank["shape"]
        if (shape["heads"] != heads or shape["kv_heads"] != heads or shape["dim"] != spec.head_dim
                or shape["layout"] != "BSHD" or shape["dtype"] != "bf16"):
            raise ValueError("target tuning native local-head/QKV geometry differs from backbone/SP")
        for side, length in (("q", shape["q"]), ("k", shape["kv"]), ("v", shape["kv"])):
            if shape[side + "_strides"] != [length * heads * spec.head_dim, heads * spec.head_dim, spec.head_dim, 1]:
                raise ValueError("target tuning requires contiguous native BSHD QKV")
    identity = runtime_identity(backbone)
    expected = {"torch": identity["torch"], "torch_cuda": identity["torch_cuda"],
                "python_abi": identity["soabi"], "cxx11_abi": identity["cxx11_abi"]}
    if any(request["abi"][key] != value for key, value in expected.items()):
        raise ValueError("target tuning ABI differs from the Bench2 CUDA12.8 runtime")
    return request


def validate_target_autotune(accepted, family, backbone, sp, device_uuids, *, build_spec=None, cache_directory=None):
    """Recheck raw A/B evidence and optionally the exact canonical native cache.

    The accepted JSON is an index, not an independently trusted attestation.
    Retain the source record and its hash-addressed raw artifact tree. Nothing
    here imports Torch, loads a binary, probes a GPU or invokes a tuner.
    """
    if not isinstance(accepted, dict):
        raise ValueError("real accepted Bench2 target autotune result required")
    evidence_path = accepted.get("evidence_path")
    if not isinstance(evidence_path, str) or not evidence_path:
        raise ValueError("target tuning cached evidence path required")
    path = Path(evidence_path).expanduser().resolve()
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != accepted.get("evidence_sha256"):
        raise ValueError("target tuning cached evidence hash mismatch")
    loader = importlib.util.spec_from_file_location("_bench2_target_materializer", ROOT / "scripts/bench2_autotune_result.py")
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    recomputed = module.materialize(accepted.get("tuning_record_path", ""), accepted.get("artifact_root", ""),
                                   family, backbone, sp)
    keys = ("schema_version", "accepted", "device_family", "backbone", "sp", "runtime", "parameters",
            "source_snapshot_sha256", "evidence_sha256", "request_sha256", "device_uuids", "target_request",
            "artifact_root", "tuning_record_path")
    if any(accepted.get(key) != recomputed.get(key) for key in keys):
        raise ValueError("accepted target result differs from verified original tuning evidence")
    request = validate_target_request(recomputed["target_request"], family, backbone, sp, device_uuids)
    if build_spec is not None:
        from draftkv.vendor.b01 import source_generation
        from draftkv.vendor.b01.build import compatibility_generation, validate_abi
        abi = validate_abi(build_spec.get("abi"))
        if (build_spec.get("source_generation") != source_generation()
                or build_spec.get("compatibility_generation") != compatibility_generation()
                or build_spec.get("arch") != {"H100": "9.0", "B200": "10.0"}[family]):
            raise ValueError("target tuning native source/build generation or architecture differs")
        pairs = (("torch", "torch_version"), ("torch_cuda", "torch_cuda"),
                 ("python_abi", "soabi"), ("cxx11_abi", "cxx11_abi"))
        if any(request["abi"][left] != abi[right] for left, right in pairs):
            raise ValueError("target tuning ABI differs from canonical B01 build spec")
        if cache_directory is None:
            raise ValueError("canonical B01 cache required with target build spec")
        from draftkv.vendor.b01.cache import verify_cache
        paths = verify_cache(cache_directory, build_spec, abi)
        if hashlib.sha256(paths["offload_fa2_ext"].read_bytes()).hexdigest() != request["b01"]["kernel_sha256"]:
            raise ValueError("target tuning kernel differs from the exact canonical B01 binary")
    return accepted
