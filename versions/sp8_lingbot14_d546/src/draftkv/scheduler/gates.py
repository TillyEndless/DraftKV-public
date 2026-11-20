"""Fail-closed CPU gates; no fallback kernels or invented runtime commands."""

from pathlib import Path
import shutil

from .common import command, digest, object_hash, read_json
from .manifests import validate_inputs, validate_manifest


def pinned(spec):
    if not isinstance(spec, dict) or not spec.get("path") or not spec.get("sha256"):
        raise ValueError("missing pinned file")
    path = Path(spec["path"])
    if not path.is_absolute() or not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"missing absolute nonempty file: {path}")
    if digest(path) != spec["sha256"]:
        raise ValueError(f"SHA256 mismatch: {path}")
    return path


def checkpoint(config):
    spec = config.get("checkpoint", {})
    shards = spec.get("files", [])
    if spec.get("real_weights") is not True or not shards:
        raise ValueError("real checkpoint shard inventory absent")
    files = {str(pinned(s).resolve()) for s in shards}
    index = spec.get("index")
    if index:
        p = pinned(index)
        names = set(read_json(p)["weight_map"].values())
        if not names or not {str((p.parent / name).resolve()) for name in names} <= files:
            raise ValueError("checkpoint index references unpinned/missing shards")
    if not spec.get("format") or not spec.get("source_revision"):
        raise ValueError("checkpoint format/source provenance missing")


def semantics(config):
    spec = config.get("runtime", {})
    pinned(spec.get("adapter"))
    if spec.get("algorithm") != config["algorithm"] or not spec.get("kernel_id"):
        raise ValueError("kernel/algorithm binding absent")
    sources = spec.get("source_files", [])
    if not sources:
        raise ValueError("algorithm static source inventory missing")
    for source in sources:
        pinned(source)
    if config["algorithm"] == "freekv":
        raft = spec.get("raft", {})
        pinned(raft.get("extension"))
        if raft.get("original_raft") is not True or not raft.get("torch_abi"):
            raise ValueError("original RAFT and current Torch ABI binding required")
    if config["algorithm"] == "draftkv":
        for key in ("b01_producer", "b01_consumer"):
            pinned(spec.get(key))
    if spec.get("trace_schema") != "draftkv.scheduler.trace.v1":
        raise ValueError("native Forward trace instrumentation missing")
    if not isinstance(spec.get("argv"), list) or not spec["argv"]:
        raise ValueError("native generator argv missing")
    if spec["argv"][0] != "{python}" or "{adapter}" not in spec["argv"] or "{request}" not in spec["argv"]:
        raise ValueError("native argv must use the pinned Python, adapter and frozen request")


def environment(config, probe=False):
    spec = config.get("environment", {})
    executable = Path(spec.get("python", ""))
    if not executable.is_file() or not executable.is_absolute():
        raise ValueError("runtime Python missing")
    pinned(spec.get("cpu_probe"))
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            raise ValueError(f"{tool} missing")
    if probe:
        import json
        proof = json.loads(command([str(executable), spec["cpu_probe"]["path"]], timeout=90))
        expected = {"torch_version": spec["torch_version"], "cuda_version": spec["cuda_version"],
                    "torch_abi": spec["torch_abi"], "sm": "sm90", "kernel_id": config["runtime"]["kernel_id"],
                    "adapter_sha256": config["runtime"]["adapter"]["sha256"]}
        if any(proof.get(k) != v for k, v in expected.items()) or proof.get("ok") is not True:
            raise ValueError("CPU native import/ABI probe failed or binding mismatch")
        if config["algorithm"] == "freekv":
            raft = config["runtime"]["raft"]
            if proof.get("raft_loaded") is not True or proof.get("raft_sha256") != raft["extension"]["sha256"] or raft["torch_abi"] != spec["torch_abi"]:
                raise ValueError("original RAFT load failed under current Torch ABI")
        return proof
    return {"probe": "DEFERRED_UNTIL_DISPATCH"}


def spatial_parallel(config):
    sp = config.get("sp")
    if type(sp) is not int or sp < 1:
        raise ValueError("SP must be a positive integer")
    # Prior measured memory/SP envelopes are telemetry only.
    # Runtime CUDA/NCCL execution is authoritative.
    for key in ("generation_no_progress_s", "evaluation_max_s"):
        value = config.get("timeouts", {}).get(key)
        if value is not None and type(value) not in (int, float):
            raise ValueError("timeout must be numeric when supplied")


def preflight(config, *, probe=False):
    if config.get("mode") == "evaluation_only":
        from .scoring import preflight_scoring
        try:
            evidence = preflight_scoring(config)
            return {"config_sha256": object_hash(config), "gates": [{"gate": "evaluation_only", "status": "PASS", "evidence": evidence}],
                    "state": "READY", "reason": ""}
        except Exception as exc:
            return {"config_sha256": object_hash(config), "gates": [{"gate": "evaluation_only", "status": "BLOCKED_ENVIRONMENT", "reason": str(exc)}],
                    "state": "BLOCKED_ENVIRONMENT", "reason": str(exc)}
    reports = []
    def run(name, blocked, call):
        try:
            evidence = call()
            reports.append(dict(gate=name, status="PASS", evidence=evidence))
        except Exception as exc:
            reports.append(dict(gate=name, status=blocked, reason=f"{type(exc).__name__}: {exc}"))
    run("dataset_manifest", "BLOCKED_INPUT", lambda: validate_manifest(config))
    run("conditioning", "BLOCKED_INPUT", lambda: validate_inputs(config, validate_manifest(config)))
    run("checkpoint", "BLOCKED_INPUT", lambda: checkpoint(config))
    run("kernel_raft_b01_trace", "BLOCKED_SEMANTICS", lambda: semantics(config))
    run("backbone_sp_memory", "BLOCKED_SEMANTICS", lambda: spatial_parallel(config))
    # Run the probe only after source, data, semantics and memory gates pass.
    run("environment", "BLOCKED_ENVIRONMENT", lambda: environment(config, probe=probe and all(r["status"] == "PASS" for r in reports)))
    failed = [r for r in reports if r["status"] != "PASS"]
    return {"config_sha256": object_hash(config), "gates": reports,
            "state": failed[0]["status"] if failed else "READY",
            "reason": "; ".join(r["gate"] + ": " + r["reason"] for r in failed)}
