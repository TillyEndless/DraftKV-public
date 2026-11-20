"""Minimal opt-in LongLive-2.0-style generation timing.

One synchronized E2E wall interval per video, optional diffusion-only interval
when the backbone exposes genuine sampling boundaries, and one PyTorch GPU
allocated-memory peak. No CUDA Events, per-block timers, or VAE sub-timers.
"""
from contextvars import ContextVar
import json
import math
import os
from pathlib import Path
import time

_ACTIVE = ContextVar("draftkv_generation_timer", default=None)


def diffusion_start():
    """Mark the start of genuine DiT sampling (no-op when profiling is off)."""
    state = _ACTIVE.get()
    if state is None:
        return
    if state["diffusion_started"] is not None:
        raise RuntimeError("duplicate diffusion_start")
    state["cuda"].synchronize(state["device"])
    state["diffusion_started"] = time.perf_counter()


def diffusion_end():
    """Mark the end of genuine DiT sampling, before VAE decode."""
    state = _ACTIVE.get()
    if state is None:
        return
    if state["diffusion_started"] is None or state["diffusion_elapsed_s"] is not None:
        raise RuntimeError("diffusion_end without a matching diffusion_start")
    state["cuda"].synchronize(state["device"])
    state["diffusion_elapsed_s"] = time.perf_counter() - state["diffusion_started"]


def run_profiled(plan, call):
    """Measure an entire generation invocation, with exactly two E2E fences.

    Diffusion-only marks are optional, and are emitted only by backbones with
    a source-audited native sampling seam. Disabled mode directly calls the
    original function without importing torch or invoking any CUDA operation.
    """
    if not plan.get("gpu_profile", False):
        return call()

    import torch
    from .latency_build import compilation_watch, prepare_profile_builds
    preparation = prepare_profile_builds(plan)
    watch = compilation_watch(torch, preparation)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    global_rank = int(os.environ.get("RANK", str(local_rank)))
    device = torch.device("cuda", local_rank)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    state = dict(cuda=torch.cuda, device=device, diffusion_started=None,
                 diffusion_elapsed_s=None)
    try:
        with watch:
            started = time.perf_counter()
            watch.attach_runtime_loader(torch)
            token = _ACTIVE.set(state)
            try:
                result = call()
            finally:
                _ACTIVE.reset(token)
            torch.cuda.synchronize(device)
            e2e_elapsed = time.perf_counter() - started
    except BaseException:
        # Preserve the original failure without another CUDA fence or teardown.
        try:
            folder = Path(plan['run_dir']) / 'traces'
            folder.mkdir(parents=True, exist_ok=True)
            (folder / f'latency_profile_failure_rank{global_rank}.json').write_text(
                json.dumps(dict(status='FAILED', latency_admission=watch.report()), indent=2) + '\n')
        except OSError:
            pass
        raise
    allocated = int(torch.cuda.max_memory_allocated(device))
    diffusion = state["diffusion_elapsed_s"]
    if diffusion is not None and (not math.isfinite(diffusion) or diffusion <= 0):
        raise RuntimeError("invalid diffusion-only latency")
    if state["diffusion_started"] is not None and diffusion is None:
        raise RuntimeError("incomplete diffusion timing markers")

    report = {
        "schema_version": 3,
        "status": "COMPLETE" if not watch.reasons else "COMPILE_CONTAMINATED",
        "scope": ("generation_through_vae_decode_and_native_video_write"
                  if plan["backbone"] == "SolarWM33B" else
                  "generation_through_vae_decode_excluding_video_encoding"),
        "backbone": plan["backbone"],
        "algorithm": plan["algorithm"],
        "case_id": plan["request"]["case_id"],
        "rank": global_rank,
        "local_rank": local_rank,
        "physical_gpu_id": (plan.get("selected_devices") or [None] * (local_rank + 1))[local_rank],
        "raw_generation_wall_s": e2e_elapsed,
        "raw_diffusion_wall_s": diffusion,
        "e2e_latency_s": e2e_elapsed if not watch.reasons else None,
        "diffusion_latency_s": diffusion if not watch.reasons else None,
        "latency_admission": watch.report(),
        "build_preparation": preparation,
        "peak_allocated_bytes": allocated,
    }
    folder = Path(plan["run_dir"]) / "traces"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"gpu_profile_rank{global_rank}.json"
    with target.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def aggregate_profiles(plan):
    """Summarize one video; do not add rank times or overlap decode times."""
    if not plan.get("gpu_profile", False):
        return None
    folder = Path(plan["run_dir"]) / "traces"
    records = []
    for rank in range(plan["sp"]):
        path = folder / f"gpu_profile_rank{rank}.json"
        item = json.loads(path.read_text(encoding="utf-8"))
        from .latency_build import POLICY
        admission = item.get('latency_admission', {})
        if (item.get('schema_version') != 3 or item.get('status') != 'COMPLETE'
                or admission.get('status') != 'PASS' or admission.get('policy') != POLICY
                or admission.get('runtime_cold_included') is not True
                or admission.get('build_time_subtracted') is not False
                or item.get('raw_generation_wall_s') != item.get('e2e_latency_s')):
            rejected = dict(status='LATENCY_NOT_ADMITTED', rank=rank, profile=str(path),
                reason='compile-contaminated or legacy unaudited timing; raw profile retained',
                raw_generation_wall_s=item.get('raw_generation_wall_s', item.get('e2e_latency_s')),
                e2e_latency_s=None, latency_admission=admission)
            target = Path(plan['run_dir']) / 'gpu_profile_rejected.json'
            if not target.exists():
                target.write_text(json.dumps(rejected, indent=2) + '\n')
            raise ValueError(f'formal latency not admitted: {path}; see {target}')
        latency = item.get("e2e_latency_s")
        memory = item.get("peak_allocated_bytes")
        diffusion = item.get("diffusion_latency_s")
        if (item.get("schema_version") != 3 or item.get("status") != "COMPLETE"
                or item.get("rank") != rank
                or item.get("algorithm") != plan["algorithm"]
                or item.get("backbone") != plan["backbone"]
                or item.get("case_id") != plan["request"]["case_id"]
                or not isinstance(latency, (int, float)) or not math.isfinite(latency)
                or latency < 0 or type(memory) is not int or memory < 0
                or (diffusion is not None and
                    (not isinstance(diffusion, (int, float))
                     or not math.isfinite(diffusion) or diffusion <= 0))):
            raise ValueError(f"invalid minimal GPU profile: {path}")
        records.append(item)

    peak = max(item["peak_allocated_bytes"] for item in records)
    if plan["backbone"] == "SolarWM33B":
        # SolarWM decodes on the same physical GPU in a separate process after
        # releasing its sampler. Its existing decode receipt already measures
        # the PyTorch peak; never launch another timed decode profile.
        path = Path(plan["run_dir"]) / "decode_receipt.json"
        decode = json.loads(path.read_text(encoding="utf-8"))
        if (decode.get("status") != "COMPLETE"
                or type(decode.get("peak_allocated_bytes")) is not int
                or decode["peak_allocated_bytes"] < 0):
            raise ValueError(f"invalid SolarWM decode receipt: {path}")
        if (decode.get('latency_admission', {}).get('status') != 'PASS'
                or decode.get('latency_admission', {}).get('policy') != POLICY):
            raise ValueError(f'SolarWM decode compilation was not admitted: {path}')
        peak = max(peak, decode["peak_allocated_bytes"])

    diffusion_values = [r["diffusion_latency_s"] for r in records]
    if any(v is None for v in diffusion_values) and any(v is not None for v in diffusion_values):
        raise ValueError("inconsistent diffusion timing availability across ranks")
    diffusion_s = max(diffusion_values) if all(v is not None for v in diffusion_values) else None
    generated_frames = plan["request"]["frames"]
    if type(generated_frames) is not int or generated_frames < 1:
        raise ValueError("invalid generated output frame count")
    report = {
        "schema_version": 3,
        "status": "COMPLETE",
        "backbone": plan["backbone"],
        "algorithm": plan["algorithm"],
        "case_id": plan["request"]["case_id"],
        "sp": plan["sp"],
        "e2e_latency_s": max(item["e2e_latency_s"] for item in records),
        "diffusion_latency_s": diffusion_s,
        "model_only_fps": generated_frames / diffusion_s if diffusion_s is not None else None,
        "output_frames": generated_frames,
        "peak_gpu_memory_gib": peak / (1024 ** 3),
        "peak_allocated_bytes": peak,
        "peak_memory_scope": "max_per_process_pytorch_allocated_not_NVML_device_total",
        "per_rank": records,
        "formal_latency_admitted": True,
        "latency_policy": POLICY,
        "raw_generation_wall_s": max(item['raw_generation_wall_s'] for item in records),
    }
    path = Path(plan["run_dir"]) / "gpu_profile.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return {
        "path": str(path),
        "e2e_latency_s": report["e2e_latency_s"],
        "diffusion_latency_s": diffusion_s,
        "model_only_fps": report["model_only_fps"],
        "peak_gpu_memory_gib": report["peak_gpu_memory_gib"],
        "formal_latency_admitted": True,
        "latency_policy": POLICY,
    }
