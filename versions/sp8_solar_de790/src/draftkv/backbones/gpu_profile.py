"""Shared opt-in generation timing and six-metric profile export.

One synchronized E2E wall interval per video, optional diffusion-only interval
when the backbone exposes genuine sampling boundaries, PyTorch allocated and
reserved peaks, and external NVML process/device peaks. No per-block timers.
"""
from contextvars import ContextVar
from .memory_profile import (GPU_PEAK_MEMORY_SCOPE, current_recorder,
                             finalize_memory_profile, memory_phase)
import json
import math
import os
from pathlib import Path
import time

_ACTIVE = ContextVar("draftkv_generation_timer", default=None)
# Version 2 declared allocated memory as GPU Peak Memory. Never silently give
# an old profile the new NVML meaning. Version 3 is used by experiment branches.
# Version 4 added NVML; version 5 also requires unmodified raw timing and build admission.
PROFILE_SCHEMA_VERSION = 5
PROFILE_METRICS = (
    'e2e_latency_s', 'diffusion_latency_s', 'peak_pytorch_allocated_bytes',
    'peak_pytorch_reserved_bytes', 'generation_nvml_process_sampled_peak_bytes',
    'generation_nvml_device_sampled_peak_bytes',
)


def diffusion_start():
    """Mark the start of genuine DiT sampling (no-op when profiling is off)."""
    state = _ACTIVE.get()
    if state is None:
        return
    if state["diffusion_started"] is not None:
        raise RuntimeError("duplicate diffusion_start")
    state["cuda"].synchronize(state["device"])
    state["diffusion_started"] = time.perf_counter()
    memory_phase('diffusion')


def diffusion_end():
    """Mark the end of genuine DiT sampling, before VAE decode."""
    state = _ACTIVE.get()
    if state is None:
        return
    if state["diffusion_started"] is None or state["diffusion_elapsed_s"] is not None:
        raise RuntimeError("diffusion_end without a matching diffusion_start")
    state["cuda"].synchronize(state["device"])
    state["diffusion_elapsed_s"] = time.perf_counter() - state["diffusion_started"]
    memory_phase('post_diffusion')


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
    memory = current_recorder()
    try:
        with watch:
            started = time.perf_counter()
            watch.attach_runtime_loader(torch)
            token = _ACTIVE.set(state)
            try:
                if memory is not None:
                    memory.begin_generation(torch)
                result = call()
            finally:
                _ACTIVE.reset(token)
            torch.cuda.synchronize(device)
            e2e_elapsed = time.perf_counter() - started
    except BaseException:
        try:
            folder = Path(plan['run_dir']) / 'traces'
            folder.mkdir(parents=True, exist_ok=True)
            (folder / f'latency_profile_failure_rank{global_rank}.json').write_text(
                json.dumps(dict(status='FAILED', latency_admission=watch.report()),indent=2)+'\n')
        except OSError:
            pass
        raise
    allocated = int(torch.cuda.max_memory_allocated(device))
    reserved = int(torch.cuda.max_memory_reserved(device))
    if memory is not None:
        memory.end_generation()
    diffusion = state["diffusion_elapsed_s"]
    if diffusion is not None and (not math.isfinite(diffusion) or diffusion <= 0):
        raise RuntimeError("invalid diffusion-only latency")
    if state["diffusion_started"] is not None and diffusion is None:
        raise RuntimeError("incomplete diffusion timing markers")

    report = {
        "schema_version": PROFILE_SCHEMA_VERSION,
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
        "peak_pytorch_allocated_bytes": allocated,
        "peak_pytorch_reserved_bytes": reserved,
    }
    folder = Path(plan["run_dir"]) / "traces"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"gpu_profile_rank{global_rank}.json"
    with target.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def _reject_latency(plan, reason, record):
    folder = Path(plan['run_dir'])
    with (folder/'gpu_profile_rejected.json').open('w',encoding='utf-8') as stream:
        json.dump(dict(status='NOT_ADMITTED',reason=reason,e2e_latency_s=None,
            diffusion_latency_s=None,raw_record=record),stream,indent=2,allow_nan=False)
    raise ValueError('GPU profile not admitted: '+reason)


def aggregate_profiles(plan, *, require_complete=True):
    """Summarize one video; do not add rank times or overlap decode times."""
    if not plan.get("gpu_profile", False):
        return None
    profile_path = Path(plan['run_dir']) / 'gpu_profile.json'
    if profile_path.exists():
        raise FileExistsError(f'refuse to overwrite an existing profile: {profile_path}')
    folder = Path(plan["run_dir"]) / "traces"
    records = []
    for rank in range(plan["sp"]):
        path = folder / f"gpu_profile_rank{rank}.json"
        item = json.loads(path.read_text(encoding="utf-8"))
        from .latency_build import POLICY
        admission = item.get('latency_admission',{})
        if (item.get('schema_version') != PROFILE_SCHEMA_VERSION
                or admission.get('status') != 'PASS'
                or admission.get('policy') != POLICY
                or admission.get('runtime_cold_included') is not True
                or admission.get('build_time_subtracted') is not False
                or item.get('e2e_latency_s') != item.get('raw_generation_wall_s')
                or item.get('diffusion_latency_s') != item.get('raw_diffusion_wall_s')):
            _reject_latency(plan,'unverified build admission or altered raw timing',item)
        latency = item.get("e2e_latency_s")
        memory = item.get("peak_allocated_bytes")
        diffusion = item.get("diffusion_latency_s")
        reserved = item.get('peak_pytorch_reserved_bytes')
        if (item.get("schema_version") != PROFILE_SCHEMA_VERSION or item.get("status") != "COMPLETE"
                or item.get("rank") != rank
                or item.get("algorithm") != plan["algorithm"]
                or item.get("backbone") != plan["backbone"]
                or item.get("case_id") != plan["request"]["case_id"]
                or type(latency) not in (int, float) or not math.isfinite(latency)
                or latency <= 0 or type(memory) is not int or memory < 0
                or item.get('peak_pytorch_allocated_bytes') != memory
                or type(reserved) is not int or reserved < memory
                or (diffusion is not None and
                    (type(diffusion) not in (int, float)
                     or not math.isfinite(diffusion) or diffusion <= 0))):
            raise ValueError(f"invalid minimal GPU profile: {path}")
        records.append(item)

    peak = max(item["peak_allocated_bytes"] for item in records)
    reserved_values = [item.get('peak_pytorch_reserved_bytes') for item in records]
    if plan["backbone"] == "SolarWM33B":
        # SolarWM decodes on the same physical GPU in a separate process after
        # releasing its sampler. Its existing decode receipt already measures
        # the PyTorch peak; never launch another timed decode profile.
        path = Path(plan["run_dir"]) / "decode_receipt.json"
        decode = json.loads(path.read_text(encoding="utf-8"))
        if decode.get('latency_admission',{}).get('status') != 'PASS':
            _reject_latency(plan,'decode compilation admission failed',decode)
        if (decode.get("status") != "COMPLETE"
                or type(decode.get("peak_allocated_bytes")) is not int
                or decode["peak_allocated_bytes"] < 0
                or type(decode.get('peak_reserved_bytes')) is not int
                or decode['peak_reserved_bytes'] < decode['peak_allocated_bytes']):
            raise ValueError(f"invalid SolarWM decode receipt: {path}")
        peak = max(peak, decode["peak_allocated_bytes"])
        reserved_values.append(decode.get('peak_reserved_bytes'))

    diffusion_values = [r["diffusion_latency_s"] for r in records]
    if any(v is None for v in diffusion_values) and any(v is not None for v in diffusion_values):
        raise ValueError("inconsistent diffusion timing availability across ranks")
    diffusion_s = max(diffusion_values) if all(v is not None for v in diffusion_values) else None
    generated_frames = plan["request"]["frames"]
    if type(generated_frames) is not int or generated_frames < 1:
        raise ValueError("invalid generated output frame count")
    report = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "status": "COMPLETE",
        "backbone": plan["backbone"],
        "algorithm": plan["algorithm"],
        "case_id": plan["request"]["case_id"],
        "measurement_kind": ('profiler_correctness_fixture' if plan.get('profiler_correctness_fixture')
                             else 'video_generation'),
        "sp": plan["sp"],
        "e2e_latency_s": max(item["e2e_latency_s"] for item in records),
        "diffusion_latency_s": diffusion_s,
        "model_only_fps": generated_frames / diffusion_s if diffusion_s is not None else None,
        "output_frames": generated_frames,
        "peak_allocated_bytes": peak,
        "peak_pytorch_allocated_bytes": peak,
        "peak_pytorch_reserved_bytes": max(reserved_values) if all(type(v) is int for v in reserved_values) else None,
        "peak_memory_scope": GPU_PEAK_MEMORY_SCOPE,
        "per_rank": records,
        "latency_admission": dict(status='PASS',policy=POLICY,
            runtime_cold_included=True,build_time_subtracted=False),
        "build_preparation": [r.get('build_preparation') for r in records],
    }
    memory = finalize_memory_profile(plan['run_dir'], expected_sp=plan['sp'],
                                     worker_returncode=0, timing_profile=report)
    for key in ('generation_nvml_process_sampled_peak_bytes',
                'generation_nvml_device_sampled_peak_bytes', 'peak_gpu_memory_bytes',
                'peak_gpu_memory_gib', 'nvml_requested_interval_ms'):
        report[key] = memory[key]
    for key in PROFILE_METRICS[2:]:
        value = report[key]
        report[key.removesuffix('_bytes') + '_gib'] = value / (1024 ** 3) if value is not None else None
    missing = [key for key in PROFILE_METRICS if report[key] is None]
    if not memory['memory_comparison_eligible']:
        missing.append('complete_rank_and_NVML_receipts')
    report.update(profile_complete=not missing, missing_metrics=missing,
                  status='COMPLETE' if not missing else 'INCOMPLETE',
                  memory_comparison_eligible=memory['memory_comparison_eligible'],
                  memory_window='generation_including_conditioning_diffusion_and_vae_decode',
                  sp_aggregation='maximum per GPU/rank; never sum ranks')
    with profile_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    if missing and require_complete:
        raise ValueError('incomplete six-metric GPU profile: ' + ', '.join(missing)
                         + '; evidence retained at ' + str(profile_path))
    return profile_reference(report, profile_path)


def profile_reference(report, path):
    """One export contract for the runtime and generation helper scripts."""
    keys = (*PROFILE_METRICS, 'schema_version', 'profile_complete', 'missing_metrics',
            'peak_gpu_memory_bytes', 'peak_gpu_memory_gib', 'peak_memory_scope',
            'model_only_fps', 'nvml_requested_interval_ms', 'memory_comparison_eligible',
            'measurement_kind','latency_admission','build_preparation')
    keys += tuple(key.removesuffix('_bytes') + '_gib' for key in PROFILE_METRICS[2:])
    return dict(path=str(path), **{key: report[key] for key in keys})


def read_generation_profile(run):
    """Read current complete exports; historical allocated profiles stay historical."""
    path = Path(run) / 'gpu_profile.json'
    report = json.loads(path.read_text())
    if (report.get('schema_version') != PROFILE_SCHEMA_VERSION
            or report.get('status') != 'COMPLETE' or not report.get('profile_complete')
            or report.get('latency_admission',{}).get('status') != 'PASS'
            or report.get('measurement_kind') != 'video_generation'
            or report.get('peak_memory_scope') != GPU_PEAK_MEMORY_SCOPE
            or any(report.get(key) is None for key in PROFILE_METRICS)
            or report.get('peak_gpu_memory_bytes') != report.get('generation_nvml_device_sampled_peak_bytes')):
        raise ValueError(f'current complete six-metric profile required: {path}; do not relabel historical allocated peaks')
    return report
