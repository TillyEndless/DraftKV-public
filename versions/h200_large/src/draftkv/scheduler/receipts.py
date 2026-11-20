"""Decode media, validate native evidence, and invoke only pinned evaluators."""

from fractions import Fraction
import json
import math
from pathlib import Path

from .common import command, digest, object_hash, read_json, atomic_json
from .gates import pinned


def media_receipt(video, config):
    data = json.loads(command(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                               "-show_entries", "stream=codec_type,width,height,nb_read_frames,avg_frame_rate,duration:format=format_name,duration",
                               "-of", "json", str(video)], timeout=180))
    streams = data.get("streams", [])
    if len(streams) != 1 or int(streams[0].get("nb_read_frames", 0)) != config["expected_frames"]:
        raise ValueError("incorrect decoded video frame count")
    stream = streams[0]
    if "mp4" not in data.get("format", {}).get("format_name", ""):
        raise ValueError("output is not an MP4")
    fps = Fraction(stream["avg_frame_rate"])
    if fps != Fraction(str(config["fps"])):
        raise ValueError("video FPS differs from frozen output timeline")
    duration = float(stream.get("duration", data["format"]["duration"]))
    if abs(duration - config["expected_frames"] / config["fps"]) > 1 / config["fps"]:
        raise ValueError("video duration differs from frozen timeline")
    # Inspect every decoded timestamp as well as the nominal rate.
    timestamps = command(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "frame=best_effort_timestamp_time", "-of", "csv=p=0", str(video)], timeout=180)
    times = [float(line.split(",")[0]) for line in timestamps.splitlines() if line.strip() and line.split(",")[0]]
    if len(times) != config["expected_frames"] or any(abs(t - i / config["fps"]) > 1e-4 for i, t in enumerate(times)):
        raise ValueError("video frame timestamps do not align with requested poses")
    command(["ffmpeg", "-v", "error", "-xerror", "-i", str(video), "-map", "0:v:0", "-f", "null", "-"], timeout=300)
    return dict(path=str(Path(video).resolve()), sha256=digest(video), ffprobe=data, frames=len(times),
                fps=float(fps), first_timestamp=times[0], last_timestamp=times[-1], decoded=True)


def trace_rows(path, task_id, config_sha, token, *, partial=False):
    rows = []
    if not Path(path).exists() and partial:
        return rows
    with Path(path).open() as stream:
        for line in stream:
            if partial and not line.endswith("\n"):
                break
            row = json.loads(line)
            if row.get("schema") != "draftkv.scheduler.trace.v1" or any(row.get(k) != v for k, v in
                    {"task_id": task_id, "config_sha256": config_sha, "attempt_token": token}.items()):
                raise ValueError("native trace identity/schema mismatch")
            rows.append(row)
    return rows


def trace_receipt(path, config, task_id, token):
    rows = trace_rows(path, task_id, object_hash(config), token)
    by_event = {}
    for row in rows:
        by_event.setdefault(row.get("event"), []).append(row)
    for name in ("checkpoint_loaded", "conditioning", "algorithm_hook", "forward", "generation_complete"):
        if not by_event.get(name):
            raise ValueError(f"native trace lacks {name}")
    if not any(r.get("checkpoint_sha256") == object_hash(config["checkpoint"]) for r in by_event["checkpoint_loaded"]):
        raise ValueError("loaded checkpoint provenance mismatch")
    if not any(r.get("case_id") == config["case_id"] and r.get("inputs_sha256") == object_hash(config["inputs"])
               and r.get("manifest_sha256") == config["manifest_sha256"] for r in by_event["conditioning"]):
        raise ValueError("actual conditioning mismatch")
    if not any(r.get("kernel_id") == config["runtime"]["kernel_id"] and r.get("algorithm") == config["algorithm"]
               and isinstance(r.get("calls"), int) and r["calls"] > 0 for r in by_event["algorithm_hook"]):
        raise ValueError("actual algorithm hook calls not proven")
    progress = [r for r in rows if r.get("event") in ("forward", "kernel", "chunk")]
    seqs = [r.get("progress_seq") for r in progress]
    if any(type(x) is not int or x < 0 for x in seqs) or any(b <= a for a, b in zip(seqs, seqs[1:])):
        raise ValueError("kernel/chunk progress must increase")
    complete = by_event["generation_complete"][-1]
    if complete.get("frames") != config["expected_frames"] or complete.get("complete_target") is not True:
        raise ValueError("target generation incomplete")
    if complete.get("posthoc_trim_or_resample") is not False:
        raise ValueError("posthoc temporal conversion is forbidden")
    return dict(path=str(Path(path).resolve()), sha256=digest(path), events=len(rows),
                last_kernel_or_chunk=progress[-1], hook_calls=sum(r.get("calls", 0) for r in by_event["algorithm_hook"]))


def evaluator_preflight(config):
    spec = config.get("evaluator", {})
    protocol = read_json(pinned(spec.get("protocol")))
    pinned(spec.get("adapter"))
    expected = "orbia" if config["benchmark"] == "bench2" else "vbench-long"
    if protocol.get("evaluator") != expected or protocol.get("official_mode") is not True or protocol.get("verified") is not True:
        raise ValueError("official evaluator protocol not verified")
    if protocol.get("reference_manifest_sha256") != config["manifest_sha256"] or not protocol.get("verification_reference"):
        raise ValueError("official evaluator reference binding absent")
    if protocol.get("adapter_sha256") != spec["adapter"]["sha256"] or not protocol.get("official_parameters"):
        raise ValueError("official parameters/adapter pin mismatch")
    if expected == "vbench-long" and protocol.get("long_video_mode") is not True:
        raise ValueError("Bench1 requires VBench-Long")
    if not protocol.get("dependency_versions"):
        raise ValueError("official evaluator dependency pins missing")
    requirement = spec.get("memory", {})
    # Evaluator memory/SP estimates are telemetry only; runtime is authoritative.
    # Run this in the evaluator's Python, not the scheduler's environment.
    code = ("import importlib.metadata as m,json,sys; "
            "print(json.dumps({k:m.version(k) for k in json.loads(sys.argv[1])}))")
    versions = json.loads(command([spec["python"], "-c", code, json.dumps(protocol["dependency_versions"])]))
    if versions != protocol["dependency_versions"]:
        raise ValueError("official evaluator dependency version mismatch")
    return protocol


def evaluate(config, folder, request, run):
    """The reviewed adapter must call the official per-case/subset API.

    A five-case score is an official case evaluation, never a full benchmark pass.
    GPU evaluator execution uses the same owned process/lease monitor as generation.
    """
    try:
        protocol = evaluator_preflight(config)
    except Exception as exc:
        return {"state": "BLOCKED_ENVIRONMENT", "reason": str(exc), "metrics": None}
    evaluation = dict(request, protocol=protocol, video=read_json(folder / "video_receipt.json"))
    atomic_json(folder / "evaluation_request.json", evaluation)
    request_sha = digest(folder / "evaluation_request.json")
    spec = config["evaluator"]
    returncode = run([spec["python"], spec["adapter"]["path"], "--request", str(folder / "evaluation_request.json"),
                      "--output", str(folder / "official_result.json")], "EVALUATING")
    if returncode:
        raise RuntimeError(f"official evaluator exit {returncode}")
    result = read_json(folder / "official_result.json")
    if result.get("request_sha256") != request_sha or result.get("official_mode") is not True or result.get("case_ids") != [config["case_id"]]:
        raise ValueError("official result request/case mismatch")
    if result.get("evaluator") != protocol["evaluator"] or result.get("reference_manifest_sha256") != config["manifest_sha256"]:
        raise ValueError("official result protocol mismatch")
    metrics = result.get("scores")
    if not isinstance(metrics, dict) or not metrics or any(type(v) not in (int, float) or not math.isfinite(v) for v in metrics.values()):
        raise ValueError("official metrics absent/non-finite")
    raw = pinned(result.get("raw_official_output"))
    if folder.resolve() not in raw.resolve().parents:
        raise ValueError("raw evaluator output must be retained in this attempt")
    return dict(state="OFFICIAL_CASE_EVALUATION_PASS", metrics=metrics,
                result_sha256=digest(folder / "official_result.json"), result_path=str((folder / "official_result.json").resolve()),
                raw_sha256=digest(raw), raw_path=str(raw.resolve()), request_sha256=request_sha,
                request_path=str((folder / "evaluation_request.json").resolve()),
                protocol_sha256=spec["protocol"]["sha256"], evaluator=protocol["evaluator"])


def verified_success(task):
    """Hash drift always blocks reuse/promotion; it never silently reruns success."""
    try:
        result = json.loads(task["result"])
        if task["state"] != "COMPLETE" or result["acceptance"] != "VIDEO_E2E_PASS":
            return False
        config = json.loads(task["config"])
        for key in ("video", "trace"):
            if digest(result[key]["path"]) != result[key]["sha256"]:
                return False
        return result["config_sha256"] == object_hash(config) and result["exit_code"] == 0
    except (OSError, KeyError, ValueError, TypeError):
        return False


def official_case_verified(task):
    if not verified_success(task):
        return False
    try:
        evaluation = json.loads(task["result"])["evaluation"]
        metrics = evaluation["metrics"]
        if evaluation["state"] != "OFFICIAL_CASE_EVALUATION_PASS" or not isinstance(metrics, dict) or not metrics:
            return False
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in metrics.values()):
            return False
        return all(digest(evaluation[k + "_path"]) == evaluation[k + "_sha256"] for k in ("result", "raw", "request"))
    except (KeyError, OSError, ValueError, TypeError):
        return False
