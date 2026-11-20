"""Decode all frames and freeze a hash-pinned ORBIA video/run manifest."""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Mapping, Sequence

from draftkv.benchmarks.datasets import (
    contained_file, require_sha, required_text, sha256_file, unique_json,
)
from draftkv.benchmarks.orbia import ORBIADataset, ORBIADependencyError, FPS, SHORT_FRAMES
from .official import EvaluationError


def plain(value):
    if isinstance(value, Mapping):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def execution_scope(ids, frozen_ids):
    """Only the complete frozen order is a full 480-case evaluation."""
    return "full480" if len(ids) == 480 and list(ids) == list(frozen_ids) else "smoke5" if len(ids) == 5 else "debug-subset"


def build_generation_manifest(context, dataset: ORBIADataset, output: str | Path, *, ffprobe="ffprobe", ffmpeg="ffmpeg") -> dict:
    """Stage byte-identical generated MP4s and bind native requests to generation.

    Runtime artifacts may live in separate per-case directories or an external
    artifact root. Hard links (or exact byte copies across filesystems) provide
    upstream's required <case_id>.mp4 layout without decoding or encoding writes.
    """
    from draftkv.runtime.config import json_sha256
    from draftkv.benchmarks.orbia_inputs import ORBIAInterface, prepare_condition
    from draftkv.benchmarks.orbia_requests import validate_request_binding
    from draftkv.backbones.planning import _conditioning_binding
    from draftkv.backbones.contracts import BackboneError

    output = Path(output).resolve()
    generation = context.generation
    ids = list(dataset.select(generation.get("case_ids")))
    artifacts = generation.get("artifacts", [])
    if ([r.get("case_id") for r in artifacts] != ids or ids != list(context.state.case_ids)
            or generation.get("dataset_sha256") != dataset.manifest_sha256):
        raise EvaluationError("Generation selection/artifacts do not bind the frozen ORBIA dataset")
    configuration = context.config.get("backbone_config", context.config.get("backbone_adapter", {}))
    requests = configuration.get("requests", {})
    conditions, bindings, native = {}, {}, {}
    for row in artifacts:
        cid = row["case_id"]
        binding = requests.get(cid)
        if not isinstance(binding, Mapping):
            raise EvaluationError("Generation has no pinned native request: " + cid)
        path = Path(binding.get("request", ""))
        path = (path if path.is_absolute() else context.root / path).resolve()
        if sha256_file(path) != require_sha(binding.get("request_sha256"), "native request SHA256"):
            raise EvaluationError("Generation native request changed: " + cid)
        request = unique_json(path.read_bytes())
        validate_request_binding(request, context.root)
        case_row = next(row for row in dataset.manifest["cases"] if row["case_id"] == cid)
        try:
            _conditioning_binding(request, plain(case_row), context.root)
        except BackboneError as exc:
            raise EvaluationError(str(exc)) from exc
        if (request.get("case_id") != cid or request.get("backbone") != context.config["backbone"]
                or request.get("dataset_manifest_sha256") != dataset.manifest_sha256
                or request.get("conditioning_interface") != "native_camera_image_text"):
            raise EvaluationError("Generation native request identity/interface mismatch: " + cid)
        conditions[cid] = prepare_condition(dataset, cid, ORBIAInterface.CAMERA,
                                              (ORBIAInterface.CAMERA,)).provenance
        bindings[cid] = dict(request=str(path), request_sha256=binding["request_sha256"],
                             conditioning_receipt_sha256=request["conditioning_receipt_sha256"])
        native[cid] = request
    root = output / "videos"
    root.mkdir(parents=True, exist_ok=True)
    for row in artifacts:
        source = Path(row["path"]).resolve()
        if sha256_file(source) != row["sha256"]:
            raise EvaluationError("Generated video changed before staging")
        target = root / (row["case_id"] + ".mp4")
        if target.exists():
            if sha256_file(target) != row["sha256"]:
                raise EvaluationError("Staged generation video changed")
        else:
            try:
                os.link(source, target)
            except OSError:
                shutil.copyfile(source, target)
        if sha256_file(target) != row["sha256"]:
            raise EvaluationError("Generation staging changed video bytes")
    path = output / "video_manifest.json"
    if path.exists():
        doc = validate_video_manifest(dataset, path, sha256_file(path), root, ffprobe=ffprobe, ffmpeg=ffmpeg)
        if (doc.get("generation_sha256") != json_sha256(generation)
                or doc.get("native_requests") != bindings):
            raise EvaluationError("Existing manifest belongs to different generation/requests")
    else:
        checkpoint = {"source": context.config.get("checkpoint", {}).get("path", "runtime-generation"),
                      "revision": context.config.get("checkpoint", {}).get("revision") or
                                  generation.get("weight_hashes", {}).get("checkpoint"),
                      "weight_hashes": generation.get("weight_hashes", {}), "generation_git_sha": generation["git_sha"]}
        if generation.get("weight_hashes", {}).get("checkpoint"):
            checkpoint["sha256"] = generation["weight_hashes"]["checkpoint"]
        build_video_manifest(dataset, root, path,
            model=context.config["backbone"] + "_" + context.config["algorithm"], interface="camera",
            run_config={"native_requests": native, "generation_run_id": generation["run_id"],
                        "runtime_request_sha256": generation["request_sha"]},
            checkpoint=checkpoint, conditions=conditions, case_ids=ids, ffprobe=ffprobe, ffmpeg=ffmpeg)
        doc = unique_json(path.read_bytes())
        doc.update(generation_run_id=generation["run_id"], generation_sha256=json_sha256(generation),
                   generation_artifacts=artifacts, native_requests=bindings,
                   execution_scope=execution_scope(ids, dataset.case_ids), release_acceptance=False)
        path.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n")
    return dict(path=str(path), sha256=sha256_file(path), root=str(root))


def probe_video(path: Path, *, ffprobe="ffprobe", ffmpeg="ffmpeg", timeout=120) -> dict:
    """Header counts alone are insufficient: ffprobe timestamps + full ffmpeg decode.

    ORBIA officially aligns by index and ignores stored FPS. Bench2 deliberately
    additionally requires 16 FPS and constant frame PTS. No resampling is done.
    """
    command = [str(ffprobe), "-v", "error", "-select_streams", "v", "-show_streams",
               "-show_frames", "-show_entries",
               "stream=codec_type,width,height,avg_frame_rate,r_frame_rate,nb_frames,duration:"
               "frame=best_effort_timestamp_time", "-of", "json", str(path)]
    try:
        result = subprocess.run(command, capture_output=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        raise ORBIADependencyError(f"Video preflight needs working ffprobe: {exc}") from exc
    if result.returncode or result.stderr.strip():
        raise EvaluationError(f"Video ffprobe/decode failed: {path}: {result.stderr.decode(errors='replace')}")
    try:
        doc = unique_json(result.stdout)
        streams, frames = doc["streams"], doc["frames"]
        if len(streams) != 1:
            raise ValueError("exactly one video stream required")
        stream = streams[0]
        fps = float(Fraction(stream["avg_frame_rate"]))
        rate = float(Fraction(stream["r_frame_rate"]))
        pts = [float(f["best_effort_timestamp_time"]) for f in frames]
        if len(pts) != SHORT_FRAMES:
            raise ValueError(f"frame count {len(pts)}, expected 357")
        if fps != FPS or rate != FPS:
            raise ValueError("Bench2 requires exactly 16 FPS")
        if any(not math.isfinite(p) or abs(p - i / FPS) > 1e-5 for i, p in enumerate(pts)):
            raise ValueError("timestamps do not match original 16 FPS timeline starting at zero")
        if int(stream["width"]) <= 0 or int(stream["height"]) <= 0:
            raise ValueError("invalid video dimensions")
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise EvaluationError(f"ORBIA video protocol rejected {path}: {exc}") from exc
    decode = [str(ffmpeg), "-v", "error", "-xerror", "-i", str(path), "-map", "0:v:0",
              "-f", "null", "-"]
    try:
        decoded = subprocess.run(decode, capture_output=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        raise ORBIADependencyError(f"Video preflight needs working ffmpeg: {exc}") from exc
    if decoded.returncode or decoded.stderr.strip():
        raise EvaluationError(f"Video is not fully decodable: {path}: {decoded.stderr.decode(errors='replace')}")
    return {"frame_count": len(pts), "fps": fps, "first_timestamp_s": pts[0],
            "last_timestamp_s": pts[-1], "duration_s": len(pts) / fps,
            "width": stream["width"], "height": stream["height"], "fully_decoded": True}


INTERFACES = {"camera": ("world", "pose_and_action"), "action": ("world", "action_only"),
              "image_text": ("ti2v", "pose_and_action")}


def validate_run(run_config: Mapping, checkpoint: Mapping, interface: str):
    if interface not in INTERFACES or not isinstance(run_config, Mapping) or not run_config:
        raise EvaluationError("Native interface and original nonempty run_config required")
    if not isinstance(checkpoint, Mapping):
        raise EvaluationError("Original checkpoint provenance required")
    required_text(checkpoint.get("source"), "checkpoint source")
    required_text(checkpoint.get("revision"), "checkpoint revision")
    if "sha256" in checkpoint:
        require_sha(checkpoint["sha256"], "checkpoint SHA256")
    try:
        json.dumps(plain([run_config, checkpoint]), allow_nan=False)
    except (ValueError, TypeError) as exc:
        raise EvaluationError("Run/checkpoint configuration must be finite JSON") from exc


def validate_condition(record: Mapping, dataset: ORBIADataset, cid: str, interface: str):
    if (not isinstance(record, Mapping) or record.get("dataset_family") != "orbia"
            or record.get("case_id") != cid or record.get("manifest_sha256") != dataset.manifest_sha256):
        raise EvaluationError("Condition must bind the original ORBIA input; WorldAtlas rejected")
    if (record.get("interface") != interface
            or record.get("native_camera_conditioning") is not (interface == "camera")):
        raise EvaluationError("Condition native interface/camera capability claim mismatch")
    hashes = record.get("file_hashes", {})
    for name in ("case.json", "camera.npz", "prompt.txt", "q0.png"):
        if hashes.get(name) != dataset.files[f"public/{cid}/{name}"]["sha256"]:
            raise EvaluationError("Condition original input hash mismatch")
    if interface != "camera":
        helper = "src.models.camera_text.camera_prompt" if interface == "image_text" else None
        if (record.get("official_commit") != dataset.official_commit or not record.get("converted_control")
                or record.get("control_profile") != INTERFACES[interface][1]
                or not isinstance(record.get("conversion_options"), Mapping)):
            raise EvaluationError("Converted control requires official helper/options/profile provenance")
        if helper and record.get("conversion") != helper:
            raise EvaluationError("Text conditioning must use the official camera_prompt conversion")
        if interface == "action" and record.get("conversion") not in (
                "src.models.actions.block_actions", "src.models.actions.timed_actions"):
            raise EvaluationError("Action conditioning must use official action conversion")


def build_video_manifest(dataset: ORBIADataset, video_root: str | Path, output: str | Path, *,
                         model: str, interface: str, run_config: Mapping, checkpoint: Mapping,
                         conditions: Mapping[str, Mapping], case_ids: Sequence[str] | None = None,
                         ffprobe="ffprobe", ffmpeg="ffmpeg") -> dict:
    ids = dataset.select(case_ids)
    required_text(model, "model")
    if Path(model).name != model or model in (".", ".."):
        raise EvaluationError("model must be one safe path component")
    validate_run(run_config, checkpoint, interface)
    root = Path(video_root).resolve()
    output = Path(output).resolve()
    if output.is_relative_to(dataset.root):
        raise EvaluationError("Video manifests must be outside original dataset root")
    expected = {cid + ".mp4" for cid in ids}
    if {p.name for p in root.glob("*.mp4")} != expected:
        raise EvaluationError("Video directory must contain exactly selected ORBIA Case IDs; missing/extra rejected")
    if set(conditions) != set(ids):
        raise EvaluationError("Condition provenance must cover every selected case exactly")
    rows = []
    for cid in ids:
        validate_condition(conditions[cid], dataset, cid, interface)
        video = contained_file(root, cid + ".mp4")
        rows.append({"case_id": cid, "path": video.name, "sha256": sha256_file(video),
                     "video_protocol": probe_video(video, ffprobe=ffprobe, ffmpeg=ffmpeg),
                     "condition": plain(conditions[cid])})
    doc = {"schema_version": 1, "benchmark": "bench2", "dataset_family": "orbia",
           "manifest_sha256": dataset.manifest_sha256, "index_sha256": dataset.index_sha256,
           "scope": "orbia480-short" if len(ids) == 480 else "debug-subset",
           "execution_scope": execution_scope(ids, dataset.case_ids),
           "official_720_case_leaderboard": False, "model": model, "interface": interface,
           "group": INTERFACES[interface][0], "control_profile": INTERFACES[interface][1],
           "run_config": plain(run_config), "checkpoint": plain(checkpoint), "videos": rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(doc, indent=2, allow_nan=False) + "\n")
    return {"path": str(output), "sha256": sha256_file(output), "scope": doc["scope"], "cases": len(ids)}


def validate_video_manifest(dataset: ORBIADataset, path: str | Path, digest: str, video_root: str | Path,
                            *, ffprobe="ffprobe", ffmpeg="ffmpeg") -> dict:
    path, root = Path(path).resolve(), Path(video_root).resolve()
    if sha256_file(path) != require_sha(digest, "video manifest SHA256"):
        raise EvaluationError("Video manifest SHA256 mismatch")
    doc = unique_json(path.read_bytes())
    if (doc.get("schema_version") != 1 or doc.get("benchmark") != "bench2"
            or doc.get("dataset_family") != "orbia"
            or doc.get("manifest_sha256") != dataset.manifest_sha256
            or doc.get("index_sha256") != dataset.index_sha256):
        raise EvaluationError("ORBIA rejects WorldAtlas/unbound video manifest")
    rows = doc.get("videos")
    if not isinstance(rows, list) or any(not isinstance(r, dict) for r in rows):
        raise EvaluationError("Invalid ORBIA video rows")
    ids = dataset.select([r.get("case_id") for r in rows])
    scope = "orbia480-short" if len(ids) == 480 else "debug-subset"
    if doc.get("scope") != scope or doc.get("official_720_case_leaderboard") is not False:
        raise EvaluationError("Subset/full benchmark score scope mislabelled")
    if "execution_scope" in doc and doc["execution_scope"] != execution_scope(ids, dataset.case_ids):
        raise EvaluationError("Smoke/full480 execution scope mislabelled")
    interface = doc.get("interface")
    validate_run(doc.get("run_config"), doc.get("checkpoint"), interface)
    if (doc.get("group"), doc.get("control_profile")) != INTERFACES[interface]:
        raise EvaluationError("Native interface/group/control_profile mismatch")
    model = required_text(doc.get("model"), "model")
    if Path(model).name != model or model in (".", ".."):
        raise EvaluationError("Invalid model identity")
    if {p.name for p in root.glob("*.mp4")} != {cid + ".mp4" for cid in ids}:
        raise EvaluationError("Video Case ID completeness mismatch")
    for row in rows:
        if row.get("path") != row["case_id"] + ".mp4":
            raise EvaluationError("Official videos must be named <case_id>.mp4")
        video = contained_file(root, row["path"])
        if sha256_file(video) != require_sha(row.get("sha256"), "video SHA256"):
            raise EvaluationError("Generated video hash mismatch")
        actual = probe_video(video, ffprobe=ffprobe, ffmpeg=ffmpeg)
        if row.get("video_protocol") != actual:
            raise EvaluationError("Video metadata differs from decoded stream")
        validate_condition(row.get("condition"), dataset, row["case_id"], interface)
        row["resolved_video"] = str(video)
    return doc
