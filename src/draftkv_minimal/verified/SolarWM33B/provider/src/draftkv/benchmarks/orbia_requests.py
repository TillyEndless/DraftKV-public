"""CPU-only ORBIA requests with backbone-owned native camera/length protocols.

Original public assets remain read only; deterministic transformations and
native output trimming are bound in each backbone's conditioning receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from draftkv.registry import Group
from .datasets import DatasetError, sha256_file, unique_json
from .orbia import DEFAULT_LOCK, FPS, PUBLIC_FILES, SHORT_FRAMES, ORBIADataset, ORBIADatasetResolver, ORBIASpec

ROOT = Path(__file__).resolve().parents[3]
BACKBONES = ("LingBot1.3B", "LingBot14B", "SolarWM33B")
POLICY = ROOT / "configs/datasets/orbia_native_policy.json"


def row_sha(row):
    """Same canonicalization as benchmarks.runtime.resolve_dataset."""
    return hashlib.sha256(json.dumps(_plain(row), sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def _plain(value):
    if hasattr(value, "items"):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def frame_policy(frames=SHORT_FRAMES, *, backbone=None):
    """Mirror Wan's temporal stride4, complete C4 and cached VAE decode.

    Native image2video floors ((F-1)//4+1) to C4. Decoder returns
    1 + 4*(latent-1), so the smallest legal output >=357 is 365.
    """
    if backbone == "SolarWM33B":
        from draftkv.backbones.solarwm_geometry import frame_policy as h3_policy
        return h3_policy(frames)
    if type(frames) is not int or frames < 1:
        raise DatasetError("Positive integer target frame count required")
    latent = math.ceil(((frames - 1) / 4 + 1) / 4) * 4
    decoded = 1 + 4 * (latent - 1)
    return {"target_frames": frames, "native_frames": decoded, "latent_frames": latent,
            "latent_chunk_frames": 4, "chunks": latent // 4,
            "camera_padding_frames": decoded - frames, "video_trim_frames": decoded - frames,
            "padding_policy": "hold_last_camera_trim_video",
            "native_chunk_policy": "minimum_complete_c4"}


def native_dimensions(width, height, max_area=832 * 480):
    """Exact source formula: image aspect ratio, VAE stride8, spatial patch2."""
    if any(type(v) is not int or v <= 0 for v in (width, height, max_area)):
        raise DatasetError("Positive integer image dimensions/max_area required")
    ratio = height / width
    out_h = round(math.sqrt(max_area * ratio) // 8 // 2 * 2) * 8
    out_w = round(math.sqrt(max_area / ratio) // 8 // 2 * 2) * 8
    if min(out_h, out_w) < 16:
        raise DatasetError("ORBIA image aspect ratio cannot form native patch geometry")
    return out_w, out_h


def _path(path, root):
    path = Path(path).resolve()
    return str(path.relative_to(root)) if path.is_relative_to(root) else str(path)


def _resolve(path, root):
    path = Path(path)
    return (path if path.is_absolute() else root / path).resolve()


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def _source_policy(root):
    path = root / "configs/datasets/orbia_native_policy.json"
    policy = unique_json(path.read_bytes())
    for name, pin in policy["reference_source_files"].items():
        if sha256_file(root / name) != pin:
            raise DatasetError("Native ORBIA adapter source SHA mismatch: " + name)
    return path, policy


def _camera_arrays(item):
    import numpy as np
    K = item.intrinsics
    # Native ray construction accepts [fx,fy,cx,cy], so skew cannot be dropped.
    if K[0, 1] != 0 or K[1, 0] != 0:
        raise DatasetError("LingBot native intrinsics cannot represent ORBIA camera skew")
    width, height = (item.document["intrinsics"][k] for k in ("width", "height"))
    # The locked author helper hard-codes the calibration reference832x480.
    # Normalize to this reference; the native helper then scales to its actual
    # image aspect output. Net effect is fx*out_width/source_width, etc.
    intrinsics = np.asarray([[K[0, 0] * 832 / width, K[1, 1] * 480 / height,
                              K[0, 2] * 832 / width, K[1, 2] * 480 / height]], dtype=np.float64)
    padding = frame_policy()["camera_padding_frames"]
    poses = np.concatenate((item.poses_c2w, np.repeat(item.poses_c2w[-1:], padding, axis=0)))
    times = np.arange(len(poses), dtype=np.float64) / FPS
    return poses, intrinsics, times


def prepare(backbone, dataset: ORBIADataset, output, *, case_ids=None, root=ROOT, seed=42):
    """Return an immutable request index; reruns validate and reuse exact bytes.

    ``index['requests']`` plugs into ``backbone_config.requests`` for every one
    of a backbone's seven algorithms. No algorithm identity enters the request.
    """
    import numpy as np
    if backbone not in BACKBONES or type(seed) is not int or seed < 0:
        raise DatasetError("A supported Bench2 backbone and nonnegative native seed are required")
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.is_relative_to(dataset.root.resolve()) or dataset.root.resolve().is_relative_to(output):
        raise DatasetError("Prepared requests must be outside the original ORBIA dataset root")
    ids = dataset.select(case_ids)
    if backbone == "SolarWM33B":
        policy_path = root / "configs/backbones/solarwm_source_lock.json"
        source_policy = {"native_processing": "H3 native MM-RoPE / Camera PRoPE / full_length_geometry"}
    else:
        policy_path, source_policy = _source_policy(root)
    if (output / "requests.json").exists():
        result = unique_json((output / "requests.json").read_bytes())
        if (result.get("backbone") != backbone or list(result.get("requests", {})) != list(ids)
                or result.get("dataset_manifest_sha256") != dataset.manifest_sha256
                or result.get("dataset_index_sha256") != dataset.index_sha256
                or result.get("seed") != seed):
            raise DatasetError("Existing immutable ORBIA preparation belongs to different inputs")
        for binding in result["requests"].values():
            request_path = _resolve(binding["request"], root)
            if sha256_file(request_path) != binding["request_sha256"]:
                raise DatasetError("Prepared request SHA mismatch")
            validate_request_binding(unique_json(request_path.read_bytes()), root)
        return result
    if output.exists():
        raise DatasetError("Incomplete/existing ORBIA preparation; choose a new output directory")
    # Validate all selected original files/arrays before creating any output.
    items = dataset.load_cases(ids)
    if backbone == "SolarWM33B":
        from draftkv.backbones.solarwm_inputs import arrays as prepare_arrays
    else:
        prepare_arrays = _camera_arrays
    prepared = [(item, prepare_arrays(item)) for item in items]
    output.mkdir(parents=True)
    rows = {row["case_id"]: _plain(row) for row in dataset.manifest["cases"]}
    manifest = dataset.root / "manifest.json"
    if sha256_file(manifest) != dataset.manifest_sha256:
        raise DatasetError("Original ORBIA manifest changed during preparation")
    requests = {}
    for item, (poses, intrinsics, times) in prepared:
        folder = output / "cases" / item.case_id
        camera = folder / "camera"
        camera.mkdir(parents=True)
        for name, array in (("poses", poses), ("intrinsics", intrinsics), ("timestamps_s", times)):
            np.save(camera / (name + ".npy"), array, allow_pickle=False)
        dimensions = item.document["intrinsics"]
        out_w, out_h = native_dimensions(dimensions["width"], dimensions["height"])
        request = {"schema_version": 1, "execution_scope": "release_candidate", "backbone": backbone,
                   "case_id": item.case_id, "protocol": Group("bench2", backbone, "native").protocol.to_dict(),
                   "input_mode": "orbia", "conditioning_interface": "native_camera_image_text",
                   "dataset_manifest": _path(manifest, root), "dataset_manifest_sha256": dataset.manifest_sha256,
                   "runtime_dataset_sha256": dataset.manifest_sha256, "dataset_index_sha256": dataset.index_sha256,
                   "dataset_row_sha256": row_sha(rows[item.case_id]), "dataset_root": _path(dataset.root, root),
                   "official_commit": dataset.official_commit, "data_version": dataset.data_version,
                   "prompts": [item.caption], "image": _path(item.image_path, root),
                   "image_conditioning": "official_q0", "frames": SHORT_FRAMES, "fps": FPS, "seed": seed,
                   "width": 832, "height": 480, "output_width": out_w, "output_height": out_h,
                   "source_width": dimensions["width"], "source_height": dimensions["height"],
                   "native_frames": frame_policy()["native_frames"], "latent_frames": frame_policy()["latent_frames"],
                   "padding_policy": "hold_last_camera_trim_video", "native_chunk_policy": "minimum_complete_c4",
                   "frame_policy": frame_policy(), "num_frame_per_block": 4,
                   "sample_shift": 5.0 if backbone == "LingBot1.3B" else 10.0, "offload_model": True,
                   "local_attn_size": 18, "sink_size": 6, "attention_branch": "conditional",
                   "camera_dir": _path(camera, root), "camera_coordinate_system": "opencv_c2w",
                   "camera_provenance": "frozen_manifest", "camera_policy": "official_c2w_hold_last_tail",
                   "original_file_sha256": dict(item.file_hashes),
                   "original_timestamps_s": item.timestamps_s.tolist(), "original_events": _plain(item.events),
                   "original_intrinsics_K": item.intrinsics.tolist(),
                   "asset_sha256": {"image": item.file_hashes["q0.png"],
                                    "poses": sha256_file(camera / "poses.npy"),
                                    "intrinsics": sha256_file(camera / "intrinsics.npy")}}
        if backbone == "SolarWM33B":
            from draftkv.backbones.solarwm_inputs import configure
            configure(request)
        assets = [manifest, policy_path, Path(__file__), item.image_path]
        if backbone == "SolarWM33B":
            assets += [root / "src/draftkv/backbones/solarwm_inputs.py",
                       root / "src/draftkv/backbones/solarwm_geometry.py",
                       root / "configs/backbones/solarwm_runtime_lock.json"]
        assets += [dataset.root / "public" / item.case_id / name for name in PUBLIC_FILES if name != "q0.png"]
        assets += [camera / (name + ".npy") for name in ("poses", "intrinsics", "timestamps_s")]
        receipt = {"schema_version": 1, "case_id": item.case_id,
                   "dataset_manifest_sha256": dataset.manifest_sha256,
                   "dataset_row_sha256": request["dataset_row_sha256"], "native_input": dict(request),
                   "source_assets": [{"path": _path(p, root), "sha256": sha256_file(p)} for p in assets],
                   "transformation": {"recipe": "draftkv.benchmarks.orbia_requests.prepare",
                       "recipe_sha256": sha256_file(Path(__file__)), "kind": "official_camera_layout_adapter",
                       "prompt": "exact UTF-8 prompt.txt bytes including terminal newline", "image": "none; exact q0.png",
                       "poses": "original OpenCV C2W without rebase; append final pose eight times",
                       "intrinsics": "K -> [fx*832/source_width,fy*480/source_height,cx*832/source_width,cy*480/source_height]",
                       "native_processing": source_policy["native_processing"], "frame_policy": frame_policy()}}
        if backbone == "SolarWM33B":
            receipt["transformation"].update(
                recipe="draftkv.backbones.solarwm_inputs",
                recipe_sha256=sha256_file(root / "src/draftkv/backbones/solarwm_inputs.py"),
                image="native PIL LANCZOS stretch1344x768 + native H3 VisualVAE",
                poses="native first-frame-relative W2C; native C5 cadence; hold final selected camera",
                intrinsics="source normalized K; native checkpoint logd4/fixed-K PRoPE",
                native_processing="SolarWM native joint Qwen image/text, MM-RoPE and Camera PRoPE",
                frame_policy=frame_policy(backbone=backbone))
        receipt_path = folder / "conditioning_receipt.json"
        _write(receipt_path, receipt)
        request.update(conditioning_receipt=_path(receipt_path, root),
                       conditioning_receipt_sha256=sha256_file(receipt_path))
        request_path = folder / "request.json"
        _write(request_path, request)
        requests[item.case_id] = {"request": _path(request_path, root), "request_sha256": sha256_file(request_path)}
    preparation = {"schema_version": 1, "status": "READY", "scope": "CPU_INPUT_PREPARATION_NOT_VIDEO_E2E",
                   "backbone": backbone, "input_mode": "orbia", "case_ids": list(ids), "frame_policy": frame_policy(backbone=backbone),
                   "dataset_manifest_sha256": dataset.manifest_sha256, "dataset_index_sha256": dataset.index_sha256,
                   "runtime_dataset_sha256": dataset.manifest_sha256,
                   "source_policy_sha256": sha256_file(policy_path), "release_e2e": False}
    _write(output / "preparation_receipt.json", preparation)
    index = {"schema_version": 1, "backbone": backbone, "input_mode": "orbia", "seed": seed,
             "dataset_manifest_sha256": dataset.manifest_sha256, "runtime_dataset_sha256": dataset.manifest_sha256,
             "dataset_index_sha256": dataset.index_sha256, "requests": requests,
             "preparation_receipt": _path(output / "preparation_receipt.json", root),
             "preparation_receipt_sha256": sha256_file(output / "preparation_receipt.json")}
    _write(output / "requests.json", index)
    return index


def validate_request_binding(request, root=ROOT):
    """Re-read original files and check the actual adapted arrays on CPU.

    Called by the native planner as well as immutable preparation reuse. This
    complements the planner's manifest row/receipt byte checks.
    """
    import numpy as np
    root = Path(root).resolve()
    if request.get("backbone") not in BACKBONES or request.get("input_mode") != "orbia":
        raise DatasetError("ORBIA request backbone/input interface mismatch")
    cid = request.get("case_id")
    if not isinstance(cid, str) or Path(cid).name != cid or cid in (".", "..") or "\\" in cid:
        raise DatasetError("Unsafe ORBIA request case ID")
    source_root = _resolve(request["dataset_root"], root)
    source = source_root / "public" / cid
    expected_hashes = request.get("original_file_sha256", {})
    for name in PUBLIC_FILES:
        if sha256_file(source / name) != expected_hashes.get(name):
            raise DatasetError("Original ORBIA file SHA mismatch: " + name)
    manifest_path = _resolve(request["dataset_manifest"], root)
    if manifest_path != source_root / "manifest.json" or sha256_file(manifest_path) != request.get("dataset_manifest_sha256"):
        raise DatasetError("Original ORBIA manifest binding mismatch")
    rows = [r for r in unique_json(manifest_path.read_bytes())["cases"] if r["case_id"] == cid]
    if len(rows) != 1 or row_sha(rows[0]) != request.get("dataset_row_sha256"):
        raise DatasetError("Original ORBIA row binding mismatch")
    doc = unique_json((source / "case.json").read_bytes())
    if request.get("prompts") != [(source / "prompt.txt").read_bytes().decode("utf-8")]:
        raise DatasetError("ORBIA original prompt changed")
    if (_resolve(request["image"], root) != (source / "q0.png").resolve()
            or request.get("asset_sha256", {}).get("image") != expected_hashes.get("q0.png")):
        raise DatasetError("ORBIA request must use the exact original q0.png")
    if request["backbone"] == "SolarWM33B":
        from draftkv.backbones.solarwm_inputs import validate
        return validate(request, source, root)
    policy = frame_policy()
    if (request.get("frames"), request.get("native_frames"), request.get("latent_frames"), request.get("fps")) != (357, 365, 92, 16):
        raise DatasetError("ORBIA requires minimal native C4 padding365/latent92/trim357 at16FPS")
    if request.get("frame_policy") != policy or request.get("padding_policy") != policy["padding_policy"]:
        raise DatasetError("ORBIA padding/trim provenance changed")
    with np.load(source / "camera.npz", allow_pickle=False) as archive:
        original, times = archive["poses_c2w"], archive["timestamps_s"]
    camera = _resolve(request["camera_dir"], root)
    poses = np.load(camera / "poses.npy", allow_pickle=False)
    K_native = np.load(camera / "intrinsics.npy", allow_pickle=False)
    padded_times = np.load(camera / "timestamps_s.npy", allow_pickle=False)
    K = np.asarray(doc["intrinsics"]["K"], dtype=np.float64)
    width, height = (doc["intrinsics"][k] for k in ("width", "height"))
    expected = np.asarray([[K[0, 0] * 832 / width, K[1, 1] * 480 / height,
                            K[0, 2] * 832 / width, K[1, 2] * 480 / height]])
    if (original.shape != (357, 4, 4) or poses.shape != (365, 4, 4)
            or not np.array_equal(poses[:357], original)
            or not np.array_equal(poses[357:], np.repeat(original[-1:], 8, axis=0))
            or K_native.shape != (1, 4) or not np.array_equal(K_native, expected)
            or not np.array_equal(times, np.asarray(request.get("original_timestamps_s")))
            or not np.array_equal(padded_times, np.arange(365) / 16)
            or request.get("original_intrinsics_K") != K.tolist()
            or request.get("original_events") != doc["events"]):
        raise DatasetError("ORBIA native camera/intrinsics/timestamps differ from original source policy")
    if (request.get("output_width"), request.get("output_height")) != native_dimensions(width, height):
        raise DatasetError("ORBIA output dimensions differ from native image aspect policy")
    for name in ("poses", "intrinsics"):
        if sha256_file(camera / (name + ".npy")) != request.get("asset_sha256", {}).get(name):
            raise DatasetError("ORBIA prepared camera SHA mismatch: " + name)
    return True


def bind_inputs(settings, group, root=ROOT):
    """Shared release configurator adapter with real DatasetResolver validation."""
    import copy
    from draftkv.backbones.planning import resolve_request
    from draftkv.runtime.config import ConfigurationError
    root = Path(root).resolve()
    allowed = {"dataset", "dataset_config", "checkpoint", "weights", "requests_index",
               "backbone_config", "model_config", "consumer_config"}
    if set(settings) - allowed:
        raise ConfigurationError("ORBIA backbone settings contain protected/unknown keys")
    asset, options = settings["dataset"], settings.get("dataset_config", {})
    manifest = _resolve(asset["path"], root)
    dataset = ORBIADatasetResolver(root).resolve(ORBIASpec(
        _resolve(options.get("root", manifest.parent), root),
        _resolve(options["index"], root), options["index_sha256"], options.get("stage", "inference")))
    if manifest != dataset.root / "manifest.json" or asset.get("sha256") != dataset.manifest_sha256:
        raise ConfigurationError("ORBIA deployment dataset does not identify the frozen original manifest")
    index_spec = settings.pop("requests_index")
    index_path = _resolve(index_spec["path"], root)
    if sha256_file(index_path) != index_spec.get("sha256"):
        raise ConfigurationError("ORBIA native requests index SHA mismatch")
    index = unique_json(index_path.read_bytes())
    requests = index.get("requests")
    if (index.get("schema_version") != 1 or index.get("input_mode") != "orbia"
            or index.get("backbone") != group.backbone or not isinstance(requests, dict)
            or list(requests) != list(dataset.case_ids)
            or index.get("dataset_manifest_sha256") != dataset.manifest_sha256
            or index.get("runtime_dataset_sha256") != dataset.manifest_sha256
            or index.get("dataset_index_sha256") != dataset.index_sha256):
        raise ConfigurationError("ORBIA native index differs from frozen480/backbone/dataset/index")
    receipt_path = _resolve(index["preparation_receipt"], root)
    if sha256_file(receipt_path) != index.get("preparation_receipt_sha256"):
        raise ConfigurationError("ORBIA preparation receipt SHA mismatch")
    receipt = unique_json(receipt_path.read_bytes())
    if (receipt.get("status") != "READY" or receipt.get("backbone") != group.backbone
            or receipt.get("case_ids") != list(dataset.case_ids)
            or receipt.get("dataset_manifest_sha256") != dataset.manifest_sha256
            or receipt.get("dataset_index_sha256") != dataset.index_sha256
            or receipt.get("frame_policy") != frame_policy(backbone=group.backbone)):
        raise ConfigurationError("ORBIA preparation receipt differs from frozen source")
    for cid, binding in requests.items():
        if not isinstance(binding, dict) or set(binding) != {"request", "request_sha256"}:
            raise ConfigurationError("Invalid ORBIA native request binding: " + cid)
        request = resolve_request(binding, root, group)
        if (request.get("execution_scope") != "release_candidate" or request.get("case_id") != cid
                or request.get("backbone") != group.backbone
                or request.get("dataset_index_sha256") != dataset.index_sha256
                or request.get("runtime_dataset_sha256") != dataset.manifest_sha256):
            raise ConfigurationError("ORBIA native request differs from release source: " + cid)
        for name in PUBLIC_FILES:
            if request["original_file_sha256"].get(name) != dataset.files[f"public/{cid}/{name}"]["sha256"]:
                raise ConfigurationError("ORBIA native request differs from installation file index")
        validate_request_binding(request, root)
    config = settings.setdefault("backbone_config", {})
    protected = {"requests", "request", "request_sha256", "algorithm_plugin", "protocol", "case_id",
                 "dataset_manifest", "dataset_manifest_sha256", "dataset_row_sha256", "runtime_dataset_sha256",
                 "execution_scope", "input_mode", "prompts", "image", "camera_dir", "asset_sha256",
                 "conditioning_receipt", "conditioning_receipt_sha256", "frames", "native_frames", "latent_frames",
                 "width", "height", "output_width", "output_height", "fps", "seed"}
    if not isinstance(config, dict) or set(config) & protected:
        raise ConfigurationError("ORBIA backbone_config cannot override frozen inputs or algorithm plugin")
    config["requests"] = copy.deepcopy(requests)
    return settings


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", choices=BACKBONES, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--index-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-ids")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    args = parser.parse_args(argv)
    dataset = ORBIADatasetResolver(ROOT, lock=args.lock).resolve(ORBIASpec(
        args.dataset_root, args.index, args.index_sha256))
    result = prepare(args.backbone, dataset, args.output,
                     case_ids=args.case_ids.split(",") if args.case_ids else None, seed=args.seed)
    print(json.dumps({"status": "READY", "case_count": len(result["requests"]),
                      "requests_index": str(args.output / "requests.json"), "release_e2e": False}, indent=2))
    return 0
