"""CPU-safe RuntimeContext hooks for the actual frozen release datasets."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from .bench1 import MANIFEST_PINS
from .datasets import Backbone, DatasetError, DatasetResolver, DatasetSpec
from .orbia import ORBIADatasetResolver, ORBIASpec
from .orbia_requests import _plain, row_sha


def resolve_dataset(context):
    config = context.config
    asset = config.get("dataset", {})
    path = Path(asset.get("path", ""))
    pin = asset.get("sha256")
    if not pin:
        raise DatasetError("Release dataset requires an externally frozen manifest SHA256")
    options = config.get("dataset_config", {})
    if config["benchmark"] == "bench1":
        backbone = {"LongLive1.3B": Backbone.LONGLIVE_1_3B}.get(config["backbone"])
        if backbone is None or pin != MANIFEST_PINS[backbone]:
            raise DatasetError("Release Bench1 requires its exact reviewed backbone manifest pin")
        if backbone is Backbone.LONGLIVE_1_3B:
            def reject_image_input(value):
                if isinstance(value, dict):
                    for key, item in value.items():
                        if key in {"image", "images", "initial_image", "image_path", "image_sha256", "conditioning_image"} and item is not None:
                            raise DatasetError("Bench1 LongLive pure T2V forbids additional image conditioning")
                        reject_image_input(item)
                elif isinstance(value, (list, tuple)):
                    for item in value:
                        reject_image_input(item)
            for name in ("native_input", "inputs", "model_config", "consumer_config", "backbone_config"):
                reject_image_input(config.get(name, {}))
        dataset = DatasetResolver(context.root).resolve(DatasetSpec(
            backbone, path, pin, options.get("root", path.parent)))
        mode = dataset.input_mode.value
        from draftkv.registry import Group
        protocol = Group("bench1", config["backbone"], "native").protocol.to_dict()
        cases = []
        for case in dataset.cases:
            row = asdict(case)
            row["image"] = str(case.image) if case.image else None
            if backbone is Backbone.LONGLIVE_1_3B:
                for key in ("prompts", "timeline", "prompt_events"):
                    row[key] = list(row[key])
            cases.append(row)
        result = dict(case_ids=list(dataset.case_ids), cases=cases,
                      manifest=str(dataset.manifest), manifest_sha256=dataset.manifest_sha256,
                      root=str(dataset.root))
    elif config["benchmark"] == "bench2":
        # dataset.path always hashes the *official manifest*, while installation
        # receipts independently bind every original public/reference asset.
        root = Path(options.get("root", path.parent))
        if path.resolve() != (root / "manifest.json").resolve():
            raise DatasetError("ORBIA dataset.path must identify dataset root's original manifest.json")
        dataset = ORBIADatasetResolver(context.root).resolve(ORBIASpec(
            root, options.get("index"), options.get("index_sha256"), options.get("stage", "inference")))
        if dataset.manifest_sha256 != pin:
            raise DatasetError("ORBIA runtime manifest differs from the configured asset pin")
        # Validate actual arrays/images/events for every case before READY.
        cases = []
        for cid in dataset.case_ids:
            case = dataset.load_case(cid)
            row = next(row for row in dataset.manifest["cases"] if row["case_id"] == cid)
            cases.append(dict(case_id=cid, prompt=case.caption, image=str(case.image_path),
                              num_frames=case.num_frames, fps=case.fps,
                              camera=str(dataset.root / "public" / cid / "camera.npz"),
                              intrinsics=case.intrinsics.tolist(), timestamps_s=case.timestamps_s.tolist(),
                              events=_plain(case.document["events"]),
                              dataset_row_sha256=row_sha(dict(row)),
                              file_hashes=dict(case.file_hashes)))
        protocol = dict(protocol_id="orbia-short-480-357", dataset_family="orbia",
                        input_mode="t_i_camera_to_v", case_count=480, target_frames=357)
        result = dict(case_ids=list(dataset.case_ids), cases=cases, root=str(dataset.root),
                      manifest=str(path), manifest_sha256=dataset.manifest_sha256,
                      index_sha256=dataset.index_sha256, data_version=dataset.data_version,
                      official_commit=dataset.official_commit)
    else:
        raise DatasetError("Default dataset hooks support only frozen release Bench1 and ORBIA")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != pin:
        raise DatasetError("Dataset manifest changed during resolution")
    source = json.loads(raw)
    selected = set(result["case_ids"])
    rows = {row["case_id"]: hashlib.sha256(json.dumps(
        row, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")).hexdigest() for row in source["cases"] if row["case_id"] in selected}
    return dict(status="READY", protocol=protocol, dataset_sha256=pin,
                native_manifest_sha256=pin, native_case_row_hashes=rows, **result)


def prepare_dataset(context):
    # Setup validates existing assets only; it never downloads or synthesizes data.
    return resolve_dataset(context)
