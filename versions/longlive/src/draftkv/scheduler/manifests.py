"""Freeze original ORBIA rows and select a shared, reproducible short pilot."""

from collections import Counter
import hashlib
from pathlib import Path
import random

from .common import atomic_json, digest, object_hash, read_json


def freeze_orbia(raw, expected_sha, output, seed, provenance):
    if digest(raw) != expected_sha:
        raise ValueError("official raw manifest hash drift")
    source = read_json(raw)
    rows = source.get("cases", [])
    ids = [r["case_id"] for r in rows]
    if len(rows) != 720 or len(set(ids)) != 720:
        raise ValueError("require original 720-case official envelope (480 short + 240 long)")
    short = [r for r in rows if r.get("frame_count") == 357]
    if len(short) != 480 or sum(r.get("frame_count") == 961 for r in rows) != 240:
        raise ValueError("ORBIA horizon counts mismatch; WorldAtlas cannot be converted")
    for row in short:
        if row.get("parent_case_id") is not None or not row.get("source") or not row.get("template"):
            raise ValueError("missing official short metadata")
        if type(row.get("anchors")) is not int or row["anchors"] < 1:
            raise ValueError("missing anchor difficulty proxy")
    if not provenance.get("verification_reference") or not provenance.get("source"):
        raise ValueError("source provenance is required")
    manifest = dict(schema="draftkv.scheduler.orbia480.v1", dataset_family="orbia",
                    benchmark="bench2", frame_count=357, timeline_fps=16,
                    official_raw_sha256=expected_sha, provenance=provenance,
                    frozen_case_ids=[r["case_id"] for r in short], cases=short)
    output = Path(output)
    if output.exists():
        raise FileExistsError("frozen manifest must be a new file")
    atomic_json(output, manifest)
    # Stable across Python hash seeds. Prefer new template, then new source, then
    # anchor complexity. Tie-break using a seeded permutation of original IDs.
    candidates = sorted(short, key=lambda r: r["case_id"])
    random.Random(seed).shuffle(candidates)
    chosen, templates, sources, complexities = [], set(), set(), set()
    for _ in range(5):
        row = max(candidates, key=lambda r: (r["template"] not in templates,
                  r["source"] not in sources, r["anchors"] not in complexities))
        chosen.append(row)
        candidates.remove(row)
        templates.add(row["template"])
        sources.add(row["source"])
        complexities.add(row["anchors"])
    selection = dict(schema="draftkv.scheduler.selection.v1", seed=seed,
                     manifest_sha256=digest(output), generation_seed=seed,
                     case_ids=[r["case_id"] for r in chosen], cases=chosen,
                     rule="seeded permutation; greedy unseen template, source, anchor count; stable ties",
                     difficulty_definition="Official template and anchor count are proxies, not official difficulty labels",
                     coverage={key: dict(Counter(str(r[key]) for r in chosen))
                               for key in ("template", "source", "anchors")},
                     omitted_templates=sorted({r["template"] for r in short} - templates),
                     omitted_sources=sorted({r["source"] for r in short} - sources))
    atomic_json(output.with_name("orbia_five_selection.json"), selection)
    return manifest, selection


def validate_manifest(config):
    manifest = read_json(config["manifest"])
    if digest(config["manifest"]) != config["manifest_sha256"]:
        raise ValueError("manifest hash drift")
    if config["benchmark"] == "bench1":
        # Resolve original frozen source schemas; no scheduler-only relabeling
        # or invented envelope can reverse the backbone's input protocol.
        from draftkv.benchmarks import Backbone, DatasetResolver, DatasetSpec
        from draftkv.registry import Registry
        group, = Registry().select("bench1", config["backbone"], config["algorithm"])
        if config.get("input_mode") != group.protocol.input_mode:
            raise ValueError("Bench1 backbone/mode binding mismatch")
        backbone = Backbone.LONGLIVE_1_3B
        dataset = DatasetResolver(Path(config["repo_root"])).resolve(DatasetSpec(
            backbone=backbone, manifest=config["manifest"], manifest_sha256=config["manifest_sha256"],
            root=config["dataset_root"]))
        selected = [case for case in dataset.cases if case.case_id == config["case_id"]]
        if len(selected) != 1:
            raise ValueError("case not in manifest")
        case = selected[0]
        row = dict(next(r for r in manifest["cases"] if r["case_id"] == case.case_id))
        row["prompt_sha256"] = row.get("source_prompt_sha256", case.prompt_sha256)
        if case.image is not None:
            row["image"] = {"path": str(case.image.relative_to(dataset.root)), "sha256": case.image_sha256}
        return row
    ids = manifest["frozen_case_ids"]
    rows = manifest["cases"]
    count = 480 if config["benchmark"] == "bench2" else 98
    if len(ids) != count or len(set(ids)) != count or [r["case_id"] for r in rows] != ids:
        raise ValueError(f"require exact ordered {count}-case frozen mapping")
    row = next((r for r in rows if r["case_id"] == config["case_id"]), None)
    if row is None:
        raise ValueError("case not in manifest")
    if config["benchmark"] == "bench2":
        if manifest.get("dataset_family") != "orbia" or any(r.get("frame_count") != 357 for r in rows):
            raise ValueError("Bench2 requires official ORBIA short cases, 357 frames")
        if config.get("expected_frames") != 357 or config.get("fps") != 16:
            raise ValueError("release output must preserve the 357-pose/16-FPS timeline")
        selection = read_json(config["selection"])
        if digest(config["selection"]) != config["selection_sha256"]:
            raise ValueError("pilot selection hash drift")
        pilots = selection["case_ids"]
        if selection["manifest_sha256"] != config["manifest_sha256"] or len(pilots) != 5 or len(set(pilots)) != 5 or not set(pilots) <= set(ids):
            raise ValueError("pilot must have exactly five IDs from this manifest")
        if config.get("pilot") is not (config["case_id"] in pilots):
            raise ValueError("pilot flag and frozen selection disagree")
        if config.get("seed") != selection.get("generation_seed"):
            raise ValueError("generation seed must match frozen paired selection")
    return row


def validate_inputs(config, row):
    assets = config.get("inputs", {})
    required = {"image", "prompt", "camera", "case"} if config["benchmark"] == "bench2" else {"prompt"}
    if config.get("input_mode") == "t_i_to_v":
        required.add("image")
    if config.get("input_mode") == "t_to_v" and "image" in assets:
        raise ValueError("T2V must not receive an image")
    if not required <= assets.keys():
        raise ValueError("missing public input assets: " + ",".join(sorted(required - assets.keys())))
    root = Path(config["dataset_root"]).resolve(strict=True)
    for name, spec in assets.items():
        p = Path(spec["path"]).resolve(strict=True)
        if root not in p.parents or digest(p) != spec["sha256"]:
            raise ValueError(f"input path/hash mismatch: {name}")
        if config["benchmark"] == "bench2" and "public" not in p.relative_to(root).parts:
            raise ValueError("generator inputs must be ORBIA public assets")
    if config["benchmark"] == "bench2":
        provenance_spec = config.get("input_provenance", {})
        if not provenance_spec.get("path") or digest(provenance_spec["path"]) != provenance_spec.get("sha256"):
            raise ValueError("pinned official input extraction provenance missing")
        provenance = read_json(provenance_spec["path"])
        frozen = read_json(config["manifest"])
        if (provenance.get("official_dataset_revision") != frozen["provenance"]["revision"]
                or provenance.get("official_raw_manifest_sha256") != frozen["official_raw_sha256"]
                or provenance.get("case_id") != config["case_id"] or provenance.get("files") != assets
                or provenance.get("archive_verified") is not True or not provenance.get("verification_reference")):
            raise ValueError("official archive extraction/input hash binding unverified")
        archive_sha = provenance.get("public_archive_sha256", "")
        if len(archive_sha) != 64 or any(c not in "0123456789abcdef" for c in archive_sha):
            raise ValueError("official public archive hash missing")
        # Inspect the official NPZ, not a converted WorldAtlas C2W file.
        import numpy as np
        from PIL import Image
        for name in required:
            if Path(assets[name]["path"]).parent.name != config["case_id"]:
                raise ValueError("input case ID path mismatch")
        with Image.open(assets["image"]["path"]) as image:
            image.verify()
        meta = read_json(assets["case"]["path"])
        if meta.get("case_id") != config["case_id"]:
            raise ValueError("official case.json ID mismatch")
        with np.load(assets["camera"]["path"], allow_pickle=False) as camera:
            poses, times = camera["poses_c2w"], camera["timestamps_s"]
            if poses.shape != (357, 4, 4) or times.shape != (357,):
                raise ValueError("camera must contain 357 poses and times")
            if not np.isfinite(poses).all() or not np.allclose(times, np.arange(357) / 16, atol=1e-5):
                raise ValueError("invalid camera timeline/poses")
        if not Path(assets["prompt"]["path"]).read_text().strip():
            raise ValueError("empty official prompt")
    else:
        prompt = Path(assets["prompt"]["path"]).read_text()
        if hashlib.sha256(prompt.encode()).hexdigest() != row["prompt_sha256"]:
            raise ValueError("Bench1 prompt bytes mismatch")
        if config.get("input_mode") == "t_i_to_v":
            image = row["image"]
            if assets["image"]["sha256"] != image["sha256"] or Path(assets["image"]["path"]).resolve() != (root / image["path"]).resolve():
                raise ValueError("Bench1 image does not match this case's frozen mapping")


def validate_release_group(config):
    """Validate the group workflow against its real frozen runtime and ORBIA bytes.

    Legacy per-case pilots keep their separate envelope validator. New releases
    queue the ordered case selection and let native per-case checkpoints resume.
    """
    from draftkv.registry import Registry
    from draftkv.runtime.config import Config, parse_devices, resolve_path
    from draftkv.benchmarks.orbia import load_lock, validate_manifest as official_manifest
    from draftkv.benchmarks.datasets import unique_json
    root = Path(config['repo_root']).resolve()
    pin = config['runtime_config']
    runtime_path = resolve_path(pin['path'], root)
    if digest(runtime_path) != pin['sha256']:
        raise ValueError('release runtime configuration hash drift')
    group, = Registry().select('bench2', config['backbone'], config['algorithm'])
    devices = parse_devices(','.join(map(str, config['devices'])))
    if len(devices) != config['sp'] or list(devices) != config['devices']:
        raise ValueError('release device/SP identity mismatch')
    resolved = Config(root, str(runtime_path)).resolve(group, devices)
    dataset = resolved.get('dataset', {})
    manifest_path = resolve_path(dataset['path'], root)
    if (digest(manifest_path) != config['manifest_sha256']
            or dataset.get('sha256') != config['manifest_sha256']):
        raise ValueError('release ORBIA manifest hash drift')
    _, ids = official_manifest(manifest_path.read_bytes(), load_lock(root/'configs/datasets/orbia480.lock.json'))
    selection = config.get('selection')
    if not isinstance(selection, dict) or set(selection) != {'case_ids', 'limit_cases'}:
        raise ValueError('release case selection must be explicit')
    chosen, limit = selection['case_ids'], selection['limit_cases']
    if chosen is not None and (not isinstance(chosen, list) or not chosen or len(set(chosen)) != len(chosen)
                               or not set(chosen) <= set(ids)):
        raise ValueError('release selection contains invalid/duplicate frozen ORBIA case IDs')
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError('positive release case limit required')
    if (object_hash(selection) != config.get('selection_sha256')
            or config.get('case_id') != 'ALL480'
            or config.get('input_mode') != group.protocol.input_mode):
        raise ValueError('release selection/protocol identity mismatch')
    return [cid for cid in ids if chosen is None or cid in chosen][:limit]
