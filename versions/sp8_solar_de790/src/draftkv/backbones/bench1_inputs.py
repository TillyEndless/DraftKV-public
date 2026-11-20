"""Prepare frozen Bench1 native requests without importing a model or CUDA.

Native output policy is pinned to audited sources. Inputs use the frozen
WorldAttention160 text-only protocol.
"""
import argparse
import hashlib
import json
from pathlib import Path

from draftkv.registry import Group
from draftkv.runtime.config import asset_sha256
from .contracts import BackboneError
from .planning import sha256

ROOT = Path(__file__).resolve().parents[3]
MANIFEST_SHA = {
    "LongLive1.3B": "4126c9304b3ce435625f253a10a7c03ae83e750e35e232754d3e86d2dddc2624",
}


def path_value(path, root):
    path = Path(path).resolve()
    return str(path.relative_to(root)) if path.is_relative_to(root) else str(path)


def row_sha(row):
    return hashlib.sha256(json.dumps(row, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def load_policy(path, root=ROOT):
    preset = json.loads(Path(path).read_text())
    receipt_path = root / preset["provenance_receipt"]
    if sha256(receipt_path) != preset["provenance_receipt_sha256"]:
        raise BackboneError("BLOCKED_SOURCE", "native output/camera provenance receipt changed")
    receipt = json.loads(receipt_path.read_text())
    if not receipt["LongLive1.3B"].get("full_decode_method_ast_identical"):
        raise BackboneError("BLOCKED_OUTPUT_SPEC", "successful native decode method is not source verified")
    return preset, receipt, receipt_path


def prepare(backbone, manifest, runtime_dataset, output, *, native_outputs=None, case_ids=None, root=ROOT):
    root, manifest, runtime_dataset, output = map(lambda p: Path(p).resolve(),
                                                 (root, manifest, runtime_dataset, output))
    if backbone not in MANIFEST_SHA:
        raise BackboneError("BLOCKED_INPUT_PROTOCOL", "Bench1 supports only LongLive1.3B text-only WorldAttention160")
    if not manifest.is_file() or sha256(manifest) != MANIFEST_SHA[backbone]:
        raise BackboneError("BLOCKED_DATASET_BINDING", "actual frozen Bench1 manifest byte SHA mismatch")
    data = json.loads(manifest.read_text())
    if backbone == "LongLive1.3B":
        from draftkv.benchmarks.worldattention import verify_manifest
        data = verify_manifest(manifest)
    rows = data.get("cases", [])
    ids = [row["case_id"] for row in rows]
    count = 160
    if len(ids) != count or len(set(ids)) != count or data.get("selected_case_ids", ids) != ids:
        raise BackboneError("BLOCKED_DATASET_BINDING", "actual frozen backbone cases/order required")
    if case_ids:
        if len(set(case_ids)) != len(case_ids) or not set(case_ids) <= set(ids):
            raise BackboneError("BLOCKED_DATASET_BINDING", "requested case is outside frozen selection")
        rows = [row for row in rows if row["case_id"] in case_ids]
    policy_path = Path(native_outputs or root / "configs/backbones/bench1_native_outputs.json").resolve()
    preset, provenance, provenance_path = load_policy(policy_path, root)
    parameters = dict(preset["backbones"][backbone], attention_branch="conditional")
    runtime_sha = asset_sha256(runtime_dataset)
    if runtime_sha != MANIFEST_SHA[backbone]:
        raise BackboneError("BLOCKED_DATASET_BINDING", "runtime dataset differs from frozen native manifest")
    if output.exists():
        raise BackboneError("OUTPUT_EXISTS", "input preparation is immutable; choose a new output directory")
    output.mkdir(parents=True, exist_ok=False)
    requests = {}
    source_paths = [manifest, policy_path, provenance_path]
    for row in rows:
        case_id = row["case_id"]
        request = dict(parameters, execution_scope="release_candidate", case_id=case_id,
                       protocol=Group("bench1", backbone, "native").protocol.to_dict(),
                       dataset_manifest=path_value(manifest, root), dataset_manifest_sha256=MANIFEST_SHA[backbone],
                       dataset_row_sha256=row_sha(row), runtime_dataset_sha256=runtime_sha,
                       prompts=row.get("prompts", [row["prompt"]]), asset_sha256={})
        if backbone == "LongLive1.3B":
            request.update(prompt_switch_frames=data["execution_schedule"]["prompt_switch_frames"],
                           prompt_execution_schedule=data["execution_schedule"], prompt_timeline=row["timeline"])
        assets = list(source_paths)
        folder = output / "cases" / case_id
        receipt_path = folder / "conditioning_receipt.json"
        receipt = dict(schema_version=1, case_id=case_id, dataset_manifest_sha256=MANIFEST_SHA[backbone],
                       dataset_row_sha256=request["dataset_row_sha256"], native_input=dict(request),
                       source_assets=[{"path": path_value(p, root), "sha256": sha256(p)} for p in assets],
                       transformation={"kind": "protocol_adapter", "recipe": "scripts/bench1_prepare_inputs.py",
                                       "recipe_sha256": sha256(root / "scripts/bench1_prepare_inputs.py"),
                                       "native_output_source": path_value(provenance_path, root),
                                       "camera_origin": None,
                                       "image_transform": None,
                                       "prompt_transform": "six exact source prompts; nearest native C3 boundaries",
                                       "attention_branch": "conditional; locked causal_fast/distilled T2V native path, no CFG"})
        write_json(receipt_path, receipt)
        request.update(conditioning_receipt=path_value(receipt_path, root),
                       conditioning_receipt_sha256=sha256(receipt_path))
        request_path = folder / "request.json"
        write_json(request_path, request)
        requests[case_id] = {"request": path_value(request_path, root), "request_sha256": sha256(request_path)}
    preparation_receipt = dict(schema_version=1, status="READY", scope="CPU_INPUT_PREPARATION_NOT_VIDEO_E2E",
                               backbone=backbone, input_mode=parameters["input_mode"], case_ids=list(requests),
                               dataset_manifest_sha256=MANIFEST_SHA[backbone], runtime_dataset_sha256=runtime_sha,
                               output_policy=path_value(policy_path, root), output_policy_sha256=sha256(policy_path),
                               provenance_receipt=path_value(provenance_path, root), provenance_receipt_sha256=sha256(provenance_path),
                               camera_origin=None, release_e2e=False)
    write_json(output / "preparation_receipt.json", preparation_receipt)
    index = dict(schema_version=1, backbone=backbone, input_mode=parameters["input_mode"],
                 dataset_manifest_sha256=MANIFEST_SHA[backbone], runtime_dataset_sha256=runtime_sha,
                 requests=requests, preparation_receipt=path_value(output / "preparation_receipt.json", root),
                 preparation_receipt_sha256=sha256(output / "preparation_receipt.json"))
    write_json(output / "requests.json", index)
    return index


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", choices=tuple(MANIFEST_SHA), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runtime-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-outputs", type=Path)
    parser.add_argument("--case-ids", help="optional comma-separated subset; frozen order retained")
    args = parser.parse_args(argv)
    result = prepare(args.backbone, args.manifest, args.runtime_dataset, args.output,
                     native_outputs=args.native_outputs, case_ids=args.case_ids.split(",") if args.case_ids else None)
    print(json.dumps({"status": "READY", "backbone": args.backbone, "case_count": len(result["requests"]),
                      "requests_index": str(args.output / "requests.json"), "release_e2e": False}, indent=2))
    return 0
