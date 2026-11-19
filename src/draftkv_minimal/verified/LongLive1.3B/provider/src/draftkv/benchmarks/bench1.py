"""Read frozen WorldAttention160 T2V and the unchanged LingBot14 TI98 schema."""
from __future__ import annotations

import hashlib
from pathlib import Path

from .datasets import (
    Backbone, DatasetError, InputMode, ResolvedCase, ResolvedScheduledCase, ResolvedDataset,
    contained_file, image_dimensions, require_sha, required_text, sha256_file, unique_json,
)
from .worldattention import MANIFEST_SHA256

MANIFEST_PINS = {
    Backbone.LONGLIVE_1_3B: MANIFEST_SHA256,
    Backbone.LINGBOT_14B: "0bbf701145c57c8cc3d089a7102eb1002869018652e3a317095b69025cea0b1c",
}
CASE_COUNTS = {Backbone.LONGLIVE_1_3B: 160, Backbone.LINGBOT_14B: 98}


def formal_scope(backbone):
    return "bench1-worldattention-160" if Backbone(backbone.lower()) is Backbone.LONGLIVE_1_3B else "bench1-frozen-98"


def resolve_frozen_bench1(backbone, manifest_path, raw, root):
    """Authenticate exact source bytes; keep TI's existing adapter image bytes."""
    backbone = Backbone(backbone)
    if backbone is Backbone.LONGLIVE_1_3B:
        from .worldattention import verify_manifest, DATASET_ID
        doc = verify_manifest(Path(manifest_path), raw)
        cases = tuple(ResolvedScheduledCase(row["case_id"], row["prompt"],
                      hashlib.sha256(row["prompt"].encode("utf-8")).hexdigest(), None, None, None,
                      tuple(row["prompts"]), tuple(row["timeline"]), tuple(row["prompt_events"]))
                      for row in doc["cases"])
        return ResolvedDataset("bench1", DATASET_ID, backbone, InputMode.TEXT_TO_VIDEO,
                               Path(root).resolve(), Path(manifest_path), MANIFEST_PINS[backbone], cases)
    digest = hashlib.sha256(raw).hexdigest()
    if digest != MANIFEST_PINS[backbone]:
        raise DatasetError("Bench1 TI source manifest differs from the reviewed frozen 98-case pin")
    doc = unique_json(raw)
    rows = doc.get("cases")
    if not isinstance(rows, list) or len(rows) != 98 or any(not isinstance(r, dict) for r in rows):
        raise DatasetError("Bench1 LingBot14 requires exactly 98 frozen source cases")
    ids = [required_text(r.get("case_id"), "case_id") for r in rows]
    if len(set(ids)) != 98 or any(Path(cid).name != cid or "\\" in cid or cid in (".", "..") for cid in ids):
        raise DatasetError("Bench1 source case IDs must be unique safe path components")
    if (doc.get("input_protocol") != "single_initial_image+one_static_prompt_60s"
            or doc.get("semantic_prompt_switch_count") != 0 or doc.get("selected_case_ids") != ids):
        raise DatasetError("Unreviewed Bench1 LingBot14 T+I input protocol/order")
    root = Path(root).resolve()
    if not root.is_dir():
        raise DatasetError(f"Required dataset directory does not exist: {root}")
    cases = []
    for row in rows:
        cid, prompt = row["case_id"], required_text(row.get("prompt"), "prompt")
        prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        prefix = f"cases/{cid}/input/"
        condition = unique_json(contained_file(root, prefix + "conditioning.json").read_bytes())
        name = Path(required_text(condition.get("initial_image"), "initial_image")).name
        if name != "image.png":
            raise DatasetError(f"Frozen LingBot14 adapter image must be image.png: {cid}")
        image = contained_file(root, prefix + name)
        image_sha = require_sha(row.get("adapter_sha256"), "adapter image SHA256")
        if sha256_file(image) != image_sha:
            raise DatasetError(f"Frozen LingBot14 adapter image SHA256 mismatch: {cid}")
        cases.append(ResolvedCase(cid, prompt, prompt_sha, image, image_sha, image_dimensions(image)))
    return ResolvedDataset("bench1", "bench1-frozen-98-ti", backbone, InputMode.TEXT_IMAGE_TO_VIDEO,
                           root, Path(manifest_path), digest, tuple(cases))
