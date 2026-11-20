"""Read the frozen text-only WorldAttention160 Bench1 schema."""
from __future__ import annotations

import hashlib
from pathlib import Path

from .datasets import (
    Backbone, DatasetError, InputMode, ResolvedScheduledCase, ResolvedDataset,
)
from .worldattention import MANIFEST_SHA256

MANIFEST_PINS = {
    Backbone.LONGLIVE_1_3B: MANIFEST_SHA256,
}
CASE_COUNTS = {Backbone.LONGLIVE_1_3B: 160}


def formal_scope(backbone):
    if Backbone(backbone.lower()) is not Backbone.LONGLIVE_1_3B:
        raise DatasetError("Bench1 supports only LongLive1.3B text-only WorldAttention160")
    return "bench1-worldattention-160"


def resolve_frozen_bench1(backbone, manifest_path, raw, root):
    """Authenticate the exact text-only 160-case source bytes."""
    backbone = Backbone(backbone)
    formal_scope(backbone.value)
    if backbone is Backbone.LONGLIVE_1_3B:
        from .worldattention import verify_manifest, DATASET_ID
        doc = verify_manifest(Path(manifest_path), raw)
        cases = tuple(ResolvedScheduledCase(row["case_id"], row["prompt"],
                      hashlib.sha256(row["prompt"].encode("utf-8")).hexdigest(), None, None, None,
                      tuple(row["prompts"]), tuple(row["timeline"]), tuple(row["prompt_events"]))
                      for row in doc["cases"])
        return ResolvedDataset("bench1", DATASET_ID, backbone, InputMode.TEXT_TO_VIDEO,
                               Path(root).resolve(), Path(manifest_path), MANIFEST_PINS[backbone], cases)
