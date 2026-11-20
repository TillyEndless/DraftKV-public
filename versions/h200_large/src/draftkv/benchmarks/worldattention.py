"""Offline, deterministic WorldAttention160 input and native C3 timing contract."""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path

from .datasets import DatasetError, required_text, unique_json

SOURCE_SHA256 = "68d917083dc27696d86547e3b18ec84dbc643557ddd3b8dd590fc661c3ad3ea9"
SOURCE_FILE = "worldattention_interactive_example_160.jsonl"
MANIFEST_FILE = "worldattention160.manifest.json"
SOURCE_URL = "https://github.com/alibaba-damo-academy/WorldAttention"
# Repository URL confirmed by the locally archived upstream README. Byte pin
# identifies the supplied file; no unverified branch/commit URL is invented.
MANIFEST_SHA256 = "4126c9304b3ce435625f253a10a7c03ae83e750e35e232754d3e86d2dddc2624"
DATASET_ID = "bench1-worldattention-160-t2v"
CASE_IDS = tuple(f"WA_{i:03d}" for i in range(1, 161))


def native_schedule():
    """Nearest C3 boundary to each nominal 40-latent interval, ties round up.

    Keep native 240 latents and full VAE decode: F=4*(L-1)+1=957.
    The first decoded frame influenced by latent L>0 is 4*L-3. These are
    conditioning boundaries, not a promise of abrupt decoded scene changes.
    """
    nominal = [40 * i for i in range(1, 6)]
    actual = [3 * ((value + 1) // 3) for value in nominal]
    decoded = [4 * value - 3 for value in actual]
    return dict(policy="nearest_native_c3_no_chunk_split", latent_frames=240, chunk_latents=3,
                nominal_switch_latents=nominal, prompt_switch_frames=actual,
                latent_deviation=[a - n for a, n in zip(actual, nominal)],
                latent_deviation_seconds=[(a - n) / 4 for a, n in zip(actual, nominal)],
                first_decoded_frame_indices=decoded,
                decoded_switch_seconds=[value / 16 for value in decoded],
                decoded_deviation_seconds=[value / 16 - 10 * i for i, value in enumerate(decoded, 1)],
                native_frames=957, fps=16, native_duration_seconds=957 / 16,
                exact_nominal_timing=False)


def build_manifest(source: Path, *, source_url=SOURCE_URL):
    return _build_manifest(source.read_bytes(), source_url=source_url)


def _build_manifest(raw, *, source_url):
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise DatasetError("WorldAttention source JSONL SHA256 mismatch")
    lines = raw.splitlines(keepends=True)
    if len(lines) != 160:
        raise DatasetError("WorldAttention requires exactly 160 source rows")
    cases = []
    starts = [0, *native_schedule()["prompt_switch_frames"]]
    for index, line in enumerate(lines, 1):
        row = unique_json(line)
        prompts = row.get("prompts")
        if set(row) != {"prompts"} or not isinstance(prompts, list) or len(prompts) != 6:
            raise DatasetError("WorldAttention source row must contain six prompts only")
        for prompt in prompts:
            required_text(prompt, "WorldAttention prompt")
        cases.append(dict(case_id=CASE_IDS[index - 1], source_row_index=index,
                          source_row_sha256=hashlib.sha256(line).hexdigest(),
                          prompt=prompts[0], prompts=prompts,
                          timeline=[dict(prompt_index=i, start_seconds=10*i, end_seconds=10*(i+1),
                                         interval="[start_seconds,end_seconds)") for i in range(6)],
                          prompt_events=[dict(start_frame=start, text=text) for start, text in zip(starts, prompts)]))
    return dict(schema_version=1, benchmark="bench1", dataset_id=DATASET_ID,
                backbone="longlive1.3b", input_mode="t_to_v", count=160,
                source=dict(url=source_url, file=SOURCE_FILE, sha256=SOURCE_SHA256,
                            row_index_base=1, row_hash="SHA256 of raw JSONL line including its line ending"),
                frozen_case_ids=list(CASE_IDS), semantic_prompt_switch_count=5,
                execution_schedule=native_schedule(), cases=cases)


def manifest_bytes(source: Path, *, source_url=SOURCE_URL):
    return (json.dumps(build_manifest(source, source_url=source_url), ensure_ascii=False, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def verify_manifest(path: Path, raw: bytes | None = None):
    raw = path.read_bytes() if raw is None else raw
    if hashlib.sha256(raw).hexdigest() != MANIFEST_SHA256:
        raise DatasetError("WorldAttention manifest SHA256 differs from frozen release pin")
    _verify_source_bytes(raw, (path.parent / SOURCE_FILE).read_bytes())
    return unique_json(raw)


@lru_cache(maxsize=2)
def _verify_source_bytes(raw, source_raw):
    # Key on actual bytes, never a path/mtime. Repeated per-case planning avoids
    # reserializing 960 prompts, while every file read and pin stays checked.
    expected = (json.dumps(_build_manifest(source_raw, source_url=SOURCE_URL), ensure_ascii=False,
                           indent=2, allow_nan=False) + "\n").encode("utf-8")
    if raw != expected:
        raise DatasetError("WorldAttention manifest differs from exact source rows/timing/provenance")


def validate_schedule(request):
    """Fail closed before calling the native pipeline, independent of algorithm."""
    prompts = request.get("prompts")
    if (not isinstance(prompts, list) or len(prompts) != 6
            or any(not isinstance(p, str) or not p.strip() for p in prompts)
            or request.get("prompt_switch_frames") != native_schedule()["prompt_switch_frames"]
            or request.get("prompt_execution_schedule") != native_schedule()
            or request.get("latent_frames") != 240 or request.get("native_frames") != 957
            or request.get("frames") != 957 or request.get("fps") != 16):
        raise DatasetError("WorldAttention requires six exact prompts and the frozen native C3 schedule")
