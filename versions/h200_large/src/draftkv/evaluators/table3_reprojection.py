"""Table 3 protocol: WBench DA3 reprojection similarity only.

This module is CPU-only. It validates upstream provenance and formats
the 30 registered Bench2 group rows without inventing missing scores.
"""
from __future__ import annotations

import csv
import hashlib
import math
import re
import subprocess
from pathlib import Path

from draftkv.registry import Registry

WBENCH_COMMIT = "edc4e80e642baac4f2b5764dec9f1a491200af8f"
PROTOCOL_ID = "wbench-da3-reprojection-masked-ssim11-lpipsalex128-v2"
ALL_METRICS = ("reprojection_psnr", "reprojection_ssim", "reprojection_lpips")
HEADERS = ("Backbone", "Method", *ALL_METRICS)
NAME_MAP = {
    "native": "Native Baseline", "draftkv": "DraftKV",
    "shadowkv": "ShadowKV", "infinigen": "InfiniGen",
    "infllm": "InfLLM", "freekv": "FreeKV", "worldkv": "WorldKV",
    "future-forcing": "Future Forcing", "deep-forcing": "Deep Forcing",
    "worldtrace-landmark": "WorldTrace-LM",
}
CASE_NAME = re.compile(r"^case_(\d+)(?:_combined)?\.mp4$")
ATLAS_NAME = re.compile(r"^memory-([a-z0-9_]+)-memory_pool_30s_(\d+)\.mp4$")
GROUP_ORDER = ("LingBot1.3B", "LingBot14B", "SolarWM33B")


def verify_wbench_source(root: Path):
    """Require the exact official WBench commit and unchanged reprojection sources."""
    root = root.resolve(strict=True)
    head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                          text=True, capture_output=True)
    if head.returncode or head.stdout.strip() != WBENCH_COMMIT:
        raise ValueError(f"WBench must be pinned to {WBENCH_COMMIT}")
    files = ["src/metrics/consistency/reconstruction_consistency.py",
             "tools/run_da3_depth.py"]
    if any(not (root / f).is_file() for f in files):
        raise ValueError("WBench DA3/reprojection source is missing")
    dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--", *files],
                           text=True, capture_output=True)
    if dirty.returncode or dirty.stdout.strip():
        raise ValueError("Official WBench reprojection source is modified")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def file_identity(path: Path, *, with_hash: bool = False) -> dict:
    p = path.resolve(strict=True)
    s = p.stat()
    return {
        "path": str(p), "size": s.st_size, "mtime_ns": s.st_mtime_ns,
        "sha256": sha256(p) if with_hash else None,
    }


def parse_video(path: Path) -> str:
    """Extract a case ID; no trajectory, turn, or frame selection is inferred."""
    atlas = ATLAS_NAME.fullmatch(path.name)
    if atlas:
        return str(int(atlas.group(2)))
    ordinary = CASE_NAME.fullmatch(path.name)
    if ordinary:
        return str(int(ordinary.group(1)))
    raise ValueError(f"Unsupported video filename: {path.name}")


def discover_videos(source, *, allow_nested=True) -> dict[str, dict]:
    """Bind existing videos by case identity without applying evaluation metadata."""
    if isinstance(source, dict):
        result = {}
        for key, value in source.items():
            cid = str(int(str(key).removeprefix("case_")))
            video = Path(value).expanduser().resolve(strict=True)
            try:
                parsed = parse_video(video)
            except ValueError:
                parsed = None  # Explicit mapping supports native.mp4.
            if parsed is not None and parsed != cid:
                raise ValueError(f"Map ID {cid} conflicts with {video.name}")
            if cid in result:
                raise ValueError(f"Duplicate case ID {cid}")
            result[cid] = {"video": str(video)}
        return result
    folder = Path(source).expanduser()
    if not folder.is_dir():
        raise ValueError(f"Video directory does not exist: {folder}")
    paths = sorted(folder.rglob("*.mp4") if allow_nested else folder.glob("*.mp4"))
    result = {}
    for video in paths:
        try:
            cid = parse_video(video)
        except ValueError:
            continue
        if cid in result:
            raise ValueError(f"Duplicate videos for case {cid}: {result[cid]['video']} vs {video}")
        result[cid] = {"video": str(video.resolve())}
    if not result:
        raise ValueError(f"No supported MP4 videos found in {folder}")
    return result


def finite_score(v):
    if v is None or isinstance(v, bool):
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def summarise(cases: dict[str, dict], expected_ids: set[str], *, require_complete=True):
    """Per-case mean and actual denominator for each raw reprojection metric."""
    valid = {}
    for metric in ALL_METRICS:
        values = [finite_score((cases.get(cid) or {}).get("scores", {}).get(metric))
                  for cid in sorted(expected_ids, key=int)]
        observed = [v for v in values if v is not None]
        valid[metric] = {"mean": sum(observed) / len(observed) if observed else None,
                         "n": len(observed), "expected": len(expected_ids)}
    full = all(valid[k]["mean"] is not None
               and (not require_complete or valid[k]["n"] == len(expected_ids))
               for k in ALL_METRICS)
    return {"metrics": valid, "reprojection_complete": full,
            "expected_cases": len(expected_ids)}


def matrix_rows(group_summaries: dict) -> list[dict]:
    rows = []
    for backbone in GROUP_ORDER:
        for g in Registry().select("bench2", backbone):
            record = group_summaries.get(f"{backbone}/{g.algorithm}", {})
            report = record.get("report", {})
            means = report.get("metrics", {})
            row = {"Backbone": backbone, "Method": NAME_MAP[g.algorithm]}
            for metric in ALL_METRICS:
                row[metric] = (means.get(metric) or {}).get("mean")
            rows.append(row)
    assert len(rows) == 30
    return rows


def export_matrix(output: Path, rows: list[dict]):
    output.mkdir(parents=True, exist_ok=True)
    with (output / "table3.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADERS)
        w.writeheader()
        for row in rows:
            w.writerow({k: "" if v is None else f"{v:.6f}" if isinstance(v, float) else v
                        for k, v in row.items()})
    with (output / "table3.md").open("w") as f:
        f.write("# Table 3 — WBench Reprojection Similarity\n\n")
        f.write("Raw WBench reprojection PSNR (dB), masked SSIM, and dense-patch LPIPS. Not a WBench leaderboard score.\n\n")
        f.write("| " + " | ".join(HEADERS) + " |\n")
        f.write("|" + "|".join("---" for _ in HEADERS) + "|\n")
        for row in rows:
            f.write("| " + " | ".join("—" if row.get(k) is None else
                    f"{row[k]:.4f}" if isinstance(row.get(k), float) else
                    str(row.get(k)) for k in HEADERS) + " |\n")
