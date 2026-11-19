"""Short-video-only ORBIA *derived* scores from immutable official outputs.

This is NOT the upstream 720-case ORBIA leaderboard Overall. The upstream
short-only T6 and Overall must remain null. Derived T6 equals the original
official scoring component `temporal.short`; Overall substitutes that value
in the unchanged official tier weights without renormalizing any weights.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from .official import EvaluationError

SCHEMA = "draftkv.orbia.short_video_derived.v1"
TIERS = ("T1", "T2", "T3", "T4", "T5", "T6")
# Exact weights from pinned ORBIA f7d8ddd src/metrics/scoring.json.
OFFICIAL_WEIGHTS = {"T1": 0.25, "T2": 0.25, "T3": 0.10,
                    "T4": 0.10, "T5": 0.10, "T6": 0.20}


def _finite_score(value, label):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvaluationError(f"Invalid ORBIA numeric score: {label}")
    score = float(value)
    if not math.isfinite(score) or not 0.0 <= score <= 100.0:
        raise EvaluationError(f"Out-of-range ORBIA numeric score: {label}")
    return score


def derive_short_video_scores(official_scores: dict, *, official_sha256: str) -> dict:
    """Derive only from the actual official scored JSON, not raw video statistics.

    Missing official short temporal components remain null. A partial tier set
    never produces an Overall. Official scores and T1..T5 are never edited.
    """
    if (not isinstance(official_scores, dict)
            or official_scores.get("protocol") != "orbia-1.0"
            or not isinstance(official_scores.get("rows"), list)
            or not isinstance(official_scores.get("components"), list)):
        raise EvaluationError("Official ORBIA score structure/version invalid")
    rows = official_scores["rows"]
    components = official_scores["components"]
    if len({r.get("model") for r in rows}) != len(rows):
        raise EvaluationError("Official ORBIA model rows not unique")
    if len(rows) != len(components):
        raise EvaluationError("Official ORBIA components/model coverage mismatch")
    component_map = {}
    for row, component in zip(rows, components):
        if not isinstance(row, dict) or not isinstance(component, dict):
            raise EvaluationError("Official ORBIA score/component row invalid")
        model = row.get("model")
        if component.get("model") not in (None, model):
            raise EvaluationError("Official ORBIA component/model mismatch")
        component_map[model] = component
    out = []
    for row in rows:
        model = row["model"]
        if row.get("T6") is not None or row.get("Overall") is not None:
            raise EvaluationError("Short-only official T6 and Overall must stay null")
        if any(t not in row for t in TIERS):
            raise EvaluationError("Official ORBIA T1..T6 coverage incomplete")
        scores = {t: _finite_score(row[t], t) for t in TIERS[:5]}
        temporal = component_map[model].get("temporal", {})
        if temporal is None:
            temporal = {}
        if not isinstance(temporal, dict):
            raise EvaluationError("Official temporal component is not a mapping")
        scores["T6"] = _finite_score(temporal.get("short"), "temporal.short")
        overall = (sum(scores[t] * weight for t, weight in OFFICIAL_WEIGHTS.items())
                   if all(scores[t] is not None for t in TIERS) else None)
        out.append(dict(model=model, group=row.get("group"),
                        cases=row.get("cases"), **scores, Overall=overall,
                        complete=overall is not None))
    return dict(schema=SCHEMA, official_scoring_version="orbia-1.0",
                official_scores_sha256=official_sha256,
                short_only=True, leaderboard_comparable=False,
                t6_source="official_scores.components[].temporal.short",
                formula="Overall = .25*T1 + .25*T2 + .10*T3 + .10*T4 + .10*T5 + .20*T6_short",
                weights=OFFICIAL_WEIGHTS, rows=out)


def calculate_from_file(official_file: str | Path) -> dict:
    official_file = Path(official_file)
    data = official_file.read_bytes()
    return derive_short_video_scores(json.loads(data),
                                    official_sha256=hashlib.sha256(data).hexdigest())


def _render_csv(doc: dict) -> str:
    import io
    stream = io.StringIO(newline="")
    fields = ["model", "group", "cases", *TIERS, "Overall", "complete"]
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows({key: r.get(key) for key in fields} for r in doc["rows"])
    return stream.getvalue()


def render_short_video_scores(official_file: str | Path, destination: str | Path) -> dict:
    """Automatic post-scoring sidecars; never touch the upstream scores folder."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    doc = calculate_from_file(official_file)
    (destination / "short_video_scores.json").write_text(
        json.dumps(doc, indent=2, allow_nan=False, ensure_ascii=False) + "\n")
    (destination / "short_video_scores.csv").write_text(_render_csv(doc))
    return doc


def verify_short_video_scores(official_file: str | Path, destination: str | Path) -> dict:
    """Byte-level verification of the derived sidecars against pinned raw scores."""
    destination = Path(destination)
    expected = calculate_from_file(official_file)
    json_path = destination / "short_video_scores.json"
    csv_path = destination / "short_video_scores.csv"
    if (not json_path.is_file() or not csv_path.is_file()
            or json.loads(json_path.read_bytes()) != expected
            or csv_path.read_text() != _render_csv(expected)):
        raise EvaluationError("Derived short-video ORBIA scores changed or are incomplete")
    return expected


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compute derived short-only ORBIA T6 and Overall")
    parser.add_argument("--official-scores", required=True, type=Path,
                        help="Upstream, unmodified official/scores/scores.json")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Independent output directory for short_video_scores.json/csv")
    args = parser.parse_args(argv)
    derived = render_short_video_scores(args.official_scores, args.output_dir)
    print(json.dumps(dict(schema=derived["schema"], output=str(args.output_dir.resolve()),
                          rows=derived["rows"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
