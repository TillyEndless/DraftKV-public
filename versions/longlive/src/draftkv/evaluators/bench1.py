"""Generation-independent, resumable per-case official VBench-Long scoring."""
from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import json
import math
from pathlib import Path
import subprocess
import uuid

from draftkv.benchmarks.bench1 import MANIFEST_PINS, CASE_COUNTS, formal_scope
from draftkv.benchmarks.datasets import Backbone, sha256_file, unique_json
from draftkv.runtime.config import json_sha256
from .official import EvaluationError
from .vbench_long import VBenchLong, DIMENSIONS, canonical_sha, parse_raw_result, write_new


def verified_inputs(dataset, generation, backbone):
    """Authenticate the source and generation before invoking any evaluator."""
    try:
        model = Backbone(backbone.lower())
        pin = MANIFEST_PINS[model]
    except (ValueError, KeyError) as error:
        raise EvaluationError("Bench1 scoring supports only LongLive1.3B WorldAttention160") from error
    path = Path(dataset["manifest"])
    raw = path.read_bytes()
    import hashlib
    if hashlib.sha256(raw).hexdigest() != pin or generation.get("dataset_sha256") != pin:
        raise EvaluationError("Bench1 scoring requires its original reviewed source manifest pin")
    rows = unique_json(raw).get("cases", [])
    ids = [row.get("case_id") for row in rows if isinstance(row, dict)]
    count = CASE_COUNTS[Backbone(backbone.lower())]
    if len(ids) != count or len(set(ids)) != count or ids != dataset.get("case_ids"):
        raise EvaluationError("Bench1 scoring requires all backbone-specific frozen source IDs in original order")
    selected = generation.get("case_ids")
    videos = generation.get("artifacts")
    if (generation.get("status") != "SUCCEEDED" or not generation.get("run_id")
            or not generation.get("weight_hashes") or not isinstance(selected, list) or not selected
            or selected != [cid for cid in ids if cid in set(selected)]
            or not isinstance(videos, list) or [v.get("case_id") for v in videos] != selected):
        raise EvaluationError("Bench1 scoring requires complete ordered generation/weight provenance for the selected cases")
    seen = set()
    for video in videos:
        path = Path(video.get("path", ""))
        if (path.suffix.lower() != ".mp4" or not path.is_file() or path.stat().st_size == 0
                or sha256_file(path) != video.get("sha256") or path.resolve() in seen):
            raise EvaluationError("Bench1 generation MP4 missing, shared or SHA256 mismatch")
        seen.add(path.resolve())
    return ids, {row["case_id"]: row for row in rows}


def probe_video(path, *, ffprobe="ffprobe"):
    process = subprocess.run([ffprobe, "-v", "error", "-count_frames", "-show_streams", "-show_format",
        "-of", "json", str(path)], capture_output=True, text=True, timeout=120, check=False)
    if process.returncode:
        raise EvaluationError("Bench1 generated MP4 failed ffprobe: " + process.stderr[-600:])
    data = json.loads(process.stdout)
    streams = [s for s in data.get("streams", []) if s.get("codec_type") == "video"]
    if (len(streams) != 1 or "mp4" not in data.get("format", {}).get("format_name", "").split(",")
            or int(streams[0].get("nb_read_frames", "0")) <= 0
            or int(streams[0].get("width", 0)) <= 0 or int(streams[0].get("height", 0)) <= 0
            or float(data.get("format", {}).get("duration", 0)) <= 0):
        raise EvaluationError("Bench1 scoring requires a decodable nonempty MP4 video")
    return data


class Bench1Evaluator:
    def __init__(self, repo_root, config, *, backend=None):
        self.root = Path(repo_root).resolve()
        self.config = dict(config)
        self.backend = backend or VBenchLong(self.root, config)

    def preflight(self):
        return self.backend.preflight()

    @staticmethod
    def _receipt(receipt_path, binding):
        receipt = unique_json(receipt_path.read_bytes())
        if receipt.get("status") != "COMPLETE" or receipt.get("binding") != binding:
            raise EvaluationError("Scoring receipt has different status/input binding")
        raw = receipt["raw_result"]
        path, attempt = Path(raw["path"]), receipt_path.parent
        if not path.resolve().is_relative_to((attempt / "raw").resolve()) or sha256_file(path) != raw["sha256"]:
            raise EvaluationError("Scoring raw result changed")
        dimensions = tuple(receipt["scores"])
        from .quality import AVG6_DIMENSIONS
        if set(dimensions) not in (set(DIMENSIONS), set(AVG6_DIMENSIONS)):
            raise EvaluationError("Legacy scoring dimensions differ from Quality7/Avg6")
        scores = parse_raw_result(path, attempt / "input", dimensions)
        if scores != receipt["scores"]:
            raise EvaluationError("Scoring receipt differs from original dimension results")
        names = {"invocation.json", "process.json", "stdout.log", "stderr.log", "video_probe.json"}
        if set(receipt["evidence_sha256"]) != names:
            raise EvaluationError("Scoring receipt requires original invocation/logs/probe")
        for name, expected in receipt["evidence_sha256"].items():
            if sha256_file(attempt / name) != expected:
                raise EvaluationError("Cached scoring receipt evidence changed")
        return dict(receipt_path=str(receipt_path), receipt_sha256=sha256_file(receipt_path), **receipt)

    @staticmethod
    def _cached(case_root, binding):
        for receipt_path in sorted(case_root.glob("attempts/*/receipt.json"), reverse=True):
            try:
                return Bench1Evaluator._receipt(receipt_path, binding)
            except (OSError, ValueError, KeyError, TypeError):
                continue
        return None

    def evaluate(self, dataset, generation, backbone, output_dir, *, metadata=None,
                 mode="missing-only", dimensions=DIMENSIONS, batch_size=16, clip_mode="auto"):
        frozen_ids, source_rows = verified_inputs(dataset, generation, backbone)
        if hasattr(self.backend, "run_dimension_batch"):
            from .quality_pipeline import evaluate_quality
            result = evaluate_quality(self, generation["artifacts"], output_dir=output_dir,
                dataset_sha256=generation["dataset_sha256"], backbone=backbone,
                method=(metadata or {}).get("method", "unspecified"), generation_sha256=json_sha256(generation),
                metadata=dict(metadata or {}, source_rows_sha256=canonical_sha(source_rows)),
                scope=formal_scope(backbone) if generation["case_ids"] == frozen_ids else "exploratory-subset",
                mode=mode, dimensions=dimensions, batch_size=batch_size)
            if backbone.lower() == "longlive1.3b":
                from .segment_clip import attach_clip
                if clip_mode not in ("auto", "off"):
                    raise EvaluationError("CLIP mode must be auto or off")
                return attach_clip(self, result, generation["artifacts"], output_dir,
                                   mode=mode, enabled=clip_mode == "auto")
            return result
        # Compatibility path for legacy injected backends/receipt verification.
        if mode == "aggregate-only":
            raise EvaluationError("Legacy backend has no model-free aggregate mode")
        preflight = self.preflight()
        if preflight["status"] != "READY":
            return preflight
        metadata = dict(metadata or {})
        metadata.pop("method", None)  # Original v1 binding predates the table label.
        evaluator_identity = canonical_sha(preflight)
        identity = dict(generation_sha256=json_sha256(generation), evaluator_sha256=evaluator_identity,
                        dataset_sha256=generation["dataset_sha256"], backbone=backbone)
        output = Path(output_dir).resolve() / canonical_sha(identity)
        output.mkdir(parents=True, exist_ok=True)
        # flock is released by the OS after interruption; retries need no stale-lock deletion.
        with (output / ".lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise EvaluationError("This generation already has an active scoring process") from exc
            return self._evaluate_locked(dataset, generation, backbone, output, identity, source_rows,
                                         frozen_ids, preflight, metadata or {})

    def _evaluate_locked(self, dataset, generation, backbone, output, identity, source_rows,
                         frozen_ids, preflight, metadata):
        successes, failures = [], []
        generation_sha = json_sha256(generation)
        for video in generation["artifacts"]:
            cid = video["case_id"]
            if Path(cid).name != cid or cid in (".", "..") or "\\" in cid:
                raise EvaluationError("Unsafe scoring case ID")
            binding = dict(**identity, case_id=cid, video_sha256=video["sha256"],
                native_case_row_sha256=canonical_sha(source_rows[cid]), metadata=metadata)
            case_root = output / "cases" / cid
            cached = self._cached(case_root, binding)
            if cached:
                successes.append(cached)
                continue
            attempt = case_root / "attempts" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex)
            attempt.mkdir(parents=True)
            started = datetime.now(timezone.utc).isoformat()
            try:
                probe = probe_video(video["path"], ffprobe=self.config.get("ffprobe", "ffprobe"))
                write_new(attempt / "video_probe.json", probe)
                result = self.backend.run_case(video, attempt)
                if sha256_file(Path(video["path"])) != video["sha256"]:
                    raise EvaluationError("Generation MP4 changed during evaluation")
                receipt = dict(schema="draftkv.bench1.vbench_long.case.v1", status="COMPLETE",
                    evaluator="vbench-long", long_video_mode=True, binding=binding,
                    original_prompt=source_rows[cid]["prompt"],
                    original_prompts=source_rows[cid].get("prompts", [source_rows[cid]["prompt"]]), generation_run_id=generation["run_id"],
                    started_at=started, ended_at=datetime.now(timezone.utc).isoformat(),
                    **result, evidence_sha256={name: sha256_file(attempt / name) for name in
                        ("invocation.json", "process.json", "stdout.log", "stderr.log", "video_probe.json")})
                write_new(attempt / "receipt.json", receipt)
                successes.append(dict(receipt_path=str(attempt / "receipt.json"),
                    receipt_sha256=sha256_file(attempt / "receipt.json"), **receipt))
            except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
                failed = dict(schema="draftkv.bench1.vbench_long.case.v1", status="FAILED", binding=binding,
                    generation_run_id=generation["run_id"], started_at=started,
                    ended_at=datetime.now(timezone.utc).isoformat(), reasons=[str(exc)])
                failed["artifacts"] = {p.name: dict(path=str(p),sha256=sha256_file(p)) for p in attempt.iterdir() if p.is_file()}
                write_new(attempt / "receipt.json", failed)
                failures.append(dict(case_id=cid, receipt_path=str(attempt / "receipt.json"), reasons=[str(exc)]))
        # Do not accept results if source, dependencies, weights or input files drifted mid-run.
        postflight = self.preflight()
        if postflight != preflight or json_sha256(generation) != generation_sha:
            raise EvaluationError("VBench-Long source/runtime changed during scoring; receipts cannot be accepted")
        verified_inputs(dataset, generation, backbone)
        full = (not failures and generation["case_ids"] == frozen_ids and len(successes) == len(frozen_ids))
        means = {name: math.fsum(r["scores"][name] for r in successes) / len(frozen_ids) for name in DIMENSIONS} if full else None
        from .quality import cohort_quality_fields
        quality = cohort_quality_fields(means, formal=full, failed=bool(failures))
        summary = dict(**quality, schema="draftkv.bench1.vbench_long.summary.v1", status="FAILED" if failures else "COMPLETE",
            evaluator="vbench-long", long_video_mode=True, scope=formal_scope(backbone) if full else "exploratory-subset",
            generation_run_id=generation["run_id"], generation_sha256=generation_sha,
            dataset_sha256=generation["dataset_sha256"], evaluator_sha256=identity["evaluator_sha256"],
            frozen_case_ids=frozen_ids, requested_case_ids=generation["case_ids"], valid_scored_count=len(successes),
            formal_mean=means, scores=means, overall=None, case_receipts=[{k:r[k] for k in
                ("receipt_path", "receipt_sha256")} for r in successes], failures=failures,
            provenance=preflight, release_acceptance=False)
        path = output / ("summary-" + uuid.uuid4().hex + ".json")
        write_new(path, summary)
        return dict(summary, summary_path=str(path), summary_sha256=sha256_file(path))
