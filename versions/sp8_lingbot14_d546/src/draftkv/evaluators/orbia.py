"""Execute the pinned ORBIA evaluate.py, preserving its complete native outputs.

This module never calculates or renormalizes a metric. Short-only T6/Overall
remain null exactly as upstream computes them. Calling preflight never scores.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path

from draftkv.benchmarks.datasets import DatasetError, sha256_file, unique_json
from draftkv.benchmarks.orbia import (
    DEFAULT_LOCK, ORBIADataset, ORBIADependencyError, load_lock,
)
from draftkv.benchmarks.orbia_source import REFERENCE_SCRIPT, verify_source
from .official import EvaluationError, EvaluationReport, Status
from .orbia_video import validate_video_manifest
from .orbia_short import render_short_video_scores, verify_short_video_scores

TIERS = ("T1", "T2", "T3", "T4", "T5", "T6")


def cpu_reference_preflight(dataset: ORBIADataset, official_root: str | Path, *,
                            python: str = sys.executable, case_ids=None,
                            lock: str | Path = DEFAULT_LOCK) -> EvaluationReport:
    """Decode registered RGB-D, masks, geometry and anchors with official code.

    Requires numpy and Pillow only; deliberately independent of CUDA and metric
    weights. Reference/source hashes are checked before this external process.
    """
    from draftkv.benchmarks.orbia import reference_files
    provenance = {"metrics": None}
    try:
        source_lock = load_lock(lock)
        source = verify_source(official_root, source_lock)
        ids = dataset.select(case_ids)
        if dataset.stage != "references":
            raise ORBIADependencyError("Official reference assets required; install references stage")
        for cid in ids:
            dataset.load_case(cid)
            prefix = f"references/{cid}/"
            ref = unique_json(dataset._checked(prefix + "reference.json").read_bytes())
            for name in reference_files(ref):
                dataset._checked(prefix + name)
        result = subprocess.run([str(python), "-c", REFERENCE_SCRIPT, str(dataset.root), json.dumps(ids)],
                                cwd=source, capture_output=True, timeout=300, check=False)
        provenance.update(official_commit=source_lock["official_commit"], case_ids=list(ids),
                          returncode=result.returncode, stdout=result.stdout.decode(errors="replace"),
                          stderr=result.stderr.decode(errors="replace"))
        if result.returncode:
            if result.returncode == 86 or "ModuleNotFoundError" in provenance["stderr"] or "ImportError" in provenance["stderr"]:
                raise ORBIADependencyError("Official CPU loader dependency missing: " + provenance["stderr"])
            raise EvaluationError("Official reference CPU preflight failed: " + provenance["stderr"])
        return EvaluationReport(Status.READY, provenance=provenance)
    except (ORBIADependencyError, OSError, subprocess.SubprocessError) as exc:
        return EvaluationReport(Status.BLOCKED_DEPENDENCY, (str(exc),), provenance)

# Based on pinned requirements/{evaluation-constraints,sam}.txt and actual
# worker imports. Probe in the requested official interpreter, never the host.
DEPENDENCY_SCRIPT = r'''
import importlib, importlib.metadata, json, pathlib, sys
reasons, versions = [], {}
pins = json.loads(sys.argv[2])
for name, expected in pins.items():
    try:
        actual = importlib.metadata.version(name)
        versions[name] = actual
        # Official CUDA wheels use PEP440 local versions such as +cu128.
        if expected and actual.split('+', 1)[0] != expected:
            reasons.append(name + ': expected ' + expected + ', found ' + actual)
    except importlib.metadata.PackageNotFoundError:
        reasons.append('missing official dependency: ' + name)
for module in ('numpy','PIL','yaml','cv2','scipy','torch','pyiqa','lpips','gsplat','open3d','decord'):
    try: importlib.import_module(module)
    except Exception as e: reasons.append(module + ': ' + str(e))
try:
    import torch
    if not torch.cuda.is_available(): reasons.append('official full evaluator requires CUDA')
    if str(torch.version.cuda) != '12.8': reasons.append('official environment requires CUDA 12.8 torch build')
except Exception: pass
try:
    import numpy
    major, minor = map(int, numpy.__version__.split('.')[:2])
    if major != 1 or minor < 26: reasons.append('official numpy constraint is >=1.26,<2')
except Exception: pass
try:
    import yaml
    doc = yaml.safe_load(pathlib.Path(sys.argv[1]).read_text())
    template = yaml.safe_load(pathlib.Path('configs/evaluation.yaml').read_text())
    for section in ('evaluator','quality'):
        # Optional worker interpreters are None in the official generated config;
        # contract.py interprets that as this same interpreter. Include paths
        # added by download_weights.py, not only string-valued template defaults.
        for key in set(template.get(section,{})) | set(doc.get(section,{})):
            if key.endswith(('_repo','_root','_checkpoint','_config','_python')) or key == 'torch_home':
                configured = doc.get(section,{}).get(key)
                if key.endswith('_python') and configured is None:
                    continue
                if not isinstance(configured,str) or not pathlib.Path(configured).is_absolute() or not pathlib.Path(configured).exists():
                    reasons.append('missing absolute official config asset: '+section+'.'+key)
except Exception as e: reasons.append('official evaluation config: '+str(e))
print(json.dumps({'reasons':reasons, 'versions':versions, 'python':sys.version}))
'''


@dataclass(frozen=True)
class ORBIAVideos:
    manifest: str | Path
    manifest_sha256: str
    root: str | Path


@dataclass(frozen=True)
class ORBIAEvaluatorSpec:
    official_root: str | Path
    config: str | Path
    python: str = sys.executable
    devices: str = "0"
    threads: int = 4
    tiers: tuple[str, ...] = TIERS
    stage: str = "full"
    geometry_root: str | Path | None = None
    installation_receipt: str | Path | None = None
    installation_receipt_sha256: str | None = None


class ORBIAOfficialEvaluator:
    def __init__(self, spec: ORBIAEvaluatorSpec, *, lock: str | Path = DEFAULT_LOCK,
                 ffprobe="ffprobe", ffmpeg="ffmpeg"):
        self.spec = spec
        self.lock_path = lock
        self.lock = load_lock(lock)
        self.ffprobe, self.ffmpeg = ffprobe, ffmpeg
        if not spec.tiers or len(set(spec.tiers)) != len(spec.tiers) or set(spec.tiers) - set(TIERS):
            raise EvaluationError("tiers must be unique official T1..T6 names")
        if spec.stage not in ("full", "geometry", "precompute"):
            raise EvaluationError("Invalid official evaluation stage")
        devices = spec.devices.split(",")
        if not devices or any(not (d.isdigit() or re.fullmatch(r"GPU-[a-zA-Z0-9-]+", d)) for d in devices) or len(set(devices)) != len(devices):
            raise EvaluationError("devices must contain distinct GPU IDs or inventory-bound UUIDs")
        if type(spec.threads) is not int or spec.threads < 1:
            raise EvaluationError("threads must be a positive integer")

    def command(self, dataset: ORBIADataset, official_manifest: Path, output: Path) -> list[str]:
        spec = self.spec
        command = [str(spec.python), str(Path(spec.official_root).resolve() / "evaluate.py"),
                   "--videos", str(official_manifest.resolve()), "--dataset", str(dataset.root),
                   "--config", str(Path(spec.config).resolve()), "--output", str(output.resolve()),
                   "--devices", spec.devices, "--threads", str(spec.threads), "--stage", spec.stage,
                   "--tiers", *spec.tiers]
        if spec.geometry_root is not None:
            command.extend(["--geometry-root", str(Path(spec.geometry_root).resolve())])
        return command

    def _dependency_probe(self, source: Path) -> dict:
        # Exact pins come from audited upstream requirements; non-pinned helper
        # packages remain presence checks and all installed versions are saved.
        pins = {"torch": "2.10.0", "torchvision": "0.25.0", "xformers": "0.0.34",
                "transformers": "4.45.2", "pyiqa": "0.1.15", "numpy": None,
                "Pillow": None, "PyYAML": None, "scipy": None, "lpips": None,
                "open3d": "0.19.0", "decord": "0.6.0", "gsplat": None,
                "peft": "0.10.0", "trl": "0.8.6", "accelerate": "1.8.0",
                "datasets": "3.6.0", "diffusers": "0.33.1", "deepspeed": "0.15.4",
                "qwen-vl-utils": "0.0.11"}
        result = subprocess.run([str(self.spec.python), "-c", DEPENDENCY_SCRIPT,
                                 str(Path(self.spec.config).resolve()), json.dumps(pins)],
                                cwd=source, capture_output=True, timeout=120, check=False)
        if result.returncode:
            raise ORBIADependencyError("Official dependency probe failed: " + result.stderr.decode(errors="replace"))
        try:
            probe = unique_json(result.stdout.splitlines()[-1])
            if not isinstance(probe.get("reasons"), list) or not isinstance(probe.get("versions"), dict):
                raise DatasetError("invalid dependency probe schema")
        except (IndexError, DatasetError) as exc:
            raise ORBIADependencyError("Official dependency probe returned no valid diagnostic JSON") from exc
        probe.update(stdout=result.stdout.decode(errors="replace"), stderr=result.stderr.decode(errors="replace"))
        return probe

    def _check(self, dataset: ORBIADataset, videos: ORBIAVideos) -> tuple[EvaluationReport, dict | None]:
        provenance = {"evaluator": "orbia", "official_commit": self.lock["official_commit"],
                      "scoring_version": self.lock["scoring_version"],
                      "dataset_manifest_sha256": dataset.manifest_sha256,
                      "dataset_index_sha256": dataset.index_sha256,
                      "dataset_root": str(dataset.root),
                      "data_version": dataset.data_version, "requested_tiers": list(self.spec.tiers),
                      "stage": self.spec.stage, "official_720_case_leaderboard": False,
                      "short_only_limitation": "Official scoring leaves T6 and Overall null without long cases"}
        try:
            provenance.update(self.verify_installation())
            doc = validate_video_manifest(dataset, videos.manifest, videos.manifest_sha256, videos.root,
                                          ffprobe=self.ffprobe, ffmpeg=self.ffmpeg)
            provenance.update(scope=doc["scope"], execution_scope=doc.get("execution_scope", "debug-subset"),
                              release_acceptance=False,
                              generation_sha256=doc.get("generation_sha256"),
                              generation_run_id=doc.get("generation_run_id"),
                              case_ids=[r["case_id"] for r in doc["videos"]],
                              model=doc["model"], group=doc["group"], control_profile=doc["control_profile"],
                              video_manifest_sha256=videos.manifest_sha256,
                              original_run_config=doc["run_config"], checkpoint=doc["checkpoint"])
            if dataset.stage != "references":
                raise ORBIADependencyError("Official reference assets required; install references stage")
            source = verify_source(self.spec.official_root, self.lock)
            provenance["source_files"] = self.lock["source_files"]
            cpu = cpu_reference_preflight(dataset, source, python=self.spec.python,
                case_ids=provenance["case_ids"], lock=self.lock_path)
            provenance["reference_preflight"] = cpu.as_dict()
            if cpu.status != Status.READY:
                return EvaluationReport(cpu.status, cpu.reasons, provenance), doc
            config = Path(self.spec.config).resolve()
            if not config.is_file():
                raise ORBIADependencyError(f"Official evaluation.local.yaml missing: {config}")
            provenance["evaluator_config_sha256"] = sha256_file(config)
            probe = self._dependency_probe(source)
            provenance["dependencies"] = probe
            if probe["reasons"]:
                return EvaluationReport(Status.BLOCKED_DEPENDENCY, tuple(probe["reasons"]), provenance), doc
            return EvaluationReport(Status.READY, provenance=provenance), doc
        except ORBIADependencyError as exc:
            return EvaluationReport(Status.BLOCKED_DEPENDENCY, (str(exc),), provenance), None
        except (OSError, subprocess.SubprocessError) as exc:
            return EvaluationReport(Status.BLOCKED_DEPENDENCY, (str(exc),), provenance), None

    def verify_installation(self):
        """Rehash installed official weights/source files before any model work."""
        if self.spec.installation_receipt is None:
            return {}
        receipt_path = Path(self.spec.installation_receipt).resolve()
        if sha256_file(receipt_path) != self.spec.installation_receipt_sha256:
            raise EvaluationError("Official evaluator installation receipt changed")
        receipt = unique_json(receipt_path.read_bytes())
        if (receipt.get("schema") != "draftkv.bench2.orbia.install.v1"
                or receipt.get("official_commit") != self.lock["official_commit"]
                or receipt.get("official_source_files") != self.lock["source_files"]
                or not receipt.get("files")):
            raise EvaluationError("Official evaluator installation receipt source mismatch")
        for name, digest in receipt["files"].items():
            path = (receipt_path.parent / name).resolve()
            if not path.is_relative_to(receipt_path.parent) or sha256_file(path) != digest:
                raise EvaluationError("Official evaluator installed asset SHA mismatch: " + name)
        return {"installation_receipt_sha256": self.spec.installation_receipt_sha256}

    def preflight(self, dataset: ORBIADataset, videos: ORBIAVideos) -> EvaluationReport:
        return self._check(dataset, videos)[0]

    def archived_report(self, videos: ORBIAVideos, output: str | Path) -> dict:
        """Verify the immutable official outputs before reuse or workflow audit."""
        output = Path(output).resolve()
        report = unique_json((output / "evaluation_report.json").read_bytes())
        provenance = report.get("provenance", {})
        if report.get("status") != "COMPLETE" or not provenance.get("output_files"):
            raise EvaluationError("No complete archived official evaluation")
        actual_files = {str(p.relative_to(output)) for p in output.rglob("*")
                        if p.is_file() and p != output / "evaluation_report.json"}
        if actual_files != set(provenance["output_files"]):
            raise EvaluationError("Archived official output inventory changed")
        for name, digest in provenance["output_files"].items():
            path = (output / name).resolve()
            if not path.is_relative_to(output) or sha256_file(path) != digest:
                raise EvaluationError("Archived official output SHA mismatch: " + name)
        if (provenance.get("video_manifest_sha256") != videos.manifest_sha256
                or sha256_file(output / "video_manifest.json") != videos.manifest_sha256
                or sha256_file(output / "evaluation.local.yaml") != sha256_file(Path(self.spec.config))
                or provenance.get("official_commit") != self.lock["official_commit"]
                or provenance.get("requested_tiers") != list(self.spec.tiers)
                or provenance.get("stage") != self.spec.stage):
            raise EvaluationError("Archived official evaluation input/config/source mismatch")
        doc = unique_json(Path(videos.manifest).read_bytes())
        rows = [{"model": doc["model"], "case_id": r["case_id"],
                 "video": str((Path(videos.root) / r["path"]).resolve()),
                 "group": doc["group"], "control_profile": doc["control_profile"]} for r in doc["videos"]]
        if [unique_json(line) for line in (output / "official_videos.jsonl").read_bytes().splitlines()] != rows:
            raise EvaluationError("Archived official JSONL differs from this generation")
        request = unique_json((output / "official_request.json").read_bytes())
        if request != {key: value for key, value in provenance.items() if key not in ("returncode", "output_files")}:
            raise EvaluationError("Archived official invocation differs from original request")
        if provenance.get("returncode") != 0:
            raise EvaluationError("Archived official process did not succeed")
        if self.spec.stage == "full":
            scores = unique_json((output / "official/scores/scores.json").read_bytes())
            if scores != report.get("scores"):
                raise EvaluationError("Archived scores differ from raw official scores")
            # Derived short-video scores are independent sidecars. Historical
            # completed archives without sidecars remain immutable and valid.
            derived_files = (output / "short_video_scores.json",
                             output / "short_video_scores.csv")
            if any(p.is_file() for p in derived_files):
                verify_short_video_scores(output / "official/scores/scores.json", output)
            paths = list((output / "official" / doc["model"] / "results").glob("*/report.json"))
            if {p.parent.name for p in paths} != {r["case_id"] for r in rows}:
                raise EvaluationError("Archived official case coverage mismatch")
            for path in paths:
                case = unique_json(path.read_bytes())
                if case.get("status") != "complete" or case.get("model") != doc["model"] or case.get("case_id") != path.parent.name:
                    raise EvaluationError("Archived official case report is incomplete")
        return report

    def evaluate(self, dataset: ORBIADataset, videos: ORBIAVideos, output: str | Path,
                 *, timeout: float = 86400, resume: bool = False) -> EvaluationReport:
        output = Path(output).resolve()
        if output.is_relative_to(dataset.root):
            raise EvaluationError("Evaluation output must be outside original dataset root")
        output.mkdir(parents=True, exist_ok=resume)
        provenance = {}
        preserve_success = False
        diagnostic = output.parent / "resume_failures" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")

        def save(report):
            destination = diagnostic if preserve_success and report.status != Status.COMPLETE else output
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "evaluation_report.json").write_text(json.dumps(report.as_dict(), indent=2, allow_nan=False) + "\n")
            return report

        try:
            prior = output / "evaluation_report.json"
            if resume and prior.exists():
                preserve_success = unique_json(prior.read_bytes()).get("status") == "COMPLETE"
            report, doc = self._check(dataset, videos)
            provenance = dict(report.provenance)
            if report.status != Status.READY:
                return save(report)
            if resume and prior.exists():
                old = unique_json(prior.read_bytes())
                if old.get("status") == "COMPLETE":
                    saved = self.archived_report(videos, output)
                    return EvaluationReport(Status.COMPLETE, tuple(saved["reasons"]), saved["provenance"], saved["scores"])
                history = output / "attempts" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                history.mkdir(parents=True)
                import shutil
                for name in ("evaluation_report.json", "official_request.json", "stdout.log", "stderr.log", "traceback.log"):
                    if (output / name).exists():
                        shutil.copyfile(output / name, history / name)
                # Keep reports and worker failure logs before upstream's resume
                # updates them. Large geometry tensors stay in its native cache.
                for path in (output / "official").rglob("*"):
                    if path.is_file() and path.suffix in (".json", ".jsonl", ".csv", ".log"):
                        target = history / path.relative_to(output)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(path, target)
                # Upstream's worker resumes only matching evaluation_context.
                # Require our stronger byte-bound input identity as well.
                for name, expected in (("video_manifest.json", videos.manifest_sha256),
                                       ("evaluation.local.yaml", provenance["evaluator_config_sha256"])):
                    if (output / name).exists() and sha256_file(output / name) != expected:
                        raise EvaluationError("Resume input changed: " + name)
                old_provenance = old.get("provenance", {})
                for name in ("official_commit", "scoring_version", "requested_tiers", "stage", "video_manifest_sha256"):
                    if name in old_provenance and old_provenance[name] != provenance.get(name):
                        raise EvaluationError("Resume evaluator identity changed: " + name)
            # Preserve exact input manifest and native official JSONL separately.
            raw_manifest = Path(videos.manifest).read_bytes()
            import hashlib
            if hashlib.sha256(raw_manifest).hexdigest() != videos.manifest_sha256:
                raise EvaluationError("Video manifest changed after preflight")
            (output / "video_manifest.json").write_bytes(raw_manifest)
            config = Path(self.spec.config).resolve()
            if sha256_file(config) != provenance["evaluator_config_sha256"]:
                raise EvaluationError("Official evaluator config changed after preflight")
            config_snapshot = output / "evaluation.local.yaml"
            config_snapshot.write_bytes(config.read_bytes())
            for row in doc["videos"]:
                if sha256_file(Path(row["resolved_video"])) != row["sha256"]:
                    raise EvaluationError("Video changed after preflight")
            manifest = output / "official_videos.jsonl"
            rows = [{"model": doc["model"], "case_id": r["case_id"], "video": r["resolved_video"],
                     "group": doc["group"], "control_profile": doc["control_profile"]} for r in doc["videos"]]
            manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
            raw = output / "official"
            command = self.command(dataset, manifest, raw)
            command[command.index("--config") + 1] = str(config_snapshot)
            provenance.update(command=command, cwd=str(Path(self.spec.official_root).resolve()),
                              official_video_manifest_sha256=sha256_file(manifest))
            request = output / "official_request.json"
            request.write_text(json.dumps(provenance, indent=2) + "\n")
            with (output / "stdout.log").open("wb") as stdout, (output / "stderr.log").open("wb") as stderr:
                result = subprocess.run(command, cwd=Path(self.spec.official_root).resolve(),
                                        stdout=stdout, stderr=stderr, timeout=timeout, check=False)
            provenance["returncode"] = result.returncode
            if result.returncode:
                raise EvaluationError(f"Official ORBIA evaluator failed: returncode {result.returncode}; see raw logs")
            if self.spec.stage != "full":
                return save(EvaluationReport(Status.COMPLETE, ("Geometry/precompute only; no metrics produced",), provenance))
            scores = unique_json((raw / "scores/scores.json").read_bytes())
            # Keep all official nested components, ranks, nulls and raw values.
            if scores.get("protocol") != self.lock["scoring_version"] or not isinstance(scores.get("components"), list):
                raise EvaluationError("Official scores protocol/components missing")
            score_rows = scores.get("rows")
            if (not isinstance(score_rows, list) or len(score_rows) != 1
                    or score_rows[0].get("model") != doc["model"]
                    or score_rows[0].get("cases") != len(rows)
                    or any(t not in score_rows[0] for t in (*TIERS, "Overall"))):
                raise EvaluationError("Official score rows/model/case coverage invalid")
            if score_rows[0]["T6"] is not None or score_rows[0]["Overall"] is not None:
                raise EvaluationError("Short-only official scoring must preserve null T6/Overall")
            paths = list((raw / doc["model"] / "results").glob("*/report.json"))
            if {p.parent.name for p in paths} != {r["case_id"] for r in rows}:
                raise EvaluationError("Official per-case reports incomplete")
            for path in paths:
                case = unique_json(path.read_bytes())
                if (case.get("status") != "complete" or case.get("case_id") != path.parent.name
                        or case.get("model") != doc["model"]):
                    raise EvaluationError("Official per-case report incomplete/identity mismatch")
            if not (raw / "scores/scores.csv").is_file():
                raise EvaluationError("Official scores.csv missing")
            # Pinned author scoring remains untouched. Explicitly derive
            # short-only T6 from its official temporal.short component, then
            # reuse the official Overall tier weights in a separate artifact.
            render_short_video_scores(raw / "scores/scores.json", output)
            provenance["output_files"] = {str(p.relative_to(output)): sha256_file(p)
                for p in output.rglob("*") if p.is_file() and p != output / "evaluation_report.json"}
            return save(EvaluationReport(Status.COMPLETE, provenance=provenance, scores=scores))
        except (DatasetError, EvaluationError, OSError, subprocess.SubprocessError) as exc:
            failure_root = diagnostic if preserve_success else output
            failure_root.mkdir(parents=True, exist_ok=True)
            (failure_root / "traceback.log").write_text(traceback.format_exc())
            save(EvaluationReport(Status.FAILED, (str(exc),), provenance))
            raise EvaluationError(str(exc)) from exc
