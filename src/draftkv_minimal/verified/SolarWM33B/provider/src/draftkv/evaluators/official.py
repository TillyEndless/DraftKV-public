"""Audited official evaluator invocation; no approximate scoring implementations.

An integration supplies a pinned adapter to the *official* API. Merely installing
ordinary VBench never enables VBench-Long. Protocol claims require an externally
reviewed, hash-pinned configuration; this repository ships no such claims.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Mapping

from draftkv.benchmarks.datasets import (
    DatasetError, contained_file, require_sha, required_text, sha256_file, unique_json,
)


class EvaluationError(ValueError):
    """Invalid/missing input or official evaluator execution failure."""


class Status(str, Enum):
    READY = "READY"
    COMPLETE = "COMPLETE"
    BLOCKED_DEPENDENCY = "BLOCKED_DEPENDENCY"
    BLOCKED_PROTOCOL = "BLOCKED_PROTOCOL"
    FAILED = "FAILED"


@dataclass(frozen=True)
class DependencyPin:
    distribution: str
    version: str


@dataclass(frozen=True)
class HookSpec:
    # Reviewed adapter file, invoked by this interpreter; no shell evaluation.
    adapter: str | Path
    sha256: str
    dependencies: tuple[DependencyPin, ...]


@dataclass(frozen=True)
class EvaluationInputs:
    reference_manifest: str | Path
    reference_sha256: str
    video_manifest: str | Path
    video_manifest_sha256: str
    video_root: str | Path


@dataclass(frozen=True)
class EvaluationReport:
    status: Status
    reasons: tuple[str, ...] = ()
    provenance: Mapping = field(default_factory=dict)
    scores: Mapping | None = None

    def as_dict(self) -> dict:
        return {
            "status": self.status.value, "reasons": list(self.reasons),
            "provenance": dict(self.provenance), "scores": self.scores,
        }


@dataclass(frozen=True)
class EvaluatorDefinition:
    name: str
    benchmark: str
    dataset_family: str
    requires_long_video: bool = False


class EvaluatorRegistry:
    def __init__(self):
        self._entries = {
            "vbench-long": EvaluatorDefinition("vbench-long", "bench1", "bench1", True),
            "orbia": EvaluatorDefinition("orbia", "bench2", "orbia"),
        }

    def get(self, name: str) -> EvaluatorDefinition:
        try:
            return self._entries[name]
        except KeyError as exc:
            raise EvaluationError(f"Unknown official evaluator: {name}") from exc

    def register(self, definition: EvaluatorDefinition) -> None:
        if definition.name in self._entries:
            raise EvaluationError(f"Evaluator already registered: {definition.name}")
        self._entries[definition.name] = definition


class OfficialEvaluator:
    def __init__(
        self, repo_root: str | Path, name: str, *,
        protocol: str | Path | None = None, protocol_sha256: str | None = None,
        hook: HookSpec | None = None, registry: EvaluatorRegistry | None = None,
    ):
        root = Path(repo_root)
        if not root.is_absolute() or not root.is_dir():
            raise EvaluationError("repo_root must be an existing absolute directory")
        self.repo_root = root.resolve()
        self.definition = (registry or EvaluatorRegistry()).get(name)
        self.protocol = protocol
        self.protocol_sha256 = protocol_sha256
        self.hook = hook

    def _path(self, path: str | Path) -> Path:
        path = Path(path)
        return (path if path.is_absolute() else self.repo_root / path).resolve()

    def _pinned_json(self, path: str | Path, digest: str, label: str) -> tuple[dict, str]:
        resolved = self._path(path)
        try:
            raw = resolved.read_bytes()
            expected = require_sha(digest, label)
            actual = hashlib.sha256(raw).hexdigest()
            if actual != expected:
                raise EvaluationError(f"{label} mismatch: {resolved}")
            return unique_json(raw), actual
        except (OSError, DatasetError) as exc:
            raise EvaluationError(str(exc)) from exc

    def _inputs(self, inputs: EvaluationInputs) -> dict:
        reference, reference_sha = self._pinned_json(
            inputs.reference_manifest, inputs.reference_sha256, "reference manifest SHA256"
        )
        videos, videos_sha = self._pinned_json(
            inputs.video_manifest, inputs.video_manifest_sha256, "video manifest SHA256"
        )
        definition = self.definition
        family = reference.get("dataset_family", reference.get("benchmark"))
        if (reference.get("benchmark") != definition.benchmark
                or family != definition.dataset_family):
            raise EvaluationError(
                f"Evaluator {definition.name} rejects dataset family {family!r}; "
                "cross-dataset adaptation/resampling is forbidden"
            )
        if type(reference.get("schema_version")) is not int or reference["schema_version"] != 1:
            raise EvaluationError("Unsupported reference manifest schema_version")
        ids = reference.get("frozen_case_ids")
        if (not isinstance(ids, list) or not ids
                or any(not isinstance(i, str) or not i.strip() for i in ids)
                or len(set(ids)) != len(ids)):
            raise EvaluationError("Reference must contain unique frozen_case_ids")
        if definition.benchmark == "bench1":
            from draftkv.benchmarks.bench1 import CASE_COUNTS, MANIFEST_PINS
            from draftkv.benchmarks.datasets import Backbone
            try:
                backbone = Backbone(reference.get("backbone", "").lower())
            except ValueError as error:
                raise EvaluationError("Bench1 reference requires an explicit supported backbone") from error
            if backbone not in CASE_COUNTS:
                raise EvaluationError("Bench1 LingBot14B and its T+I benchmark have been retired")
            if len(ids) != CASE_COUNTS[backbone]:
                raise EvaluationError("Bench1 reference must contain its backbone-specific frozen case count")
            if backbone is Backbone.LONGLIVE_1_3B:
                from draftkv.benchmarks.worldattention import CASE_IDS
                if (ids != list(CASE_IDS) or (reference_sha != MANIFEST_PINS[backbone]
                        and reference.get("original_manifest_sha256") != MANIFEST_PINS[backbone])):
                    raise EvaluationError("LongLive evaluator requires the pinned WorldAttention160 source")
        if definition.name == "orbia" and len(ids) != 480:
            raise EvaluationError("ORBIA Bench2 release requires exactly 480 frozen official case IDs")
        if (type(videos.get("schema_version")) is not int or videos["schema_version"] != 1
                or videos.get("reference_manifest_sha256") != reference_sha
                or videos.get("benchmark") != definition.benchmark
                or videos.get("dataset_family") != definition.dataset_family):
            raise EvaluationError("Video manifest schema/reference/dataset mismatch")
        rows = videos.get("videos")
        if (not isinstance(rows, list) or any(not isinstance(r, dict) for r in rows)
                or [r.get("case_id") for r in rows] != ids):
            raise EvaluationError("Video case IDs/order must match frozen reference exactly")
        root = self._path(inputs.video_root)
        checked = []
        try:
            for row in rows:
                path = contained_file(root, row.get("path"))
                digest = require_sha(row.get("sha256"), "video SHA256")
                if sha256_file(path) != digest:
                    raise EvaluationError(f"Video SHA256 mismatch: {row['case_id']}")
                checked.append({"case_id": row["case_id"], "path": str(path), "sha256": digest})
        except DatasetError as exc:
            raise EvaluationError(str(exc)) from exc
        return {
            "evaluator": definition.name, "benchmark": definition.benchmark,
            "dataset_family": definition.dataset_family,
            "reference_manifest": str(self._path(inputs.reference_manifest)),
            "reference_manifest_sha256": reference_sha,
            "video_manifest_sha256": videos_sha,
            "case_ids": ids, "videos": checked,
        }

    def preflight(self, inputs: EvaluationInputs) -> EvaluationReport:
        # Data integrity errors always raise, even with no official dependencies.
        provenance = self._inputs(inputs)
        return self.protocol_preflight(inputs.reference_sha256, provenance)

    def protocol_preflight(self, reference_sha256: str, provenance: dict | None = None) -> EvaluationReport:
        """Check official protocol/dependencies before any video is generated."""
        provenance = dict(provenance or {})
        if self.definition.name == "orbia":
            return EvaluationReport(Status.BLOCKED_PROTOCOL, (
                "Legacy envelopes cannot certify official ORBIA inputs; use ORBIAOfficialEvaluator "
                "with the frozen manifest, installation index and decoded video preflight",
            ), provenance)
        if self.protocol is None or self.protocol_sha256 is None:
            return EvaluationReport(Status.BLOCKED_PROTOCOL, ("No pinned, verified official protocol",), provenance)
        config, config_sha = self._pinned_json(self.protocol, self.protocol_sha256, "protocol SHA256")
        provenance.update({"protocol_sha256": config_sha, "protocol": config})
        checks = (
            (type(config.get("schema_version")) is int and config["schema_version"] == 1, "protocol schema_version must be 1"),
            (config.get("verified") is True, "official parameters have not been verified"),
            (isinstance(config.get("verification_reference"), str) and bool(config["verification_reference"].strip()), "missing protocol verification reference"),
            (config.get("evaluator") == self.definition.name, "evaluator identity mismatch"),
            (config.get("dataset_family") == self.definition.dataset_family, "protocol dataset mismatch"),
            (config.get("official_mode") is True, "official_mode must be true"),
            (isinstance(config.get("protocol_id"), str) and bool(config["protocol_id"].strip()), "missing fixed protocol_id"),
            (isinstance(config.get("official_parameters"), dict) and bool(config["official_parameters"]), "official protocol parameters are unverified/empty"),
            (not self.definition.requires_long_video or config.get("long_video_mode") is True, "VBench-Long requires explicit long_video_mode=true"),
            (config.get("reference_manifest_sha256") == reference_sha256, "protocol is not pinned to this reference manifest"),
        )
        reasons = tuple(reason for valid, reason in checks if not valid)
        if reasons:
            return EvaluationReport(Status.BLOCKED_PROTOCOL, reasons, provenance)
        hook = self.hook
        if hook is None or not hook.dependencies:
            return EvaluationReport(Status.BLOCKED_DEPENDENCY, ("Missing pinned official adapter/dependencies",), provenance)
        dependency_versions = {}
        missing = []
        for dependency in hook.dependencies:
            if not dependency.distribution.strip() or not dependency.version.strip():
                missing.append("Official dependency name/version must be fixed")
                continue
            try:
                actual = importlib.metadata.version(dependency.distribution)
            except importlib.metadata.PackageNotFoundError:
                missing.append(f"Missing official dependency: {dependency.distribution}=={dependency.version}")
                continue
            if actual != dependency.version:
                missing.append(f"Dependency version mismatch: {dependency.distribution}: {actual} != {dependency.version}")
            dependency_versions[dependency.distribution] = actual
        if config.get("dependency_versions") != {d.distribution: d.version for d in hook.dependencies}:
            return EvaluationReport(Status.BLOCKED_PROTOCOL, ("Protocol dependency pins differ from adapter",), provenance)
        try:
            adapter_sha = sha256_file(self._path(hook.adapter))
            if adapter_sha != require_sha(hook.sha256, "adapter SHA256"):
                missing.append("Official adapter SHA256 mismatch")
        except DatasetError as exc:
            missing.append(str(exc))
        if config.get("adapter_sha256") != hook.sha256:
            return EvaluationReport(Status.BLOCKED_PROTOCOL, ("Protocol adapter pin differs from hook",), provenance)
        provenance.update({"dependency_versions": dependency_versions, "adapter_sha256": hook.sha256})
        if missing:
            return EvaluationReport(Status.BLOCKED_DEPENDENCY, tuple(missing), provenance)
        return EvaluationReport(Status.READY, provenance=provenance)

    def evaluate(self, inputs: EvaluationInputs, output_dir: str | Path, *, timeout: float = 3600) -> EvaluationReport:
        report = self.preflight(inputs)
        output = self._path(output_dir)
        # Require a new run directory: never consume a stale official result.
        output.mkdir(parents=True, exist_ok=False)
        report_path = output / "evaluation_report.json"
        if report.status is not Status.READY:
            report_path.write_text(json.dumps(report.as_dict(), indent=2) + "\n", encoding="utf-8")
            return report
        request = output / "official_request.json"
        result_path = output / "official_result.json"
        request.write_text(json.dumps(report.provenance, indent=2) + "\n", encoding="utf-8")
        expected_request_sha = sha256_file(request)
        assert self.hook is not None
        command = [sys.executable, str(self._path(self.hook.adapter)), "--request", str(request), "--output", str(result_path)]
        # Write READY provenance before invocation so even failed runs remain auditable.
        report_path.write_text(json.dumps(report.as_dict(), indent=2) + "\n", encoding="utf-8")
        try:
            with (output / "official_stdout.log").open("w") as stdout, (output / "official_stderr.log").open("w") as stderr:
                subprocess.run(command, cwd=self.repo_root, check=True, timeout=timeout, stdout=stdout, stderr=stderr)
            if sha256_file(request) != expected_request_sha:
                raise DatasetError("Official adapter modified the pinned request")
            result = unique_json(result_path.read_bytes())
        except (OSError, subprocess.SubprocessError, DatasetError) as exc:
            failed = EvaluationReport(Status.FAILED, (str(exc),), report.provenance)
            report_path.write_text(json.dumps(failed.as_dict(), indent=2) + "\n", encoding="utf-8")
            raise EvaluationError(f"Official evaluator failed; see {output}: {exc}") from exc
        scores = result.get("scores")
        if (result.get("official_mode") is not True
                or result.get("evaluator") != self.definition.name
                or result.get("request_sha256") != expected_request_sha
                or result.get("case_ids") != report.provenance["case_ids"]
                or (self.definition.requires_long_video and result.get("long_video_mode") is not True)
                or not isinstance(scores, dict) or not scores
                or any(not isinstance(k, str) or type(v) not in (int, float) or not math.isfinite(v) for k, v in scores.items())):
            failed = EvaluationReport(Status.FAILED, ("Official result provenance/scores rejected",), report.provenance)
            report_path.write_text(json.dumps(failed.as_dict(), indent=2) + "\n", encoding="utf-8")
            raise EvaluationError("Official result is missing/mismatched provenance or finite numeric scores")
        complete = EvaluationReport(Status.COMPLETE, provenance={
            **report.provenance, "request_sha256": expected_request_sha,
            "official_result_sha256": sha256_file(result_path),
        }, scores=scores)
        report_path.write_text(json.dumps(complete.as_dict(), indent=2) + "\n", encoding="utf-8")
        return complete
