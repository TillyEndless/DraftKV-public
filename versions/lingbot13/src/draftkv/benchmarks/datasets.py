"""Resolve frozen Bench1 inputs without sampling, substitutions or implicit fallback.

Manifest format and pinning requirements are documented in evaluator_protocol.md.
No remote paths are accessed until a caller explicitly resolves a dataset.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping


class DatasetError(ValueError):
    """Input data is absent, malformed, mismatched or not pinned."""


class InputMode(str, Enum):
    TEXT_IMAGE_TO_VIDEO = "t_i_to_v"
    TEXT_TO_VIDEO = "t_to_v"


class Backbone(str, Enum):
    LONGLIVE_1_3B = "longlive1.3b"
    LINGBOT_14B = "lingbot14b"  # Legacy identifier retained only for explicit rejection.


INPUT_MODES = {
    Backbone.LONGLIVE_1_3B: InputMode.TEXT_TO_VIDEO,
}
DEFAULT_ROOTS = {
    Backbone.LONGLIVE_1_3B: Path(
        "configs/datasets/bench1"
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise DatasetError(f"Cannot read required file {path}: {exc}") from exc
    return digest.hexdigest()


def require_sha(value: object, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise DatasetError(f"{label} must be a lowercase SHA256 hex digest")
    return value


def unique_json(raw: bytes) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise DatasetError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        def invalid_constant(value):
            raise DatasetError(f"Non-finite JSON constant: {value}")

        result = json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid_constant)
    except (ValueError, UnicodeError) as exc:
        raise DatasetError(f"Invalid JSON: {exc}") from exc
    if not isinstance(result, dict):
        raise DatasetError("Manifest/result must be a JSON object")
    return result


def required_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DatasetError(f"{label} must be a nonempty string")
    return value


def positive_int(value: object, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise DatasetError(f"{label} must be a positive integer")
    return value


def contained_file(root: Path, relative: object) -> Path:
    name = required_text(relative, "file path")
    if Path(name).is_absolute() or ".." in Path(name).parts:
        raise DatasetError(f"File path must be relative without traversal: {name}")
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise DatasetError(f"Missing file or file escapes dataset root: {path}")
    return path


def image_dimensions(path: Path) -> tuple[int, int]:
    """Decode with Pillow; missing decoder or invalid images are hard errors."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise DatasetError("Image validation requires Pillow; no header-only fallback") from exc
    try:
        with Image.open(path) as img:
            dimensions = img.size
            img.verify()
        with Image.open(path) as img:
            img.load()
        return dimensions
    except Exception as exc:
        raise DatasetError(f"Invalid image {path}: {exc}") from exc


@dataclass(frozen=True)
class DatasetSpec:
    backbone: Backbone
    manifest: str | Path | None = None
    manifest_sha256: str | None = None
    root: str | Path | None = None


@dataclass(frozen=True)
class ResolvedCase:
    case_id: str
    prompt: str
    prompt_sha256: str
    image: Path | None
    image_sha256: str | None
    image_dimensions: tuple[int, int] | None


@dataclass(frozen=True)
class ResolvedScheduledCase(ResolvedCase):
    """Only scheduled T2V adds fields; the frozen TI case schema is unchanged."""
    prompts: tuple[str, ...]
    timeline: tuple[dict, ...]
    prompt_events: tuple[dict, ...]


@dataclass(frozen=True)
class ResolvedDataset:
    benchmark: str
    dataset_id: str
    backbone: Backbone
    input_mode: InputMode
    root: Path
    manifest: Path
    manifest_sha256: str
    cases: tuple[ResolvedCase, ...]

    @property
    def case_ids(self) -> tuple[str, ...]:
        return tuple(case.case_id for case in self.cases)


class DatasetResolver:
    def __init__(self, repo_root: str | Path, environ: Mapping[str, str] | None = None):
        self.repo_root = Path(repo_root)
        if not self.repo_root.is_absolute() or not self.repo_root.is_dir():
            raise DatasetError("repo_root must be an existing absolute directory")
        self.repo_root = self.repo_root.resolve()
        self.environ = os.environ if environ is None else environ

    def _path(self, value: str | Path) -> Path:
        path = Path(value)
        return (path if path.is_absolute() else self.repo_root / path).resolve()

    def resolve(self, spec: DatasetSpec) -> ResolvedDataset:
        # ORBIA has a distinct official on-disk schema. Keep Bench1 untouched.
        from .orbia import ORBIADatasetResolver, ORBIASpec
        if isinstance(spec, ORBIASpec):
            return ORBIADatasetResolver(self.repo_root, self.environ).resolve(spec)
        try:
            backbone = Backbone(spec.backbone)
        except ValueError as exc:
            raise DatasetError(f"Unsupported backbone: {spec.backbone}") from exc
        if backbone is not Backbone.LONGLIVE_1_3B:
            raise DatasetError("Bench1 LingBot14B and its T+I benchmark have been retired")
        prefix = "DRAFTKV_BENCH1_LONGLIVE"
        manifest_name = spec.manifest or self.environ.get(prefix + "_MANIFEST")
        if not manifest_name:
            raise DatasetError(f"Required manifest missing: config or {prefix}_MANIFEST")
        manifest_path = self._path(manifest_name)
        expected_sha = require_sha(
            spec.manifest_sha256 or self.environ.get(prefix + "_MANIFEST_SHA256"),
            "Pinned manifest SHA256",
        )
        try:
            raw = manifest_path.read_bytes()
        except OSError as exc:
            raise DatasetError(f"Cannot read manifest {manifest_path}: {exc}") from exc
        actual_sha = hashlib.sha256(raw).hexdigest()
        if actual_sha != expected_sha:
            raise DatasetError("Manifest SHA256 mismatch (frozen inputs changed)")
        manifest = unique_json(raw)
        # The reviewed server manifests predate the portable envelope schema.
        # Authenticate their actual bytes, then parse their original schema.
        from .bench1 import MANIFEST_PINS, resolve_frozen_bench1
        if backbone is Backbone.LONGLIVE_1_3B and expected_sha != MANIFEST_PINS[backbone]:
            raise DatasetError("LongLive requires the frozen WorldAttention160 manifest pin")
        if expected_sha == MANIFEST_PINS[backbone]:
            root = self._path(spec.root or self.environ.get(prefix + "_ROOT") or manifest_path.parent)
            return resolve_frozen_bench1(backbone, manifest_path, raw, root)
