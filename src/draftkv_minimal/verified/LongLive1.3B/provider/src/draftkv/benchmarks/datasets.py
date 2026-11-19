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
    LINGBOT_14B = "lingbot14b"


INPUT_MODES = {
    Backbone.LONGLIVE_1_3B: InputMode.TEXT_TO_VIDEO,
    Backbone.LINGBOT_14B: InputMode.TEXT_IMAGE_TO_VIDEO,
}
DEFAULT_ROOTS = {
    Backbone.LONGLIVE_1_3B: Path(
        "configs/datasets/bench1"
    ),
    Backbone.LINGBOT_14B: Path(
        "datasets/bench1_penguin_random100_ti_98case_20261025"
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
        mode = INPUT_MODES[backbone]
        prefix = "DRAFTKV_BENCH1_" + (
            "LONGLIVE" if backbone is Backbone.LONGLIVE_1_3B else "LINGBOT"
        )
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
        if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
            raise DatasetError("Unsupported dataset manifest schema_version")
        if manifest.get("benchmark") != "bench1":
            raise DatasetError("Expected bench1 manifest")
        dataset_id = required_text(manifest.get("dataset_id"), "dataset_id")
        if manifest.get("backbone") != backbone.value or manifest.get("input_mode") != mode.value:
            raise DatasetError("Manifest backbone/input_mode mismatch")
        ids = manifest.get("frozen_case_ids")
        if (not isinstance(ids, list) or len(ids) != 98
                or any(not isinstance(i, str) or not i.strip() for i in ids)
                or len(set(ids)) != 98):
            raise DatasetError("Bench1 requires exactly 98 unique frozen case IDs")
        cases = manifest.get("cases")
        if (not isinstance(cases, list) or len(cases) != len(ids)
                or any(not isinstance(c, dict) for c in cases)
                or [c.get("case_id") for c in cases] != ids):
            raise DatasetError("Case count/order must match frozen_case_ids exactly")
        root = self._path(
            spec.root or self.environ.get(prefix + "_ROOT")
            or manifest.get("dataset_root") or DEFAULT_ROOTS[backbone]
        )
        if not root.is_dir():
            raise DatasetError(f"Required dataset directory does not exist: {root}")
        resolved = []
        for case in cases:
            case_id = case["case_id"]
            prompt = required_text(case.get("prompt"), f"{case_id}.prompt")
            prompt_sha = require_sha(case.get("prompt_sha256"), f"{case_id}.prompt_sha256")
            if hashlib.sha256(prompt.encode("utf-8")).hexdigest() != prompt_sha:
                raise DatasetError(f"Prompt SHA256 mismatch: {case_id}")
            image = case.get("image")
            image_path = image_sha = dimensions = None
            if mode is InputMode.TEXT_TO_VIDEO:
                if image is not None:
                    raise DatasetError(f"T2V case must not include an image: {case_id}")
            else:
                if not isinstance(image, dict) or image.get("case_id") != case_id:
                    raise DatasetError(f"Missing or misaligned image: {case_id}")
                image_sha = require_sha(image.get("sha256"), f"{case_id}.image.sha256")
                image_path = contained_file(root, image.get("path"))
                if sha256_file(image_path) != image_sha:
                    raise DatasetError(f"Image SHA256 mismatch: {case_id}")
                dimensions = (
                    positive_int(image.get("width"), "image.width"),
                    positive_int(image.get("height"), "image.height"),
                )
                if image_dimensions(image_path) != dimensions:
                    raise DatasetError(f"Image dimensions mismatch: {case_id}")
            resolved.append(ResolvedCase(case_id, prompt, prompt_sha, image_path, image_sha, dimensions))
        return ResolvedDataset(
            "bench1", dataset_id, backbone, mode, root, manifest_path, actual_sha, tuple(resolved)
        )
