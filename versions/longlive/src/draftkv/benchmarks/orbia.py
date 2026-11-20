"""Frozen ORBIA short-case inputs. Never sample cases or rewrite source assets.

The upstream manifest is not DraftKV's Bench1/envelope format. Its raw bytes
and original row order are authoritative. See configs/datasets/orbia480.lock.json.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .datasets import (
    DatasetError, contained_file, image_dimensions, require_sha, required_text,
    sha256_file, unique_json,
)

SHORT_FRAMES = 357
FPS = 16
SHORT_CASES = 480
PUBLIC_FILES = ("case.json", "camera.npz", "q0.png", "prompt.txt")
DEFAULT_LOCK = Path(__file__).resolve().parents[3] / "configs/datasets/orbia480.lock.json"


class ORBIADependencyError(DatasetError):
    """Official assets, download transport or CPU decoder dependencies unavailable."""

    status = "BLOCKED_DEPENDENCY"


def frozen(value):
    if isinstance(value, dict):
        return MappingProxyType({k: frozen(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(frozen(v) for v in value)
    return value


def load_lock(path: str | Path = DEFAULT_LOCK) -> dict:
    try:
        lock = unique_json(Path(path).read_bytes())
    except OSError as exc:
        raise ORBIADependencyError(f"ORBIA source lock unavailable: {path}") from exc
    if (lock.get("schema_version") != 1 or lock.get("dataset_family") != "orbia"
            or lock.get("benchmark") != "bench2"):
        raise DatasetError("ORBIA rejects WorldAtlas/other dataset locks")
    return lock


def git_blob_sha1(raw: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()


def verify_distribution(raw: bytes, asset: Mapping, *, lfs: bool = False) -> str:
    """Authenticate HF small files or LFS pointers against the frozen Git blob ID.

    Missing LFS digests must be acquired from that exact pointer; a hash computed
    from a newly downloaded untrusted archive is never accepted as its own pin.
    """
    if git_blob_sha1(raw) != asset["git_blob_sha1"]:
        raise DatasetError("Official distribution Git blob hash mismatch")
    digest = hashlib.sha256(raw).hexdigest()
    if not lfs:
        if len(raw) != asset["size"]:
            raise DatasetError("Official manifest size mismatch")
        if asset.get("sha256") and digest != require_sha(asset["sha256"], "manifest SHA256"):
            raise DatasetError("Official manifest SHA256 mismatch")
    return digest


def validate_manifest(raw: bytes, lock: Mapping) -> tuple[dict, tuple[str, ...]]:
    verify_distribution(raw, lock["distribution"]["manifest"])
    doc = unique_json(raw)
    if doc.get("dataset_family", "orbia") != "orbia" or doc.get("benchmark", "bench2") != "bench2":
        raise DatasetError("ORBIA rejects WorldAtlas/cross-dataset input")
    rows = doc.get("cases")
    if not isinstance(rows, list) or len(rows) != 720:
        raise DatasetError("Official ORBIA manifest requires 480 short + 240 long cases")
    ids, short, long_count = [], [], 0
    for row in rows:
        if not isinstance(row, dict):
            raise DatasetError("Invalid official case row")
        cid = required_text(row.get("case_id"), "case_id")
        if Path(cid).name != cid or cid in (".", "..") or "\\" in cid:
            raise DatasetError("Case ID must be one safe path component")
        ids.append(cid)
        count = row.get("frame_count")
        if type(count) is not int or count not in (357, 961):
            raise DatasetError("ORBIA rejects historical WorldAtlas/unknown frame counts")
        if count == SHORT_FRAMES:
            if row.get("parent_case_id") is not None:
                raise DatasetError("Short case must not have a parent")
            short.append(cid)
        else:
            long_count += 1
    if len(set(ids)) != 720 or len(short) != SHORT_CASES or long_count != 240:
        raise DatasetError("Official ORBIA manifest count/duplicate mismatch")
    for row in rows:
        if row["frame_count"] == 961 and row.get("parent_case_id") not in short:
            raise DatasetError("Long case parent is absent from official short cases")
    return doc, tuple(short)


def reference_files(document: Mapping) -> tuple[str, ...]:
    """Exact asset fields used by upstream src.data.bundle._reference_files."""
    try:
        names = [f[k] for f in document["frames"].values() for k in ("rgb", "depth", "valid")]
        names += [a[k] for a in document["anchors"] for k in ("q0_mask", "qe_mask", "qe_region")]
        names += list(document["observability"].values()) + [document["qe_scene_region"]]
    except (KeyError, TypeError) as exc:
        raise DatasetError("Invalid official reference document") from exc
    if not names or not document["anchors"]:
        raise DatasetError("Reference RGB-D/anchors required")
    for name in names:
        required_text(name, "reference file")
        if Path(name).is_absolute() or ".." in Path(name).parts or "\\" in name:
            raise DatasetError("Reference asset escapes case root")
    return tuple(dict.fromkeys(names))


@dataclass(frozen=True)
class ORBIASpec:
    dataset_root: str | Path | None = None
    index: str | Path | None = None
    index_sha256: str | None = None
    stage: str = "inference"


@dataclass(frozen=True)
class ORBIAInput:
    dataset_family: str
    case_id: str
    image_path: Path
    caption: str  # exact UTF-8 contents, including any terminal newline
    poses_c2w: Any  # original archive values, never rebased in this data layer
    intrinsics: Any
    timestamps_s: Any
    events: Mapping
    document: Mapping
    source_manifest_sha256: str
    file_hashes: Mapping[str, str]

    @property
    def num_frames(self) -> int:
        return SHORT_FRAMES

    @property
    def fps(self) -> int:
        return FPS


@dataclass(frozen=True)
class ORBIADataset:
    root: Path
    manifest_sha256: str
    index_sha256: str
    data_version: str
    case_ids: tuple[str, ...]
    files: Mapping
    stage: str
    manifest: Mapping
    official_commit: str

    def _checked(self, relative: str) -> Path:
        path = contained_file(self.root, relative)
        if (relative not in self.files or not isinstance(self.files[relative], Mapping)
                or sha256_file(path) != self.files[relative].get("sha256")):
            raise DatasetError(f"ORBIA file hash mismatch/unindexed asset: {relative}")
        return path

    def load_case(self, case_id: str) -> ORBIAInput:
        if case_id not in self.case_ids:
            raise DatasetError(f"Case is not in frozen ORBIA 480 short manifest: {case_id}")
        try:
            import numpy as np
            # Check the decoder dependency separately from invalid image bytes.
            # image_dimensions deliberately reports malformed images as data errors.
            __import__("PIL.Image")
        except ImportError as exc:
            raise ORBIADependencyError("ORBIA CPU input parsing requires numpy and Pillow") from exc
        prefix = f"public/{case_id}/"
        paths = {name: self._checked(prefix + name) for name in PUBLIC_FILES}
        doc = unique_json(paths["case.json"].read_bytes())
        row = next(r for r in self.manifest["cases"] if r["case_id"] == case_id)
        if (doc.get("case_id") != case_id or doc.get("frame_count") != SHORT_FRAMES
                or doc.get("fps") != FPS or doc.get("parent_case_id") is not None):
            raise DatasetError("ORBIA public case identity/frame count/FPS mismatch")
        if any(doc.get(k) != row.get(k) for k in ("source", "template", "parent_case_id")):
            raise DatasetError("Public case differs from official manifest")
        if doc.get("dataset_family", "orbia") != "orbia":
            raise DatasetError("ORBIA rejects WorldAtlas input")
        try:
            with np.load(paths["camera.npz"], allow_pickle=False) as a:
                poses, times = a["poses_c2w"].copy(), a["timestamps_s"].copy()
            K = np.asarray(doc["intrinsics"]["K"], dtype=np.float64)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise DatasetError("Invalid ORBIA camera NPZ/intrinsics") from exc
        if (poses.shape != (SHORT_FRAMES, 4, 4) or times.shape != (SHORT_FRAMES,)
                or not np.issubdtype(poses.dtype, np.number) or not np.issubdtype(times.dtype, np.number)
                or not np.isfinite(poses).all()
                or not np.isfinite(times).all()):
            raise DatasetError("Invalid ORBIA camera shape/non-finite values")
        R = poses[:, :3, :3]
        if (not np.allclose(poses[:, 3], [0, 0, 0, 1], atol=1e-9, rtol=0)
                or not np.allclose(R.swapaxes(1, 2) @ R, np.eye(3), atol=1e-6, rtol=0)
                or not np.allclose(np.linalg.det(R), 1, atol=1e-6, rtol=0)):
            raise DatasetError("ORBIA camera must be OpenCV C2W proper SE(3)")
        if not np.allclose(times, np.arange(SHORT_FRAMES) / FPS, atol=1e-6, rtol=0):
            raise DatasetError("ORBIA timestamps must preserve original 16 FPS timeline")
        if (K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0
                or not np.allclose(K[2], [0, 0, 1], atol=1e-9, rtol=0)):
            raise DatasetError("Invalid ORBIA intrinsics K")
        try:
            dimensions = (doc["intrinsics"]["width"], doc["intrinsics"]["height"])
            if any(type(v) is not int or v <= 0 for v in dimensions):
                raise DatasetError("Invalid ORBIA image dimensions")
            if image_dimensions(paths["q0.png"]) != dimensions:
                raise DatasetError("ORBIA image/intrinsics resolution mismatch")
            events = doc["events"]
            for required in ("q0", "qe1", "qc_first", "qc_revisit"):
                if required not in events:
                    raise DatasetError(f"Missing official event: {required}")
            for event in events.values():
                i = event["frame_index"]
                if type(i) is not int or not 0 <= i < SHORT_FRAMES:
                    raise DatasetError("ORBIA event outside trajectory")
                if abs(float(event["timestamp_s"]) - times[i]) > 1e-6:
                    raise DatasetError("ORBIA event time differs from camera timestamp")
            pairs = doc["revisit_pairs"]
            if "C" not in pairs:
                raise DatasetError("Missing official revisit pair C")
            for pair in pairs.values():
                for role in ("first", "revisit"):
                    if pair[role + "_frame_index"] != events[pair[role + "_event"]]["frame_index"]:
                        raise DatasetError("ORBIA revisit/event mismatch")
            caption = paths["prompt.txt"].read_bytes().decode("utf-8")
            required_text(caption, "ORBIA original caption")
        except (KeyError, TypeError, UnicodeError, ValueError) as exc:
            raise DatasetError(f"Invalid ORBIA public input: {exc}") from exc
        for array in (poses, times, K):
            array.flags.writeable = False
        return ORBIAInput("orbia", case_id, paths["q0.png"], caption, poses, K, times,
                          frozen(events), frozen(doc), self.manifest_sha256,
                          frozen({name: self.files[prefix + name]["sha256"] for name in PUBLIC_FILES}))

    def load_cases(self, case_ids: Sequence[str] | None = None) -> tuple[ORBIAInput, ...]:
        ids = self.select(case_ids)
        return tuple(self.load_case(cid) for cid in ids)

    def select(self, case_ids: Sequence[str] | None = None) -> tuple[str, ...]:
        if case_ids is None:
            return self.case_ids
        ids = tuple(case_ids)
        if not ids or len(set(ids)) != len(ids) or any(cid not in self.case_ids for cid in ids):
            raise DatasetError("Subset must contain unique frozen ORBIA short case IDs")
        selected = tuple(cid for cid in self.case_ids if cid in ids)
        if selected != ids:
            raise DatasetError("ORBIA subset must preserve official manifest order")
        return selected


class ORBIADatasetResolver:
    def __init__(self, repo_root: str | Path, environ: Mapping | None = None,
                 *, lock: str | Path = DEFAULT_LOCK):
        self.repo_root = Path(repo_root).resolve()
        self.environ = os.environ if environ is None else environ
        self.lock = load_lock(lock)

    def _path(self, value) -> Path:
        p = Path(value)
        return (p if p.is_absolute() else self.repo_root / p).resolve()

    def resolve(self, spec: ORBIASpec = ORBIASpec()) -> ORBIADataset:
        if spec.stage not in ("inference", "references"):
            raise DatasetError("stage must be inference or references")
        root = spec.dataset_root or self.environ.get("DRAFTKV_ORBIA_ROOT")
        index = spec.index or self.environ.get("DRAFTKV_ORBIA_INDEX")
        pin = spec.index_sha256 or self.environ.get("DRAFTKV_ORBIA_INDEX_SHA256")
        if not root or not index or not pin:
            raise ORBIADependencyError("ORBIA requires dataset-root and externally pinned installation index")
        root, index = self._path(root), self._path(index)
        expected = require_sha(pin, "ORBIA index SHA256")
        if sha256_file(index) != expected:
            raise DatasetError("ORBIA index SHA256 mismatch")
        receipt = unique_json(index.read_bytes())
        if (receipt.get("schema_version") != 1 or receipt.get("dataset_family") != "orbia"
                or receipt.get("official_commit") != self.lock["official_commit"]):
            raise DatasetError("ORBIA rejects WorldAtlas/unpinned installation index")
        raw = contained_file(root, "manifest.json").read_bytes()
        manifest, ids = validate_manifest(raw, self.lock)
        digest = hashlib.sha256(raw).hexdigest()
        if receipt.get("manifest_sha256") != digest or receipt.get("case_ids") != list(ids):
            raise DatasetError("ORBIA manifest/index hash or case order mismatch")
        if (receipt.get("distribution_revision") != self.lock["distribution"]["revision"]
                or receipt.get("manifest_git_blob_sha1") != self.lock["distribution"]["manifest"]["git_blob_sha1"]
                or receipt.get("data_version") != "Orbia/Orbia:manifest-sha256:" + digest
                or receipt.get("benchmark") != "bench2"
                or receipt.get("stage") not in ("inference", "references")):
            raise DatasetError("ORBIA installation receipt differs from the frozen distribution/version")
        if spec.stage == "references" and receipt.get("stage") != "references":
            raise ORBIADependencyError("Official reference assets not installed; run reference stage")
        files = receipt.get("files")
        if not isinstance(files, dict):
            raise DatasetError("Missing ORBIA per-file index")
        dataset = ORBIADataset(root, digest, expected, required_text(receipt.get("data_version"), "data version"),
                               ids, frozen(files), spec.stage, frozen(manifest), self.lock["official_commit"])
        checked = set()
        for cid in ids:
            for name in PUBLIC_FILES:
                relative = f"public/{cid}/{name}"
                dataset._checked(relative)
                checked.add(relative)
            if spec.stage == "references":
                relative = f"references/{cid}/reference.json"
                ref = dataset._checked(relative)
                checked.add(relative)
                doc = unique_json(ref.read_bytes())
                if doc.get("case_id") != cid:
                    raise DatasetError("Reference identity mismatch")
                for name in reference_files(doc):
                    relative = f"references/{cid}/{name}"
                    dataset._checked(relative)
                    checked.add(relative)
        for name in files:
            parts = Path(name).parts
            if len(parts) < 3 or parts[0] not in ("public", "references") or parts[1] not in ids:
                raise DatasetError("Installation index contains non-ORBIA short-case assets")
            if name not in checked and (parts[0] == "public" or spec.stage == "references"):
                dataset._checked(name)
        return dataset
