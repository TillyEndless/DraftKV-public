"""Source integrity and error taxonomy. Standard library only."""

import hashlib
from functools import lru_cache
import io
import json
import os
import time
from pathlib import Path
import tarfile

from ._source_lock import MANIFEST_SHA256

ROOT = Path(__file__).resolve().parent


class B01Error(RuntimeError):
    def __init__(self, category, message):
        self.category = category
        super().__init__(f"{category}: {message}")


def _file_identity(stat):
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _hash_file(path, identity):
    # Never cache a pathname alone. In-place edits (even with restored mtime),
    # atomic replacement and truncation invalidate the inode/ctime identity.
    result = hashlib.sha256()
    with open(path, 'rb') as stream:
        if _file_identity(os.fstat(stream.fileno())) != identity:
            raise B01Error('LogicError', 'file changed before hashing')
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
        if _file_identity(os.fstat(stream.fileno())) != identity:
            raise B01Error('LogicError', 'file changed while hashing')
    return result.hexdigest()


_file_digest = lru_cache(maxsize=256)(_hash_file)


def digest(path):
    """Full SHA on first use/change; reuse only the same observed file version.

    This is a process-local optimization for ordinary source/build files, not
    an adversarial-filesystem guarantee. Every call still stats the file and
    every caller still compares its digest against the expected SHA.
    """
    path = Path(path).resolve(strict=True)
    identity = _file_identity(path.stat())
    # Some Linux/NFS timestamps advance only once per clock tick. A newly
    # written file can be edited twice with the same size/mtime/ctime tuple.
    # Never populate or consult the memo while timestamps are recent/future.
    # Published build artifacts become eligible once their version settles.
    settled = time.time_ns() - max(identity[-2:]) > 2_000_000_000
    result = (_file_digest if settled else _hash_file)(str(path), identity)
    if _file_identity(path.stat()) != identity:
        raise B01Error('LogicError', 'file changed during verification')
    return result


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def confined(root, relative):
    root = Path(root).resolve()
    path = root / relative
    if Path(relative).is_absolute() or ".." in Path(relative).parts or path.is_symlink():
        raise B01Error("PathError", f"unsafe relative path: {relative}")
    if not path.resolve().is_relative_to(root):
        raise B01Error("PathError", f"path escapes root: {relative}")
    return path


def verify_sources(root=ROOT):
    root = Path(root)
    try:
        if digest(root / "source_manifest.json") != MANIFEST_SHA256:
            raise B01Error("LogicError", "source manifest hash mismatch")
        manifest = json.loads((root / "source_manifest.json").read_text())
        for archive_key, entries in (("source_archive", manifest["files"]),
                                     ("evidence_archive", [e for e in manifest["evidence"] if e["path"].startswith("evidence/")])):
            archive = manifest[archive_key]
            path = confined(root, archive["path"])
            if digest(path) != archive["sha256"]:
                raise B01Error("LogicError", f"archive hash mismatch: {archive['path']}")
            files = _snapshot(str(path.resolve()), archive["sha256"])
            if set(files) != {item["path"] for item in entries}:
                raise B01Error("LogicError", "snapshot layout has extra or missing files")
            for item in entries:
                if hashlib.sha256(files[item["path"]]).hexdigest() != item["sha256"]:
                    raise B01Error("LogicError", f"snapshot file hash mismatch: {item['path']}")
        for item in manifest["evidence"]:
            if not item["path"].startswith("evidence/") and digest(confined(root, item["path"])) != item["sha256"]:
                raise B01Error("LogicError", f"license hash mismatch: {item['path']}")
        if (root / "sources").exists() or (root / "evidence").exists():
            raise B01Error("LogicError", "unreviewed loose snapshot directory")
        return manifest
    except (OSError, ValueError, KeyError) as exc:
        raise B01Error("PathError", f"invalid source bundle: {exc}") from exc


def _read_tar(data):
    """Read regular members only, with bounded sizes and no filesystem extraction."""
    files = {}
    total = 0
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
            for member in archive:
                name = member.name
                if not member.isfile() or name in files or Path(name).is_absolute() or ".." in Path(name).parts or "\\" in name:
                    raise B01Error("PathError", f"unsafe snapshot member: {name}")
                total += member.size
                if member.size > 16 * 1024**2 or total > 32 * 1024**2 or len(files) >= 256:
                    raise B01Error("LogicError", "snapshot exceeds audited size/count bounds")
                files[name] = archive.extractfile(member).read()
        return files
    except (tarfile.TarError, EOFError, OSError) as exc:
        raise B01Error("LogicError", f"invalid snapshot archive: {exc}") from exc


@lru_cache(maxsize=4)
def _snapshot(path, expected_hash):
    # Outer verification rehashes the archive on every call. Cache only the
    # decompressed immutable bytes; a changed archive never reuses this cache.
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_hash:
        raise B01Error("LogicError", "snapshot changed while verifying")
    return _read_tar(data)


def snapshot_files(root=ROOT, *, evidence=False):
    manifest = verify_sources(root)
    archive = manifest["evidence_archive" if evidence else "source_archive"]
    # Return a fresh mapping; immutable member bytes cannot be modified in place.
    return dict(_snapshot(str(confined(root, archive["path"]).resolve()), archive["sha256"]))


def extract_sources(destination, root=ROOT):
    """CPU-only, hash-verified extraction into a NEW external build directory."""
    destination = Path(destination).resolve()
    if destination.is_relative_to(ROOT.parents[3]):
        raise B01Error("PathError", "source extraction must be outside the release checkout")
    if destination.exists():
        raise B01Error("PathError", "source extraction destination must be new")
    files = snapshot_files(root)
    destination.mkdir(parents=True)
    for relative, data in files.items():
        path = confined(destination, relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as output:
            output.write(data)
    return destination / "sources"


def source_generation(root=ROOT):
    verify_sources(root)
    return MANIFEST_SHA256
