"""Re-runnable official-source ORBIA deployment, separate from the product CLI.

python -m draftkv.benchmarks.orbia_install --help
No GPU, sampling, reconstruction or scoring runs here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tarfile
import tempfile
import urllib.error
import urllib.request
import urllib.parse
from pathlib import Path

from .datasets import DatasetError, contained_file, require_sha, sha256_file, unique_json
from .orbia import (
    DEFAULT_LOCK, PUBLIC_FILES, ORBIADatasetResolver, ORBIADependencyError,
    ORBIASpec, load_lock, reference_files, validate_manifest, verify_distribution,
)


class _DownloadRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None and urllib.parse.urlparse(newurl).netloc != urllib.parse.urlparse(req.full_url).netloc:
            redirected.remove_header("Authorization")
        return redirected


def _request(url: str, *, start=None):
    request = urllib.request.Request(url)
    if start is not None:
        request.add_header("Range", f"bytes={start}-")
    # Authentication is optional. Never print credentials or forward the bearer
    # token to signed CDN redirects. Gated dataset approval remains external.
    if urllib.parse.urlparse(url).hostname == "huggingface.co":
        token = os.environ.get("HF_TOKEN")
        if not token:
            token_file = Path(os.environ.get("HF_TOKEN_PATH") or
                              Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "token")
            if token_file.is_file():
                token = token_file.read_text().strip()
        if token:
            request.add_header("Authorization", "Bearer " + token)
    return urllib.request.build_opener(_DownloadRedirect()).open(request, timeout=30)


def download(url: str, target: Path, *, size: int | None = None, sha256: str | None = None):
    """Bounded streaming download; authenticate before atomic promotion."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if ((size is not None and target.stat().st_size != size)
                or (sha256 and sha256_file(target) != sha256)):
            raise DatasetError(f"Cached asset integrity mismatch: {target}")
        return
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as out:
            tmp = Path(out.name)
        # Official archives may outlive a CDN connection. Resume only when the
        # server proves the exact requested byte offset with Content-Range.
        last_error = None
        for attempt in range(4):
            offset = tmp.stat().st_size
            try:
                response = _request(url, start=offset) if offset else _request(url)
                with response:
                    status = getattr(response, "status", 200)
                    if offset and status == 206:
                        content_range = response.headers.get("Content-Range", "")
                        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
                        if (not match or int(match[1]) != offset or int(match[2]) < offset
                                or (size is not None and int(match[3]) != size)):
                            raise DatasetError("Archive resume Content-Range mismatch")
                    elif offset and status == 200:
                        offset = 0  # Server ignored Range: discard partial bytes.
                    elif status not in (200, 206):
                        raise ORBIADependencyError("Unexpected archive HTTP status: " + str(status))
                    with tmp.open("ab" if offset else "wb") as out:
                        total = offset
                        while chunk := response.read(1024 * 1024):
                            total += len(chunk)
                            if size is not None and total > size:
                                raise DatasetError("Download exceeds pinned size")
                            out.write(chunk)
                if size is None or tmp.stat().st_size == size:
                    break
                last_error = ORBIADependencyError(
                    f"Truncated official download: {tmp.stat().st_size}/{size} bytes")
            except (OSError, urllib.error.URLError) as exc:
                last_error = exc
        else:
            raise ORBIADependencyError(f"Official distribution transfer incomplete after 4 attempts: {url}: {last_error}")
        if size is not None and tmp.stat().st_size != size:
            raise DatasetError("Download size mismatch")
        if sha256 and sha256_file(tmp) != sha256:
            raise DatasetError("Download SHA256 mismatch")
        tmp.replace(target)
    except (OSError, urllib.error.URLError) as exc:
        raise ORBIADependencyError(f"Official distribution download unavailable: {url}: {exc}") from exc
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def lfs_digest(pointer: bytes, asset: dict) -> str:
    verify_distribution(pointer, asset, lfs=True)
    match = re.fullmatch(rb"version https://git-lfs.github.com/spec/v1\noid sha256:([0-9a-f]{64})\nsize ([0-9]+)\n?", pointer)
    if not match or int(match[2]) != asset["size"]:
        raise DatasetError("Official LFS pointer must supply actual SHA256 and pinned size")
    digest = require_sha(match[1].decode(), "official archive SHA256")
    if asset.get("sha256") and digest != asset["sha256"]:
        raise DatasetError("Official LFS SHA256 differs from source lock")
    return digest


def extract_short(archive_path: Path, root: Path, folder: str, ids: tuple[str, ...]):
    """Extract only official short IDs; do not alter existing source files."""
    seen = set()
    extracted = []
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            parts = Path(member.name).parts
            if (Path(member.name).is_absolute() or ".." in parts or "\\" in member.name
                    or not parts or parts[0] != folder or not (member.isfile() or member.isdir())):
                raise DatasetError(f"Unsafe official archive member: {member.name}")
            if member.name in seen:
                raise DatasetError(f"Duplicate archive member: {member.name}")
            seen.add(member.name)
            if len(parts) < 2 or parts[1] not in ids:
                continue
            target = root / member.name
            if not target.resolve().is_relative_to(root.resolve()) or target.is_symlink():
                raise DatasetError("Archive member escapes dataset root")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            extracted.append(member.name)
            with archive.extractfile(member) as stream:
                if target.exists():
                    digest = hashlib.sha256()
                    while chunk := stream.read(1024 * 1024):
                        digest.update(chunk)
                    if digest.hexdigest() != sha256_file(target):
                        raise DatasetError(f"Existing dataset asset differs; use a new root: {target}")
                else:
                    tmp = None
                    try:
                        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as out:
                            tmp = Path(out.name)
                            shutil.copyfileobj(stream, out, 1024 * 1024)
                        tmp.replace(target)
                    finally:
                        if tmp is not None:
                            tmp.unlink(missing_ok=True)
    return tuple(extracted)


def install(dataset_root: Path, cache: Path, index: Path, *, stage="inference", lock_path=DEFAULT_LOCK) -> dict:
    lock = load_lock(lock_path)
    if stage not in ("inference", "references"):
        raise DatasetError("stage must be inference or references")
    root, cache, index = map(lambda p: Path(p).resolve(), (dataset_root, cache, index))
    if index.is_relative_to(root) or cache.is_relative_to(root):
        raise DatasetError("Cache and read-only index must be outside original dataset root")
    if index.exists():
        raise FileExistsError(f"Choose a new installation index: {index}")
    distribution = lock["distribution"]
    base, revision = distribution["base_url"], distribution.get("revision") or "main"
    manifest_asset = distribution["manifest"]
    manifest_path = cache / (manifest_asset["git_blob_sha1"] + ".manifest.json")
    download(f"{base}/resolve/{revision}/manifest.json", manifest_path, size=manifest_asset["size"],
             sha256=manifest_asset.get("sha256"))
    raw = manifest_path.read_bytes()
    _, ids = validate_manifest(raw, lock)
    installed_manifest = root / "manifest.json"
    if installed_manifest.exists() and installed_manifest.read_bytes() != raw:
        raise DatasetError("Existing dataset manifest differs from frozen official bytes")
    archives = {}
    # Acquire all pins before downloading any large archive.
    for folder in ("public", "references") if stage == "references" else ("public",):
        asset = distribution[folder]
        pointer = cache / (asset["git_blob_sha1"] + ".lfs")
        download(f"{base}/raw/{revision}/{asset['path']}", pointer, size=None)
        digest = lfs_digest(pointer.read_bytes(), asset)
        archives[folder] = {"sha256": digest, "size": asset["size"], "path": asset["path"],
                            "git_blob_sha1": asset["git_blob_sha1"]}
    installed_names = []
    for folder, asset in archives.items():
        archive = cache / (asset["sha256"] + ".tar.gz")
        download(f"{base}/resolve/{revision}/{asset['path']}", archive,
                 size=asset["size"], sha256=asset["sha256"])
        root.mkdir(parents=True, exist_ok=True)
        installed_names.extend(extract_short(archive, root, folder, ids))
    root.mkdir(parents=True, exist_ok=True)
    if installed_manifest.exists():
        if installed_manifest.read_bytes() != raw:
            raise DatasetError("Existing dataset manifest differs from frozen official bytes")
    else:
        with installed_manifest.open("xb") as stream:
            stream.write(raw)
    files = {}
    for cid in ids:
        names = [f"public/{cid}/{n}" for n in PUBLIC_FILES]
        if stage == "references":
            name = f"references/{cid}/reference.json"
            doc = unique_json(contained_file(root, name).read_bytes())
            if doc.get("case_id") != cid:
                raise DatasetError("Reference case identity mismatch")
            names += [name, *(f"references/{cid}/{n}" for n in reference_files(doc))]
        for name in names:
            contained_file(root, name)
    # Index every authenticated archive file, including optional geometry or
    # annotations beyond the official loader's required fields. Do not absorb
    # arbitrary pre-existing files from a directory listing into official data.
    for name in installed_names:
        path = contained_file(root, name)
        files[name] = {"sha256": sha256_file(path), "size": path.stat().st_size}
    digest = hashlib.sha256(raw).hexdigest()
    receipt = {"schema_version": 1, "dataset_family": "orbia", "benchmark": "bench2",
               "official_commit": lock["official_commit"], "stage": stage,
               "data_version": "Orbia/Orbia:manifest-sha256:" + digest,
               "distribution_revision": distribution.get("revision"),
               "manifest_sha256": digest, "manifest_git_blob_sha1": manifest_asset["git_blob_sha1"],
               "case_ids": list(ids), "archives": archives, "files": files,
               "licenses": lock["licenses"]}
    index.parent.mkdir(parents=True, exist_ok=True)
    with index.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    index.chmod(0o444)
    return {"status": "COMPLETE", "stage": stage, "case_count": len(ids),
            "manifest_sha256": digest, "dataset_root": str(root), "index": str(index),
            "index_sha256": sha256_file(index), "archives": archives}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--index-sha256", help="required for --verify-only")
    parser.add_argument("--stage", choices=("inference", "references"), default="inference")
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--parse-inputs", action="store_true", help="CPU parse every installed short case")
    parser.add_argument("--report", type=Path, required=True, help="new report path outside dataset root")
    args = parser.parse_args(argv)
    if args.report.resolve().is_relative_to(args.dataset_root.resolve()):
        parser.error("report must be outside original dataset root")
    if args.report.exists():
        raise FileExistsError(args.report)
    report = {}
    try:
        if not args.verify_only:
            report = install(args.dataset_root, args.cache, args.index, stage=args.stage, lock_path=args.lock)
        else:
            report = {"index_sha256": args.index_sha256}
        data = ORBIADatasetResolver(Path.cwd(), lock=args.lock).resolve(ORBIASpec(
            args.dataset_root, args.index, report["index_sha256"], args.stage))
        if args.parse_inputs:
            # Stream cases: do not retain all full-resolution image/camera arrays.
            for cid in data.case_ids:
                data.load_case(cid)
        report.update(status="COMPLETE", case_count=len(data.case_ids),
                      manifest_sha256=data.manifest_sha256, parsed_inputs=args.parse_inputs,
                      metrics=None)
        code = 0
    except ORBIADependencyError as exc:
        report.update(status=exc.status, reasons=[str(exc)], metrics=None)
        code = 2
    except (DatasetError, OSError, tarfile.TarError) as exc:
        report.update(status="FAILED", reasons=[str(exc)], metrics=None)
        code = 1
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
