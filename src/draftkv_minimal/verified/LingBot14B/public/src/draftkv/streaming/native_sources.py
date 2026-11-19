# SPDX-License-Identifier: Apache-2.0
"""Assemble explicitly verified native build inputs outside the public package.

Only project-owned glue ships here. FA/B01-derived headers are read from the
caller's authorized runtime root, never downloaded or located in private caches.
"""
from importlib.resources import files
from pathlib import Path
import hashlib
import json
import os
import tempfile

from draftkv.runtime.io import require
from draftkv.streaming.external import runtime_root
from draftkv.streaming.dependency import requirement


def native_contract():
    return json.loads(files('draftkv').joinpath('data/native_sources.json').read_text())


def verified_archive():
    path = runtime_root() / 'src/draftkv/vendor/b01/source_snapshot.tar.gz'
    require(path.is_file(), 'EXTERNAL_RUNTIME_DEPENDENCY_REQUIRED: install the authorized B01 source_snapshot.tar.gz under source_root; see docs/INSTALL.md')
    require(hashlib.sha256(path.read_bytes()).hexdigest() == requirement()['source_archive_sha256'], 'B01 archive SHA256 differs; no substitution permitted')
    return path


def verify_external_native(root):
    root = Path(root).resolve()
    result = []
    for row in native_contract()['external_sources']:
        path = root / row['source_relative']
        require(path.resolve().is_relative_to(root), 'External native source escapes explicit root')
        require(path.is_file(), 'EXTERNAL_RUNTIME_DEPENDENCY_REQUIRED: authorized native header missing: ' + row['source_relative'])
        require(hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256'], 'SOURCE_HASH_MISMATCH: ' + row['source_relative'])
        result.append(row['source_relative'])
    return result


def assemble_native_sources():
    """Runtime-only content-addressed source assembly; performs no compilation."""
    root = runtime_root()
    verify_external_native(root)
    public = files('draftkv.algorithms.draftkv.historical.orbia.streaming')
    payload = {}
    for row in native_contract()['public_sources']:
        raw = public.joinpath(row['relative']).read_bytes()
        require(hashlib.sha256(raw).hexdigest() == row['sha256'], 'PUBLIC_NATIVE_HASH_MISMATCH: ' + row['relative'])
        payload[row['relative']] = raw
    for row in native_contract()['external_sources']:
        payload[row['relative']] = (root / row['source_relative']).read_bytes()
    digest = hashlib.sha256()
    for name, raw in sorted(payload.items()):
        digest.update(name.encode()); digest.update(raw)
    cache = Path(os.environ.get('DRAFTKV_PREPACK_BUILD', str(Path(tempfile.gettempdir()) / 'draftkv-native-build'))).resolve()
    package = Path(__file__).resolve().parents[1]
    require(not cache.is_relative_to(package), 'Build cache cannot be inside the public package')
    destination = cache / ('sources-' + digest.hexdigest())
    for name, raw in payload.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            require(target.read_bytes() == raw, 'Native source cache content mismatch')
        else:
            # Atomic publication also permits independent distributed ranks.
            fd, temporary = tempfile.mkstemp(dir=target.parent)
            try:
                with os.fdopen(fd, 'wb') as stream: stream.write(raw)
                os.replace(temporary, target)
            finally:
                if os.path.exists(temporary): os.unlink(temporary)
    return destination
