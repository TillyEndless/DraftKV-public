"""Explicit, source-verified access to external kernel and native dependencies.

No external source is located by searching development checkouts or caches.
Only a caller-supplied runtime root and the packaged historical locks are used.
"""
from importlib import import_module, util
from pathlib import Path
import hashlib
import os
import sys
import types

from draftkv.runtime.io import require
from draftkv.streaming.integrity import source_lock


def runtime_root():
    value = os.environ.get('DRAFTKV_EXTERNAL_RUNTIME_ROOT')
    require(bool(value), 'EXTERNAL_RUNTIME_DEPENDENCY_REQUIRED: configure source_root in the asset config; see docs/INSTALL.md')
    root = Path(value).resolve()
    require(root.is_dir(), 'EXTERNAL_RUNTIME_DEPENDENCY_REQUIRED: source_root is missing')
    return root


def external_path(profile, relative):
    require(profile in ('longlive', 'orbia'), 'Unknown historical core profile')
    root = runtime_root()
    rel = Path(relative)
    require(not rel.is_absolute() and '..' not in rel.parts, 'Unsafe external dependency path')
    path = root / rel
    require(path.resolve().is_relative_to(root), 'External dependency escapes explicit source root')
    lock = source_lock('longlive' if profile == 'longlive' else 'lingbot')['files']
    if path.is_file():
        expected = lock.get(relative)
        require(expected is not None, 'External dependency has no historical fingerprint: ' + relative)
        require(hashlib.sha256(path.read_bytes()).hexdigest() == expected, 'SOURCE_HASH_MISMATCH: ' + relative)
    return path


def _namespace(name, path):
    existing = sys.modules.get(name)
    if existing is not None:
        locations = getattr(existing, '__path__', ())
        require(str(path) in locations, 'External package already loaded from another source: ' + name)
        return existing
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    module.__package__ = name
    sys.modules[name] = module
    return module


def external_symbol(profile, module, symbol):
    """Load an actual pinned dependency symbol, never a stand-in implementation."""
    require(module.startswith('draftkv.vendor.b01'), 'Undeclared external kernel module')
    root = runtime_root()
    _namespace('draftkv.vendor', root / 'src/draftkv/vendor')
    # Verify every Python wrapper before the package loader can execute imports.
    lock = source_lock('longlive' if profile == 'longlive' else 'lingbot')['files']
    for rel in lock:
        if rel.startswith('src/draftkv/vendor/b01/') and rel.endswith('.py'):
            path = external_path(profile, rel)
            require(path.is_file(), 'EXTERNAL_RUNTIME_DEPENDENCY_REQUIRED: ' + rel)
    parent = 'draftkv.vendor.b01'
    if parent not in sys.modules:
        path = external_path(profile, 'src/draftkv/vendor/b01/__init__.py')
        spec = util.spec_from_file_location(parent, path, submodule_search_locations=[str(path.parent)])
        loaded = util.module_from_spec(spec)
        sys.modules[parent] = loaded
        try:
            spec.loader.exec_module(loaded)
        except BaseException:
            sys.modules.pop(parent, None)
            raise
    result = import_module(module)
    require(Path(result.__file__).resolve().is_relative_to(root), 'External dependency source drift')
    external_path(profile, str(Path(result.__file__).resolve().relative_to(root)))
    return getattr(result, symbol)
