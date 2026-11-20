"""Content-addressed native artifacts; no implicit builds or fallback operators."""

import importlib.util
import json
from pathlib import Path
import sys

from .build import EXPORTS, MODULES, assert_abi, compatibility_generation, current_abi, check_clean_checkout
from .integrity import B01Error, canonical_hash, confined, digest, source_generation

_LOADED = {}


def cache_key(spec, toolchain):
    return canonical_hash(dict(spec=spec, toolchain=toolchain))


def check_exports(modules):
    if set(modules) != set(EXPORTS):
        raise B01Error("CUDA_ABI", "native module set mismatch")
    for name, exports in EXPORTS.items():
        for symbol in exports:
            if not callable(getattr(modules[name], symbol, None)):
                raise B01Error("CUDA_ABI", f"missing native export: {name}.{symbol}")
    producer = modules["producer_plan_ext"].ProducerPlan
    if any(not callable(getattr(producer, name, None)) for name in ("initialize", "enqueue", "bytes")):
        raise B01Error("CUDA_ABI", "ProducerPlan method set mismatch")


def _import_modules(paths):
    modules = {}
    try:
        for name, path in paths.items():
            identity = (str(Path(path).resolve()), digest(path))
            if name in _LOADED:
                old_identity, module = _LOADED[name]
                if identity != old_identity or sys.modules.get(name) is not module:
                    raise B01Error("CUDA_ABI", f"cannot mix native builds in one process: {name}")
                modules[name] = module
                continue
            if name in sys.modules:
                raise B01Error("CUDA_ABI", f"native module already loaded; use fresh process: {name}")
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            modules[name] = module
        check_exports(modules)
        for name, module in modules.items():
            _LOADED[name] = ((str(Path(paths[name]).resolve()), digest(paths[name])), module)
        return modules
    except (ImportError, OSError, AttributeError) as exc:
        raise B01Error("CUDA_ABI", f"native load failed: {exc}") from exc


def finalize_cache(directory, spec, toolchain):
    """Called only after successful native build. Verify exports before publishing receipt."""
    directory = Path(directory)
    paths = {name: directory / (name + spec["abi"]["ext_suffix"]) for name in MODULES}
    for path in paths.values():
        if not path.is_file() or not path.stat().st_size:
            raise B01Error("CUDA_ABI", f"missing built binary: {path}")
    _import_modules(paths)
    if spec["source_generation"] != source_generation() or spec["compatibility_generation"] != compatibility_generation():
        raise B01Error("CUDA_ABI", "source/build layer changed during compilation")
    check_clean_checkout(spec["clean_commit"])
    record = dict(schema=1, validation="BUILD_EXPORTS_ONLY", key=cache_key(spec, toolchain), spec=spec,
                  toolchain=toolchain, binaries={name: dict(path=path.name, sha256=digest(path)) for name, path in paths.items()})
    target = directory / "receipt.json"
    temp = directory / "receipt.json.tmp"
    temp.write_text(json.dumps(record, indent=2) + "\n")
    temp.replace(target)
    return record


def verify_cache(directory, expected_spec, actual_abi):
    assert_abi(expected_spec["abi"], actual_abi)
    if expected_spec["source_generation"] != source_generation() or expected_spec["compatibility_generation"] != compatibility_generation():
        raise B01Error("CUDA_ABI", "source or build compatibility generation changed")
    directory = Path(directory)
    try:
        record = json.loads((directory / "receipt.json").read_text())
        key = cache_key(record["spec"], record["toolchain"])
        if record["schema"] != 1 or record["validation"] != "BUILD_EXPORTS_ONLY" or record["spec"] != expected_spec or record["key"] != key or directory.name != key:
            raise B01Error("CUDA_ABI", "cache identity/ABI mismatch")
        if set(record["binaries"]) != set(MODULES):
            raise B01Error("CUDA_ABI", "cache has incomplete module set")
        paths = {}
        for name, binary in record["binaries"].items():
            if binary["path"] != name + actual_abi["ext_suffix"]:
                raise B01Error("CUDA_ABI", "binary name/Python ABI mismatch")
            path = confined(directory, binary["path"])
            if not path.stat().st_size or digest(path) != binary["sha256"]:
                raise B01Error("CUDA_ABI", f"binary hash mismatch: {name}")
            paths[name] = path
        return paths
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise B01Error("CUDA_ABI", f"corrupt or incomplete cache: {exc}") from exc


def load_native(directory, expected_spec):
    """Explicit runtime entry point. Imports current Torch but never calls CUDA init."""
    try:
        import torch
    except ImportError as exc:
        raise B01Error("MissingDependency", "model environment has no Torch") from exc
    return _import_modules(verify_cache(directory, expected_spec, current_abi(torch)))
