"""Portable layered JSON configuration; see runtime/CONTRACT.md for the ABI.

Config is selected by --config or DRAFTKV_CONFIG. Relative paths use the Git
root; ${ENV_VAR}/~ paths are expanded and undefined variables fail. Merge order
is defaults -> backbone -> algorithm -> exact group, recursively for objects.
Release dataset assets require a frozen external SHA-256 pin during execution;
all configured assets are hashed before loading a model. Directory hashing is
defined by asset_sha256(). Optional integrations are never imported to plan.
"""

import copy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

from draftkv.registry import Registry


class ConfigurationError(ValueError):
    pass


def repo_root(explicit: str | None = None) -> Path:
    requested = Path(explicit).expanduser().resolve() if explicit else None
    if requested is not None and not requested.is_dir():
        raise ConfigurationError(f"repository root does not exist: {requested}")
    command = ["git", "-C", str(requested)] if requested is not None else ["git"]
    try:
        result = subprocess.run(
            [*command, "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ConfigurationError(
            "cannot find Git root; provide --repo-root pointing to a Git repository"
        ) from exc
    root = Path(result.stdout.strip()).resolve()
    if requested is not None and requested != root:
        raise ConfigurationError(
            f"--repo-root must be the Git root {root}, not {requested}"
        )
    return root


def resolve_path(value: str, root: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError("path must be a nonempty string")
    expanded = os.path.expanduser(os.path.expandvars(value))
    if re.search(r"\$(?:\{[^}]*\}|[A-Za-z_][A-Za-z0-9_]*)", expanded):
        raise ConfigurationError(f"undefined environment variable in path: {value}")
    path = Path(expanded)
    return (root / path if not path.is_absolute() else path).resolve()


def parse_devices(value: str) -> tuple[int, ...]:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]+(?:,[0-9]+)*", value):
        raise ConfigurationError(
            "--devices must be comma-separated nonnegative CUDA IDs, e.g. 0,2,3,4"
        )
    parts = tuple(part.lstrip("0") or "0" for part in value.split(","))
    if any(len(part) > 10 for part in parts):
        raise ConfigurationError(
            "CUDA IDs must fit a nonnegative signed 32-bit integer"
        )
    devices = tuple(int(part) for part in parts)
    if len(devices) not in (1, 2, 4, 8):
        raise ConfigurationError(
            "SP must be 1, 2, 4 or 8 (the number of selected devices)"
        )
    if len(set(devices)) != len(devices):
        raise ConfigurationError("--devices must contain unique CUDA IDs")
    if any(device > 2**31 - 1 for device in devices):
        raise ConfigurationError(
            "CUDA IDs must fit a nonnegative signed 32-bit integer"
        )
    return devices


def merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        result[key] = (
            merge(result[key], value)
            if isinstance(value, dict) and isinstance(result.get(key), dict)
            else copy.deepcopy(value)
        )
    return result


def json_sha256(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


class Config:
    def __init__(self, root: Path, path: str | None = None):
        self.root = root
        path = path if path is not None else os.environ.get("DRAFTKV_CONFIG")
        self.path = resolve_path(path, root) if path else None
        self.data = {}
        if self.path:
            try:
                self.data = json.loads(
                    self.path.read_text(encoding="utf-8"),
                    parse_constant=lambda v: self._invalid_constant(v),
                    object_pairs_hook=self._unique_keys,
                )
            except (OSError, ValueError) as exc:
                raise ConfigurationError(
                    f"cannot load config {self.path}: {exc}"
                ) from exc
        if not isinstance(self.data, dict):
            raise ConfigurationError("config must be a JSON object")
        version = self.data.get("schema_version", 1)
        if type(version) is not int or version != 1:
            raise ConfigurationError("unsupported config schema_version; expected 1")
        for key, expected in (
            ("defaults", dict),
            ("groups", list),
            ("registry", list),
            ("backbones", dict),
            ("algorithms", dict),
        ):
            if key in self.data and not isinstance(self.data[key], expected):
                raise ConfigurationError(f"config {key} must be a {expected.__name__}")
        unknown = set(self.data) - {
            "schema_version",
            "defaults",
            "groups",
            "registry",
            "backbones",
            "algorithms",
        }
        if unknown:
            raise ConfigurationError(f"unknown config keys: {sorted(unknown)}")
        self.registry = Registry(self.data.get("registry", []))
        self.overrides = {}
        for entry in self.data.get("groups", []):
            if not isinstance(entry, dict):
                raise ConfigurationError("config groups entries must be objects")
            try:
                if any(
                    not isinstance(entry[key], str)
                    for key in ("benchmark", "backbone", "algorithm")
                ):
                    raise ConfigurationError(
                        "group benchmark/backbone/algorithm must be strings"
                    )
                (group,) = self.registry.select(
                    entry["benchmark"], entry["backbone"], entry["algorithm"]
                )
            except KeyError as exc:
                raise ConfigurationError(f"group config missing {exc.args[0]}") from exc
            if group.key in self.overrides:
                raise ConfigurationError(f"duplicate group config: {group.key}")
            if "release_target" in entry:
                raise ConfigurationError(
                    "release_target is owned by Registry, not group configuration"
                )
            self.overrides[group.key] = {
                k: v
                for k, v in entry.items()
                if k not in ("benchmark", "backbone", "algorithm")
            }

    @staticmethod
    def _invalid_constant(value):
        raise ValueError(f"non-finite JSON number: {value}")

    @staticmethod
    def _unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def resolve(self, group, devices: tuple[int, ...]) -> dict:
        # Reuse model and algorithm settings across benchmarks without repeating groups.
        result = copy.deepcopy(self.data.get("defaults", {}))
        for section, name in (
            ("backbones", group.backbone),
            ("algorithms", group.algorithm),
        ):
            matches = [
                v
                for k, v in self.data.get(section, {}).items()
                if k.lower() == name.lower()
            ]
            if len(matches) > 1 or any(not isinstance(v, dict) for v in matches):
                raise ConfigurationError(
                    f"{section}.{name} must be one unambiguous object"
                )
            if matches:
                result = merge(result, matches[0])
        result = merge(result, self.overrides.get(group.key, {}))
        weights = result.get("weights", {})
        if not isinstance(weights, dict):
            raise ConfigurationError("weights must be an object of named assets")
        assets = [(key, result.get(key)) for key in ("dataset", "checkpoint")]
        assets.extend((f"weights.{key}", value) for key, value in weights.items())
        for key, asset in assets:
            if asset is not None:
                if not isinstance(asset, dict) or "path" not in asset:
                    raise ConfigurationError(f"{key} must be an object with a path")
                asset["path"] = str(resolve_path(asset["path"], self.root))
                digest = asset.get("sha256")
                if digest is not None and (
                    not isinstance(digest, str)
                    or not re.fullmatch(r"[a-fA-F0-9]{64}", digest)
                ):
                    raise ConfigurationError(
                        f"{key}.sha256 must be a 64-character SHA-256 hex digest"
                    )
                if digest:
                    asset["sha256"] = digest.lower()
        if not isinstance(result.get("consumer_config", {}), dict):
            raise ConfigurationError("consumer_config must be an object")
        result.setdefault("consumer_config", {})
        if type(result.get("gpu_profile", False)) is not bool:
            raise ConfigurationError("gpu_profile must be a JSON boolean")
        result.setdefault("gpu_profile", False)
        for key in ("model_config", "algorithm_config"):
            if not isinstance(result.get(key, {}), dict):
                raise ConfigurationError(f"{key} must be an object")
            result.setdefault(key, {})
        if "model_dir" in result:
            result["model_dir"] = str(resolve_path(result["model_dir"], self.root))
        result["protocol"] = group.protocol.to_dict()
        result.update(group.to_dict())
        if group.benchmark == "bench1" and group.release_target:
            result["dataset_id"] = group.dataset_id
        result.update(
            selected_devices=list(devices),
            sp=len(devices),
            git_root=str(self.root),
            config_path=str(self.path) if self.path else None,
        )
        return result


def asset_sha256(path: Path) -> str:
    """File SHA256; directory SHA256 of sorted [relative POSIX path, file SHA256].

    Directory symlinks are rejected, including symlinked files, to avoid silent
    omission or traversal outside the configured dataset/checkpoint. Empty
    directories are rejected. Large files are streamed in 1 MiB chunks.
    """
    if path.is_file():
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    if not path.is_dir():
        raise FileNotFoundError(f"required asset not found: {path}")
    entries = []
    for item in sorted(path.rglob("*")):
        if item.is_symlink():
            raise ConfigurationError(f"asset directory contains a symlink: {item}")
        if item.is_file():
            entries.append([item.relative_to(path).as_posix(), asset_sha256(item)])
        elif not item.is_dir():
            raise ConfigurationError(f"unsupported asset file type: {item}")
    if not entries:
        raise ConfigurationError(f"asset directory is empty: {path}")
    return hashlib.sha256(
        json.dumps(entries, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()
