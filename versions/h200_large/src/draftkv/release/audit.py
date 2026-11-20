"""CPU-only release binding checks, never a model/GPU readiness certificate.

The release wrapper and the plan builder share the same current-checkout file
snapshot. Historical receipts, example deployments, caches and model assets do
not participate in this snapshot. Native has no algorithm plugin, but still
needs the same adapter/config drift protection as plugin methods.
"""
import csv
import hashlib
import json
import os
from pathlib import Path

from draftkv.registry import Registry, algorithm_package
from draftkv.runtime.config import json_sha256


class ReleaseBindingError(ValueError):
    pass


RESOURCE_KEYS = ("KVRAG_NATIVE_DIRECT_B01", "LINGBOT14_LOW_MEMORY_STREAM",
                 "LINGBOT14_FSDP_FULL_SHARD")
CONFIG_FILES = (
    "configs/backbones/algorithm_sources.json",
    "configs/backbones/source_locks.json",
    "configs/backbones/solarwm_source_lock.json",
    "configs/backbones/effective_budgets.json",
    "configs/algorithms/draftkv/scorer.json",
    "configs/algorithms/draftkv/a2_frozen_defaults.json",
    "configs/algorithms/draftkv/release_bindings.json",
    "configs/algorithms/worldtrace_landmark/policy.json",
    "configs/profiles/lingbot14b_draftkv_sp8_bf16_20261102.json",
)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(root, value):
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ReleaseBindingError(f"nonportable release file: {value}")
    actual = (root / path).resolve()
    if not actual.is_relative_to(root) or not actual.is_file():
        raise ReleaseBindingError(f"release file missing or outside checkout: {value}")
    return actual


def release_source_files(root, *, require_complete=True):
    """Current adapter and directly consumed release configs, without self-locks."""
    root = Path(root).resolve()
    source = root / "src/draftkv"
    if not source.is_dir():
        if require_complete:
            raise ReleaseBindingError("release adapter source tree missing")
        # Small historical plan fixtures need not pretend to be a checkout.
        return {}
    names = {p.relative_to(root).as_posix() for p in source.rglob("*.py")}
    names.update(CONFIG_FILES)
    names.update(p.relative_to(root).as_posix()
                 for p in (root / "configs/algorithms").glob("*/source_lock.json"))
    if not require_complete:
        names = {name for name in names if (root / name).is_file()}
    return {name: _sha(_relative(root, name)) for name in sorted(names)}


def validate_plugin_binding(root, group, plugin, *, expected_files=None):
    """Require official release methods to use this checkout's canonical entry.

    Caller decides whether a plan is an official release. Diagnostic fixtures
    and extension registries are not silently promoted to release groups.
    """
    root = Path(root).resolve()
    if group.algorithm == "native":
        if plugin:
            raise ReleaseBindingError("Native may not bind an algorithm plugin")
        return
    expected_entry = f"src/draftkv/algorithms/{algorithm_package(group.algorithm)}/entry.py"
    if not isinstance(plugin, dict) or plugin.get("entry") != expected_entry:
        raise ReleaseBindingError(f"canonical plugin required for {group.algorithm}: {expected_entry}")
    configured_root = Path(plugin.get("root", "."))
    configured_root = (root / configured_root if not configured_root.is_absolute()
                       else configured_root).resolve()
    if configured_root != root:
        raise ReleaseBindingError("release plugin root must be the current checkout")
    files = plugin.get("files")
    if not isinstance(files, dict):
        raise ReleaseBindingError("release plugin source lock missing")
    expected_files = release_source_files(root) if expected_files is None else expected_files
    if files != expected_files:
        raise ReleaseBindingError("release plugin adapter/config snapshot differs; regenerate the source-bound configuration")
    if group.algorithm == "infinigen" and plugin.get("config", {}).get("scoring_granularity", "frame") != "frame":
        raise ReleaseBindingError("InfiniGen release requires frame scoring")


def _locked_files(root, files, prefix=""):
    if not isinstance(files, dict) or not files:
        raise ReleaseBindingError("source-lock file map is empty")
    for name, digest in files.items():
        path = _relative(root, str(Path(prefix) / name))
        if not isinstance(digest, str) or len(digest) != 64 or _sha(path) != digest:
            raise ReleaseBindingError(f"source-lock drift: {path.relative_to(root).as_posix()}")


def _loader_probe(backbone, algorithm):
    # Import lazily: this module is also used by planning and source-lock tools.
    from draftkv.backbones.models import native_capture_loading
    before = {key: os.environ.get(key) for key in RESOURCE_KEYS}
    records = []
    with native_capture_loading({"backbone": backbone, "algorithm": algorithm},
                                lambda event, **fields: records.append((event, fields))):
        during = {key: os.environ.get(key) for key in RESOURCE_KEYS}
    after = {key: os.environ.get(key) for key in RESOURCE_KEYS}
    if after != before:
        raise ReleaseBindingError("constructor resource controls leaked into generation")
    if backbone == "LingBot14B" and any(during[key] != "1" for key in RESOURCE_KEYS):
        raise ReleaseBindingError("LingBot14B unified constructor resource policy missing")
    if backbone != "LingBot14B" and (during != before or records):
        raise ReleaseBindingError("LingBot14B resource policy changed another backbone")
    return "BF16_STREAMING_FULL_SHARD_CONSTRUCTOR_ONLY" if backbone == "LingBot14B" else "BACKBONE_NATIVE_POLICY"


def audit_release_sources(root, groups=None):
    """Audit source bindings and actual constructor context using no CUDA/model."""
    root = Path(root).resolve()
    selected = tuple(Registry().groups if groups is None else groups)
    official = {g.key for g in Registry().groups}
    gaps = []
    try:
        snapshot = release_source_files(root)
    except (OSError, ValueError) as exc:
        snapshot = {}
        gaps.append(str(exc))
    backbone_locks = json.loads((root / "configs/backbones/source_locks.json").read_text())
    for backbone in sorted({g.backbone for g in selected} - {"SolarWM33B"}):
        prefix = "src/draftkv/vendor/backbones/" + ("longlive13" if backbone == "LongLive1.3B" else "lingbot14")
        try:
            _locked_files(root, backbone_locks[backbone]["files"], prefix)
        except (OSError, KeyError, ValueError) as exc:
            gaps.append(f"{backbone}: {exc}")
    for algorithm in sorted({g.algorithm for g in selected}):
        lock = root / f"configs/algorithms/{algorithm_package(algorithm)}/source_lock.json"
        if lock.is_file():
            try:
                _locked_files(root, json.loads(lock.read_text())["files"])
            except (OSError, KeyError, ValueError) as exc:
                gaps.append(f"{algorithm}: {exc}")
    inventory_path = root / "configs/backbones/algorithm_sources.json"
    try:
        inventory = json.loads(inventory_path.read_text())
        rows = {tuple(r[k] for k in ("benchmark", "backbone", "algorithm")): r
                for r in inventory["groups"]}
        for group in selected:
            current = rows[group.key]
            _locked_files(root, current["current_source_inventory"])
            entry = current.get("current_plugin_entry")
            if entry is not None:
                expected = f"src/draftkv/algorithms/{algorithm_package(group.algorithm)}/entry.py"
                if group.algorithm == "native" or entry != expected:
                    raise ReleaseBindingError(f"current plugin metadata is not canonical: {'/'.join(group.key)}")
                if _sha(_relative(root, entry)) != current.get("current_plugin_sha256"):
                    raise ReleaseBindingError(f"current_plugin_sha256 drift: {'/'.join(group.key)}")
            if "importance_config_sha256" in current:
                config_path = current.get("importance_config")
                if (not isinstance(config_path, str) or
                        _sha(_relative(root, config_path)) != current["importance_config_sha256"]):
                    raise ReleaseBindingError(f"importance_config_sha256 drift: {'/'.join(group.key)}")
    except (OSError, KeyError, ValueError) as exc:
        gaps.append(f"current algorithm source inventory: {exc}")
    try:
        binding = json.loads((root / "configs/algorithms/draftkv/release_bindings.json").read_text())
        scorer = binding["importance_scorer"]
        if _sha(_relative(root, scorer["path"])) != scorer["sha256"]:
            raise ReleaseBindingError("DraftKV scorer configuration source-lock drift")
    except (OSError, KeyError, ValueError) as exc:
        gaps.append(str(exc))
    rows = []
    for group in selected:
        row_gaps = []
        entry = ("src/draftkv/backbones/runtime.py" if group.algorithm == "native" else
                 f"src/draftkv/algorithms/{algorithm_package(group.algorithm)}/entry.py")
        try:
            if group.key not in official:
                raise ReleaseBindingError("unregistered release group")
            entry_sha = _sha(_relative(root, entry))
            loader = _loader_probe(group.backbone, group.algorithm)
        except (OSError, ValueError) as exc:
            entry_sha, loader = "", "UNVERIFIED"
            row_gaps.append(str(exc))
        rows.append(dict(benchmark=group.benchmark, backbone=group.backbone,
                         method=group.algorithm, entry=entry, source_sha=entry_sha,
                         adapter_config_sha=json_sha256(snapshot), low_memory_loader=loader,
                         default_command_valid="CPU_BINDING_VALID" if not gaps and not row_gaps else "BLOCKED",
                         gpu_runtime="NOT_TESTED", reason="; ".join(gaps + row_gaps)))
    all_gaps = gaps + [row["reason"] for row in rows if row["reason"] and row["reason"] not in gaps]
    return dict(status="BLOCKED" if all_gaps else "CPU_BINDING_VALID", scope="CPU_SOURCE_AND_CONSTRUCTOR_CONTEXT_ONLY",
                group_count=len(rows), registry_counts={b: sum(r["benchmark"] == b for r in rows) for b in ("bench1", "bench2")},
                adapter_config_sha=json_sha256(snapshot), rows=rows, gaps=list(dict.fromkeys(all_gaps)),
                runtime_assets="NOT_TESTED", kernel_abi="NOT_TESTED", gpu_forward="NOT_TESTED", release_e2e=False)


def require_release_sources(root, groups=None):
    report = audit_release_sources(root, groups)
    if report["status"] == "BLOCKED":
        raise ReleaseBindingError("BLOCKED_RELEASE_BINDING: " + "; ".join(report["gaps"]))
    return report


def write_source_audit_csv(report, path):
    columns = ("benchmark", "backbone", "method", "entry", "source_sha", "adapter_config_sha",
               "low_memory_loader", "default_command_valid", "gpu_runtime", "reason")
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(report["rows"])
