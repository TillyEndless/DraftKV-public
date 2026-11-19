"""Logical Bench1 dataset roots; no host-specific absolute paths."""
import os
from pathlib import Path
from draftkv.registry import BENCH1_DATASETS


def dataset_root(backbone, cache, repo, override=None):
    if backbone not in BENCH1_DATASETS:
        raise ValueError("Bench1 supports only LongLive1.3B text-only WorldAttention160")
    identity = BENCH1_DATASETS[backbone]
    variable = "DRAFTKV_BENCH1_T2V_ROOT"
    explicit = override or os.environ.get(variable)
    if explicit:
        result = Path(explicit).expanduser().resolve()
        if not result.is_dir():
            raise ValueError(f"missing {identity} dataset root: {result}")
        return result
    candidates = [Path(cache) / "datasets" / identity]
    parent = os.environ.get("DRAFTKV_DATASET_HOME")
    if parent:
        candidates.insert(0, Path(parent).expanduser() / identity)
    candidates.append(Path(repo) / "configs/datasets/bench1")
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    raise ValueError(f"missing {identity}; set {variable}, --dataset-root, or install its assets")
