"""Normalize release input aliases before installation or request binding."""
from pathlib import Path


def dataset_root_override(args):
    values = [getattr(args, name, None) for name in ("dataset_root", "dataset_dir", "orbia_root")]
    roots = {str(Path(value).expanduser().resolve()) for value in values if value}
    if len(roots) > 1:
        raise ValueError("conflicting dataset roots; use one --dataset-root/--dataset-dir/--orbia-root")
    return next(iter(roots), None)
