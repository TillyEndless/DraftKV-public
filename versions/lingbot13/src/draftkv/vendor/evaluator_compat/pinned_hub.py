"""Resolve two already hash-verified official Hub trees without a network probe.

No model function, arguments, weights, or evaluator formula is replaced. The
VBench preflight authenticates these complete source trees before scoring.
"""
import functools
import os
from pathlib import Path


def install():
    import torch.hub
    original = torch.hub.load
    trees = {"facebookresearch/dino": "facebookresearch_dino_main",
             "facebookresearch/dinov2": "facebookresearch_dinov2_main"}

    @functools.wraps(original)
    def load(repo_or_dir, model, *args, **kwargs):
        repo, _, revision = str(repo_or_dir).partition(":")
        if kwargs.get("source", "github") == "github" and repo in trees and revision in ("", "main"):
            path = Path(os.environ["TORCH_HOME"]) / "hub" / trees[repo]
            if not (path / "hubconf.py").is_file():
                raise FileNotFoundError(f"pinned offline Hub source missing: {path}")
            repo_or_dir = str(path)
            kwargs["source"] = "local"
        return original(repo_or_dir, model, *args, **kwargs)

    torch.hub.load = load
