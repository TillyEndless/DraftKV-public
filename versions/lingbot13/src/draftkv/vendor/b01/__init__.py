"""Original B01 snapshots plus an explicit, fail-closed build/runtime contract.

Importing this package never imports Torch, compiles CUDA, or discovers GPUs.
"""

from .contract import PlanSpec, BoundPlan
from .integrity import B01Error, verify_sources, source_generation

__all__ = ["B01Error", "PlanSpec", "BoundPlan", "verify_sources", "source_generation"]
