"""Portable native backbone adapters and version-1 algorithm hook ABI."""

from .catalog import coverage, model_spec
from .contracts import ABI_VERSION, BackboneError, InputMode
from .runtime import NativeBackboneAdapter


def create_adapter(context):
    """Factory for adapter.plugin in the existing unified Runtime."""
    return NativeBackboneAdapter(context)


__all__ = ["ABI_VERSION", "BackboneError", "InputMode", "coverage", "model_spec", "create_adapter"]
