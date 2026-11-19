"""CPU-import-safe public integration ABI. See runtime/CONTRACT.md.

Declarations do not establish model support or release readiness. All optional
model, kernel, dataset and evaluator imports occur only during real execution.
"""

from .adapters import (
    AdapterNotImplemented,
    AdapterRegistry,
    AdapterSpec,
    AlgorithmPlugin,
    BackboneAdapter,
    Capabilities,
)
from .config import ConfigurationError
from .engine import RuntimeEngine
from .state import RuntimeContext, RunState

__all__ = [
    "AdapterNotImplemented",
    "AdapterRegistry",
    "AdapterSpec",
    "AlgorithmPlugin",
    "BackboneAdapter",
    "Capabilities",
    "ConfigurationError",
    "RuntimeContext",
    "RuntimeEngine",
    "RunState",
]
