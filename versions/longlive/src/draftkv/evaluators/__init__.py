"""Official evaluator adapters with fail-closed protocol preflight."""

from .official import (
    DependencyPin, EvaluationError, EvaluationInputs, EvaluationReport,
    EvaluatorDefinition, EvaluatorRegistry, HookSpec, OfficialEvaluator, Status,
)
from .orbia import ORBIAEvaluatorSpec, ORBIAOfficialEvaluator, ORBIAVideos, cpu_reference_preflight
from .orbia_video import build_video_manifest

__all__ = [
    "preflight", "evaluate", "audit",
    "DependencyPin", "EvaluationError", "EvaluationInputs", "EvaluationReport",
    "EvaluatorDefinition", "EvaluatorRegistry", "HookSpec", "OfficialEvaluator", "Status",
    "ORBIAEvaluatorSpec", "ORBIAOfficialEvaluator", "ORBIAVideos", "build_video_manifest",
    "cpu_reference_preflight",
]

from .runtime import preflight, evaluate, audit
