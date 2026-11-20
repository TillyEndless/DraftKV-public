"""One release binding for H3 installation, deployment and checkpoint admission."""
import json
from pathlib import Path

from .contracts import BackboneError

ROOT = Path(__file__).resolve().parents[3]


def release_binding(root=ROOT):
    config = json.loads((Path(root) / "configs/backbones/solarwm_model.json").read_text())
    return config["checkpoint_subfolder"], config["checkpoint_step"]


def download_patterns(root=ROOT):
    package, _ = release_binding(root)
    return ["SolarWM-h3-33B-base/**", package + "/**", "support/**"]


def release_tag(root=ROOT):
    _, step = release_binding(root)
    return f"solarwm-ema{step}"


def validate_checkpoint(path, expected_step):
    """Keep explicit paths supported, but never substitute another stage/step."""
    path = Path(path)
    for name in ("COMPLETE.json", "checkpoint-manifest.json", "ema.pt"):
        if not (path / name).is_file():
            raise BackboneError("BLOCKED_CHECKPOINT", "missing H3 Stage2 asset " + name)
    metadata = [json.loads((path / name).read_text())
                for name in ("COMPLETE.json", "checkpoint-manifest.json")]
    steps = [row.get("step") for row in metadata]
    if steps != [expected_step, expected_step]:
        raise BackboneError(
            "BLOCKED_CHECKPOINT",
            f"H3 release requires Stage2 EMA {expected_step}; found steps {steps} at {path}. "
            "Prepare a new release config with the matching checkpoint; retain old run evidence.",
        )
    # The official loader also verifies the complete camera/encoder/LoRA contract.
    if metadata[1].get("contract", {}).get("stage") != "stage2":
        raise BackboneError("BLOCKED_CHECKPOINT", "H3 release requires a Stage2 EMA checkpoint")
