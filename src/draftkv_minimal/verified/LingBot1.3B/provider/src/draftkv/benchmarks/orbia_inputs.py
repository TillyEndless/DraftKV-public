"""Select an honest native interface; official helpers own action/text conversion."""
from __future__ import annotations

import json
import math
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping

from .datasets import DatasetError, unique_json
from .orbia import ORBIADataset, ORBIADependencyError, ORBIAInput, frozen, load_lock, DEFAULT_LOCK
from .orbia_source import CONTROL_SCRIPT, verify_source


class ORBIAInterface(str, Enum):
    CAMERA = "camera"
    ACTION = "action"
    IMAGE_TEXT = "image_text"


@dataclass(frozen=True)
class ORBIACondition:
    interface: ORBIAInterface
    group: str
    control_profile: str
    native_camera_conditioning: bool
    input: ORBIAInput
    payload: Mapping
    provenance: Mapping


def prepare_condition(dataset: ORBIADataset, case_id: str, interface: ORBIAInterface,
                      native_interfaces: tuple[ORBIAInterface, ...], *,
                      official_root: str | Path | None = None, python: str = sys.executable,
                      action_mode: str = "block", conversion_options: Mapping | None = None,
                      lock_path: str | Path = DEFAULT_LOCK) -> ORBIACondition:
    interface = ORBIAInterface(interface)
    if interface not in native_interfaces:
        raise DatasetError(f"Backbone does not declare native {interface.value} capability")
    item = dataset.load_case(case_id)
    if item.dataset_family != "orbia":
        raise DatasetError("ORBIA rejects WorldAtlas input")
    provenance = {"dataset_family": "orbia", "case_id": case_id,
                  "interface": interface.value,
                  "native_camera_conditioning": interface is ORBIAInterface.CAMERA,
                  "manifest_sha256": item.source_manifest_sha256,
                  "file_hashes": dict(item.file_hashes), "original_caption": item.caption,
                  "original_camera": str(dataset.root / "public" / case_id / "camera.npz"),
                  "camera_convention": "OpenCV C2W; X right, Y down, Z forward",
                  "trajectory_operation": "none"}
    if interface is ORBIAInterface.CAMERA:
        payload = {"image_path": item.image_path, "prompt": item.caption,
                   "poses_c2w": item.poses_c2w, "K": item.intrinsics,
                   "timestamps_s": item.timestamps_s}
        return ORBIACondition(interface, "world", "pose_and_action", True, item,
                              frozen(payload), frozen(provenance))
    if official_root is None:
        raise ORBIADependencyError("Official checkout required for action/text conversion")
    lock = load_lock(lock_path)
    source = verify_source(official_root, lock)
    options = dict(conversion_options or {})
    if interface is ORBIAInterface.IMAGE_TEXT:
        if set(options) - {"native_fps"}:
            raise DatasetError("Text conversion accepts only official native_fps")
        fps = options.get("native_fps")
        if fps is not None and (type(fps) not in (int, float) or not math.isfinite(fps) or fps <= 0):
            raise DatasetError("native_fps must be positive and finite")
        mode, group, profile = "text", "ti2v", "pose_and_action"
        helper = "src.models.camera_text.camera_prompt"
    else:
        if action_mode not in ("block", "timed"):
            raise DatasetError("Official action_mode must be block or timed")
        # Calibration belongs to the native model. Do not invent its turn speed.
        required = {"yaw_step_deg", "pitch_step_deg"} if action_mode == "block" else {
            "yaw_deg_per_s", "pitch_deg_per_s", "control_seconds"}
        if not required.issubset(options):
            raise DatasetError("Action conversion requires explicit model-calibrated turning rates")
        for key in required - {"control_seconds"}:
            value = options[key]
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise DatasetError("Action turning rates must be positive and finite")
        mode, group, profile = action_mode, "world", "action_only"
        helper = "src.models.actions." + action_mode + "_actions"
    command = [str(python), "-c", CONTROL_SCRIPT, str(dataset.root), case_id, mode,
               json.dumps(options, allow_nan=False)]
    try:
        result = subprocess.run(command, cwd=source, capture_output=True, timeout=120, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        raise ORBIADependencyError(f"Official control converter unavailable: {exc}") from exc
    if result.returncode:
        raise ORBIADependencyError(f"Official control converter failed ({result.returncode}): "
                                   + result.stderr.decode(errors="replace"))
    converted = unique_json(result.stdout)
    provenance.update(official_commit=lock["official_commit"], conversion=helper,
                      conversion_options=options, control_profile=profile,
                      upstream_caption_operation="load_model_case uses prompt.txt.strip()",
                      trajectory_operation="official load_model_case: inv(C2W[0]) @ C2W; conversion only",
                      converted_control=converted)
    payload = {"image_path": item.image_path, "prompt": converted["prompt"] if mode == "text" else item.caption,
               "control": converted}
    return ORBIACondition(interface, group, profile, False, item, frozen(payload), frozen(provenance))
