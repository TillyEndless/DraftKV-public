"""Validate an external checkout before executing pinned official Python code."""
from __future__ import annotations

import subprocess
from pathlib import Path

from .datasets import sha256_file
from .orbia import ORBIADependencyError


def verify_source(root: str | Path, lock: dict) -> Path:
    root = Path(root).resolve()
    if not (root / "evaluate.py").is_file():
        raise ORBIADependencyError(f"Pinned ORBIA evaluator checkout missing: {root}")
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                              text=True, check=True, timeout=20).stdout.strip()
        dirty = subprocess.run(["git", "diff", "--no-ext-diff", "--exit-code", "HEAD", "--"],
                               cwd=root, capture_output=True, check=True, timeout=20)
        untracked = subprocess.run(["git", "ls-files", "--others", "--exclude-standard"],
                                   cwd=root, capture_output=True, text=True, check=True, timeout=20)
    except (OSError, subprocess.SubprocessError) as exc:
        raise ORBIADependencyError(f"ORBIA source checkout missing or modified: {exc}") from exc
    if head != lock["official_commit"] or dirty.returncode:
        raise ORBIADependencyError("Official ORBIA source commit mismatch/dirty checkout")
    if any(Path(p).suffix == ".py" for p in untracked.stdout.splitlines()):
        raise ORBIADependencyError("Untracked Python could shadow official ORBIA modules")
    for name, digest in lock["source_files"].items():
        path = root / name
        if not path.is_file() or sha256_file(path) != digest:
            raise ORBIADependencyError(f"Official source SHA256 mismatch: {name}")
    return root


# Executed in an isolated process with the official checkout as cwd. Importing
# upstream's generic `src` package in DraftKV's host process would cause collisions.
CONTROL_SCRIPT = r'''
import json, sys
if sys.version_info < (3, 10):
    print('ORBIA official code requires Python >= 3.10', file=sys.stderr)
    sys.exit(86)
from src.models import actions, camera_text
from src.models.case import load_model_case
root, cid, mode, options = sys.argv[1:]
options = json.loads(options)
case = load_model_case(root, cid)
if mode == 'text':
    result = camera_text.camera_prompt(case, native_fps=options.get('native_fps'))
elif mode == 'block':
    spec = actions.BlockActionSpec(**options)
    result = actions.block_actions(case.poses_c2w, events=case.event_frames,
                                  template=case.template, spec=spec)
elif mode == 'timed':
    spec = actions.TimedActionSpec(**options)
    result = actions.timed_actions(case.poses_c2w, events=case.event_frames,
                                  template=case.template, spec=spec)
else:
    raise ValueError('invalid control interface')
def encode(x):
    if hasattr(x, 'tolist'): return x.tolist()
    raise TypeError(type(x).__name__)
print(json.dumps(result, default=encode, allow_nan=False))
'''


REFERENCE_SCRIPT = r'''
import json, sys
if sys.version_info < (3, 10):
    print('ORBIA official code requires Python >= 3.10', file=sys.stderr)
    sys.exit(86)
from src.data.case_loader import load_case
for cid in json.loads(sys.argv[2]):
    case = load_case(sys.argv[1], cid, references=True)
    case.to_evaluation_case()
    print(cid, 'REFERENCE_VALID', flush=True)
'''
