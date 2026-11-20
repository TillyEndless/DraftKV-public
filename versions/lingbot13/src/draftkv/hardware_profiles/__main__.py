"""python -m draftkv.hardware_profiles: offline by default, target execution explicit."""

import argparse
import importlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

from .profiles import (ProfileError, load_catalog, load_profile, read_json, recipes_for,
                       require, request_family, select_profile, verify_binding, verify_provenance)
from .runner import TunePolicy, candidate_configs, run_tuner


def task_spec(argv, request, baseline, policy):
    argv = list(argv)
    if "--output" in argv:
        index = argv.index("--output") + 1
        argv[index] += ".target.json"
    return {"schema_version": 1, "kind": "draftkv_b01_light_autotune", "status": "GPU_GATE_REQUIRED",
            "target_hardware": request_family(request), "request": request,
            "baseline_profile_sha256": baseline["sha256"],
            "resources": {"gpu_count": request["parallel"]["sp"], "exclusive": True,
                          "max_runtime_s": policy.total_timeout_s},
            "command": [sys.executable, "-m", "draftkv.hardware_profiles", *argv, "--execute"],
            "required_gate": "CUDA_GOLDEN", "automatic_enqueue": False,
            "output_contract": {"real_checkpoint_forward": "NOT_RUN", "video_e2e": "NOT_RUN",
                                "official_eval": "NOT_RUN", "publish": False}}


def execute_worker(argv, timeout_s):
    """Outer process-group watchdog, including a hung native Frontier/kernel call."""
    command = [sys.executable, "-m", "draftkv.hardware_profiles", *argv, "--worker"]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, start_new_session=True,
                               env={**os.environ, "DRAFTKV_GPU_PROFILE_WORKER_PARENT": str(os.getpid())})
    try:
        stdout, stderr = process.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.communicate(timeout=3)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
        raise ProfileError("GPU_GATE_REQUIRED", "Frontier/worker timeout; worker group discarded; no winner") from exc
    require(process.returncode == 0, "GPU_GATE_REQUIRED", "target adapter worker failed: " + stderr[-1000:] + stdout[-1000:])
    try:
        return json.loads(stdout)
    except ValueError as exc:
        raise ProfileError("GPU_GATE_REQUIRED", "adapter polluted JSON worker output") from exc


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("query", "tune"))
    parser.add_argument("--catalog", default="configs/gpu_profiles/catalog.json")
    parser.add_argument("--request", required=True, help="externally captured physical/ABI/QKV JSON")
    parser.add_argument("--artifact-root", required=True, help="external root for source/kernel/raw evidence")
    parser.add_argument("--asset-roots", help="external JSON mapping archive asset IDs to local roots")
    parser.add_argument("--baseline", help="explicit old original-B01 bound profile")
    parser.add_argument("--target-hardware", choices=("H100", "B200"), help="explicit scheduler target; checked against actual probe")
    parser.add_argument("--adapter", help="trusted target module:factory (loaded only with --execute)")
    parser.add_argument("--scheduler-lease", help="unique Scheduler allocation with profile_autotune purpose; required for --execute")
    parser.add_argument("--output", help="new output file; existing evidence is never overwritten")
    parser.add_argument("--gpu-gate", action="store_true", help="verify saved Golden evidence; does not run GPU")
    parser.add_argument("--execute", action="store_true", help="run actual target hardware tuner")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if args.worker:
            require(args.execute and os.environ.get("DRAFTKV_GPU_PROFILE_WORKER_PARENT") == str(os.getppid()),
                    "GPU_GATE_REQUIRED", "native worker requires parent process-group watchdog")
        catalog, request = load_catalog(args.catalog), read_json(args.request)
        assets = read_json(args.asset_roots) if args.asset_roots else None
        request_family(request)
        if args.output and not args.worker:
            require(not Path(args.output).exists(), "BLOCKED_PROFILE", "refusing to overwrite existing evidence output")
        if args.action == "query":
            require(not args.execute and not args.worker, "BLOCKED_PROFILE", "query never executes GPU")
            recipes = recipes_for(catalog, request)
            for recipe in recipes:
                verify_provenance(recipe["provenance"], args.artifact_root, assets)
            try:
                result = select_profile(catalog, request, args.artifact_root, gpu=args.gpu_gate, assets=assets)
            except ProfileError as exc:
                if recipes and exc.code == "BLOCKED_PROFILE" and not catalog["profiles"] and not args.gpu_gate:
                    result = {"status": "GPU_GATE_REQUIRED", "validation_level": "STATIC",
                              "recipes": recipes, "reason": "frozen historical values only; exact physical binding and Golden required"}
                else:
                    raise
        else:
            require(args.target_hardware == request_family(request), "UNSUPPORTED_GPU", "explicit target-hardware differs from request")
            if args.execute:
                from .deployment import verify_scheduler_lease
                verify_scheduler_lease(args.scheduler_lease, request)
            require(not args.gpu_gate, "BLOCKED_PROFILE", "tune runs fresh validations; gpu-gate is query-only")
            require(bool(args.baseline), "BLOCKED_PROFILE", "explicit old profile required for A/B")
            baseline = load_profile(args.baseline)
            verify_binding(baseline, request, args.artifact_root, catalog, assets=assets)
            candidates = candidate_configs(request)
            policy = TunePolicy()
            if not args.execute:
                require(not args.worker, "BLOCKED_PROFILE", "worker requires explicit execution")
                spec_argv = [item for item in argv if item != "--execute"]
                result = {"status": "GPU_GATE_REQUIRED", "validation_level": "STATIC", "winner": None,
                          "candidates": candidates, "task_spec": task_spec(spec_argv, request, baseline, policy)}
            elif not args.worker:
                require(bool(args.adapter), "GPU_GATE_REQUIRED", "original B01 target adapter required")
                result = execute_worker(argv, policy.total_timeout_s)
            else:
                require(args.adapter and ":" in args.adapter, "GPU_GATE_REQUIRED", "adapter must be module:factory")
                module, factory = args.adapter.split(":", 1)
                adapter = getattr(importlib.import_module(module), factory)(request=request, artifact_root=args.artifact_root)
                result = run_tuner(catalog, request, baseline, args.artifact_root, adapter, assets=assets)
        if args.output and not args.worker:
            # Exclusive creation also protects a baseline/output collision.
            with Path(args.output).open("x", encoding="utf-8") as stream:
                json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n")
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 0
    except (ProfileError, OSError, ImportError, AttributeError, TypeError, ValueError) as exc:
        print(json.dumps({"status": getattr(exc, "code", "GPU_GATE_REQUIRED"), "validation_level": "STATIC",
                          "winner": None, "reason": str(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
