"""``draftkv`` console entry point. This module has no torch/GPU dependency.

Offline example::

    draftkv run --benchmark bench1 --devices 0,2 --dry-run

All relative paths (including --config) are relative to the Git root, never the
caller's current directory. --repo-root can select a repository from elsewhere.
Omitted backbone/algorithm filters independently expand registered groups.
SUCCEEDED means only that the requested hooks/commands returned successfully;
it is not certification of algorithm accuracy, release readiness, or E2E PASS.
"""

import argparse
import json
import sys

from .registry import RegistryError, safe_name, canonical_method
from .runtime.config import Config, ConfigurationError, parse_devices, repo_root
from .runtime.engine import execute_isolated as execute
from .runtime.state import inspect_runs


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="draftkv", description=__doc__)
    subcommands = result.add_subparsers(dest="command", required=True)
    for command in ("pipeline", "setup", "run", "evaluate", "inspect"):
        child = subcommands.add_parser(command)
        child.add_argument("--benchmark", required=True, help="bench1 or bench2")
        child.add_argument(
            "--backbone", help="optional Registry backbone filter (case insensitive)"
        )
        child.add_argument(
            "--algorithm", "--method", type=canonical_method,
            help="optional Registry method filter (case insensitive)"
        )
        child.add_argument(
            "--devices",
            required=True,
            help="physical CUDA IDs, e.g. 0,2,3,4; SP=1/2/4/8",
        )
        if command == 'run':
            child.add_argument('--ablation-config', help='explicit mechanism experiment YAML; never used by ordinary runners')
            child.add_argument('--production-plan', help='frozen worker plan for explicit ablation run')
            child.add_argument('--ablation-device', default='H200')
            child.add_argument('--ablation-output-root', default='outputs/ablations')
        child.add_argument("--worldtrace-preset", choices=("author-default", "table2-budget-matched"))
        child.add_argument("--gpu-profile", action="store_true",
                           help="opt-in synchronized GPU latency and peak allocator memory per case/rank")
        child.add_argument(
            "--config",
            help="JSON configuration path relative to repo root, or absolute (default: DRAFTKV_CONFIG)",
        )
        child.add_argument("--repo-root", help="explicit Git repository root")
        child.add_argument("--cache", help="installed asset/environment cache for the unified pipeline")
        child.add_argument("--dataset-root", "--dataset-dir", help="explicit frozen benchmark data root for the unified pipeline")
        if command == "pipeline":
            child.add_argument("--skip-install", action="store_true",
                               help="reuse installed dependencies in the unified release workflow")
        child.add_argument(
            "--dry-run",
            action="store_true",
            help="plan only; no runtime imports, asset reads, GPU probes or run files",
        )
        child.add_argument(
            "--preflight",
            action="store_true",
            help="check integrations, assets and hardware without loading a model or generating",
        )
        child.add_argument(
            "--resume",
            metavar="RUN_ID",
            help="resume verified case checkpoints into a NEW run (one group, run/pipeline)",
        )
        child.add_argument(
            "--source-run",
            metavar="RUN_ID",
            help="explicit successful generation to evaluate (one group)",
        )
        child.add_argument(
            "--case-ids",
            help="comma-separated case IDs; frozen dataset order is preserved",
        )
        child.add_argument(
            "--limit-cases",
            type=int,
            help="positive limit after case filtering; never a full benchmark claim",
        )
    return result


def main(argv: list[str] | None = None) -> int:
    cli = parser()
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = cli.parse_args(arguments)
    try:
        if getattr(args, 'ablation_config', None):
            from .ablation.config import read_config, resolve
            from pathlib import Path
            base = repo_root(args.repo_root)
            path = Path(args.ablation_config)
            if not path.is_absolute():
                path = base/path
            try:
                explicit = resolve(read_config(path))
            except ValueError as exc:
                cli.error(str(exc))
            if explicit is not None:
                from .run import main as ablation_main
                forwarded = ['--ablation-config', str(path), '--backbone', args.backbone or '',
                             '--benchmark', args.benchmark, '--method', args.algorithm or 'draftkv',
                             '--devices', args.devices, '--device', args.ablation_device,
                             '--output-root', args.ablation_output_root, '--repo-root', str(base)]
                if args.production_plan:
                    forwarded.extend(['--production-plan', args.production_plan])
                if args.dry_run:
                    forwarded.append('--dry-run')
                return ablation_main(forwarded)
        devices = parse_devices(args.devices)
        root = repo_root(args.repo_root)
        # The unified pipeline uses the same installation, deployment and
        # persistent queue as the release wrapper. Dry runs stay
        # side-effect free and explicit configs retain the normal runtime path.
        if (args.benchmark.lower() in ("bench1", "bench2") and args.command == "pipeline"
                and not args.config and not __import__("os").environ.get("DRAFTKV_CONFIG")
                and not args.dry_run and not args.preflight and not args.resume):
            import subprocess
            workflow = root / ("scripts/" + args.benchmark.lower() + "_release.py")
            forwarded = ["all", "--benchmark", args.benchmark.lower(), "--devices", args.devices]
            if args.skip_install:
                forwarded.append("--skip-install")
            if args.gpu_profile:
                forwarded.append("--gpu-profile")
            for key in ("backbone", "algorithm", "case_ids", "limit_cases", "worldtrace_preset"):
                value = getattr(args, key)
                if value is not None:
                    forwarded += ["--" + key.replace("_", "-"), str(value)]
            if args.cache:
                forwarded += ["--cache", args.cache]
            if args.dataset_root:
                forwarded += ["--dataset-root", args.dataset_root]
            return subprocess.call([sys.executable, str(workflow), *forwarded])
        if getattr(args, "skip_install", False):
            raise ConfigurationError("--skip-install requires pipeline without an explicit runtime config, preflight, dry-run or resume")
        if (args.cache or args.dataset_root) and not args.dry_run:
            raise ConfigurationError("--cache/--dataset-root require pipeline without an explicit runtime config")
        if args.gpu_profile and args.command not in ("run", "pipeline"):
            raise ConfigurationError("--gpu-profile is only valid with run/pipeline")
        config = Config(root, args.config)
        groups = config.registry.select(args.benchmark, args.backbone, args.algorithm)
        for run_id in (args.resume, args.source_run):
            if run_id is not None:
                safe_name(run_id, "run ID")
        if args.limit_cases is not None and args.limit_cases < 1:
            raise ConfigurationError("--limit-cases must be positive")
        case_ids = args.case_ids.split(",") if args.case_ids is not None else None
        if case_ids is not None and (
            any(not v.strip() or v != v.strip() for v in case_ids)
            or len(set(case_ids)) != len(case_ids)
        ):
            raise ConfigurationError(
                "--case-ids must contain unique nonempty IDs without surrounding whitespace"
            )
        if args.dry_run and args.preflight:
            raise ConfigurationError("--dry-run and --preflight are mutually exclusive")
        if args.resume and (
            len(groups) != 1
            or args.command not in ("run", "pipeline")
            or args.preflight
            or args.dry_run
        ):
            raise ConfigurationError(
                "--resume requires one selected group and a real run/pipeline"
            )
        if args.source_run and (
            len(groups) != 1 or args.command != "evaluate" or args.dry_run
        ):
            raise ConfigurationError(
                "--source-run requires one selected group and evaluate without --dry-run"
            )
        if args.command == "inspect" and any(
            (args.preflight, args.resume, args.source_run, case_ids, args.limit_cases)
        ):
            raise ConfigurationError(
                "inspect does not accept execution/selection options"
            )
        # Resolve every group before creating any run, so invalid input is atomic.
        resolved = [config.resolve(group, devices) for group in groups]
        for item in resolved:
            item["selection"] = {"case_ids": case_ids, "limit_cases": args.limit_cases}
            if args.gpu_profile:
                item["gpu_profile"] = True
            if args.worldtrace_preset and item["algorithm"] == "worldtrace-landmark":
                from .backbones.budgets import effective_budget, author_budget_policy
                from .backbones.catalog import model_spec
                cfg = item.setdefault("backbone_config", {})
                cfg.setdefault("algorithm_plugin", {}).setdefault("config", {})["preset"] = args.worldtrace_preset
                cfg["effective_budget"] = effective_budget(model_spec(item["backbone"]), item["algorithm"], preset=args.worldtrace_preset).to_dict()
                cfg["author_budget_policy"] = author_budget_policy(model_spec(item["backbone"]), item["algorithm"], preset=args.worldtrace_preset)
        if args.dry_run:
            payload = {
                "command": args.command,
                "dry_run": True,
                "groups": resolved,
                "group_count": len(groups),
                "status": "PLANNED",
            }
        elif args.command == "inspect":
            payload = {
                "command": "inspect",
                "runs": inspect_runs(root, groups, devices),
            }
        else:
            rows = [
                execute(
                    root,
                    group,
                    item,
                    args.command,
                    ["draftkv", *arguments],
                    preflight=args.preflight,
                    resume=args.resume,
                    source_run=args.source_run,
                )
                for group, item in zip(groups, resolved)
            ]
            payload = {"command": args.command, "runs": rows}
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
        return (
            1
            if any(
                row["status"] not in ("SUCCEEDED", "READY")
                for row in payload.get("runs", [])
            )
            and args.command != "inspect"
            else 0
        )
    except (ConfigurationError, RegistryError, OSError) as exc:
        cli.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
