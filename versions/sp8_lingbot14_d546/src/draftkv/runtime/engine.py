"""Fail-closed execution, case checkpoints and explicit generation/evaluation lineage."""

from dataclasses import asdict
from datetime import datetime, timezone
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import traceback

from draftkv.registry import safe_name
from .adapters import (
    AdapterRegistry,
    cpu_preflight,
    invoke_hook,
    load_callable,
    require_success,
    runtime_environment,
)
from .config import ConfigurationError, asset_sha256, json_sha256
from .devices import device_mapping
from .state import RunState, create_context, run_parent, write_json


def verify_assets(context, command):
    model_dir = context.config.get("model_dir")
    if model_dir is not None and not Path(model_dir).is_dir():
        raise FileNotFoundError(f"model source directory not found: {model_dir}")
    assets = {name: context.config.get(name) for name in ("dataset", "checkpoint")}
    assets.update(
        {f"weights.{k}": v for k, v in context.config.get("weights", {}).items()}
    )
    for name, asset in assets.items():
        if asset is None:
            if command == "evaluate" and name == "checkpoint":
                continue
            raise ConfigurationError(f"{name}.path is required for {command}")
        if (
            name == "dataset"
            and context.config.get("release_target")
            and not asset.get("sha256")
        ):
            raise ConfigurationError(
                "release dataset requires an externally frozen SHA-256 pin"
            )
        actual = asset_sha256(Path(asset["path"]))
        if name != "dataset":
            context.state.weight_hashes[name] = actual
        if name in ("dataset", "checkpoint"):
            setattr(context.state, f"{name}_sha", actual)
        context.save()
        if asset.get("sha256") and asset["sha256"] != actual:
            raise ConfigurationError(
                f"{name} SHA-256 mismatch: expected {asset['sha256']}, got {actual}"
            )


def resolve_dataset(context):
    hooks = context.config.get("dataset_hooks", {})
    if not isinstance(hooks, dict) or set(hooks) - {"setup", "resolve"}:
        raise ConfigurationError(
            "dataset_hooks must contain only setup/resolve entry points"
        )
    with cpu_preflight():
        result = invoke_hook(
            hooks.get("resolve", "draftkv.benchmarks:resolve_dataset"),
            context,
            "dataset.resolve",
        )
    if result.get("protocol") != context.config["protocol"]:
        raise ConfigurationError(
            "dataset protocol mismatch (family, modality, case count or target frames)"
        )
    ids = result.get("case_ids")
    if (
        not isinstance(ids, list)
        or len(ids) != context.config["protocol"]["case_count"]
        or any(not isinstance(v, str) or not v for v in ids)
        or len(set(ids)) != len(ids)
    ):
        raise ConfigurationError(
            "dataset must expose the complete ordered frozen case IDs"
        )
    if result.get("dataset_sha256") != context.state.dataset_sha:
        raise ConfigurationError(
            "dataset resolver must bind the hashed configured dataset asset"
        )
    selection = context.config.get("selection", {})
    requested = selection.get("case_ids")
    if requested:
        if any(v not in ids for v in requested):
            raise ConfigurationError(
                "--case-ids includes cases outside the frozen dataset"
            )
        # Preserve release order even when filters are entered in another order.
        ids = [v for v in ids if v in requested]
    limit = selection.get("limit_cases")
    if limit:
        ids = ids[:limit]
    context.dataset = result
    context.state.case_ids = ids
    context.save()


def load_run(context, group, run_id):
    safe_name(run_id, "run ID")
    directory = run_parent(context.root, group) / run_id
    if (
        not directory.resolve().is_relative_to(
            run_parent(context.root, group).resolve()
        )
        or directory.is_symlink()
    ):
        raise ConfigurationError("source run escapes group directory")
    try:
        state = RunState(**json.loads((directory / "state.json").read_text()))
    except (OSError, ValueError, TypeError) as exc:
        raise ConfigurationError(f"invalid source run {run_id}: {exc}") from exc
    if (state.run_id, state.benchmark, state.backbone, state.algorithm) != (
        run_id,
        *group.key,
    ):
        raise ConfigurationError("source run identity mismatch")
    try:
        requested = json.loads((directory / "request_config.json").read_text())
        resolved = json.loads((directory / "resolved_config.json").read_text())
    except (OSError, ValueError) as exc:
        raise ConfigurationError(
            "source configuration snapshots are missing or corrupt"
        ) from exc
    if (
        state.git_root != str(context.root)
        or json_sha256(requested) != state.request_sha
        or json_sha256(resolved) != state.config_sha
    ):
        raise ConfigurationError(
            "source configuration snapshots disagree with manifest"
        )
    if (
        state.git_sha != context.state.git_sha
        or state.code_sha != context.state.code_sha
        or state.request_sha != context.state.request_sha
    ):
        raise ConfigurationError(
            "source run code/configuration/selection differs; start a new run"
        )
    if (
        state.dataset_sha != context.state.dataset_sha
        or state.weight_hashes != context.state.weight_hashes
    ):
        raise ConfigurationError(
            "source run assets changed; old results cannot be reused"
        )
    if state.case_ids != context.state.case_ids:
        raise ConfigurationError("source run frozen selection differs")
    if (
        context.state.command != "evaluate"
        and state.integration_sha != context.state.integration_sha
    ):
        raise ConfigurationError(
            "source adapter capabilities or implementation changed"
        )
    return directory, state


def validated_artifacts(directory, state, *, complete=False):
    try:
        receipt = json.loads((directory / "case_results.json").read_text())
    except FileNotFoundError:
        if not complete and not state.completed_case_ids:
            return []
        raise ConfigurationError("source case checkpoint is missing")
    if receipt.get("run_id") != state.run_id or not isinstance(
        receipt.get("artifacts"), list
    ):
        raise ConfigurationError("source case checkpoint identity mismatch")
    rows = receipt["artifacts"]
    if any(
        not isinstance(r, dict) or not {"case_id", "path", "sha256", "run_id"} <= set(r)
        for r in rows
    ):
        raise ConfigurationError("source case checkpoint has malformed artifacts")
    ids = [r["case_id"] for r in rows]
    if (
        ids != state.completed_case_ids
        or len(set(ids)) != len(ids)
        or not set(ids) <= set(state.case_ids)
    ):
        raise ConfigurationError("source case checkpoint selection mismatch")
    if complete and set(ids) != set(state.case_ids):
        raise ConfigurationError("source generation is incomplete")
    paths = set()
    for row in rows:
        path = Path(row["path"])
        if (
            row.get("run_id") != state.run_id
            or not path.resolve().is_relative_to(directory.resolve())
            or any(
                p.is_symlink()
                for p in (path, *path.parents)
                if p.is_relative_to(directory)
            )
            or not path.is_file()
            or path.stat().st_size == 0
        ):
            raise ConfigurationError("source artifact missing or outside its run")
        if path.resolve() in paths:
            raise ConfigurationError("source cases share an artifact")
        paths.add(path.resolve())
        if asset_sha256(path) != row["sha256"]:
            raise ConfigurationError("source artifact hash mismatch")
    return rows


def resume_cases(context, group, run_id):
    directory, state = load_run(context, group, run_id)
    if state.status not in ("FAILED", "BLOCKED_INTEGRATION", "SUCCEEDED"):
        raise ConfigurationError("cannot resume an active or preflight-only run")
    context.state.parent_run_id = run_id
    for index, row in enumerate(validated_artifacts(directory, state)):
        target = context.run_dir / "resumed" / f"{index:04d}{Path(row['path']).suffix}"
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(row["path"], target)
        context.complete_case(
            row["case_id"], target, expected_sha256=row["sha256"], resumed_from=run_id
        )
    context.record(
        "resume",
        {
            "parent_run_id": run_id,
            "reused_case_ids": context.state.completed_case_ids[:],
        },
    )


def generation_receipt(context):
    if context.pending_case_ids:
        raise ConfigurationError(
            f"generation did not produce all requested cases: {context.pending_case_ids}"
        )
    # Rehash immediately: recording completion does not license later modification.
    validated_artifacts(context.run_dir, context.state, complete=True)
    receipt = {
        "status": "SUCCEEDED",
        "run_id": context.state.run_id,
        "case_ids": context.state.case_ids,
        "artifacts": list(context.artifacts.values()),
        "git_sha": context.state.git_sha,
        "request_sha": context.state.request_sha,
        "dataset_sha256": context.state.dataset_sha,
        "weight_hashes": context.state.weight_hashes,
    }
    if context.state.algorithm == 'draftkv':
        receipt.update(context.state.method_provenance.get('host_pages', {}))
    write_json(context.run_dir / "generation.json", receipt)
    context.generation = receipt
    return {"run_id": receipt["run_id"], "generation_sha256": json_sha256(receipt)}


def verify_generation_files(context):
    for row in context.generation["artifacts"]:
        path = Path(row["path"])
        if (
            not path.is_file()
            or path.stat().st_size == 0
            or asset_sha256(path) != row["sha256"]
        ):
            raise ConfigurationError(
                "generation artifacts changed during evaluation/audit"
            )


def evaluation_source(context, group, run_id=None):
    if run_id is None:
        candidates = []
        for manifest in sorted(run_parent(context.root, group).glob("*/state.json")):
            if manifest.parent == context.run_dir:
                continue
            # Corrupt and incompatible manifests are not considered successful.
            try:
                _, state = load_run(context, group, manifest.parent.name)
                if (
                    state.status in ("SUCCEEDED", "FAILED", "BLOCKED_INTEGRATION")
                    and state.stages.get("generate") == "SUCCEEDED"
                ):
                    candidates.append(state.run_id)
            except (ConfigurationError, KeyError, TypeError):
                continue
        if len(candidates) != 1:
            raise ConfigurationError(
                "evaluate needs one compatible successful generation; use --source-run RUN_ID when ambiguous"
            )
        run_id = candidates[0]
    directory, state = load_run(context, group, run_id)
    if state.status not in ("SUCCEEDED", "FAILED", "BLOCKED_INTEGRATION") or state.stages.get("generate") != "SUCCEEDED":
        raise ConfigurationError("source run has no successful generation")
    rows = validated_artifacts(directory, state, complete=True)
    receipt = json.loads((directory / "generation.json").read_text())
    if (
        receipt.get("run_id") != run_id
        or receipt.get("case_ids") != state.case_ids
        or receipt.get("artifacts") != rows
        or receipt.get("git_sha") != state.git_sha
        or receipt.get("dataset_sha256") != state.dataset_sha
        or receipt.get("weight_hashes") != state.weight_hashes
    ):
        raise ConfigurationError("source generation receipt mismatch")
    context.state.source_run_id = run_id
    context.generation = receipt
    context.record(
        "generation.source",
        {"source_run_id": run_id, "generation_sha256": json_sha256(receipt)},
    )


class RuntimeEngine:
    def __init__(self, registry=None):
        self.registry = registry or AdapterRegistry()

    def execute(
        self,
        root,
        group,
        resolved,
        command,
        argv,
        *,
        preflight=False,
        resume=None,
        source_run=None,
        report_path=None,
    ):
        if command not in ("setup", "run", "evaluate", "pipeline"):
            raise ConfigurationError(f"unsupported execution command: {command}")
        if resume and (command not in ("run", "pipeline") or preflight):
            raise ConfigurationError("resume requires run/pipeline execution")
        if source_run and command != "evaluate":
            raise ConfigurationError("source_run requires evaluate execution")
        resolved = copy.deepcopy(resolved)
        context = create_context(root, group, resolved, argv)
        context.state.command = command
        context.state.status = "RUNNING"
        context.save()
        if report_path:
            write_json(Path(report_path), asdict(context.state))

        def stage(name, action):
            context.state.phase = name
            context.state.stages[name] = "RUNNING"
            context.state.stage_exit_codes[name] = None
            context.save()
            value = action()
            context.state.stages[name] = "SUCCEEDED"
            context.state.stage_exit_codes[name] = 0
            context.save()
            return value

        try:
            with runtime_environment(context, bind_devices=False):
                # Adapter presence/capability checks precede asset reads and factories.
                builder = (
                    stage(
                        "preflight.adapters", lambda: self.registry.preflight(context)
                    )
                    if command != "evaluate"
                    else None
                )
                if command == "evaluate":

                    def map_evaluation_devices():
                        context.config["device_mapping"] = device_mapping(
                            resolved["selected_devices"]
                        )
                        context.state.device_mapping = context.config["device_mapping"]

                    stage("preflight.devices", map_evaluation_devices)
                if command in ("setup", "pipeline") and not preflight:
                    hooks = resolved.get("dataset_hooks", {})

                    def prepare_dataset():
                        with cpu_preflight():
                            invoke_hook(
                                hooks.get(
                                    "setup", "draftkv.benchmarks:prepare_dataset"
                                ),
                                context,
                                "dataset.setup",
                            )

                    stage("setup.dataset", prepare_dataset)
                stage("preflight.assets", lambda: verify_assets(context, command))
                stage("preflight.dataset", lambda: resolve_dataset(context))
                if command != "evaluate":
                    stage("preflight.native_dataset", lambda: self.registry.validate_dataset(context))
                if command == "evaluate":
                    stage(
                        "source", lambda: evaluation_source(context, group, source_run)
                    )
                if command in ("pipeline", "evaluate"):

                    def preflight_hooks():
                        with cpu_preflight():
                            load_callable(
                                resolved.get("evaluator", "draftkv.evaluators:evaluate")
                            )
                            if command == "pipeline":
                                load_callable(
                                    resolved.get("auditor", "draftkv.evaluators:audit")
                                )
                            report = load_callable(
                                resolved.get(
                                    "evaluator_preflight",
                                    "draftkv.evaluators:preflight",
                                )
                            )(context)
                            require_success(report, "evaluator preflight", ("READY",))
                            context.record("preflight.evaluator", report)

                    stage("preflight.hooks", preflight_hooks)
                # Persist the actual hardware mapping in the resolved manifest too.
                write_json(context.run_dir / "resolved_config.json", context.config)
                context.state.config_sha = json_sha256(context.config)
                if preflight:
                    context.state.status = "READY"
                else:
                    if resume:
                        stage("resume", lambda: resume_cases(context, group, resume))
                    with runtime_environment(context):
                        adapter = stage("adapter.load", builder) if builder else None
                        if command in ("setup", "pipeline"):
                            stage("setup.adapter", lambda: adapter.setup(context))
                        if command in ("run", "pipeline"):

                            def generate():
                                if context.pending_case_ids:
                                    adapter.generate(context)
                                context.record("generate", generation_receipt(context))

                            stage("generate", generate)
                        if command in ("pipeline", "evaluate"):

                            def evaluate():
                                report = load_callable(
                                    resolved.get(
                                        "evaluator", "draftkv.evaluators:evaluate"
                                    )
                                )(context)
                                require_success(
                                    report,
                                    "evaluate",
                                    ("SUCCEEDED", "COMPLETE", "PASS", "PASSED"),
                                )
                                self._check_receipt(context, report, "evaluator")
                                verify_generation_files(context)
                                context.record("evaluate", report)

                            stage("evaluate", evaluate)
                        if command == "pipeline" or (command == "evaluate" and resolved.get("benchmark") == "bench2"):

                            def audit():
                                report = load_callable(
                                    resolved.get("auditor", "draftkv.evaluators:audit")
                                )(context)
                                require_success(
                                    report,
                                    "audit",
                                    ("SUCCEEDED", "COMPLETE", "PASS", "PASSED"),
                                )
                                self._check_receipt(context, report, "audit")
                                verify_generation_files(context)
                                if report.get("evaluation_sha256") != json_sha256(
                                    context.results["evaluate"]
                                ):
                                    raise ConfigurationError(
                                        "audit receipt must bind the current evaluation"
                                    )
                                context.record("audit", report)

                            stage("audit", audit)
                    context.state.status = "SUCCEEDED"
        except (Exception, KeyboardInterrupt, SystemExit) as exc:
            context.state.status = (
                "BLOCKED_INTEGRATION"
                if isinstance(exc, (NotImplementedError, ImportError)) or (str(getattr(exc, "code", "")).startswith("BLOCKED") or getattr(exc, "code", None) == "GPU_GATE_REQUIRED")
                else "FAILED"
            )
            if context.state.phase:
                context.state.stages[context.state.phase] = context.state.status
                context.state.stage_exit_codes[context.state.phase] = 1
            context.state.error_type = type(exc).__name__
            context.state.failure_reason = str(exc)
            trace = traceback.format_exc()
            path = context.run_dir / "traceback.txt"
            path.write_text(trace, encoding="utf-8")
            context.state.trace_path = str(path)
            with context.stderr_path.open("a", encoding="utf-8") as stderr:
                stderr.write(trace)
            if isinstance(exc, KeyboardInterrupt):
                raise
        finally:
            context.state.finished_at = datetime.now(timezone.utc).isoformat()
            context.state.exit_code = (
                0 if context.state.status in ("SUCCEEDED", "READY") else 1
            )
            context.save()
        return asdict(context.state)

    @staticmethod
    def _check_receipt(context, report, label):
        if report.get("run_id") != context.state.run_id or report.get(
            "generation_sha256"
        ) != json_sha256(context.generation):
            raise ConfigurationError(
                f"{label} receipt does not bind this run and its generation"
            )


def execute(root, group, resolved, command, argv, **options):
    """Embedded entry for a dedicated interpreter; CLI uses execute_isolated."""
    return RuntimeEngine().execute(root, group, resolved, command, argv, **options)


def execute_isolated(root, group, resolved, command, argv, **options):
    """A fresh interpreter per group prevents CUDA state leaking across device sets."""
    payload = {
        "root": str(root),
        "group": group.to_dict(),
        "resolved": resolved,
        "command": command,
        "argv": argv,
        "options": options,
    }
    environment = os.environ.copy()
    environment.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1"
    )
    source = str(Path(__file__).resolve().parents[2])
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (source, environment.get("PYTHONPATH")))
    )
    with tempfile.TemporaryDirectory(prefix="draftkv-worker-") as temporary:
        transport = Path(temporary)
        payload["report_path"] = str(transport / "result.json")
        with (
            (transport / "stdout").open("w+b") as stdout,
            (transport / "stderr").open("w+b") as stderr,
        ):
            result = subprocess.run(
                [sys.executable, "-m", "draftkv.runtime.worker"],
                input=json.dumps(payload),
                stdout=stdout,
                stderr=stderr,
                text=True,
                env=environment,
                cwd=root,
            )
            try:
                row = json.loads((transport / "result.json").read_text())
                RunState(**row)
            except (OSError, ValueError, TypeError) as exc:
                stderr.seek(0)
                raise ConfigurationError(
                    f"runtime worker failed (exit {result.returncode}): {stderr.read()[-2000:].decode(errors='replace')}"
                ) from exc
            directory = run_parent(root, group) / row["run_id"]
            write_json(
                directory / "worker.json",
                {
                    "argv": [sys.executable, "-m", "draftkv.runtime.worker"],
                    "returncode": result.returncode,
                },
            )
            for source, name in ((stdout, "stdout.log"), (stderr, "stderr.log")):
                source.seek(0)
                with (directory / name).open("ab") as target:
                    shutil.copyfileobj(source, target)
            if result.returncode != 0 or row["status"] in ("CREATED", "RUNNING"):
                row.update(
                    status="FAILED",
                    error_type="WorkerProcessError",
                    exit_code=result.returncode or 1,
                    finished_at=datetime.now(timezone.utc).isoformat(),
                )
                # The latest durable state may contain partial case checkpoints.
                durable = json.loads((directory / "state.json").read_text())
                durable.update(
                    {
                        k: row[k]
                        for k in ("status", "error_type", "exit_code", "finished_at")
                    }
                )
                if durable.get("phase"):
                    durable["stages"][durable["phase"]] = "FAILED"
                path = directory / "worker_failure.txt"
                path.write_text(
                    f"isolated worker exited with code {result.returncode}\n"
                )
                with (directory / "stderr.log").open("a") as target:
                    target.write(path.read_text())
                durable["trace_path"] = str(path)
                write_json(directory / "state.json", durable)
                row = durable
    return row
