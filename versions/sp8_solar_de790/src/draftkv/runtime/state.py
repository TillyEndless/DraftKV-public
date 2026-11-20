"""One manifest schema for every execution stage and isolated per-run evidence."""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import uuid

from .config import ConfigurationError, asset_sha256, json_sha256


def write_json(path: Path, value) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def git_sha(root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ConfigurationError(f"cannot resolve Git HEAD at {root}") from exc


def git_dirty(root: Path) -> bool | None:
    """Unknown is explicit for imported/test manifests without a Git checkout."""
    result = subprocess.run(["git", "-C", str(root), "status", "--porcelain"],
                            capture_output=True, text=True, check=False)
    return bool(result.stdout.strip()) if result.returncode == 0 else None


def method_provenance(root, group, resolved):
    if group.algorithm not in ("deep-forcing", "future-forcing"):
        return {}
    from draftkv.algorithms.forcing.budget import policy_budget
    from draftkv.backbones.catalog import MODELS
    spec = MODELS[group.backbone]
    cfg = resolved.get("backbone_config", {})
    settings = cfg.get("algorithm_plugin", {}).get("config", {})
    policy = policy_budget(spec.window, spec.chunk, group.algorithm,
                           preset=settings.get("preset", "author_primary"))
    source_file = root / "configs/assets/bench1.sources.json"
    sources = json.loads(source_file.read_text())["assets"] if source_file.is_file() else {}
    names = ("longlive-1.3b", "wan21-t2v-1.3b") if group.backbone == "LongLive1.3B" else ("lingbot-v2-14b",)
    data_root = resolved.get("dataset_config", {}).get("root")
    if data_root is not None:
        path = Path(data_root).expanduser()
        data_root = str((path if path.is_absolute() else root / path).resolve())
    hyperparameters = (dict(fusion="sum", max_topc_reuse=7, statistics_cadence="every_attention_call",
                            selection_cadence="first_call_on_native_window_overflow")
                       if group.algorithm == "deep-forcing" else
                       dict(lambda_history_future=.5, future_horizon=6, merge_threshold=.95,
                            proxy_history_frames=policy["proxy_history_frames"],
                            statistics_cadence="once_per_clean_frame",
                            selection_cadence="first_noisy_call_per_chunk"))
    hyperparameters.update(preset=policy["preset"], attention_consumer="native_fa2",
        score_backend=("torch_signed_dot_sum" if group.algorithm == "deep-forcing"
                       else settings.get("score_backend", "fa2")))
    return dict(budget=policy, method_hyperparameters=hyperparameters,
                backbone_revisions={name: sources[name] for name in names if name in sources},
                evaluator_revision=sources.get("vbench-long"),
                evaluator_binding=resolved.get("evaluation_config", {}),
                dataset_id=group.dataset_id, dataset_binding=resolved.get("dataset", {}),
                dataset_resolved_root=data_root,
                frame_tokens=1560 if group.backbone == "LongLive1.3B" else None,
                frame_geometry="native per-case geometry; VAE8 / patch2x2",
                case_geometries={})


@dataclass
class RunState:
    run_id: str
    benchmark: str
    backbone: str
    algorithm: str
    selected_devices: list[int]
    sp: int
    git_sha: str
    config_sha: str
    dataset_sha: str | None
    checkpoint_sha: str | None
    consumer_config: dict
    created_at: str
    exec_argv: list[str]
    git_root: str
    config_path: str | None
    status: str = "CREATED"
    error_type: str | None = None
    trace_path: str | None = None
    command: str = "run"
    phase: str | None = None
    request_sha: str | None = None
    parent_run_id: str | None = None
    source_run_id: str | None = None
    model_config: dict = field(default_factory=dict)
    algorithm_config: dict = field(default_factory=dict)
    weight_hashes: dict = field(default_factory=dict)
    device_mapping: dict = field(default_factory=dict)
    case_ids: list[str] = field(default_factory=list)
    completed_case_ids: list[str] = field(default_factory=list)
    stages: dict = field(default_factory=dict)
    finished_at: str | None = None
    code_sha: str | None = None
    integration_sha: str | None = None
    exit_code: int | None = None
    git_dirty: bool | None = None
    method_provenance: dict = field(default_factory=dict)
    stage_exit_codes: dict = field(default_factory=dict)
    failure_reason: str | None = None


@dataclass
class RuntimeContext:
    """Contract shared by adapters, dataset hooks and evaluator hooks.

    CUDA_VISIBLE_DEVICES uses selected GPU UUIDs, so a child process sees local
    ordinals 0..sp-1. Hooks must preserve the original algorithm's semantics.
    Resolved config and consumer_config are passed through without translating
    algorithm knobs. Return values are saved as evidence, never release approval.
    """

    root: Path
    run_dir: Path
    config: dict
    state: RunState
    results: dict = field(default_factory=dict)
    artifacts: dict = field(default_factory=dict)
    dataset: dict = field(default_factory=dict)
    generation: dict = field(default_factory=dict)

    @property
    def stdout_path(self) -> Path:
        return self.run_dir / "stdout.log"

    @property
    def stderr_path(self) -> Path:
        return self.run_dir / "stderr.log"

    def save(self) -> None:
        write_json(self.run_dir / "state.json", asdict(self.state))

    def record(self, stage: str, result) -> None:
        self.results[stage] = result
        write_json(self.run_dir / "stage_results.json", self.results)

    @property
    def pending_case_ids(self) -> tuple[str, ...]:
        return tuple(case for case in self.state.case_ids if case not in self.artifacts)

    def complete_case(
        self, case_id: str, path, *, expected_sha256=None, resumed_from=None
    ):
        """Commit one nonempty current-run file after generation; safe retry boundary."""
        if case_id not in self.state.case_ids or case_id in self.artifacts:
            raise ConfigurationError(f"unknown or duplicate completed case: {case_id}")
        path = Path(path)
        candidate = path if path.is_absolute() else self.run_dir / path
        if any(p.is_symlink() for p in (candidate, *candidate.parents)):
            # macOS temp roots can themselves be symlinks; only test below run_dir.
            relative = candidate.absolute().relative_to(self.run_dir)
            if any(
                (self.run_dir / Path(*relative.parts[:i])).is_symlink()
                for i in range(1, len(relative.parts) + 1)
            ):
                raise ConfigurationError("case artifact cannot be a symlink")
        path = candidate.resolve()
        if (
            not path.is_relative_to(self.run_dir.resolve())
            or not path.is_file()
            or path.stat().st_size == 0
        ):
            raise ConfigurationError(
                "case artifact must be a nonempty file inside the current run"
            )
        if path in [Path(r["path"]) for r in self.artifacts.values()]:
            raise ConfigurationError("different cases cannot share the same artifact")
        digest = asset_sha256(path)
        if expected_sha256 is not None and expected_sha256 != digest:
            raise ConfigurationError(f"case artifact hash mismatch: {case_id}")
        self.artifacts[case_id] = {
            "case_id": case_id,
            "path": str(path),
            "sha256": digest,
            "run_id": self.state.run_id,
            "resumed_from": resumed_from,
        }
        self.state.completed_case_ids = list(self.artifacts)
        write_json(
            self.run_dir / "case_results.json",
            {"run_id": self.state.run_id, "artifacts": list(self.artifacts.values())},
        )
        self.save()


def run_parent(root: Path, group) -> Path:
    parent = root / "runs" / group.benchmark / group.backbone / group.algorithm
    if not parent.resolve().is_relative_to(root.resolve()):
        raise ConfigurationError(f"runs directory escapes repository root: {parent}")
    return parent


def create_context(
    root: Path, group, resolved: dict, argv: list[str]
) -> RuntimeContext:
    revision = git_sha(root)
    now = datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%dT%H%M%S.%fZ") + "-" + uuid.uuid4().hex
    parent = run_parent(root, group)
    parent.mkdir(parents=True, exist_ok=True)
    run_dir = parent / run_id
    run_dir.mkdir(exist_ok=False)
    # Persist scheduler lineage before any optional import, asset read or model
    # call. CLI stdout is written only at exit and cannot be a recovery index.
    attempt = os.environ.get("DRAFTKV_ATTEMPT_TOKEN")
    if attempt and os.environ.get("DRAFTKV_ATTEMPT_PHASE") is not None:
        phase = os.environ.get("DRAFTKV_ATTEMPT_PHASE")
        if (len(attempt) != 32 or any(c not in "0123456789abcdef" for c in attempt)
                or phase not in ("run", "evaluate")):
            raise ConfigurationError("invalid scheduler attempt identity")
        write_json(run_dir / "scheduler_attempt.json", {
            "schema_version": 1, "attempt_token": attempt, "phase": phase,
            "run_id": run_id, "benchmark": group.benchmark,
            "backbone": group.backbone, "algorithm": group.algorithm,
            "git_sha": revision, "request_sha": json_sha256(resolved),
            "config_path": resolved["config_path"],
            "config_file_sha256": asset_sha256(Path(resolved["config_path"])),
        })
    # Create the audit files before loading any optional runtime modules.
    (run_dir / "stdout.log").touch()
    (run_dir / "stderr.log").touch()
    write_json(run_dir / "resolved_config.json", resolved)
    write_json(run_dir / "request_config.json", resolved)
    state = RunState(
        run_id=run_id,
        benchmark=group.benchmark,
        backbone=group.backbone,
        algorithm=group.algorithm,
        selected_devices=resolved["selected_devices"],
        sp=resolved["sp"],
        git_sha=revision,
        git_dirty=git_dirty(root),
        method_provenance=method_provenance(root, group, resolved),
        config_sha=json_sha256(resolved),
        dataset_sha=None,
        checkpoint_sha=None,
        consumer_config=resolved["consumer_config"],
        created_at=now.isoformat(),
        exec_argv=list(argv),
        git_root=str(root),
        config_path=resolved["config_path"],
        request_sha=json_sha256(resolved),
        model_config=resolved.get("model_config", {}),
        algorithm_config=resolved.get("algorithm_config", {}),
        code_sha=json_sha256(
            {
                str(p.relative_to(root)): asset_sha256(p)
                for p in sorted((root / "src").rglob("*.py"))
            }
        ),
    )
    context = RuntimeContext(root, run_dir, resolved, state)
    context.save()
    return context


def inspect_runs(root: Path, groups, devices: tuple[int, ...]) -> list[dict]:
    rows = []
    for group in groups:
        for path in sorted(run_parent(root, group).glob("*/state.json")):
            if not path.resolve().is_relative_to(root.resolve()):
                raise ConfigurationError(
                    f"run manifest escapes repository root: {path}"
                )
            try:
                row = json.loads(path.read_text(encoding="utf-8"))
                RunState(**row)  # A corrupt/incompatible manifest is never hidden.
            except (ValueError, TypeError, OSError) as exc:
                raise ConfigurationError(f"invalid run manifest {path}: {exc}") from exc
            if (row["benchmark"], row["backbone"], row["algorithm"]) != group.key:
                raise ConfigurationError(
                    f"run manifest group disagrees with its directory: {path}"
                )
            if row["selected_devices"] == list(devices):
                rows.append(row)
    return rows
