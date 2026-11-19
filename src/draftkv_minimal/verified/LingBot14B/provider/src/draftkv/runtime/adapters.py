"""Lazy adapter registry and composition. Public ABI is documented in CONTRACT.md."""

from contextlib import contextmanager, redirect_stderr, redirect_stdout
import builtins
from dataclasses import asdict, dataclass
import importlib
import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import shutil
import string
import subprocess
import sys
from typing import Protocol

from .config import ConfigurationError, asset_sha256, json_sha256, resolve_path
from .devices import device_mapping, validate_hardware


class AdapterNotImplemented(NotImplementedError):
    """Integration is absent; never means that Native should be selected."""


class MissingDependency(ImportError):
    pass


class BackboneAdapter(Protocol):
    def setup(self, context): ...
    def generate(self, context): ...


class AlgorithmPlugin(Protocol):
    def attach(self, backbone: BackboneAdapter, context): ...


@dataclass(frozen=True)
class Capabilities:
    identity: str
    input_modes: tuple[str, ...]
    protocols: tuple[str, ...]
    supported_sp: tuple[int, ...]
    min_free_memory_mib: dict[str, int]
    streaming_kv: bool
    required_weights: tuple[str, ...] = ("checkpoint",)
    kernels: tuple[dict, ...] = ()
    gpu_names: tuple[str, ...] = ()
    requires_streaming_kv: bool = False

    @classmethod
    def parse(cls, value):
        try:
            caps = value if isinstance(value, cls) else cls(**value)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(f"invalid adapter capabilities: {exc}") from exc
        for name in ("input_modes", "protocols", "required_weights", "gpu_names"):
            entries = getattr(caps, name)
            if not isinstance(entries, (tuple, list)) or any(
                not isinstance(v, str) or not v for v in entries
            ):
                raise ConfigurationError(f"capabilities.{name} must contain strings")
        if (
            not caps.input_modes
            or not caps.protocols
            or not isinstance(caps.identity, str)
            or not caps.identity
        ):
            raise ConfigurationError(
                "capability identity, input modes and protocols are required"
            )
        if (
            not isinstance(caps.supported_sp, (tuple, list))
            or not caps.supported_sp
            or any(
                type(v) is not int or v not in (1, 2, 4, 8) for v in caps.supported_sp
            )
            or len(set(caps.supported_sp)) != len(caps.supported_sp)
        ):
            raise ConfigurationError(
                "capabilities.supported_sp must enumerate SP1/2/4/8"
            )
        if (
            not isinstance(caps.min_free_memory_mib, dict)
            or any(
                type(v) is not int
                or v < 0
                or k not in {str(s) for s in caps.supported_sp}
                for k, v in caps.min_free_memory_mib.items()
            )
            or set(caps.min_free_memory_mib) != {str(s) for s in caps.supported_sp}
        ):
            raise ConfigurationError(
                "capabilities must declare free memory MiB for every supported SP"
            )
        if any(
            type(getattr(caps, k)) is not bool
            for k in ("streaming_kv", "requires_streaming_kv")
        ):
            raise ConfigurationError("streaming KV capability flags must be booleans")
        if not isinstance(caps.kernels, (tuple, list)) or any(
            not isinstance(k, dict) for k in caps.kernels
        ):
            raise ConfigurationError("capabilities.kernels must be asset objects")
        return caps


class _NoAcceleratorImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {
            "torch",
            "transformers",
            "triton",
            "cupy",
            "flash_attn",
            "xformers",
        }:
            raise AdapterNotImplemented(
                f"preflight entry point must be CPU import-safe: {fullname}"
            )
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if (
            spec
            and isinstance(spec.loader, importlib.machinery.ExtensionFileLoader)
            and fullname.split(".")[0] not in {"PIL", "numpy"}
        ):
            # Stdlib and known CPU image/array decoders are safe. Unknown binary
            # extensions are deferred to factory/attach/generate in the worker.
            origin = Path(spec.origin).resolve()
            if (
                "site-packages" in origin.parts
                or "dist-packages" in origin.parts
                or not origin.is_relative_to(Path(sys.base_prefix).resolve())
            ):
                raise AdapterNotImplemented(
                    f"preflight cannot load binary extension: {fullname}"
                )
        return None


@contextmanager
def cpu_preflight():
    # MetaPathFinder alone does not see modules already loaded by another
    # CPU test or caller. Guard both import APIs before sys.modules lookup.
    guard = _NoAcceleratorImports()
    original_import, original_import_module = builtins.__import__, importlib.import_module
    forbidden = {"torch", "transformers", "triton", "cupy", "flash_attn", "xformers"}

    def check(name):
        if isinstance(name, str) and name.split(".", 1)[0] in forbidden:
            raise AdapterNotImplemented(f"preflight entry point must be CPU import-safe: {name}")

    def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 0:
            check(name)
        return original_import(name, globals, locals, fromlist, level)

    def safe_import_module(name, package=None):
        absolute = importlib.util.resolve_name(name, package) if name.startswith(".") else name
        check(absolute)
        return original_import_module(name, package)

    sys.meta_path.insert(0, guard)
    builtins.__import__, importlib.import_module = safe_import, safe_import_module
    try:
        yield
    finally:
        builtins.__import__, importlib.import_module = original_import, original_import_module
        sys.meta_path.remove(guard)


def load_callable(reference: str, *, adapter: bool = False):
    error = AdapterNotImplemented if adapter else MissingDependency
    if not isinstance(reference, str) or reference.count(":") != 1:
        raise ConfigurationError("entry point must be 'module:callable'")
    module_name, attribute = reference.split(":")
    if not module_name or not attribute or not attribute.isidentifier():
        raise ConfigurationError(f"invalid entry point: {reference!r}")
    try:
        module = importlib.import_module(module_name)
    except (ImportError, OSError) as exc:
        raise error(
            f"cannot load {reference}: {exc}; install the real implementation"
        ) from exc
    target = getattr(module, attribute, None)
    if not callable(target):
        raise error(f"{reference} is not installed as a callable")
    return target


@contextmanager
def runtime_environment(context, *, bind_devices=True):
    """Worker-local environment; UUIDs avoid CUDA/nvidia-smi ordinal ambiguity."""
    mapping = context.config.get("device_mapping", {})
    changes = {
        "CUDA_VISIBLE_DEVICES": mapping.get(
            "cuda_visible_devices", ",".join(map(str, context.state.selected_devices))
        ),
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "WORLD_SIZE": str(context.state.sp),
    }
    changes.update(
        {
            key: None
            for key in {
                "RANK",
                "LOCAL_RANK",
                "LOCAL_WORLD_SIZE",
                "GROUP_RANK",
                "ROLE_RANK",
                "MASTER_ADDR",
                "MASTER_PORT",
            }
            | {key for key in os.environ if key.startswith("TORCHELASTIC_")}
        }
    )
    if not bind_devices:
        changes = {}
    old = {k: os.environ.get(k) for k in changes}
    old_cwd = Path.cwd()
    for key, value in changes.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    try:
        os.chdir(context.root)
        with (
            context.stdout_path.open("a", encoding="utf-8") as stdout,
            context.stderr_path.open("a", encoding="utf-8") as stderr,
        ):
            with redirect_stdout(stdout), redirect_stderr(stderr):
                yield
    finally:
        os.chdir(old_cwd)
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def require_success(
    result, stage, allowed=("READY", "SUCCEEDED", "COMPLETE", "PASS", "PASSED")
):
    if isinstance(result, dict) and str(result.get("status", "")).startswith("BLOCKED"):
        raise AdapterNotImplemented(f"{stage} blocked: {result}")
    if not isinstance(result, dict) or result.get("status") not in allowed:
        raise RuntimeError(
            f"{stage} must return an explicit successful status; got {result!r}"
        )
    return result


def adapter_result(result, stage):
    # None means the adapter completed normally; any structured return must
    # explicitly succeed. In particular BLOCKED and False cannot be swallowed.
    if result is not None:
        require_success(result, stage)


def invoke_hook(reference: str, context, stage: str):
    result = load_callable(reference)(context)
    require_success(result, stage)
    context.record(stage, result)
    return result


def template_values(context) -> dict:
    return {
        "repo_root": str(context.root),
        "run_dir": str(context.run_dir),
        "config_path": context.config["config_path"] or "",
        "resolved_config_path": str(context.run_dir / "resolved_config.json"),
        "benchmark": context.state.benchmark,
        "backbone": context.state.backbone,
        "algorithm": context.state.algorithm,
        "devices": ",".join(map(str, context.state.selected_devices)),
        "sp": str(context.state.sp),
        "checkpoint": context.config.get("checkpoint", {}).get("path", ""),
        "dataset": context.config.get("dataset", {}).get("path", ""),
    }


def render_command(template, values: dict) -> list[str]:
    if (
        not isinstance(template, list)
        or not template
        or any(not isinstance(arg, str) or not arg for arg in template)
    ):
        raise ConfigurationError(
            "adapter command must be a nonempty array of nonempty strings (no shell)"
        )
    formatter = string.Formatter()
    result = []
    try:
        for arg in template:
            for _, name, spec, conversion in formatter.parse(arg):
                if name is not None and (name not in values or spec or conversion):
                    raise ConfigurationError(
                        f"unsupported command placeholder: {name!r}"
                    )
            result.append(arg.format_map(values))
    except ValueError as exc:
        raise ConfigurationError(f"invalid command template: {exc}") from exc
    if not result[0]:
        raise ConfigurationError("adapter executable must not be empty")
    return result


class CommandAdapter:
    """Compatibility route: explicit capabilities and generation receipts required."""

    def __init__(self, spec, context):
        self.commands = spec.get("commands", {})
        self.context = context
        if not isinstance(self.commands, dict):
            raise ConfigurationError("adapter.commands must be an object")
        for template in self.commands.values():
            argv = render_command(template, template_values(context))
            executable = argv[0]
            path = resolve_path(executable, context.root) if "/" in executable else None
            if not (
                (path and path.is_file() and os.access(path, os.X_OK))
                or (path is None and shutil.which(executable))
            ):
                raise AdapterNotImplemented(
                    f"adapter executable is not installed/executable: {executable}"
                )

    def execute(self, stage, context):
        if stage not in self.commands:
            raise AdapterNotImplemented(f"adapter has no {stage} command")
        argv = render_command(self.commands[stage], template_values(context))
        with (context.run_dir / "commands.jsonl").open("a", encoding="utf-8") as out:
            out.write(
                json.dumps(
                    {
                        "stage": stage,
                        "argv": argv,
                        "cwd": str(context.root),
                        "CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"],
                        "WORLD_SIZE": str(context.state.sp),
                    }
                )
                + "\n"
            )
        with (
            context.stdout_path.open("a", encoding="utf-8") as stdout,
            context.stderr_path.open("a", encoding="utf-8") as stderr,
        ):
            subprocess.run(
                argv,
                cwd=context.root,
                env=os.environ.copy(),
                stdout=stdout,
                stderr=stderr,
                check=True,
            )

    def setup(self, context):
        self.execute("setup", context)

    def generate(self, context):
        self.execute("run", context)
        receipt = json.loads((context.run_dir / "generation.json").read_text())
        if receipt.get("run_id") != context.state.run_id:
            raise ConfigurationError("generation receipt belongs to another run")
        for row in receipt.get("artifacts", []):
            context.complete_case(
                row["case_id"], row["path"], expected_sha256=row["sha256"]
            )


class ComposedAdapter:
    def __init__(self, backbone_factory, algorithm_factory, context):
        self.backbone = backbone_factory(context)
        self.algorithm = algorithm_factory(context)
        for obj, method in (
            (self.backbone, "setup"),
            (self.backbone, "generate"),
            (self.algorithm, "attach"),
        ):
            if not callable(getattr(obj, method, None)):
                raise AdapterNotImplemented(
                    f"adapter does not implement {method}(context)"
                )

    def setup(self, context):
        adapter_result(self.backbone.setup(context), "backbone.setup")

    def generate(self, context):
        adapter_result(
            self.algorithm.attach(self.backbone, context), "algorithm.attach"
        )
        adapter_result(self.backbone.generate(context), "backbone.generate")


@dataclass(frozen=True)
class AdapterSpec:
    descriptor: str
    factory: str
    preflight: str

    def __post_init__(self):
        for reference in (self.descriptor, self.factory, self.preflight):
            if not isinstance(reference, str) or reference.count(":") != 1:
                raise ConfigurationError("adapter references must be module:callable")
            module, attribute = reference.split(":")
            if (
                not all(part.isidentifier() for part in module.split("."))
                or not attribute.isidentifier()
            ):
                raise ConfigurationError(f"invalid adapter reference: {reference!r}")

    @classmethod
    def for_module(cls, module):
        return cls(f"{module}:describe", f"{module}:create", f"{module}:preflight")


class AdapterRegistry:
    """Import-free declarations. Modules implement describe/create/preflight lazily."""

    def __init__(self):
        self.backbones = {
            name: AdapterSpec("draftkv.backbones.runtime:describe",
                              "draftkv.backbones.runtime:create",
                              "draftkv.backbones.runtime:preflight")
            for name, module in (
                ("LongLive1.3B", "longlive"),
                ("LingBot1.3B", "lingbot"),
                ("LingBot14B", "lingbot"),
                ("SolarWM33B", "solarwm"),
            )
        }
        self.algorithms = {
            name: AdapterSpec("draftkv.backbones.runtime:describe_algorithm",
                              "draftkv.backbones.runtime:create_algorithm",
                              "draftkv.backbones.runtime:preflight_algorithm")
            for name in (
                "native",
                "draftkv",
                "shadowkv",
                "infinigen",
                "infllm",
                "freekv",
                "deep-forcing",
                "future-forcing",
                "worldkv",
                "worldtrace-landmark",
            )
        }

    def register_backbone(self, name, spec):
        self.backbones[name] = spec

    def register_algorithm(self, name, spec):
        self.algorithms[name] = spec

    def _spec(self, config, kind, name):
        supplied = config.get(f"{kind}_adapter")
        if supplied is not None and not (kind == "backbone" and isinstance(supplied, dict)
                                             and not {"descriptor", "factory", "preflight"} & set(supplied)):
            try:
                return AdapterSpec(**supplied)
            except TypeError as exc:
                raise ConfigurationError(f"invalid {kind}_adapter: {exc}") from exc
        registry = self.backbones if kind == "backbone" else self.algorithms
        if name not in registry:
            raise AdapterNotImplemented(f"no registered {kind} adapter for {name}")
        return registry[name]

    def validate_dataset(self, context):
        """Bind native requests after the frozen dataset has been resolved."""
        if context.config.get("adapter") is not None:
            return
        spec = self._spec(context.config, "backbone", context.config["backbone"])
        if spec.preflight == "draftkv.backbones.runtime:preflight":
            with cpu_preflight():
                result = load_callable("draftkv.backbones.runtime:validate_dataset", adapter=True)(context)
                require_success(result, "native dataset binding", ("READY",))
            context.record("preflight.native_dataset", result)

    def preflight(self, context):
        config = context.config
        legacy = config.get("adapter")
        specs = []
        caps = []
        factories = []
        validators = []
        origins = {}
        checks = []
        kernels = []
        with cpu_preflight():
            if legacy is not None:
                if not isinstance(legacy, dict) or set(legacy) != {
                    "commands",
                    "capabilities",
                }:
                    raise AdapterNotImplemented(
                        "legacy adapters require commands and explicit backbone/algorithm capabilities; use composed adapters"
                    )
                caps = [
                    Capabilities.parse(legacy["capabilities"][k])
                    for k in ("backbone", "algorithm")
                ]
                command = CommandAdapter(legacy, context)
                stages = (
                    ("setup", "run")
                    if context.state.command == "pipeline"
                    else (context.state.command,)
                )
                if any(stage not in command.commands for stage in stages):
                    raise AdapterNotImplemented(
                        f"adapter commands missing required stages: {stages}"
                    )
            else:
                for kind in ("backbone", "algorithm"):
                    spec = self._spec(config, kind, config[kind])
                    specs.append(spec)
                    caps.append(
                        Capabilities.parse(
                            load_callable(spec.descriptor, adapter=True)(config)
                        )
                    )
                    factories.append(load_callable(spec.factory, adapter=True))
                    validators.append(load_callable(spec.preflight, adapter=True))
                    for reference in asdict(spec).values():
                        module = sys.modules[reference.split(":")[0]]
                        origin = getattr(module, "__file__", None)
                        origins[reference] = (
                            {
                                "path": str(Path(origin).resolve()),
                                "sha256": asset_sha256(Path(origin)),
                            }
                            if origin
                            else None
                        )
            for kind, capability in zip(("backbone", "algorithm"), caps):
                if capability.identity != config[kind]:
                    raise ConfigurationError(
                        f"{kind} identity mismatch: requested {config[kind]}, got {capability.identity}; fallback forbidden"
                    )
                if (
                    kind == "algorithm"
                    and config[kind] != "native"
                    and not capability.kernels
                ):
                    raise AdapterNotImplemented(
                        f"{config[kind]} must declare a pinned kernel/source implementation"
                    )
                protocol = config["protocol"]
                if (
                    protocol["input_mode"] not in capability.input_modes
                    or protocol["protocol_id"] not in capability.protocols
                ):
                    raise ConfigurationError(
                        f"{capability.identity} does not support current dataset protocol/input modality"
                    )
                for weight in capability.required_weights:
                    asset = (
                        config.get("checkpoint")
                        if weight == "checkpoint"
                        else config.get("weights", {}).get(weight)
                    )
                    if not asset:
                        raise ConfigurationError(
                            f"{capability.identity} requires weight {weight}"
                        )
                for kernel in capability.kernels:
                    if not isinstance(kernel.get("path"), str):
                        raise ConfigurationError("kernel requires a path")
                    path = resolve_path(kernel["path"], context.root)
                    # A pinned implementation inventory may legitimately contain
                    # zero-byte Python package markers (for example __init__.py).
                    # Existence + exact SHA-256 is the installation contract here;
                    # non-empty checks belong to produced runtime artifacts, not
                    # immutable source files.
                    if not path.is_file():
                        raise AdapterNotImplemented(
                            f"algorithm kernel not installed: {path}"
                        )
                    actual = asset_sha256(path)
                    if kernel.get("sha256") != actual:
                        raise ConfigurationError(
                            f"algorithm kernel requires a matching SHA-256: {path}"
                        )
                    kernels.append(
                        {
                            "identity": capability.identity,
                            "path": str(path),
                            "sha256": actual,
                        }
                    )
            if caps[1].requires_streaming_kv and not caps[0].streaming_kv:
                raise ConfigurationError(
                    "algorithm requires streaming KV unsupported by backbone"
                )
            if config.get("streaming_kv", True) and not all(
                c.streaming_kv for c in caps
            ):
                raise ConfigurationError(
                    "streaming KV requested but unsupported by adapter pair"
                )
            # Missing adapters are reported before touching the hardware probe.
            mapping = device_mapping(config["selected_devices"])
            # Bench2 launch policy is runtime-authoritative: historical/free-memory
            # estimates are telemetry, never generation admission gates.
            validate_hardware(caps, mapping, enforce_memory=config.get("benchmark") != "bench2")
            config["device_mapping"] = mapping
            context.state.device_mapping = mapping
            with runtime_environment(context):
                for validator in validators:
                    checks.append(
                        require_success(
                            validator(context), "adapter preflight", ("READY",)
                        )
                    )
        context.state.integration_sha = json_sha256(
            {
                "capabilities": [asdict(c) for c in caps],
                "entry_points": [asdict(s) for s in specs],
                "origins": origins,
                "compatibility": [r.get("fingerprint", {}) for r in checks],
            }
        )
        context.record(
            "preflight.adapters",
            {
                "status": "READY",
                "capabilities": [asdict(c) for c in caps],
                "entry_points": [asdict(s) for s in specs],
                "device_mapping": mapping,
                "checks": checks,
                "entry_point_hashes": origins,
                "kernels": kernels,
            },
        )
        # Factory invocation stays outside CPU preflight, after all data checks.
        return (
            (lambda: command)
            if legacy is not None
            else (lambda: ComposedAdapter(*factories, context))
        )


class PluginAdapter:
    """Compatibility for explicit old factories; never a default algorithm fallback.

    Runtime v2 still requires generate(context) and per-case completion receipts.
    An old run-only plugin can be inspected/setup but cannot silently qualify a run.
    """
    def __init__(self, reference, stages, context):
        self.plugin = load_callable(reference, adapter=True)(context)
        for stage in stages:
            if not callable(getattr(self.plugin, stage, None)):
                raise AdapterNotImplemented(f"plugin has no {stage}(context)")

    def setup(self, context):
        result = self.plugin.setup(context)
        adapter_result(result, "plugin.setup")
        return result

    def run(self, context):
        result = self.plugin.run(context)
        adapter_result(result, "plugin.run")
        return result

    def generate(self, context):
        if not callable(getattr(self.plugin, "generate", None)):
            raise AdapterNotImplemented("legacy run-only plugin needs a Runtime v2 generate/case-receipt bridge")
        result = self.plugin.generate(context)
        adapter_result(result, "plugin.generate")
        return result
