"""Existing Runtime PluginAdapter bridge; no CLI, scheduler or engine changes."""

import json
import os
from pathlib import Path
import subprocess

from draftkv.registry import Group, algorithm_package
from .contracts import BackboneError
from .planning import make_plan, sha256


class NativeBackboneAdapter:
    def __init__(self, context):
        self.group = Group(context.state.benchmark, context.state.backbone, context.state.algorithm)

    def _plan(self, context):
        configuration = backbone_configuration(context.config)
        if not isinstance(configuration, dict):
            raise BackboneError("BLOCKED_CONFIGURATION", "backbone_adapter external paths and frozen request required")
        plan = make_plan(Path(context.root), self.group, context.state.sp, configuration,
                         context.config.get("checkpoint", {}).get("path", ""), context.run_dir)
        plan["gpu_profile"] = context.config.get("gpu_profile", False)
        plan["selected_devices"] = list(context.state.selected_devices)
        plan["selected_device_metadata"] = list((getattr(context.state, "device_mapping", {}) or {}).get("ranks", []))
        if self.group.algorithm == "worldtrace-landmark":
            evaluation = context.config.get("evaluation_config", {})
            revision = None
            if evaluation.get("official_root"):
                try:
                    revision = subprocess.check_output(
                        ["git", "-C", evaluation["official_root"], "rev-parse", "HEAD"],
                        text=True, stderr=subprocess.DEVNULL).strip()
                except (OSError, subprocess.CalledProcessError):
                    pass
            plan["evaluator_provenance"] = dict(configuration=evaluation, revision=revision)
        return plan

    def setup(self, context):
        plan = self._plan(context)
        return {"status": "READY", "sp": plan["sp"],
                "input_mode": plan["request"]["input_mode"], "release_e2e": False,
                "note": "source/assets checked without importing torch; ABI/kernel execution is not certified"}

    def run(self, context):
        plan = self._plan(context)
        page_fields = {key: plan[key] for key in ('host_pages_backend', 'host_pages_policy', 'host_pages_role') if key in plan}
        if page_fields:
            context.state.method_provenance['host_pages'] = page_fields
            context.save()
        if plan["algorithm"] in ("deep-forcing", "future-forcing"):
            context.state.method_provenance.setdefault("case_geometries", {})[
                plan["request"]["case_id"]] = plan["native_output_geometry"]
            context.save()
        path = Path(context.run_dir) / "backbone_plan.json"
        with path.open("x") as output:
            json.dump(plan, output, indent=2, allow_nan=False)
        if Path(plan["output"]).exists():
            raise BackboneError("OUTPUT_EXISTS", plan["output"])
        argv = [plan["python"], "-m", "torch.distributed.run", "--standalone",
                "--log-dir", str(Path(context.run_dir) / "rank_logs"), "--redirects", "3", "--tee", "3",
                f"--nproc_per_node={plan['sp']}", "-m", "draftkv.backbones.worker", "--plan", str(path)]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = context.config.get("device_mapping", {}).get(
            "cuda_visible_devices", ",".join(map(str, context.state.selected_devices)))
        environment["PYTHONPATH"] = os.pathsep.join((str(Path(__file__).resolve().parents[2]),
                                                    environment.get("PYTHONPATH", "")))
        from .memory_profile import memory_session, finalize_memory_profile
        with memory_session(plan, environment):
            with (Path(context.run_dir) / "backbone_command.json").open("x") as output:
                json.dump({"argv": argv, "cwd": str(context.root),
                           "CUDA_VISIBLE_DEVICES": environment["CUDA_VISIBLE_DEVICES"]}, output, indent=2)
            with context.stdout_path.open("a") as stdout, context.stderr_path.open("a") as stderr:
                subprocess.run(argv, cwd=context.root, env=environment, stdout=stdout, stderr=stderr, check=True)
        video = Path(plan["output"])
        if not video.is_file() or not video.stat().st_size:
            raise BackboneError("MISSING_OUTPUT", "author entry returned without a real video")
        probe = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                                "-show_entries", "stream=width,height,avg_frame_rate,nb_read_frames", "-of", "json", str(video)],
                               check=True, capture_output=True, text=True)
        streams = json.loads(probe.stdout)["streams"]
        if len(streams) != 1:
            raise BackboneError("OUTPUT_SPEC_MISMATCH", "expected one video stream")
        from draftkv.algorithms.forcing.geometry import output_request
        stream, request = streams[0], output_request(plan)
        from fractions import Fraction
        if (int(stream["nb_read_frames"]) != request["frames"]
                or Fraction(stream["avg_frame_rate"]) != request["fps"]
                or stream["width"] != request["output_width"] or stream["height"] != request["output_height"]):
            raise BackboneError("OUTPUT_SPEC_MISMATCH", str(stream))
        for rank in range(plan["sp"]):
            trace = Path(context.run_dir) / "traces" / f"backbone_rank{rank}.jsonl"
            events = [json.loads(line) for line in trace.read_text().splitlines()]
            if not events or not any(row["event"] == "weights_loaded" for row in events) or events[-1]["event"] != "worker_complete":
                raise BackboneError("TRACE_INCOMPLETE", f"rank {rank} missing real model completion")
            validate_algorithm_trace(plan, events)
        from .gpu_profile import aggregate_profiles
        gpu_profile = aggregate_profiles(plan)
        result = {"status": "SUCCEEDED", "video": str(video), "sha256": sha256(video), "stream": stream,
                  **page_fields,
                  "trace_dir": str(Path(context.run_dir) / "traces"), "algorithm": plan["algorithm"],
                  "scope": "NATIVE_MODEL_EXECUTION_OFFICIAL_EVALUATION_SEPARATE", "release_e2e": False}
        if gpu_profile is not None:
            result["gpu_profile"] = gpu_profile
            memory = finalize_memory_profile(context.run_dir, expected_sp=plan['sp'], worker_returncode=0)
            result['memory_profile'] = dict(path=str(Path(context.run_dir) / 'memory_profile.json'),
                comparison_eligible=memory['memory_comparison_eligible'])
        with (Path(context.run_dir) / "backbone_result.json").open("x") as output:
            json.dump(result, output, indent=2)
        return result


# Runtime v2 composition is CPU import safe. Model factories stay in workers.
def backbone_configuration(config):
    value = config.get("backbone_config", config.get("backbone_adapter"))
    if not isinstance(value, dict) or {"descriptor", "factory", "preflight"} & set(value):
        raise BackboneError("BLOCKED_CONFIGURATION", "draftkv.backbones requires backbone_config source/assets/request bindings")
    return value



def _protocol(config):
    group = Group(config["benchmark"], config["backbone"], config["algorithm"])
    return group, group.protocol.to_dict()


def describe(config):
    from .catalog import model_spec
    group, protocol = _protocol(config)
    backbone_configuration(config)
    spec = model_spec(group.backbone)
    return {"identity": group.backbone, "input_modes": [protocol["input_mode"]],
            "protocols": [protocol["protocol_id"]], "supported_sp": list(spec.native_sp),
            "min_free_memory_mib": {str(sp): 1 for sp in spec.native_sp}, "streaming_kv": True}


def describe_algorithm(config):
    group, protocol = _protocol(config)
    implemented = {"native", "draftkv", "infllm", "shadowkv", "infinigen", "freekv", "worldkv", "worldtrace-landmark", "deep-forcing", "future-forcing"}
    if group.algorithm not in implemented:
        raise BackboneError("BLOCKED_ALGORITHM_SEMANTICS", "unregistered original hook " + group.algorithm)
    from .catalog import model_spec
    model = model_spec(group.backbone)
    supported = model.draftkv_sp if group.algorithm == "draftkv" else model.native_sp
    kernels = []
    if group.algorithm != "native":
        configuration = backbone_configuration(config)
        plugin = configuration.get("algorithm_plugin")
        expected = "src/draftkv/algorithms/" + algorithm_package(group.algorithm) + "/entry.py"
        if not isinstance(plugin, dict) or plugin.get("entry") != expected:
            raise BackboneError("BLOCKED_ALGORITHM_SEMANTICS", "bind the audited " + expected + " plugin; legacy embedded owners are not the canonical release path")
        root = Path(config["git_root"])
        source = external_plugin_path(plugin, root)
        from .planning import verify_files
        verify_files(source, plugin.get("files", {}))
        for relative, digest in plugin.get("files", {}).items():
            path = source / relative
            kernels.append({"path": str(path), "sha256": digest})
        if not kernels:
            raise BackboneError("BLOCKED_SOURCE", "algorithm plugin requires a complete SHA-256 source inventory")
    return {"identity": group.algorithm, "input_modes": [protocol["input_mode"]],
            "protocols": [protocol["protocol_id"]], "supported_sp": list(supported),
            "min_free_memory_mib": {str(sp): 1 for sp in supported}, "streaming_kv": True,
            "requires_streaming_kv": group.algorithm == "draftkv", "kernels": kernels}


def external_plugin_path(plugin, root):
    from .contracts import external_path
    return external_path(plugin["root"], root)


class RuntimeBackbone(NativeBackboneAdapter):
    def _case_contexts(self, context):
        from copy import copy, deepcopy
        from .contracts import external_path
        from .planning import read_json
        configuration = backbone_configuration(context.config)
        requests = configuration.get("requests")
        if not isinstance(requests, dict) or not requests:
            raise BackboneError("BLOCKED_DATASET_BINDING", "draftkv.backbones requires a SHA-frozen native request for every selected case in backbone_config.requests")
        ids = context.state.case_ids
        if not ids:
            # Adapter preflight precedes Dataset resolution. Apply the same
            # filter/limit in frozen request order before planning model cases.
            ids = list(requests)
            selection = context.config.get("selection", {})
            requested = selection.get("case_ids")
            if requested:
                if any(case_id not in requests for case_id in requested):
                    raise BackboneError("BLOCKED_DATASET_BINDING", "selection includes cases outside the frozen native requests")
                ids = [case_id for case_id in ids if case_id in requested]
            limit = selection.get("limit_cases")
            if limit:
                ids = ids[:limit]
        for index, case_id in enumerate(ids):
            binding = requests.get(case_id)
            if not isinstance(binding, dict) or set(binding) != {"request", "request_sha256"}:
                raise BackboneError("BLOCKED_DATASET_BINDING", "missing exact request/SHA binding for " + case_id)
            path = external_path(binding["request"], context.root)
            if not path.is_file() or sha256(path) != binding["request_sha256"]:
                raise BackboneError("BLOCKED_INPUT_PROTOCOL", "native request changed: " + case_id)
            request = read_json(path)
            if request.get("case_id") != case_id or request.get("execution_scope") != "release_candidate":
                raise BackboneError("BLOCKED_DATASET_BINDING", "diagnostic/different-case request cannot become release evidence: " + case_id)
            if request.get("protocol") != self.group.protocol.to_dict():
                raise BackboneError("BLOCKED_INPUT_PROTOCOL", "native request protocol differs from Runtime Dataset")
            if context.state.dataset_sha and request.get("runtime_dataset_sha256") != context.state.dataset_sha:
                raise BackboneError("BLOCKED_DATASET_BINDING", "native request is not bound to the Runtime dataset asset")
            if context.dataset:
                manifest_sha = context.dataset.get("native_manifest_sha256")
                row_sha = context.dataset.get("native_case_row_hashes", {}).get(case_id)
                if (not manifest_sha or not row_sha
                        or request.get("dataset_manifest_sha256") != manifest_sha
                        or request.get("dataset_row_sha256") != row_sha):
                    raise BackboneError("BLOCKED_DATASET_BINDING", "native conditioning is not bound to the resolved frozen Dataset row: " + case_id)
            child = copy(context)
            child.config = deepcopy(context.config)
            child.config["backbone_config"] = dict(configuration, **binding)
            if self.group.algorithm == "draftkv":
                plugin = child.config["backbone_config"].get("algorithm_plugin", {})
                settings = plugin.setdefault("config", {})
                settings["profile_admission"] = "runtime"
                settings.pop("hardware_profile_binding", None)
            child.run_dir = context.run_dir / "cases" / f"{index:04d}"
            yield case_id, child

    def validate_dataset(self, context):
        cases = list(self._case_contexts(context))
        return {"status": "READY", "case_count": len(cases), "release_e2e": False}

    def setup(self, context):
        plans = [self._plan(child) for _, child in self._case_contexts(context)]
        return {"status": "READY", "case_count": len(plans), "release_e2e": False}

    def generate(self, context):
        for case_id, child in self._case_contexts(context):
            if case_id not in context.pending_case_ids:
                continue
            child.run_dir.mkdir(parents=True, exist_ok=False)
            result = self.run(child)
            context.complete_case(case_id, result["video"], expected_sha256=result["sha256"])
            context.record("native.case." + case_id, result)
        return {"status": "SUCCEEDED", "release_e2e": False}


def validate_dataset(context):
    """Recheck actual Dataset manifest/row bindings after Dataset resolution."""
    return RuntimeBackbone(context).validate_dataset(context)


def create(context):
    return RuntimeBackbone(context)


def preflight(context):
    _protocol(context.config)
    adapter = RuntimeBackbone(context)
    plans = [adapter._plan(child) for _, child in adapter._case_contexts(context)]
    return {"status": "READY", "fingerprint": {"case_count": len(plans), "source_files": [p["source_files"] for p in plans]}, "release_e2e": False}


class RuntimeAlgorithm:
    def attach(self, backbone, context):
        describe_algorithm(context.config)
        return {"status": "READY", "consumer": "worker_hook_registry", "release_e2e": False}


def create_algorithm(context):
    return RuntimeAlgorithm()


def preflight_algorithm(context):
    describe_algorithm(context.config)
    original_provider = None
    if context.state.algorithm in ("shadowkv", "infinigen", "freekv", "worldkv", "worldtrace-landmark"):
        import importlib
        module = importlib.import_module("draftkv.algorithms." + algorithm_package(context.state.algorithm) + ".entry")
        report = module.preflight(context.config, Path(context.root))
        if report.get("status", "READY") != "READY":
            raise BackboneError(report.get("status", "BLOCKED_DEPENDENCY"), str(report.get("reason", report)))
        original_provider = report
    return {"status": "READY", "fingerprint": {"algorithm": context.state.algorithm,
            "original_provider": original_provider}, "release_e2e": False}


NATIVE_CONSUMERS = {"FA3": ("flash_attn_interface._flash_attn_forward", "flash_attn_3._C"),
                    "FA2": ("flash_attn.flash_attn_func", "flash_attn_2_cuda")}


def _native_allowed_recipe(plan, row):
    """Replay the production resolver against actual, inventory-bound device metadata."""
    from draftkv.streaming.table2_native import runtime_config
    from draftkv.backbones.catalog import MODELS
    from draftkv.backbones.sequence_parallel import local_heads
    device = row.get("native_device") or {}
    capability = device.get("capability")
    if (not isinstance(capability, list) or len(capability) != 2
            or any(type(v) is not int for v in capability) or not device.get("uuid")
            or not isinstance(device.get("name"), str)):
        raise ValueError("missing actual consumer device identity")
    inventory = plan.get("selected_device_metadata")
    if not inventory:
        raise ValueError("plan lacks the admitted device inventory")
    normalize_uuid = lambda x: str(x).lower().removeprefix("gpu-").replace("-", "")
    admitted = [d for d in inventory if normalize_uuid(d.get("uuid")) == normalize_uuid(device["uuid"])]
    normalize_name = lambda x: " ".join(str(x).lower().replace("nvidia", "").split())
    if len(admitted) != 1 or normalize_name(admitted[0].get("name")) != normalize_name(device["name"]):
        raise ValueError("consumer device differs from the admitted inventory")
    name = device["name"].upper()
    families = (("H100", [9, 0]), ("H200", [9, 0]), ("4090", [8, 9]),
                ("5090", [12, 0]), ("B200", [10, 0]))
    for family, expected in families:
        if family in name and capability != expected:
            raise ValueError("consumer compute capability differs from the admitted GPU")
    spec = MODELS[plan["backbone"]]
    heads = local_heads(spec.heads, spec.family, plan.get("sp", 1))
    if row.get("local_heads") != heads:
        raise ValueError("consumer local heads differ from the planned topology")
    settings = (plan.get("algorithm_plugin") or {}).get("config") or {}
    recipe = runtime_config(settings, capability=tuple(capability),
                            native_version=plan.get("native_attention_version"),
                            heads=heads, default_slots=True, physical_sp=plan.get("sp", 1),
                            device_name=device["name"])
    if recipe is None:
        raise ValueError("plan does not admit native FlashAttention streaming")
    return recipe


def _validate_native_consumers(plan, events, rows):
    """Check actual calls against plan, CUDA inventory, exact recipe and DMA/slot evidence."""
    import re
    from draftkv.streaming.table2_native import native_config_sha256, native_receipt_identity
    from draftkv.streaming.slot_native import slot_geometry
    hexdigest = lambda value: bool(re.fullmatch(r"[a-f0-9]{64}", str(value or "")))
    settings = (plan.get("algorithm_plugin") or {}).get("config") or {}
    expected_profile = (settings.get("hardware_profile_binding") or {}).get("sha256")
    checkpoint = plan.get("checkpoint_sha256")
    positions = {(r.get("forward_id"), r.get("layer")): r for r in events
                 if r.get("event") == "draftkv_time_position"}
    headers=[r for r in events if r.get('event')=='draftkv_native_consumer_identity']
    admitted={r.get('identity_id'):r.get('identity') for r in headers}
    if not admitted or len(admitted)!=len(headers):
        raise BackboneError('TRACE_INCOMPLETE','missing/duplicate native consumer identity header')
    identities = set()
    for row in rows:
        try:
            backend = row.get("consumer_backend")
            symbol, extension = NATIVE_CONSUMERS[backend]
            if (row.get("consumer_kind") != "native_flash_attention" or row.get("consumer_symbol") != symbol
                    or row.get("consumer_extension") != extension or not hexdigest(row.get("source_sha"))):
                raise ValueError("missing real native FlashAttention symbol/extension receipt")
            if not hexdigest(checkpoint) or row.get("checkpoint_sha") != checkpoint:
                raise ValueError("native consumer checkpoint identity differs from the plan")
            if row.get("profile_sha") != expected_profile:
                raise ValueError("native consumer hardware profile differs from the bound profile")
            identity_id,identity=native_receipt_identity(row)
            if row.get('native_consumer_id')!=identity_id or admitted.get(identity_id)!=identity:
                raise ValueError('native consumer extension/profile/checkpoint differs from its run identity')
            recipe = _native_allowed_recipe(plan, row)
            if backend != f"FA{recipe['version']}" or row.get("device_profile") != recipe["device_profile"]:
                raise ValueError("native FlashAttention backend differs from the device policy")
            identities.add((row["source_sha"], row["native_device"]["uuid"]))
            if row.get("final_join") is not True:
                raise ValueError("native consumer lacks completed final join")
            total = row.get("kv_total_tokens")
            if type(total) is not int or total < 1:
                raise ValueError("native consumer lacks actual KV extent")
            historical = row.get("event") == "draftkv_b01_attention"
            slots = recipe.get("kv_slots", 0) if historical else 0
            if slots:
                splits, used, cap = slot_geometry(total, recipe["splits"], slots, recipe["kv_slot_tokens"])
                recipe = dict(recipe, splits=splits)
                units = (row["local_heads"] // recipe["head_group"]) * splits
                if (row.get("native_executor") != "bounded_slots" or row.get("kv_slots") != used
                        or row.get("kv_slot_tokens") != cap or row.get("external_splits") != splits
                        or row.get("receive_capacity_bytes") != used * 2 * cap * recipe["head_group"] * 128 * 2
                        or row.get("slot_unit_count") != units or row.get("slot_reuse_count") != max(0, units-used)
                        or row.get("slot_completion_protocol") != "consumer_ready_before_reuse"
                        or row.get("slot_final_consumer_complete") is not True):
                    raise ValueError("bounded KV slot capacity/reuse differs from the planned executor")
            elif "kv_slots" in row or row.get("native_executor") != "direct":
                raise ValueError("unplanned bounded KV slots or executor")
            actual = row.get("native_runtime_config")
            if (actual != recipe or row.get("native_config_sha256") != native_config_sha256(actual)
                    or row.get("producer_schedule") != recipe["schedule"]
                    or row.get("submission") != recipe["submission"]):
                raise ValueError("native consumer runtime recipe differs from the plan")
            if not historical:
                if row.get("h2d_bytes") != 0:
                    raise ValueError("resident native consumer reports historical H2D")
                continue
            position = positions.get((row.get("forward_id"), row.get("layer")))
            tokens = position.get("history_tokens") if position else None
            if (row.get("producer") not in ("cpp_async", "cpp_unit_batch") or type(tokens) is not int or tokens <= 0
                    or type(row.get("h2d_bytes")) is not int
                    or row["h2d_bytes"] != tokens * row["local_heads"] * 128 * 4):
                raise ValueError("historical native attention lacks exact pinned-host C++ producer/H2D bytes")
            if position.get("token_budget") is not None:
                prefix = row.get("fixed_prefix_tokens", 0)
                if (type(prefix) is not int or prefix < 0 or total != position["token_budget"] + prefix
                        or (prefix and plan["backbone"] != "SolarWM33B")):
                    raise ValueError("native KV extent differs from its actual source-token map")
        except (ValueError, TypeError, KeyError, ZeroDivisionError) as exc:
            raise BackboneError("TRACE_INCOMPLETE", str(exc)) from exc
    if len(identities) != 1:
        raise BackboneError("TRACE_INCOMPLETE", "native consumer extension/device identity changed within one rank")


def validate_algorithm_trace(plan, events):
    from draftkv.algorithms.draftkv.experiment import experiment_scope
    with experiment_scope((plan.get('algorithm_plugin') or {}).get('config', {}), plan['backbone']):
        return _validate_algorithm_trace(plan, events)


def _validate_algorithm_trace(plan, events):
    """Acceptance reads actual per-layer consumer receipts, never configuration labels."""
    if plan["algorithm"] == "native":
        return
    from .catalog import MODELS
    if plan["algorithm"] != "draftkv":
        layers = set(range(MODELS[plan["backbone"]].layers))
        active = None
        for event in events:
            kind = event.get("event")
            if kind == "algorithm_forward_begin":
                if active is not None or type(event.get("clean")) is not bool or type(event.get("start_frame")) is not int or event["start_frame"] < 0:
                    raise BackboneError("TRACE_INCOMPLETE", "overlapping or invalid native forward transaction")
                active = event["forward_id"]
            elif kind in ("attention", "clean_commit", "algorithm_forward_complete"):
                if active is None or event.get("forward_id") != active:
                    raise BackboneError("TRACE_INCOMPLETE", "consumer/commit outside its native forward transaction")
                if kind == "algorithm_forward_complete":
                    active = None
        if active is not None:
            raise BackboneError("TRACE_INCOMPLETE", "unterminated native forward transaction")
        begins = [r for r in events if r.get("event") == "algorithm_forward_begin"]
        completes = [r for r in events if r.get("event") == "algorithm_forward_complete"]
        ids = [r.get("forward_id") for r in begins]
        if not ids or ids != list(range(1, len(ids)+1)) or [r.get("forward_id") for r in completes] != ids:
            raise BackboneError("TRACE_INCOMPLETE", "missing original algorithm forward transactions")
        rows = [r for r in events if r.get("event") == "attention"]
        if {r.get("forward_id") for r in rows} != set(ids):
            raise BackboneError("TRACE_INCOMPLETE", "unexpected/missing per-forward attention evidence")
        for begin in begins:
            consumed = [r for r in rows if r.get("forward_id") == begin["forward_id"]]
            if (len(consumed) != len(layers) or {r.get("layer") for r in consumed} != layers
                    or any(r.get("consumer") != "native_fa2" or type(r.get("native_fa2_calls")) is not int or r["native_fa2_calls"] < 1
                           or r.get("consumer_symbol") != ("flash_attn.flash_attn_func" if plan["backbone"] == "SolarWM33B" else "wan.modules.attention.flash_attention(version=2)")
                           or r.get("clean") != begin.get("clean")
                           or r.get("start_frame") != begin.get("start_frame")
                           or r.get("attention_branch") != begin.get("attention_branch") for r in consumed)):
                raise BackboneError("TRACE_INCOMPLETE", "every original algorithm layer must prove actual Native FA2 consumption")
            commits = [r for r in events if r.get("event") == "clean_commit" and r.get("forward_id") == begin["forward_id"]]
            if (len(commits) != int(begin["clean"]) or any(r.get("layers") != len(layers) or r.get("start_frame") != begin["start_frame"]
                           or r.get("attention_branch") != begin.get("attention_branch") for r in commits)):
                raise BackboneError("TRACE_INCOMPLETE", "original algorithm clean transaction did not commit")
        if plan["algorithm"] == "worldtrace-landmark":
            configurations = [e for e in events if e.get("event") == "worldtrace_configuration"]
            reads = [e for e in events if e.get("event") == "worldtrace_read"]
            detectors = [e for e in events if e.get("event") == "worldtrace_detector"]
            if (not configurations or len(reads) != len(rows)
                    or len(detectors) != sum(int(b["clean"]) for b in begins)):
                raise BackboneError("TRACE_INCOMPLETE", "WorldTrace configuration/read/detector path missing")
        return
    if any(row.get("consumer") == "native_fa2" or row.get("consumer_kind") == "native_fa2" for row in events):
        raise BackboneError("BLOCKED_ALGORITHM_SEMANTICS", "DraftKV invoked Native FA2")
    layers = set(range(MODELS[plan["backbone"]].layers))
    real = [r for r in events if r.get("event") in ("draftkv_b01_attention", "draftkv_b01_resident")]
    if {r.get("layer") for r in real} != layers:
        raise BackboneError("TRACE_INCOMPLETE", "every actual model layer must prove canonical Streaming FA2 consumption")
    begins = [r for r in events if r.get("event") == "algorithm_forward_begin"]
    completes = [r for r in events if r.get("event") == "algorithm_forward_complete"]
    ids = [r.get("forward_id") for r in begins]
    if (not ids or ids != list(range(1, len(ids) + 1))
            or [r.get("forward_id") for r in completes] != ids
            or {r.get("forward_id") for r in real} != set(ids)):
        raise BackboneError("TRACE_INCOMPLETE", "missing or unmatched whole-model forward receipts")
    for forward_id in ids:
        consumed = [r.get("layer") for r in real if r.get("forward_id") == forward_id]
        if len(consumed) != len(layers) or set(consumed) != layers:
            raise BackboneError("TRACE_INCOMPLETE", "each model forward must consume exactly once in every layer")
        begin = begins[forward_id - 1]
        if any(r.get("start_frame") != begin.get("start_frame") or r.get("clean") != begin.get("clean")
               for r in real if r.get("forward_id") == forward_id):
            raise BackboneError("TRACE_INCOMPLETE", "real consumer receipts differ from their model-forward transaction")
        if begin.get("clean") is True and not any(r.get("event") == "draftkv_clean_commit"
                                                  and r.get("forward_id") == forward_id for r in events):
            raise BackboneError("TRACE_INCOMPLETE", "each clean model forward requires its own atomic commit receipt")
    import re
    symbols = {"b01_ready_ext.fwd", "b01_frontier_opt_ext.fwd", "fa2_streaming_consumer_ext.streaming_fwd",
               "offload_fa2_ext.fwd"}
    native = [r for r in real if r.get("consumer_kind") == "native_flash_attention"
              or r.get("consumer_backend") in NATIVE_CONSUMERS]
    if native and len(native) != len(real):
        raise BackboneError("TRACE_INCOMPLETE", "mixed native FlashAttention and B01 consumer receipts in one run")
    if native:
        _validate_native_consumers(plan, events, native)
    else:
        from draftkv.streaming.table2_native import runtime_config
        settings=(plan.get('algorithm_plugin') or {}).get('config') or {}
        for device in plan.get('selected_device_metadata') or []:
            name=str(device.get('name','')).upper()
            for family,cap in (('H100',(9,0)),('H200',(9,0)),('4090',(8,9)),('5090',(12,0)),('B200',(10,0))):
                if family in name and runtime_config(settings,capability=cap,heads=MODELS[plan['backbone']].heads,default_slots=True) is not None:
                    raise BackboneError('TRACE_INCOMPLETE','B01 receipt conflicts with the planned native device policy')
    for row in real:
        for field in ('host_pages_backend', 'host_pages_policy', 'host_pages_role'):
            if field in plan and row.get(field) != plan[field]:
                raise BackboneError("TRACE_INCOMPLETE", "actual host-page policy differs from generation plan")
        if native:
            continue
        if (row.get("consumer_kind") != "b01_streaming" or row.get("consumer_symbol") not in symbols
                or any(not re.fullmatch(r"[a-f0-9]{64}", str(row.get(name, "")))
                       for name in ("source_sha", "profile_sha", "checkpoint_sha"))):
            raise BackboneError("TRACE_INCOMPLETE", "missing real Streaming FA2 symbol/source/profile/checkpoint receipt")
        if row["event"] == "draftkv_b01_attention" and (not row.get("producer_plan_id") or row.get("producer_symbol") != "producer_plan_ext.ProducerPlan"
                                                           or type(row.get("h2d_bytes")) is not int or row["h2d_bytes"] <= 0):
            raise BackboneError("TRACE_INCOMPLETE", "historical attention lacks actual original ProducerPlan/H2D receipt")
    block_mode = plan.get('algorithm_plugin', {}).get('config', {}).get('retrieval_granularity') == 'block'
    block_events = any(row.get('retrieval_granularity') == 'block' for row in real)
    if block_mode or block_events:
        from draftkv.algorithms.draftkv.block_trace import validate_block_coordinates
        validate_block_coordinates(plan, events, real)
    elif plan.get('attention_position_policy') == 'chrono_compact':
        from draftkv.algorithms.draftkv.position import compact_map
        positions = [r for r in events if r.get('event') == 'draftkv_time_position']
        indexed = {(r.get('forward_id'), r.get('layer')): r for r in positions}
        if len(positions) != len(real) or len(indexed) != len(real):
            raise BackboneError('TRACE_INCOMPLETE', 'every DraftKV consumer requires a unique Chrono-Compact map')
        for row in real:
            position = indexed.get((row['forward_id'], row['layer']), {})
            sink = tuple(position.get('sink_ids', ()))
            selected = tuple(row.get('selected_frame_ids', ()))
            current = tuple(position.get('current_ids', ()))
            spec = MODELS[plan['backbone']]
            if (position.get('position_policy') != 'chrono_compact'
                    or position.get('scoring_qk_space') != 'native_post_rope'
                    or position.get('start_frame') != row['start_frame']
                    or position.get('clean') != row['clean']
                    or sink != tuple(range(min(spec.sink, row['start_frame'])))
                    or current != tuple(range(row['start_frame'], row['start_frame'] + len(current)))
                    or not 1 <= len(current) <= spec.chunk
                    or position.get('selected_ids') != list(selected)
                    or position.get('packing_policy') != 'chronological_frame_id'
                    or selected != tuple(sorted(selected))):
                raise BackboneError('TRACE_INCOMPLETE', 'invalid chronological DraftKV source IDs or coordinate contract')
            mapping = compact_map(sink, selected, current)
            packed = sink + selected + current
            if (len(packed) > spec.window
                    or position.get('chronological_selected_ids') != sorted(selected)
                    or position.get('spatial_and_camera_policy') != 'native_unchanged'
                    or position.get('value_policy') != 'native_unchanged'
                    or position.get('packed_source_ids') != list(packed)
                    or position.get('packed_temporal_slots') != list(range(len(packed)))
                    or position.get('current_temporal_slots') != [mapping[fid] for fid in current]
                    or position.get('raw_temporal_source') != 'clean_context.raw_key_direct_no_inverse'):
                raise BackboneError('TRACE_INCOMPLETE', 'Chrono-Compact chronological slot mapping differs')
    if plan["request"]["latent_frames"] > MODELS[plan["backbone"]].window:
        if not any(r.get("event") == "draftkv_b01_attention" and r.get("selected_frame_ids") for r in real):
            raise BackboneError("TRACE_INCOMPLETE", "long request never consumed actual historical KV")
    if not any(r.get("event") == "draftkv_clean_commit" for r in events):
        raise BackboneError("TRACE_INCOMPLETE", "canonical DraftKV clean commit is absent")
