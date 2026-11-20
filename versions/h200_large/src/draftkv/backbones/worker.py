"""Isolated native GPU worker. Importing this module does not start inference."""

import argparse
from contextlib import contextmanager
import importlib.util
import json
import os
from pathlib import Path
import sys

from .catalog import MODELS
from .contracts import BackboneError
from .failure_scope import isolated_cli_worker
from .hooks import HookRegistry, HookSession, bind_model
from .models import LingbotCore, LongliveCore, activate_sources
from .gpu_profile import run_profiled
from .planning import read_json, sha256, verify_files


class Trace:
    """Bounded asynchronous JSONL sink; never block attention on filesystem writes.

    Event serialization remains on the calling thread: a completed enqueue owns
    immutable JSON bytes, so mutable algorithm metadata cannot race the writer.
    The writer flushes batches; close() drains all events and propagates I/O errors.
    A bounded queue deliberately applies backpressure rather than losing evidence.
    """

    _STOP = object()

    def __init__(self, folder, rank):
        import atexit
        import queue
        import threading
        self.rank = rank
        self.path = Path(folder) / f"backbone_rank{rank}.jsonl"
        self._stream = self.path.open("x", buffering=1024 * 1024)
        self._queue = queue.Queue(maxsize=2048)
        self._error = None
        self._closed = False
        self.handle = self  # Retain worker's trace.handle.close() API.
        self._thread = threading.Thread(target=self._drain, name=f"draftkv-trace-rank{rank}", daemon=True)
        self._thread.start()
        # Ordinary interpreter shutdown drains this sink. Fatal CLI failures
        # use a separate synchronous error artifact and bypass exit-time joins.
        atexit.register(self.close)

    def _drain(self):
        try:
            rows = 0
            while True:
                line = self._queue.get()
                if line is self._STOP:
                    break
                self._stream.write(line)
                rows += 1
                if rows >= 64:
                    self._stream.flush()
                    rows = 0
            self._stream.flush()
        except BaseException as exc:
            self._error = exc
        finally:
            try:
                self._stream.close()
            except BaseException as exc:
                self._error = exc

    def _enqueue(self, line):
        import queue
        while True:
            if self._error is not None:
                raise RuntimeError("TRACE_IO_FAILED") from self._error
            try:
                self._queue.put(line, timeout=0.25)
                return
            except queue.Full:
                continue

    def __call__(self, event, **fields):
        if self._closed:
            raise RuntimeError("TRACE_CLOSED")
        line = json.dumps({**fields, "schema_version": 1, "event": event, "rank": self.rank},
                          allow_nan=False) + "\n"
        self._enqueue(line)

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._enqueue(self._STOP)
        self._thread.join()
        if self._error is not None:
            raise RuntimeError("TRACE_IO_FAILED") from self._error


def configure_environment(plan):
    from .latency_build import CACHE_VARIABLES
    # Eliminate inherited method selection; a prior group cannot activate an
    # algorithm, CPU diagnostic, disabled clean commit, or alternate consumer.
    for name in list(os.environ):
        if name in {"DRAFTKV_ATTEMPT_TOKEN", "DRAFTKV_TASK_ID", *CACHE_VARIABLES}:
            continue  # Scheduler ownership and infrastructure cache bindings.
        if name.startswith(("LINGBOT_", "LINGBOT13_", "LINGBOT14_", "KVRAG_", "DRAFTKV_", "H3_", "RAG_", "OFFLOAD_AUTOTUNE_")):
            os.environ.pop(name)
    os.environ.update(LINGBOT_RAG="off", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      TOKENIZERS_PARALLELISM="false")
    if (plan.get("backbone") == "LingBot14B" and plan["algorithm"] == "future-forcing"
            and not any(key in os.environ for key in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"))):
        # Variable-length cache/statistic temporaries otherwise strand allocator
        # fragments on 24-GiB cards. This changes allocation only, before CUDA
        # initialization; an explicitly supplied allocator policy takes priority.
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    if (plan.get("draftkv_loading_profile") ==
            "lingbot14b-sp8-bf16-stream-20261102"):
        if (plan["algorithm"] != "draftkv" or plan["backbone"] != "LingBot14B"
                or plan["sp"] != 8 or not plan.get("algorithm_plugin")):
            raise BackboneError("BLOCKED_CONFIGURATION", "invalid DraftKV SP8 low-memory profile")
        # Matching 4090 SP8 profiling: per-rank phase totals, no added CUDA barrier.
        path = Path(plan["run_dir"]) / "traces"
        path.mkdir(parents=True, exist_ok=True)
        os.environ["KVRAG_PROFILE_OUTPUT"] = str(path / "kvrag_profile")
    if plan["algorithm"] == "draftkv" and not plan.get("algorithm_plugin"):
        if plan.get("backbone") == "LingBot14B":
            raise BackboneError("BLOCKED_ALGORITHM_SEMANTICS",
                                "LingBot14B DraftKV requires its audited algorithm plugin")
        # Historical non-14B compatibility only. Published LingBot14B methods
        # use a plugin and the backbone's constructor-only loading policy.
        history = plan["history_plan"]
        os.environ.update(LINGBOT_RAG="draftmap", DRAFTKV_HIERARCHICAL_V1="1",
                          DRAFTKV_B01_AUTOTUNE_MODE="off", KVRAG_NATIVE_DIRECT_B01="1",
                          KVRAG_PINNED_FRAME_CAPACITY=str(history["capacity_frames"]),
                          KVRAG_PINNED_CPU_BANK_MAX_BYTES=str(history["limit_bytes_per_rank"]),
                          OFFLOAD_AUTOTUNE_NATIVE_DIR=plan["kernel_dir"],
                          DRAFTKV_B01_BLOCK_M="64",
                          KVRAG_B01_TRACE_DIR=str(Path(plan["run_dir"]) / "traces"))
        if plan["backbone"] == "LongLive1.3B":
            os.environ.update(LINGBOT_RAG="off",
                              RAG_DRAFTMAP_GRID_H=str(plan["request"]["height"] // 16),
                              RAG_DRAFTMAP_GRID_W=str(plan["request"]["width"] // 16))
        elif plan["backbone"] == "LingBot1.3B":
            os.environ.update(LINGBOT_SINGLE_OWNER="1", LINGBOT13_EXPERIMENTAL_STREAMING="0",
                              LINGBOT_DRAFTMAP_REFERENCE="native")


def runtime_identity(plan, trace):
    """Record admitted runtime provenance once, before model construction."""
    from draftkv.runtime.config import ConfigurationError, json_sha256
    from draftkv.runtime.state import git_sha
    import torch
    root = Path(plan["repo_root"]).resolve()
    commit = plan.get("repo_commit")
    if not commit:
        try:
            commit = git_sha(root)
        except ConfigurationError:
            commit = None
    plugin = plan.get("algorithm_plugin")
    plugin_root = Path(plugin["root"]) if plugin else None
    if plugin_root is not None and not plugin_root.is_absolute():
        plugin_root = root / plugin_root
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    low_memory = plan["backbone"] == "LingBot14B"
    trace("runtime_identity", repo_commit=commit,
          repo_commit_status="known" if commit else "unknown_non_git_snapshot",
          backbone=plan["backbone"], algorithm=plan["algorithm"],
          plugin_entry=plugin["entry"] if plugin else None,
          plugin_source_sha256=sha256(plugin_root / plugin["entry"]) if plugin else None,
          plugin_files_sha256=json_sha256(plugin["files"]) if plugin else None,
          adapter_source_sha256=json_sha256(plan["adapter_source_files"])
          if "adapter_source_files" in plan else None,
          checkpoint_sha256=plan["checkpoint_sha256"],
          checkpoint_hashes=plan["checkpoint_hashes"], sp=plan["sp"],
          distributed_topology=plan["distributed_topology"],
          cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
          local_device=local_rank,
          gpu_name=None, gpu_identity_status="deferred_until_weights_loaded",
          torch=str(torch.__version__), cuda=torch.version.cuda,
          model_loading_policy="strict_bf16_streaming" if low_memory else "author_default",
          fsdp_strategy=("FULL_SHARD" if low_memory else "author_default") if plan["sp"] > 1 else "none",
          effective_config={"request": plan["request"],
                            "effective_budget": plan["effective_budget"],
                            "author_budget_policy": plan["author_budget_policy"],
                            "plugin_config": plugin.get("config") if plugin else None})


def _load_plugin(plan, core, trace):
    plugin = plan["algorithm_plugin"]
    root = Path(plugin["root"])
    verify_files(root, plugin["files"])
    path = root / plugin["entry"]
    spec = importlib.util.spec_from_file_location("_draftkv_external_algorithm", path)
    if spec is None or spec.loader is None:
        raise BackboneError("BLOCKED_ALGORITHM_SEMANTICS", "plugin entry cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(root))
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    registry = HookRegistry()
    module.register(registry)
    hook, registration = registry.build(plan["algorithm"], core)
    if registration.source_sha256 != sha256(path):
        raise BackboneError("SOURCE_SHA_MISMATCH", "plugin registration differs from audited entry")
    if plan.get("bounded_acceptance"):
        from .acceptance import ForwardObserver, AttentionProbe
        observer = AttentionProbe if plan["bounded_acceptance"]["mode"] == "probe" else ForwardObserver
        hook = observer(hook, plan, plan["bounded_acceptance"])
    semantic_sp = 1 if plan.get("sp_mode") == "query_sequence_replicated_kv" else plan["sp"]
    session = HookSession(core.spec, hook, registration, trace, plan["request"], semantic_sp,
                          physical_sp=plan["sp"], sp_mode=plan.get("sp_mode", "single"))
    bind_model(core, session)
    return session


def _capture_session(core):
    options = core.plan.get("bounded_acceptance")
    if not options or options["mode"] not in ("capture", "capture_resident"):
        return False
    from .acceptance import NativeCaptureSession
    bind_model(core, NativeCaptureSession(core, options))
    return True


def _bounded_run(plan, trace, call):
    from .acceptance import AcceptanceStop
    try:
        call()
    except AcceptanceStop as stop:
        options = plan.get("bounded_acceptance")
        if not options or stop.record["mode"] != options["mode"]:
            raise
        trace("bounded_acceptance_complete", **stop.record)
        if plan["sp"] > 1:
            import torch.distributed as dist
            if dist.is_initialized():
                _shutdown_distributed(plan)
        return True
    if plan.get("bounded_acceptance"):
        raise BackboneError("BLOCKED_ACCEPTANCE", "model returned without reaching the bounded acceptance target")
    return False


def _clear_cuda_allocator():
    import torch
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def _shutdown_distributed(plan):
    import torch.distributed as dist
    if plan["backbone"] == "SolarWM33B":
        # Native H3 creates multiple NCCL groups. Release unused allocator slabs
        # before collective teardown, which can allocate temporary CUDA memory.
        # This is post-execution cleanup, not a memory admission estimate.
        _clear_cuda_allocator()
        dist.barrier()
    dist.destroy_process_group()


@contextmanager
def _forcing_native_teardown(entry, plan, trace):
    """Release unreachable model/cache cycles before the author's final barrier.

    LingBot's existing cleanup is gated by its historical B01 constructor flag.
    Forcing restores that flag before generation, so reuse allocator cleanup at
    the native entry's teardown seam. Only this module's distributed reference
    is wrapped; model collectives and native source bytes remain unchanged.
    """
    if (plan["backbone"] != "LingBot14B" or
            plan.get("algorithm") not in ("deep-forcing", "future-forcing")):
        yield
        return
    original = entry.dist

    class TeardownDistributed:
        def __getattr__(self, name):
            return getattr(original, name)

        def barrier(self, *args, **kwargs):
            import gc
            import torch
            gc.collect()
            _clear_cuda_allocator()
            trace("forcing_pre_nccl_cleanup",
                  allocated_bytes=torch.cuda.memory_allocated(),
                  reserved_bytes=torch.cuda.memory_reserved(),
                  constructor_controls_restored=os.environ.get("KVRAG_NATIVE_DIRECT_B01", "0") == "0")
            return original.barrier(*args, **kwargs)

    entry.dist = TeardownDistributed()
    try:
        yield
    finally:
        entry.dist = original


def _trace_native(core, trace):
    """Observe genuine model calls and QKV seams without changing their results."""
    import importlib
    target = getattr(core.model, "module", core.model)
    for layer, block in enumerate(target.blocks):
        block.self_attn._draftkv_trace_layer = layer

    def before(module, args, kwargs):
        t = kwargs.get("t")
        core.trace_start = int(kwargs.get("current_start", 0))
        trace("forward_begin", current_start=int(kwargs.get("current_start", 0)),
              clean=bool((t == 0).all()) if t is not None else None)

    def after(module, args, kwargs, result):
        t = kwargs.get("t")
        trace("forward_end", current_start=int(kwargs.get("current_start", 0)),
              clean=bool((t == 0).all()) if t is not None else None)

    target.register_forward_pre_hook(before, with_kwargs=True)
    target.register_forward_hook(after, with_kwargs=True)
    if core.plan.get("sp_mode") == "query_sequence_replicated_kv":
        # The author non-Ulysses model keeps exact full Q/K/V, RoPE, native
        # cache and camera tensors. Partition only the cached attention Q,
        # then restore its sequence before projection/residual/block output.
        import types
        from .hooks import compile_seam
        for layer, block in enumerate(target.blocks):
            def bridge(attn, q, k, v, nk, nv, rq, rk, tokens, grid, freqs, cache, _layer=layer):
                trace("qkv_route", layer=_layer, current_start=core.trace_start, frame_tokens=int(tokens),
                      query_shape=list(q.shape), key_shape=list(k.shape),
                      visible_kv_shape=list(nk.shape), sp=1,
                      physical_sp=core.plan["sp"], sp_mode=core.plan["sp_mode"],
                      raw_query_present=rq is not None, raw_key_present=rk is not None)
                return core.native_fa2(q, nk, nv)
            block.self_attn.forward = types.MethodType(
                compile_seam(block.self_attn.forward.__func__, bridge, family="lingbot"), block.self_attn)
        return
    for name in ("wan.modules.model_fast", "wan.distributed.sequence_parallel"):
        module = importlib.import_module(name)
        original = module.route_attention

        def route(attn, q, k, v, nk, nv, start, tokens, *args, _original=original, **kwargs):
            result = _original(attn, q, k, v, nk, nv, start, tokens, *args, **kwargs)
            trace("qkv_route", layer=attn._draftkv_trace_layer, current_start=int(start),
                  frame_tokens=int(tokens), query_shape=list(q.shape), key_shape=list(k.shape),
                  visible_kv_shape=list(result[0].shape), sp=int(kwargs.get("sp_size", 1)),
                  raw_query_present=kwargs.get("raw_query") is not None,
                  raw_key_present=kwargs.get("raw_key") is not None)
            return result

        module.route_attention = route


def _trace_embedded_owner(plan, trace):
    if plan["algorithm"] != "draftkv" or plan.get("algorithm_plugin"):
        return
    import importlib
    name, cls = (("lingbot13_native_owner", "LingBot13NativeOwner") if plan["backbone"] == "LingBot1.3B"
                 else ("lingbot14_native_owner", "LingBot14NativeOwner"))
    owner_class = getattr(importlib.import_module(name), cls)
    original = owner_class.__init__

    def initialize(owner, *args, **kwargs):
        original(owner, *args, **kwargs)
        geometry = plan["history_plan"]
        if (owner.budget["tokens_per_frame"] != geometry["frame_tokens"]
                or owner.budget["heads_per_rank"] != geometry["heads_per_rank"]):
            raise BackboneError("BLOCKED_HISTORY_CAPACITY", "actual native geometry differs from planned history allocation")
        trace("history_capacity_bound", **geometry)
        owner.observer = lambda event, **fields: trace("owner." + event, **fields)

    owner_class.__init__ = initialize


def validate_host_memory(plan):
    """Capacity is runtime-observed only; no pre-allocation host-memory gate."""
    return


def validate_camera(plan):
    import numpy as np
    request = plan["request"]
    poses = np.load(Path(request["camera_dir"]) / "poses.npy", allow_pickle=False)
    intrinsics = np.load(Path(request["camera_dir"]) / "intrinsics.npy", allow_pickle=False)
    if (poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) != request["native_frames"]
            or not np.isfinite(poses).all() or not np.allclose(poses[:, 3, :], (0, 0, 0, 1))
            or intrinsics.ndim != 2 or intrinsics.shape[1] != 4 or len(intrinsics) not in (1, len(poses))
            or not np.isfinite(intrinsics).all() or np.any(intrinsics[:, :2] <= 0)):
        raise BackboneError("BLOCKED_INPUT_PROTOCOL", "native pose/intrinsics geometry mismatch")
    if not np.allclose(poses[:, :3, :3] @ poses[:, :3, :3].transpose(0, 2, 1), np.eye(3), atol=1e-3):
        raise BackboneError("BLOCKED_INPUT_PROTOCOL", "camera rotations must be orthonormal")
    if request["native_frames"] > request["frames"]:
        if not np.allclose(poses[request["frames"]:], poses[request["frames"] - 1]):
            raise BackboneError("BLOCKED_INPUT_PROTOCOL", "padded camera tail differs from frozen hold-last policy")
        if len(intrinsics) > 1 and not np.allclose(intrinsics[request["frames"]:], intrinsics[request["frames"] - 1]):
            raise BackboneError("BLOCKED_INPUT_PROTOCOL", "padded intrinsics tail differs from frozen hold-last policy")


def validate_kernel_device(plan):
    if not plan.get("kernel_abi"):
        return
    import torch
    abi = plan["kernel_abi"]
    capability = torch.cuda.get_device_capability(int(os.environ.get("LOCAL_RANK", "0")))
    if str(torch.__version__) != abi["torch"] or "sm" + "".join(map(str, capability)) not in abi["sm"]:
        raise BackboneError("BLOCKED_DEPENDENCY", "B01 binary does not match worker Torch or device SM; rebuild required")


def _run_longlive(plan, spec, trace):
    import torch
    import torch.distributed as dist
    from .sequence_parallel import native_attention, require_torchrun
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = require_torchrun(plan["sp"]) if plan["sp"] > 1 else 0
    torch.cuda.set_device(local_rank)
    core = LongliveCore(spec, plan, trace).load(device=f"cuda:{local_rank}")
    if plan["sp_mode"] == "query_sequence_replicated_kv":
        unsharded = core.native_fa2
        core.native_fa2 = native_attention(unsharded, plan["sp"])
        core.native_fa2._draftkv_unsharded = unsharded
    if _capture_session(core):
        pass
    elif plan["algorithm_plugin"]:
        _load_plugin(plan, core, trace)
    elif plan["algorithm"] == "native":
        import types
        from .hooks import compile_seam
        target = core.model.get_base_model() if hasattr(core.model, "get_base_model") else core.model

        def before(module, args, kwargs):
            t = kwargs.get("t")
            trace("forward_begin", current_start=int(kwargs.get("current_start", 0)),
                  clean=bool((t == 0).all()))
            core.trace_start = int(kwargs.get("current_start", 0))

        def after(module, args, kwargs, output):
            trace("forward_end", current_start=int(kwargs.get("current_start", 0)),
                  output_shape=list(output.shape))

        target.register_forward_pre_hook(before, with_kwargs=True)
        target.register_forward_hook(after, with_kwargs=True)
        for layer, block in enumerate(target.blocks):
            def tap(attn, q, k, v, nk, nv, rq, rk, tokens, grid, freqs, cache, _layer=layer):
                trace("qkv_route", layer=_layer, current_start=core.trace_start,
                      query_shape=list(q.shape), current_key_shape=list(k.shape),
                      visible_kv_shape=list(nk.shape), frame_tokens=int(tokens),
                      consumer="native_fa2", raw_query_present=rq is not None, raw_key_present=rk is not None)
                return core.native_fa2(q, nk, nv)

            block.self_attn.forward = types.MethodType(
                compile_seam(block.self_attn.forward.__func__, tap, family="longlive"), block.self_attn)
    request = plan["request"]
    trace("launch", algorithm=plan["algorithm"], backbone=plan["backbone"], sp=plan["sp"],
          sp_mode=plan["sp_mode"], input_mode=request["input_mode"], input_binding=request["image_conditioning"],
          source_files=plan["source_files"], dataset_manifest_sha256=request["dataset_manifest_sha256"])
    result = run_profiled(plan, lambda: core.generate(request))
    # All ranks execute the exact same model/cache lifecycle.  Only rank0 owns
    # the release artifact; a final barrier prevents another rank from exiting
    # while rank0 is still encoding it.
    if rank == 0:
        from wan.utils.utils import cache_video
        cache_video(tensor=result[None], save_file=plan["output"], fps=request["fps"],
                    nrow=1, normalize=True, value_range=(-1, 1))
    if plan["sp"] > 1:
        dist.barrier()


def _report_worker_failure(plan, error):
    """Persist the original exception without CUDA, collectives or trace joins."""
    import traceback
    formatted = ''.join(traceback.format_exception(type(error), error, error.__traceback__))
    rank = os.environ.get('RANK', '0')
    artifact = None
    reporting_error = None
    try:
        if plan.get('run_dir'):
            folder = Path(plan['run_dir']) / 'traces'
            folder.mkdir(parents=True, exist_ok=True)
            # RANK is supplied by torchrun; sanitize before using it as a filename.
            suffix = str(int(rank))
            artifact = folder / f'worker_failure_rank{suffix}.json'
            with artifact.open('w', encoding='utf-8') as output:
                json.dump(dict(event='worker_failure', rank=rank, pid=os.getpid(),
                    algorithm=plan.get('algorithm'), backbone=plan.get('backbone'),
                    error_type=type(error).__name__, message=str(error), traceback=formatted),
                    output, ensure_ascii=True, indent=2)
                output.flush()
                os.fsync(output.fileno())
    except BaseException as failure:
        # An unwritable report must never replace the original model exception.
        reporting_error = f'{type(failure).__name__}: {failure}'
    try:
        sys.stderr.write(f'WORKER_FAILURE rank={rank} artifact={artifact}\n' + formatted)
        if reporting_error:
            sys.stderr.write(f'WORKER_FAILURE_REPORT_ERROR {reporting_error}\n')
        sys.stderr.flush()
    except BaseException:
        pass


def main(argv=None):
    """Imported callers receive the original error; only the CLI exits hard."""
    plan = {}
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--plan", required=True)
        options = parser.parse_args(argv)
        plan = read_json(options.plan)
        return _run_plan(plan)
    except BaseException as error:
        if not (isinstance(error, SystemExit) and error.code in (None, 0)):
            try:
                _report_worker_failure(plan, error)
            except BaseException:
                pass  # Reporting must not replace the original failure.
        raise


def _run_plan(plan):
    from draftkv.algorithms.draftkv.experiment import experiment_scope
    with experiment_scope((plan.get('algorithm_plugin') or {}).get('config', {}), plan['backbone']):
        return _run_plan_bound(plan)


def _run_plan_bound(plan):
    if ((plan.get("benchmark") == "bench1" and plan.get("backbone") == "LingBot14B")
            or plan.get("request", {}).get("input_mode") == "bench1_ti"):
        raise BackboneError("BLOCKED_INPUT_PROTOCOL", "Bench1 LingBot14B and its T+I benchmark have been retired")
    from draftkv.runtime.config import asset_sha256, json_sha256
    hashes = {"base": asset_sha256(Path(plan["checkpoint"]))}
    if plan.get("lora_checkpoint"):
        hashes["lora"] = asset_sha256(Path(plan["lora_checkpoint"]))
    if plan["backbone"] == "SolarWM33B":
        hashes.update(native_base=asset_sha256(Path(plan["assets_dir"])), silence=sha256(plan["silence_path"]))
    if hashes != plan.get("checkpoint_hashes") or json_sha256(hashes) != plan.get("checkpoint_sha256"):
        raise BackboneError("BLOCKED_CHECKPOINT", "base/LoRA bytes changed after planning")
    verify_files(plan["source_root"], plan["source_files"])
    if plan.get("adapter_source_files") is not None:
        if not plan["adapter_source_files"]:
            raise BackboneError("BLOCKED_SOURCE", "empty release adapter source lock")
        verify_files(plan["repo_root"], plan["adapter_source_files"])
    if plan.get("longlive_provider"):
        from .longlive_provider import verify_provider
        verify_provider(plan["longlive_provider"])
    request = plan["request"]
    if sha256(request["dataset_manifest"]) != request["dataset_manifest_sha256"]:
        raise BackboneError("BLOCKED_INPUT_PROTOCOL", "frozen dataset changed after planning")
    assets = {"image": Path(request["image"])} if request.get("image") else {}
    if request.get("camera_dir"):
        assets.update(poses=Path(request["camera_dir"]) / "poses.npy", intrinsics=Path(request["camera_dir"]) / "intrinsics.npy")
    for name, path in assets.items():
        if sha256(path) != request["asset_sha256"][name]:
            raise BackboneError("BLOCKED_INPUT_PROTOCOL", "prepared conditioning changed after planning")
    if request.get("input_conversion"):
        raise BackboneError("BLOCKED_INPUT_PROTOCOL", "retired generated-image conversion is forbidden")
    if plan["kernel_dir"]:
        verify_files(plan["kernel_dir"], plan["kernel_sha256"])
    if plan["backbone"] == "SolarWM33B":
        from .planning import probe_environment
        current = probe_environment(plan["python"], "solarwm")
        if current != plan["environment_metadata"]:
            raise BackboneError("SOURCE_ABI_DRIFT", "H3 runtime changed after planning")
    configure_environment(plan)
    validate_host_memory(plan)
    validate_kernel_device(plan)
    folder = Path(plan["run_dir"]) / "traces"
    folder.mkdir(exist_ok=True)
    rank = int(os.environ.get("RANK", "0"))
    trace = Trace(folder, rank)
    runtime_identity(plan, trace)
    if plan['algorithm'] == 'draftkv' and 'host_pages_policy' in plan:
        from draftkv.streaming.binding import NativeBinding
        if plan['algorithm_plugin']['config'].get('gpu_prepack',False):
            actual_pages=dict(host_pages_backend='cudaHostRegister_THP',host_pages_policy='systems',
                              host_pages_role='clean_gpu_prepared_archive')
        else:
            actual_pages = NativeBinding.read(plan['algorithm_plugin']['config'], plan['repo_root']).host_pages_metadata
        if any(plan.get(key) != value for key, value in actual_pages.items()):
            raise BackboneError('SOURCE_ABI_DRIFT', 'host-page policy changed after planning')
        trace('host_pages_policy', **actual_pages)
    if plan["backbone"] == "LingBot14B" and plan["algorithm"] == "future-forcing":
        trace("cuda_allocator_config", value=os.environ.get("PYTORCH_ALLOC_CONF",
              os.environ.get("PYTORCH_CUDA_ALLOC_CONF")), scope="worker_resource_policy")
    spec = MODELS[plan["backbone"]]
    from .sequence_parallel import rank_mapping
    trace("rank_mapping", **{key: value for key, value in rank_mapping(plan["sp"]).items() if key != "rank"})
    from .acceptance import validate_acceptance
    validate_acceptance(plan, spec)
    activate_sources(plan)
    from .latency_build import prepare_profile_builds
    prepare_profile_builds(plan, trace)
    if spec.family == "solarwm":
        from .solarwm import SolarwmCore
        core = SolarwmCore(spec, plan, trace).load()
        if _capture_session(core):
            pass
        elif plan.get("algorithm_plugin"):
            _load_plugin(plan, core, trace)
        trace("launch", algorithm=plan["algorithm"], backbone=plan["backbone"], sp=plan["sp"],
              input_mode="orbia", native_entry="h3_sgf_rollout", effective_budget=plan["effective_budget"])
        if _bounded_run(plan, trace, lambda: run_profiled(plan, core.generate)):
            trace.handle.close()
            return
        if plan["sp"] > 1:
            _shutdown_distributed(plan)
        trace("worker_complete", output=plan["output"], release_e2e=False)
        trace.handle.close()
        return
    if spec.family == "longlive":
        if _bounded_run(plan, trace, lambda: _run_longlive(plan, spec, trace)):
            trace.handle.close()
            return
        trace("worker_complete", output=plan["output"], release_e2e=False)
        trace.handle.close()
        if plan["sp"] > 1:
            import torch.distributed as dist
            if dist.is_initialized():
                dist.destroy_process_group()
        return
    # Import exactly the pinned author's generate module; it owns distributed
    # initialization, seed broadcast, image decoding, save_video and teardown.
    entry_path = Path(plan["native_root"]) / "generate.py"
    module_spec = importlib.util.spec_from_file_location("_draftkv_author_generate", entry_path)
    entry = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(entry)
    if plan["backbone"] != "SolarWM33B":
        validate_camera(plan)
    _trace_embedded_owner(plan, trace)

    def run_causal(args, cfg, img, device, rank, mode="causal_fast"):
        core = LingbotCore(spec, plan, trace).load(cfg, device_id=device, rank=rank, sp=plan["sp"])
        if _capture_session(core):
            pass
        elif plan["algorithm_plugin"]:
            _load_plugin(plan, core, trace)
        else:
            _trace_native(core, trace)
        result = run_profiled(plan, lambda: core.generate(plan["request"], img))
        if result is not None:
            if plan['algorithm'] == 'draftkv':
                validate_finite_video(result)
                trace('decoded_tensor_finite', finite=True, frames=int(result.shape[1]))
            trace("decoded_tensor", layout="C,F,H,W", shape=list(result.shape))
            from draftkv.algorithms.forcing.geometry import output_request
            result = validate_native_video(result, output_request(plan))
        return result

    entry.run_causal = run_causal
    if plan["sp_mode"] == "query_sequence_replicated_kv":
        from .lingbot_distributed import query_sequence_generate
        entry.generate = query_sequence_generate(entry.generate, heads=spec.heads, sp=plan["sp"])
    req = plan["request"]
    sys.argv = [str(entry_path), "--task", spec.task, "--infer_mode", "causal_fast",
                "--size", f"{req['height']}*{req['width']}", "--ckpt_dir", plan["checkpoint"],
                "--assets_dir", plan["assets_dir"], "--image", req["image"],
                "--action_path", req["camera_dir"], "--prompt", req["prompts"][0],
                "--frame_num", str(req["native_frames"]), "--chunk_size", str(spec.chunk),
                "--base_seed", str(req["seed"]), "--ulysses_size", str(plan["sp"]),
                "--local_attn_size", str(spec.window), "--sink_size", str(spec.sink),
                "--offload_model", str(req["offload_model"]), "--save_file", plan["output"],
                "--save_dir", plan["run_dir"]]
    if plan["sp"] > 1:
        sys.argv += ["--dit_fsdp", "--t5_fsdp"]
    trace("launch", algorithm=plan["algorithm"], backbone=plan["backbone"], sp=plan["sp"],
          input_mode=req["input_mode"], dataset_manifest_sha256=req["dataset_manifest_sha256"],
          input_conversion=req.get("input_conversion"), native_input_mode="image_camera",
          source_files=plan["source_files"], consumer="original_b01_streaming_fa2" if plan["algorithm"] == "draftkv" else "native_fa2")
    with _forcing_native_teardown(entry, plan, trace):
        bounded = _bounded_run(plan, trace, lambda: entry.generate(entry._parse_args()))
    # Only a successful normal/bounded return may enter collective teardown.
    # A failing peer can strand another rank in a native barrier: destroying
    # NCCL here would hide the original exception before elastic can observe it.
    if plan["sp"] > 1:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()
    if bounded:
        trace.handle.close()
        return
    trace("worker_complete", output=plan["output"], release_e2e=False)
    trace.handle.close()


def validate_finite_video(result):
    """Bounded per-frame observation before encoding can hide NaN/Inf pixels."""
    import torch
    for frame in result.unbind(1):
        if not bool(torch.isfinite(frame).all()):
            raise BackboneError('BLOCKED_OUTPUT_SPEC', 'decoded DraftKV video contains NaN/Inf')


def validate_native_video(result, request):
    """Trim only the explicitly bound native C4 tail after exact decode checks."""
    if result.ndim != 4:
        raise BackboneError("BLOCKED_OUTPUT_SPEC", "native decoded video must have C,F,H,W layout")
    expected = request["frames"]
    padded = request.get("padding_policy") == "hold_last_camera_trim_video"
    decoded = request["native_frames"] if padded else expected
    if result.shape[1] != decoded:
        raise BackboneError("BLOCKED_OUTPUT_SPEC", "native decoded frame count differs from frozen request")
    if ((result.shape[3], result.shape[2]) != (request["output_width"], request["output_height"])
            or result.shape[0] != 3 or (padded and decoded < expected)):
        raise BackboneError("BLOCKED_OUTPUT_SPEC", "native decoded dimensions/padding differ from frozen request")
    return result[:, :expected] if padded else result


def _cli_main(argv=None):
    """Let elastic record failures, then bypass Python/NCCL destructors."""
    with isolated_cli_worker():
        try:
            from torch.distributed.elastic.multiprocessing.errors import record
            return record(main)(argv)
        except BaseException as error:
            if isinstance(error, SystemExit) and error.code in (None, 0):
                return
            # Also expose CLI/elastic-reporting errors. main() already persisted the
            # original model exception synchronously, before any unsafe cleanup.
            import traceback
            try:
                traceback.print_exception(type(error), error, error.__traceback__, file=sys.stderr)
            except BaseException:
                pass
            for stream in (sys.stdout, sys.stderr):
                try:
                    stream.flush()
                except BaseException:
                    pass
            status = error.code if isinstance(error, SystemExit) else 1
            os._exit(status if type(status) is int and 0 < status < 256 else 1)


if __name__ == "__main__":
    _cli_main()
