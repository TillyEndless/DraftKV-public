"""One core loader/forward per family, using external author source and weights.

These classes run in isolated workers (``wan`` is an upstream top-level package).
No model or algorithm source is copied into the release repository.
"""

import ast
import importlib
import importlib.abc
import importlib.util
import inspect
import os
from contextlib import contextmanager
import sys
import textwrap
from pathlib import Path

from .contracts import BackboneError
from .failure_scope import retain_fatal_failure


class _GuardLoader(importlib.abc.Loader):
    def __init__(self, path):
        self.path = path

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        tree = ast.parse(self.path.read_bytes(), filename=str(self.path))
        guards = [node for node in tree.body if isinstance(node, ast.If)
                  and "_unified_method_env.environ.get('LINGBOT_RAG')" in ast.unparse(node.test)
                  and "refusing native Local Attention fallback" in ast.unparse(node)]
        if len(guards) != 1:
            raise BackboneError("SOURCE_ABI_DRIFT", "LingBot14 method-only import guard changed")
        # This snapshot has an import-time DraftMap-only gate. The adapter
        # already verifies the selected algorithm before importing; removing
        # this single gate permits Native and registered hooks on the SAME
        # source. No mathematical expression or runtime branch is changed.
        tree.body.remove(guards[0])
        module.__file__ = str(self.path)
        exec(compile(ast.fix_missing_locations(tree), str(self.path), "exec"), module.__dict__)


class _GuardFinder(importlib.abc.MetaPathFinder):
    def __init__(self, path):
        self.path = path

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "wan.modules.model_fast":
            return importlib.util.spec_from_file_location(fullname, self.path, loader=_GuardLoader(self.path))
        return None


def activate_sources(plan):
    """Only the worker may activate top-level author packages; mixing is an error."""
    if "wan" in sys.modules or "solarwm" in sys.modules:
        raise BackboneError("MODEL_NAMESPACE_COLLISION", "use a fresh worker for each model/run")
    roots = [str(Path(path).resolve()) for path in plan["import_roots"]]
    sys.path[:0] = roots
    if plan.get("backbone") == "SolarWM33B":
        spec = importlib.util.find_spec("solarwm")
        if spec is None or not spec.origin or not Path(spec.origin).resolve().is_relative_to(Path(plan["source_root"]).resolve()):
            raise BackboneError("SOURCE_ABI_DRIFT", "SolarWM must import from the locked native source")
    native = Path(plan["native_root"])
    fast = native / "wan/modules/model_fast.py"
    if fast.is_file() and "refusing native Local Attention fallback" in fast.read_text():
        if plan["algorithm"] != "draftkv" or plan.get("algorithm_plugin"):
            sys.meta_path.insert(0, _GuardFinder(fast))


def _native_fa2(version=2):
    module = importlib.import_module("wan.modules.attention")
    if version not in (2,3):
        raise BackboneError("BLOCKED_DEPENDENCY", "native attention version must be 2 or 3")
    if version==2 and not getattr(module, "FLASH_ATTN_2_AVAILABLE", False):
        raise BackboneError("BLOCKED_DEPENDENCY", "the author's FlashAttention 2 extension is required")
    if version==3:
        try:importlib.import_module('flash_attn_interface')
        except ImportError as error:
            raise BackboneError("BLOCKED_DEPENDENCY", "FlashAttention 3 was explicitly requested") from error
    module.FLASH_ATTN_3_AVAILABLE = version==3
    # DraftKV resolves the architecture policy before this call. Other
    # algorithms retain their explicit plan; no missing-backend fallback.
    original = getattr(module,'_draftkv_original_flash_attention',module.flash_attention)
    module._draftkv_original_flash_attention=original

    def fa2(q, k, v, *args, **kwargs):
        kwargs["version"] = version
        return original(q, k, v, *args, **kwargs)

    module.flash_attention = fa2
    return module.attention


def bootstrap_algorithm(model, plan, trace):
    """Run an original algorithm's weight bootstrap before any model sharding."""
    if plan["algorithm"] != "infinigen":
        return
    module = importlib.import_module("draftkv.algorithms.infinigen.entry")
    callback = getattr(module, "bootstrap", None)
    if not callable(callback):
        raise BackboneError("BLOCKED_ALGORITHM_SEMANTICS", "InfiniGen requires its original pre-FSDP partial-weight bootstrap")
    callback(model, plan, trace)


def pre_shard_configure(original, callback):
    """Guard the native sharding seam and delegate all configuration unchanged."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    shard_calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                   and isinstance(node.func, ast.Name) and node.func.id == "shard_fn"
                   and len(node.args) == 1 and isinstance(node.args[0], ast.Name)
                   and node.args[0].id == "model"]
    if len(shard_calls) != 1 or not {"model", "dit_fsdp", "shard_fn"} <= set(inspect.signature(original).parameters):
        raise BackboneError("SOURCE_ABI_DRIFT", "native pre-FSDP _configure_model seam changed")
    def configure(instance, model, *args, **kwargs):
        callback(model)
        return original(instance, model, *args, **kwargs)
    return configure


class LingbotCore:
    """Shared WanI2VCausal loader for both 1.3B and 14B configurations."""

    def __init__(self, spec, plan, trace):
        self.spec, self.plan, self.trace = spec, plan, trace

    def load(self, config, *, device_id, rank, sp):
        from .sequence_parallel import model_topology, native_attention
        topology = model_topology(self.spec.family, self.spec.heads, sp)
        if (self.plan.get("sp_mode", topology["sp_mode"]) != topology["sp_mode"]
                or self.plan.get("attention_sp", topology["attention_sp"]) != topology["attention_sp"]):
            raise BackboneError("UNSUPPORTED", "planned attention topology differs from model heads and physical world")
        if (config.num_heads != self.spec.heads or config.num_layers != self.spec.layers
                or config.dim != self.spec.heads * self.spec.head_dim):
            raise BackboneError("SOURCE_ABI_DRIFT", "native configuration head/layer/dimension differs from selected model")
        from draftkv.streaming.device_profiles import native_version_for_plan
        version=native_version_for_plan(self.plan,f"cuda:{device_id}")
        self.trace('native_attention_backend',requested=self.plan.get('native_attention_version',2),
                   resolved=version)
        self.native_fa2 = _native_fa2(version)
        if topology["sp_mode"] == "query_sequence_replicated_kv":
            self.native_fa2 = native_attention(self.native_fa2, sp)
        import wan
        source = Path(wan.__file__).resolve()
        if not source.is_relative_to(Path(self.plan["native_root"]).resolve()):
            raise BackboneError("SOURCE_ABI_DRIFT", f"wrong author module: {source}")
        constructor = wan.WanI2VCausal
        original = constructor._configure_model
        if self.plan["algorithm"] == "infinigen":
            constructor._configure_model = pre_shard_configure(
                original, lambda model: bootstrap_algorithm(model, self.plan, self.trace))
        try:
            from .budgets import bound_budget
            camera_only = (self.plan['algorithm'] == 'draftkv' and
                           self.plan.get('algorithm_plugin', {}).get('config', {}).get('retrieval_ablation') == 'camera-only')
            allocation = bound_budget(self, 'draftkv') if camera_only else None
            with lingbot14_low_memory_loading(self.plan, self.trace):
                self.pipeline = constructor(
                    config=config, checkpoint_dir=self.plan["checkpoint"], assets_dir=self.plan["assets_dir"],
                    device_id=device_id, rank=rank, use_sp=topology["native_use_sp"],
                    t5_fsdp=sp > 1, dit_fsdp=sp > 1, t5_cpu=False, convert_model_dtype=False,
                    local_attn_size=allocation.capacity if allocation else self.spec.window,
                    sink_size=allocation.sink if allocation else self.spec.sink, infer_mode="causal_fast")
        finally:
            constructor._configure_model = original
        self.model = self.pipeline.model
        if self.pipeline.sp_size != topology["attention_sp"]:
            raise BackboneError("SOURCE_ABI_DRIFT", "native cache/Ulysses world differs from selected attention topology")
        self.trace("distributed_topology_bound", **topology,
                   parameter_sharding="author_fsdp" if sp > 1 else "none",
                   fsdp_strategy=("FULL_SHARD" if self.plan.get("backbone") == "LingBot14B"
                                  else "author_default") if sp > 1 else "none",
                   text_encoder="author_fsdp_all_ranks" if sp > 1 else "native",
                   vae_decode="native_rank0", positions="native_full_sequence_rope_and_camera")
        # Query device properties only after the author has selected this rank's
        # CUDA device and constructed the model; telemetry must not lazy-init a
        # default cuda:0 context on every rank before that selection.
        import torch
        self.trace("weights_loaded", checkpoint=self.plan["checkpoint"], task=self.spec.task,
                   layers=self.spec.layers, heads=self.spec.heads, rank=rank,
                   cuda_device=int(device_id),
                   gpu_name=torch.cuda.get_device_name(device_id) if torch.cuda.is_initialized() else None)
        return self

    def forward(self, **native_kwargs):
        """Raw native Forward: preserve caches, timestep, RoPE and camera kwargs."""
        return self.model(**native_kwargs)

    def generate(self, request, image):
        from .lingbot_vae_memory import install_lingbot14_vae_cpu_staging
        install_lingbot14_vae_cpu_staging(self)
        # Native source owns generation; opt-in timing only observes the shared
        # VAE-encode and final clean-forward seams, across all KV algorithms.
        from contextlib import nullcontext
        if self.plan.get("gpu_profile", False):
            from .lingbot_diffusion_timing import lingbot_diffusion_timing
            timing = lingbot_diffusion_timing(self, request)
        else:
            timing = nullcontext()
        with (lingbot_vae_history_release(self, request),
              infinigen_vae_profile_offload(self), timing):
            return self.pipeline.generate(
                request["prompts"][0], image, action_path=request["camera_dir"],
                chunk_size=self.spec.chunk, max_area=request["height"] * request["width"],
                frame_num=request["native_frames"], seed=request["seed"],
                # Do not reimplement the author's scheduler or override its timestep list.
                shift=request["sample_shift"], offload_model=request["offload_model"])


@contextmanager
def lingbot_vae_history_release(core, request):
    """Release completed-video histories at the native outer decode boundary."""
    if (core.plan.get("backbone") != "LingBot14B"
            or core.plan.get("algorithm") not in ("shadowkv", "worldtrace-landmark")):
        yield
        return
    callback = getattr(core, "release_after_rollout", None)
    expected = request.get("latent_frames")
    if not callable(callback) or type(expected) is not int or expected < 1:
        raise BackboneError("BLOCKED_PROVIDER_ABI", "VAE decode needs a bound completed-rollout release")
    vae = core.pipeline.vae
    original = vae.decode
    had_instance_decode = "decode" in vars(vae)
    instance_decode = vars(vae).get("decode")

    def decode(*args, **kwargs):
        callback(expected_frames=expected)
        return original(*args, **kwargs)

    vae.decode = decode
    try:
        yield
    finally:
        if had_instance_decode:
            vae.decode = instance_decode
        else:
            del vae.decode


@contextmanager
def infinigen_vae_profile_offload(core):
    """End frozen forecast GPU residency at the native rollout/decode boundary.

    The native generator calls this outer VAE decode only after every model
    forward and clean commit. Its model.cpu() cannot see the frozen modules
    inside InfiniGen's ordinary Python list. Keep the same profiles and hooks,
    and restore their original devices before a subsequent generation.
    """
    if (core.plan.get("backbone") != "LingBot14B"
            or core.plan.get("algorithm") != "infinigen"):
        yield
        return
    from draftkv.algorithms.infinigen.forecast import (
        offload_forecast_profiles, restore_forecast_profiles)
    restored = restore_forecast_profiles(core.model)
    core.trace("infinigen_frozen_profile_storage", phase="before_generation",
               movement=restored)
    vae = core.pipeline.vae
    original = vae.decode
    had_instance_decode = "decode" in vars(vae)
    instance_decode = vars(vae).get("decode")

    def decode(*args, **kwargs):
        moved = offload_forecast_profiles(core.model)
        core.trace("infinigen_frozen_profile_storage",
                   phase="after_rollout_before_vae_decode", movement=moved)
        return original(*args, **kwargs)

    vae.decode = decode
    try:
        yield
    finally:
        # Python binding restoration only: fatal CUDA errors must not cause
        # an extra device move or fence during exception unwinding.
        if had_instance_decode:
            vae.decode = instance_decode
        else:
            del vae.decode


@contextmanager
def lingbot14_low_memory_loading(plan, trace):
    """Reuse the locked author's strict BF16 streaming loader and FULL_SHARD.

    Its existing resource controls share a historical B01 prefix. Scope them
    solely to construction, then restore them before any model Forward. The
    backbone, not the selected attention/KV algorithm, determines this resource
    policy. In particular, this must never activate the author's DraftMap
    generation branch for Native or an algorithm plugin.
    """
    if plan.get("backbone") != "LingBot14B":
        yield
        return
    keys = {"KVRAG_NATIVE_DIRECT_B01": "1", "LINGBOT14_LOW_MEMORY_STREAM": "1",
            "LINGBOT14_FSDP_FULL_SHARD": "1"}
    previous = {key: os.environ.get(key) for key in keys}
    # Preserve opt-in/WT observer event identities while keeping one backbone
    # policy. A profile changes telemetry, never eligibility for this loader.
    from .planning import DRAFTKV_SP8_BF16_PROFILE
    profiled_draftkv = (plan.get("algorithm") == "draftkv" and plan.get("sp") == 8
                       and plan.get("draftkv_loading_profile") == DRAFTKV_SP8_BF16_PROFILE)
    event = ("draftkv_sp8_loading_resources" if profiled_draftkv
             else "worldtrace_loading_resources" if plan.get("algorithm") == "worldtrace-landmark"
             else "lingbot14_loading_resources")
    fatal = False
    try:
        os.environ.update(keys)
        trace(event, backbone=plan["backbone"], algorithm=plan.get("algorithm"),
              low_memory=True, checkpoint_load="author_strict_tensorwise_f32_to_bf16",
              fsdp="FULL_SHARD", scope="constructor_only", attention="algorithm_unchanged")
        yield
    except BaseException as error:
        fatal = retain_fatal_failure(error)
        raise
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        if not fatal:
            trace(event + "_restored", constructor_controls_restored=True)


# Compatibility for existing capture/WT observers; there is one implementation
# and the ordinary model constructor calls it for every LingBot14B method.
native_capture_loading = lingbot14_low_memory_loading


class LongliveCore:
    """Source-locked 1.3B base+LoRA loader, shared by Native and injected hooks.

    Bench1 uses the actual frozen text-only manifest and native T2V Forward.
    The historical 240-latent result does not freeze the new dataset output spec.
    """

    def __init__(self, spec, plan, trace):
        self.spec, self.plan, self.trace = spec, plan, trace

    def load(self, *, device="cuda:0"):
        import torch
        from draftkv.streaming.device_profiles import native_version_for_plan
        version=native_version_for_plan(self.plan,device)
        self.trace('native_attention_backend',requested=self.plan.get('native_attention_version',2),
                   resolved=version)
        self.native_fa2 = _native_fa2(version)
        from wan.modules.causal_model import CausalWanModel
        # Exactly the architecture and W12/S3 of the audited checkpoint Forward.
        self.model = CausalWanModel(model_type="t2v", dim=1536, ffn_dim=8960,
                                   freq_dim=256, num_heads=12, num_layers=30,
                                   local_attn_size=12, sink_size=3)
        state = torch.load(self.plan["checkpoint"], map_location="cpu", weights_only=True, mmap=True)["generator"]
        if len(state) != 825 or any(not name.startswith("model.") for name in state):
            raise BackboneError("CHECKPOINT_ABI_DRIFT", "expected audited 825-key model.* generator")
        self.model.load_state_dict({name[6:]:tensor for name, tensor in state.items()}, strict=True)
        self.model.eval().to(device=device, dtype=torch.bfloat16)
        lora = self.plan.get("lora_checkpoint")
        if lora:
            import peft
            from utils.lora_utils import configure_lora_for_model
            if str(peft.__version__).startswith("0.21."):
                # Reuse the audited native float-Linear workaround from the
                # historical entry. Never apply it to a quantized target.
                if any(type(module.weight).__module__.startswith("torchao") for module in self.model.modules()
                       if getattr(module, "weight", None) is not None):
                    raise BackboneError("BLOCKED_DEPENDENCY", "quantized LongLive targets need an original supported PEFT/TorchAO binding")
                from peft.tuners.lora import torchao as peft_torchao
                peft_torchao.is_torchao_available = lambda: False
            self.model = configure_lora_for_model(
                self.model, model_name="generator", is_main_process=False,
                lora_config=dict(type="lora", rank=256, alpha=256, dropout=0.0, verbose=False))
            state = torch.load(lora, map_location="cpu", weights_only=True)
            result = peft.set_peft_model_state_dict(self.model, state.get("generator_lora", state))
            if (getattr(result, "unexpected_keys", ())
                    or any("lora_" in key for key in getattr(result, "missing_keys", ()))):
                raise BackboneError("CHECKPOINT_ABI_DRIFT", "unexpected LoRA keys")
            self.model.eval()
        bootstrap_algorithm(self.model, self.plan, self.trace)
        self.trace("weights_loaded", checkpoint=self.plan["checkpoint"], lora=lora,
                   layers=30, heads=12)
        return self

    def forward(self, **native_kwargs):
        return self.model(**native_kwargs)

    def generate(self, request):
        from .longlive_generation import generate
        return generate(self, request)


def relocated_asset_init(cls, assets):
    """Run the author's constructor with only its asset string literals relocated.

    The isolated worker does not chdir or create links inside the author's tree.
    Source locks and the exact literal prefix make drift fail closed.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(cls.__init__)))
    prefix = "wan_models/Wan2.1-T2V-1.3B/"
    changed = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.startswith(prefix):
            node.value = str(Path(assets) / node.value[len(prefix):])
            changed += 1
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "super" and not node.args:
            # The function is recompiled outside its original class body.
            node.args = [ast.Name(id="_AuthorClass", ctx=ast.Load()), ast.Name(id="self", ctx=ast.Load())]
    if not changed:
        raise BackboneError("SOURCE_ABI_DRIFT", "author asset constructor no longer has the locked path literals")
    namespace = dict(cls.__init__.__globals__, _AuthorClass=cls)
    exec(compile(ast.fix_missing_locations(tree), inspect.getsourcefile(cls) or "<author-assets>", "exec"), namespace)
    instance = cls.__new__(cls)
    namespace["__init__"](instance)
    return instance
