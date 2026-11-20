"""Portable binding of the external, frozen LongLive V1/B01 implementation.

Only source locations and the input-owned prompt-switch guard are rebound.
Retrieval math, native cached Forward, ProducerPlan, consumer and all-layer
CleanSession execute from the locked historical sources.
"""

import ast
from contextvars import ContextVar
import importlib.util
from pathlib import Path
import sys
import types

from .contracts import BackboneError, external_path
from .planning import read_json, verify_files


BINDING = "longlive_hierarchical_v1_original_b01"
REQUIRED = {
    "handler": ("longlive1_draftmap_candidate.py", "longlive1_native_hook.py", "hierarchical_v1_selector.py"),
    "support": ("registry.py", "ragged_frame_index.py", "lingbot_native_candidate/unified_core/draftmap.py"),
    "math": ("compat/__init__.py", "compat/native_math.py", "canonical/__init__.py", "canonical/core.py",
             "reference/spatial2d_draft_importance.py"),
    "author": ("draftmap_perlayer.py",),
}
LEGACY_SWITCHES = (40, 80, 120, 160, 200)


def resolve_provider(config, root):
    if (not isinstance(config, dict) or not config.get("source_lock")
            or not isinstance(config.get("roots"), dict) or not set(REQUIRED).issubset(config["roots"])):
        raise BackboneError("BLOCKED_DEPENDENCY", "LongLive V1 requires longlive_provider source roots and lock")
    lock = read_json(external_path(config["source_lock"], root))
    if lock.get("binding") != BINDING:
        raise BackboneError("BLOCKED_ALGORITHM_SEMANTICS", "original LongLive Hierarchical V1/B01 provider required")
    roots = {}
    for role, required in REQUIRED.items():
        source = external_path(config["roots"][role], root)
        files = lock["roots"][role]["files"]
        if not set(required).issubset(files):
            raise BackboneError("BLOCKED_SOURCE", f"unlocked LongLive provider dependencies: {role}")
        verify_files(source, files)
        roots[role] = {"path": str(source), "files": files}
    return {"binding": BINDING, "roots": roots}


def verify_provider(provider):
    if provider.get("binding") != BINDING:
        raise BackboneError("BLOCKED_ALGORITHM_SEMANTICS", "wrong LongLive provider binding")
    for role, required in REQUIRED.items():
        record = provider["roots"][role]
        if not set(required).issubset(record["files"]):
            raise BackboneError("BLOCKED_SOURCE", f"missing {role} provider lock")
        verify_files(record["path"], record["files"])


def bind_prompt_guards(source, switches):
    """Rebind only two input guards; preserve the original algorithm AST.

    WorldAttention guards are discharged only inside the wrapper's proven
    text-selection/crossattention-invalidation transaction, including clean.
    Outside it the author's guard still rejects unhandled switches.
    """
    from draftkv.benchmarks.worldattention import native_schedule
    switches = tuple(switches)
    if switches not in ((), LEGACY_SWITCHES, tuple(native_schedule()["prompt_switch_frames"])):
        raise BackboneError("BLOCKED_INPUT_PROTOCOL", "unimplemented LongLive prompt-switch schedule")
    tree = ast.parse(source)
    cls = next((node for node in tree.body if isinstance(node, ast.ClassDef)
                and node.name == "LongLive1DraftMap"), None)
    changed = []
    if cls:
        for method in cls.body:
            if not isinstance(method, ast.FunctionDef) or method.name not in ("attend", "_attend_native_b01"):
                continue
            for node in ast.walk(method):
                if (isinstance(node, ast.Compare) and isinstance(node.left, ast.Name)
                        and node.left.id == "current_frame" and len(node.ops) == 1
                        and isinstance(node.ops[0], ast.In) and isinstance(node.comparators[0], ast.Tuple)
                        and ast.literal_eval(node.comparators[0]) == LEGACY_SWITCHES):
                    node.comparators[0] = ast.Name(id="_draftkv_prompt_switches", ctx=ast.Load())
                    changed.append(method.name)
    if sorted(changed) != ["_attend_native_b01", "attend"]:
        raise BackboneError("SOURCE_ABI_DRIFT", "exactly two original LongLive prompt-switch guards required")
    return ast.fix_missing_locations(tree)


def _module(name, path, *, tree=None, namespace=None):
    if name in sys.modules:
        raise BackboneError("MODEL_NAMESPACE_COLLISION", f"external provider already imported: {name}")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    module.__dict__.update(namespace or {})
    if tree is None:
        spec.loader.exec_module(module)
    else:
        exec(compile(tree, str(path), "exec"), module.__dict__)
    if "." in name:
        parent, attribute = name.rsplit(".", 1)
        setattr(sys.modules[parent], attribute, module)
    return module


def install(core, caches, torch):
    """One original handler per real native layer, sharing its native cache."""
    provider = core.plan["longlive_provider"]
    verify_provider(provider)
    roots = {role: Path(record["path"]) for role, record in provider["roots"].items()}
    for name in ("registry", "ragged_frame_index", "compat", "canonical", "hierarchical_v1_selector",
                 "longlive1_native_hook", "longlive1_draftmap_candidate"):
        if name in sys.modules or any(key.startswith(name + ".") for key in sys.modules):
            raise BackboneError("MODEL_NAMESPACE_COLLISION", f"use a fresh provider worker: {name}")
    # Load the inventory explicitly; do not let an unrelated registry.py or
    # compat package elsewhere on sys.path satisfy these algorithm imports.
    for name, role, relative in (
        ("registry", "support", "registry.py"),
        ("ragged_frame_index", "support", "ragged_frame_index.py"),
        ("canonical", "math", "canonical/__init__.py"),
        ("canonical.core", "math", "canonical/core.py"),
        ("compat", "math", "compat/__init__.py"),
        ("compat.native_math", "math", "compat/native_math.py"),
        ("hierarchical_v1_selector", "handler", "hierarchical_v1_selector.py"),
    ):
        _module(name, roots[role] / relative)
    hook = _module("longlive1_native_hook", roots["handler"] / "longlive1_native_hook.py")
    source = Path(core.plan["native_root"]) / "wan/modules/causal_model.py"
    hook.SOURCE = source
    path = roots["handler"] / "longlive1_draftmap_candidate.py"
    switches = tuple(core.plan["request"]["prompt_switch_frames"])
    from .longlive_conditioning import PromptSwitchGuard
    guards = PromptSwitchGuard(switches) if switches not in ((), LEGACY_SWITCHES) else switches
    module = _module("longlive1_draftmap_candidate", path,
                     tree=bind_prompt_guards(path.read_bytes(), switches),
                     namespace={"_draftkv_prompt_switches": guards})
    module.SOURCE = source
    module.AUTHOR = roots["author"] / "draftmap_perlayer.py"
    module.SHARED = roots["support"] / "lingbot_native_candidate/unified_core/draftmap.py"
    model = core.model.get_base_model() if hasattr(core.model, "get_base_model") else core.model
    geometry = core.plan["history_plan"]
    if len(model.blocks) != core.spec.layers or len(caches) != core.spec.layers:
        raise BackboneError("SOURCE_ABI_DRIFT", "LongLive requires every native layer/cache")
    handlers = module.install_model_layers(
        model, torch, core.native_fa2, frame_tokens=geometry["frame_tokens"],
        heads=geometry["heads_per_rank"], head_dim=core.spec.head_dim,
        pinned_capacity_frames=geometry["capacity_frames"],
        max_total_pinned_bytes=geometry["limit_bytes_per_rank"], allow_gpu_sink_recache=False)
    active = ContextVar("longlive_attention_owner", default=None)
    selector = sys.modules["hierarchical_v1_selector"]
    select_v1 = selector.select_v1

    def observe_selection(*args, **kwargs):
        decision = select_v1(*args, **kwargs)
        owner = active.get()
        if owner is not None:
            owner._draftkv_decision = decision
        return decision

    selector.select_v1 = observe_selection
    for layer, handler in enumerate(handlers):
        # Historical direct-B01 settings; never select serial/native or g4
        # comparison modes. Original state/events retain async tensor lifetime.
        handler._b01_coalesce_frame_dma = True
        handler._e2e_events = []
        original = handler.attend

        def observe(owner, *args, _original=original, _layer=layer, **kwargs):
            owner._draftkv_decision = None
            token = active.set(owner)
            try:
                result = _original(*args, **kwargs)
            finally:
                active.reset(token)
            start, count, selected, visible = owner.calls[-1]
            decision = owner._draftkv_decision
            core.trace("qkv_route", layer=_layer, current_start=start * owner.frame_tokens,
                       start_frame=start, frame_count=count, frame_tokens=owner.frame_tokens,
                       query_shape=list(args[0].shape), current_key_shape=list(args[4].shape),
                       sink_frame_ids=list(range(min(owner.sink, start))), retrieved_frame_ids=list(selected),
                       local_frame_ids=list(decision.local) if decision else list(range(min(owner.sink, start), start)),
                       salient_frame_ids=list(decision.salient) if decision else [],
                       current_frame_ids=list(range(start, start + count)), visible_tokens=visible,
                       clean=owner.clean_mode, cache_owner="original_LongLive1DraftMap_per_layer",
                       frame_page_mapping="one_full_BF16_frame_per_pinned_slot_absolute_id",
                       consumer="original_b01_streaming_fa2" if selected else "native_fa2_warmup",
                       commit_generation=owner._b01_commit_generation)
            return result

        handler.attend = types.MethodType(observe, handler)
    core.trace("history_capacity_bound", **geometry)
    core.trace("algorithm_provider_bound", binding=BINDING, prompt_switch_frames=list(switches),
               source_roots=provider["roots"], clean_session="original_NativeCleanSession")
    return CleanBinding(module.NativeCleanSession(tuple(zip(handlers, caches))), handlers, core.trace)


class CleanBinding:
    """Observe the original all-layer transaction, including its failure path."""
    def __init__(self, session, handlers, trace):
        self.session, self.handlers, self.trace = session, handlers, trace

    def run(self, call, *, timestep):
        try:
            result = self.session.run(call, timestep=timestep)
        except BaseException:
            self.trace("clean_commit_failed", invalidated=[h.invalidated for h in self.handlers])
            raise
        self.trace("clean_commit", layers=len(self.handlers),
                   committed_frames=[h.frame_index.total_frames for h in self.handlers],
                   commit_generations=[h._b01_commit_generation for h in self.handlers],
                   transaction="original_all_layer_D2H_fenced_atomic_publication")
        return result
