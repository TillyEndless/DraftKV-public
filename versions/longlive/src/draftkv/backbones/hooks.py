"""Algorithm registration and transactional adapter for native attention seams."""

import ast
from contextlib import contextmanager
from dataclasses import dataclass
import functools
import inspect
import textwrap
import types

from .catalog import ALGORITHM_REQUIREMENTS, algorithm_requirements
from .contracts import ABI_VERSION, BackboneError, QKVContext
from .failure_scope import bypass_failed_transaction_rollback


@dataclass(frozen=True)
class HookRegistration:
    algorithm: str
    factory: object
    semantics: frozenset[str]
    source_sha256: str
    consumer: str
    abi_version: int = ABI_VERSION


class HookRegistry:
    """Runtime registration is separate from the release's audited coverage.

    A registration records the provider's semantic claims, NOT E2E acceptance.
    The release catalog remains blocked until that provider is source audited.
    """

    def __init__(self):
        self.registrations = {}

    def register(self, registration: HookRegistration):
        import re
        algorithm = registration.algorithm
        if algorithm not in ALGORITHM_REQUIREMENTS or algorithm == "native":
            raise BackboneError("BLOCKED_ALGORITHM_SEMANTICS", "Native uses the author's unmodified path")
        required = frozenset(ALGORITHM_REQUIREMENTS[algorithm])
        expected = "streaming_fa2" if algorithm == "draftkv" else "native_fa2"
        if (registration.abi_version != ABI_VERSION or not callable(registration.factory)
                or not required <= registration.semantics or registration.consumer != expected
                or not re.fullmatch(r"[a-f0-9]{64}", registration.source_sha256)):
            raise BackboneError("BLOCKED_ALGORITHM_SEMANTICS", f"incomplete {algorithm} implementation declaration")
        if algorithm in self.registrations:
            raise BackboneError("DUPLICATE_HOOK", algorithm)
        self.registrations[algorithm] = registration

    def build(self, algorithm, host):
        if algorithm not in self.registrations:
            raise BackboneError("BLOCKED_ALGORITHM_SEMANTICS", f"no original {algorithm} hook registered")
        registration = self.registrations[algorithm]
        required = frozenset(algorithm_requirements(algorithm, host.spec.backbone)
                             if algorithm == "draftkv" else ALGORITHM_REQUIREMENTS[algorithm])
        if not required <= registration.semantics:
            raise BackboneError("BLOCKED_ALGORITHM_SEMANTICS", "missing model-specific retrieval capabilities")
        hook = registration.factory(host)
        if any(not callable(getattr(hook, name, None)) for name in ("route", "consume", "after_attention")):
            raise BackboneError("BLOCKED_ALGORITHM_SEMANTICS", "incomplete attention lifecycle")
        owner = getattr(hook, "owner", None)
        if any(not callable(getattr(owner, name, None)) for name in
               ("begin_forward", "history_ids", "frame_pages", "commit_clean", "rollback")):
            raise BackboneError("BLOCKED_ALGORITHM_SEMANTICS", "missing historical cache owner")
        return hook, registration


class HookSession:
    def __init__(self, spec, hook, registration, trace, conditioning, sp, *,
                 physical_sp=None, sp_mode="single"):
        self.spec, self.hook, self.registration = spec, hook, registration
        from .budgets import effective_budget
        self.budget = (hook.budget if registration.algorithm in ("deep-forcing", "future-forcing", "worldtrace-landmark")
                       else effective_budget(spec, registration.algorithm))
        self.trace, self.conditioning, self.sp = trace, conditioning, sp
        self.physical_sp = sp if physical_sp is None else physical_sp
        self.sp_mode = sp_mode
        self.active = None
        self.pending = {}
        self.poisoned = False
        self.forward_sequence = 0
        self.expected_layers = frozenset(range(spec.layers))

    @contextmanager
    def forward(self, *, start_frame, clean, timestep):
        if self.active is not None or self.poisoned:
            raise BackboneError("DIRTY_HISTORY", "nested or previously failed forward")
        self.forward_sequence += 1
        self.active = dict(start_frame=start_frame, clean=clean, timestep=timestep)
        self.trace("algorithm_forward_begin", forward_id=self.forward_sequence, start_frame=start_frame, clean=clean,
                   attention_branch=self.conditioning.get("attention_branch"))
        self.seen = set()
        abandon = False
        try:
            self.hook.owner.begin_forward(**self.active)
            yield
            if self.pending or self.seen != self.expected_layers:
                raise BackboneError("PARTIAL_NATIVE_WIRING", "not all model layers completed attention")
            if clean:
                self.hook.owner.commit_clean(timestep=timestep)
                self.trace("clean_commit", layers=self.spec.layers, start_frame=start_frame, forward_id=self.forward_sequence, attention_branch=self.conditioning.get("attention_branch"))
            self.trace("algorithm_forward_complete", forward_id=self.forward_sequence, start_frame=start_frame, clean=clean, layers=self.spec.layers, attention_branch=self.conditioning.get("attention_branch"))
        except BaseException as error:
            self.poisoned = True
            abandon = bypass_failed_transaction_rollback(error)
            if not abandon:
                self.hook.owner.rollback()
                self.trace("rollback", start_frame=start_frame, forward_id=self.forward_sequence,
                           attention_branch=self.conditioning.get("attention_branch"))
            raise
        finally:
            # A fatal isolated worker is about to exit without destructors.
            # Keep asynchronous source references alive until process exit;
            # rollback fences or trace backpressure could hide the root error.
            if not abandon:
                self.active = None
                self.pending.clear()

    def attend(self, layer, query, key, value, native_keys, native_values, raw_query,
               raw_key, frame_tokens, grid_sizes, freqs, native_fa2, native_cache=None, *, fixed_prefix=None, native_geometry=None):
        if self.active is None or layer in self.seen or layer in self.pending:
            raise BackboneError("PARTIAL_NATIVE_WIRING", "attention outside a unique layer transaction")
        context = QKVContext(layer, query, key, value, raw_query, raw_key,
                             native_keys, native_values, self.active["start_frame"],
                             frame_tokens, key.shape[1] // frame_tokens, self.budget.sink,
                             self.budget.capacity, self.sp, grid_sizes, freqs,
                             self.conditioning, self.active["clean"], self.active["timestep"], native_cache, self.forward_sequence, fixed_prefix, native_geometry)
        if key.shape[1] % frame_tokens:
            raise BackboneError("KV_CONSTRAINT", "partial frame in native current tokens")
        selection = self.hook.route(context)
        # FullKV's explicit experimental owner may lift the rolling-window
        # validator, without changing production/native cache capacities.
        validation_context = getattr(self.hook, 'validation_context', lambda c: c)(context)
        selection.validate(validation_context)
        selected_history = selection.local_ids + selection.retrieved_ids
        contains = getattr(self.hook.owner, "contains_history_ids", None)
        present = (contains(layer, selected_history) if callable(contains)
                   else set(selected_history) <= set(self.hook.owner.history_ids(layer)))
        if not present:
            raise BackboneError("KV_CONSTRAINT", "selected history absent from this layer's cache owner")
        self.pending[layer] = selection
        native_calls = 0
        def counted_native(q, k, v, *, head_index=None):
            nonlocal native_calls
            native_calls += 1
            consumer = native_fa2
            if (self.sp_mode == "query_sequence_replicated_kv"
                    and selection.metadata.get("representation") in ("query_head_paged", "frame_head_full")):
                consumer = getattr(native_fa2, "_draftkv_unsharded", native_fa2)
            if context.fixed_prefix is not None:
                import torch
                pk, pv = context.fixed_prefix
                if head_index is not None:
                    pk, pv = pk[:, :, head_index:head_index+1], pv[:, :, head_index:head_index+1]
                k, v = torch.cat((pk, k), 1).contiguous(), torch.cat((pv, v), 1).contiguous()
            return consumer(q, k, v)
        if self.registration.consumer == "native_fa2":
            # Enforce the consumer here; plugins may not replace FA2 with SDPA,
            # DraftKV streaming or an approximate result for another algorithm.
            if selection.metadata.get("representation") in ("query_head_paged", "frame_head_full"):
                result = self.hook.consume(context, selection, counted_native)
                if native_calls < 1:
                    raise BackboneError("BLOCKED_ALGORITHM_SEMANTICS", "retrieved selection never called Native FA2")
            else:
                result = counted_native(selection.query, selection.keys, selection.values)
        else:
            result = self.hook.consume(context, selection, native_fa2)
        observe = getattr(self.hook, "observe_attention_result", None)
        if observe is not None:
            observe(context, selection, result, native_calls)
        self.hook.after_attention(context, selection)
        self.trace("attention", layer=layer, attention_branch=self.conditioning.get("attention_branch"), clean=context.clean, start_frame=context.start_frame, forward_id=context.forward_id,
                   sink_ids=selection.sink_ids, local_ids=selection.local_ids,
                   retrieved_ids=selection.retrieved_ids, current_ids=selection.current_ids,
                   pages=([] if getattr(self.hook.owner,"settings",{}).get("bounded_trace",False) else [vars(page) for page in selection.pages]), consumer=self.registration.consumer,
                   native_fa2_calls=native_calls,
                   consumer_symbol=("flash_attn.flash_attn_func" if getattr(self.spec, "family", None) == "solarwm" else "wan.modules.attention.flash_attention(version=2)") if native_calls else None)
        del self.pending[layer]
        self.seen.add(layer)
        return result


def compile_seam(function, bridge, *, family, source_text=None):
    """Replace only the cached native attention call; QKV/RoPE/cache math stays native.

    LongLive has two cached branches; LingBot has one native consumer per SP or
    non-SP forward. The legacy route/finish callbacks become identity/no-op for
    externally registered plugins. The plugin owns the complete lifecycle.
    Source files on disk are never changed.
    """
    tree = ast.parse(textwrap.dedent(source_text or inspect.getsource(function)))
    definition = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
    definition.decorator_list = []
    current_key = None
    if family == "lingbot":
        routes = [node for node in ast.walk(definition)
                  if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                  and node.func.id == "route_attention" and len(node.args) >= 4]
        if (len(routes) != 1 or not isinstance(routes[0].args[2], ast.Name)
                or routes[0].args[2].id not in ("key", "roped_key")):
            raise BackboneError("SOURCE_ABI_DRIFT", "native LingBot current-key seam changed")
        # Native SP names its post-RoPE key `key`; non-SP uses `roped_key`.
        # The consumer may call either query `routed_query`, so its name
        # cannot identify the current-key tensor.
        current_key = routes[0].args[2].id
    calls = 0
    for node in ast.walk(definition):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if family == "lingbot" and node.func.id in ("route_attention", "finish_attention"):
            node.func.id = "_adapter_identity_route" if node.func.id == "route_attention" else "_adapter_finish"
        if node.func.id not in ("attention", "flash_attention") or len(node.args) != 3:
            continue
        q, k, v = (ast.unparse(arg) for arg in node.args)
        if q not in ("query", "roped_query", "routed_query") or not any(name in k for name in ("cache", "temp_k", "k_cat")):
            continue
        if family == "longlive" and q != "roped_query":
            continue
        names = (["self", q, "roped_key", "v", k, v, "q", "k", "frame_seqlen", "grid_sizes", "freqs", "kv_cache"]
                 if family == "longlive" else
                 ["self", q, current_key, "v", k, v,
                  "q", "k", "frame_seqlen", "grid_sizes", "freqs", "kv_cache"])
        node.func.id = "_adapter_attention"
        node.args = [ast.parse(name, mode="eval").body for name in names]
        node.keywords = []
        calls += 1
    expected = 2 if family == "longlive" else 1
    if calls != expected:
        raise BackboneError("SOURCE_ABI_DRIFT", f"expected {expected} cached attention calls, found {calls}")
    namespace = dict(function.__globals__, _adapter_attention=bridge,
                     _adapter_identity_route=lambda module, q, k, v, nk, nv, *a, **kw: (nk, nv),
                     _adapter_finish=lambda module: None)
    exec(compile(ast.fix_missing_locations(tree), inspect.getsourcefile(function) or "<native-seam>", "exec"), namespace)
    return namespace[definition.name]


def _native_cache_identity(cache):
    """FSDP may rebuild containers while preserving their rolling K/V buffers."""
    if isinstance(cache, (list, tuple)) and cache:
        tensors = [row.get(name) for row in cache if isinstance(row, dict) for name in ("k", "v")]
        if len(tensors) == 2 * len(cache) and all(
                callable(getattr(t, "data_ptr", None)) and t.numel() for t in tensors):
            return ("kv_storage", tuple((str(t.device), t.data_ptr(), tuple(t.shape),
                                        tuple(t.stride()), str(t.dtype)) for t in tensors))
    return ("object", id(cache)) if cache is not None else None


def _native_frontier_cache_identity(cache):
    """Identify a native Wan stream across FSDP container and float casts.

    Wan keeps its integer frontier tensors in place for a stream's lifetime.
    FSDP preserves these tensors even when it rebuilds containers or casts K/V.
    bind_model retains the first cache, so counter identities cannot be reused.
    Independent CFG caches have independent counters, even with shared K/V.
    """
    if not isinstance(cache, (list, tuple)) or not cache:
        raise BackboneError("SOURCE_ABI_DRIFT", "native Wan stream needs a nonempty cache")
    counters = []
    for layer in cache:
        if not isinstance(layer, dict):
            raise BackboneError("SOURCE_ABI_DRIFT", "native Wan cache layer must be a mapping")
        pair = tuple(layer.get(name) for name in ("global_end_index", "local_end_index"))
        if any(str(getattr(t, "dtype", None)) not in ("torch.int32", "torch.int64")
               or not callable(getattr(t, "numel", None)) or t.numel() != 1 for t in pair):
            raise BackboneError("SOURCE_ABI_DRIFT", "native Wan stream needs integer frontier counters")
        counters.append(tuple(id(t) for t in pair))
    return ("native_frontier_counters", tuple(counters))


# Preserve the existing private DraftKV import used by its lifecycle tests.
_draftkv_cache_identity = _native_frontier_cache_identity


def bind_model(core, session):
    """Bind the exact same model to one plugin. Called only inside an isolated worker."""
    if callable(getattr(core, "bind_attention", None)):
        return core.bind_attention(session)
    target = getattr(core.model, "module", core.model)
    if hasattr(target, "get_base_model"):
        target = target.get_base_model()
    modules = [block.self_attn for block in target.blocks]
    if len(modules) != core.spec.layers:
        raise BackboneError("SOURCE_ABI_DRIFT", "model block count changed")
    current_only = session.registration.algorithm == "draftkv"
    if current_only and core.spec.family == "lingbot" and getattr(core, "pipeline", None) is not None:
        from .draftkv_native import install_empty_native_cache
        install_empty_native_cache(core.pipeline, core.spec.family)
    for layer, module in enumerate(modules):
        original = module.forward.__func__
        native_fa2 = core.native_fa2

        def bridge(attn, query, key, value, nk, nv, rq, rk, tokens, grid, freqs, cache,
                   _layer=layer, _consumer=native_fa2):
            # SP padding is removed by native code before this seam.
            rq, rk = rq[:, :query.shape[1]], rk[:, :key.shape[1]]
            return current_session[0].attend(_layer, query, key, value, nk, nv, rq, rk,
                                  int(tokens), grid, freqs, _consumer, cache)

        if current_only:
            from .draftkv_native import compile_current_only
            compiled = compile_current_only(original, bridge, family=core.spec.family)
        else:
            compiled = compile_seam(original, bridge, family=core.spec.family)
        module.forward = types.MethodType(compiled, module)
    # FSDP reconstructs containers and may cast floating K/V at every forward.
    # Wan's integer frontier tensors retain stream identity across both changes.
    # Keep independent CFG counters separate even when they share float storage.
    # Strong references prevent identity reuse; non-FSDP comparator dispatch and
    # Forcing's existing rolling-storage contract remain unchanged.
    # Portable historical providers keep cache attributes on attention modules;
    # swap those references at the whole-forward boundary, never inside a layer.
    from copy import copy, deepcopy
    branch_name = getattr(core, "plan", {}).get("request", {}).get("attention_branch", "native")
    sessions, native_caches, current_session = {}, {}, [session]
    module_states = {}
    forward_sequence = [0]
    camera = [None]
    release_history = session.registration.algorithm in ("shadowkv", "worldtrace-landmark")
    rollout_sessions, released_rollout = {}, [()]

    def activate(selected):
        previous = current_session[0]
        if previous is selected:
            return
        module_states[id(previous)] = [{name: value for name, value in vars(module).items()
                                       if name.startswith("_portable_")} for module in modules]
        for index, module in enumerate(modules):
            for name in list(vars(module)):
                if name.startswith("_portable_"):
                    delattr(module, name)
            for name, value in module_states.get(id(selected), [{} for _ in modules])[index].items():
                setattr(module, name, value)
        current_session[0] = selected

    def select_session(params):
        cache = params.get("kv_cache")
        key = (_native_cache_identity(cache) if session.registration.algorithm in
               ("deep-forcing", "future-forcing") else id(cache) if cache is not None else None)
        if session.registration.algorithm == "draftkv":
            key = _native_frontier_cache_identity(cache)
        if (core.spec.family == "lingbot" and session.physical_sp > 1
                and session.registration.algorithm in
                ("infllm", "shadowkv", "infinigen", "freekv", "worldkv")):
            key = _native_frontier_cache_identity(cache)
        if cache is not None and session.registration.algorithm == "worldtrace-landmark":
            from draftkv.algorithms.worldtrace_landmark.backbone import native_cache_identity
            key = native_cache_identity(cache)
        if key not in sessions:
            branch = branch_name if not sessions else branch_name + ":native_cache_" + str(len(sessions))
            if not sessions:
                selected = session
            else:
                host = copy(core)
                host.plan = deepcopy(core.plan)
                host.plan.setdefault("request", {})["attention_branch"] = branch
                hook = session.registration.factory(host)
                if host.plan.get("bounded_acceptance"):
                    from .acceptance import ForwardObserver, AttentionProbe
                    options = host.plan["bounded_acceptance"]
                    observer = AttentionProbe if options["mode"] == "probe" else ForwardObserver
                    hook = observer(hook, host.plan, options)
                selected = HookSession(session.spec, hook, session.registration, session.trace,
                    host.plan["request"], session.sp, physical_sp=session.physical_sp, sp_mode=session.sp_mode)
                if camera[0] is not None and callable(getattr(hook, "set_camera_poses", None)):
                    hook.set_camera_poses(camera[0])
            selected.conditioning = dict(selected.conditioning, attention_branch=branch)
            sessions[key] = selected
            native_caches[key] = cache
        selected = sessions[key]
        activate(selected)
        if release_history:
            rollout_sessions[id(selected)] = selected
        return selected

    if release_history:
        def release_after_rollout(*, expected_frames):
            """Release every completed CFG history before the separate VAE phase."""
            if type(expected_frames) is not int or expected_frames <= 0:
                raise BackboneError("DIRTY_HISTORY", "rollout release needs a positive frame count")
            # Retain all stream identities, but validate only owners touched
            # since the last successful release. A new video's frame count
            # need not match previously closed videos. Without a new forward,
            # repeat calls still validate the exact last rollout's frontier.
            branches = tuple(rollout_sessions.values()) or released_rollout[0] or (session,)
            previous = current_session[0]
            try:
                # Validate all branches before dropping any history. Portable
                # providers inspect attributes on the shared attention modules,
                # including those belonging to an inactive CFG session.
                for selected in branches:
                    activate(selected)
                    owner = selected.hook.owner
                    if (selected.active is not None or selected.pending or selected.poisoned
                            or getattr(owner, "active", None) is not None
                            or getattr(owner, "pending", {}) or getattr(owner, "poisoned", False)
                            or getattr(owner, "frontier", None) != expected_frames):
                        raise BackboneError("DIRTY_HISTORY", "cannot release an incomplete CFG rollout")
                    if not callable(getattr(owner, "release_after_rollout", None)):
                        raise BackboneError("DIRTY_HISTORY", "cache owner has no successful rollout release")
                    validate = getattr(owner, "validate_release_after_rollout", None)
                    if not getattr(owner, "closed", False) and callable(validate):
                        validate(expected_frames=expected_frames)
                for selected in branches:
                    activate(selected)
                    owner = selected.hook.owner
                    if not getattr(owner, "closed", False):
                        owner.release_after_rollout(expected_frames=expected_frames)
                released_rollout[0] = branches
                rollout_sessions.clear()
            finally:
                activate(previous)

        core.release_after_rollout = release_after_rollout

    if callable(getattr(session.hook, "set_camera_poses", None)):
        from draftkv.algorithms.worldkv.source import capture_native_camera
        def bind_camera(poses):
            camera[0] = poses.detach().clone()
            session.hook.set_camera_poses(camera[0])
            for other in sessions.values():
                if other is not session:
                    other.hook.set_camera_poses(camera[0])
        camera_source_sha = capture_native_camera(core.pipeline, bind_camera)
        core.trace("worldkv_native_camera_seam", source_sha256=camera_source_sha,
                   capture="absolute_interpolation_before_native_relative_plucker")

    forward_name = "_forward_inference" if core.spec.family == "longlive" and hasattr(target, "_forward_inference") else "forward"
    original_forward = getattr(target, forward_name)
    forward_signature = inspect.signature(original_forward)

    @functools.wraps(original_forward)
    def forward(*args, **kwargs):
        bound = forward_signature.bind(*args, **kwargs)
        bound.apply_defaults()
        params = bound.arguments
        timestep = params.get("t")
        frame_tokens = params.get("frame_seqlen")
        if frame_tokens is None and core.spec.family == "longlive" and params.get("x") is not None:
            inputs = params["x"]
            latent = inputs[0] if isinstance(inputs, (list, tuple)) and inputs else inputs
            patch = target.patch_size
            frame_tokens = latent.shape[-2] * latent.shape[-1] // (patch[1] * patch[2])
        if timestep is None or not frame_tokens:
            raise BackboneError("SOURCE_ABI_DRIFT", "native t/frame_seqlen missing")
        start = int(params.get("current_start", 0))
        if start % int(frame_tokens):
            raise BackboneError("KV_CONSTRAINT", "unaligned absolute frame offset")
        if current_only and modules:
            from .draftkv_native import discard_native_history
            discarded = discard_native_history(params.get("kv_cache"))
            core.trace("draftkv_native_history_storage", policy="cpu_history_no_native_gpu_rolling_kv",
                       layers=len(modules), native_kv_storage_bytes=0,
                       discarded_input_storage_bytes=discarded)
        selected = select_session(params)
        selected.forward_sequence = forward_sequence[0]
        try:
            with selected.forward(start_frame=start // int(frame_tokens),
                                  clean=bool((timestep == 0).all()), timestep=timestep):
                return original_forward(*args, **kwargs)
        finally:
            forward_sequence[0] = selected.forward_sequence

    setattr(target, forward_name, forward)
