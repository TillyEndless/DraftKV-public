"""DraftKV's native seam: current QKV only, no GPU rolling history.

Keep the author's QKV, absolute RoPE, SP collectives and output projection.
Historical attention belongs exclusively to the CPU-backed DraftKV owner.
Other algorithms continue to use the unmodified native rolling-cache seam.
"""
import ast
import functools
import inspect
import textwrap

from .contracts import BackboneError


def _drift(message):
    raise BackboneError("SOURCE_ABI_DRIFT", "DraftKV current-only seam: " + message)


def _assigns(node, name):
    return isinstance(node, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id == name for t in node.targets)


def _cut_at_current_end(body):
    indices = [i for i, node in enumerate(body) if _assigns(node, "current_end")]
    if len(indices) != 1:
        _drift("expected one native current_end boundary")
    prefix = body[:indices[0]]
    # The frozen variants must produce BOTH current absolute-position tensors
    # before touching rolling history. Online-relative latentmem is not an
    # interchangeable source and must fail closed.
    for name in ("query", "key"):
        calls = [n for stmt in prefix for n in ast.walk(stmt)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                 and n.func.id == "causal_rope_apply" and n.args
                 and isinstance(n.args[0], ast.Name)
                 and n.args[0].id == ("q" if name == "query" else "k")]
        if len(calls) != 1:
            _drift("missing native absolute current " + name + " RoPE")
    return prefix


def advance_frontier(cache, end):
    """Identity counters survive FSDP/CFG and sink replay; no KV is stored."""
    frontier = max(int(cache["global_end_index"].item()), int(end))
    cache["global_end_index"].fill_(frontier)
    cache["local_end_index"].zero_()
    return frontier


def compile_current_only(function, bridge, *, family, source_text=None):
    tree = ast.parse(textwrap.dedent(source_text or inspect.getsource(function)))
    definition = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
    definition.decorator_list = []
    body = definition.body
    if family == "lingbot":
        routes = [n for n in ast.walk(definition) if isinstance(n, ast.Call)
                  and isinstance(n.func, ast.Name) and n.func.id == "route_attention"]
        if len(routes) != 1 or len(routes[0].args) < 4:
            _drift("expected one LingBot route")
        query, key = (ast.unparse(n) for n in routes[0].args[1:3])
        if (query, key) not in (("query", "key"), ("roped_query", "roped_key")):
            _drift("unknown LingBot current Q/K")
        prefix = _cut_at_current_end(body)
        ends = [i for i, n in enumerate(body) if isinstance(n, ast.Expr)
                and ast.unparse(n.value) == "kv_cache['local_end_index'].fill_(local_end_index)"]
        if len(ends) != 1:
            _drift("expected native rolling tail")
        suffix = body[ends[0] + 1:]
        sp = query == "query"
        # The old direct-B01 branch's output reuse fence belongs to its deleted
        # cache/consumer resources. The canonical consumer joins its own streams.
        suffix = [n for n in suffix if not (
            isinstance(n, ast.If) and ast.unparse(n.test) == "_direct_state is not None")]
        result = "x_local" if sp else "x"
        middle = ast.parse(f"""
{result} = _draftkv_attention(self, {query}, {key}, v, {key}, v,
                            q, k, frame_seqlen, grid_sizes, freqs, kv_cache)
_draftkv_advance(kv_cache, current_start + {query}.shape[1])
""").body
        definition.body = prefix + middle + suffix
    elif family == "longlive":
        branches = [n for n in body if isinstance(n, ast.If)
                    and ast.unparse(n.test) == "kv_cache is None"]
        if len(branches) != 1:
            _drift("expected LongLive cached branch")
        branch = branches[0]
        prefix = _cut_at_current_end(branch.orelse)
        branch.orelse = prefix + ast.parse("""
x = _draftkv_attention(self, roped_query, roped_key, v, roped_key, v,
                       q, k, frame_seqlen, grid_sizes, freqs, kv_cache)
current_end = _draftkv_advance(kv_cache, current_start + roped_query.shape[1])
local_end_index = 0
cache_update_info = None
""").body
        # Native whole-model _apply_cache_updates still receives its tuple, but
        # None carries no rolling data, and the local storage frontier is zero.
    else:
        _drift("unsupported family")
    if any(isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name)
           and n.value.id == "kv_cache" and isinstance(n.slice, ast.Constant)
           and n.slice.value in ("k", "v") for n in ast.walk(definition)):
        _drift("rolling K/V access survived the current-only rewrite")
    namespace = dict(function.__globals__, _draftkv_attention=bridge,
                     _draftkv_advance=advance_frontier)
    tree = ast.fix_missing_locations(tree)
    exec(compile(tree, inspect.getsourcefile(function) or "<draftkv-current-only>", "exec"), namespace)
    return namespace[definition.name]


def install_empty_native_cache(pipeline, family):
    """Intercept before allocation: zero-token K/V, native integer identities.

    Calling the native allocator with zero capacity preserves its container,
    device, dtype, counters and LongLive extra fields without ever allocating a
    window-sized GPU tensor. Cross-attention/text KV is unaffected.
    """
    name = "_initialize_self_kv_cache" if family == "lingbot" else "_initialize_kv_cache"
    original = getattr(pipeline, name, None)
    if not callable(original):
        _drift("native self KV allocator missing")
    if getattr(original, "_draftkv_empty_history", False):
        return
    signature = inspect.signature(original)
    capacity = "shape" if family == "lingbot" else "kv_cache_size_override"
    if capacity not in signature.parameters:
        _drift("native allocator capacity ABI changed")

    @functools.wraps(original)
    def allocate(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        if family == "lingbot":
            shape = list(bound.arguments[capacity])
            if len(shape) != 4:
                _drift("expected native B,T,H,D cache")
            shape[1] = 0
            bound.arguments[capacity] = shape
        else:
            bound.arguments[capacity] = 0
        result = original(*bound.args, **bound.kwargs)
        rows = result if family == "lingbot" else pipeline.kv_cache1
        if any(row[name].shape[1] != 0 or row[name].untyped_storage().nbytes() != 0
               for row in rows for name in ("k", "v")):
            _drift("native allocator did not honor zero history capacity")
        return result

    allocate._draftkv_empty_history = True
    setattr(pipeline, name, allocate)


def discard_native_history(cache):
    """Handle caller-supplied raw-forward caches as well as native allocators."""
    if not isinstance(cache, (list, tuple)):
        _drift("native cache list required")
    discarded = 0
    for row in cache:
        for name in ("k", "v"):
            value = row.get(name)
            if value is None or value.ndim != 4:
                _drift("native K/V placeholder missing")
            # Empty views can still own a full GPU rolling allocation.
            storage_bytes = value.untyped_storage().nbytes()
            if value.shape[1] or storage_bytes:
                discarded += storage_bytes
                shape = list(value.shape)
                shape[1] = 0
                row[name] = value.new_empty(shape)
        # Historical legacy latent-memory lists are unused by DraftKV as well.
        for name in ("cpu_k_frames", "cpu_v_frames"):
            if row.get(name):
                _drift("unexpected legacy native CPU history")
    return discarded
