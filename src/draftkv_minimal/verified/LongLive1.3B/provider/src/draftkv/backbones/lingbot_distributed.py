"""Small source-guarded adapter for LingBot's query-sequence topology.

The frozen generate function owns NCCL initialization, T5/DiT FSDP, seed
broadcast, image conditioning, rank0 save and communicator teardown. Only its
head-divisibility assertion needs a different contract when Ulysses head
sharding is replaced by exact query partitioning. No vendor file is edited.
"""
from __future__ import annotations

import ast
import functools
import inspect
import textwrap

from .contracts import BackboneError
from .sequence_parallel import model_topology


def validate_query_world(heads: int, sp: int, world: int):
    topology = model_topology("lingbot", heads, sp)
    if world != sp or topology["sp_mode"] != "query_sequence_replicated_kv":
        raise BackboneError("UNSUPPORTED", "query-sequence adapter requires the exact physical world and full-head topology")


def query_sequence_generate(original, *, heads: int, sp: int, source_text=None):
    """Preserve the author lifecycle and replace only the Ulysses-only guard."""
    validate_query_world(heads, sp, sp)
    tree = ast.parse(textwrap.dedent(source_text or inspect.getsource(original)))
    definition = next((node for node in tree.body if isinstance(node, ast.FunctionDef)), None)
    if definition is None:
        raise BackboneError("SOURCE_ABI_DRIFT", "native generate function missing")
    expected = ast.dump(ast.parse("cfg.num_heads % args.ulysses_size == 0", mode="eval").body)
    guards = [node for node in ast.walk(definition) if isinstance(node, ast.Assert)
              and ast.dump(node.test) == expected]
    if len(guards) != 1:
        raise BackboneError("SOURCE_ABI_DRIFT", "native generate Ulysses head-divisibility guard changed")
    # Keep the separate args.ulysses_size == world_size assertion and every
    # original init_distributed_group/seed/decoder/save/barrier call verbatim.
    guard = guards[0]
    guard.test = ast.parse("_query_world(cfg.num_heads, args.ulysses_size, world_size)", mode="eval").body

    def check(actual_heads, actual_sp, world):
        if actual_heads != heads or actual_sp != sp:
            raise BackboneError("SOURCE_ABI_DRIFT", "native task heads/SP differ from the selected model")
        validate_query_world(actual_heads, actual_sp, world)
        return True

    namespace = dict(original.__globals__, _query_world=check)
    definition.decorator_list = []
    exec(compile(ast.fix_missing_locations(tree), inspect.getsourcefile(original) or "<lingbot-query-world>", "exec"), namespace)
    return functools.wraps(original)(namespace[definition.name])
