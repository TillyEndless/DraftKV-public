"""Exact attention transport and model topology for the benchmark backbones.

LingBot14B uses author-native Ulysses head sharding. LingBot1.3B and LongLive
use replicated model activations and K/V with
partitioned attention queries. Twelve heads cannot divide SP8 without
padding or changing the model. We therefore shard only the query
sequence at the attention consumer, keep exact full-head K/V on every rank, and
all-gather the attention output in original query order.  This is mathematically
exact for the native non-causal cached attention call and does not alter K/V,
RoPE, retrieval, cache ownership, or attention normalization.

This module declares static capability only.  It is not GPU/E2E acceptance.
"""
from __future__ import annotations

import os

from .contracts import BackboneError

SUPPORTED_SP = (1, 2, 4, 8)


def topology_for(family: str, sp: int, heads: int | None = None) -> str:
    if type(sp) is not int or sp not in SUPPORTED_SP:
        raise BackboneError("UNSUPPORTED", f"SP{sp} is outside the release topology set")
    if family not in ("lingbot", "longlive", "solarwm"):
        raise BackboneError("UNSUPPORTED", f"unknown distributed family {family}")
    if heads is not None and (type(heads) is not int or heads <= 0):
        raise BackboneError("UNSUPPORTED", "head count must be a positive integer")
    if family == "solarwm" and heads != 56:
        raise BackboneError("UNSUPPORTED", "H3 requires exactly 56 unpadded heads")
    if sp == 1:
        return "single"
    if family == "solarwm":
        if heads != 56 or heads % sp:
            raise BackboneError("UNSUPPORTED", "H3 requires exactly 56 unpadded heads")
        return "native_head_shard"
    if family == "longlive":
        return "query_sequence_replicated_kv"
    if family == "lingbot":
        # A single exact topology across all 1.3B algorithms also retains the
        # frozen B01 full-12-head consumer geometry at SP2 and SP4.
        if heads == 12 or (heads is not None and heads % sp):
            return "query_sequence_replicated_kv"
        return "native_head_shard"
    raise BackboneError("UNSUPPORTED", f"unknown distributed family {family}")


def attention_sp(family: str, sp: int, heads: int | None = None) -> int:
    """Logical head-shard world used by algorithm/cache semantics.

    The query topology distributes only the query sequence and replicates exact full-head
    K/V, so its algorithm-owned history remains an SP1 head geometry even when
    the physical torchrun world contains 2/4/8 ranks.
    """
    return sp if topology_for(family, sp, heads) == "native_head_shard" else 1


def local_heads(global_heads: int, family: str, sp: int) -> int:
    mode = topology_for(family, sp, global_heads)
    if mode != "native_head_shard":
        return global_heads
    if global_heads % sp:
        raise BackboneError(
            "UNSUPPORTED",
            f"native head-sharded SP{sp} cannot divide {global_heads} model heads",
        )
    return global_heads // sp


def query_bounds(length: int, sp: int, rank: int) -> tuple[int, int]:
    if (type(length) is not int or length <= 0 or type(sp) is not int or sp not in SUPPORTED_SP
            or type(rank) is not int or not 0 <= rank < sp or length < sp):
        raise BackboneError("UNSUPPORTED", "invalid query-sequence SP geometry")
    begin = length * rank // sp
    end = length * (rank + 1) // sp
    if begin >= end:
        raise BackboneError(
            "UNSUPPORTED",
            f"SP{sp} has more ranks than query tokens ({length})",
        )
    return begin, end


def rank_mapping(sp: int, environ=None) -> dict:
    """Validate physical single-node rank/device ownership without importing CUDA.

    CUDA remaps the ordered visible list to local ordinal 0..SP-1. Physical
    ordinals need not be contiguous (UUID lists are also accepted), but every
    rank must own one distinct visible device and belong to the same world.
    """
    if type(sp) is not int or sp not in SUPPORTED_SP:
        raise BackboneError("UNSUPPORTED", "invalid physical SP world")
    env = os.environ if environ is None else environ
    try:
        rank = int(env.get("RANK", "0")) if sp == 1 else int(env["RANK"])
        local_rank = int(env.get("LOCAL_RANK", "0")) if sp == 1 else int(env["LOCAL_RANK"])
        world = int(env.get("WORLD_SIZE", "1")) if sp == 1 else int(env["WORLD_SIZE"])
        local_world = int(env.get("LOCAL_WORLD_SIZE", str(world)))
    except (KeyError, ValueError) as exc:
        raise BackboneError("UNSUPPORTED", "SP>1 requires an exact torchrun rank environment") from exc
    if world != sp or local_world != sp or rank != local_rank or not 0 <= rank < sp:
        raise BackboneError("UNSUPPORTED", "torchrun rank/world differs from selected SP topology")
    visible = None
    if "CUDA_VISIBLE_DEVICES" in env:
        visible = [device.strip() for device in env["CUDA_VISIBLE_DEVICES"].split(",")]
        identities = [str(int(device)) if device.isdecimal() else device.lower() for device in visible]
        if (len(visible) != sp or any(not device or device == "-1" for device in visible)
                or len(set(identities)) != sp):
            raise BackboneError("UNSUPPORTED", "CUDA_VISIBLE_DEVICES must map each SP rank to one distinct device")
    return {"rank": rank, "local_rank": local_rank, "world_size": world,
            "visible_device": visible[local_rank] if visible is not None else None}


def require_torchrun(sp: int):
    """Initialize/verify the existing single-node torchrun world lazily."""
    import torch
    import torch.distributed as dist

    mapping = rank_mapping(sp)
    rank, local_rank = mapping["rank"], mapping["local_rank"]
    if sp == 1:
        return rank
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    if dist.get_world_size() != sp or dist.get_rank() != rank or dist.get_backend() != "nccl":
        raise BackboneError("UNSUPPORTED", "initialized NCCL group differs from selected SP topology")
    return rank



def gather_query_output(local_output, total_length: int, sp: int, rank: int):
    """Gather uneven query slices without ever feeding collective padding to attention."""
    begin, end = query_bounds(total_length, sp, rank)
    expected = end - begin
    if local_output.ndim != 4 or local_output.shape[1] != expected:
        raise BackboneError("PARTIAL_NATIVE_WIRING", "local attention output length differs from assigned query slice")
    if sp == 1:
        return local_output
    import torch
    import torch.distributed as dist
    if not dist.is_initialized() or dist.get_world_size() != sp or dist.get_rank() != rank:
        raise BackboneError("UNSUPPORTED", "query gather rank/world differs from selected topology")
    lengths = [query_bounds(total_length, sp, r)[1] - query_bounds(total_length, sp, r)[0]
               for r in range(sp)]
    maximum = max(lengths)
    padded = local_output.contiguous()
    if expected < maximum:
        padded = torch.cat([local_output, local_output.new_zeros(
            (local_output.shape[0], maximum - expected, *local_output.shape[2:]))], dim=1)
    shards = [torch.empty_like(padded) for _ in range(sp)]
    dist.all_gather(shards, padded)
    return torch.cat([shard[:, :length] for shard, length in zip(shards, lengths)], dim=1)

def query_sequence_attention(query, sp: int, compute):
    """Run exact attention on one Q slice and all-gather output along sequence.

    K/V are intentionally supplied by the caller unchanged/replicated.  Uneven
    query partitions are padded only for the collective, then trimmed before
    concatenation; padding never enters attention.
    """
    if query.ndim != 4 or any(int(size) <= 0 for size in query.shape):
        raise BackboneError("KV_CONSTRAINT", "attention query must have positive B,T,H,D geometry")
    query_bounds(int(query.shape[1]), sp, 0)
    if sp == 1:
        output = compute(query)
        if tuple(output.shape) != tuple(query.shape):
            raise BackboneError("PARTIAL_NATIVE_WIRING", "attention output differs from query shape")
        return output, {
            "mode": "single", "sp": 1, "rank": 0,
            "query_start": 0, "query_stop": int(query.shape[1]),
        }

    rank = require_torchrun(sp)
    begin, end = query_bounds(int(query.shape[1]), sp, rank)
    local_query = query[:, begin:end].contiguous()
    local_output = compute(local_query)
    if tuple(local_output.shape) != tuple(local_query.shape):
        raise BackboneError("PARTIAL_NATIVE_WIRING", "SP attention output differs from local query shape")

    lengths = [query_bounds(int(query.shape[1]), sp, r)[1] -
               query_bounds(int(query.shape[1]), sp, r)[0] for r in range(sp)]
    output = gather_query_output(local_output, int(query.shape[1]), sp, rank)
    if tuple(output.shape) != tuple(query.shape):
        raise BackboneError("PARTIAL_NATIVE_WIRING", "all-gather did not reconstruct the native query sequence")
    return output, {
        "mode": "query_sequence_replicated_kv", "sp": sp, "rank": rank,
        "query_start": begin, "query_stop": end, "query_lengths": lengths,
        "kv_replication": "exact_full_heads_full_tokens_per_rank",
        "head_padding": False,
    }


def native_attention(native_fa2, sp: int):
    """Wrap Native FA2; preserve full-head/full-token K/V on every rank."""
    if sp == 1:
        return native_fa2

    def wrapped(q, k, v):
        if (q.ndim != 4 or k.ndim != 4 or tuple(k.shape) != tuple(v.shape)
                or q.shape[0] != k.shape[0] or tuple(q.shape[2:]) != tuple(k.shape[2:])):
            raise BackboneError("KV_CONSTRAINT", "replicated Native FA2 Q/K/V head geometry differs")
        result, _ = query_sequence_attention(q, sp, lambda local_q: native_fa2(local_q, k, v))
        return result

    wrapped._draftkv_unsharded = native_fa2
    return wrapped


def model_topology(family: str, heads: int, sp: int) -> dict:
    """One contract for planning, loader, cache, retrieval and B01 geometry.

    FSDP parameter sharding is independent of attention head sharding. In the
    query topology, native ``use_sp`` must be False: the author keeps the full
    latent sequence, camera/positions and all heads, allocates full-head KV,
    and skips Ulysses all-to-all. T5 and DiT still use all physical FSDP ranks.
    """
    mode = topology_for(family, sp, heads)
    semantic_sp = attention_sp(family, sp, heads)
    return {"sp": sp, "sp_mode": mode, "attention_sp": semantic_sp,
            "native_use_sp": mode == "native_head_shard", "fsdp": sp > 1,
            "heads_per_rank": local_heads(heads, family, sp),
            "cache_layout": "full_tokens_local_heads" if mode == "native_head_shard" else "replicated_full_tokens_full_heads",
            "score_aggregation": "author_head_collective" if semantic_sp > 1 else "complete_query_and_heads_per_rank",
            "output_restore": "native_reverse_all_to_all" if mode == "native_head_shard" else ("all_gather_query_order" if sp > 1 else "identity"),
            "head_padding": False, "gpu_verified": False}
