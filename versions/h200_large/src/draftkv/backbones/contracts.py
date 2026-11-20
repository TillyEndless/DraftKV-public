"""Versioned backbone/plugin ABI. Importing contracts never imports GPU libraries."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

ABI_VERSION = 1


class BackboneError(RuntimeError):
    def __init__(self, code: str, detail: str):
        self.code = code
        super().__init__(f"{code}: {detail}")


class InputMode(str, Enum):
    BENCH1_T2V = "bench1_t2v"
    ORBIA = "orbia"


@dataclass(frozen=True)
class FramePage:
    """Absolute frame ID and exact token interval; pages may span frame boundaries."""

    frame_id: int
    page_id: int
    token_start: int
    token_stop: int
    source_token_ids: tuple[int, ...] = ()


class CacheOwner(Protocol):
    """The algorithm owns one historical bank, distinct from native rolling scratch.

    begin/commit are whole-forward transactions across ALL layers. route() may
    stage data but must never publish noisy K/V. A failed forward poisons the
    transaction; rollback must fence asynchronous transfers before releasing refs.
    """

    def begin_forward(self, *, start_frame: int, clean: bool, timestep: Any) -> None: ...
    def history_ids(self, layer: int) -> tuple[int, ...]: ...
    def frame_pages(self, layer: int, frame_id: int) -> tuple[FramePage, ...]: ...
    def commit_clean(self, *, timestep: Any) -> None: ...
    def rollback(self) -> None: ...


@dataclass(frozen=True)
class QKVContext:
    layer: int
    query: Any                 # post-RoPE native [B,T,H,D], padding removed
    key: Any                   # current post-RoPE K; not historical scratch
    value: Any
    raw_query: Any             # normalized pre-RoPE Q
    raw_key: Any
    native_keys: Any           # native rolling view; current-only for DraftKV
    native_values: Any
    start_frame: int           # absolute, never the rolling cache slot
    frame_tokens: int
    current_frames: int
    sink_frames: int
    window_frames: int
    sp: int
    grid_sizes: Any
    freqs: Any
    conditioning: Mapping[str, Any]
    clean: bool
    timestep: Any
    native_cache: Any = None
    forward_id: int = 0         # worker-local whole-model transaction sequence
    fixed_prefix: Any = None    # model condition K/V, outside video budget
    native_geometry: Any = None # backbone-owned rotary/camera callbacks


@dataclass(frozen=True)
class AttentionSelection:
    query: Any
    keys: Any
    values: Any
    sink_ids: tuple[int, ...]
    local_ids: tuple[int, ...]
    retrieved_ids: tuple[int, ...]
    current_ids: tuple[int, ...]
    pages: tuple[FramePage, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self, context: QKVContext) -> None:
        partitions = (self.sink_ids, self.local_ids, self.retrieved_ids, self.current_ids)
        flat = [fid for part in partitions for fid in part]
        if (any(type(fid) is not int or fid < 0 for fid in flat)
                or len(flat) != len(set(flat))):
            raise BackboneError("KV_CONSTRAINT", "sink/local/retrieved/current IDs must be unique and disjoint")
        expected = tuple(range(context.start_frame, context.start_frame + context.current_frames))
        if self.current_ids != expected or any(fid >= context.start_frame for fid in flat[:-len(expected)]):
            raise BackboneError("KV_CONSTRAINT", "future or missing current frames")
        if self.sink_ids != tuple(range(min(context.sink_frames, context.start_frame))):
            raise BackboneError("KV_CONSTRAINT", "native sink frames must be retained in order")
        if (any(fid >= context.sink_frames for fid in self.sink_ids)
                or any(fid < context.sink_frames for fid in self.local_ids + self.retrieved_ids)):
            raise BackboneError("KV_CONSTRAINT", "sink range or historical partition changed")
        representation = self.metadata.get("representation", "exact")
        expected_tokens = len(flat) * context.frame_tokens
        if representation == "device_archive":
            # Runtime provenance remains on the GPU. CPU audit exports are
            # optional and never construct the next execution input.
            import torch
            budget = self.metadata.get('history_budget')
            ids = self.metadata.get('device_source_tokens')
            if (type(budget) is not int or not 0 <= budget <= context.start_frame-len(self.sink_ids)
                    or self.metadata.get('archive_frontier') != context.start_frame
                    or self.retrieved_ids or self.local_ids or self.pages):
                raise BackboneError('KV_CONSTRAINT', 'invalid device archive budget/frontier')
            expected_tokens = (len(self.sink_ids)+budget+len(self.current_ids))*context.frame_tokens
            if (not isinstance(ids, torch.Tensor) or ids.shape != (expected_tokens,)
                    or ids.dtype != torch.int64 or ids.device != context.query.device
                    or not ids.is_contiguous()):
                raise BackboneError('KV_CONSTRAINT', 'invalid device archive source map')
            torch._assert_async(((ids[1:]>ids[:-1]).all()), 'device archive source IDs not chronological/unique')
            torch._assert_async(((ids>=0)&(ids<(context.start_frame+context.current_frames)*context.frame_tokens)).all(),
                                'device archive source IDs out of causal range')
        elif representation == "device_blocks":
            # Exact dynamic provenance stays on the GPU; no fabricated host IDs.
            import torch
            budget = self.metadata.get("history_budget")
            offsets = self.metadata.get("device_offsets")
            slots = self.metadata.get("device_slots")
            takes = self.metadata.get("device_takes")
            if (type(budget) is not int or not 0 <= budget <= context.start_frame-len(self.sink_ids)
                    or self.metadata.get("archive_frontier") != context.start_frame
                    or self.retrieved_ids or self.pages):
                raise BackboneError("KV_CONSTRAINT", "invalid device block budget/frontier")
            expected_tokens = (len(self.sink_ids)+budget+len(self.current_ids))*context.frame_tokens
            for tensor,shape,dtype in ((offsets,(1,2,expected_tokens),torch.int64),
                                       (slots,(expected_tokens,),torch.int32)):
                if (not isinstance(tensor,torch.Tensor) or tuple(tensor.shape)!=shape
                        or tensor.dtype!=dtype or tensor.device!=context.query.device or not tensor.is_contiguous()):
                    raise BackboneError("KV_CONSTRAINT", "invalid device block layout")
            if not isinstance(takes,torch.Tensor) or not takes.is_cuda or takes.dtype!=torch.int32:
                raise BackboneError("KV_CONSTRAINT", "missing device selection")
        elif representation == "head_paged":
            # Original token selectors choose a different exact set for each
            # head. The union of absolute frames is provenance, not allocation.
            # Original providers can submit either Python rows or a CPU NumPy
            # integer map. Use one vectorized check, not O(heads*tokens)
            # Python int/set conversions on every attention layer.
            import numpy as np
            maps = self.metadata.get("head_token_ids")
            heads, tokens = context.query.shape[2], self.keys.shape[1]
            if maps is None or self.pages:
                raise BackboneError("KV_CONSTRAINT", "head-paged KV requires one exact token map per head and no shared pages")
            try:
                ids = np.asarray(maps)
            except (TypeError, ValueError) as exc:
                raise BackboneError("KV_CONSTRAINT", "invalid head-paged token map") from exc
            if ids.shape != (heads, tokens) or ids.dtype.kind not in ("i", "u"):
                raise BackboneError("KV_CONSTRAINT", "head-paged token map shape or integer dtype changed")
            maximum = (context.start_frame + context.current_frames) * context.frame_tokens
            if np.any(ids >= maximum) or np.any(ids < 0):
                raise BackboneError("KV_CONSTRAINT", "head-paged map contains future or negative tokens")
            frame_tokens = context.frame_tokens
            mandatory_frames = np.asarray(self.sink_ids + self.current_ids, dtype=np.int64)
            # Short videos: absolute coordinate counting avoids any coordinate
            # compression. Long videos: compress only the selected frame domain,
            # independent of total history length. Both paths are exact.
            absolute = heads * maximum <= 4_000_000
            domain = None if absolute else np.sort(np.asarray(flat, dtype=np.int64))
            frame_count = maximum // frame_tokens if absolute else len(domain)
            dense_width = frame_count * frame_tokens
            if heads * dense_width <= 4_000_000 and dense_width > 0:
                if absolute:
                    compressed_token = ids.astype(np.int64, copy=False)
                    mandatory_frame_index = mandatory_frames
                else:
                    source_frame = ids // frame_tokens
                    compressed_frame = np.searchsorted(domain, source_frame)
                    if (np.any(compressed_frame >= len(domain)) or
                        np.any(domain[np.minimum(compressed_frame, len(domain) - 1)] != source_frame)):
                        raise BackboneError("KV_CONSTRAINT", "head-paged frame union differs from exact per-head token maps")
                    compressed_token = compressed_frame * frame_tokens + ids % frame_tokens
                    mandatory_frame_index = np.searchsorted(domain, mandatory_frames)
                    if mandatory_frames.size and (np.any(mandatory_frame_index >= len(domain))
                            or np.any(domain[np.minimum(mandatory_frame_index, len(domain) - 1)] != mandatory_frames)):
                        raise BackboneError("KV_CONSTRAINT", "head-paged map lacks mandatory sink/current tokens")
                offsets = np.arange(heads, dtype=np.int64)[:, None] * dense_width
                multiplicity = np.bincount(
                    (compressed_token + offsets).ravel(),
                    minlength=heads * dense_width,
                ).reshape(heads, frame_count, frame_tokens)
                if np.any(multiplicity > 1):
                    raise BackboneError("KV_CONSTRAINT", "head-paged map has duplicate tokens")
                if mandatory_frames.size:
                    mandatory_tokens = (
                        mandatory_frame_index[:, None] * frame_tokens
                        + np.arange(frame_tokens, dtype=np.int64)[None, :]
                    ).ravel()
                    if np.any(multiplicity.reshape(heads, dense_width)[:, mandatory_tokens] != 1):
                        raise BackboneError("KV_CONSTRAINT", "head-paged map lacks mandatory sink/current tokens")
                occupied_frames = np.flatnonzero(np.any(multiplicity, axis=(0, 2)))
                if (set(occupied_frames.tolist()) != set(flat) if absolute
                        else occupied_frames.size != len(domain)):
                    raise BackboneError("KV_CONSTRAINT", "head-paged frame union differs from exact per-head token maps")
            else:
                # Last-resort exact validator when even selected token volume
                # exceeds the bounded 32 MiB count-table budget.
                ordered = np.sort(ids, axis=1)
                if np.any(ordered[:, 1:] == ordered[:, :-1]):
                    raise BackboneError("KV_CONSTRAINT", "head-paged map has duplicate tokens")
                mandatory = np.concatenate([
                    np.arange(fid * frame_tokens, (fid + 1) * frame_tokens, dtype=np.int64)
                    for fid in mandatory_frames
                ]) if mandatory_frames.size else np.empty(0, dtype=np.int64)
                if mandatory.size:
                    for row in ordered:
                        positions = np.searchsorted(row, mandatory)
                        if np.any(positions >= tokens) or np.any(row[np.minimum(positions, tokens - 1)] != mandatory):
                            raise BackboneError("KV_CONSTRAINT", "head-paged map lacks mandatory sink/current tokens")
                if set(np.unique(ids // frame_tokens).tolist()) != set(flat):
                    raise BackboneError("KV_CONSTRAINT", "head-paged frame union differs from exact per-head token maps")
            expected_tokens = tokens
        elif representation == "episodic_landmark":
            sources = self.metadata.get("slot_source_ids", ())
            positions = self.metadata.get("virtual_frame_ids", ())
            if (not isinstance(sources, tuple) or not isinstance(positions, tuple)
                    or len(sources) != len(positions)
                    or set(sources) != set(self.retrieved_ids)
                    or any(type(t) is not int or not 0 <= t < context.start_frame for t in positions)
                    or tuple(sorted(set(positions))) != positions
                    or self.pages):
                raise BackboneError("KV_CONSTRAINT", "invalid landmark source/virtual-slot map")
            expected_tokens = (len(self.sink_ids) + len(sources) + len(self.local_ids)
                               + len(self.current_ids)) * context.frame_tokens
        elif representation in ("compressed", "paged"):
            ordered = sorted(self.pages, key=lambda page: page.token_start)
            cursor = 0
            extents = {}
            for page in ordered:
                if page.token_start != cursor or page.token_stop <= cursor:
                    raise BackboneError("KV_CONSTRAINT", "compressed page map must cover each output token once")
                cursor = page.token_stop
                extents[page.frame_id] = extents.get(page.frame_id, 0) + page.token_stop - page.token_start
            if not ordered or self.metadata.get("position_map") is None:
                raise BackboneError("KV_CONSTRAINT", "compressed KV requires explicit RoPE/PRoPE position mapping")
            if any(extents.get(fid) != context.frame_tokens for fid in self.sink_ids + self.current_ids):
                raise BackboneError("KV_CONSTRAINT", "sink/current tokens cannot be compressed")
            if cursor > expected_tokens:
                raise BackboneError("KV_CONSTRAINT", "compressed representation exceeds exact token capacity")
            if len(self.metadata["position_map"]) != cursor:
                raise BackboneError("KV_CONSTRAINT", "position map must cover each output token")
            expected_tokens = cursor
        elif representation != "exact":
            raise BackboneError("KV_CONSTRAINT", "unknown KV representation")
        if expected_tokens > context.window_frames * context.frame_tokens:
            raise BackboneError("KV_CONSTRAINT", "native physical attention capacity exceeded")
        if (self.keys.shape != self.values.shape or self.keys.shape[1] != expected_tokens
                or self.keys.shape[0] != context.query.shape[0]
                or self.keys.shape[2:] != context.query.shape[2:]
                or self.query.shape != context.query.shape):
            raise BackboneError("KV_CONSTRAINT", "paired native Q/K/V tensor layout changed")
        for tensor in (self.query, self.keys, self.values):
            if tensor.dtype != context.query.dtype or tensor.device != context.query.device:
                raise BackboneError("KV_CONSTRAINT", "Q/K/V dtype or device changed")
        for page in self.pages:
            if page.frame_id not in flat or page.page_id < 0 or not 0 <= page.token_start < page.token_stop:
                raise BackboneError("KV_CONSTRAINT", "invalid absolute frame/page mapping")
            if page.source_token_ids and (len(page.source_token_ids) != page.token_stop - page.token_start
                                          or any(token < 0 for token in page.source_token_ids)):
                raise BackboneError("KV_CONSTRAINT", "invalid exact source-token mapping")


class AlgorithmHook(Protocol):
    """No model loading in plugins. Native FA2 is supplied by the model adapter.

    A streaming plugin must retain ProducerPlan, pinned source tensors, frontier
    and transfer events until the consumer joins. No implicit native fallback.
    RoPE/PRoPE transforms, compressed position maps and camera use are the
    algorithm's responsibility; the adapter passes native metadata unchanged.
    """

    owner: CacheOwner

    def route(self, context: QKVContext) -> AttentionSelection: ...
    def after_attention(self, context: QKVContext, selection: AttentionSelection) -> None: ...
    def consume(self, context: QKVContext, selection: AttentionSelection,
                native_fa2: Callable) -> Any: ...


def external_path(value: str, root: Path, *, resolve_symlinks: bool = True) -> Path:
    import os
    from draftkv.runtime.config import resolve_path
    resolved = resolve_path(value, root)
    if resolve_symlinks:
        return resolved
    # A venv discovers pyvenv.cfg through its invoked Python path, so retain
    # that path after the normal external-path input validation above.
    path = Path(os.path.expanduser(os.path.expandvars(value)))
    return Path(os.path.abspath(path if path.is_absolute() else root / path))
