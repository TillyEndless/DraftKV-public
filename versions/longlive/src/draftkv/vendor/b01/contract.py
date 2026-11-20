"""CPU planning and a thin dispatcher to immutable native Producer/Consumer.

No Python DMA, attention, frame selection, ring buffer, or fallback implementation.
"""

from dataclasses import dataclass
import math
import re

from .build import ARCHES
from .integrity import B01Error, source_generation

# A failed native submission can still own live asynchronous GPU work. Retain
# buffers even if a caller drops its reference after an exception. Worker exit
# is the recovery boundary; no poisoned-context reuse or silent fallback.
_POISONED_PLANS = []
_LIVE_PLANS = {}


def require(condition, message):
    if not condition:
        raise B01Error("LogicError", message)


@dataclass(frozen=True)
class PlanSpec:
    # IDs are in caller-selected packed order, NOT sorted by this vendor.
    frame_ids: tuple
    frame_tokens: int
    history_start: int
    kv_tokens: int
    layer: int
    layers: int
    arch: str
    generation: str
    clean_commit: str
    consumer: str = "token"
    heads: int = 12
    head_group: int = 4
    gate_tiles: int = 4
    variant: int = 9
    block_m: int = 64
    tile_n: int = 64
    splits: int = 1
    layout: str = "grouped"
    dispatch: str = "all"
    host_pages: str = "pytorch"
    preverified_source: bool = False  # Only from a verified worker-owned session.

    def __post_init__(self):
        require(type(self.frame_ids) is tuple and bool(self.frame_ids), "ordered nonempty tuple of absolute Frame IDs required")
        require(all(type(i) is int and i >= 0 for i in self.frame_ids) and len(set(self.frame_ids)) == len(self.frame_ids), "absolute Frame IDs must be unique nonnegative integers")
        for name in ("frame_tokens", "history_start", "kv_tokens", "layer", "layers", "heads", "head_group", "gate_tiles", "variant", "block_m", "tile_n", "splits"):
            require(type(getattr(self, name)) is int, f"{name} must be an integer")
        require(self.frame_tokens > 0 and self.history_start >= 0 and self.history_end <= self.kv_tokens, "history budget exceeds stable storage")
        require(0 <= self.layer < self.layers and 0 < self.heads < 2**31, "invalid layer count/index or local heads")
        require(self.kv_tokens < 2**31, "native int32 token/tile range exceeded")
        if not isinstance(self.arch, str) or self.arch not in ARCHES:
            raise B01Error("UnsupportedArchitecture", f"unknown arch: {self.arch}")
        require(self.consumer in ("token", "ready", "frontier_opt", "autotune"), "unknown original Consumer")
        require(self.layout == "grouped", "only original group-major layout is connected")
        require(self.dispatch in ("all", "early"), "unsupported producer dispatch")
        require(self.host_pages in ("pytorch", "base4k", "thp2m", "thp2m_slab"), "unknown host page policy")
        if self.preverified_source:
            # Full source verification occurred before the worker registered its
            # native modules. Never re-hash archived members per Attention.
            from .integrity import MANIFEST_SHA256
            require(self.generation == MANIFEST_SHA256,
                    "verified worker source generation changed")
        else:
            require(self.generation == source_generation(),
                    "Source Generation differs from reviewed source bundle")
        require(isinstance(self.clean_commit, str) and bool(re.fullmatch(r"[0-9a-f]{40}", self.clean_commit)), "full clean release commit required")
        require(0 < self.gate_tiles < 2**31 and self.gate_tiles & (self.gate_tiles - 1) == 0, "gate_tiles must be a positive int32 power of two")
        if self.consumer in ("ready", "frontier_opt"):
            require(self.heads == 12 and self.head_group in (1, 2, 3, 4, 6, 12), "group-major requires exactly 12 local heads and valid head_group")
        if self.consumer == "autotune":
            require(self.head_group > 0 and self.heads % self.head_group == 0, "group must divide actual local heads")
            require(self.block_m in (64, 128) and self.tile_n in (32, 64, 128), "tile not compiled by original Autotune provider")
            require(1 <= self.splits <= 16 and (self.splits == 1 or self.block_m == 64), "original SplitKV requires M64 and splits 1..16")
        else:
            require(self.block_m == 64 and self.splits == 1, "legacy bridge supports M64/S1 only")
        if self.consumer == "frontier_opt":
            require(self.variant in (0, 1, 2, 3, 4, 5, 6, 9), "unsupported original frontier variant")

    @property
    def history_end(self):
        return self.history_start + len(self.frame_ids) * self.frame_tokens

    @property
    def block_n(self):
        if self.consumer == "autotune":
            return self.tile_n
        if self.consumer == "token":
            return 32 if self.arch == "8.9" else 64
        return 64 if self.consumer == "frontier_opt" and self.variant in (4, 9) else 32

    @property
    def groups(self):
        return 1 if self.consumer == "token" else self.heads // self.head_group

    @property
    def first_tile(self):
        return self.history_start // self.block_n

    @property
    def last_tile(self):
        return (self.history_end - 1) // self.block_n

    def entries(self):
        """(head-group, packed-frame-rank, ready-value), descending contiguous suffix.

        At a partial frame boundary, ceil avoids releasing the still-incomplete
        crossing tile. The final copy can release the tile touching the already
        initialized prefix. Ungated prefix/suffix must be initialized beforehand.
        """
        entries = []
        for group in range(self.groups):
            for rank in reversed(range(len(self.frame_ids))):
                start = self.history_start + rank * self.frame_tokens
                value = (start + self.block_n - 1) // self.block_n if rank else self.first_tile
                entries.append((group, rank, value))
        return tuple(entries)


def _identity(tensor):
    return (tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()), str(tensor.device),
            str(tensor.dtype), tensor._version)


def disjoint(tensors):
    # K/V may be disjoint views of the ORIGINAL registered slab. Comparing
    # storage pointers alone incorrectly rejects that allocator's valid ABI.
    intervals = [(t.data_ptr(), t.data_ptr() + math.prod(t.shape) *
                  (4 if str(t.dtype) in ("torch.int32", "torch.float32", "int32", "fp32") else 2))
                 for t in tensors]
    # Sorting exact contiguous memory intervals is equivalent to the original
    # all-pairs check and avoids O(R^2) Python comparisons for direct frames.
    intervals.sort()
    return all(end <= nxt_start
               for (_, end), (nxt_start, _) in zip(intervals, intervals[1:]))


def consumer_module(spec):
    return {"token": "fa2_streaming_consumer_ext", "ready": "b01_ready_ext",
            "frontier_opt": "b01_frontier_opt_ext", "autotune": "offload_fa2_ext"}[spec.consumer]


def split_workspace(torch, spec, q):
    """Original binding ABI: FP32 [S,1,H,Q,D] and [S,1,H,Q]."""
    shape = (spec.splits, 1, spec.heads, q.shape[1])
    return (torch.empty((*shape, 128) if spec.splits > 1 else (0,), device=q.device, dtype=torch.float32),
            torch.empty(shape if spec.splits > 1 else (0,), device=q.device, dtype=torch.float32))


def consume(modules, spec, q, k, v, frontier, scale, out, lse, workspace, *, first, last):
    args = (q, k, v, frontier, scale, first, last, spec.gate_tiles, out, lse)
    module = modules[consumer_module(spec)]
    if spec.consumer == "token":
        return module.streaming_fwd(*args)
    if spec.consumer == "ready":
        return module.fwd(*args, spec.head_group)
    if spec.consumer == "frontier_opt":
        return module.fwd(*args, spec.head_group, spec.variant)
    return module.fwd(*args, spec.head_group, spec.block_m, spec.block_n, spec.splits, *workspace)


class BoundPlan:
    """Single-use native plan. References survive until both streams complete.

    A poisoned plan must be retained until the owning process/context terminates;
    do not free/recycle its buffers or attempt fallback with a waiting Consumer.
    """

    def __init__(self, spec, cache_directory, build_spec, src_k, src_v, dst_k, dst_v,
                 frontier, h2d_stream, consumer_stream, *, outside_history_initialized=False, native_modules=None):
        from .cache import load_native
        try:
            import torch
        except ImportError as exc:
            raise B01Error("MissingDependency", "Torch required only for explicit native binding") from exc
        self.torch = torch
        self.spec = spec
        require(spec.generation == build_spec["source_generation"] and spec.clean_commit == build_spec["clean_commit"] and spec.arch == build_spec["arch"], "build/runtime generation, commit or arch mismatch")
        require(outside_history_initialized is True, "caller must initialize ungated KV prefix/suffix before binding")
        # A verified worker binding owns native modules throughout this process.
        # This is not a persistent-path cache; the worker itself must be
        # recreated to rebind another compiled source/ABI generation.
        self.modules = native_modules if native_modules is not None else load_native(cache_directory, build_spec)
        # Original C++ ProducerPlan accepts independently pinned contiguous
        # frame tensors. For the token consumer (one group), bind those pages
        # directly instead of forcing a new packed pinned staging allocation.
        # Group-major consumers retain the established packed layout.
        self.direct_frames = isinstance(src_k, (tuple, list))
        if self.direct_frames:
            require(spec.groups == 1 and isinstance(src_v, (tuple, list)),
                    "direct frame sources require the original one-group token consumer")
            require(len(src_k) == len(src_v) == len(spec.frame_ids),
                    "direct frame count differs from selected frame IDs")
            src_k, src_v = tuple(src_k), tuple(src_v)
        self.sources = (src_k, src_v)
        self._source_tensors = (*src_k, *src_v) if self.direct_frames else (src_k, src_v)
        self.targets = (dst_k, dst_v, frontier)
        self.streams = (h2d_stream, consumer_stream)
        self.state = "BOUND"
        self._retained = []
        spec_group_heads = spec.heads if spec.consumer == "token" else spec.head_group
        expected_src = (spec.groups, len(spec.frame_ids) * spec.frame_tokens, spec_group_heads, 128)
        expected_dst = (spec.groups, spec.kv_tokens, spec_group_heads, 128)
        for tensor in self._source_tensors:
            require(tensor.device.type == "cpu" and tensor.is_pinned(), "pinned CPU K/V sources required")
            shape = (spec.frame_tokens, spec.heads, 128) if self.direct_frames else expected_src
            require(tensor.dtype == torch.bfloat16 and tensor.is_contiguous() and tuple(tensor.shape) == shape,
                    "invalid pinned source shape/dtype/layout")
        for tensor in (dst_k, dst_v):
            require(tensor.is_cuda and tensor.dtype == torch.bfloat16 and tensor.is_contiguous() and tuple(tensor.shape) == expected_dst, "invalid stable destination shape/dtype/layout")
        require(frontier.is_cuda and frontier.dtype == torch.int32 and frontier.is_contiguous() and tuple(frontier.shape) == (spec.groups,), "invalid frontier")
        require(all(t.device == frontier.device for t in self.targets), "all destination tensors must share logical CUDA device")
        require(all(stream.device == frontier.device for stream in self.streams), "explicit streams must match logical destination device")
        require(h2d_stream.cuda_stream != consumer_stream.cuda_stream, "two explicit streams required")
        major, minor = torch.cuda.get_device_capability(frontier.device)
        require(f"{major}.{minor}" == spec.arch, "actual device architecture differs from compiled plan")
        # Avoid aliased K/V/frontier storage. Full KV allocations must remain stable.
        require(disjoint(self._source_tensors) and disjoint(self.targets),
                "K/V/frontier storage aliasing unsupported")
        self.identities = tuple(_identity(t) for t in (*self._source_tensors, *self.targets))
        dk, dv, sk, sv, indices, values = [], [], [], [], [], []
        for group, rank, value in spec.entries():
            a = rank * spec.frame_tokens
            b = a + spec.frame_tokens
            dk.append(dst_k[group, spec.history_start + a:spec.history_start + b])
            dv.append(dst_v[group, spec.history_start + a:spec.history_start + b])
            sk.append(src_k[rank] if self.direct_frames else src_k[group, a:b])
            sv.append(src_v[rank] if self.direct_frames else src_v[group, a:b])
            indices.append(group)
            values.append(value)
        self.native = self.modules["producer_plan_ext"].ProducerPlan(dk, dv, sk, sv, frontier, indices, values)
        self.entry_count = len(indices)

    def run(self, q, out, lse, scale, *, skip_ungated_warmup=False):
        """Original all/early submission order, retaining both streams until join.

        Early warms the ungated consumer before partial submission (original
        Runner rule for CUDA lazy loading). The optional skip requires that the
        same executable/shape has already completed a successful native warmup.
        Failed early work poisons the worker;
        it must never recycle storage or fall back to another attention provider.
        """
        require(type(skip_ungated_warmup) is bool, "skip_ungated_warmup must be boolean")
        require(self.state == "BOUND", "plan must be newly bound and unpoisoned")
        require(tuple(_identity(t) for t in (*self._source_tensors, *self.targets)) == self.identities,
                "bound storage/source generation changed")
        t = self.torch
        s = self.spec
        device = self.targets[-1].device
        require(q.is_cuda and q.device == device and q.dtype == t.bfloat16 and q.is_contiguous(), "invalid Q")
        require(q.ndim == 4 and tuple(q.shape[::2]) == (1, s.heads) and 0 < q.shape[1] < 2**31 and q.shape[3] == 128, "Q must be [1,Lq,local_heads,128] within int32 range")
        require(out.device == device and out.dtype == t.bfloat16 and tuple(out.shape) == tuple(q.shape) and out.is_contiguous(), "invalid output")
        require(lse.device == device and lse.dtype == t.float32 and tuple(lse.shape) == (1, s.heads, q.shape[1]) and lse.is_contiguous(), "invalid LSE")
        require(type(scale) in (int, float) and 1.401298464324817e-45 <= scale <= 3.4028234663852886e38 and math.isfinite(scale), "softmax scale must be positive finite float32")
        all_tensors = (*self.sources, *self.targets, q, out, lse)
        require(disjoint((*self.targets, q, out, lse)), "Q/output/LSE cannot alias live buffers")
        self._retained.extend((q, out, lse))
        self.state = "SUBMITTING"
        _LIVE_PLANS[id(self)] = self
        try:
            workspace = split_workspace(t, s, q) if s.consumer == "autotune" else ()
            self._retained.extend(workspace)
            def forward(first, last):
                return consume(self.modules, s, q, *self.targets[:2], self.targets[2],
                               scale, out, lse, workspace, first=first, last=last)
            if s.dispatch == "early" and not skip_ungated_warmup:
                # Discarded warmup, exactly as original Runner._forward(False).
                # Loads compute AND merge kernels before a waiting launch. The
                # final output/LSE are always overwritten by the gated call.
                forward(0, -1)
                warm_done = t.cuda.Event()
                warm_done.record(t.cuda.current_stream(device))
                warm_done.synchronize()
            self.initialized = t.cuda.Event()
            self.producer_done = t.cuda.Event()
            self.consumer_done = t.cuda.Event()
            # Fence any caller preparation/ungated initialization on current stream.
            with t.cuda.device(device):
                prepared = t.cuda.Event()
                prepared.record(t.cuda.current_stream(device))
                with t.cuda.stream(self.streams[0]):
                    self.streams[0].wait_event(prepared)
                    self.native.initialize(s.last_tile + 1)
                    self.initialized.record(self.streams[0])
                    self.native.enqueue(0, 1 if s.dispatch == "early" else self.entry_count)
                    if s.dispatch == "all":
                        self.producer_done.record(self.streams[0])
                with t.cuda.stream(self.streams[1]):
                    self.streams[1].wait_event(self.initialized)
                    result = forward(s.first_tile, s.last_tile)
                    self.consumer_done.record(self.streams[1])
                if s.dispatch == "early":
                    with t.cuda.stream(self.streams[0]):
                        self.native.enqueue(1, self.entry_count)
                        self.producer_done.record(self.streams[0])
            self.state = "IN_FLIGHT"
            return result
        except Exception:
            self._poison()
            raise

    def _poison(self):
        self.state = "POISONED"
        _LIVE_PLANS.pop(id(self), None)
        _POISONED_PLANS.append(self)

    def join(self):
        require(self.state == "IN_FLIGHT", "only successfully submitted plan can join")
        try:
            self.producer_done.synchronize()
            self.consumer_done.synchronize()
            self.state = "COMPLETE"
            _LIVE_PLANS.pop(id(self), None)
        except Exception:
            self._poison()
            raise
