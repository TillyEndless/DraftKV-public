"""Per-layer original C++ ProducerPlan → pinned H2D → CUDA Frontier → B01 FA2.

Only the canonical, receipt-verified vendor loader may import native extensions.
Historical IDs arrive in chronological Frame ID order from DraftKV; score ranking is selector-only. No native attention fallback.
"""
from __future__ import annotations
from contextlib import nullcontext
import time
import types
import uuid

from draftkv.vendor.b01 import BoundPlan
from .direct_source import DirectGroupMajorSource
from draftkv.vendor.b01.contract import consume, consumer_module, split_workspace
from draftkv.vendor.b01.integrity import B01Error
from draftkv.backbones.failure_scope import retain_fatal_failure
from .config import source_bytes


class _Pages:
    def __init__(self, mode, *, quality=False):
        if quality and mode == 'pytorch':
            from .quality_pages import QualityPages
            self.pages = QualityPages()
            self.slab = False
            return
        module = types.ModuleType('b01_original_host_pages')
        exec(compile(source_bytes('bench/host_pages.py'), 'b01-original/bench/host_pages.py', 'exec'), module.__dict__)
        self.pages = module.HostPages('thp2m' if mode == 'thp2m_slab' else mode)
        self.slab = mode == 'thp2m_slab'
    def copy(self, value):
        return self.pages.copy(value)
    def close(self):
        self.pages.close()


def _check_qkv(torch, q, k, v):
    if (q.ndim != 4 or q.shape[0] != 1 or q.shape[-1] != 128
            or q.dtype != torch.bfloat16 or not q.is_cuda or k.shape != v.shape
            or k.ndim != 4 or k.shape[0] != 1 or k.shape[2:] != q.shape[2:]
            or min(q.shape) < 1 or min(k.shape) < 1):
        raise B01Error('LogicError', 'B01 requires batch-one BF16 D128 native Q/K/V')
    if any(t.device != q.device or t.dtype != q.dtype or not t.is_contiguous() for t in (q, k, v)):
        raise B01Error('LogicError', 'same-device contiguous BF16 operands required')


def _group(tensor, groups, group):
    return tensor.reshape(tensor.shape[0], groups, group, 128).permute(1, 0, 2, 3).contiguous()


def _pack_sources(torch, rows, spec, pages):
    width = spec.heads if spec.consumer == 'token' else spec.head_group
    packed = [_group(torch.cat(column, dim=0), spec.groups, width) for column in rows]
    if pages.slab:
        # Original thp2m_slab: one registered THP slab, disjoint contiguous
        # K/V subviews; the C++ transfer list still publishes each group/frame.
        slab = pages.copy(torch.cat([t.reshape(-1) for t in packed]))
        count = packed[0].numel()
        return [slab[:count].view_as(packed[0]), slab[count:].view_as(packed[1])]
    return [pages.copy(t) for t in packed]


def _direct_frame_sources(torch, history, spec):
    """Exact pinned per-frame sources for B01's original single-head-group ABI.

    No permutation, packing, new pinned allocation, or KV arithmetic. Fallback
    for unpinned or grouped layouts. Source order is the caller-selected physical Frame ID order (chronological for DraftKV).
    Each view is kept alive and checked by the original BoundPlan until join.
    """
    if spec.groups != 1:
        return None
    keys, values = [], []
    for fid in spec.frame_ids:
        k, v = history[fid]
        if (not all(t.device.type == 'cpu' and t.dtype == torch.bfloat16
                    and t.is_pinned() and t.is_contiguous()
                    and tuple(t.shape) == (spec.frame_tokens, spec.heads, 128)
                    for t in (k, v))):
            return None
        keys.append(k)
        values.append(v)
    return tuple(keys), tuple(values)


def _stage_grouped_sources(torch, rows, spec, cache):
    """Pack directly into a reusable pinned group-major slab (one CPU copy).

    Only the exact original B01 group-major layout is produced. The caller
    must join the native Producer/Consumer before this slab can be reused.
    """
    shape = (2, spec.groups, len(spec.frame_ids) * spec.frame_tokens, spec.head_group, 128)
    slab = cache.get('slab')
    if slab is None or tuple(slab.shape) != shape:
        slab = torch.empty(shape, device='cpu', dtype=torch.bfloat16, pin_memory=True)
        cache['slab'] = slab
    for side, column in enumerate(rows):
        for rank, frame in enumerate(column):
            native = frame.reshape(spec.frame_tokens, spec.groups, spec.head_group, 128)
            view = native.permute(1, 0, 2, 3)
            slab[side, :, rank * spec.frame_tokens:(rank + 1) * spec.frame_tokens].copy_(view)
    return [slab[0], slab[1]]


def _early_warmup_ticket(staging_cache, spec, binding, module_name, q, resident_k,
                         *, serial_reference=False):
    """Return cache entry only for a source/ABI/shape-matched warmed worker.

    Not a global process flag: each generation worker owns staging_cache.
    No CUDA API, event or synchronization occurs in this lookup. The cache
    is updated only AFTER the original plan.join() successfully completes.
    """
    if staging_cache is None or spec.dispatch != 'early' or serial_reference:
        return None, None, False
    warmed = staging_cache.setdefault('_b01_verified_early_warmup', set())
    binary = binding.runtime.hashes.get(module_name) if binding.runtime else None
    key = (str(q.device), spec.generation, binary, module_name,
           spec.consumer, spec.heads, spec.head_group, spec.block_m, spec.block_n,
           spec.splits, spec.gate_tiles, spec.variant,
           tuple(q.shape), tuple(resident_k.shape),
           tuple(q.stride()), tuple(resident_k.stride()))
    return warmed, key, key in warmed


def run_layer(q, resident_k, resident_v, history, frame_ids, parameters=None, *, binding=None,
              layer=0, layers=1, sp=1, validate_kv=False, _record_golden=False,
              serial_reference=False, _trace_label=None, staging_cache=None,
              direct_source=None, gpu_krope_plan=None):
    """History[id]=(CPU K,V), each [T,H,128], in the caller-selected RoPE basis.

    Resident [1,L,H,128] prefix/suffix are initialized. History slots are owned by
    the ProducerPlan. The call returns after BOTH native streams have completed.
    """
    if binding is None:
        raise B01Error('MissingDependency', 'canonical NativeBinding is required; ambient modules are forbidden')
    if parameters is not None and parameters != binding.parameters:
        raise B01Error('LogicError', 'caller parameters differ from the bound profile')
    if type(serial_reference) is not bool:
        raise B01Error('LogicError', 'serial_reference must be an explicit bool')
    if not frame_ids:
        if serial_reference:
            raise B01Error('LogicError', 'serial comparison requires nonempty real historical H2D')
        return run_resident(q, resident_k, resident_v, binding=binding, layer=layer, layers=layers, sp=sp, _record_golden=_record_golden)
    import torch
    _check_qkv(torch, q, resident_k, resident_v)
    p = binding.parameters
    tokens = p['frame_tokens']
    sink = p['sink_tokens']
    spec = binding.plan(frame_ids, frame_tokens=tokens, history_start=sink,
                        kv_tokens=resident_k.shape[1], layer=layer, layers=layers, heads=q.shape[2])
    profile = binding.live_profile(q, resident_k, resident_v, spec, sp=sp, validation_only=_record_golden)
    started = time.perf_counter_ns()
    pages = None
    retained = False
    try:
        # Normal RAG plugins retain their original CPU transfer path. Only
        # canonical DraftKV Block Top-K can supply the native final Group-major
        # buffer directly, eliminating its former whole-history CPU ck/cv and
        # the subsequent regrouping copy. No per-attention packed CPU duplicate.
        prepacked = None
        width = spec.heads if spec.consumer == 'token' else spec.head_group
        if direct_source is not None:
            if history is not None or not isinstance(direct_source, DirectGroupMajorSource):
                raise B01Error('LogicError', 'direct Group-major input requires exclusive verified ownership')
            source = direct_source.verify(spec)
            source_mode = ('direct_groupmajor_gpu_krope_8x16'
                           if gpu_krope_plan is not None else 'direct_groupmajor_pinned')
            prepacked = True
        else:
            rows = [[], []]
            for fid in spec.frame_ids:
                pair = history[fid]
                if len(pair) != 2 or any(t.device.type != 'cpu' or t.dtype != torch.bfloat16
                        or tuple(t.shape) != (tokens, spec.heads, 128) for t in pair):
                    raise B01Error('LogicError', 'history must be paired exact CPU BF16 frame pages')
                for column, value in zip(rows, pair):
                    column.append(value)
            width = spec.heads if spec.consumer == 'token' else spec.head_group
            direct_frames = (_direct_frame_sources(torch, history, spec)
                             if p['host_pages'] == 'pytorch' else None)
            if direct_frames is None:
                # Original system THP/NUMA sources remain unchanged.
                if (staging_cache is not None and p['host_pages'] == 'pytorch'
                        and spec.groups > 1):
                    layer_cache = staging_cache.setdefault(layer, {})
                    source = _stage_grouped_sources(torch, rows, spec, layer_cache)
                    source_mode = 'reused_grouped_pinned_slab'
                else:
                    pages = _Pages(p['host_pages'], quality=binding.host_pages_metadata['host_pages_policy'] == 'quality')
                    source = _pack_sources(torch, rows, spec, pages)
                    source_mode = 'packed_pinned'
            else:
                source = direct_frames
                source_mode = 'direct_pinned_frames'
        destination = [_group(t[0], spec.groups, width) for t in (resident_k, resident_v)]
        if gpu_krope_plan is not None:
            if direct_source is None or serial_reference:
                raise B01Error('LogicError',
                               'GPU K-RoPE requires direct 8x16 source and Streaming FA2')
            # Real CUDA module loading MUST finish before a waiting FA2
            # consumer can be enqueued; do not lazily import/compile kernels
            # after the Frontier wait has begun.
            gpu_krope_plan.validate(direct_source, spec, q.device)
            gpu_krope_plan.preload(destination[0], spec)
        frontier = torch.empty(spec.groups, dtype=torch.int32, device=q.device)
        fetch, compute = torch.cuda.Stream(device=q.device), torch.cuda.Stream(device=q.device)
        output = torch.empty_like(q)
        lse = torch.empty((1, spec.heads, q.shape[1]), device=q.device, dtype=torch.float32)
        prepared = time.perf_counter_ns()
        plan = BoundPlan(spec, binding.cache_directory, binding.build_spec, *source, *destination,
                         frontier, fetch, compute, outside_history_initialized=True,
                         native_modules=binding.runtime.modules if binding.runtime else None)
        plan_end = time.perf_counter_ns()
        # Events are observations on the ACTUAL native streams. They enclose
        # kernel/DMA activity and include waits; no SM-active overlap is inferred.
        events = {key: torch.cuda.Event(enable_timing=True) for key in
                  ('origin', 'h2d_begin', 'frontier_initialized', 'h2d_end', 'consumer_begin', 'consumer_end', 'join')}
        events['origin'].record(torch.cuda.current_stream(q.device))
        native = plan.native
        if gpu_krope_plan is not None:
            from .rotating_producer import RotatingProducer
            native = RotatingProducer(native, spec, source, destination,
                                     frontier, plan.modules, gpu_krope_plan)
        producer_submissions = []
        def trace_range(suffix):
            return torch.profiler.record_function(_trace_label + '/' + suffix) if _trace_label else nullcontext()
        class ObservedProducer:
            def initialize(self, value):
                events['h2d_begin'].record(fetch)
                begin_ns = time.perf_counter_ns()
                with trace_range('producer/initialize'):
                    native.initialize(value)
                producer_submissions.append(dict(operation='initialize', begin_ns=begin_ns, end_ns=time.perf_counter_ns()))
                events['frontier_initialized'].record(fetch)
            def enqueue(self, begin, end):
                begin_ns = time.perf_counter_ns()
                with trace_range(f'producer/enqueue/{begin}/{end}'):
                    native.enqueue(begin, end)
                producer_submissions.append(dict(operation='enqueue', begin=begin, end=end,
                    begin_ns=begin_ns, end_ns=time.perf_counter_ns()))
                events['h2d_end'].record(fetch)
            def bytes(self):
                return native.bytes()
        plan.native = ObservedProducer()
        module_name = consumer_module(spec)
        symbol = 'streaming_fwd' if spec.consumer == 'token' else 'fwd'
        consumer = getattr(plan.modules[module_name], symbol)
        def observed_consumer(*args):
            # The original early warmup runs on the caller stream. Do not
            # label its discarded ungated launch as the measured gated call.
            if args[6] < args[5]:
                with trace_range('consumer/warmup_ungated'):
                    return consumer(*args)
            events['consumer_begin'].record(compute)
            with trace_range('consumer/gated'):
                result = consumer(*args)
            events['consumer_end'].record(compute)
            return result
        plan.modules = dict(plan.modules)
        plan.modules[module_name] = types.SimpleNamespace(**{symbol: observed_consumer})
        # A successful first native EARLY invocation has already initialized
        # the consumer kernel. Do not redo its synchronizing ungated warmup on
        # every layer. This is worker-scoped, opt-in through the existing
        # reusable staging_cache; test/microbench calls without a cache keep
        # exact historical behavior. All producer/frontier/consumer ordering
        # and native CUDA code remain unchanged.
        warmed, warm_key, skip_ungated_warmup = _early_warmup_ticket(
            staging_cache, spec, binding, module_name, q, resident_k,
            serial_reference=serial_reference)
        try:
            if serial_reference:
                from .comparator import run_serial_reference
                run_serial_reference(plan, q, output, lse, 128 ** -.5)
            else:
                plan.run(q, output, lse, 128 ** -.5,
                         skip_ungated_warmup=skip_ungated_warmup)
            join_begin_ns = time.perf_counter_ns()
            plan.join()
            # Only a completed/joined original run makes the next call safe.
            if warmed is not None and warm_key is not None:
                warmed.add(warm_key)
            join_end_ns = time.perf_counter_ns()
        except BaseException:
            # BoundPlan quarantines native storage. HostPages owners must ALSO
            # survive worker exit when mmap-backed pinned allocations are live.
            if plan.state == 'POISONED':
                _POISONED_PAGE_OWNERS.append(pages)
                retained = True
            raise
        events['join'].record(torch.cuda.current_stream(q.device)); events['join'].synchronize()
        observed = {key: events['origin'].elapsed_time(events[key]) for key in events if key != 'origin'}
        from draftkv.vendor.b01.cache import verify_cache
        from draftkv.vendor.b01.build import current_abi
        from draftkv.vendor.b01.integrity import digest
        if binding.runtime:
            extensions = binding.runtime.hashes
        else:
            extension_paths = verify_cache(binding.cache_directory, binding.build_spec, current_abi(torch))
            extensions = {name: digest(path) for name, path in extension_paths.items()}
        metrics = {**binding.host_pages_metadata,
                   'kernel_sha': extensions[module_name], 'extension_sha256': extensions,
                   'producer_plan_id': f'{spec.generation}:{layer}:{uuid.uuid4().hex}',
                   'producer_symbol': 'producer_plan_ext.ProducerPlan',
                   'consumer_symbol': f'{module_name}.{symbol}', 'consumer_kind': 'b01_streaming',
                   'source_sha': spec.generation, 'profile': profile,
                   'cpu_perf_counter_ns': {'prepare_begin': started, 'plan_begin': prepared, 'plan_end': plan_end},
                   'producer_submissions': producer_submissions,
                   'join_cpu_wait_ns': join_end_ns - join_begin_ns,
                   'execution_schedule': 'serial_reference' if serial_reference else 'streaming',
                   'skipped_repeated_early_warmup': bool(skip_ungated_warmup),
                   'cuda_relative_ms': observed, 'h2d_bytes': plan.native.bytes(),
                   'frontier_publications': [{'group': g, 'rank': r, 'value': v} for g, r, v in spec.entries()],
                   'frontier_publish_timestamps_ms': None,
                   'page_evidence': [] if pages is None else pages.pages.evidence,
                   'source_mode': source_mode,
                   'gpu_krope_8x16': gpu_krope_plan is not None,
                   'gpu_temporal_rotated_pages': (
                       sum(gpu_krope_plan.rotated_ranks) * spec.groups
                       if gpu_krope_plan is not None else 0),
                   'consumer_active_math_intervals': None,
                   'active_math_unavailable': 'CUDA event envelopes include Frontier waits; attach Nsight activity',
                   'true_overlap_verified': False, 'peak_gpu_memory_bytes': torch.cuda.max_memory_allocated(q.device)}
        if validate_kv:
            for rank, fid in enumerate(spec.frame_ids):
                begin, end = sink + rank * tokens, sink + (rank + 1) * tokens
                if prepacked:
                    # Debug-only exact reference. When GPU K-RoPE is enabled
                    # the CPU source holds raw temporal K for shifted tokens;
                    # Frontier-protected GPU destination is the rotated value.
                    for side,(dst,src) in enumerate(zip(destination, source)):
                        for group in range(spec.groups):
                            picked = dst[group,begin:end].cpu()
                            original = src[group,rank*tokens:(rank+1)*tokens]
                            if gpu_krope_plan is not None and side == 0:
                                from .krope_8x16 import cpu_reference_for_validation
                                expected=cpu_reference_for_validation(
                                    gpu_krope_plan,original,rank=rank)
                                ulps=(picked.view(torch.int16).int()
                                      -expected.view(torch.int16).int()).abs()
                                # +0.0 and -0.0 are numerically identical;
                                # their BF16 bit patterns differ by 2**15.
                                ulps.masked_fill_((picked == 0) & (expected == 0),0)
                                if int(ulps.max()) > 1:
                                    delta=(picked.float()-expected.float()).abs()
                                    idx=(ulps>1).nonzero()[:3].tolist()
                                    raise B01Error('LogicError',
                                       'GPU K-RoPE CPU reference differs: ' +
                                       f'max_ulps={int(ulps.max())} max_abs={float(delta.max())} ' +
                                       f'indices={idx} actual={[float(picked[tuple(i)]) for i in idx]} ' +
                                       f'expected={[float(expected[tuple(i)]) for i in idx]}')
                                metrics['gpu_krope_cpu_max_ulp']=max(
                                    metrics.get('gpu_krope_cpu_max_ulp',0),int(ulps.max()))
                            elif not torch.equal(picked, original):
                                raise B01Error('LogicError',
                                    'B01 ProducerPlan changed direct packed K/V')
                else:
                    for dst, src in zip(destination, history[fid]):
                        copied = dst[:, begin:end].permute(1, 0, 2, 3).reshape(tokens, spec.heads, 128)
                        if not torch.equal(copied.cpu(), src):
                            raise B01Error('LogicError', 'C++ ProducerPlan changed selected full-resolution KV')
            metrics['retrieved_kv_bitwise_equal'] = (
                gpu_krope_plan is None or metrics.get('gpu_krope_cpu_max_ulp',0)==0)
            if gpu_krope_plan is not None:
                metrics['retrieved_kv_within_one_bf16_ulp']=True
            import hashlib
            def tensor_sha(tensor):
                return hashlib.sha256(tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
            logical = [dst.permute(1, 0, 2, 3).reshape(1, spec.kv_tokens, spec.heads, 128) for dst in destination]
            logical_source = ([torch.cat(src, dim=0).unsqueeze(0) for src in source]
                              if source_mode == 'direct_pinned_frames' else
                              [src.permute(1, 0, 2, 3).reshape(1, len(spec.frame_ids) * tokens, spec.heads, 128)
                               for src in source])
            metrics['source_sha256'] = [tensor_sha(tensor) for tensor in logical_source]
            metrics['source_pinned'] = [all(t.is_pinned() for t in frames)
                                        if source_mode == 'direct_pinned_frames'
                                        else frames.is_pinned() for frames in source]
            metrics['destination_sha256'] = [tensor_sha(tensor) for tensor in logical]
            metrics['nan_survivors'] = sum(int(torch.isnan(tensor).sum().item()) for tensor in logical)
            metrics['frontier_final'] = frontier.cpu().tolist()
            metrics['frontier_stalled'] = any(value != spec.first_tile for value in metrics['frontier_final'])
            metrics['_validation_lse'] = lse
        metrics['final_join'] = True
        return output, metrics
    except BaseException as error:
        # Packing can fail before BoundPlan exists or marks itself poisoned.
        # A fatal CLI must retain pinned owners rather than synchronize/free
        # them while another rank or native stream is still waiting.
        if retain_fatal_failure(error, pages):
            retained = True
        raise
    finally:
        if not retained and pages is not None:
            pages.close()


_POISONED_PAGE_OWNERS = []


def run_resident(q, k, v, *, binding, layer=0, layers=1, sp=1, _record_golden=False):
    """Zero-history warmup uses the SAME original Streaming Consumer ungated.

    Original binding explicitly disables its Frontier when last_tile < first.
    It does not call NativeFA2, SDPA, or a Python attention implementation.
    """
    import torch
    from draftkv.vendor.b01.cache import load_native
    _check_qkv(torch, q, k, v)
    spec = binding.resident_spec(frame_tokens=binding.parameters['frame_tokens'], kv_tokens=k.shape[1],
                                 layer=layer, layers=layers, heads=q.shape[2])
    profile = binding.live_profile(q, k, v, spec, sp=sp, validation_only=_record_golden)
    prepared_ns = time.perf_counter_ns()
    modules = binding.runtime.modules if binding.runtime else load_native(binding.cache_directory, binding.build_spec)
    width = spec.heads if spec.consumer == 'token' else spec.head_group
    grouped = [_group(t[0], spec.groups, width) for t in (k, v)]
    frontier = torch.zeros(spec.groups, dtype=torch.int32, device=q.device)
    out = torch.empty_like(q); lse = torch.empty((1, spec.heads, q.shape[1]), dtype=torch.float32, device=q.device)
    workspace = split_workspace(torch, spec, q) if spec.consumer == 'autotune' else ()
    name = consumer_module(spec)
    events = {key: torch.cuda.Event(enable_timing=True) for key in ('origin', 'consumer_begin', 'consumer_end', 'join')}
    events['origin'].record(torch.cuda.current_stream(q.device))
    try:
        events['consumer_begin'].record(torch.cuda.current_stream(q.device))
        consume(modules, spec, q, *grouped, frontier, 128 ** -.5, out, lse,
                workspace, first=0, last=-1)
        events['consumer_end'].record(torch.cuda.current_stream(q.device))
        events['join'].record(torch.cuda.current_stream(q.device))
        # Caller stream owns all resident input/output tensors until completion.
        events['join'].synchronize()
    except BaseException:
        # A binding/async error can follow a partial launch. Keep every owner
        # alive through worker exit; never recycle or attempt another consumer.
        _POISONED_RESIDENT.append((q, k, v, grouped, frontier, out, lse, workspace, binding, modules))
        raise
    if binding.runtime:
        extensions = binding.runtime.hashes
    else:
        import json
        receipt = json.loads((binding.cache_directory / 'receipt.json').read_text())
        extensions = {name: record['sha256'] for name, record in receipt['binaries'].items()}
    return out, {**binding.host_pages_metadata, 'final_join': True,
                 'producer_plan_id': None, 'consumer_kind': 'b01_streaming',
                 'kernel_sha': extensions[name], 'extension_sha256': extensions,
                 'cpu_perf_counter_ns': {'prepare_begin': prepared_ns},
                 'cuda_relative_ms': {key: events['origin'].elapsed_time(events[key]) for key in events if key != 'origin'},
                 'consumer_symbol': name + ('.streaming_fwd' if spec.consumer == 'token' else '.fwd'),
                 'source_sha': spec.generation, 'profile': profile, 'h2d_bytes': 0,
                 'warmup': 'all_visible_KV_resident_frontier_disabled', 'true_overlap_verified': False}


_POISONED_RESIDENT = []
