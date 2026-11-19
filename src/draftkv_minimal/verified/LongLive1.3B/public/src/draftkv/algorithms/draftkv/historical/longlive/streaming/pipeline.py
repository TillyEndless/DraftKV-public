# SPDX-License-Identifier: Apache-2.0
"""Per-layer original C++ ProducerPlan → pinned H2D → CUDA Frontier → B01 FA2.

Only the canonical, receipt-verified vendor loader may import native extensions.
Historical IDs arrive in chronological Frame ID order from DraftKV; score ranking is selector-only. No native attention fallback.
"""
from __future__ import annotations
from draftkv.runtime import profiling as pm
from draftkv.streaming.external import external_symbol, external_path
from contextlib import nullcontext
import time
import types
import uuid
from draftkv.algorithms.draftkv.historical.longlive.backbones.failure_scope import retain_fatal_failure

class _Pages:

    def __init__(self, mode, *, quality=False):
        if not quality or mode != 'pytorch':
            raise ValueError('Only canonical Full quality host pages are supported')
        from .quality_pages import QualityPages
        self.pages = QualityPages()
        self.slab = False
        return

    def copy(self, value):
        return self.pages.copy(value)

    def close(self):
        self.pages.close()

def _check_qkv(torch, q, k, v):
    B01Error = external_symbol('longlive', 'draftkv.vendor.b01.integrity', 'B01Error')
    if q.ndim != 4 or q.shape[0] != 1 or q.shape[-1] != 128 or (q.dtype != torch.bfloat16) or (not q.is_cuda) or (k.shape != v.shape) or (k.ndim != 4) or (k.shape[0] != 1) or (k.shape[2:] != q.shape[2:]) or (min(q.shape) < 1) or (min(k.shape) < 1):
        raise B01Error('LogicError', 'B01 requires batch-one BF16 D128 native Q/K/V')
    if any((t.device != q.device or t.dtype != q.dtype or (not t.is_contiguous()) for t in (q, k, v))):
        raise B01Error('LogicError', 'same-device contiguous BF16 operands required')

def _group(tensor, groups, group):
    return tensor.reshape(tensor.shape[0], groups, group, 128).permute(1, 0, 2, 3).contiguous()

def _pack_sources(torch, rows, spec, pages):
    width = spec.heads if spec.consumer == 'token' else spec.head_group
    packed = [_group(torch.cat(column, dim=0), spec.groups, width) for column in rows]
    if pages.slab:
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
    (keys, values) = ([], [])
    for fid in spec.frame_ids:
        (k, v) = history[fid]
        if not all((t.device.type == 'cpu' and t.dtype == torch.bfloat16 and t.is_pinned() and t.is_contiguous() and (tuple(t.shape) == (spec.frame_tokens, spec.heads, 128)) for t in (k, v))):
            return None
        keys.append(k)
        values.append(v)
    return (tuple(keys), tuple(values))

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
    for (side, column) in enumerate(rows):
        for (rank, frame) in enumerate(column):
            native = frame.reshape(spec.frame_tokens, spec.groups, spec.head_group, 128)
            view = native.permute(1, 0, 2, 3)
            slab[side, :, rank * spec.frame_tokens:(rank + 1) * spec.frame_tokens].copy_(view)
    return [slab[0], slab[1]]

def run_layer(q, resident_k, resident_v, history, frame_ids, parameters=None, *, binding=None, layer=0, layers=1, sp=1, validate_kv=False, _record_golden=False, serial_reference=False, _trace_label=None, staging_cache=None):
    """History[id]=(CPU K,V), each [T,H,128], in the caller-selected RoPE basis.

    Resident [1,L,H,128] prefix/suffix are initialized. History slots are owned by
    the ProducerPlan. The call returns after BOTH native streams have completed.
    """
    B01Error = external_symbol('longlive', 'draftkv.vendor.b01.integrity', 'B01Error')
    BoundPlan = external_symbol('longlive', 'draftkv.vendor.b01', 'BoundPlan')
    consumer_module = external_symbol('longlive', 'draftkv.vendor.b01.contract', 'consumer_module')
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
    spec = binding.plan(frame_ids, frame_tokens=tokens, history_start=sink, kv_tokens=resident_k.shape[1], layer=layer, layers=layers, heads=q.shape[2])
    profile = binding.live_profile(q, resident_k, resident_v, spec, sp=sp, validation_only=_record_golden)
    started = pm.clock_ns()
    pages = None
    retained = False
    try:
        rows = [[], []]
        for fid in spec.frame_ids:
            pair = history[fid]
            if len(pair) != 2 or any((t.device.type != 'cpu' or t.dtype != torch.bfloat16 or tuple(t.shape) != (tokens, spec.heads, 128) for t in pair)):
                raise B01Error('LogicError', 'history must be paired exact CPU BF16 frame pages')
            for (column, value) in zip(rows, pair):
                column.append(value)
        width = spec.heads if spec.consumer == 'token' else spec.head_group
        direct_source = _direct_frame_sources(torch, history, spec) if p['host_pages'] == 'pytorch' else None
        if direct_source is None:
            if staging_cache is not None and p['host_pages'] == 'pytorch' and (spec.groups > 1):
                layer_cache = staging_cache.setdefault(layer, {})
                source = _stage_grouped_sources(torch, rows, spec, layer_cache)
                source_mode = 'reused_grouped_pinned_slab'
            else:
                pages = _Pages(p['host_pages'], quality=binding.host_pages_metadata['host_pages_policy'] == 'quality')
                source = _pack_sources(torch, rows, spec, pages)
                source_mode = 'packed_pinned'
        else:
            source = direct_source
            source_mode = 'direct_pinned_frames'
        destination = [_group(t[0], spec.groups, width) for t in (resident_k, resident_v)]
        frontier = torch.empty(spec.groups, dtype=torch.int32, device=q.device)
        (fetch, compute) = (torch.cuda.Stream(device=q.device), torch.cuda.Stream(device=q.device))
        output = torch.empty_like(q)
        lse = torch.empty((1, spec.heads, q.shape[1]), device=q.device, dtype=torch.float32)
        prepared = pm.clock_ns()
        plan = BoundPlan(spec, binding.cache_directory, binding.build_spec, *source, *destination, frontier, fetch, compute, outside_history_initialized=True, native_modules=binding.runtime.modules if binding.runtime else None)
        plan_end = pm.clock_ns()
        events = {key: pm.event(torch) for key in ('origin', 'h2d_begin', 'frontier_initialized', 'h2d_end', 'consumer_begin', 'consumer_end', 'join')}
        pm.record(events['origin'], torch.cuda.current_stream(q.device))
        native = plan.native
        producer_submissions = []

        def trace_range(suffix):
            return nullcontext()

        class ObservedProducer:

            def initialize(self, value):
                pm.record(events['h2d_begin'], fetch)
                begin_ns = pm.clock_ns()
                with trace_range('producer/initialize'):
                    native.initialize(value)
                producer_submissions.append(dict(operation='initialize', begin_ns=begin_ns, end_ns=pm.clock_ns()))
                pm.record(events['frontier_initialized'], fetch)

            def enqueue(self, begin, end):
                begin_ns = pm.clock_ns()
                with trace_range(f'producer/enqueue/{begin}/{end}'):
                    native.enqueue(begin, end)
                producer_submissions.append(dict(operation='enqueue', begin=begin, end=end, begin_ns=begin_ns, end_ns=pm.clock_ns()))
                pm.record(events['h2d_end'], fetch)

            def bytes(self):
                return native.bytes()
        if pm.enabled():
            plan.native = ObservedProducer()
        module_name = consumer_module(spec)
        symbol = 'streaming_fwd' if spec.consumer == 'token' else 'fwd'
        consumer = getattr(plan.modules[module_name], symbol)

        def observed_consumer(*args):
            if args[6] < args[5]:
                with trace_range('consumer/warmup_ungated'):
                    return consumer(*args)
            pm.record(events['consumer_begin'], compute)
            with trace_range('consumer/gated'):
                result = consumer(*args)
            pm.record(events['consumer_end'], compute)
            return result
        plan.modules = dict(plan.modules)
        if pm.enabled():
            plan.modules[module_name] = types.SimpleNamespace(**{symbol: observed_consumer})
        try:
            if serial_reference:
                from .comparator import run_serial_reference
                run_serial_reference(plan, q, output, lse, 128 ** (-0.5))
            else:
                plan.run(q, output, lse, 128 ** (-0.5))
            join_begin_ns = pm.clock_ns()
            plan.join()
            join_end_ns = pm.clock_ns()
        except BaseException:
            if plan.state == 'POISONED':
                _POISONED_PAGE_OWNERS.append(pages)
                retained = True
            raise
        pm.record(events['join'], torch.cuda.current_stream(q.device))
        pm.diagnostic_sync(events['join'])
        observed = pm.relative(events)
        verify_cache = external_symbol('longlive', 'draftkv.vendor.b01.cache', 'verify_cache')
        current_abi = external_symbol('longlive', 'draftkv.vendor.b01.build', 'current_abi')
        digest = external_symbol('longlive', 'draftkv.vendor.b01.integrity', 'digest')
        if binding.runtime:
            extensions = binding.runtime.hashes
        else:
            extension_paths = verify_cache(binding.cache_directory, binding.build_spec, current_abi(torch))
            extensions = {name: digest(path) for (name, path) in extension_paths.items()}
        metrics = {**binding.host_pages_metadata, 'kernel_sha': extensions[module_name], 'extension_sha256': extensions, 'producer_plan_id': f'{spec.generation}:{layer}:{uuid.uuid4().hex}', 'producer_symbol': 'producer_plan_ext.ProducerPlan', 'consumer_symbol': f'{module_name}.{symbol}', 'consumer_kind': 'b01_streaming', 'source_sha': spec.generation, 'profile': profile, 'cpu_perf_counter_ns': {'prepare_begin': started, 'plan_begin': prepared, 'plan_end': plan_end}, 'producer_submissions': producer_submissions, 'join_cpu_wait_ns': join_end_ns - join_begin_ns, 'execution_schedule': 'serial_reference' if serial_reference else 'streaming', 'cuda_relative_ms': observed, 'h2d_bytes': plan.native.bytes(), 'frontier_publications': [{'group': g, 'rank': r, 'value': v} for (g, r, v) in spec.entries()], 'frontier_publish_timestamps_ms': None, 'page_evidence': [] if pages is None else pages.pages.evidence, 'source_mode': source_mode, 'consumer_active_math_intervals': None, 'active_math_unavailable': 'CUDA event envelopes include Frontier waits; attach Nsight activity', 'true_overlap_verified': False, 'peak_gpu_memory_bytes': pm.peak_allocated(torch, q.device)} if pm.enabled() else {**binding.host_pages_metadata, 'kernel_sha': extensions[module_name], 'extension_sha256': extensions, 'source_sha': spec.generation, 'profile': profile, 'final_join': True, 'consumer_kind': 'b01_streaming'}
        if validate_kv:
            for (rank, fid) in enumerate(spec.frame_ids):
                (begin, end) = (sink + rank * tokens, sink + (rank + 1) * tokens)
                for (dst, src) in zip(destination, history[fid]):
                    copied = dst[:, begin:end].permute(1, 0, 2, 3).reshape(tokens, spec.heads, 128)
                    if not torch.equal(copied.cpu(), src):
                        raise B01Error('LogicError', 'C++ ProducerPlan changed selected full-resolution KV')
            metrics['retrieved_kv_bitwise_equal'] = True
            import hashlib

            def tensor_sha(tensor):
                return hashlib.sha256(tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
            logical = [dst.permute(1, 0, 2, 3).reshape(1, spec.kv_tokens, spec.heads, 128) for dst in destination]
            logical_source = [torch.cat(src, dim=0).unsqueeze(0) for src in source] if direct_source is not None else [src.permute(1, 0, 2, 3).reshape(1, len(spec.frame_ids) * tokens, spec.heads, 128) for src in source]
            metrics['source_sha256'] = [tensor_sha(tensor) for tensor in logical_source]
            metrics['source_pinned'] = [all((t.is_pinned() for t in frames)) if direct_source is not None else frames.is_pinned() for frames in source]
            metrics['destination_sha256'] = [tensor_sha(tensor) for tensor in logical]
            metrics['nan_survivors'] = sum((int(torch.isnan(tensor).sum().item()) for tensor in logical))
            metrics['frontier_final'] = frontier.cpu().tolist()
            metrics['frontier_stalled'] = any((value != spec.first_tile for value in metrics['frontier_final']))
            metrics['_validation_lse'] = lse
        if pm.enabled():
            pm.observe(route='b01_streaming', fetch_attention_ms=observed.get('join'), h2d_transfer_ms=observed.get('h2d_end', 0) - observed.get('h2d_begin', 0), h2d_bytes=plan.native.bytes(), receive_buffer_bytes=sum((x.numel() * x.element_size() for x in destination)), source='pipeline.run_layer CUDA event envelopes')
        metrics['final_join'] = True
        return (output, metrics)
    except BaseException as error:
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
    consume = external_symbol('longlive', 'draftkv.vendor.b01.contract', 'consume')
    consumer_module = external_symbol('longlive', 'draftkv.vendor.b01.contract', 'consumer_module')
    split_workspace = external_symbol('longlive', 'draftkv.vendor.b01.contract', 'split_workspace')
    import torch
    load_native = external_symbol('longlive', 'draftkv.vendor.b01.cache', 'load_native')
    _check_qkv(torch, q, k, v)
    spec = binding.resident_spec(frame_tokens=binding.parameters['frame_tokens'], kv_tokens=k.shape[1], layer=layer, layers=layers, heads=q.shape[2])
    profile = binding.live_profile(q, k, v, spec, sp=sp, validation_only=_record_golden)
    prepared_ns = pm.clock_ns()
    modules = binding.runtime.modules if binding.runtime else load_native(binding.cache_directory, binding.build_spec)
    width = spec.heads if spec.consumer == 'token' else spec.head_group
    grouped = [_group(t[0], spec.groups, width) for t in (k, v)]
    frontier = torch.zeros(spec.groups, dtype=torch.int32, device=q.device)
    out = torch.empty_like(q)
    lse = torch.empty((1, spec.heads, q.shape[1]), dtype=torch.float32, device=q.device)
    workspace = split_workspace(torch, spec, q) if spec.consumer == 'autotune' else ()
    name = consumer_module(spec)
    events = {key: pm.event(torch, required=key == 'join') for key in ('origin', 'consumer_begin', 'consumer_end', 'join')}
    pm.record(events['origin'], torch.cuda.current_stream(q.device))
    try:
        pm.record(events['consumer_begin'], torch.cuda.current_stream(q.device))
        consume(modules, spec, q, *grouped, frontier, 128 ** (-0.5), out, lse, workspace, first=0, last=-1)
        pm.record(events['consumer_end'], torch.cuda.current_stream(q.device))
        pm.record(events['join'], torch.cuda.current_stream(q.device))
        events['join'].synchronize()
    except BaseException:
        _POISONED_RESIDENT.append((q, k, v, grouped, frontier, out, lse, workspace, binding, modules))
        raise
    if binding.runtime:
        extensions = binding.runtime.hashes
    else:
        import json
        receipt = json.loads((binding.cache_directory / 'receipt.json').read_text())
        extensions = {name: record['sha256'] for (name, record) in receipt['binaries'].items()}
    if pm.enabled():
        pm.observe(route='b01_resident', fetch_attention_ms=pm.elapsed(events['origin'], events['join']), h2d_transfer_ms=0, h2d_bytes=0, receive_buffer_bytes=0, source='pipeline.run_resident native consumer, no historical transfer')
    return (out, {**binding.host_pages_metadata, 'final_join': True, 'producer_plan_id': None, 'consumer_kind': 'b01_streaming', 'kernel_sha': extensions[name], 'extension_sha256': extensions, 'cpu_perf_counter_ns': {'prepare_begin': prepared_ns}, 'cuda_relative_ms': pm.relative(events), 'consumer_symbol': name + ('.streaming_fwd' if spec.consumer == 'token' else '.fwd'), 'source_sha': spec.generation, 'profile': profile, 'h2d_bytes': 0, 'warmup': 'all_visible_KV_resident_frontier_disabled', 'true_overlap_verified': False} if pm.enabled() else dict(final_join=True, profile=profile, consumer_kind='b01_streaming', kernel_sha=extensions[name], extension_sha256=extensions, source_sha=spec.generation))
_POISONED_RESIDENT = []
