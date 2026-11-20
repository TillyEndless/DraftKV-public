"""Chrono-Compact attention coordinates; retrieval stays native post-RoPE.

Keep the native K/V bank for scoring/provenance. Clean commits additionally own
raw temporal K components, so reads never invert a rounded rotated key. Only
selected K pages are materialized, on pinned CPU memory, for original B01 DMA.
"""
from concurrent.futures import ThreadPoolExecutor
from draftkv.backbones.contracts import BackboneError

POSITION_POLICY = 'chrono_compact'
PACKING_POLICY = 'chronological_frame_id'


def require_packing_policy(settings):
    policy = settings.get('packing_policy', PACKING_POLICY)
    if policy != PACKING_POLICY:
        raise BackboneError('BLOCKED_ALGORITHM_SEMANTICS',
                            'canonical DraftKV requires chronological_frame_id packing')
    return policy


def require_policy(settings):
    policy = settings.get('position_policy', POSITION_POLICY)
    if policy != POSITION_POLICY and not ((settings.get('gpu_prepack',False) or settings.get('position_reference_a',False)) and policy=='native_post_rope'):
        raise BackboneError('BLOCKED_ALGORITHM_SEMANTICS',
                            'canonical DraftKV requires position_policy=chrono_compact')
    return policy


def compact_map(sink, selected, current):
    # Fail closed rather than silently deriving a different chronological
    # mapping from score-ranked physical K/V pages.
    ordered = tuple(sink) + tuple(selected) + tuple(current)
    if (not current or any(type(i) is not int or i < 0 for i in ordered)
            or len(set(ordered)) != len(ordered) or tuple(sorted(ordered)) != ordered):
        raise BackboneError('KV_CONSTRAINT', 'Chrono-Compact requires unique chronological source IDs')
    return {fid: slot for slot, fid in enumerate(ordered)}


class WanTime:
    dimensions = 44
    layout = 'wan_complex_t0:44'

    def __init__(self, context):
        import torch
        freqs = context.freqs
        if (not isinstance(freqs, torch.Tensor) or freqs.ndim != 2
                or freqs.shape[1] != 64 or not freqs.is_complex()):
            raise BackboneError('KV_CONSTRAINT', 'native Wan D128 complex frequency table required')
        self.coefficients = (freqs[:context.window_frames, :22].detach(),)
        self.prefix = context.fixed_prefix
        if self.prefix is not None:
            raise BackboneError('KV_CONSTRAINT', 'unexpected fixed prefix on Wan DraftKV')

    def extract(self, raw):
        return raw[..., :44]

    def assign(self, target, temporal):
        target[..., :44] = temporal

    def rotate(self, raw, slot, dtype, *, cpu=None):
        import torch
        phase = (cpu or self.coefficients)[0][slot].to(raw.device)
        value = torch.view_as_complex(raw.to(torch.float64).reshape(*raw.shape[:-1], 22, 2))
        # Native causal_rope_apply casts back to raw dtype, then to V dtype.
        return torch.view_as_real(value * phase).flatten(-2).to(raw.dtype).to(dtype)


class H3Time:
    dimensions = 32
    layout = 'h3_rotate_half_t0:16+48:64'

    def __init__(self, context):
        geometry = context.native_geometry
        if (geometry is None or geometry.rope.rope_freq_dim != 16
                or geometry.rows != context.frame_tokens):
            raise BackboneError('KV_CONSTRAINT', 'native H3 MM-RoPE geometry required')
        positions = geometry.positions[::geometry.rows][:context.window_frames]
        cos, sin = geometry.rope(positions)
        self.coefficients = (cos[:, :16].detach(), sin[:, :16].detach())
        self.prefix = getattr(geometry, 'chrono_compact_prefix', None)
        if context.fixed_prefix is not None and self.prefix is None:
            raise BackboneError('KV_CONSTRAINT', 'H3 Chrono-Compact requires raw-derived unshifted prefix')
        if self.prefix is not None:
            if (context.fixed_prefix is None or len(self.prefix) != 2
                    or any(a.shape != b.shape or a.dtype != b.dtype or a.device != b.device
                           for a, b in zip(self.prefix, context.fixed_prefix))):
                raise BackboneError('KV_CONSTRAINT', 'H3 fixed-prefix geometry changed')

    def extract(self, raw):
        import torch
        return torch.cat((raw[..., :16], raw[..., 48:64]), -1)

    def assign(self, target, temporal):
        target[..., :16], target[..., 48:64] = temporal[..., :16], temporal[..., 16:]

    def rotate(self, raw, slot, dtype, *, cpu=None):
        import torch
        coefficients = cpu or self.coefficients
        cos, sin = (x[slot].to(device=raw.device, dtype=raw.dtype) for x in coefficients)
        first, second = raw.chunk(2, -1)
        # Preserve H3 _apply_rotary_emb's dtype and operation order, including
        # intermediate BF16 rounding; camera PRoPE uses disjoint last32 dims.
        return torch.cat((first * cos + (-second) * sin,
                          second * cos + first * sin), -1).to(dtype)


def geometry_for(context, family):
    if family == 'solarwm':
        return H3Time(context)
    if family not in ('lingbot', 'longlive') or context.native_geometry is not None:
        raise BackboneError('KV_CONSTRAINT', 'unsupported DraftKV native rotary family')
    return WanTime(context)


class ChronoCompact:
    def __init__(self, family):
        self.family = family
        self.raw_temporal = {}
        self.cache = {}
        self.cpu_coefficients = None
        self.layout = None
        self.coefficient_signature = None
        self.coefficient_source = ()
        # Threads exist only for cold miss bursts; elementwise FP64 math is unchanged.
        self._cpu_rotate_pool = None
        # A concurrent eight-rank SP8 test showed occasional slowdowns
        # for 5 local heads; avoid parallel CPU work on small page batches.
        self._parallel_min_bytes = 24 * 1024 * 1024

    @staticmethod
    def _signature(tensors):
        return tuple((str(t.device), int(t.data_ptr()), int(t._version),
                      tuple(t.shape), tuple(t.stride()), str(t.dtype)) for t in tensors)

    def cpu_coefficients_for(self, geometry):
        """Reuse unchanged native RoPE coefficients without repeated GPU D2H.

        A source tensor is retained so a freed GPU allocation cannot appear to
        be the old table under a recycled data_ptr. In-place writes increment
        the saved version counter; layout/dtype/shape/stride are also checked.
        Existing non-block Chrono-Compact routing remains unchanged.
        """
        signature = self._signature(geometry.coefficients)
        if (self.layout == geometry.layout and
                self.coefficient_signature == signature and
                self.cpu_coefficients is not None):
            return self.cpu_coefficients
        coefficients = tuple(x.to('cpu', copy=True) for x in geometry.coefficients)
        self.cpu_coefficients = coefficients
        self.coefficient_signature = signature
        self.coefficient_source = geometry.coefficients
        self.layout = geometry.layout
        self.cache.clear()
        return coefficients

    def route(self, context, keys, sink, selected, current, bank):
        import torch
        geometry = geometry_for(context, self.family)
        mapping = compact_map(sink, selected, current)
        signature = self._signature(geometry.coefficients)
        # Keep source tensors alive so a freed allocation cannot reuse an old
        # data_ptr with a reset version counter. In-place mutations bump _version.
        if self.layout != geometry.layout or self.coefficient_signature != signature:
            coefficients = tuple(x.to('cpu', copy=True) for x in geometry.coefficients)
            if (self.layout != geometry.layout or self.cpu_coefficients is None
                    or len(coefficients) != len(self.cpu_coefficients)
                    or any(not torch.equal(a, b) for a, b in zip(coefficients, self.cpu_coefficients))):
                self.cache.clear()
                self.cpu_coefficients, self.layout = coefficients, geometry.layout
            self.coefficient_signature = signature
            self.coefficient_source = geometry.coefficients
        coefficients = self.cpu_coefficients
        if max(mapping.values()) >= len(coefficients[0]):
            raise BackboneError('KV_CONSTRAINT', 'Chrono-Compact slot exceeds native rotary table')

        def rebase(native, raw):
            if all(mapping[fid] == fid for fid in current):
                return native.contiguous()
            result = native.clone()
            for rank, fid in enumerate(current):
                lo, hi = rank * context.frame_tokens, (rank + 1) * context.frame_tokens
                geometry.assign(result[:, lo:hi], geometry.rotate(
                    geometry.extract(raw[:, lo:hi]), mapping[fid], native.dtype))
            return result.contiguous()

        query = rebase(context.query, context.raw_query)
        offset = (len(sink) + len(selected)) * context.frame_tokens
        keys[:, offset:].copy_(rebase(context.key, context.raw_key))
        old = self.cache.get(context.layer, {})
        new, consumer_bank = {}, {}
        cold_ids = tuple(fid for fid in selected
                         if mapping[fid] != fid and old.get((fid, mapping[fid])) is None)
        cold_bytes = sum(bank[fid][0].numel() * bank[fid][0].element_size()
                         for fid in cold_ids)
        if len(cold_ids) < 6 or cold_bytes < self._parallel_min_bytes:
            # Exact original synchronous source and destination lifetime:
            # allocate, copy, rotate, then continue to the next selected page.
            for fid in selected:
                native_key, value = bank[fid]
                slot = mapping[fid]
                if slot == fid:
                    consumer_bank[fid] = native_key, value
                    continue
                key = old.get((fid, slot))
                if key is None:
                    raw = self.raw_temporal[context.layer][fid]
                    key = torch.empty_like(native_key, device='cpu', pin_memory=True)
                    key.copy_(native_key)
                    geometry.assign(key, geometry.rotate(
                        raw, slot, native_key.dtype, cpu=self.cpu_coefficients))
                new[(fid, slot)] = key
                consumer_bank[fid] = key, value
        else:
            # Cold independent pages are materialized in input order and
            # re-rotated in a bounded pool with unchanged FP64 arithmetic.
            misses = []
            for fid in selected:
                native_key, value = bank[fid]
                slot = mapping[fid]
                if slot == fid:
                    consumer_bank[fid] = native_key, value
                    continue
                key = old.get((fid, slot))
                if key is None:
                    raw = self.raw_temporal[context.layer][fid]
                    key = torch.empty_like(native_key, device='cpu', pin_memory=True)
                    misses.append((key, native_key, raw, slot))
                new[(fid, slot)] = key
                consumer_bank[fid] = key, value

            def rebase_miss(item):
                key, native_key, raw, slot = item
                key.copy_(native_key)
                geometry.assign(key, geometry.rotate(
                    raw, slot, native_key.dtype, cpu=self.cpu_coefficients))

            if self._cpu_rotate_pool is None:
                self._cpu_rotate_pool = ThreadPoolExecutor(
                    max_workers=4, thread_name_prefix='draftkv-chrono-k')
            for _ in self._cpu_rotate_pool.map(rebase_miss, misses):
                pass  # finish every page before the cache is published
        self.cache[context.layer] = new  # bounded to this read's R, not history×slots
        packed = sink + selected + current
        evidence = dict(position_policy=POSITION_POLICY, packing_policy=PACKING_POLICY,
            scoring_qk_space='native_post_rope',
            temporal_layout=geometry.layout, temporal_real_dims=geometry.dimensions,
            sink_ids=list(sink), selected_ids=list(selected), chronological_selected_ids=sorted(selected),
            current_ids=list(current), packed_source_ids=list(packed),
            packed_temporal_slots=[mapping[fid] for fid in packed],
            current_temporal_slots=[mapping[fid] for fid in current],
            raw_temporal_source='clean_context.raw_key_direct_no_inverse',
            spatial_and_camera_policy='native_unchanged', value_policy='native_unchanged',
            fixed_prefix_policy='unshifted_native' if geometry.prefix is not None else 'none',
            transformed_cache_pages=len(new),
            transformed_cache_bytes=sum(t.numel()*t.element_size() for t in new.values()))
        return query, consumer_bank, geometry.prefix, evidence

    def stage_clean(self, pending):
        staged = {}
        for layer, row in pending.items():
            context = row['context']
            if not context.clean:
                raise BackboneError('DIRTY_HISTORY', 'raw temporal keys require a clean model commit')
            geometry = geometry_for(context, self.family)
            staged[layer] = {}
            for rank, fid in enumerate(row['current_ids']):
                lo, hi = rank*context.frame_tokens, (rank+1)*context.frame_tokens
                staged[layer][fid] = geometry.extract(context.raw_key[0, lo:hi]).detach().to('cpu', copy=True).contiguous()
        # Construct complete replacement maps before publication (LongLive's
        # clean sink recache replaces only those source IDs, never the frontier).
        return {layer: {**self.raw_temporal.get(layer, {}), **rows} for layer, rows in staged.items()}

    def publish(self, banks):
        self.raw_temporal = banks
        self.cache.clear()

    def raw_bytes(self):
        return sum(t.numel()*t.element_size() for rows in self.raw_temporal.values() for t in rows.values())
