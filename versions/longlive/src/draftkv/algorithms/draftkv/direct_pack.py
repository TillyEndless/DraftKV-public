"""Direct Block Top-K CPU materialization into final B01 Group-major pages.

No intermediate complete ck/cv history tensor or second Group-major re-pack.
The default GPU 8x16 path packs raw temporal K only where the logical
Chrono-Compact slot differs from the source ID; all other spatial/camera
components and V are bitwise native. The native B01 fetch stream rotates those
temporal components AFTER H2D but BEFORE Frontier publication.
The original CPU-only oracle remains independently available.
"""
from __future__ import annotations


def materialize_block_history(bank, raw_temporal, history, *, geometry, coefficients,
                              sink_count, frame_tokens, heads, head_group,
                              cache, layer, pin_memory, chrono=True,
                              gpu_krope=False):
    import torch
    from draftkv.streaming.direct_source import DirectGroupMajorSource

    if heads % head_group or head_group <= 0:
        raise ValueError('B01 group-major head division mismatch')
    budget_tokens = sum(len(ids) for _, ids in history)
    if budget_tokens == 0 or budget_tokens % frame_tokens:
        raise ValueError('Block Top-K requires a nonempty whole-frame token budget')
    groups = heads // head_group
    shape = (2, groups, budget_tokens, head_group, 128)
    # No additional CPU history matrix: this one slab is the native B01 DMA
    # source. Its storage is safely reusable after the previous run_layer joins.
    row = cache.setdefault(layer, {})
    slab = row.get('slab')
    if (slab is None or tuple(slab.shape) != shape
            or slab.is_pinned() != bool(pin_memory)):
        slab = torch.empty(shape, dtype=torch.bfloat16, device='cpu',
                           pin_memory=pin_memory)
        row['slab'] = slab

    # One pinned byte of rotation metadata per historical token, independent
    # of head group and physical SP. No persistent GPU KV or per-block
    # temporary history cache is allocated.
    flags = None
    if gpu_krope:
        flags = row.get('gpu_rotation_flags')
        if flags is None or flags.numel() != budget_tokens:
            # Flags are ~KB per layer; avoid costly CUDA pinned page registration
            # just to upload them. The original pinned K/V slab is unchanged.
            flags = torch.empty((budget_tokens,), dtype=torch.uint8, device='cpu')
            row['gpu_rotation_flags'] = flags
        flags.zero_()
    full_frame = tuple(range(frame_tokens))
    cursor = 0
    source_map = []
    for fid, token_ids in history:
        n = len(token_ids)
        begin, end = cursor, cursor + n
        if n == 0 or end > budget_tokens:
            raise ValueError('invalid selected historical block length')
        key, value = bank[fid]
        full = token_ids == full_frame
        index = None if full else torch.tensor(token_ids, dtype=torch.long)
        for group in range(groups):
            hs = slice(group * head_group, (group + 1) * head_group)
            for side, src in enumerate((key, value)):
                dst = slab[side, group, begin:end]
                region = src[:, hs]
                if full:
                    dst.copy_(region)
                else:
                    torch.index_select(region, 0, index, out=dst)

        first_slot = sink_count + begin // frame_tokens
        last_slot = sink_count + (end - 1) // frame_tokens
        if chrono and (first_slot != fid or last_slot != fid):
            # The raw temporal components were captured before the original
            # native RoPE; no inverse rotation on rounded post-RoPE Keys.
            saved = raw_temporal[fid]
            raw = saved if full else saved.index_select(0, index)
            offset = 0
            while offset < n:
                slot = sink_count + (begin + offset) // frame_tokens
                segment_end = min(n, ((begin + offset) // frame_tokens + 1)
                                  * frame_tokens - begin)
                if slot != fid:
                    for group in range(groups):
                        hs = slice(group * head_group, (group + 1) * head_group)
                        target = slab[0, group, begin + offset:begin + segment_end]
                        if gpu_krope:
                            # Only the raw BF16 temporal components move
                            # unrotated; the fetch stream will apply precisely
                            # the native FP64 Wan or BF16 H3 operation.
                            geometry.assign(target, raw[offset:segment_end, hs])
                        else:
                            geometry.assign(target, geometry.rotate(
                                raw[offset:segment_end, hs], slot,
                                slab.dtype, cpu=coefficients))
                    if gpu_krope:
                        flags[begin + offset:begin + segment_end].fill_(1)
                offset = segment_end
        source_map.extend(fid * frame_tokens + t for t in token_ids)
        cursor = end

    if cursor != budget_tokens:
        raise ValueError('direct Block Top-K did not fill the pinned source')
    if gpu_krope:
        return slab, tuple(source_map), flags
    return slab, tuple(source_map)


def bound_direct_source(slab, *, budget, frame_tokens, prefix_tokens,
                        sink_tokens, total_tokens, heads, head_group):
    from draftkv.streaming.direct_source import DirectGroupMajorSource
    return DirectGroupMajorSource(
        slab[0], slab[1], tuple(range(budget)), frame_tokens,
        prefix_tokens + sink_tokens, prefix_tokens + total_tokens,
        heads, heads // head_group, head_group)
