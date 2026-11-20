"""Export actual device selection/consumption evidence at a whole-forward boundary.

The route/consumer hot path only retains small GPU metadata tensors. One batched
copy per dtype drains them after the last layer; no historical KV is copied.
"""
import hashlib
import json

from draftkv.backbones.contracts import BackboneError


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':')).encode()).hexdigest()


def stage(hook, context, selection, row):
    pending = selection.metadata['draftkv_plan']
    decision = pending['decision']
    source = pending['direct_source']
    tensors = {'source': selection.metadata['device_source_tokens']}
    if decision is not None:
        tensors.update(history=decision.device_tokens, candidates=decision.candidates)
        if decision.scores is not None:
            tensors['scores'] = decision.scores
        if getattr(source, 'logical_source_tokens', None) is not None:
            tensors.update(consumed=source.logical_source_tokens,
                           permutation=source.physical_to_logical, spans=source.metadata,
                           pointers=source.directory.pointers, mapping=source.directory.mapping,
                           starts=source.directory.starts, sizes=source.directory.sizes)
    mode = hook.settings.get('retrieval_ablation', 'full')
    if mode == 'camera-only':
        chunks = getattr(hook, '_device_camera_only_chunks', None)
    elif mode == 'full' and hook.host.spec.family != 'longlive':
        scope = getattr(hook, '_device_camera_scope', None)
        chunks = scope[-1] if scope is not None and scope[0] == context.start_frame else None
    else:
        chunks = None
    if chunks is not None:
        tensors['chunks'] = chunks
    audit = dict(row=row, position=dict(pending['position']), tensors=tensors,
                 mode=mode, sink=list(selection.sink_ids), current=list(selection.current_ids),
                 protected_count=decision.protected_count if decision else 0,
                 source_group=source.head_group if source is not None else 0,
                 source_heads=source.heads if source is not None else 0)
    if not hasattr(hook, '_device_audits'):
        hook._device_audits = []
    hook._device_audits.append(audit)


def verify_dma(audit, data):
    """Check the consumed producer's actual pointer spans against its logical tokens."""
    import numpy as np
    history = data.get('history', [])
    if not history:
        return None
    if 'consumed' not in data:
        # The CPU-RoPE experiment has its own selected-only bridge; this task
        # does not label that bridge as GPU archive consumption.
        if audit['position'].get('cpu_krope_selected_only_bridge'):
            return None
        raise ValueError('GPU archive lacks actual consumed source-token metadata')
    if data['consumed'] != history:
        raise ValueError('consumer source tokens differ from the actual GPU selection')
    n = len(history)
    permutation = data['permutation']
    if sorted(permutation) != list(range(n)):
        raise ValueError('consumer source permutation is not bijective')
    ft = audit['position']['frame_tokens']
    ids = np.asarray(history, dtype=np.int64)[permutation]
    mapping = np.asarray(data['mapping'], dtype=np.int64).reshape(ft, 2)
    pointers = np.asarray(data['pointers'], dtype=np.int64).reshape(-1, 2)
    starts, sizes = np.asarray(data['starts']), np.asarray(data['sizes'])
    block, offset = mapping[ids % ft].T
    heads, group = audit['source_heads'], audit['source_group']
    groups = heads // group
    spans = np.asarray(data['spans'], dtype=np.int64).reshape(groups, -1, 4)
    width = group * 128 * 2
    for g, rows in enumerate(spans):
        base = pointers[ids // ft, 0] + starts[block] * 2 * heads * 128 * 2
        keys = base + (g * 2 * sizes[block] + offset) * width
        values = keys + sizes[block] * width
        cursor = 0
        for key, value, begin, count in rows:
            if count == 0:
                continue
            if begin != cursor or count < 1 or begin + count > n:
                raise ValueError('producer span coverage differs from the selected history')
            end = begin + count
            if (key != keys[begin] or value != values[begin]
                    or not np.array_equal(keys[begin:end], key + np.arange(count) * width)
                    or not np.array_equal(values[begin:end], value + np.arange(count) * width)):
                raise ValueError('producer pointers do not read the selected source KV')
            cursor = end
        if cursor != n:
            raise ValueError('producer did not cover the complete selected history')
    return dict(schema='draftkv.device-selection.v1',
                history_source_sha256=digest(history), producer_spans_sha256=digest(data['spans']),
                source_permutation_sha256=digest(permutation), history_tokens=n,
                producer_pointer_map_verified=True)


def flush(hook):
    import torch
    audits = getattr(hook, '_device_audits', [])
    if not audits:
        return
    batches = {}
    for index, audit in enumerate(audits):
        for name, tensor in audit['tensors'].items():
            kind = 'float' if tensor.is_floating_point() else 'int'
            batches.setdefault((str(tensor.device), kind), []).append((index, name, tensor.detach().reshape(-1)))
    exported = [{} for _ in audits]
    for (_, kind), entries in batches.items():
        dtype = torch.float32 if kind == 'float' else torch.int64
        values = torch.cat([t.to(dtype=dtype) for _, _, t in entries]).to('cpu').tolist()
        cursor = 0
        for index, name, tensor in entries:
            exported[index][name] = values[cursor:cursor + tensor.numel()]
            cursor += tensor.numel()
    try:
        for audit, data in zip(audits, exported):
            emit(hook, audit, data)
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        raise BackboneError('TRACE_INCOMPLETE', 'device archive audit: ' + str(exc)) from exc
    finally:
        hook._device_audits = []


def emit(hook, audit, data):
    """Build compact trace records from actual exported IDs, scores and DMA spans."""
    from draftkv.backbones.budgets import bound_budget
    from .block_policy import configured_protection
    from draftkv.retrieval.block_topk_p3 import select_blocks
    allocation = bound_budget(hook.host, 'draftkv')
    row, pos = audit['row'], audit['position']
    start, ft = row['start_frame'], pos['frame_tokens']
    sink, current = audit['sink'], audit['current']
    history = data.get('history', [])
    if data['source'] != sorted(set(data['source'])):
        raise ValueError('GPU source map is not unique and chronological')
    expected_edges = [f * ft + t for f in sink for t in range(ft)] + history + [f * ft + t for f in current for t in range(ft)]
    if data['source'] != expected_edges:
        raise ValueError('actual source map differs from the consumed history/current ranges')
    source_sha = digest(data['source'])
    frames = list(dict.fromkeys(t // ft for t in history))
    local = list(range(max(allocation.sink, start-allocation.local), start)) if allocation.local else []
    older = list(range(allocation.sink, local[0] if local else start))
    p = min(configured_protection(hook.host.spec.backbone), allocation.retrieved, len(older))
    protected = older[-p:] if p else []
    remote = [f for f in frames if f not in protected and f not in local]
    proof = verify_dma(audit, data)
    base = dict(layer=row['layer'], start_frame=start, forward_id=row['forward_id'], clean=row['clean'])
    mode, candidates = audit['mode'], data.get('candidates', [])
    if mode == 'camera-only':
        if history != [f * ft + t for f in frames for t in range(ft)]:
            raise ValueError('Camera-only consumed partial spatial frames')
        budget = len(frames)
    elif history:
        decision = select_blocks(data['scores'], candidates, pos['history_budget'], query_start=start,
                                 grid_h=pos['grid_hw'][0], grid_w=pos['grid_hw'][1], protect_recent=audit['protected_count'])
        actual = [f * ft + t for f, tokens in decision.token_indices for t in tokens]
        if actual != history:
            raise ValueError('actual GPU selection differs from score replay')
        protected, remote = list(decision.protected), list(decision.remote_ids)
        budget = decision.history_budget
        hook.host.trace('draftkv_block_topk', **base, candidate_ids=candidates,
            softmax_domain=('candidate_subset_softmax_including_true_latest_P' if mode == 'full' and hook.host.spec.family != 'longlive'
                            else 'all_nonsink_history_including_protected'),
            policy_name=f'direct_block_topk_p{configured_protection(hook.host.spec.backbone)}',
            protected_frame_target=configured_protection(hook.host.spec.backbone), recent_exclusion_gap=0,
            frame_importance_used=False, threshold_z=None, relative_floor=None, block_scores=data['scores'],
            protected_ids=protected, protected_frame_count=len(protected), remote_equivalent_frames=budget-len(protected),
            requested_history_frames=budget, frame_tokens=ft, remote_token_budget=decision.remote_token_budget,
            remote_token_count=decision.remote_token_budget, history_token_count=len(history), remote_source_frame_ids=remote,
            ranked_blocks=[dict(frame=f,block=b,importance=s,available_tokens=a,taken_tokens=n) for f,b,s,a,n in decision.ranked_blocks],
            partial_boundary_block=decision.partial_boundary_block,
            selected_source_tokens=[dict(frame=f,tokens=list(ts)) for f,ts in decision.token_indices if f not in protected],
            transfer_page_ids=list(range(budget)), source_token_map_sha256=source_sha)
    else:
        budget = 0
    chunk = hook.host.spec.chunk
    if mode in ('full', 'camera-only') and hook.host.spec.family != 'longlive' and (history or data.get('chunks')):
        first = (allocation.sink+chunk-1)//chunk
        stop = max(first, min(start//chunk-2, (start-allocation.local)//chunk))
        chunks = data.get('chunks', [])
        event_key = (start, row['layer'], mode)
        seen = getattr(hook, '_device_audit_camera_events', set())
        if event_key not in seen:
            if mode == 'camera-only':
                hook.host.trace('draftkv_camera_only_chunk_route', **base,
                    selected_camera_chunk_ids=chunks, eligible_camera_chunk_ids=list(range(first,stop)),
                    camera_top_chunks=hook.camera_top_chunks, remote_selection_policy='camera_rank_then_frame_id_truncate_after_reserving_P',
                    remote_whole_frame_ids=remote, protected_frame_ids=protected, history_budget_frames=budget-len(local),
                    exact_token_budget=(budget-len(local))*ft,remote_history_frames=len(remote),no_backfill=True,
                    early_history_fallback=False,draft_importance_computed=False,draft_descriptors_persisted=False,
                    block_importance_ranking=False,transport='native_' + row['consumer_backend'].lower() + '_streaming')
            else:
                hook.host.trace('camera_gated_draftkv_route', **base, selected_camera_chunk_ids=chunks,
                    eligible_camera_chunk_ids=list(range(first,stop)),requested_camera_top_m=hook.camera_top_chunks,
                    true_latest_protected_ids=protected,selected_remote_frame_candidates=candidates,
                    actual_history_budget_frames=budget,early_history_fallback=False,no_backfill=True,
                    scorer_domain='candidate_subset_softmax_including_true_latest_P')
            seen.add(event_key)
            hook._device_audit_camera_events = seen
    pos.update(selected_ids=frames, protected_ids=protected, packed_source_frame_ids=sink+frames+current,
        source_token_map_sha256=source_sha, raw_temporal_source='clean_context.raw_key_direct_no_inverse',
        spatial_and_camera_policy='native_unchanged',value_policy='native_unchanged',
        fixed_prefix_policy='unshifted_native' if hook.host.spec.family=='solarwm' else 'none',
        current_temporal_slots=list(range(len(sink)+budget,len(sink)+budget+len(current))))
    if allocation.local:
        pos.update(local_window_ids=local,local_budget_frames=allocation.local,retrieved_history_frames=budget-len(local))
    hook.host.trace('draftkv_time_position', **base, **pos)
    row.update(selected_frame_ids=frames,source_frame_ids=frames,source_token_map_sha256=source_sha,
               device_selection_audit=proof,device_selection_export='whole_forward_boundary')
    hook.host.trace(row.pop('event'), **row)
