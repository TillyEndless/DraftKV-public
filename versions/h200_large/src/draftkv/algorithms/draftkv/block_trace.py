"""Replay block decisions and source/time maps in the formal rollout validator."""
import hashlib
import json

from draftkv.backbones.contracts import BackboneError
from draftkv.retrieval.block_topk_p3 import select_blocks
from .block_policy import configured_protection, applied_protection


def validate_block_coordinates(plan, events, real):
    if plan.get('algorithm_plugin', {}).get('config', {}).get('retrieval_ablation') == 'camera-only':
        return validate_camera_only_coordinates(plan, events, real)
    from draftkv.backbones.catalog import MODELS, BLOCK_DRAFTKV_BACKBONES
    from draftkv.backbones.budgets import effective_budget
    if plan['backbone'] not in BLOCK_DRAFTKV_BACKBONES:
        raise BackboneError('TRACE_INCOMPLETE', 'block trace on an unconfigured backbone')
    spec=MODELS[plan['backbone']]
    allocation=effective_budget(spec,'draftkv')
    capacity=allocation.retrieved
    positions=[r for r in events if r.get('event')=='draftkv_time_position']
    reads=[r for r in events if r.get('event')=='draftkv_block_topk']
    key=lambda r:(r.get('forward_id'),r.get('layer'))
    indexed={key(r):r for r in positions};selected={key(r):r for r in reads}
    if len(positions)!=len(real) or len(indexed)!=len(real) or len(selected)!=len(reads):
        raise BackboneError('TRACE_INCOMPLETE','missing or duplicate block source/time maps')
    consumed_reads=set()
    for row in real:
        try:
            pos=indexed[key(row)];t=row['start_frame'];ft=pos['frame_tokens']
            h,w=pos['grid_hw'];budget=min(capacity,max(0,t-allocation.sink))
            sink=tuple(range(min(allocation.sink,t)));current=tuple(pos['current_ids'])
            assert row.get('retrieval_granularity')=='block'
            assert type(ft) is int and ft==h*w and ft>0
            assert pos['start_frame']==t and pos['clean']==row['clean']
            assert 1<=len(current)<=spec.chunk and current==tuple(range(t,t+len(current)))
            assert pos['position_policy']=='chrono_compact'
            assert pos['packing_policy']=='chronological_frame_id' and pos['packing_unit']=='source_token'
            assert pos['scoring_qk_space']=='native_post_rope'
            assert pos['raw_temporal_source']=='clean_context.raw_key_direct_no_inverse'
            assert pos['spatial_and_camera_policy']==pos['value_policy']=='native_unchanged'
            assert pos['fixed_prefix_policy']==('unshifted_native' if spec.family=='solarwm' else 'none')
            assert pos['temporal_rule']=='packed_token_offset//frame_tokens'
            assert pos['history_budget']==budget and pos['history_tokens']==budget*ft
            assert pos['sink_ids']==list(sink)
            assert pos['current_temporal_slots']==list(range(len(sink)+budget,len(sink)+budget+len(current)))
            assert len(sink)+budget+len(current)<=spec.window
            history=();protected=();remote=()
            if budget:
                read=selected[key(row)];consumed_reads.add(key(row))
                candidates=tuple(range(allocation.sink,t))
                assert read['candidate_ids']==list(candidates)
                assert read['softmax_domain']=='all_nonsink_history_including_protected'
                assert read['protected_frame_target']==configured_protection(plan['backbone'])
                assert read['policy_name']==f'direct_block_topk_p{configured_protection(plan["backbone"])}'
                assert read['recent_exclusion_gap']==0 and read['frame_importance_used'] is False
                assert read['threshold_z'] is None and read['relative_floor'] is None
                d=select_blocks(read['block_scores'],candidates,budget,query_start=t,
                                grid_h=h,grid_w=w,protect_recent=applied_protection(plan['backbone'],budget))
                history=d.token_indices;protected=d.protected;remote=d.remote_ids
                assert read['protected_ids']==list(protected)
                assert read['protected_frame_count']==len(protected)
                assert read['remote_equivalent_frames']==budget-len(protected)
                assert read['requested_history_frames']==budget and read['frame_tokens']==ft
                assert read['remote_token_budget']==read['remote_token_count']==d.remote_token_budget
                assert read['history_token_count']==budget*ft
                assert read['remote_source_frame_ids']==list(remote)
                assert read['ranked_blocks']==[dict(frame=f,block=b,importance=s,available_tokens=a,taken_tokens=n)
                                              for f,b,s,a,n in d.ranked_blocks]
                assert read['partial_boundary_block']==d.partial_boundary_block
                assert read['selected_source_tokens']==[dict(frame=f,tokens=list(ts)) for f,ts in history if f not in protected]
                assert read['transfer_page_ids']==list(range(budget))
            source=list(range(len(sink)*ft))
            source.extend(f*ft+token for f,ts in history for token in ts)
            source.extend(range(t*ft,(t+len(current))*ft))
            digest=hashlib.sha256(json.dumps(source,separators=(',',':')).encode()).hexdigest()
            assert source==sorted(set(source))
            assert len(source)==pos['token_budget']==(len(sink)+budget+len(current))*ft
            assert pos['source_token_map_sha256']==row['source_token_map_sha256']==digest
            if budget:assert read['source_token_map_sha256']==digest
            frames=[f for f,_ in history]
            assert pos['selected_ids']==row['selected_frame_ids']==frames
            assert pos['packed_source_frame_ids']==list(sink)+frames+list(current)
            assert pos['protected_ids']==list(protected)
            assert row['transfer_page_ids']==list(range(budget))
            assert row['event']==('draftkv_b01_attention' if budget else 'draftkv_b01_resident')
        except (AssertionError,KeyError,TypeError,ValueError,IndexError) as exc:
            raise BackboneError('TRACE_INCOMPLETE','invalid block selection, source-token budget or Chrono-Compact map') from exc
    if consumed_reads!=set(selected):
        raise BackboneError('TRACE_INCOMPLETE','unconsumed block selection trace')


def validate_camera_only_coordinates(plan, events, real):
    """Verify whole camera chunks and independent L/P domains without Draft replay."""
    from draftkv.backbones.catalog import MODELS
    from draftkv.backbones.budgets import effective_budget
    spec = MODELS[plan['backbone']]
    allocation = effective_budget(spec, 'draftkv')
    p = configured_protection(plan['backbone'])
    k = (allocation.retrieved - p) // spec.chunk
    key = lambda r: (r.get('forward_id'), r.get('layer'))
    positions = [r for r in events if r.get('event') == 'draftkv_time_position']
    indexed = {key(r): r for r in positions}
    routes = [r for r in events if r.get('event') == 'draftkv_camera_only_chunk_route']
    scopes = {(r.get('start_frame'), r.get('layer')): r for r in routes}
    if (len(positions) != len(real) or len(indexed) != len(real)
            or len(scopes) != len(routes)
            or any(r.get('event') in ('draftkv_block_topk', 'camera_gated_draftkv_route') for r in events)):
        raise BackboneError('TRACE_INCOMPLETE', 'missing/duplicate Camera-only maps or unexpected Draft scoring')
    for row in real:
        try:
            pos = indexed[key(row)]
            t, ft = row['start_frame'], pos['frame_tokens']
            assert type(ft) is int and ft > 0 and ft == pos['grid_hw'][0] * pos['grid_hw'][1]
            assert pos['start_frame'] == t and pos['clean'] == row['clean'] and t % spec.chunk == 0
            sink = tuple(range(min(allocation.sink, t)))
            local = tuple(range(max(allocation.sink, t - allocation.local), t))
            older = tuple(range(allocation.sink, local[0] if local else t))
            keep = min(p, allocation.retrieved, len(older))
            protected = older[-keep:] if keep else ()
            sink_chunks = (allocation.sink + spec.chunk - 1) // spec.chunk
            stop = min(t // spec.chunk - 2, (t - allocation.local) // spec.chunk)
            eligible = tuple(range(sink_chunks, max(sink_chunks, stop)))
            remote = ()
            if eligible or protected:
                route = scopes[(t, row['layer'])]
                chunks = tuple(route['selected_camera_chunk_ids'])
                assert route['eligible_camera_chunk_ids'] == list(eligible)
                assert len(chunks) == len(set(chunks)) == min(k, len(eligible))
                assert set(chunks).issubset(eligible) and route['camera_top_chunks'] == k
                remote = tuple(sorted(fid for chunk in chunks for fid in range(chunk * spec.chunk, (chunk + 1) * spec.chunk)))
                assert set(remote).issubset(older) and not set(remote).intersection(protected)
                assert route['remote_whole_frame_ids'] == list(remote)
                assert route['protected_frame_ids'] == list(protected)
                assert route['history_budget_frames'] == len(remote) + len(protected)
                assert route['exact_token_budget'] == (len(remote) + len(protected)) * ft
                assert route['remote_history_frames'] == len(remote)
                assert route['no_backfill'] is True and route['early_history_fallback'] is False
                assert all(route[name] is False for name in ('draft_importance_computed', 'draft_descriptors_persisted', 'block_importance_ranking'))
                assert route['transport'] == 'original_b01_streaming_fa2'
            history = tuple(sorted(local + protected + remote))
            assert len(history) == len(set(history))
            assert len(protected) + len(remote) <= allocation.retrieved
            current = tuple(pos['current_ids'])
            assert 1 <= len(current) <= spec.chunk and current == tuple(range(t, t + len(current)))
            total = len(sink) + len(history) + len(current)
            assert total <= allocation.capacity == spec.window
            assert pos['position_policy'] == 'chrono_compact' and pos['temporal_rule'] == 'packed_token_offset//frame_tokens'
            assert pos['packing_policy'] == 'chronological_frame_id' and pos['packing_unit'] == 'source_token'
            assert pos['scoring_qk_space'] == 'native_post_rope'
            assert pos['raw_temporal_source'] == 'clean_context.raw_key_direct_no_inverse'
            assert pos['spatial_and_camera_policy'] == pos['value_policy'] == 'native_unchanged'
            assert pos['fixed_prefix_policy'] == ('unshifted_native' if spec.family == 'solarwm' else 'none')
            assert pos['sink_ids'] == list(sink) and pos['protected_ids'] == list(protected)
            assert pos['selected_ids'] == row['selected_frame_ids'] == list(history)
            assert pos['history_budget'] == len(history) and pos['history_tokens'] == len(history) * ft
            assert pos['current_temporal_slots'] == list(range(len(sink) + len(history), total))
            assert pos['packed_source_frame_ids'] == list(sink + history + current)
            if allocation.local:
                assert pos['local_window_ids'] == list(local) and pos['local_budget_frames'] == allocation.local
                assert pos['retrieved_history_frames'] == len(protected) + len(remote)
            source = [fid * ft + token for fid in sink + history + current for token in range(ft)]
            digest = hashlib.sha256(json.dumps(source, separators=(',', ':')).encode()).hexdigest()
            assert source == sorted(set(source)) and len(source) == pos['token_budget'] == total * ft
            assert pos['source_token_map_sha256'] == row['source_token_map_sha256'] == digest
            assert row['transfer_page_ids'] == list(range(len(history)))
            assert row['event'] == ('draftkv_b01_attention' if history else 'draftkv_b01_resident')
            assert row['retrieval_granularity'] == 'block'
        except (AssertionError, KeyError, TypeError, ValueError, IndexError) as exc:
            raise BackboneError('TRACE_INCOMPLETE', 'invalid Camera-only L/R/P, whole chunks or Chrono-Compact map') from exc
