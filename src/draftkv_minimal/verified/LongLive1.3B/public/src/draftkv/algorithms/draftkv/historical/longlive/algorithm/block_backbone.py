# SPDX-License-Identifier: Apache-2.0
"""Token-matched Direct Block Top-K for all four canonical backbones.

P2/P3/P5 full frames stay protected. B01 transport pages are frame_tokens-sized chunks
of a source-ordered token stream, not semantic source frame IDs. Chrono-Compact
uses packed offset // frame_tokens; native spatial/camera dimensions, V and
H3 condition/audio prefixes retain their existing policy.
"""
from __future__ import annotations
from draftkv.streaming.external import external_symbol, external_path
from dataclasses import replace
import hashlib
import json
from draftkv.algorithms.draftkv.historical.longlive.backbones.contracts import AttentionSelection, BackboneError, FramePage
from draftkv.algorithms.draftkv.historical.longlive.retrieval.block_topk_p3 import block_probabilities, select_blocks
from draftkv.algorithms.draftkv.historical.longlive.streaming.pipeline import run_layer, run_resident
from .backbone import DraftKVHook
from .block_policy import configured_protection, applied_protection
from .position import geometry_for

class BlockDraftKVHook(DraftKVHook):

    def __init__(self, host):
        from .entry import configure_retrieval
        configure_retrieval(host.spec.backbone, host.plan['algorithm_plugin'].get('config', {}))
        super().__init__(host)

    def route(self, context):
        import torch
        (h, w) = self._verify_context(context)
        if self.host.spec.family not in ('longlive', 'lingbot', 'solarwm'):
            raise BackboneError('KV_CONSTRAINT', 'Block DraftKV supports LongLive, LingBot and SolarWM only')
        (layer, ft) = (context.layer, context.frame_tokens)
        self.active['forward_id'] = context.forward_id
        current = tuple(range(context.start_frame, context.start_frame + context.current_frames))
        bank = self.committed.get(layer, {})
        if set(bank) != set(range(self.next_frame)):
            raise BackboneError('DIRTY_HISTORY', 'whole-model clean frontier mismatch')
        sink = tuple(range(min(context.sink_frames, context.start_frame)))
        candidates = tuple((fid for fid in bank if context.sink_frames <= fid < context.start_frame))
        from draftkv.algorithms.draftkv.historical.longlive.backbones.budgets import bound_budget
        capacity = bound_budget(self.host, 'draftkv').retrieved
        budget = min(capacity, len(candidates))
        if capacity < 1:
            raise BackboneError('KV_CONSTRAINT', 'Block DraftKV requires positive history capacity')
        decision = None
        if candidates:
            probs = block_probabilities(context.query, [self.descriptors[layer][fid] for fid in candidates], h, w, config=self.importance)
            masses = probs.float().sum(dim=2)
            (heads, queries) = probs.shape[1:3]
            scores = masses.sum(dim=1)
            if context.sp > 1:
                import torch.distributed as dist
                if not dist.is_initialized() or dist.get_world_size() != context.sp:
                    raise BackboneError('KV_CONSTRAINT', 'exact native SP head group required')
                dist.all_reduce(scores, op=dist.ReduceOp.SUM)
            scores = scores / (heads * queries * context.sp)
            decision = select_blocks(scores[0].detach().cpu().tolist(), candidates, budget, query_start=context.start_frame, grid_h=h, grid_w=w, protect_recent=applied_protection(self.host.spec.backbone, budget))
        history = decision.token_indices if decision else ()
        local = decision.protected if decision else ()
        remote = decision.remote_ids if decision else ()
        selected_sources = tuple((fid for (fid, _) in history))
        total = (len(sink) + budget + len(current)) * ft
        shape = (1, total, *context.query.shape[2:])
        (k, v) = (torch.empty(shape, device=context.key.device, dtype=context.key.dtype) for _ in range(2))
        for (i, fid) in enumerate(sink):
            for (dst, src) in zip((k, v), self.sinks[layer][fid]):
                dst[:, i * ft:(i + 1) * ft].copy_(src)
        geometry = geometry_for(context, self.host.spec.family)
        coefficients = tuple((x.to('cpu') for x in geometry.coefficients))
        q = context.query.clone()
        offset = (len(sink) + budget) * ft
        k[:, offset:].copy_(context.key)
        v[:, offset:].copy_(context.value)
        for (i, fid) in enumerate(current):
            slot = len(sink) + budget + i
            (a, b) = (i * ft, (i + 1) * ft)
            if slot != fid:
                geometry.assign(q[:, a:b], geometry.rotate(geometry.extract(context.raw_query[:, a:b]), slot, q.dtype))
                geometry.assign(k[:, offset + a:offset + b], geometry.rotate(geometry.extract(context.raw_key[:, a:b]), slot, k.dtype))
        pin = context.query.is_cuda
        cpu_shape = (budget * ft, *context.query.shape[2:])
        (ck, cv) = (torch.empty(cpu_shape, dtype=context.key.dtype, device='cpu', pin_memory=pin) for _ in range(2))
        (cursor, pages, source_map) = (0, [], [])
        for (i, fid) in enumerate(sink):
            pages.append(FramePage(fid, i, i * ft, (i + 1) * ft))
            source_map.extend(range(fid * ft, (fid + 1) * ft))
        for (fid, token_ids) in history:
            index = torch.tensor(token_ids, dtype=torch.long)
            n = len(token_ids)
            ck[cursor:cursor + n].copy_(bank[fid][0].index_select(0, index))
            cv[cursor:cursor + n].copy_(bank[fid][1].index_select(0, index))
            raw = self.time_coordinates.raw_temporal[layer][fid].index_select(0, index)
            p = 0
            while p < n:
                slot = len(sink) + (cursor + p) // ft
                end = min(n, ((cursor + p) // ft + 1) * ft - cursor)
                if slot != fid:
                    geometry.assign(ck[cursor + p:cursor + end], geometry.rotate(raw[p:end], slot, ck.dtype, cpu=coefficients))
                p = end
            start = len(sink) * ft + cursor
            pages.append(FramePage(fid, len(pages), start, start + n, () if fid in local else token_ids))
            source_map.extend((fid * ft + t for t in token_ids))
            cursor += n
        if cursor != budget * ft:
            raise BackboneError('KV_CONSTRAINT', 'exact historical token budget mismatch')
        for (i, fid) in enumerate(current):
            start = offset + i * ft
            pages.append(FramePage(fid, len(pages), start, start + ft))
            source_map.extend(range(fid * ft, (fid + 1) * ft))
        if source_map != sorted(set(source_map)) or len(source_map) != total:
            raise BackboneError('KV_CONSTRAINT', 'physical KV source token order mismatch')
        transfer_ids = tuple(range(budget))
        consumer_bank = {i: (ck[i * ft:(i + 1) * ft], cv[i * ft:(i + 1) * ft]) for i in transfer_ids}
        source_sha = hashlib.sha256(json.dumps(source_map, separators=(',', ':')).encode()).hexdigest()
        position = dict(position_policy='chrono_compact', packing_policy='chronological_frame_id', packing_unit='source_token', scoring_qk_space='native_post_rope', grid_hw=[h, w], frame_tokens=ft, sink_ids=list(sink), selected_ids=list(selected_sources), current_ids=list(current), protected_ids=list(local), history_budget=budget, raw_temporal_source='clean_context.raw_key_direct_no_inverse', spatial_and_camera_policy='native_unchanged', value_policy='native_unchanged', fixed_prefix_policy='unshifted_native' if geometry.prefix is not None else 'none', temporal_rule='packed_token_offset//frame_tokens', spatial_policy='native_unchanged', current_temporal_slots=list(range(len(sink) + budget, len(sink) + budget + len(current))), packed_source_frame_ids=list(sink + selected_sources + current), source_token_map_sha256=source_sha, token_budget=total, history_tokens=budget * ft)
        pending = dict(context=context, current_ids=current, decision=decision, selected=selected_sources, consumed=False, consumer_bank=consumer_bank, transfer_ids=transfer_ids, fixed_prefix=geometry.prefix, position=position)
        self.pending[layer] = pending
        if decision:
            self.host.trace('draftkv_block_topk', layer=layer, chunk=context.start_frame, forward_id=context.forward_id, step=self.denoise_step, clean=context.clean, candidate_ids=list(candidates), softmax_domain='all_nonsink_history_including_protected', policy_name=f'direct_block_topk_p{configured_protection(self.host.spec.backbone)}', protected_frame_target=configured_protection(self.host.spec.backbone), recent_exclusion_gap=0, frame_importance_used=False, config=self.importance.to_dict(), config_sha256=self.importance_sha256, block_scores=list(decision.block_scores), protected_ids=list(local), remote_token_budget=decision.remote_token_budget, remote_token_count=sum((len(ts) for (f, ts) in history if f not in local)), ranked_blocks=[dict(frame=f, block=b, importance=s, available_tokens=a, taken_tokens=n) for (f, b, s, a, n) in decision.ranked_blocks], partial_boundary_block=decision.partial_boundary_block, selected_source_tokens=[dict(frame=f, tokens=list(ts)) for (f, ts) in history if f not in local], remote_source_frame_ids=list(remote), history_token_count=budget * ft, source_token_map_sha256=source_sha, chronological_source_tokens=True, requested_history_frames=budget, protected_frame_count=len(local), remote_equivalent_frames=budget - len(local), frame_tokens=ft, transfer_page_ids=list(transfer_ids), threshold_z=None, relative_floor=None, position=position)
        self.host.trace('draftkv_time_position', layer=layer, start_frame=context.start_frame, forward_id=context.forward_id, clean=context.clean, **position)
        return AttentionSelection(q.contiguous(), k, v, sink, local, remote, current, tuple(pages), dict(representation='paged', packing_policy='chronological_frame_id', packing_unit='source_token', position_map=tuple(((i // ft, t % ft) for (i, t) in enumerate(source_map))), draftkv_plan=pending, packed_source_token_sha256=source_sha))

    def consume(self, context, selection, native_fa2):
        pending = self.pending.get(context.layer)
        if pending is None or selection.metadata.get('draftkv_plan') is not pending:
            raise BackboneError('DIRTY_HISTORY', 'block consumer differs from staged owner')
        (keys, values) = (selection.keys, selection.values)
        prefix_tokens = 0
        if pending['fixed_prefix'] is not None:
            import torch
            (pk, pv) = pending['fixed_prefix']
            prefix_tokens = pk.shape[1]
            keys = torch.cat((pk, keys), 1).contiguous()
            values = torch.cat((pv, values), 1).contiguous()
        p = {**self.binding.parameters, 'sink_tokens': prefix_tokens + len(selection.sink_ids) * context.frame_tokens, 'frame_tokens': context.frame_tokens}
        binding = replace(self.binding, parameters=p)
        physical_sp = int(self.host.plan.get('sp', context.sp))

        def execute(query):
            if pending['transfer_ids']:
                return run_layer(query, keys, values, pending['consumer_bank'], pending['transfer_ids'], binding=binding, layer=context.layer, layers=self.host.spec.layers, sp=physical_sp, staging_cache=self._b01_pinned_staging)
            return run_resident(query, keys, values, binding=binding, layer=context.layer, layers=self.host.spec.layers, sp=physical_sp)
        if self.host.plan.get('sp_mode') == 'query_sequence_replicated_kv' and physical_sp > 1:
            from draftkv.algorithms.draftkv.historical.longlive.backbones.sequence_parallel import query_sequence_attention
            holder = {}

            def local_compute(query):
                (result, holder['metrics']) = execute(query)
                return result
            (output, sp_trace) = query_sequence_attention(selection.query, physical_sp, local_compute)
            metrics = holder['metrics']
            metrics['sequence_parallel'] = sp_trace
        else:
            (output, metrics) = execute(selection.query)
        pending['consumed'] = True
        profile = metrics.pop('profile')
        self.host.trace('draftkv_b01_attention' if pending['transfer_ids'] else 'draftkv_b01_resident', layer=context.layer, start_frame=context.start_frame, retrieval_granularity='block', selected_frame_ids=list(pending['selected']), clean=context.clean, forward_id=context.forward_id, denoise_step=self.denoise_step, transfer_page_ids=list(pending['transfer_ids']), source_frame_ids=list(pending['selected']), source_token_map_sha256=pending['position']['source_token_map_sha256'], checkpoint_sha=self.checkpoint_sha, profile_sha=profile.get('profile', profile).get('sha256'), **metrics)
        return output
