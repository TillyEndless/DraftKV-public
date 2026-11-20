"""Token-matched Direct Block Top-K for all four canonical backbones.

P2/P3/P5 full frames stay protected. B01 transport pages are frame_tokens-sized chunks
of a source-ordered token stream, not semantic source frame IDs. Chrono-Compact
uses packed offset // frame_tokens; native spatial/camera dimensions, V and
H3 condition/audio prefixes retain their existing policy.
"""
from dataclasses import replace
import hashlib
import json

from draftkv.backbones.contracts import AttentionSelection, BackboneError, FramePage
from draftkv.retrieval.block_topk_p3 import block_probabilities, select_blocks
from draftkv.streaming.pipeline import run_layer, run_resident
from .backbone import DraftKVHook
from .block_policy import configured_protection, applied_protection
from .position import geometry_for
from .direct_pack import materialize_block_history, bound_direct_source
from draftkv.streaming.krope_8x16 import GpuKRoPEPlan


class BlockDraftKVHook(DraftKVHook):
    def __init__(self, host):
        from .entry import configure_retrieval
        configure_retrieval(host.spec.backbone, host.plan['algorithm_plugin'].get('config', {}))
        super().__init__(host)

    def _history_budget(self, context, candidates, budget):
        return budget

    def _history_staging_device(self, context):
        return 'cpu'

    def _validate_sink(self, pair):
        if any(src.device.type != 'cpu' for src in pair):
            raise BackboneError('KV_CONSTRAINT', 'persistent sink KV must be on CPU')

    def _use_direct_history(self, context):
        return True

    def _native_runtime_config(self, context):
        from draftkv.streaming.table2_native import runtime_config
        return runtime_config(getattr(self,'settings',{}),device=context.query.device,
                              native_version=self.host.plan.get('native_attention_version'),
                              heads=context.query.shape[2],default_slots=True,
                              physical_sp=self.host.plan.get('sp',getattr(context,'sp',1)))

    def _select_history(self, context, candidates, budget, h, w):
        decision = None
        if candidates:
            # Every non-sink historical block, including protected P, enters
            # softmax; only selection ranking excludes the latest full P frames.
            probs = block_probabilities(context.query,
                self._scoring_descriptors(context.layer, candidates, context.query.device),
                h, w, config=self.importance)
            masses = probs.float().sum(dim=2)  # sum query regions, never key blocks
            heads, queries = probs.shape[1:3]
            scores = masses.sum(dim=1)
            if context.sp > 1:
                import torch.distributed as dist
                if not dist.is_initialized() or dist.get_world_size() != context.sp:
                    raise BackboneError('KV_CONSTRAINT', 'exact native SP head group required')
                dist.all_reduce(scores, op=dist.ReduceOp.SUM)
            scores = scores / (heads * queries * context.sp)
            decision = select_blocks(scores[0], candidates, budget,
                query_start=context.start_frame, grid_h=h, grid_w=w,
                protect_recent=applied_protection(self.host.spec.backbone, budget),
                pool_h=self.importance.pool_h,pool_w=self.importance.pool_w)
        return decision

    def route(self, context):
        import torch
        from draftkv.streaming.canonical_archive import ensure_archive
        h, w = self._verify_context(context)
        archive = ensure_archive(self, context)
        from .device_archive_route import enabled, route as device_archive_route
        if enabled(self, context):
            return device_archive_route(self, context, h, w, archive)
        from .device_archive_groups import enabled as group_enabled, route as group_route
        if group_enabled(self, context):
            return group_route(self, context, h, w, archive)
        if self.host.spec.family not in ('longlive', 'lingbot', 'solarwm'):
            raise BackboneError('KV_CONSTRAINT', 'Block DraftKV supports LongLive, LingBot and SolarWM only')
        layer, ft = context.layer, context.frame_tokens
        self.active['forward_id'] = context.forward_id
        current = tuple(range(context.start_frame, context.start_frame+context.current_frames))
        bank = self.committed.get(layer, {})
        if set(bank) != set(range(self.next_frame)):
            raise BackboneError('DIRTY_HISTORY', 'whole-model clean frontier mismatch')
        sink = tuple(range(min(context.sink_frames, context.start_frame)))
        candidates = tuple(fid for fid in bank if context.sink_frames <= fid < context.start_frame)
        from draftkv.backbones.budgets import bound_budget
        allocation = bound_budget(self.host, 'draftkv')
        # L is an explicit mandatory recent window, separate from P inside R.
        # It shares the CPU archive and chronological B01 transport with R;
        # no historical KV becomes persistent GPU state.
        if allocation.local and getattr(self, 'settings', {}).get('retrieval_ablation') != 'camera-only':
            raise BackboneError('KV_CONSTRAINT', 'local Block DraftKV allocation requires Camera-only')
        local_window = candidates[-min(allocation.local, len(candidates)):] if allocation.local else ()
        if local_window:
            candidates = tuple(fid for fid in candidates if fid < local_window[0])
        capacity = allocation.retrieved
        budget = min(capacity, len(candidates))
        if capacity < 1:
            raise BackboneError('KV_CONSTRAINT', 'Block DraftKV requires positive history capacity')
        budget = self._history_budget(context, candidates, budget)
        decision = self._select_history(context, candidates, budget, h, w)
        history = decision.token_indices if decision else ()
        protected = decision.protected if decision else ()
        local = tuple(sorted(local_window + protected))
        remote = decision.remote_ids if decision else ()
        retrieved_budget = budget
        if local_window:
            history = tuple(sorted(history + tuple((fid, tuple(range(ft))) for fid in local_window)))
            budget += len(local_window)
        selected_sources = tuple(fid for fid, _ in history)
        total = (len(sink)+budget+len(current))*ft
        # Default CUDA release path: select exactly the canonical A2/Block
        # Top-K tokens, then assemble them directly in the final pinned
        # B01 ProducerPlan Group-major H2D source. No temporary full CPU KV
        # tensor or second CPU regrouping. CPU fixture and external Systems
        # host-page policy retain the original path unchanged.
        direct = (context.query.is_cuda
                  and self._use_direct_history(context)
                  and self._history_staging_device(context) == 'cpu'
                  and getattr(self.binding, 'host_pages_policy', 'quality') == 'quality'
                  and budget > 0)
        geometry = geometry_for(context, self.host.spec.family)
        # Camera A2 / Block Top-K selection is already frozen. This
        # acceleration changes ONLY the temporal position transform of
        # selected K after H2D. Never import the 900d66d SP1/32x32 selector.
        gpu_krope = (direct and getattr(self, 'position_policy', 'chrono_compact')
                     == 'chrono_compact'
                     and getattr(self, 'settings', {}).get('gpu_krope_8x16', True))
        if gpu_krope and (self.importance.pool_h, self.importance.pool_w) != (8,16):
            raise BackboneError('BLOCKED_ALGORITHM_SEMANTICS',
                                'GPU K-RoPE requires exact 8x16 Block Top-K')
        prefix = geometry.prefix
        prefix_tokens = prefix[0].shape[1] if direct and prefix is not None else 0
        chrono=self.position_policy=='chrono_compact' if hasattr(self,'position_policy') else True
        coefficients = (self.time_coordinates.cpu_coefficients_for(geometry)
                        if chrono and not gpu_krope else None)
        resident = None
        if (direct and self._native_runtime_config(context) is not None
                and getattr(self, 'settings', {}).get('direct_resident_kv', True)):
            from draftkv.streaming.resident_kv import prepare_resident
            from draftkv.streaming.table2_native import reserve_native_dense
            native_cfg=self._native_runtime_config(context)
            dense = None if native_cfg.get('kv_slots') else reserve_native_dense(context.query,prefix_tokens+total,
                binding=self.binding,cache=self._b01_pinned_staging,
                config=self._native_runtime_config(context),family=self.host.spec.family,
                capacity_tokens=prefix_tokens+context.window_frames*ft)
            q, resident = prepare_resident(context, geometry, sink, current, budget,
                bank, self._validate_sink, chrono=chrono,destination=dense)
            k, v = context.key, context.value
            full_k = full_v = None
        else:
            if direct:
                full_shape = (1, prefix_tokens + total, *context.query.shape[2:])
                full_k, full_v = (torch.empty(full_shape, device=context.key.device,
                                               dtype=context.key.dtype) for _ in range(2))
                if prefix_tokens:
                    full_k[:, :prefix_tokens].copy_(prefix[0])
                    full_v[:, :prefix_tokens].copy_(prefix[1])
                k, v = full_k[:, prefix_tokens:], full_v[:, prefix_tokens:]
            else:
                shape = (1, total, *context.query.shape[2:])
                k, v = (torch.empty(shape, device=context.key.device, dtype=context.key.dtype)
                        for _ in range(2))
                full_k, full_v = None, None
            for i, fid in enumerate(sink):
                if fid not in bank:
                    raise BackboneError('DIRTY_HISTORY', 'CPU committed sink missing from this layer')
                self._validate_sink(bank[fid])
                for dst, src in zip((k, v), bank[fid]):
                    dst[:, i*ft:(i+1)*ft].copy_(src.unsqueeze(0))
            q = context.query.clone()
            offset = (len(sink)+budget)*ft
            k[:, offset:].copy_(context.key)
            v[:, offset:].copy_(context.value)
            for i, fid in enumerate(current):
                slot = len(sink)+budget+i
                a, b = i*ft, (i+1)*ft
                if chrono and slot != fid:
                    geometry.assign(q[:, a:b], geometry.rotate(geometry.extract(context.raw_query[:, a:b]),slot,q.dtype))
                    geometry.assign(k[:, offset+a:offset+b], geometry.rotate(geometry.extract(context.raw_key[:, a:b]),slot,k.dtype))
        offset = (len(sink)+budget)*ft
        direct_source = None
        gpu_krope_plan = None
        if direct:
            params = self.binding.parameters
            heads = context.query.shape[2]
            from draftkv.streaming.device_profiles import transport_group
            head_group=transport_group(self._native_runtime_config(context),params,heads)
            if heads % head_group:
                raise BackboneError('KV_CONSTRAINT', 'B01 head group must divide local heads')
            from draftkv.streaming.canonical_prepack import bind_archive
            archive_bound = (bind_archive(bank,history,head_group=head_group,
                frame_tokens=ft,history_start=prefix_tokens+len(sink)*ft,
                total=prefix_tokens+total,device=context.query.device,
                device_tokens=getattr(decision,"device_tokens",None) if not local_window else None,
                cache=self._b01_pinned_staging.setdefault(('archive_registry',layer),{})) if (gpu_krope or not chrono)
                and self._native_runtime_config(context) is not None else None)
            if archive_bound is not None:
                direct_source,history_map=archive_bound
                if gpu_krope:
                    # Every archived temporal component is raw. Rotate once
                    # to its logical compact slot, including unchanged slots.
                    raw_flags=torch.ones(budget*ft,dtype=torch.uint8)
                    gpu_krope_plan=GpuKRoPEPlan.from_flags(raw_flags,
                        sink_frames=len(sink),frame_tokens=ft,family=self.host.spec.family,
                        head_group=head_group,geometry=geometry,device=context.query.device)
            else:
                prepared = materialize_block_history(
                    bank, self.time_coordinates.raw_temporal.get(layer, {}), history,
                    geometry=geometry, coefficients=coefficients, chrono=chrono,
                    sink_count=len(sink), frame_tokens=ft, heads=heads,
                    head_group=head_group,
                    cache=self._b01_pinned_staging.setdefault('direct_block', {}),
                    layer=layer, pin_memory=True, gpu_krope=gpu_krope)
                if gpu_krope:
                    slab, history_map, raw_flags = prepared
                    # Compatibility/CPU-RoPE ablations retain the selected slab.
                    # GPU K-RoPE still follows its H2D before publication.
                    gpu_krope_plan = GpuKRoPEPlan.from_flags(
                        raw_flags, sink_frames=len(sink), frame_tokens=ft,
                        family=self.host.spec.family, head_group=head_group,
                        geometry=geometry, device=context.query.device)
                else:
                    slab, history_map = prepared
                direct_source = bound_direct_source(
                    slab, budget=budget, frame_tokens=ft,
                    prefix_tokens=prefix_tokens, sink_tokens=len(sink)*ft,
                    total_tokens=total, heads=heads, head_group=head_group)
            ck, cv = None, None
        else:
            pin = context.query.is_cuda
            cpu_shape = (budget*ft, *context.query.shape[2:])
            ck, cv = (torch.empty(cpu_shape, dtype=context.key.dtype,
                                 device=self._history_staging_device(context),
                                 pin_memory=pin and self._history_staging_device(context)=='cpu')
                      for _ in range(2))
            history_map = ()
        cursor, pages, source_map = 0, [], []
        full_frame_tokens = tuple(range(ft))
        raw_temporal = self.time_coordinates.raw_temporal.get(layer, {})
        for i, fid in enumerate(sink):
            pages.append(FramePage(fid, i, i*ft, (i+1)*ft))
            source_map.extend(range(fid*ft,(fid+1)*ft))
        for fid, token_ids in history:
            n = len(token_ids)
            begin, finish = cursor, cursor + n
            if not direct:
                dst_k, dst_v = ck[begin:finish], cv[begin:finish]
                entire_frame = token_ids == full_frame_tokens
                if entire_frame:
                    dst_k.copy_(bank[fid][0])
                    dst_v.copy_(bank[fid][1])
                    index = None
                else:
                    index = torch.tensor(token_ids, dtype=torch.long, device=bank[fid][0].device)
                    torch.index_select(bank[fid][0], 0, index, out=dst_k)
                    torch.index_select(bank[fid][1], 0, index, out=dst_v)
                first_slot = len(sink) + begin // ft
                last_slot = len(sink) + (finish - 1) // ft
                if chrono and (first_slot != fid or last_slot != fid):
                    saved_raw = raw_temporal[fid]
                    raw = saved_raw if entire_frame else saved_raw.index_select(0, index)
                    p = 0
                    while p < n:
                        slot = len(sink) + (begin+p) // ft
                        end = min(n, ((begin+p)//ft+1)*ft-begin)
                        if slot != fid:
                            geometry.assign(ck[begin+p:begin+end],
                                geometry.rotate(raw[p:end], slot, ck.dtype, cpu=coefficients))
                        p = end
            start = len(sink)*ft+cursor
            pages.append(FramePage(fid,len(pages),start,start+n,() if fid in local else token_ids))
            source_map.extend(fid*ft+t for t in token_ids)
            cursor += n
        if cursor != budget*ft:
            raise BackboneError('KV_CONSTRAINT','exact historical token budget mismatch')
        for i,fid in enumerate(current):
            start=offset+i*ft
            pages.append(FramePage(fid,len(pages),start,start+ft))
            source_map.extend(range(fid*ft,(fid+1)*ft))
        if source_map != sorted(set(source_map)) or len(source_map)!=total:
            raise BackboneError('KV_CONSTRAINT','physical KV source token order mismatch')
        # Transport page IDs are NOT semantic source frame IDs. For the
        # optimized path, native ProducerPlan reads the final pinned slab
        # directly; the old per-page CPU bank exists only in the CPU oracle /
        # explicit Systems host-page mode.
        if direct and source_map[len(sink)*ft:len(sink)*ft + budget*ft] != list(history_map):
            raise BackboneError('KV_CONSTRAINT', 'direct DMA source identity differs from A2 selection')
        transfer_ids = tuple(range(budget))
        consumer_bank = ({} if direct else
                         {i:(ck[i*ft:(i+1)*ft],cv[i*ft:(i+1)*ft]) for i in transfer_ids})
        source_sha = hashlib.sha256(json.dumps(source_map,separators=(',',':')).encode()).hexdigest()
        position = dict(position_policy='chrono_compact' if chrono else 'native_post_rope',
            packing_policy='chronological_frame_id',packing_unit='source_token',
            scoring_qk_space='native_post_rope',grid_hw=[h,w],frame_tokens=ft,
            sink_ids=list(sink),selected_ids=list(selected_sources),current_ids=list(current),
            protected_ids=list(protected),history_budget=budget,
            raw_temporal_source='clean_context.raw_key_direct_no_inverse' if chrono else 'unused',
            spatial_and_camera_policy='native_unchanged',value_policy='native_unchanged',
            fixed_prefix_policy='unshifted_native' if geometry.prefix is not None else 'none',
            temporal_rule='packed_token_offset//frame_tokens' if chrono else 'native_absolute_frame_id',spatial_policy='native_unchanged',
            current_temporal_slots=list(range(len(sink)+budget,len(sink)+budget+len(current))) if chrono else list(current),
            packed_source_frame_ids=list(sink+selected_sources+current),
            source_token_map_sha256=source_sha,token_budget=total,history_tokens=budget*ft)
        if allocation.local:
            position.update(local_window_ids=list(local_window), local_budget_frames=allocation.local,
                            retrieved_history_frames=retrieved_budget)
        pending = dict(context=context,current_ids=current,decision=decision,selected=selected_sources,
            consumed=False,consumer_bank=consumer_bank,transfer_ids=transfer_ids,
            direct_source=direct_source,gpu_krope_plan=gpu_krope_plan,
            full_gpu_k=full_k,full_gpu_v=full_v,
            fixed_prefix=geometry.prefix,position=position,resident_kv=resident)
        self.pending[layer]=pending
        if decision:
            camera_only = getattr(self, 'settings', {}).get('retrieval_ablation') == 'camera-only'
            self.host.trace('draftkv_camera_only_packing' if camera_only else 'draftkv_block_topk',
                layer=layer,chunk=context.start_frame,
                forward_id=context.forward_id,step=self.denoise_step,clean=context.clean,
                candidate_ids=list(self._camera_chunk_cache[context.start_frame][2] if
                    not camera_only and hasattr(self,'_camera_chunk_cache') and context.start_frame in self._camera_chunk_cache
                    else candidates),
                softmax_domain=('not_applicable_camera_only' if camera_only else
                    ('candidate_subset_softmax_including_true_latest_P' if hasattr(self,'_camera_chunk_cache')
                     else 'all_nonsink_history_including_protected')),
                policy_name=('a2_camera_only_whole_chunks' if camera_only
                             else f'direct_block_topk_p{configured_protection(self.host.spec.backbone)}'),
                protected_frame_target=configured_protection(self.host.spec.backbone),
                recent_exclusion_gap=0,frame_importance_used=False,
                config=self.importance.to_dict(),config_sha256=self.importance_sha256,
                block_scores=list(decision.block_scores),protected_ids=list(protected),
                remote_token_budget=decision.remote_token_budget,
                remote_token_count=sum(len(ts) for f,ts in history if f not in local),
                ranked_blocks=[dict(frame=f,block=b,importance=s,available_tokens=a,taken_tokens=n)
                               for f,b,s,a,n in decision.ranked_blocks],
                partial_boundary_block=decision.partial_boundary_block,
                selected_source_tokens=[dict(frame=f,tokens=list(ts)) for f,ts in history if f not in local],
                remote_source_frame_ids=list(remote),history_token_count=budget*ft,
                source_token_map_sha256=source_sha,chronological_source_tokens=True,
                requested_history_frames=retrieved_budget,protected_frame_count=len(protected),
                remote_equivalent_frames=retrieved_budget-len(protected),frame_tokens=ft,
                transfer_page_ids=list(transfer_ids),
                threshold_z=None,relative_floor=None,position=position)
        self.host.trace('draftkv_time_position',layer=layer,start_frame=context.start_frame,
            forward_id=context.forward_id,clean=context.clean,**position)
        return AttentionSelection(q.contiguous(),k,v,sink,local,remote,current,tuple(pages),
            dict(representation='paged',resident_kv=resident,packing_policy='chronological_frame_id',packing_unit='source_token',position_map=tuple((i//ft if chrono else t//ft,t%ft) for i,t in enumerate(source_map)),
                 draftkv_plan=pending,packed_source_token_sha256=source_sha))

    def consume(self,context,selection,native_fa2):
        pending=self.pending.get(context.layer)
        if pending is None or selection.metadata.get('draftkv_plan') is not pending:
            raise BackboneError('DIRTY_HISTORY','block consumer differs from staged owner')
        resident = pending.get('resident_kv')
        if resident is not None:
            keys, values = selection.keys, selection.values
            prefix_tokens = resident.prefix_tokens
        elif pending['direct_source'] is not None:
            keys, values = pending['full_gpu_k'], pending['full_gpu_v']
            prefix_tokens = keys.shape[1] - selection.keys.shape[1]
        else:
            keys, values = selection.keys, selection.values
            prefix_tokens = 0
            if pending['fixed_prefix'] is not None:
                import torch
                pk, pv = pending['fixed_prefix']
                prefix_tokens = pk.shape[1]
                keys = torch.cat((pk, keys), 1).contiguous()
                values = torch.cat((pv, values), 1).contiguous()
        p={**self.binding.parameters,'sink_tokens':prefix_tokens+len(selection.sink_ids)*context.frame_tokens,
           'frame_tokens':context.frame_tokens}
        binding=replace(self.binding,parameters=p)
        physical_sp=int(self.host.plan.get('sp',context.sp))
        native_config=self._native_runtime_config(context)
        def execute(query):
            if native_config is not None:
                from draftkv.streaming.table2_native import run_direct_native
                if pending['transfer_ids'] and pending['direct_source'] is None:
                    raise BackboneError('KV_CONSTRAINT','Table2 native runtime requires direct pinned source')
                return run_direct_native(query,keys,values,
                    source=pending['direct_source'],rotation=pending['gpu_krope_plan'],
                    binding=binding,cache=self._b01_pinned_staging,
                    config=native_config,family=self.host.spec.family,
                    **({'resident_kv': resident} if resident is not None else {}))
            if pending['transfer_ids']:
                return run_layer(query, keys, values,
                    (None if pending['direct_source'] is not None else pending['consumer_bank']),
                    pending['transfer_ids'], binding=binding, layer=context.layer,
                    layers=self.host.spec.layers, sp=physical_sp,
                    staging_cache=self._b01_pinned_staging,
                    **({'direct_source':pending['direct_source'],
                        'gpu_krope_plan':pending['gpu_krope_plan']}
                       if pending['direct_source'] is not None else {}))
            return run_resident(query,keys,values,binding=binding,
                layer=context.layer,layers=self.host.spec.layers,sp=physical_sp)
        if self.host.plan.get('sp_mode')=='query_sequence_replicated_kv' and physical_sp>1:
            from draftkv.backbones.sequence_parallel import query_sequence_attention
            holder={}
            def local_compute(query):
                result,holder['metrics']=execute(query)
                return result
            output,sp_trace=query_sequence_attention(selection.query,physical_sp,local_compute)
            metrics=holder['metrics'];metrics['sequence_parallel']=sp_trace
        else:
            output,metrics=execute(selection.query)
        pending['consumed']=True
        profile=metrics.pop('profile')
        event = 'draftkv_b01_attention' if pending['transfer_ids'] else 'draftkv_b01_resident'
        receipt = dict(event=event,
            layer=context.layer,start_frame=context.start_frame,retrieval_granularity='block',
            selected_frame_ids=(None if pending.get('device_archive') else list(pending['selected'])),
            clean=context.clean,forward_id=context.forward_id,denoise_step=self.denoise_step,
            transfer_page_ids=list(pending['transfer_ids']),
            source_frame_ids=(None if pending.get('device_archive') else list(pending['selected'])),
            selected_ids_device_resident=(bool(pending.get('device_archive'))
                and not pending['position'].get('cpu_krope_selected_only_bridge',False)),
            source_token_map_sha256=pending['position']['source_token_map_sha256'],
            checkpoint_sha=self.checkpoint_sha,profile_sha=profile.get('profile',profile).get('sha256'),
            fixed_prefix_tokens=prefix_tokens,**metrics)
        if receipt.get('consumer_kind') == 'native_flash_attention':
            from draftkv.streaming.table2_native import native_receipt_identity
            identity_id,identity=native_receipt_identity(receipt)
            receipt['native_consumer_id']=identity_id
            identities=getattr(self,'_native_consumer_headers',set())
            if identity_id not in identities:
                self.host.trace('draftkv_native_consumer_identity',identity_id=identity_id,identity=identity)
                identities.add(identity_id)
                self._native_consumer_headers=identities
        if pending.get('device_archive'):
            from .device_archive_audit import stage
            stage(self,context,selection,receipt)
        else:
            self.host.trace(receipt.pop('event'),**receipt)
        return output

    def after_attention(self,context,selection):
        if self.done | {context.layer} == self.expected_layers:
            from .device_archive_audit import flush
            flush(self)
        return super().after_attention(context,selection)

    def rollback(self):
        self._device_audits = []
        return super().rollback()
