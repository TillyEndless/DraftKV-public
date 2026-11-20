"""Explicit experimental owner, sharing production selection, packing and commit ABI.

A0 returns the exact production factory. Other profiles extend the canonical
block owner via narrow seams; no model, FlashAttention or baseline is copied.
FullKV bypasses scoring/Top-K entirely. Residency never enlarges selected context.
"""
from copy import copy
from dataclasses import replace

from draftkv.algorithms.draftkv.block_backbone import BlockDraftKVHook
from draftkv.algorithms.draftkv.block_policy import applied_protection
from draftkv.backbones.contracts import BackboneError
from draftkv.retrieval.block_topk_p3 import BlockDecision, block_probabilities, select_blocks
from .config import AblationError, resolve
from .instrumentation import Instrumentation
from .scheduler import Page, Route, make_scheduler


def create_ablation_hook(core):
    from pathlib import Path
    import hashlib
    source = core.plan.get('ablation_source')
    if not isinstance(source, dict) or not Path(source.get('path', '')).is_file():
        raise AblationError('ablation requires an explicit configuration file and frozen SHA256')
    if hashlib.sha256(Path(source['path']).read_bytes()).hexdigest() != source.get('sha256'):
        raise AblationError('ablation configuration file changed after planning')
    from .config import read_config
    if resolve(read_config(source['path'])) != core.plan['ablation']:
        raise AblationError('worker ablation fields differ from explicit config file')
    expected = core.plan.get('ablation_expected_backend')
    if expected is not None:
        import torch
        from .config import resolve_attention_backend
        actual = resolve_attention_backend(torch.cuda.get_device_capability(), stable_backend='native_FA2' if expected=='native_FA2' else 'production_b01')
        if actual != expected:
            raise AblationError(f'actual architecture backend {actual} differs from planned {expected}')
    config = resolve({'ablation': core.plan['ablation']})
    if config['profile'] == 'draftkv_full':
        from draftkv.algorithms.draftkv.entry import _create_production_hook
        hook = _create_production_hook(core)
        return ObservedProductionHook(hook, core, config) if callable(getattr(hook, 'route', None)) else hook
    if config['retrieval']['selector'] == 'all_history':
        return MechanismAblationHook(core, config)
    if core.plan['algorithm_plugin'].get('config', {}).get('gpu_device_route'):
        from .device_runtime import DeviceMechanismAblationHook
        return DeviceMechanismAblationHook(core, config)
    return MechanismAblationHook(core, config)


class ObservedProductionHook:
    """Instrumentation-only proxy; original factory/owner/components execute A0."""
    def __init__(self, hook, host, config):
        self.hook, self.host, self.config = hook, host, config
        self.ranges = Instrumentation(nvtx=bool(host.plan.get('ablation_nvtx', False)))

    def __getattr__(self, name):
        return getattr(self.hook, name)

    def route(self, context):
        with self.ranges.range('route'):
            selection = self.hook.route(context)
        from .reporting import record_route
        record_route(self.host, self.config, context, selection)
        return selection

    def consume(self, context, selection, native_fa2):
        with self.ranges.range('attention'):
            return self.hook.consume(context, selection, native_fa2)


class MechanismAblationHook(BlockDraftKVHook):
    def __init__(self, host, config):
        # Do not silently turn an accelerated provider into a different baseline.
        # Its device-only route representation needs its own selector/residency
        # implementation. The canonical block provider is the controlled adapter
        # for mechanism experiments; A0 accepts every production provider.
        settings = host.plan['algorithm_plugin'].get('config', {})
        if config['retrieval']['selector'] != 'all_history' and (settings.get('gpu_device_route') or settings.get('gpu_prepack')):
            raise AblationError('Mechanism profiles require the canonical block provider; accelerated device/prepacked route is not yet supported. Use an unmodified canonical production plan for paired A0/variant runs.')
        self.ablation = config
        super().__init__(host)
        self.ranges = Instrumentation(nvtx=bool(host.plan.get('ablation_nvtx', False)))
        self.fullkeys = {}
        self.scheduler = self.transport = None
        self.position_policy = config['retrieval']['position_mode']
        self._ablation_ids = None
        self._route_started = False
        # FullKV does not store or use compact descriptors; M2 also avoids bank.
        self.use_bank = config['history']['searchable_index'] == 'draft_bank'
        self.use_draft_descriptors = self.use_bank

    @property
    def gpu_history(self):
        return self.ablation['history']['residency'] == 'gpu'

    def _history_budget(self, context, candidates, budget):
        mode = self.ablation['retrieval']['selector']
        if mode == 'all_history':
            horizon = self.ablation['history']['max_frames']
            return len(candidates) if horizon is None else min(horizon, len(candidates))
        if mode == 'local_only' and self.ablation['history']['local_window_mode'] == 'production':
            return applied_protection(self.host.spec.backbone, budget) if budget else 0
        return budget

    def _select_history(self, context, candidates, budget, h, w):
        import torch
        mode = self.ablation['retrieval']['selector']
        if not candidates or not budget:
            return None
        if mode in ('all_history', 'local_only'):
            selected = candidates[-budget:]
            ids = tuple((f, tuple(range(h*w))) for f in selected)
            # All selected entries are guaranteed context here, no Top-K call.
            return BlockDecision(selected, (), ids, (), (), budget, 0, False)
        with self.ranges.range('draft_score'):
            if mode == 'draft':
                prob = block_probabilities(context.query,
                    self._scoring_descriptors(context.layer, candidates, context.query.device), h, w, config=self.importance)
            elif mode == 'exact_region_relevance':
                from .oracle import ExactRegionRelevanceOracle
                source = self.fullkeys.get(context.layer) if self.ablation['history']['fullkey_residency'] == 'gpu' else None
                keys = torch.cat([(source[f] if source is not None else self.committed[context.layer][f][0]).to(context.query.device)
                                  for f in candidates], 0).unsqueeze(0)
                prob = ExactRegionRelevanceOracle().score(context.query, keys, h, w, config=self.importance)
            else:
                raise AblationError(f'unimplemented selector {mode}')
            scores = prob.float().sum(2).sum(1)
            if context.sp > 1:
                import torch.distributed as dist
                if not dist.is_initialized() or dist.get_world_size() != context.sp:
                    raise AblationError('exact native SP head group required')
                dist.all_reduce(scores)
            scores = scores / (prob.shape[1]*prob.shape[2]*context.sp)
        with self.ranges.range('index_select'):
            if self.ablation['retrieval']['granularity'] == 'block':
                return select_blocks(scores[0], candidates, budget,
                    query_start=context.start_frame, grid_h=h, grid_w=w,
                    protect_recent=applied_protection(self.host.spec.backbone, budget),
                    pool_h=self.importance.pool_h, pool_w=self.importance.pool_w)
            # SAME draft block probabilities; aggregate block mass into frames.
            # Rank whole frames after retaining the identical protected P set.
            frame_scores = scores[0].reshape(len(candidates), -1).sum(-1).detach().cpu().tolist()
            p = applied_protection(self.host.spec.backbone, budget)
            protected = candidates[-p:] if p else ()
            remote = tuple(sorted(sorted((f for f in candidates if f not in protected),
                key=lambda f: (-frame_scores[candidates.index(f)], f))[:budget-p]))
            selected = tuple(sorted(protected+remote))
            return BlockDecision(protected, remote, tuple((f, tuple(range(h*w))) for f in selected),
                                 (), tuple(scores[0].detach().cpu().tolist()), budget, (budget-p)*h*w, False)

    def _history_staging_device(self, context):
        return context.query.device if self.gpu_history else 'cpu'

    def _native_runtime_config(self, context):
        cfg = super()._native_runtime_config(context)
        if cfg is not None and self.ablation['attention']['prefetch_schedule']=='serial':
            return {**cfg, 'schedule':'serial_barrier'}
        return cfg

    def _native_serial(self, context):
        return (self.ablation['attention']['prefetch_schedule']=='serial'
                and self._native_runtime_config(context) is not None)

    def _use_direct_history(self, context):
        # Explicit page schedulers own their transfers and need the real page
        # bank. Native serial instead changes only HeadGroupPlan's barrier.
        return (self.ablation['attention']['prefetch_schedule']=='production'
                or self._native_serial(context))

    def _validate_sink(self, pair):
        expected = self.host.plan.get('ablation_device_type', 'cuda') if self.gpu_history else 'cpu'
        if any(t.device.type != expected for t in pair):
            raise AblationError('ablation persistent history residency mismatch')

    def validation_context(self, context):
        if self.ablation['retrieval']['selector'] == 'all_history':
            return replace(context, window_frames=max(context.window_frames, self.pending[context.layer]['position']['token_budget']//context.frame_tokens))
        return context

    def begin_forward(self, **kwargs):
        super().begin_forward(**kwargs)
        self._route_started = False
        if self.scheduler:
            self.scheduler.on_denoise_step_start(self.denoise_step,
                epoch=(self.branch, self.next_frame, self.commit_generation))

    def _route_full_history(self, context):
        import torch
        from draftkv.backbones.contracts import AttentionSelection, FramePage
        self._verify_context(context)
        from draftkv.streaming.canonical_archive import ensure_archive
        ensure_archive(self,context)
        self.active['forward_id'] = context.forward_id
        bank = self.committed.get(context.layer, {})
        horizon = self.ablation['history']['max_frames']
        expected = (set(range(self.next_frame)) if horizon is None else
                    set(range(min(context.sink_frames,self.next_frame))) |
                    set(range(max(context.sink_frames,self.next_frame-horizon),self.next_frame)))
        if set(bank) != expected:
            raise BackboneError('DIRTY_HISTORY', 'FullKV clean frontier/horizon mismatch')
        sink = tuple(range(min(context.sink_frames, context.start_frame)))
        eligible = tuple(f for f in bank if context.sink_frames <= f < context.start_frame)
        horizon = self.ablation['history']['max_frames']
        selected = eligible if horizon is None else eligible[-horizon:]
        current = tuple(range(context.start_frame, context.start_frame+context.current_frames))
        # Whole native frames; no descriptor, score, sorting, Top-K, token map,
        # CPU gather, Chrono rebasing or source-map hashing on this reference.
        order = sink+selected
        keys = torch.cat([*(bank[f][0].unsqueeze(0) for f in order), context.key],1).contiguous()
        values = torch.cat([*(bank[f][1].unsqueeze(0) for f in order), context.value],1).contiguous()
        pages = tuple(FramePage(f,f,i*context.frame_tokens,(i+1)*context.frame_tokens)
                      for i,f in enumerate(order+current))
        pending = dict(context=context,current_ids=current,selected=selected,consumed=False,
                       transfer_ids=(),consumer_bank={},fixed_prefix=context.fixed_prefix,
                       position=dict(history_budget=len(selected),token_budget=keys.shape[1]))
        self.pending[context.layer] = pending
        return AttentionSelection(context.query,keys,values,sink,selected,(),current,pages,
                                  dict(draftkv_plan=pending,full_history_native_reference=True))

    def route(self, context):
        from .planning import check_memory
        check_memory(self.ablation, self.host.spec, history_frames=self.next_frame,
                     frame_tokens=context.frame_tokens)
        # Production pack and Chrono processing remain inside measured route.
        with self.ranges.range('route'), self.ranges.range('gpu_gather' if self.gpu_history else 'cpu_pack'):
            result = (self._route_full_history(context) if self.ablation['retrieval']['selector'] == 'all_history'
                      else super().route(context))
        from .reporting import record_route
        record_route(self.host, self.ablation, context, result)
        pending = self.pending[context.layer]
        if self.position_policy == 'native_post_rope' and context.fixed_prefix is not None:
            pending['fixed_prefix'] = context.fixed_prefix
        if (self.scheduler is None and self.ablation['attention']['prefetch_schedule'] != 'production'
                and not self._native_serial(context)):
            from .transport import TorchPageTransport
            self.transport = TorchPageTransport(context.query.device, self.ranges)
            self.scheduler = make_scheduler(self.ablation['attention']['prefetch_schedule'], self.transport,
                                            layers=self.host.spec.layers)
            self.scheduler.on_denoise_step_start(self.denoise_step,
                epoch=(self.branch, self.next_frame, self.commit_generation))
        if self.scheduler:
            self.scheduler.on_layer_start(context.layer)
            decision = pending['decision']
            tokens = tuple(f*context.frame_tokens+t for f, ts in decision.token_indices for t in ts) if decision else ()
            pages = []
            for page_id in pending['transfer_ids']:
                a, b = page_id*context.frame_tokens, (page_id+1)*context.frame_tokens
                source = tokens[a:b]
                # Include source order, absolute clean generation and actual
                # compact destination slot: corrected pages cannot alias stale K.
                identity = (self.branch, self.commit_generation, context.layer, source,
                            len(result.sink_ids)+page_id if self.position_policy == 'chrono_compact' else None)
                pair = pending['consumer_bank'][page_id]
                pages.append(Page(identity, sum(t.numel()*t.element_size() for t in pair), pair))
            pending['ablation_route'] = Route(context.layer, tuple(pages))
            self.scheduler.on_route_ready(pending['ablation_route'])
        return result

    def consume(self, context, selection, native_fa2):
        import torch
        pending = self.pending[context.layer]
        # Canonical host production consumer unchanged for selector ablations.
        if not self.gpu_history and self.scheduler is None:
            with self.ranges.range('attention'):
                return super().consume(context, selection, native_fa2)
        if self.scheduler:
            ready = self.scheduler.on_attention_start()
            pairs = [ticket.pair for _, ticket in ready]
        else:
            pairs = [pending['consumer_bank'][i] for i in pending['transfer_ids']]
        offset = len(selection.sink_ids)*context.frame_tokens
        with self.ranges.range('gpu_gather'):
            for k, v in pairs:
                n = k.shape[0]
                selection.keys[:, offset:offset+n].copy_(k.unsqueeze(0), non_blocking=True)
                selection.values[:, offset:offset+n].copy_(v.unsqueeze(0), non_blocking=True)
                offset += n
        with self.ranges.range('attention', stream=torch.cuda.current_stream(context.query.device) if context.query.is_cuda else None):
            # Reuse the SAME production kernel for the canonical block provider,
            # with selected KV ready. FullKV uses normal architecture-native FA.
            if self.ablation['retrieval']['selector'] == 'all_history':
                from draftkv.streaming.headgroup import native_attention
                keys, values = selection.keys, selection.values
                if pending['fixed_prefix'] is not None:
                    pk, pv = pending['fixed_prefix']
                    keys, values = torch.cat((pk, keys), 1), torch.cat((pv, values), 1)
                from draftkv.streaming.device_profiles import native_version_for_plan
                version = native_version_for_plan(self.host.plan,context.query.device)
                output = native_attention(selection.query, keys, values, version=version)[0]
                pending['consumed'] = True
            else:
                transfers = pending['transfer_ids']
                pending['transfer_ids'] = ()
                try:
                    output = super().consume(context, selection, native_fa2)
                finally:
                    pending['transfer_ids'] = transfers
        if self.scheduler:
            self.scheduler.on_attention_end()
            self.host.trace('ablation_prefetch', layer=context.layer, denoise_step=self.denoise_step,
                            prediction_source='previous_denoise_same_layer_route', **self.scheduler.metrics())
        return output

    def _stage_clean_layer(self, layer, pending):
        if self.use_bank and not self.gpu_history:
            return super()._stage_clean_layer(layer, pending)
        import torch
        from draftkv.algorithms.draftkv.position import geometry_for
        from draftkv.retrieval.draftattention2_importance import descriptor
        c = pending['context']
        if not self.gpu_history and c.key.is_cuda:
            row = super()._stage_clean_layer(layer,pending)
            row['fullkeys'] = ({fid:c.key[0,i*c.frame_tokens:(i+1)*c.frame_tokens].detach().clone()
                for i,fid in enumerate(pending['current_ids'])}
                if self.ablation['history']['searchable_index']=='full_keys'
                and self.ablation['history']['fullkey_residency']=='gpu' else {})
            row['ready_event'] = torch.cuda.Event()
            row['ready_event'].record()
            return row
        rows, desc, raw, fullkeys = {}, {}, {}, {}
        for i, fid in enumerate(pending['current_ids']):
            a, b = i*c.frame_tokens, (i+1)*c.frame_tokens
            # GPU history clones detached clean BF16 K/V without CPU offload.
            pair = tuple(t[0, a:b].detach().clone() if self.gpu_history else
                         t[0, a:b].detach().to('cpu', copy=True) for t in (c.key, c.value))
            if not self.gpu_history and c.key.is_cuda:
                pair = tuple(t.pin_memory() for t in pair)
            rows[fid] = pair
            if self.use_bank:
                h, w = self.geometry[:2]
                desc[fid] = descriptor(c.key[:, a:b], h, w, capacity=self.importance.capacity,
                    pool_h=self.importance.pool_h, pool_w=self.importance.pool_w).detach()
            if self.position_policy == 'chrono_compact':
                temporal = geometry_for(c, self.host.spec.family).extract(c.raw_key[0, a:b]).detach()
                raw[fid] = temporal.clone() if self.gpu_history else temporal.to('cpu', copy=True)
            if self.ablation['history']['searchable_index'] == 'full_keys' and self.ablation['history']['fullkey_residency'] == 'gpu':
                fullkeys[fid] = c.key[0, a:b].detach().clone()
        ready = None
        if c.key.is_cuda:
            ready = torch.cuda.Event()
            ready.record()  # same caller stream for every staged layer
        return dict(rows=rows, descriptors=desc, raw=raw, fullkeys=fullkeys, ready_event=ready,
                    nbytes=sum(t.numel()*t.element_size() for pair in rows.values() for t in pair), registered=False)

    def commit_clean(self, *, timestep):
        if self.use_bank and not self.gpu_history:
            super().commit_clean(timestep=timestep)
        else:
            if (self.active is None or not self.active['clean'] or timestep is not self.active['timestep']
                    or self.done != self.expected_layers or set(self.pending) != self.done
                    or any(not p['consumed'] for p in self.pending.values())):
                raise BackboneError('DIRTY_HISTORY', 'ablation clean commit requires entire successful model')
            staged = {l: p['cpu_stage'] for l, p in self.pending.items()}
            for row in staged.values():
                if row.get('completion') is not None:
                    row['completion'].wait()
            events = [row['ready_event'] for row in staged.values() if row.get('ready_event') is not None]
            if events:
                events[-1].synchronize()  # one fence, before any bank/frontier publication
            ids = [p['current_ids'] for p in self.pending.values()]
            if any(x != ids[0] for x in ids):
                raise BackboneError('DIRTY_HISTORY', 'clean layer ID sets differ')
            for l, row in staged.items():
                self.committed[l] = {**self.committed.get(l, {}), **row['rows']}
                if self.use_bank:
                    self.descriptors[l] = {**self.descriptors.get(l, {}), **row['descriptors']}
                self.time_coordinates.raw_temporal[l] = {**self.time_coordinates.raw_temporal.get(l, {}), **row['raw']}
                self.fullkeys[l] = {**self.fullkeys.get(l, {}), **row.get('fullkeys', {})}
            if not self.active['recache']:
                self.next_frame = ids[0][-1]+1
                self.reserved += sum(row['nbytes'] for row in staged.values())
            if self.ablation['retrieval']['selector'] == 'all_history' and self.ablation['history']['max_frames'] is not None:
                horizon = self.ablation['history']['max_frames']
                from draftkv.backbones.budgets import effective_budget
                sink = effective_budget(self.host.spec,'draftkv').sink
                retain = set(range(min(sink,self.next_frame))) | set(range(max(sink,self.next_frame-horizon),self.next_frame))
                self.committed = {l: {f: pair for f,pair in rows.items() if f in retain}
                                  for l,rows in self.committed.items()}
                self.reserved = sum(t.numel()*t.element_size() for rows in self.committed.values()
                                    for pair in rows.values() for t in pair)
            self.commit_generation += 1
            self.host.trace('ablation_clean_commit', frontier=self.next_frame, residency=self.ablation['history']['residency'],
                            searchable_index=self.ablation['history']['searchable_index'], kv_bytes=self.reserved)
            self.pending, self.active, self.done = {}, None, set()
        if self.scheduler:
            # New clean generation invalidates all old prediction identities.
            self.scheduler.drain()
            self.scheduler.previous.clear()
            self.transport.close()  # include outstanding wasted DMA before publication returns
            self.host.trace('ablation_prefetch_summary', **self.scheduler.metrics())
        from .instrumentation import report_envelopes
        report_envelopes(self.host, self.ranges)

    def rollback(self):
        if self.scheduler:
            self.scheduler.drain()
        if self.transport:
            self.transport.close()
        super().rollback()
