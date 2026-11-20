"""HookSession glue around existing frozen scoring/selection and canonical B01.

The mathematical implementation is imported from the reviewed retrieval package.
The owner retains native scoring history and applies Chrono-Compact only to
attention temporal coordinates, publishing whole-forward clean commits.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import os
from pathlib import Path

from draftkv.backbones.contracts import AttentionSelection, BackboneError, FramePage
from draftkv.retrieval import SelectionContext, select_v1
from draftkv.streaming.binding import NativeBinding
from draftkv.streaming.pipeline import run_layer, run_resident

# Formal P3/z3 policy. The frozen selector default remains ceil(R/2).
CANONICAL_RECENT_PROTECTION = 3
CANONICAL_SALIENT_Z = 3.0
CANONICAL_RELATIVE_FLOOR = 1.5

def canonical_protect_recent(budget: int) -> int:
    if type(budget) is not int or budget < 1:
        raise ValueError("DRAFTKV_INVALID_HISTORY_BUDGET")
    return min(CANONICAL_RECENT_PROTECTION, budget)



class DraftKVHook:
    def __init__(self, host):
        self.host, self.owner = host, self
        plan = host.plan
        self.settings = plan['algorithm_plugin'].get('config', {})
        from .experiment import archive_page_policy
        self.archive_page_policy = archive_page_policy(self.settings, getattr(host.spec, 'backbone', ''))
        from .position import ChronoCompact, require_policy
        self.position_policy = require_policy(self.settings)
        from .position import require_packing_policy
        self.packing_policy = require_packing_policy(self.settings)
        self.time_coordinates = ChronoCompact(host.spec.family)
        repo_root = Path(plan.get('repo_root', Path(__file__).resolve().parents[4]))
        if self.settings.get('gpu_prepack', False):
            from types import SimpleNamespace
            self.binding = SimpleNamespace(build_spec={})
        else:
            self.binding = NativeBinding.read(self.settings, repo_root)
        # CPU-only source-contract fixtures deliberately inject a stub with no
        # compiled build spec. Production bindings always carry a generation.
        if self.binding.build_spec.get('source_generation') is not None:
            self.binding = self.binding.prepare_runtime()
        from draftkv.retrieval.draftattention2_importance import load_importance_config
        importance_path = Path(self.settings.get(
            'importance_config', 'configs/algorithms/draftkv/scorer.json'))
        if not importance_path.is_absolute():
            importance_path = repo_root / importance_path
        self.importance_path = importance_path.resolve()
        self.importance = load_importance_config(self.importance_path)
        self.importance_sha256 = hashlib.sha256(self.importance_path.read_bytes()).hexdigest()
        expected_importance_sha = self.settings.get('importance_config_sha256')
        request = plan.get('request', {})
        if (plan.get('benchmark') == 'bench2' and request.get('execution_scope') == 'release_candidate'
                and not self.settings.get('gpu_prepack', False)):
            canonical_path = (repo_root / 'configs/algorithms/draftkv/scorer.json').resolve()
            canonical_sha = hashlib.sha256(canonical_path.read_bytes()).hexdigest()
            if expected_importance_sha is None:
                raise BackboneError('BLOCKED_ALGORITHM_SEMANTICS',
                                    'Bench2 release DraftKV requires a pinned importance config SHA256')
            if expected_importance_sha != canonical_sha:
                raise BackboneError('SOURCE_SHA_MISMATCH',
                                    'Bench2 DraftKV importance SHA256 must equal the canonical fixed8x16 scorer')
        if expected_importance_sha is not None and expected_importance_sha != self.importance_sha256:
            raise BackboneError('SOURCE_SHA_MISMATCH',
                                'DraftKV importance config SHA256 differs from deployment binding')
        # Reusable pinned producer staging for grouped B01 layouts only.
        # One session owner, exact shape and original source order per layer.
        self._b01_pinned_staging = {}
        # Lifetime of THP registered CUDA host regions matches historical KV:
        # preserve every owner until this worker exits; never overwrite pages.
        self._clean_registered_owners = []
        self.committed = {}  # layer -> absolute ID -> immutable post-native-RoPE CPU K/V
        # Complete historical K/V and raw temporal keys remain CPU-owned.
        # Draft-K descriptors are retained on their original scoring device
        # (CUDA in production), avoiding repeated historical descriptor H2D.
        self.descriptors = {}  # layer -> absolute ID -> GPU FP16 mean/max Draft-K
        self.pending = {}
        self.active = None
        self.done = set()
        self.expected_layers = frozenset(range(host.spec.layers))
        self.next_frame = 0
        self.invalidated = False
        self.geometry = None
        self.commit_generation = 0
        self.limit = None  # capacity is runtime-observed; OOM is non-gating
        self.reserved = 0
        self.forward_index = -1
        self.denoise_step = 0
        self.step_scope = None
        self.observations = None
        request = plan.get('request', {})
        self.branch = request.get('attention_branch')
        if not isinstance(self.branch, str) or not self.branch:
            raise BackboneError('BLOCKED_ALGORITHM_SEMANTICS', 'explicit native attention_branch required; CFG branches need separate model/cache owners')
        self.run_id = Path(plan.get('run_dir', '/cpu-fixture')).name
        self.case_id = request.get('case_id', 'cpu-fixture')
        self.checkpoint_sha = plan.get('checkpoint_sha256', plan.get('checkpoint_sha', ''))
        if not self.checkpoint_sha:
            raise BackboneError('BLOCKED_CHECKPOINT', 'worker must bind the actual loaded checkpoint manifest SHA')

    def begin_forward(self, *, start_frame, clean, timestep):
        if self.invalidated or (self.active is not None and self.done != self.expected_layers):
            raise BackboneError('DIRTY_HISTORY', 'previous DraftKV model transaction incomplete or failed')
        # Native LongLive prompt switches replay its protected sink at original
        # positions. Keep that clean replacement separate from a new chunk.
        recache = start_frame < self.next_frame
        if recache and not (self.host.spec.family == 'longlive' and clean and start_frame == 0):
            raise BackboneError('DIRTY_HISTORY', 'only the native LongLive clean sink replay may revisit committed IDs')
        if not recache and start_frame != self.next_frame:
            raise BackboneError('DIRTY_HISTORY', 'absolute chunk does not follow the clean committed frontier')
        scope = (start_frame, self.next_frame, recache)
        self.denoise_step = self.denoise_step + 1 if scope == self.step_scope else 0
        self.step_scope = scope
        self.forward_index += 1
        self.active = {'start_frame': start_frame, 'clean': clean, 'timestep': timestep, 'recache': recache}
        self.pending, self.done = {}, set()

    def _verify_context(self, context):
        import torch
        if (self.active is None or context.layer in self.pending or context.layer in self.done
                or context.start_frame != self.active['start_frame'] or context.clean != self.active['clean']):
            raise BackboneError('PARTIAL_NATIVE_WIRING', 'DraftKV layer does not belong to the active model transaction')
        if not 0 <= context.layer < self.host.spec.layers:
            raise BackboneError('PARTIAL_NATIVE_WIRING', 'layer outside native model')
        if (context.query.shape != context.key.shape or context.key.shape != context.value.shape
                or context.raw_query.shape != context.query.shape or context.raw_key.shape != context.key.shape
                or context.query.shape[0] != 1 or context.query.shape[-1] != 128
                or context.query.dtype != torch.bfloat16):
            raise BackboneError('KV_CONSTRAINT', 'exact native batch-one BF16 D128 Q/K/V required')
        h, w = (int(context.grid_sizes[0][i]) for i in (1, 2))
        if h * w != context.frame_tokens or context.key.shape[1] != context.current_frames * context.frame_tokens:
            raise BackboneError('KV_CONSTRAINT', 'explicit native spatial grid and whole unpadded frames required')
        geometry = (h, w, context.query.shape[2], context.frame_tokens, context.sp)
        if self.geometry not in (None, geometry):
            raise BackboneError('KV_CONSTRAINT', 'native grid/SP/head geometry changed inside a historical owner')
        self.geometry = geometry
        forward_geometry = (context.start_frame, context.current_frames, context.frame_tokens,
                            context.sink_frames, context.window_frames, geometry)
        if self.active.get('forward_geometry', forward_geometry) != forward_geometry:
            raise BackboneError('DIRTY_HISTORY', 'all native layers must stage identical current absolute IDs and geometry')
        self.active['forward_geometry'] = forward_geometry
        if context.current_frames > self.host.spec.chunk:
            raise BackboneError('KV_CONSTRAINT', 'current chunk exceeds the frozen native current-frame capacity')
        if self.active['recache'] and context.current_frames != context.sink_frames:
            raise BackboneError('DIRTY_HISTORY', 'LongLive sink replay must replace exactly its native protected sink')
        return h, w

    def _scoring_descriptors(self, layer, frame_ids, device):
        """Reuse persistent Draft-K descriptors directly on the scoring device."""
        rows = tuple(self.descriptors[layer][fid] for fid in frame_ids)
        if not rows or any(item.device != device for item in rows):
            raise BackboneError('KV_CONSTRAINT',
                                'DraftKV historical descriptors must match the scoring device')
        return rows

    def route(self, context):
        import torch
        from draftkv.retrieval.draftattention2_importance import frame_mass
        from draftkv.retrieval.hierarchical_v1 import reduce_frame_mass
        h, w = self._verify_context(context)
        current = tuple(range(context.start_frame, context.start_frame + context.current_frames))
        layer = context.layer
        self.active['forward_id'] = context.forward_id
        bank = self.committed.get(layer, {})
        if set(bank) != set(range(self.next_frame)):
            raise BackboneError('DIRTY_HISTORY', 'per-layer bank differs from the whole-model clean frontier')
        sink = tuple(range(min(context.sink_frames, context.start_frame)))
        candidates = tuple(fid for fid in bank if context.sink_frames <= fid < context.start_frame)
        from draftkv.backbones.budgets import bound_budget
        capacity = bound_budget(self.host, "draftkv").retrieved
        budget = min(capacity, len(candidates))
        if capacity < 0 or (candidates and budget < 1):
            raise BackboneError('KV_CONSTRAINT', 'native physical window leaves no historical budget')
        decision = None
        if candidates:
            # DraftAttention2-derived scorer: the exact post-QKNorm/post-RoPE
            # Q/K at the native seam are pooled before time compaction. Mean and max
            # draft probabilities are normalized separately then fused. Full K/V
            # remain exact BF16; no mixed-precision attention is introduced.
            masses = frame_mass(
                context.query, self._scoring_descriptors(layer, candidates, context.query.device),
                h, w, config=self.importance)
            if context.sp > 1:
                import torch.distributed as dist
                if not dist.is_initialized() or dist.get_world_size() != context.sp:
                    raise BackboneError('KV_CONSTRAINT', 'retrieval must use the exact native SP head-shard group')
                shards = [torch.empty_like(masses) for _ in range(context.sp)]
                dist.all_gather(shards, masses)
                masses = torch.cat(shards, dim=1)
            scores = reduce_frame_mass(masses)
            decision = select_v1(
                scores, candidates, budget, query_start=context.start_frame,
                sink_ids=sink, protect_recent=canonical_protect_recent(budget),
                salient_z=CANONICAL_SALIENT_Z, relative_floor=CANONICAL_RELATIVE_FLOOR,
                context=SelectionContext(layer=layer, chunk=context.start_frame))
            # Ranked IDs are used only to determine the selected SET and for
            # diagnostic traces. Every physical KV source must instead follow
            # chronological Frame ID order, matching the Chrono-Compact RoPE
            # slots passed to the B01 Producer/Consumer.
            selected = decision.chronological
            if selected != tuple(sorted(decision.ranked)):
                raise BackboneError('KV_CONSTRAINT', 'DraftKV chronological selection differs from ranked selection')
            # Hierarchical V1 remains the frame-budget policy; only its score
            # source changes to consumer-aligned DraftAttention2 importance.
            self.host.trace('draftkv_importance', layer=layer, chunk=context.start_frame,
                            config=self.importance.to_dict(), config_sha256=self.importance_sha256,
                            candidate_ids=list(candidates), frame_scores=list(decision.frame_scores))
            self.host.trace('draftkv_hierarchical_v1', **decision.trace.to_dict())
        else:
            selected = ()
        count = len(sink) + len(selected) + len(current)
        shape = (1, count * context.frame_tokens, *context.query.shape[2:])
        k, v = (torch.empty(shape, dtype=context.key.dtype, device=context.key.device) for _ in range(2))
        for offset, fid in enumerate(sink):
            if fid not in bank:
                raise BackboneError('DIRTY_HISTORY', 'CPU committed sink missing from this layer')
            begin = offset * context.frame_tokens
            # The native Sink remains semantically protected, but is NOT resident
            # on GPU across forwards. Copy into this layer's temporary KV only.
            for destination, source in zip((k, v), bank[fid]):
                if source.device.type != 'cpu':
                    raise BackboneError('KV_CONSTRAINT', 'persistent sink KV must be on CPU')
                destination[:, begin:begin + context.frame_tokens].copy_(source.unsqueeze(0))
        start = (len(sink) + len(selected)) * context.frame_tokens
        k[:, start:].copy_(context.key); v[:, start:].copy_(context.value)
        # Only clean native current Q/K/V can enter the staged bank. Noisy
        # forwards never allocate or overwrite a historical full-resolution page.
        self.pending[layer] = {'context': context, 'current_ids': current, 'decision': decision,
                               'selected': selected, 'consumed': False}
        compact_query, consumer_bank, fixed_prefix, position = self.time_coordinates.route(
            context, k, sink, selected, current, bank)
        self.pending[layer].update(consumer_bank=consumer_bank, fixed_prefix=fixed_prefix, position=position)
        self.host.trace('draftkv_time_position', layer=layer, start_frame=context.start_frame,
                        forward_id=context.forward_id, clean=context.clean, **position)
        local = decision.local if decision else ()
        remote = tuple(fid for fid in selected if fid not in set(local))
        order = sink + selected + current
        if order != tuple(sorted(order)):
            raise BackboneError('KV_CONSTRAINT', 'DraftKV physical packing must follow chronological Frame IDs')
        if position['packed_source_ids'] != list(order) or position['packed_temporal_slots'] != list(range(len(order))):
            raise BackboneError('KV_CONSTRAINT', 'DraftKV physical packing and Chrono-Compact slots disagree')
        pages = tuple(FramePage(fid, fid, rank * context.frame_tokens, (rank + 1) * context.frame_tokens)
                      for rank, fid in enumerate(order))
        return AttentionSelection(compact_query, k, v, sink, local, remote, current,
                                  pages, {'draftkv_plan': self.pending[layer], 'packed_frame_ids': order,
                                          'selected_frame_ids': selected,
                                          'importance_ranked_frame_ids': decision.ranked if decision else (),
                                          'packing_policy': self.packing_policy,
                                          'time_position': position})

    def consume(self, context, selection, native_fa2):
        pending = self.pending.get(context.layer)
        if pending is None or selection.metadata.get('draftkv_plan') is not pending:
            raise BackboneError('DIRTY_HISTORY', 'consumer selection differs from the staged owner')
        keys, values = selection.keys, selection.values
        prefix_tokens = 0
        if pending['fixed_prefix'] is not None:
            import torch
            pk, pv = pending['fixed_prefix']
            prefix_tokens = pk.shape[1]
            keys, values = torch.cat((pk, keys), 1).contiguous(), torch.cat((pv, values), 1).contiguous()
        p = {**self.binding.parameters, 'sink_tokens': prefix_tokens + len(selection.sink_ids) * context.frame_tokens,
             'frame_tokens': context.frame_tokens}
        binding = replace(self.binding, parameters=p)
        selected = pending['selected']
        expected_order = selection.sink_ids + selected + selection.current_ids
        if (selected != tuple(sorted(selected))
                or selection.metadata.get('packed_frame_ids') != expected_order
                or selection.metadata.get('packing_policy') != self.packing_policy
                or expected_order != tuple(sorted(expected_order))):
            raise BackboneError('KV_CONSTRAINT', 'DraftKV nonchronological B01 handoff forbidden')
        physical_sp = int(self.host.plan.get('sp', context.sp))
        sequence_parallel = (self.host.plan.get('sp_mode') == 'query_sequence_replicated_kv'
                             and physical_sp > 1)

        def execute(query):
            if selected:
                return run_layer(query, keys, values,
                                 pending['consumer_bank'], selected,
                                 binding=binding, layer=context.layer,
                                 layers=self.host.spec.layers, sp=physical_sp,
                                 staging_cache=self._b01_pinned_staging)
            # Warmup/all-fit has no fetched history: use the original ungated
            # B01 Consumer, without fabricating a Frame ID/ProducerPlan/H2D.
            return run_resident(query, keys, values, binding=binding,
                                layer=context.layer, layers=self.host.spec.layers,
                                sp=physical_sp)

        if sequence_parallel:
            from draftkv.backbones.sequence_parallel import query_sequence_attention
            holder = {}
            def local_compute(query):
                result, observed = execute(query)
                holder['metrics'] = observed
                return result
            output, sp_trace = query_sequence_attention(selection.query, physical_sp, local_compute)
            metrics = holder['metrics']
            metrics['sequence_parallel'] = sp_trace
        else:
            output, metrics = execute(selection.query)
        pending['consumed'] = True
        decision = pending['decision']
        profile = metrics.pop('profile')
        self._observe(context, selected, decision, metrics, profile)
        self.host.trace('draftkv_b01_attention' if selected else 'draftkv_b01_resident',
                        layer=context.layer, start_frame=context.start_frame, clean=context.clean,
                        run_id=self.run_id, case_id=self.case_id, branch=self.branch,
                        forward_index=self.forward_index, forward_id=context.forward_id, denoise_step=self.denoise_step,
                        commit_generation=self.commit_generation,
                        sink_recache=self.active['recache'], selected_frame_ids=list(selected),
                        local_ids=list(decision.local) if decision else [],
                        salient_ids=list(decision.salient) if decision else [],
                        profile_sha=profile.get('profile', profile).get('sha256'),
                        checkpoint_sha=self.checkpoint_sha, **metrics)
        return output

    def _observe(self, context, selected, decision, metrics, profile):
        """Append actual event values using the existing telemetry journal API.

        CPU perf-counter and per-sample CUDA event origins remain DIFFERENT
        domains. No unmeasured calibration, Nsight SHA or active MMA is invented.
        materialize_journal requires externally supplied capture/calibration first.
        """
        from draftkv.telemetry import ObservationWriter
        cpu, cuda = metrics.get('cpu_perf_counter_ns'), metrics.get('cuda_relative_ms')
        if not cpu or not cuda:
            return
        folder = Path(self.host.plan['run_dir']) / 'traces'
        if self.observations is None:
            branch_id = hashlib.sha256(self.branch.encode()).hexdigest()[:12]
            self.observation_path = folder / f'draftkv_observations_rank{os.environ.get("RANK", "0")}_{branch_id}.jsonl'
            # Pending overlap observations are evidence, but per-event fsync-like
            # flushes must not serialize every attention layer on shared storage.
            # A bounded record batch is flushed periodically and always at close/atexit.
            self.observations = ObservationWriter(self.observation_path,
                                                   run_id=self.run_id, flush_every=128)
        sample_id = f'{self.branch}:forward{self.forward_index}:layer{context.layer}'
        cpu_domain = f'perf_counter_ns:{os.getpid()}'
        cuda_domain = f'cuda_event_relative:{sample_id}'
        if metrics['producer_plan_id'] is not None:
            for event, key in (('producer_plan_begin', 'plan_begin'), ('producer_plan_end', 'plan_end')):
                self.observations.mark(sample_id, event, cpu[key], clock_domain=cpu_domain, event_unit='ns')
            for event, key in (('gpu_h2d_begin', 'h2d_begin'), ('gpu_h2d_end', 'h2d_end'), ('frontier_ready', 'h2d_end')):
                self.observations.mark(sample_id, event, cuda[key], clock_domain=cuda_domain, event_unit='ms')
        for event, key in (('consumer_kernel_begin', 'consumer_begin'),
                           ('consumer_kernel_end', 'consumer_end'), ('final_join', 'join')):
            self.observations.mark(sample_id, event, cuda[key], clock_domain=cuda_domain, event_unit='ms')
        self.host.trace('draftkv_telemetry_pending', run_id=self.run_id, case_id=self.case_id,
                        branch=self.branch, observation_journal=self.observation_path.name,
                        sample_id=sample_id, layer=context.layer, chunk=context.start_frame,
                        denoise_step=self.denoise_step, forward_index=self.forward_index, forward_id=context.forward_id,
                        commit_generation=self.commit_generation, selected_frame_ids=list(selected),
                        local_ids=list(decision.local) if decision else [],
                        salient_ids=list(decision.salient) if decision else [],
                        history_frame_ids=list(range(self.next_frame)),
                        producer_plan_id=metrics['producer_plan_id'],
                        consumer_kind=metrics['consumer_kind'], kernel_symbol=metrics['consumer_symbol'],
                        source_sha=metrics['source_sha'], kernel_sha=metrics.get('kernel_sha'),
                        extension_sha256=metrics.get('extension_sha256'),
                        profile_sha=profile['profile']['sha256'], checkpoint_sha=self.checkpoint_sha,
                        cpu_clock_domain=cpu_domain, cuda_clock_domain=cuda_domain,
                        critical_path={'definition': 'pipeline_origin_to_final_join',
                                       'begin': {'value': cpu['prepare_begin'], 'clock_domain': cpu_domain, 'event_unit': 'ns'},
                                       'end': {'value': cuda['join'], 'clock_domain': cuda_domain, 'event_unit': 'ms'}},
                        consumer_active_math_intervals=None,
                        active_math_unavailable='event envelopes include Frontier waits; no Nsight SM activity capture',
                        baseline_reference=None, evidence_scope='PENDING_OBSERVATIONS_NEEDS_CAPTURE_AND_CLOCK_CALIBRATION')

    def after_attention(self, context, selection):
        pending = self.pending.get(context.layer)
        if pending is None or not pending['consumed']:
            raise BackboneError('PARTIAL_NATIVE_WIRING', 'DraftKV attention did not execute original B01')
        if context.clean:
            # Stage after EACH layer on every backbone, not only SolarWM.
            # Otherwise pending holds all layers' current GPU Q/K/V until the
            # whole-model commit and inflates peak GPU memory. Publication still
            # remains atomic at successful whole-model commit.
            staged = self._stage_clean_layer(context.layer, pending)
            self.pending[context.layer] = dict(current_ids=pending['current_ids'],
                                               consumed=True, cpu_stage=staged)
        self.done.add(context.layer)
        if not context.clean:
            # Keep only completion markers for a noisy transaction, never QKV.
            del self.pending[context.layer]
            if self.done == self.expected_layers:
                self.active = None

    def _stage_clean_layer(self, layer, pending):
        """Copy one completed layer privately; publish only at model commit."""
        import torch
        from draftkv.retrieval.draftattention2_importance import descriptor
        from .registered_pages import RegisteredKVSlab
        c = pending['context']; h, w = self.geometry[:2]
        archive = getattr(self, '_canonical_clean_archive', None)
        if archive is None and c.key.is_cuda:
            from draftkv.streaming.canonical_archive import ensure_archive
            archive = ensure_archive(self,c)
        # Recache must remain private until commit; it cannot overwrite an
        # existing immutable arena slice in place.
        if archive is not None and not self.active.get('recache', False):
            from .position import geometry_for
            return archive.stage(c, pending['current_ids'],
                geometry_for(c,self.host.spec.family), importance=(h,w,self.importance),
                descriptors=getattr(self,'use_draft_descriptors',True))
        rows, descriptors = {}, {}
        slab = None
        if c.key.is_cuda:
            if getattr(self, 'archive_page_policy', 'thp2m') == 'base4k':
                slab = RegisteredKVSlab(c.key, c.value, len(pending['current_ids']), c.frame_tokens, page_policy='base4k')
            else:
                slab = RegisteredKVSlab(c.key, c.value, len(pending['current_ids']), c.frame_tokens)
            self._clean_registered_owners.append(slab)
        for rank, fid in enumerate(pending['current_ids']):
            a, b = rank*c.frame_tokens, (rank+1)*c.frame_tokens
            sources = (c.key[0, a:b], c.value[0, a:b])
            pair = (slab.frame(rank) if slab is not None else
                    tuple(torch.empty_like(t, device='cpu', pin_memory=c.key.is_cuda) for t in sources))
            for dst, src in zip(pair, sources):
                dst.copy_(src, non_blocking=False)
            rows[fid] = pair
            # Descriptor returns fresh pooled FP16 storage; detach without D2H.
            if getattr(self, 'use_draft_descriptors', True):
                descriptors[fid] = descriptor(c.key[:, a:b], h, w,
                                             capacity=self.importance.capacity,
                                             pool_h=self.importance.pool_h,
                                             pool_w=self.importance.pool_w).detach()
            if (any(t.device.type != 'cpu' for t in pair)
                    or (fid in descriptors and descriptors[fid].device != c.key.device)):
                raise BackboneError('KV_CONSTRAINT',
                                    'DraftKV CPU KV / device-resident descriptor ownership mismatch')
        if getattr(self,'settings',{}).get('position_reference_a',False):
            raw={}
        else:
            temporal = self.time_coordinates.stage_clean({layer: pending})[layer]
            raw = {fid: temporal[fid] for fid in pending['current_ids']}
        nbytes = len(pending['current_ids'])*2*c.frame_tokens*c.query.shape[2]*128*2
        return dict(rows=rows, descriptors=descriptors, raw=raw,
                    nbytes=nbytes, registered=slab is not None)

    def commit_clean(self, *, timestep):
        import torch
        from draftkv.retrieval.draftattention2_importance import descriptor
        if (self.active is None or not self.active['clean'] or timestep is not self.active['timestep']
                or self.done != self.expected_layers
                or set(self.pending) != self.done or any(not p['consumed'] for p in self.pending.values())):
            raise BackboneError('DIRTY_HISTORY', 'only an entire successful native clean forward may commit')
        staged_ids = [pending['current_ids'] for pending in self.pending.values()]
        if any(ids != staged_ids[0] for ids in staged_ids[1:]):
            raise BackboneError('DIRTY_HISTORY', 'clean commit current ID sets differ across native layers')
        # Every per-layer copy remains private until ALL model layers succeed.
        staged, staged_descriptors, temporal_banks = {}, {}, {}
        additional = 0
        new_registered_owners = []
        for layer, pending in self.pending.items():
            row = pending.get('cpu_stage')
            if row is None:
                row = self._stage_clean_layer(layer, pending)
            if row.get('completion') is not None:
                row['completion'].wait()
                self.host.trace('draftkv_gpu_prepare_complete',layer=layer,**row['completion'].times())
            staged[layer], staged_descriptors[layer] = row['rows'], row['descriptors']
            temporal_banks[layer] = {**self.time_coordinates.raw_temporal.get(layer, {}), **row['raw']}
            if not self.active['recache']:
                additional += row['nbytes']
            if row['registered']:
                new_registered_owners.append(layer)
        # D2H copies above complete before any bank/descriptor/sink is visible.
        from .device_archive_route import publish as publish_device_archive
        publish_device_archive(self, staged, staged_descriptors)
        banks = {layer: {**self.committed.get(layer, {}), **rows} for layer, rows in staged.items()}
        descriptors = {layer: {**self.descriptors.get(layer, {}), **rows} for layer, rows in staged_descriptors.items()}
        self.time_coordinates.publish(temporal_banks)
        self.committed, self.descriptors = banks, descriptors
        if not self.active['recache']:
            self.next_frame = max(next(iter(staged.values()))) + 1
            self.reserved += additional
        self.commit_generation += 1
        descriptor_sample = next((value for bank in descriptors.values()
                                  for value in bank.values()), None)
        descriptor_device_type = (descriptor_sample.device.type
                                  if descriptor_sample is not None else 'none')
        descriptor_gpu_bytes = sum(
            t.numel() * t.element_size()
            for bank in descriptors.values() for t in bank.values()
            if t.is_cuda
        )
        self.host.trace('draftkv_clean_commit', start_frame=self.active['start_frame'],
                        forward_id=self.active['forward_id'], forward_index=self.forward_index,
                        sink_recache=self.active['recache'], generation=self.commit_generation,
                        committed_ids=list(next(iter(staged.values()))), layers=len(staged),
                        importance_config=self.importance.to_dict(), importance_config_sha256=self.importance_sha256,
                        pinned_history_bytes=self.reserved, committed_frontier=self.next_frame,
                        raw_temporal_key_bytes=self.time_coordinates.raw_bytes(),
                        descriptor_key_source='clean_context.key_native_post_rope',
                        full_kv_source='native_post_rope', position_policy=self.position_policy,
                        history_kv_device='cpu', history_descriptor_device=descriptor_device_type,
                        persistent_video_gpu_kv_bytes=0,
                        persistent_gpu_draft_descriptor_bytes=descriptor_gpu_bytes,
                        gpu_kv_lifetime='current_chunk_and_per_layer_temporary_attention_only',
                        archive_page_policy=getattr(self, 'archive_page_policy', 'thp2m'),
                        clean_pinned_allocator=('bounded_ahead_of_use_arena' if getattr(self,'_canonical_clean_archive',None) is not None and not self.active['recache'] else (('base4k_cuda_host_register' if getattr(self, 'archive_page_policy', 'thp2m') == 'base4k' else 'thp_cuda_host_register') if new_registered_owners
                                                else 'torch_pinned_original')),
                        clean_registered_layer_count=len(new_registered_owners))
        self.active, self.pending, self.done = None, {}, set()

    def rollback(self):
        # The canonical bridge joins successful calls; poisoned native plans and
        # mmap pin owners are quarantined until worker exit, never recycled here.
        self.invalidated = True
        self.active, self.pending, self.done = None, {}, set()

    def history_ids(self, layer):
        return tuple(sorted(self.committed.get(layer, {})))

    def contains_history_ids(self, layer, frame_ids):
        bank = self.committed.get(layer, {})
        return all(fid in bank for fid in frame_ids)

    def frame_pages(self, layer, frame_id):
        if frame_id not in self.committed.get(layer, {}):
            raise BackboneError('KV_CONSTRAINT', 'frame is not clean committed in this layer')
        tokens = self.geometry[3]
        return (FramePage(frame_id, frame_id, 0, tokens),)
