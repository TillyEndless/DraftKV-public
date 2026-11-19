# SPDX-License-Identifier: Apache-2.0
"""HookSession glue around existing frozen scoring/selection and canonical B01.

The mathematical implementation is imported from the reviewed retrieval package.
The owner retains native scoring history and applies Chrono-Compact only to
attention temporal coordinates, publishing whole-forward clean commits.
"""
from __future__ import annotations
from draftkv.runtime import profiling as pm
from draftkv.streaming.external import external_symbol, external_path
from dataclasses import replace
import hashlib
import os
from pathlib import Path
from draftkv.algorithms.draftkv.historical.orbia.backbones.contracts import AttentionSelection, BackboneError, FramePage
from draftkv.algorithms.draftkv.historical.orbia.streaming.binding import NativeBinding
from draftkv.algorithms.draftkv.historical.orbia.streaming.pipeline import run_layer, run_resident
CANONICAL_RECENT_PROTECTION = 3
CANONICAL_SALIENT_Z = 3.0
CANONICAL_RELATIVE_FLOOR = 1.5

def canonical_protect_recent(budget: int) -> int:
    if type(budget) is not int or budget < 1:
        raise ValueError('DRAFTKV_INVALID_HISTORY_BUDGET')
    return min(CANONICAL_RECENT_PROTECTION, budget)

class DraftKVHook:

    def __init__(self, host):
        (self.host, self.owner) = (host, self)
        plan = host.plan
        self.settings = plan['algorithm_plugin'].get('config', {})
        from ..policy import archive_page_policy
        self.archive_page_policy = archive_page_policy(self.settings, getattr(host.spec, 'backbone', ''))
        from .position import ChronoCompact, require_policy
        self.position_policy = require_policy(self.settings)
        from .position import require_packing_policy
        self.packing_policy = require_packing_policy(self.settings)
        self.time_coordinates = ChronoCompact(host.spec.family)
        repo_root = Path(plan.get('repo_root', Path(__file__).resolve().parents[4]))
        self.binding = NativeBinding.read(self.settings, repo_root)
        if self.binding.build_spec.get('source_generation') is not None:
            self.binding = self.binding.prepare_runtime()
        from draftkv.algorithms.draftkv.historical.orbia.retrieval.draftattention2_importance import load_importance_config
        importance_path = Path(self.settings.get('importance_config', 'configs/algorithms/draftkv/scorer.json'))
        if not importance_path.is_absolute():
            importance_path = repo_root / importance_path
        self.importance_path = importance_path.resolve()
        self.importance = load_importance_config(self.importance_path)
        self.importance_sha256 = hashlib.sha256(self.importance_path.read_bytes()).hexdigest()
        expected_importance_sha = self.settings.get('importance_config_sha256')
        request = plan.get('request', {})
        if plan.get('benchmark') == 'bench2' and request.get('execution_scope') == 'release_candidate':
            canonical_path = (repo_root / 'configs/algorithms/draftkv/scorer.json').resolve()
            canonical_sha = hashlib.sha256(canonical_path.read_bytes()).hexdigest()
            if expected_importance_sha is None:
                raise BackboneError('BLOCKED_ALGORITHM_SEMANTICS', 'Bench2 release DraftKV requires a pinned importance config SHA256')
            if expected_importance_sha != canonical_sha:
                raise BackboneError('SOURCE_SHA_MISMATCH', 'Bench2 DraftKV importance SHA256 must equal the canonical fixed8x16 scorer')
        if expected_importance_sha is not None and expected_importance_sha != self.importance_sha256:
            raise BackboneError('SOURCE_SHA_MISMATCH', 'DraftKV importance config SHA256 differs from deployment binding')
        self._b01_pinned_staging = {}
        self._clean_registered_owners = []
        self.committed = {}
        self.descriptors = {}
        self.pending = {}
        self.active = None
        self.done = set()
        self.expected_layers = frozenset(range(host.spec.layers))
        self.next_frame = 0
        self.invalidated = False
        self.geometry = None
        self.commit_generation = 0
        self.limit = None
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
        recache = start_frame < self.next_frame
        if recache and (not (self.host.spec.family == 'longlive' and clean and (start_frame == 0))):
            raise BackboneError('DIRTY_HISTORY', 'only the native LongLive clean sink replay may revisit committed IDs')
        if not recache and start_frame != self.next_frame:
            raise BackboneError('DIRTY_HISTORY', 'absolute chunk does not follow the clean committed frontier')
        scope = (start_frame, self.next_frame, recache)
        self.denoise_step = self.denoise_step + 1 if scope == self.step_scope else 0
        self.step_scope = scope
        self.forward_index += 1
        self.active = {'start_frame': start_frame, 'clean': clean, 'timestep': timestep, 'recache': recache}
        (self.pending, self.done) = ({}, set())

    def _verify_context(self, context):
        import torch
        if self.active is None or context.layer in self.pending or context.layer in self.done or (context.start_frame != self.active['start_frame']) or (context.clean != self.active['clean']):
            raise BackboneError('PARTIAL_NATIVE_WIRING', 'DraftKV layer does not belong to the active model transaction')
        if not 0 <= context.layer < self.host.spec.layers:
            raise BackboneError('PARTIAL_NATIVE_WIRING', 'layer outside native model')
        if context.query.shape != context.key.shape or context.key.shape != context.value.shape or context.raw_query.shape != context.query.shape or (context.raw_key.shape != context.key.shape) or (context.query.shape[0] != 1) or (context.query.shape[-1] != 128) or (context.query.dtype != torch.bfloat16):
            raise BackboneError('KV_CONSTRAINT', 'exact native batch-one BF16 D128 Q/K/V required')
        (h, w) = (int(context.grid_sizes[0][i]) for i in (1, 2))
        if h * w != context.frame_tokens or context.key.shape[1] != context.current_frames * context.frame_tokens:
            raise BackboneError('KV_CONSTRAINT', 'explicit native spatial grid and whole unpadded frames required')
        geometry = (h, w, context.query.shape[2], context.frame_tokens, context.sp)
        if self.geometry not in (None, geometry):
            raise BackboneError('KV_CONSTRAINT', 'native grid/SP/head geometry changed inside a historical owner')
        self.geometry = geometry
        forward_geometry = (context.start_frame, context.current_frames, context.frame_tokens, context.sink_frames, context.window_frames, geometry)
        if self.active.get('forward_geometry', forward_geometry) != forward_geometry:
            raise BackboneError('DIRTY_HISTORY', 'all native layers must stage identical current absolute IDs and geometry')
        self.active['forward_geometry'] = forward_geometry
        if context.current_frames > self.host.spec.chunk:
            raise BackboneError('KV_CONSTRAINT', 'current chunk exceeds the frozen native current-frame capacity')
        if self.active['recache'] and context.current_frames != context.sink_frames:
            raise BackboneError('DIRTY_HISTORY', 'LongLive sink replay must replace exactly its native protected sink')
        return (h, w)

    def _scoring_descriptors(self, layer, frame_ids, device):
        """Reuse persistent Draft-K descriptors directly on the scoring device."""
        rows = tuple((self.descriptors[layer][fid] for fid in frame_ids))
        if not rows or any((item.device != device for item in rows)):
            raise BackboneError('KV_CONSTRAINT', 'DraftKV historical descriptors must match the scoring device')
        return rows

    def after_attention(self, context, selection):
        pending = self.pending.get(context.layer)
        if pending is None or not pending['consumed']:
            raise BackboneError('PARTIAL_NATIVE_WIRING', 'DraftKV attention did not execute original B01')
        if context.clean:
            staged = self._stage_clean_layer(context.layer, pending)
            self.pending[context.layer] = dict(current_ids=pending['current_ids'], consumed=True, cpu_stage=staged)
        self.done.add(context.layer)
        if not context.clean:
            del self.pending[context.layer]
            if self.done == self.expected_layers:
                self.active = None

    def _stage_clean_layer(self, layer, pending):
        """Copy one completed layer privately; publish only at model commit."""
        import torch
        from draftkv.algorithms.draftkv.historical.orbia.retrieval.draftattention2_importance import descriptor
        from .registered_pages import RegisteredKVSlab
        c = pending['context']
        (h, w) = self.geometry[:2]
        archive = getattr(self, '_canonical_clean_archive', None)
        if archive is None and c.key.is_cuda:
            from draftkv.algorithms.draftkv.historical.orbia.streaming.canonical_archive import ensure_archive
            archive = ensure_archive(self, c)
        if archive is not None and (not self.active.get('recache', False)):
            from .position import geometry_for
            return archive.stage(c, pending['current_ids'], geometry_for(c, self.host.spec.family), importance=(h, w, self.importance), descriptors=getattr(self, 'use_draft_descriptors', True))
        (rows, descriptors) = ({}, {})
        slab = None
        if c.key.is_cuda:
            if getattr(self, 'archive_page_policy', 'thp2m') == 'base4k':
                slab = RegisteredKVSlab(c.key, c.value, len(pending['current_ids']), c.frame_tokens, page_policy='base4k')
            else:
                slab = RegisteredKVSlab(c.key, c.value, len(pending['current_ids']), c.frame_tokens)
            self._clean_registered_owners.append(slab)
        for (rank, fid) in enumerate(pending['current_ids']):
            (a, b) = (rank * c.frame_tokens, (rank + 1) * c.frame_tokens)
            sources = (c.key[0, a:b], c.value[0, a:b])
            pair = slab.frame(rank) if slab is not None else tuple((torch.empty_like(t, device='cpu', pin_memory=c.key.is_cuda) for t in sources))
            for (dst, src) in zip(pair, sources):
                dst.copy_(src, non_blocking=False)
            rows[fid] = pair
            if getattr(self, 'use_draft_descriptors', True):
                descriptors[fid] = descriptor(c.key[:, a:b], h, w, capacity=self.importance.capacity, pool_h=self.importance.pool_h, pool_w=self.importance.pool_w).detach()
            if any((t.device.type != 'cpu' for t in pair)) or (fid in descriptors and descriptors[fid].device != c.key.device):
                raise BackboneError('KV_CONSTRAINT', 'DraftKV CPU KV / device-resident descriptor ownership mismatch')
        temporal = self.time_coordinates.stage_clean({layer: pending})[layer]
        raw = {fid: temporal[fid] for fid in pending['current_ids']}
        nbytes = len(pending['current_ids']) * 2 * c.frame_tokens * c.query.shape[2] * 128 * 2
        return dict(rows=rows, descriptors=descriptors, raw=raw, nbytes=nbytes, registered=slab is not None)

    def commit_clean(self, *, timestep):
        import torch
        from draftkv.algorithms.draftkv.historical.orbia.retrieval.draftattention2_importance import descriptor
        if self.active is None or not self.active['clean'] or timestep is not self.active['timestep'] or (self.done != self.expected_layers) or (set(self.pending) != self.done) or any((not p['consumed'] for p in self.pending.values())):
            raise BackboneError('DIRTY_HISTORY', 'only an entire successful native clean forward may commit')
        staged_ids = [pending['current_ids'] for pending in self.pending.values()]
        if any((ids != staged_ids[0] for ids in staged_ids[1:])):
            raise BackboneError('DIRTY_HISTORY', 'clean commit current ID sets differ across native layers')
        (staged, staged_descriptors, temporal_banks) = ({}, {}, {})
        additional = 0
        new_registered_owners = []
        for (layer, pending) in self.pending.items():
            row = pending.get('cpu_stage')
            if row is None:
                row = self._stage_clean_layer(layer, pending)
            if row.get('completion') is not None:
                row['completion'].wait()
                if pm.enabled():
                    self.host.trace('draftkv_gpu_prepare_complete', layer=layer, **row['completion'].times())
            (staged[layer], staged_descriptors[layer]) = (row['rows'], row['descriptors'])
            temporal_banks[layer] = {**self.time_coordinates.raw_temporal.get(layer, {}), **row['raw']}
            if not self.active['recache']:
                additional += row['nbytes']
            if row['registered']:
                new_registered_owners.append(layer)
        from .device_archive_route import publish as publish_device_archive
        publish_device_archive(self, staged, staged_descriptors)
        banks = {layer: {**self.committed.get(layer, {}), **rows} for (layer, rows) in staged.items()}
        descriptors = {layer: {**self.descriptors.get(layer, {}), **rows} for (layer, rows) in staged_descriptors.items()}
        self.time_coordinates.publish(temporal_banks)
        (self.committed, self.descriptors) = (banks, descriptors)
        if not self.active['recache']:
            self.next_frame = max(next(iter(staged.values()))) + 1
            self.reserved += additional
        self.commit_generation += 1
        descriptor_sample = next((value for bank in descriptors.values() for value in bank.values()), None)
        descriptor_device_type = descriptor_sample.device.type if descriptor_sample is not None else 'none'
        descriptor_gpu_bytes = sum((t.numel() * t.element_size() for bank in descriptors.values() for t in bank.values() if t.is_cuda))
        self.host.trace('draftkv_clean_commit', start_frame=self.active['start_frame'], forward_id=self.active['forward_id'], forward_index=self.forward_index, sink_recache=self.active['recache'], generation=self.commit_generation, committed_ids=list(next(iter(staged.values()))), layers=len(staged), importance_config=self.importance.to_dict(), importance_config_sha256=self.importance_sha256, pinned_history_bytes=self.reserved, committed_frontier=self.next_frame, raw_temporal_key_bytes=self.time_coordinates.raw_bytes(), descriptor_key_source='clean_context.key_native_post_rope', full_kv_source='native_post_rope', position_policy=self.position_policy, history_kv_device='cpu', history_descriptor_device=descriptor_device_type, persistent_video_gpu_kv_bytes=0, persistent_gpu_draft_descriptor_bytes=descriptor_gpu_bytes, gpu_kv_lifetime='current_chunk_and_per_layer_temporary_attention_only', archive_page_policy=getattr(self, 'archive_page_policy', 'thp2m'), clean_pinned_allocator='bounded_ahead_of_use_arena' if getattr(self, '_canonical_clean_archive', None) is not None and (not self.active['recache']) else ('base4k_cuda_host_register' if getattr(self, 'archive_page_policy', 'thp2m') == 'base4k' else 'thp_cuda_host_register') if new_registered_owners else 'torch_pinned_original', clean_registered_layer_count=len(new_registered_owners))
        (self.active, self.pending, self.done) = (None, {}, set())

    def rollback(self):
        self.invalidated = True
        (self.active, self.pending, self.done) = (None, {}, set())

    def history_ids(self, layer):
        return tuple(sorted(self.committed.get(layer, {})))

    def contains_history_ids(self, layer, frame_ids):
        bank = self.committed.get(layer, {})
        return all((fid in bank for fid in frame_ids))

    def frame_pages(self, layer, frame_id):
        if frame_id not in self.committed.get(layer, {}):
            raise BackboneError('KV_CONSTRAINT', 'frame is not clean committed in this layer')
        tokens = self.geometry[3]
        return (FramePage(frame_id, frame_id, 0, tokens),)
