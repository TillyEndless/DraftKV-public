# SPDX-License-Identifier: Apache-2.0
"""Training-free A2-Frustum V2 camera scope for LingBot1.3B C4 and SolarWM-H3 C5.

Uses the exact frozen LingBot14B bilateral frustum overlap implementation,
followed by DraftKV post-RoPE 8x16 importance and B01 Streaming FA2.
Remote history is eligible only inside camera Top-K; exact recent P history
is protected. No Early Fix, fallback, threshold or candidate backfill.

LongLive1.3B's frozen text-only WorldAttention benchmark has no camera
trajectory. It cannot run geometric A2 and is intentionally not registered
here; its P3 Block Top-K runtime remains usable without inventing poses.
"""
from __future__ import annotations
from draftkv.streaming.external import external_symbol, external_path
from pathlib import Path
from draftkv.algorithms.draftkv.historical.orbia.algorithm.block_backbone import BlockDraftKVHook
from draftkv.algorithms.draftkv.historical.orbia.algorithm.block_policy import configured_protection
from draftkv.algorithms.draftkv.historical.orbia.algorithm.camera_frustum_v2 import bilateral_frustum_scores, gated_frame_candidates
from draftkv.algorithms.draftkv.historical.orbia.backbones.contracts import BackboneError
from draftkv.algorithms.draftkv.historical.orbia.retrieval.block_topk_p3 import BlockDecision, block_probabilities, select_blocks
CAMERA_POLICY = {'LingBot1.3B': (4, 3, 3), 'LingBot14B': (4, 3, 3), 'SolarWM33B': (5, 7, 5)}

def rank_camera_chunks(poses, start_frame, *, sink_frames, current_frames, top_chunks, stored_chunks, intrinsics, width, height, local_frames=0):
    """A2 eligible C-sized chunks; reuse the frozen BFO score without modification.

    One sink-ceil chunk is excluded if the sink intersects it. The last two
    historical chunks are excluded as in the original A2-C4 policy. A declared
    L window is also excluded before ranking; P is re-added separately.
    """
    import torch
    if poses is None:
        raise BackboneError('CAMERA_POSES_MISSING', 'A2 needs actual native camera poses')
    if current_frames not in (3, 4, 5) or start_frame % current_frames:
        raise BackboneError('CAMERA_GATE_ALIGNMENT', 'A2 requires aligned native C3/C4/C5')
    if poses.ndim != 3 or tuple(poses.shape[-2:]) != (4, 4):
        raise BackboneError('CAMERA_POSE_SHAPE', 'A2 requires actual absolute C2W [T,4,4]')
    if not bool(torch.isfinite(poses).all()) or not bool(torch.isfinite(intrinsics).all()):
        raise BackboneError('CAMERA_POSE_NONFINITE', 'camera geometry nonfinite')
    if intrinsics.shape != (4,) or bool((intrinsics[:2] <= 0).any()):
        raise BackboneError('CAMERA_INTRINSICS_SHAPE', 'pixel intrinsics must be [fx,fy,cx,cy]')
    if type(top_chunks) is not int or top_chunks < 1:
        raise BackboneError('BLOCKED_CONFIGURATION', 'A2 Top-K must be a positive integer')
    if type(local_frames) is not int or local_frames < 0:
        raise BackboneError('BLOCKED_CONFIGURATION', 'invalid camera local window')
    chunk = start_frame // current_frames
    if chunk > stored_chunks or len(poses) < start_frame + current_frames:
        raise BackboneError('DIRTY_HISTORY', 'native camera poses/clean KV frontier misaligned')
    sink_chunks = (sink_frames + current_frames - 1) // current_frames
    eligible_stop = min(chunk - 2, (start_frame - local_frames) // current_frames)
    eligible = tuple(range(sink_chunks, max(sink_chunks, eligible_stop)))
    if not eligible:
        return ((), eligible)
    (scores, _, _, distances) = bilateral_frustum_scores(poses, list(eligible), chunk, current_frames, intrinsics.to(device=poses.device), width, height)
    score_cpu = scores.detach().cpu().tolist()
    dist_cpu = distances.detach().cpu().tolist()
    ranked = sorted(range(len(eligible)), key=lambda i: (-score_cpu[i], dist_cpu[i], eligible[i]))
    picked = tuple((eligible[i] for i in ranked[:top_chunks]))
    if len(set(picked)) != len(picked) or not set(picked).issubset(eligible):
        raise BackboneError('CAMERA_GATE_RANK', 'A2 camera chunk selection invalid')
    return (picked, eligible)

def _native_intrinsics(host):
    """Only convert camera geometry for retrieval; never modify native PRoPE."""
    import numpy as np
    import torch
    request = host.plan['request']
    backbone = host.spec.backbone
    if backbone in ('LingBot1.3B', 'LingBot14B'):
        values = np.load(Path(request['camera_dir']) / 'intrinsics.npy', allow_pickle=False)
        if values.shape != (1, 4):
            raise BackboneError('CAMERA_INTRINSICS_SHAPE', 'LingBot camera K must be (1,4)')
        result = values[0]
    elif backbone == 'SolarWM33B':
        original = np.asarray(request['original_intrinsics_K'], dtype=np.float64)
        if original.shape != (3, 3):
            raise BackboneError('CAMERA_INTRINSICS_SHAPE', 'H3 original K must be 3x3')
        (sw, sh) = (request['source_width'], request['source_height'])
        (w, h) = (request['output_width'], request['output_height'])
        if min(sw, sh, w, h) <= 0:
            raise BackboneError('CAMERA_INTRINSICS_SHAPE', 'invalid H3 image dimensions')
        result = np.asarray([original[0, 0] * w / sw, original[1, 1] * h / sh, original[0, 2] * w / sw, original[1, 2] * h / sh])
    else:
        raise BackboneError('BLOCKED_CONFIGURATION', 'camera gate needs an actual camera-conditioned backbone')
    if not np.isfinite(result).all() or np.any(result[:2] <= 0):
        raise BackboneError('CAMERA_INTRINSICS_SHAPE', 'invalid physical camera intrinsics')
    return torch.tensor(result.copy(), dtype=torch.float32)

class CameraGatedMultiBackboneHook(BlockDraftKVHook):
    """A2 camera candidate restriction with unchanged BlockDraftKVHook/B01."""

    def __init__(self, host):
        backbone = host.spec.backbone
        if backbone not in CAMERA_POLICY:
            raise BackboneError('CAMERA_POSES_MISSING', 'LongLive text-only T2V has no camera poses for A2 BFO')
        (chunk, top, protected) = CAMERA_POLICY[backbone]
        cfg = host.plan['algorithm_plugin']['config']
        if host.spec.chunk != chunk or cfg.get('camera_top_chunks', top) != top or type(cfg.get('camera_top_chunks', top)) is not int or (cfg.get('protect_recent_frames', protected) != protected) or (cfg.get('camera_gate_mode', 'hard_top_m_no_backfill') != 'hard_top_m_no_backfill'):
            raise BackboneError('BLOCKED_CONFIGURATION', 'wrong A2 Top-K / P / C or Early Fix')
        from draftkv.algorithms.draftkv.historical.orbia.algorithm.entry import configure_retrieval
        self.camera_top_chunks = top
        self.camera_poses = None
        self._camera_chunk_cache = {}
        self._camera_events = set()
        self._frustum_intrinsics = _native_intrinsics(host)
        self._frustum_width = int(host.plan['request']['output_width'])
        self._frustum_height = int(host.plan['request']['output_height'])
        super().__init__(host)

    def set_camera_poses(self, poses):
        import torch
        if not isinstance(poses, torch.Tensor) or poses.ndim != 3 or tuple(poses.shape[-2:]) != (4, 4) or (not bool(torch.isfinite(poses).all())):
            raise BackboneError('CAMERA_POSE_SHAPE', 'absolute native camera C2W missing')
        self.camera_poses = poses.detach().clone()
        self._camera_chunk_cache.clear()
        self.host.trace('camera_gate_native_c2w', case_id=self.case_id, frames=len(poses), camera_top_chunks=self.camera_top_chunks, source='solarwm_original_orbia_c2w_native_c5' if self.host.spec.family == 'solarwm' else 'worldkv_absolute_pose_interpolation')

    def _camera_scope(self, context, candidates, requested_budget):
        start = context.start_frame
        chunk = self.host.spec.chunk
        if start % chunk:
            raise BackboneError('CAMERA_GATE_ALIGNMENT', 'native C4/C5 offset misaligned')
        keep = min(configured_protection(self.host.spec.backbone), requested_budget, len(candidates))
        protected = tuple(candidates[-keep:]) if keep else ()
        if start not in self._camera_chunk_cache:
            from draftkv.algorithms.draftkv.historical.orbia.backbones.budgets import bound_budget
            (chunks, eligible) = rank_camera_chunks(self.camera_poses, start, sink_frames=context.sink_frames, current_frames=chunk, top_chunks=self.camera_top_chunks, stored_chunks=self.next_frame // chunk, intrinsics=self._frustum_intrinsics, width=self._frustum_width, height=self._frustum_height, local_frames=bound_budget(self.host, 'draftkv').local)
            scoped = gated_frame_candidates(candidates, protected, chunks, sink_frames=context.sink_frames, chunk_frames=chunk, budget=requested_budget)
            if not set(scoped).issubset(candidates) or not set(protected).issubset(scoped):
                raise BackboneError('CAMERA_GATE_SCOPE', 'unapproved source frames or lost recent P')
            self._camera_chunk_cache[start] = (chunks, eligible, scoped, protected)
        (chunks, eligible, scoped, prior) = self._camera_chunk_cache[start]
        if protected != prior or tuple(sorted(set(scoped))) != scoped:
            raise BackboneError('CAMERA_GATE_SCOPE', 'inconsistent candidate scope across layers')
        return (chunks, eligible, scoped, protected)

    def _history_budget(self, context, candidates, budget):
        (_, _, scoped, _) = self._camera_scope(context, candidates, budget)
        return min(budget, len(scoped))

    def _select_history(self, context, candidates, budget, h, w):
        if not candidates or not budget:
            return None
        (chunks, eligible, scoped, protected) = self._camera_scope(context, candidates, budget)
        keep = len(protected)
        if budget > len(scoped) or (keep and scoped[-keep:] != protected):
            raise BackboneError('CAMERA_GATE_BUDGET', 'candidate budget or P3/P5 violation')
        probs = block_probabilities(context.query, self._scoring_descriptors(context.layer, scoped, context.query.device), h, w, config=self.importance)
        (heads, queries) = probs.shape[1:3]
        scores = probs.float().sum(dim=2).sum(dim=1)
        if context.sp > 1:
            import torch.distributed as dist
            if not dist.is_initialized() or dist.get_world_size() != context.sp:
                raise BackboneError('KV_CONSTRAINT', 'native SP group required')
            dist.all_reduce(scores, op=dist.ReduceOp.SUM)
        scores = scores / (heads * queries * context.sp)
        decision = select_blocks(scores[0], scoped, budget, query_start=context.start_frame, grid_h=h, grid_w=w, protect_recent=keep, pool_h=self.importance.pool_h, pool_w=self.importance.pool_w)
        if decision.protected != protected or sum((len(ids) for (_, ids) in decision.token_indices)) != budget * h * w:
            raise BackboneError('CAMERA_GATE_BUDGET', 'P or token budget drift')
        event = (context.start_frame, context.layer)
        if event not in self._camera_events:
            self._camera_events.add(event)
            self.host.trace('camera_gated_draftkv_route', case_id=self.case_id, layer=context.layer, start_frame=context.start_frame, forward_id=context.forward_id, requested_camera_top_m=self.camera_top_chunks, eligible_camera_chunk_ids=list(eligible), selected_camera_chunk_ids=list(chunks), true_latest_protected_ids=list(protected), selected_remote_frame_candidates=list(scoped), actual_history_budget_frames=budget, candidate_shortfall_frames=max(0, len(candidates) - len(scoped)), early_history_fallback=False, no_backfill=True, native_camera_source='solarwm_orbia_absolute_c2w' if self.host.spec.family == 'solarwm' else 'worldkv_frustum_v2_absolute_c2w', scorer_domain='candidate_subset_softmax_including_true_latest_P', block_size=[self.importance.pool_h, self.importance.pool_w], position_policy=self.position_policy)
        return decision
