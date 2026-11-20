"""Isolated Camera-Gated DraftKV: A1 Top3 / A2 Top4, case_0005.
Only remote candidate domain changes; full K history, P3, scorer, budget,
chronological packing and Streaming FA retain canonical DraftKV semantics.

Camera candidate ranking uses verbatim WorldKV-Frustum-v2 BFO mathematics.
Only the camera metric changes; A2 sink/recent eligibility is unchanged. The author recent-window exclusion
is two complete C4 chunks (recent=Local4+Current4), with the original sink
ceil-chunk exclusion. True latest P3 historical frames are always protected.
"""
from __future__ import annotations
import hashlib
import math
from pathlib import Path

import torch
import numpy as np
from draftkv.algorithms.draftkv.block_backbone import BlockDraftKVHook
from draftkv.algorithms.draftkv.block_policy import applied_protection, configured_protection
from draftkv.algorithms.worldkv.source import AUTHOR
from draftkv.backbones.contracts import BackboneError
from draftkv.retrieval.block_topk_p3 import block_probabilities, select_blocks
from draftkv.vendor.backbones.lingbot14.runtime.rag_portable.world_core import camera_selection_code

_POLICY = "worldkv_frustum_v2_bfo_topm_then_draftkv_block_importance"
_REQUIRED = frozenset({
    "draftattention2_postrope_spatial8x16_meanmax", "hierarchical_v1",
    "direct_block_topk_p3", "all_history_softmax",  # retained for baseline provenance, gated for new method
    "chronological_source_token_packing", "original_b01_producer_plan",
    "asynchronous_h2d", "cuda_frontier", "streaming_fa2",
    "clean_commit", "chrono_compact_temporal", "chronological_frame_packing",
    "worldkv_author_camera_gate", "protected_true_latest_p3",
})

GRID_H = 8
GRID_W = 8
DEPTH_MULTIPLIERS = (0.5, 1.5, 4.0)
EPS = 1e-8


def _project_so3(rotation: torch.Tensor) -> torch.Tensor:
    """Nearest proper rotation by batched polar/SVD projection."""
    u, _, vh = torch.linalg.svd(rotation.float())
    out = u @ vh
    det = torch.linalg.det(out)
    if bool((det < 0).any()):
        u = u.clone()
        u[det < 0, :, -1] *= -1
        out = u @ vh
    return out


def _chunk_pose(poses: torch.Tensor, cid: int, chunk_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    s = cid * chunk_size
    e = min(s + chunk_size, poses.shape[0])
    t = poses[s:e, :3, 3].float().mean(dim=0)
    r = _project_so3(poses[s:e, :3, :3].float().mean(dim=0, keepdim=True))[0]
    return r, t


def _sample_camera_points(intrinsics: torch.Tensor, width: int, height: int,
                          depth_scale: torch.Tensor) -> torch.Tensor:
    """8x8 normalized image rays x three z-depth shells -> [192,3]."""
    fx, fy, cx, cy = intrinsics.float()
    xs = (torch.arange(GRID_W, device=intrinsics.device, dtype=torch.float32) + 0.5) * (width / GRID_W)
    ys = (torch.arange(GRID_H, device=intrinsics.device, dtype=torch.float32) + 0.5) * (height / GRID_H)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    rays = torch.stack(((xx.reshape(-1) - cx) / fx,
                        (yy.reshape(-1) - cy) / fy,
                        torch.ones(GRID_H * GRID_W, device=intrinsics.device)), dim=-1)
    depths = depth_scale * torch.tensor(DEPTH_MULTIPLIERS, device=intrinsics.device, dtype=torch.float32)
    return (rays[:, None, :] * depths[None, :, None]).reshape(-1, 3)


def _inside(points_camera: torch.Tensor, intrinsics: torch.Tensor,
            width: int, height: int) -> torch.Tensor:
    fx, fy, cx, cy = intrinsics.float()
    z = points_camera[..., 2]
    safe_z = torch.where(z.abs() > EPS, z, torch.ones_like(z))
    u = fx * points_camera[..., 0] / safe_z + cx
    v = fy * points_camera[..., 1] / safe_z + cy
    return (z > EPS) & (u >= 0) & (u < width) & (v >= 0) & (v < height)


def bilateral_frustum_scores(
    poses: torch.Tensor,
    candidate_ids: list[int],
    chunk_id: int,
    chunk_size: int,
    intrinsics: torch.Tensor,
    width: int,
    height: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return BFO score, forward overlap, reverse overlap, and WorldKV pose distance."""
    if not candidate_ids:
        empty = torch.empty(0, device=poses.device, dtype=torch.float32)
        return empty, empty, empty, empty

    current_r, current_t = _chunk_pose(poses, chunk_id, chunk_size)
    cand_r, cand_t = zip(*[_chunk_pose(poses, cid, chunk_size) for cid in candidate_ids])
    cand_r = torch.stack(cand_r)
    cand_t = torch.stack(cand_t)

    # Scale uses only observed/current camera centers, never future trajectory.
    ce = min((chunk_id + 1) * chunk_size, poses.shape[0])
    centers = poses[:ce, :3, 3].float()
    center_mean = centers.mean(dim=0)
    radius = torch.linalg.vector_norm(centers - center_mean, dim=-1).max()
    step = torch.linalg.vector_norm(centers[1:] - centers[:-1], dim=-1)
    step_scale = step.median() * 16.0 if step.numel() else radius
    depth_scale = torch.maximum(radius, step_scale).clamp_min(1e-4)

    points = _sample_camera_points(intrinsics, width, height, depth_scale)

    # Candidate -> current.
    world_from_cand = torch.matmul(points.unsqueeze(0), cand_r.transpose(-1, -2)) + cand_t[:, None, :]
    in_current = torch.matmul(world_from_cand - current_t, current_r)
    forward = _inside(in_current, intrinsics, width, height).float().mean(dim=-1)

    # Current -> candidate.
    world_from_current = torch.matmul(points, current_r.transpose(-1, -2)) + current_t
    in_candidate = torch.matmul(world_from_current.unsqueeze(0) - cand_t[:, None, :], cand_r)
    reverse = _inside(in_candidate, intrinsics, width, height).float().mean(dim=-1)

    score = 2.0 * forward * reverse / (forward + reverse + EPS)

    # Original WorldKV distance retained ONLY for deterministic ties.
    trans_dist = ((cand_t - current_t) ** 2).sum(dim=-1)
    rel_rot = cand_r.transpose(-1, -2) @ current_r.unsqueeze(0)
    trace_val = rel_rot.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    rot_dist = torch.acos(((trace_val - 1.0) / 2.0).clamp(-1.0, 1.0))
    trans_norm = trans_dist / trans_dist.max() if bool(trans_dist.max() > 0) else trans_dist
    rot_norm = rot_dist / rot_dist.max() if bool(rot_dist.max() > 0) else rot_dist
    pose_distance = 0.5 * trans_norm + 0.5 * rot_norm
    return score, forward, reverse, pose_distance

def worldkv_camera_chunks(poses, start_frame, *, sink_frames, current_frames,
                          top_chunks, stored_chunks, intrinsics, width, height):
    """Original A2 eligible window + EXACT pinned WorldKV-Frustum-V2 BFO metric."""
    if poses is None:
        raise BackboneError("CAMERA_POSES_MISSING", "absolute C2W not captured")
    if current_frames != 4 or start_frame % current_frames:
        raise BackboneError("KV_CONSTRAINT", "frozen LingBot14B C4 required")
    if poses.ndim != 3 or tuple(poses.shape[-2:]) != (4,4):
        raise BackboneError("CAMERA_POSE_SHAPE", "WorldKV absolute C2W must be [T,4,4]")
    chunk = start_frame // current_frames
    if chunk > stored_chunks:
        raise BackboneError("DIRTY_HISTORY", "camera KV chunk is beyond clean history")
    if len(poses) < start_frame + current_frames:
        raise BackboneError("CAMERA_POSES_MISSING", "absolute camera trajectory is incomplete")
    if top_chunks not in (3,4):
        raise BackboneError("BLOCKED_CONFIGURATION", "A1/A2 only, Top3 or Top4")
    # IDENTICAL A2 eligibility; only the camera score below has changed.
    sink_chunks=(sink_frames+current_frames-1)//current_frames
    recent_chunks=max(1,(4+current_frames)//current_frames)
    eligible=tuple(range(sink_chunks,max(sink_chunks,chunk-recent_chunks)))
    if not eligible:
        return (), eligible
    # EXACT WorldKV-Frustum-V2 BFO: SO(3)-projected chunk mean,
    # 8x8x3 rays, adaptive depth shells, bidirectional harmonic overlap.
    scores, forward, reverse, pose_tiebreak = bilateral_frustum_scores(
        poses, list(eligible), chunk, current_frames,
        intrinsics.to(device=poses.device), width, height)
    score_cpu=scores.detach().cpu().tolist()
    pose_cpu=pose_tiebreak.detach().cpu().tolist()
    order=sorted(range(len(eligible)),
                 key=lambda j:(-score_cpu[j], pose_cpu[j], eligible[j]))
    picked=tuple(eligible[j] for j in order[:min(top_chunks,len(eligible))])
    if len(picked)!=min(top_chunks,len(eligible)):
        raise BackboneError("CAMERA_GATE_RANK", "WorldKV-Frustum-v2 TopK cardinality drift")
    if len(set(picked))!=len(picked) or not set(picked).issubset(eligible):
        raise BackboneError("CAMERA_GATE_RANK", "WorldKV-Frustum candidate outside A2 range")
    return picked, eligible

def gated_frame_candidates(all_candidates, protected, camera_chunks, *,
                            sink_frames, chunk_frames, budget):
    """Only camera Top-M remote frames plus mandatory native P3; never backfill."""
    all_candidates=tuple(all_candidates)
    protected=tuple(protected)
    remote=tuple(fid for c in camera_chunks
                 for fid in range(c*chunk_frames,(c+1)*chunk_frames)
                 if fid>=sink_frames and fid in all_candidates)
    gate=tuple(sorted(set(protected).union(remote)))
    if not set(protected).issubset(gate):
        raise BackboneError("CAMERA_GATE_P3", "recent P3 missing")
    # We intentionally allow fewer than the nominal R budget. The missing
    # history slots are not padded, and unrelated historical frames stay banned.
    return gate


class CameraGatedBlockDraftHook(BlockDraftKVHook):
    def __init__(self, host):
        cfg=host.plan["algorithm_plugin"]["config"]
        self.camera_top_chunks=cfg.get("camera_top_chunks")
        if type(self.camera_top_chunks) is not int or self.camera_top_chunks not in (3,4):
            raise BackboneError("BLOCKED_CONFIGURATION", "Camera Gate supports only A1 Top3 and A2 Top4")
        if host.spec.backbone!="LingBot14B" or host.spec.chunk!=4:
            raise BackboneError("BLOCKED_CONFIGURATION", "Camera Gate experiment restricted to LingBot14B C4")
        if cfg.get("gpu_prepack",False) or cfg.get("gpu_device_route",False):
            raise BackboneError("BLOCKED_CONFIGURATION", "frozen first A1/A2 uses baseline BlockDraftKVHook B01 route")
        self.camera_poses=None
        self._camera_chunk_cache={}
        self._camera_events=set()
        camera_dir=Path(host.plan["request"]["camera_dir"])
        intrinsics=np.load(camera_dir/"intrinsics.npy", allow_pickle=False)
        if intrinsics.shape!=(1,4):
            raise BackboneError("CAMERA_INTRINSICS_SHAPE", "WorldKV-Frustum-v2 expects [fx,fy,cx,cy]")
        self._frustum_intrinsics=torch.from_numpy(intrinsics[0].copy()).float()
        self._frustum_width=int(host.plan["request"]["width"])
        self._frustum_height=int(host.plan["request"]["height"])
        super().__init__(host)

    def set_camera_poses(self, poses):
        if not isinstance(poses,torch.Tensor) or poses.ndim!=3 or tuple(poses.shape[-2:])!=(4,4):
            raise BackboneError("CAMERA_POSE_SHAPE","native absolute C2W tensor missing")
        if not bool(torch.isfinite(poses).all()):
            raise BackboneError("CAMERA_POSE_NONFINITE","native interpolated C2W nonfinite")
        self.camera_poses=poses.detach().clone()
        self._camera_chunk_cache.clear()
        self.host.trace("camera_gate_native_c2w",case_id=self.case_id,frames=int(poses.shape[0]),
            shape=list(poses.shape),source="worldkv_absolute_pose_interpolation",
            camera_top_chunks=self.camera_top_chunks)

    def _camera_scope(self, context, candidates, requested_budget):
        """Freeze the same Camera Top-M + true P3 scope across all layers/steps."""
        if context.start_frame % self.host.spec.chunk:
            raise BackboneError("CAMERA_GATE_ALIGNMENT", "C4 current start not aligned")
        keep = min(configured_protection(self.host.spec.backbone), requested_budget, len(candidates))
        protected = tuple(candidates[-keep:]) if keep else ()
        start = context.start_frame
        if start not in self._camera_chunk_cache:
            chunks, eligible = worldkv_camera_chunks(
                self.camera_poses, start, sink_frames=context.sink_frames,
                current_frames=self.host.spec.chunk, top_chunks=self.camera_top_chunks,
                stored_chunks=self.next_frame // self.host.spec.chunk,
                intrinsics=self._frustum_intrinsics,
                width=self._frustum_width, height=self._frustum_height)
            scoped = gated_frame_candidates(
                candidates, protected, chunks, sink_frames=context.sink_frames,
                chunk_frames=self.host.spec.chunk, budget=requested_budget)
            if not set(scoped).issubset(candidates) or not set(protected).issubset(scoped):
                raise BackboneError("CAMERA_GATE_SCOPE", "non-camera or missing P3 history")
            self._camera_chunk_cache[start] = (chunks,eligible,scoped,protected)
        chunks,eligible,scoped,original_protected = self._camera_chunk_cache[start]
        if protected != original_protected or tuple(sorted(set(scoped))) != scoped:
            raise BackboneError("CAMERA_GATE_SCOPE", "history scope changed across layers")
        return chunks, eligible, scoped, protected

    def _history_budget(self, context, candidates, budget):
        # The parent route uses this return value to allocate pinned staging,
        # construct exact chronology, size H2D transfer and allocate attention.
        # Shrink it before any packing; never borrow frames outside Camera Top-M.
        _, _, scoped, _ = self._camera_scope(context, candidates, budget)
        return min(budget, len(scoped))

    def _select_history(self, context, candidates, budget, h, w):
        if not candidates or not budget: return None
        start = context.start_frame
        chunks,eligible,scoped,protected = self._camera_scope(context, candidates, budget)
        keep = len(protected)
        if budget > len(scoped):
            raise BackboneError("CAMERA_GATE_BUDGET", "short candidate domain was not budget-clipped")
        if scoped[-keep:] != protected:
            raise BackboneError("CAMERA_GATE_P3", "true latest P3 must be final in scoped order")
        # The main compute must use the filtered source IDs to shrink the
        # softmax domain as well as the descriptor gather, not mask AFTER scoring.
        probs=block_probabilities(
            context.query,self._scoring_descriptors(context.layer,scoped,context.query.device),
            h,w,config=self.importance)
        masses=probs.float().sum(dim=2)
        heads,queries=probs.shape[1:3]
        scores=masses.sum(dim=1)
        if context.sp>1:
            import torch.distributed as dist
            if not dist.is_initialized() or dist.get_world_size()!=context.sp:
                raise BackboneError("KV_CONSTRAINT","native SP head group required")
            dist.all_reduce(scores,op=dist.ReduceOp.SUM)
        scores=scores/(heads*queries*context.sp)
        decision=select_blocks(
            scores[0],scoped,budget,
            query_start=start,grid_h=h,grid_w=w,protect_recent=keep,
            pool_h=self.importance.pool_h,pool_w=self.importance.pool_w)
        if decision.protected!=protected or sum(len(t) for _,t in decision.token_indices)!=budget*h*w:
            raise BackboneError("CAMERA_GATE_BUDGET","P3 or exact token budget drift")
        if (start,context.layer) not in self._camera_events:
            self._camera_events.add((start,context.layer))
            self.host.trace("camera_gated_draftkv_route",case_id=self.case_id,
                layer=context.layer, start_frame=start,forward_id=context.forward_id,
                requested_camera_top_m=self.camera_top_chunks,
                eligible_camera_chunk_ids=list(eligible),
                selected_camera_chunk_ids=list(chunks),
                true_latest_protected_ids=list(protected),
                selected_remote_frame_candidates=list(scoped),
                all_nonsink_history_frame_count=len(candidates),
                gated_history_frame_count=len(scoped),
                early_history_fallback=False,
                candidate_shortfall_frames=max(0, self.host.spec.window - self.host.spec.sink - self.host.spec.chunk - len(scoped)),
                actual_history_budget_frames=budget,
                no_backfill=True,
                scorer_domain="candidate_subset_softmax_including_true_latest_P3",
                history_budget_frames=budget,
                token_budget_exact=budget*h*w,
                native_camera_source="worldkv_frustum_v2_bilateral_harmonic_absolute_c2w",
                position_policy=self.position_policy,
                block_size=[self.importance.pool_h,self.importance.pool_w])
        return decision


def create_hook(core):
    return CameraGatedBlockDraftHook(core)


def register(registry):
    from draftkv.backbones.hooks import HookRegistration
    # Register under DraftKV: inherit its audited budget and Streaming-FA ABI.
    # The separate hashed plugin/plan distinguishes A1 versus A2.
    registry.register(HookRegistration(
        "draftkv",create_hook,_REQUIRED,
        hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"streaming_fa2"))
