"""Canonical camera/scoring semantics with no host selection or token maps."""
from dataclasses import replace
from draftkv.backbones.contracts import AttentionSelection, BackboneError
from draftkv.streaming.device_archive import (
    ArchiveDirectory, gather_descriptors, select_device, bind_device_archive,
)


def enabled(hook, context):
    settings = getattr(hook, 'settings', {})
    return (context.query.is_cuda and context.sp == 1 and settings.get('device_archive_plan', True)
        and hook.host.spec.backbone in ('LingBot1.3B', 'LingBot14B', 'SolarWM33B')
        and settings.get('retrieval_ablation', 'full') in ('full', 'draft-only')
        and settings.get('gpu_krope_8x16', True)
        and getattr(hook, 'position_policy', 'chrono_compact') == 'chrono_compact'
        and getattr(hook.binding, 'host_pages_policy', 'quality') == 'quality'
        and not getattr(hook, 'gpu_history', False)
        and hook._native_runtime_config(context) is not None)


def publish(hook, staged, staged_descriptors):
    directories = getattr(hook, '_device_archive_directories', None)
    if directories is None:
        return
    updated = dict(directories)
    for layer, rows in staged.items():
        if layer in updated:
            updated[layer] = updated[layer].publication(rows, staged_descriptors[layer])
    hook._device_archive_directories = updated


def camera_scope(hook, context, capacity, device):
    # Camera BFO used CPU FP32 in the oracle. Native model autocast must not
    # turn the migrated GPU rotation products into BF16 (or change ranking).
    import torch
    with torch.autocast(device_type='cuda', enabled=False):
        return _camera_scope(hook, context, capacity, device)


def _camera_scope(hook, context, capacity, device):
    """Frozen BFO math and lexicographic ties, evaluated/ranked on the GPU."""
    import torch
    from .camera_frustum_v2 import _sample_camera_points, _inside, EPS
    from .block_policy import configured_protection
    start, chunk, sink = context.start_frame, hook.host.spec.chunk, context.sink_frames
    count = max(0, start-sink)
    p = min(configured_protection(hook.host.spec.backbone), capacity, count)
    if getattr(hook, '_device_camera_pose_owner', None) is not getattr(hook, 'camera_poses', None):
        hook._device_camera_pose_owner = hook.camera_poses
        hook._device_camera_poses = None
        hook._device_camera_scope = None
    cache = getattr(hook, '_device_camera_scope', None)
    if cache is not None and cache[0] == start:
        return cache[1:]
    if getattr(hook, 'camera_poses', None) is None:
        raise BackboneError('CAMERA_POSES_MISSING', 'device archive requires native C2W')
    poses = getattr(hook, '_device_camera_poses', None)
    if poses is None:
        poses = hook.camera_poses.to(device=device)
        hook._device_camera_poses = poses
        hook._device_camera_intrinsics = hook._frustum_intrinsics.to(device=device)
    k = hook._device_camera_intrinsics
    cid = start//chunk
    first, stop = (sink+chunk-1)//chunk, max((sink+chunk-1)//chunk, cid-2)
    eligible = torch.arange(first, stop, device=device, dtype=torch.int64)
    m = min(hook.camera_top_chunks, stop-first)
    if m:
        # Same per-chunk SO(3)-projected mean, including reflection correction.
        observed = poses[:start+chunk].reshape(cid+1, chunk, 4, 4)
        rotation = observed[:, :, :3, :3].float().mean(1)
        u, _, vh = torch.linalg.svd(rotation)
        out = u @ vh
        negative = torch.linalg.det(out) < 0
        corrected = u.clone()
        corrected[:, :, -1] *= torch.where(negative, -1., 1.)[:, None]
        rotations = corrected @ vh
        centers_by_chunk = observed[:, :, :3, 3].float().mean(1)
        cr, ct = rotations[cid], centers_by_chunk[cid]
        rr, tt = rotations[eligible], centers_by_chunk[eligible]
        centers = poses[:start+chunk, :3, 3].float()
        radius = torch.linalg.vector_norm(centers-centers.mean(0), dim=-1).max()
        step = torch.linalg.vector_norm(centers[1:]-centers[:-1], dim=-1)
        scale = torch.maximum(radius, step.median()*16.).clamp_min(1e-4)
        points = _sample_camera_points(k, hook._frustum_width, hook._frustum_height, scale)
        world = torch.matmul(points.unsqueeze(0), rr.transpose(-1, -2))+tt[:, None, :]
        forward = _inside(torch.matmul(world-ct, cr), k,
                          hook._frustum_width, hook._frustum_height).float().mean(-1)
        world_current = torch.matmul(points, cr.transpose(-1, -2))+ct
        reverse = _inside(torch.matmul(world_current.unsqueeze(0)-tt[:, None, :], rr), k,
                          hook._frustum_width, hook._frustum_height).float().mean(-1)
        score = 2.*forward*reverse/(forward+reverse+EPS)
        td = ((tt-ct)**2).sum(-1)
        relative = rr.transpose(-1, -2) @ cr.unsqueeze(0)
        rd = torch.acos(((relative.diagonal(dim1=-2, dim2=-1).sum(-1)-1.)/2.).clamp(-1., 1.))
        td = td/torch.where(td.max()>0, td.max(), torch.ones_like(td.max()))
        rd = rd/torch.where(rd.max()>0, rd.max(), torch.ones_like(rd.max()))
        distance = .5*td+.5*rd
        order = torch.argsort(distance, stable=True)
        order = order[torch.argsort(score[order], descending=True, stable=True)]
        chosen = eligible[order[:m]]
        remote = (chosen[:, None]*chunk+torch.arange(chunk, device=device)[None, :]).flatten()
    else:
        chosen = eligible
        remote = torch.empty(0, device=device, dtype=torch.int64)
    protected = torch.arange(start-p, start, device=device, dtype=torch.int64)
    frames = torch.cat((remote, protected)).sort().values
    budget = min(capacity, m*chunk+p)
    # Remote eligibility excludes two chunks; P must be disjoint from it.
    if p > 2*chunk:
        raise BackboneError('CAMERA_GATE_BUDGET', 'P overlaps excluded recent camera chunks')
    hook._device_camera_scope = (start, frames, budget, p, chosen)
    return frames, budget, p, chosen


def draft_scores(query, descriptors, h, w, config):
    """The existing FP16-rounded mean/max probability arithmetic, unchanged."""
    import torch
    from draftkv.retrieval.draftattention2_importance import spatial8x16_mean_max
    qm, qx = spatial8x16_mean_max(query, h, w, capacity=config.capacity,
                                pool_h=config.pool_h, pool_w=config.pool_w)
    km, kx = descriptors.unbind(0)
    scale = qm.shape[-1]**-.5
    logits = (torch.matmul(qm.float(), km.float().transpose(-2, -1))*scale).to(torch.float16)
    prob = logits.float().softmax(-1)
    if float(config.maxpool_weight):
        logits = (torch.matmul(qx.float(), kx.float().transpose(-2, -1))*scale).to(torch.float16)
        prob = (1.-float(config.maxpool_weight))*prob + float(config.maxpool_weight)*logits.float().softmax(-1)
    prob = prob.to(torch.float16)
    return prob.float().sum(2).sum(1)/(prob.shape[1]*prob.shape[2])


def route(hook, context, h, w, archive):
    import torch
    from draftkv.backbones.budgets import bound_budget
    from draftkv.streaming.device_profiles import transport_group
    from draftkv.streaming.krope_8x16 import GpuKRoPEPlan, phase_table
    from .block_policy import applied_protection
    from .position import geometry_for
    layer, ft, start = context.layer, context.frame_tokens, context.start_frame
    hook.active['forward_id'] = context.forward_id
    allocation = bound_budget(hook.host, 'draftkv')
    if allocation.local or context.sp != 1 or archive.packed_group is None:
        raise BackboneError('KV_CONSTRAINT', 'device archive requires SP1 packed Full/Draft-only with L0')
    if hook.next_frame != start:
        raise BackboneError('DIRTY_HISTORY', 'device archive frontier mismatch')
    directories = getattr(hook, '_device_archive_directories', None)
    if directories is None:
        directories = hook._device_archive_directories = {}
    directory = directories.get(layer)
    if directory is None:
        if start:
            raise BackboneError('DIRTY_HISTORY', 'missing published device archive directory')
        directory = directories[layer] = ArchiveDirectory(h, w, context.query.shape[2],
                                                        archive.packed_group, context.query.device)
    if directory.frontier != start:
        raise BackboneError('DIRTY_HISTORY', 'directory not atomically published at clean commit')
    sink = tuple(range(min(context.sink_frames, start)))
    current = tuple(range(start, start+context.current_frames))
    capacity = min(allocation.retrieved, max(0, start-len(sink)))
    decision = None
    if capacity:
        if getattr(hook, 'settings', {}).get('retrieval_ablation', 'full') == 'full':
            frames, budget, protected, chunks = camera_scope(hook, context, capacity, context.query.device)
        else:
            frames = torch.arange(len(sink), start, device=context.query.device)
            budget = capacity
            protected = applied_protection(hook.host.spec.backbone, budget)
        scores = draft_scores(context.query, gather_descriptors(directory, frames), h, w, hook.importance)
        decision = select_device(scores[0], frames, budget, protected, directory)
    else:
        budget = 0
    total = (len(sink)+budget+len(current))*ft
    geometry = geometry_for(context, hook.host.spec.family)
    prefix = geometry.prefix
    prefix_tokens = prefix[0].shape[1] if prefix is not None else 0
    shape = (1, prefix_tokens+total, *context.query.shape[2:])
    full_k, full_v = (torch.empty(shape, dtype=context.key.dtype, device=context.key.device) for _ in range(2))
    if prefix_tokens:
        full_k[:, :prefix_tokens].copy_(prefix[0]); full_v[:, :prefix_tokens].copy_(prefix[1])
    k, v = full_k[:, prefix_tokens:], full_v[:, prefix_tokens:]
    bank = hook.committed.get(layer, {})
    for i, fid in enumerate(sink):
        hook._validate_sink(bank[fid])
        for dst, src in zip((k, v), bank[fid]):
            dst[:, i*ft:(i+1)*ft].copy_(src.unsqueeze(0))
    offset = (len(sink)+budget)*ft
    k[:, offset:].copy_(context.key); v[:, offset:].copy_(context.value)
    q = context.query.clone()
    for i, fid in enumerate(current):
        slot, a, b = len(sink)+budget+i, i*ft, (i+1)*ft
        if slot != fid:
            geometry.assign(q[:, a:b], geometry.rotate(geometry.extract(context.raw_query[:, a:b]), slot, q.dtype))
            geometry.assign(k[:, offset+a:offset+b], geometry.rotate(geometry.extract(context.raw_key[:, a:b]), slot, k.dtype))
    source = rotation = None
    if decision is not None:
        source = bind_device_archive(directory, decision, history_start=prefix_tokens+len(sink)*ft,
                                     total=prefix_tokens+total)
        n, group = budget*ft, directory.group
        dense = torch.arange(n, device=context.query.device, dtype=torch.int64)
        rotation = GpuKRoPEPlan(range(n), dense*group*128,
            (len(sink)+dense//ft).to(torch.int32), phase_table(geometry, hook.host.spec.family, context.query.device),
            ft, len(sink), 'h3' if hook.host.spec.family=='solarwm' else 'wan', group, (True,)*budget)
        history_ids = decision.device_tokens
    else:
        history_ids = torch.empty(0, device=context.query.device, dtype=torch.int64)
    source_ids = torch.cat((torch.arange(len(sink)*ft, device=context.query.device), history_ids,
                           torch.arange(start*ft, (start+len(current))*ft, device=context.query.device)))
    position = dict(position_policy='chrono_compact', packing_policy='chronological_frame_id',
        packing_unit='source_token', scoring_qk_space='native_post_rope', grid_hw=[h,w],
        frame_tokens=ft, sink_ids=list(sink), current_ids=list(current),
        history_budget=budget, history_tokens=budget*ft, token_budget=total,
        temporal_rule='packed_token_offset//frame_tokens', source_token_map_sha256=None,
        selected_ids_device_resident=True)
    pending = dict(context=context, current_ids=current, decision=decision, selected=(), consumed=False,
        consumer_bank={}, transfer_ids=tuple(range(budget)), direct_source=source, gpu_krope_plan=rotation,
        full_gpu_k=full_k, full_gpu_v=full_v, fixed_prefix=prefix, position=position,
        device_archive=True)
    hook.pending[layer] = pending
    return AttentionSelection(q.contiguous(), k, v, sink, (), (), current, (),
        dict(representation='device_archive', history_budget=budget, archive_frontier=start,
             device_source_tokens=source_ids, device_decision=decision,
             draftkv_plan=pending, packed_source_token_sha256=None))
