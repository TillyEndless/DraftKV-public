"""Additional canonical groups, isolated from the qualified Full GPU route.

GPU selection and DMA planning cover camera-only and text-only LongLive.
CPU K-RoPE deliberately retains CPU rotation and a selected-only ID bridge;
no history-wide host scan or CPU selection is required by that ablation.
"""
from copy import copy
from dataclasses import replace
from itertools import groupby

from draftkv.backbones.contracts import AttentionSelection, BackboneError
from draftkv.streaming.device_archive import (
    ArchiveDirectory, DeviceDecision, gather_descriptors, select_device, bind_device_archive,
)
from .device_archive_route import camera_scope, draft_scores


def enabled(hook, context):
    settings = getattr(hook, 'settings', {})
    eligible = (context.query.is_cuda and context.sp == 1
        and settings.get('device_archive_plan', True)
        and hook.host.spec.backbone in ('LongLive1.3B', 'LingBot1.3B', 'LingBot14B', 'SolarWM33B')
        and settings.get('retrieval_ablation', 'full') in ('full', 'draft-only', 'camera-only')
        and getattr(hook, 'position_policy', 'chrono_compact') == 'chrono_compact'
        and getattr(hook.binding, 'host_pages_policy', 'quality') == 'quality'
        and not getattr(hook, 'gpu_history', False)
        and not getattr(hook, 'ablation', {})
        and not settings.get('gpu_prepack', False) and not settings.get('gpu_device_route', False)
        and hook._native_runtime_config(context) is not None)
    if eligible and settings.get('retrieval_ablation','full') == 'camera-only':
        from draftkv.backbones.budgets import bound_budget
        allocation = bound_budget(hook.host,'draftkv')
        local = min(allocation.local,max(0,context.start_frame-context.sink_frames))
        capacity = min(allocation.retrieved,max(0,context.start_frame-context.sink_frames-local))
        # Unusual P/L overrides can let selected remote chunks overlap P. The
        # reference then has a data-dependent underfilled shape. Preserve that
        # owner instead of reading a GPU scalar or changing its candidate set.
        eligible = camera_shape_is_static(hook,context,capacity,local)
    return eligible


def camera_shape_is_static(hook,context,capacity,local_frames):
    from .block_policy import configured_protection
    start,chunk,sink = context.start_frame,hook.host.spec.chunk,context.sink_frames
    first = (sink+chunk-1)//chunk
    stop = max(first,min(start//chunk-2,(start-local_frames)//chunk))
    p = min(configured_protection(hook.host.spec.backbone),capacity,max(0,start-sink-local_frames))
    return stop == first or not p or stop*chunk <= start-local_frames-p


class GroupArchiveDirectory(ArchiveDirectory):
    """Same stable GPU directory; Camera-only has no Draft descriptors.

    Native LongLive sink recache uses registered token-major sink owners.
    Their KV pointers are never selected by historical fetch, but their
    ownership and descriptor replacements must publish atomically too.
    """
    def __init__(self, *args, sink_frames, descriptors=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.sink_frames = sink_frames
        self.requires_descriptors = descriptors

    def publication(self, rows, descriptors):
        import torch
        from draftkv.streaming.canonical_prepack import LogicalArchiveTensor
        ids = tuple(rows)
        if (not ids or ids != tuple(range(ids[0], ids[0]+len(ids)))
                or ids[0] > self.frontier):
            raise ValueError('device archive publication must follow the clean frontier')
        end = ids[-1]+1
        if ids[0] < self.frontier and end > self.frontier:
            raise ValueError('replacement publication overlaps the frontier')
        pointers, owners, bases = [], [], {}
        for fid in ids:
            key, value = rows[fid]
            if isinstance(key, LogicalArchiveTensor):
                f = key.frame
                if (not isinstance(value, LogicalArchiveTensor) or value.frame is not f
                        or key.side != 0 or value.side != 1
                        or (f.h,f.w,f.heads,f.group) != (self.h,self.w,self.heads,self.group)
                        or not f.slab.is_pinned() or f.slab.element_size() != 2):
                    raise ValueError('invalid paired immutable packed allocation')
                identity = id(f.slab)
                if identity not in bases:
                    bases[identity] = f.slab.data_ptr()
                base = bases[identity]+f.frame*2*self.ft*self.heads*128*2
            else:
                if (fid >= self.sink_frames or not isinstance(key, torch.Tensor)
                        or not isinstance(value, torch.Tensor)
                        or key.shape != (self.ft,self.heads,128) or value.shape != key.shape
                        or key.dtype != torch.bfloat16 or value.dtype != key.dtype
                        or not key.is_pinned() or not value.is_pinned()):
                    raise ValueError('only native sink recache may publish token-major owners')
                base = key.data_ptr()  # sink KV is never a historical DMA source
            desc = descriptors.get(fid)
            if self.requires_descriptors and desc is None:
                raise ValueError('missing committed Draft descriptor')
            if desc is not None and (desc.shape != (2,1,self.heads,self.regions,128)
                    or desc.dtype != torch.float16 or not desc.is_contiguous()
                    or desc.device != self.device):
                raise ValueError('device archive descriptor layout changed')
            pointers.append((base, 0 if desc is None else desc.data_ptr()))
            owners.append((key,value,desc))
        tail = torch.tensor(pointers, device=self.device, dtype=torch.int64)
        new = copy(self)
        if ids[0] == self.frontier:
            new.pointers = torch.cat((self.pointers, tail), 0)
        else:
            new.pointers = self.pointers.clone()
            new.pointers[ids[0]:end].copy_(tail)
        new.frontier = max(self.frontier, end)
        new.owners = self.owners+(tuple(owners),)
        return new


def camera_only_scope(hook, context, capacity, local_frames):
    """Original BFO lexicographic ranking with L excluded and true P retained."""
    import torch
    from .block_policy import configured_protection
    from .camera_frustum_v2 import _sample_camera_points, _inside, EPS
    start, chunk, sink = context.start_frame, hook.host.spec.chunk, context.sink_frames
    if not camera_shape_is_static(hook,context,capacity,local_frames):
        raise BackboneError('KV_CONSTRAINT','overlapping remote/P shape belongs to the reference owner')
    p = min(configured_protection(hook.host.spec.backbone), capacity,
            max(0,start-sink-local_frames))
    first = (sink+chunk-1)//chunk
    stop = max(first,min(start//chunk-2,(start-local_frames)//chunk))
    top = min(hook.camera_top_chunks, stop-first)
    key = (start,capacity,local_frames,id(hook.camera_poses),hook.camera_top_chunks)
    cached = getattr(hook,'_device_camera_only_scope',None)
    if cached is not None and cached[0] == key:
        return cached[1]
    if hook.camera_poses is None:
        raise BackboneError('CAMERA_POSES_MISSING','Camera-only needs actual native C2W')
    with torch.autocast(device_type='cuda',enabled=False):
        # Geometry itself is exactly the qualified GPU FP32 BFO operation.
        poses = getattr(hook,'_device_camera_only_poses',None)
        if poses is None or getattr(hook,'_device_camera_only_owner',None) is not hook.camera_poses:
            poses = hook.camera_poses.to(device=context.query.device)
            hook._device_camera_only_poses = poses
            hook._device_camera_only_owner = hook.camera_poses
            hook._device_camera_only_intrinsics = hook._frustum_intrinsics.to(device=context.query.device)
        k = hook._device_camera_only_intrinsics
        eligible = torch.arange(first,stop,device=context.query.device,dtype=torch.int64)
        if top:
            cid = start//chunk
            observed = poses[:start+chunk].reshape(cid+1,chunk,4,4)
            rotation = observed[:,:,:3,:3].float().mean(1)
            u,_,vh = torch.linalg.svd(rotation)
            negative = torch.linalg.det(u@vh)<0
            corrected = u.clone()
            corrected[:,:,-1] *= torch.where(negative,-1.,1.)[:,None]
            rotations = corrected@vh
            centers_by_chunk = observed[:,:,:3,3].float().mean(1)
            cr,ct = rotations[cid],centers_by_chunk[cid]
            rr,tt = rotations[eligible],centers_by_chunk[eligible]
            centers = poses[:start+chunk,:3,3].float()
            radius = torch.linalg.vector_norm(centers-centers.mean(0),dim=-1).max()
            step = torch.linalg.vector_norm(centers[1:]-centers[:-1],dim=-1)
            scale = torch.maximum(radius,step.median()*16.).clamp_min(1e-4)
            points = _sample_camera_points(k,hook._frustum_width,hook._frustum_height,scale)
            world = torch.matmul(points.unsqueeze(0),rr.transpose(-1,-2))+tt[:,None,:]
            forward = _inside(torch.matmul(world-ct,cr),k,hook._frustum_width,hook._frustum_height).float().mean(-1)
            world_current = torch.matmul(points,cr.transpose(-1,-2))+ct
            reverse = _inside(torch.matmul(world_current.unsqueeze(0)-tt[:,None,:],rr),k,hook._frustum_width,hook._frustum_height).float().mean(-1)
            score = 2.*forward*reverse/(forward+reverse+EPS)
            td = ((tt-ct)**2).sum(-1)
            relative = rr.transpose(-1,-2)@cr.unsqueeze(0)
            rd = torch.acos(((relative.diagonal(dim1=-2,dim2=-1).sum(-1)-1.)/2.).clamp(-1.,1.))
            td = td/torch.where(td.max()>0,td.max(),torch.ones_like(td.max()))
            rd = rd/torch.where(rd.max()>0,rd.max(),torch.ones_like(rd.max()))
            distance = .5*td+.5*rd
            order = torch.argsort(distance,stable=True)
            order = order[torch.argsort(score[order],descending=True,stable=True)]
            chosen = eligible[order[:top]]
            remote = (chosen[:,None]*chunk+torch.arange(chunk,device=context.query.device)[None,:]).flatten()
        else:
            remote = torch.empty(0,device=context.query.device,dtype=torch.int64)
        protected = torch.arange(start-local_frames-p,start-local_frames,device=context.query.device)
        frames = torch.cat((remote,protected)).sort().values
    hook._device_camera_only_scope = (key,frames)
    return frames


def selection(hook, context, directory, allocation, h, w):
    import torch
    from .block_policy import applied_protection, configured_protection
    start, sink = context.start_frame,min(context.sink_frames,context.start_frame)
    mode = hook.settings.get('retrieval_ablation','full')
    local = min(allocation.local,max(0,start-sink))
    if local and mode != 'camera-only':
        raise BackboneError('KV_CONSTRAINT','local Block DraftKV allocation requires Camera-only')
    count = max(0,start-sink-local)
    capacity = min(allocation.retrieved,count)
    decision = None
    if capacity:
        if mode == 'camera-only':
            frames = camera_only_scope(hook,context,capacity,local)
            budget = frames.numel()
            ids = (frames[:,None]*directory.ft+torch.arange(directory.ft,device=frames.device)[None,:]).flatten()
            if budget:
                decision = DeviceDecision(ids,frames,None,None,None,budget,
                    min(capacity,configured_protection(hook.host.spec.backbone)))
        else:
            if mode == 'full' and hook.host.spec.family != 'longlive':
                frames,budget,p,_ = camera_scope(hook,context,capacity,context.query.device)
            else:
                frames = torch.arange(sink,start,device=context.query.device)
                budget,p = capacity,applied_protection(hook.host.spec.backbone,capacity)
            scores = draft_scores(context.query,gather_descriptors(directory,frames),h,w,hook.importance)
            decision = select_device(scores[0],frames,budget,p,directory)
    if local:
        frames = torch.arange(start-local,start,device=context.query.device)
        ids = (frames[:,None]*directory.ft+torch.arange(directory.ft,device=frames.device)[None,:]).flatten()
        if decision is None:
            decision = DeviceDecision(ids,frames,None,None,None,local,0)
        else:
            decision = replace(decision,device_tokens=torch.cat((decision.device_tokens,ids)),
                candidates=torch.cat((decision.candidates,frames)),history_budget=decision.history_budget+local)
    return decision


def cpu_krope_source(hook, context, decision, directory, geometry, sink, prefix_tokens, total):
    """Selected-only compatibility bridge required by the CPU rotation ablation."""
    from .direct_pack import materialize_block_history, bound_direct_source
    ft = directory.ft
    selected = decision.device_tokens.to(device='cpu').tolist()
    history = tuple((fid,tuple(t%ft for t in tokens)) for fid,tokens in
                    groupby(selected,key=lambda t:t//ft))
    slab,_ = materialize_block_history(hook.committed[context.layer],
        hook.time_coordinates.raw_temporal.get(context.layer,{}),history,
        geometry=geometry,coefficients=hook.time_coordinates.cpu_coefficients_for(geometry),
        chrono=True,sink_count=len(sink),frame_tokens=ft,heads=directory.heads,
        head_group=directory.group,cache=hook._b01_pinned_staging.setdefault('direct_block',{}),
        layer=context.layer,pin_memory=True,gpu_krope=False)
    return bound_direct_source(slab,budget=decision.history_budget,frame_tokens=ft,
        prefix_tokens=prefix_tokens,sink_tokens=len(sink)*ft,total_tokens=total,
        heads=directory.heads,head_group=directory.group)


def route(hook, context, h, w, archive):
    import torch
    from draftkv.backbones.budgets import bound_budget
    from draftkv.streaming.krope_8x16 import GpuKRoPEPlan, phase_table
    from .position import geometry_for
    layer,ft,start = context.layer,context.frame_tokens,context.start_frame
    hook.active['forward_id'] = context.forward_id
    allocation = bound_budget(hook.host,'draftkv')
    if archive is None or archive.packed_group is None:
        raise BackboneError('KV_CONSTRAINT','device archive groups require native packed owners')
    directories = getattr(hook,'_device_archive_directories',None)
    if directories is None:
        directories = hook._device_archive_directories = {}
    directory = directories.get(layer)
    if directory is None:
        if start:
            raise BackboneError('DIRTY_HISTORY','missing published device archive directory')
        directory = directories[layer] = GroupArchiveDirectory(h,w,context.query.shape[2],archive.packed_group,
            context.query.device,sink_frames=context.sink_frames,
            descriptors=getattr(hook,'use_draft_descriptors',True))
    recache = hook.active.get('recache',False) and start == 0 and context.clean and hook.host.spec.family == 'longlive'
    if (hook.next_frame != start or directory.frontier != start) and not recache:
        raise BackboneError('DIRTY_HISTORY','directory/clean frontier mismatch')
    sink = tuple(range(min(context.sink_frames,start)))
    current = tuple(range(start,start+context.current_frames))
    decision = selection(hook,context,directory,allocation,h,w)
    budget = 0 if decision is None else decision.history_budget
    total = (len(sink)+budget+len(current))*ft
    geometry = geometry_for(context,hook.host.spec.family)
    prefix = geometry.prefix
    prefix_tokens = prefix[0].shape[1] if prefix is not None else 0
    shape = (1,prefix_tokens+total,*context.query.shape[2:])
    full_k,full_v = (torch.empty(shape,dtype=context.key.dtype,device=context.key.device) for _ in range(2))
    if prefix_tokens:
        full_k[:,:prefix_tokens].copy_(prefix[0]);full_v[:,:prefix_tokens].copy_(prefix[1])
    k,v = full_k[:,prefix_tokens:],full_v[:,prefix_tokens:]
    bank = hook.committed.get(layer,{})
    for i,fid in enumerate(sink):
        hook._validate_sink(bank[fid])
        for dst,src in zip((k,v),bank[fid]):
            dst[:,i*ft:(i+1)*ft].copy_(src.unsqueeze(0))
    offset = (len(sink)+budget)*ft
    k[:,offset:].copy_(context.key);v[:,offset:].copy_(context.value)
    q = context.query.clone()
    for i,fid in enumerate(current):
        slot,a,b = len(sink)+budget+i,i*ft,(i+1)*ft
        if slot != fid:
            geometry.assign(q[:,a:b],geometry.rotate(geometry.extract(context.raw_query[:,a:b]),slot,q.dtype))
            geometry.assign(k[:,offset+a:offset+b],geometry.rotate(geometry.extract(context.raw_key[:,a:b]),slot,k.dtype))
    source = rotation = None
    cpu_rope = not hook.settings.get('gpu_krope_8x16',True)
    if decision is not None:
        if cpu_rope:
            source = cpu_krope_source(hook,context,decision,directory,geometry,sink,prefix_tokens,total)
        else:
            source = bind_device_archive(directory,decision,history_start=prefix_tokens+len(sink)*ft,total=prefix_tokens+total)
            n,group = budget*ft,directory.group
            dense = torch.arange(n,device=context.query.device,dtype=torch.int64)
            rotation = GpuKRoPEPlan(range(n),dense*group*128,(len(sink)+dense//ft).to(torch.int32),
                phase_table(geometry,hook.host.spec.family,context.query.device),ft,len(sink),
                'h3' if hook.host.spec.family=='solarwm' else 'wan',group,(True,)*budget)
        ids = decision.device_tokens
    else:
        ids = torch.empty(0,device=context.query.device,dtype=torch.int64)
    source_ids = torch.cat((torch.arange(len(sink)*ft,device=context.query.device),ids,
        torch.arange(start*ft,(start+len(current))*ft,device=context.query.device)))
    position = dict(position_policy='chrono_compact',packing_policy='chronological_frame_id',
        packing_unit='source_token',scoring_qk_space='native_post_rope',grid_hw=[h,w],frame_tokens=ft,
        sink_ids=list(sink),current_ids=list(current),history_budget=budget,history_tokens=budget*ft,
        token_budget=total,temporal_rule='packed_token_offset//frame_tokens',source_token_map_sha256=None,
        selected_ids_device_resident=not cpu_rope,cpu_krope_selected_only_bridge=cpu_rope)
    pending = dict(context=context,current_ids=current,decision=decision,selected=(),consumed=False,
        consumer_bank={},transfer_ids=tuple(range(budget)),direct_source=source,gpu_krope_plan=rotation,
        full_gpu_k=full_k,full_gpu_v=full_v,fixed_prefix=prefix,position=position,device_archive=True)
    hook.pending[layer] = pending
    return AttentionSelection(q.contiguous(),k,v,sink,(),(),current,(),
        dict(representation='device_archive',history_budget=budget,archive_frontier=start,
            device_source_tokens=source_ids,device_decision=decision,draftkv_plan=pending,
            cpu_krope_selected_only_bridge=cpu_rope,packed_source_token_sha256=None))
