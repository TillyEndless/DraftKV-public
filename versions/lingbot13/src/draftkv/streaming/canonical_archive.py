"""Bounded immutable clean archive shared by canonical retrieval owners.

Native-task paths use GPU block/group preparation and native Draft descriptors;
CPU/page-reference paths retain token-major storage. Publication remains a
whole-clean-forward transaction after asynchronous D2H completes.
"""
import atexit

from .archive_arena import ArenaPool


class Completion:
    def __init__(self, prepared, begin, end, wait_ms):
        self.prepared, self.begin, self.end = prepared, begin, end
        self.wait_ms = wait_ms

    def wait(self):
        self.end.synchronize()

    def times(self):
        return dict(prepare_ms=self.prepared[0].elapsed_time(self.prepared[1]),
                    d2h_ms=self.begin.elapsed_time(self.end),
                    archive_wait_ms=self.wait_ms)


class CanonicalCleanArchive:
    def __init__(self, context, *, layers, capacity, temporal_dims, page_policy='base4k',
                 packed_group=None, grid=None):
        import torch
        if context.key.dtype != torch.bfloat16 or context.key.shape[-1] != 128:
            raise ValueError('canonical clean arena requires BF16 D128')
        if temporal_dims not in (0, 32, 44):
            raise ValueError('unsupported raw temporal layout')
        self.frames = context.current_frames
        self.ft, self.heads = context.frame_tokens, context.key.shape[2]
        self.capacity, self.temporal_dims = capacity, temporal_dims
        self.device = context.key.device
        self.page_policy = page_policy
        self.packed_group, self.grid = packed_group, grid
        self.sink_frames=getattr(context,'sink_frames',0)
        if packed_group is not None and (self.heads % packed_group or grid is None
                or grid[0]*grid[1] != self.ft):
            raise ValueError('invalid canonical GPU prepack geometry')
        self.pool = ArenaPool(self.device, layers, self.frames, self.ft,
                              heads=self.heads, max_frames=capacity,
                              width=256+temporal_dims, page_policy=page_policy)
        self.copy_stream = torch.cuda.Stream(device=self.device)
        self.slots, self.next_slot = [], 0
        atexit.register(self.pool.close)

    def stage(self, context, current_ids, geometry, *, importance, descriptors=True):
        import torch
        from draftkv.retrieval.draftattention2_importance import descriptor
        ids = tuple(current_ids)
        n = len(ids)
        if (not context.clean or n != context.current_frames or
                ids != tuple(range(context.start_frame, context.start_frame+n)) or
                context.start_frame % self.frames or
                context.key.shape != (1, n*self.ft, self.heads, 128) or
                context.key.device != self.device or context.key.dtype != torch.bfloat16 or
                n != min(self.frames, self.capacity-context.start_frame)):
            raise ValueError('canonical clean arena geometry/frontier mismatch')
        elements = n*self.ft*self.heads*(256+self.temporal_dims)
        host, owner, wait_ms, index = self.pool.slice(
            context.layer, context.start_frame//self.frames, elements)
        if len(self.slots) < 2:
            self.slots.append({})
        slot = self.slots[self.next_slot % 2]
        self.next_slot += 1
        current_stream = torch.cuda.current_stream(self.device)
        if slot.get('release') is not None:
            current_stream.wait_event(slot['release'])
        # Reuse capacity-sized slots. Source tensors can die after the GPU
        # preparation; only these slots must survive asynchronous D2H.
        if slot.get('storage') is None:
            slot['storage'] = torch.empty(self.frames*self.ft*self.heads*
                                          (256+self.temporal_dims),
                                          device=self.device, dtype=torch.bfloat16)
        packed = slot['storage'][:elements]
        kv_count = n*self.ft*self.heads*128
        prepared = tuple(torch.cuda.Event(enable_timing=True) for _ in range(2))
        prepared[0].record()
        packed_descriptors = {}
        if self.packed_group is not None:
            from .canonical_prepack import prepare_kernel, token_layout
            from .prepacked import _device_geometry
            h,w=self.grid
            if importance is None or (importance[2].pool_h,importance[2].pool_w)!=(8,16):
                raise ValueError('canonical GPU archive requires production 8x16')
            blocks=token_layout(h,w)[0]
            indices,counts,starts=_device_geometry(h,w,8,16,self.packed_group,str(self.device))
            raw=context.raw_key if self.temporal_dims else context.key
            prepare_kernel()[(n,len(blocks),self.heads)](context.key,context.value,raw,
                packed,indices,counts,starts,self.ft,self.heads,self.packed_group,
                len(blocks),128,n,self.temporal_dims,self.temporal_dims==32,
                context.start_frame,self.sink_frames,
                context.key.stride(1),context.key.stride(2),context.value.stride(1),
                context.value.stride(2),raw.stride(1),raw.stride(2),True,num_warps=4)
            # Preserve the production CUDA reduction, including FP32 summation
            # order and FP16 tie rounding. A fused Triton reduction can differ
            # by one FP16 ULP and change a boundary Top-K selection.
            if descriptors:
                config=importance[2]
                for rank,fid in enumerate(ids):
                    packed_descriptors[fid]=descriptor(context.key[:,rank*self.ft:(rank+1)*self.ft],
                        h,w,capacity=config.capacity,pool_h=config.pool_h,pool_w=config.pool_w).detach()
        else:
            packed[:kv_count].view_as(context.key).copy_(context.key)
            packed[kv_count:2*kv_count].view_as(context.value).copy_(context.value)
            if self.temporal_dims:
                raw = geometry.extract(context.raw_key)
                packed[2*kv_count:].view_as(raw).copy_(raw)
        prepared[1].record()
        begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        with torch.cuda.stream(self.copy_stream):
            self.copy_stream.wait_event(prepared[1])
            begin.record()
            host.copy_(packed, non_blocking=True)
            end.record()
        slot['release'] = end
        if self.packed_group is not None:
            from .canonical_prepack import PackedFrame, LogicalArchiveTensor
            frames={fid:PackedFrame(host,i,n,*self.grid,self.heads,self.packed_group,
                                   self.temporal_dims,fid<self.sink_frames,
                                   fid<self.sink_frames) for i,fid in enumerate(ids)}
            return dict(rows={fid:tuple(LogicalArchiveTensor(f,side) for side in (0,1))
                              for fid,f in frames.items()},
                        raw={fid:LogicalArchiveTensor(f,0,True) for fid,f in frames.items()}
                            if self.temporal_dims else {},
                        descriptors=packed_descriptors,
                        registered=True,nbytes=2*kv_count*2,
                        completion=Completion(prepared,begin,end,wait_ms),
                        arena_index=index,arena_owner=owner)
        keys = host[:kv_count].view(n, self.ft, self.heads, 128)
        values = host[kv_count:2*kv_count].view_as(keys)
        raw = (host[2*kv_count:].view(n, self.ft, self.heads, self.temporal_dims)
               if self.temporal_dims else None)
        desc = {}
        if descriptors:
            h, w = importance[:2]
            config = importance[2]
            for rank, fid in enumerate(ids):
                desc[fid] = descriptor(context.key[:, rank*self.ft:(rank+1)*self.ft],
                    h, w, capacity=config.capacity, pool_h=config.pool_h,
                    pool_w=config.pool_w).detach()
        return dict(rows={fid:(keys[i], values[i]) for i,fid in enumerate(ids)},
                    raw={fid:raw[i] for i,fid in enumerate(ids)} if raw is not None else {},
                    descriptors=desc, registered=True,
                    nbytes=2*kv_count*2,
                    completion=Completion(prepared,begin,end,wait_ms),
                    arena_index=index, arena_owner=owner)


def ensure_archive(hook, context):
    """Start preparation on first real route, inside measured diffusion."""
    if not context.key.is_cuda:
        return None
    if getattr(hook, 'gpu_history', False):
        return None  # residency ablations must never allocate/offload host KV
    settings = getattr(hook, 'settings', {})
    # Allocation is independent of consumer/backend/schedule. Every host-history
    # production and ablation owner gets the same repair, including B01 FA2.
    archive = getattr(hook, '_canonical_clean_archive', None)
    if archive is None:
        from draftkv.algorithms.draftkv.position import geometry_for
        g = geometry_for(context, hook.host.spec.family)
        capacity = hook.host.plan.get('history_plan', {}).get('capacity_frames')
        if type(capacity) is not int or capacity < 1:
            raise ValueError('native pipeline requires positive history_plan.capacity_frames')
        page_policy = hook.archive_page_policy
        packed={}
        native_config=getattr(hook,'_native_runtime_config',lambda c:None)(context)
        full_reference=getattr(hook,'ablation',{}).get('retrieval',{}).get('selector')=='all_history'
        if (not full_reference and native_config is not None and getattr(hook,'_use_direct_history',lambda c:False)(context)
                and getattr(hook.binding,'host_pages_policy','quality')=='quality'):
            p=hook.binding.parameters
            heads=context.key.shape[2]
            from .device_profiles import transport_group
            group=transport_group(native_config,p,heads)
            packed=dict(packed_group=group,grid=hook.geometry[:2])
        archive = CanonicalCleanArchive(context, layers=hook.host.spec.layers,
            capacity=capacity, temporal_dims=0 if hook.position_policy=='native_post_rope' else g.dimensions,
            page_policy=page_policy,**packed)
        hook._canonical_clean_archive = archive
        hook._clean_registered_owners.append(archive)
        hook.archive_page_policy = page_policy
        hook.host.trace('draftkv_canonical_archive', allocator='bounded_ahead_of_use_arena',
            page_policy=page_policy, capacity_frames=capacity,
            chunks_per_arena=archive.pool.chunks,
            timing_scope='inside_generation_first_route',
            layout='gpu_block_group_paired' if packed else 'native_token_major_with_raw_temporal',
            head_group=packed.get('packed_group'),
            device_profile=(native_config or {}).get('device_profile'),
            preferred_head_group=(native_config or {}).get('preferred_head_group'),
            group_adjusted=(native_config or {}).get('group_adjusted',False),
            cpu_fetch_packing=not bool(packed), gpu_fused_descriptors=False,
            descriptor_reduction='unchanged_production')
    return archive
