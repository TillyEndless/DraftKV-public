"""Opt-in SM120 GPU routing + native copy engine + asynchronous immutable clean arenas."""
import time
from draftkv.backbones.contracts import AttentionSelection,BackboneError,FramePage
from .prepacked_sm120_backbone import GPUPreparedSM120Hook
from .position import geometry_for
from .block_policy import applied_protection
from draftkv.streaming.routed_sm120 import native,prepare_current,plan_route,execute_plan
from draftkv.streaming.prepacked import PreparedChunk
from draftkv.streaming.routed_sm120_arena import ArenaPool

class GPUBlockSelection(AttentionSelection):
    def validate(self,c):
        # This ABI is exclusive to this provider. Construction proves disjoint
        # frame/block domains; full token provenance is an explicit validation view.
        p=self.metadata['gpu_plan'];ft=c.frame_tokens
        if (p.start!=c.start_frame or p.total!=(p.sink+p.budget+c.current_frames)*ft
            or p.total>c.window_frames*ft or self.query.shape!=c.query.shape
            or self.keys.shape!=self.values.shape or self.keys.shape!=(1,p.total,12,128)
            or p.offsets.shape!=(1,2,p.total) or p.slots.shape!=(p.total,)
            or self.current_ids!=tuple(range(c.start_frame,c.start_frame+c.current_frames))):
            raise BackboneError('KV_CONSTRAINT','GPU block-plan ABI mismatch')
        for tensor in (self.query,self.keys,self.values):
            if tensor.device!=c.query.device or tensor.dtype!=c.query.dtype:
                raise BackboneError('KV_CONSTRAINT','GPU block-plan dtype/device mismatch')


def validation_view(selection,c):
    """Deliberately synchronizing oracle adapter. Never called by production."""
    p=selection.metadata['gpu_plan']
    if p.sources.numel()!=p.total:raise ValueError('enable route_validation for the explicit provenance view')
    sources=p.sources.cpu().tolist();slots=p.slots.cpu().tolist();ft=c.frame_tokens
    pages=[];cursor=0
    for fid in sorted(set(token//ft for token in sources)):
        tokens=tuple(token%ft for token in sources if token//ft==fid)
        pages.append(FramePage(fid,len(pages),cursor,cursor+len(tokens),tokens));cursor+=len(tokens)
    remote=tuple(page.frame_id for page in pages if page.frame_id not in selection.sink_ids+selection.local_ids+selection.current_ids)
    return AttentionSelection(selection.query,selection.keys,selection.values,selection.sink_ids,selection.local_ids,remote,selection.current_ids,tuple(pages),
        dict(representation='paged',position_map=tuple((slot,token%ft) for slot,token in zip(slots,sources))))


class GPURoutedSM120Hook(GPUPreparedSM120Hook):
    def __init__(self,host):
        # Invoke the grandparent to avoid compiling the previous SM120 identity.
        from .block_backbone import BlockDraftKVHook
        BlockDraftKVHook.__init__(self,host)
        if host.spec.backbone!='LingBot1.3B' or int(host.plan.get('sp',1))!=1:raise ValueError('SM120 route requires LingBot1.3B/SP1')
        if (self.importance.pool_h,self.importance.pool_w)!=(32,32):raise ValueError('this experiment freezes 32x32')
        self.group=12;self.batch_bytes=int(self.settings.get('prepack_dma_bytes',4194304))
        if int(self.settings.get('prepack_head_group',12))!=12:raise ValueError('this experiment freezes G12')
        self.splits=int(self.settings.get('prepack_splits',4))
        if self.splits not in (1,4):raise ValueError('frozen A/S1 and B/S4 configuration required')
        self.chrono=self.position_policy=='chrono_compact'
        if self.chrono and self.settings.get('prepack_chrono_placement','fetch')!='fetch':raise ValueError('B retains verified fetch rotation')
        self.copy_stream=None;self.execution_cache={};self.preparation_cache={};self.tickets=[];self.arena=None
        self.device_banks={};self.address_banks={};self.frame_capacity=int(self.settings.get('route_frame_capacity',128))
        self.logging_mode='bounded_gpu_block_summary'
        self.runtime=None if host.plan.get('gpu_profile', False) else native()

    def _layout(self,c):
        if self.runtime is None:
            self.runtime=native()
        import torch
        h,w=self.geometry[:2];r=((h+31)//32)*((w+31)//32)
        if c.start_frame+c.current_frames>self.frame_capacity:raise ValueError('frozen route frame capacity exceeded')
        if self.arena is None:
            latent=self.host.plan.get('request',{}).get('latent_frames')
            generations=((latent+self.host.spec.chunk-1)//self.host.spec.chunk) if isinstance(latent,int) and latent>0 else None
            self.arena=ArenaPool(c.key.device,self.host.spec.layers,self.host.spec.chunk,h*w,int(self.settings.get('route_arena_chunks',4)),max_generations=generations)
            self.copy_stream=torch.cuda.Stream(device=c.key.device)
        if c.layer not in self.device_banks:
            self.device_banks[c.layer]=torch.empty((2,12,self.frame_capacity,r,128),device=c.key.device,dtype=torch.float16)
            self.address_banks[c.layer]=torch.empty(self.frame_capacity,device=c.key.device,dtype=torch.int64)
        return self.device_banks[c.layer],self.address_banks[c.layer]

    def _collect(self,wait=False):
        remain=[]
        for ticket,pending in self.tickets:
            metrics=ticket.metrics(wait)
            if not metrics:remain.append((ticket,pending));continue
            p=pending['plan'];begin,scored,end=p.route_events
            self.host.trace('draftkv_gpu_route_attention',layer=pending['layer'],start_frame=p.start,clean=pending['clean'],
                route_wall_ms=pending['route_wall_ms'],consume_submit_ms=pending['consume_submit_ms'],
                score_gpu_ms=begin.elapsed_time(scored),route_materialize_gpu_ms=scored.elapsed_time(end),
                current_prepare_ms=p.current.prepare_start.elapsed_time(p.current.prepare_end),
                source_identity=self.runtime.source_identity,position_policy=self.position_policy,splits=self.splits,
                descriptor_d2h_bytes=p.table.numel()*8,token_metadata_h2d_bytes=0,logging_mode=self.logging_mode,
                **metrics)
        self.tickets=remain

    def route(self,c):
        import torch
        begin=time.perf_counter_ns();h,w=self._verify_context(c);ft=h*w
        if c.query.shape[2:]!=(12,128) or c.sp!=1 or c.fixed_prefix is not None:raise ValueError('SM120 route context mismatch')
        if c.start_frame!=self.next_frame:raise BackboneError('DIRTY_HISTORY','GPU route frontier mismatch')
        self._collect();bank,addresses=self._layout(c);self.active['forward_id']=c.forward_id
        sink=min(c.sink_frames,c.start_frame)
        from draftkv.backbones.budgets import bound_budget
        budget=min(bound_budget(self.host,'draftkv').retrieved,max(0,c.start_frame-sink))
        protect=min(applied_protection(self.host.spec.backbone,budget),budget) if budget else 0
        current=prepare_current(c.key,c.value,c.raw_key,h,w,32,32,self.chrono,self.preparation_cache)
        geo=geometry_for(c,self.host.spec.family);phase=torch.view_as_real(geo.coefficients[0]).flatten(-2).contiguous()
        q=c.query
        if self.chrono:
            q=q.clone()
            for frame in range(c.current_frames):
                slot=sink+budget+frame;fid=c.start_frame+frame;lo=frame*ft;hi=lo+ft
                if slot!=fid:geo.assign(q[:,lo:hi],geo.rotate(geo.extract(c.raw_query[:,lo:hi]),slot,q.dtype))
        plan=plan_route(q.contiguous(),current,bank,addresses,phase,h,w,self.importance,c.start_frame,sink,budget,protect,self.splits,self.batch_bytes,self.execution_cache,score_query=c.query,validation=self.settings.get('route_validation',False))
        pending=dict(context=c,current_ids=tuple(range(c.start_frame,c.start_frame+c.current_frames)),consumed=False,plan=plan,
            current_preparation=current,route_wall_ms=(time.perf_counter_ns()-begin)/1e6)
        self.pending[c.layer]=pending
        # Shape-only views: the native consumer exclusively reads the GPU plan.
        k=c.key[:,:1].expand(1,plan.total,12,128);v=c.value[:,:1].expand_as(k)
        return GPUBlockSelection(q.contiguous(),k,v,tuple(range(sink)),tuple(range(c.start_frame-protect,c.start_frame)),(),pending['current_ids'],(),
            dict(representation='gpu_block_plan',gpu_plan=plan,logging_mode=self.logging_mode))

    def consume(self,c,selection,native_fa2):
        pending=self.pending[c.layer];p=pending['plan']
        if selection.metadata.get('gpu_plan') is not p:raise BackboneError('DIRTY_HISTORY','GPU plan owner mismatch')
        begin=time.perf_counter_ns();ticket=execute_plan(p,self.splits)
        pending['consumed']=True
        self.tickets.append((ticket,dict(plan=p,layer=c.layer,clean=c.clean,route_wall_ms=pending['route_wall_ms'],consume_submit_ms=(time.perf_counter_ns()-begin)/1e6)))
        return p.out

    def _stage_clean_layer(self,layer,pending):
        import torch
        c=pending['context'];h,w=self.geometry[:2]
        if not c.clean:raise BackboneError('DIRTY_HISTORY','noisy KV cannot be archived')
        bank,addresses=self._layout(c)
        # Manual replay seeding has no preceding route; production always supplies it.
        chunk=pending.get('current_preparation')
        if chunk is None:chunk=prepare_current(c.key,c.value,c.raw_key,h,w,32,32,self.chrono,self.preparation_cache)
        begin=time.perf_counter_ns();slab,owner,wait,index=self.arena.slice(layer,self.commit_generation,chunk.slab.numel())
        if index not in self.arena.reported:
            self.arena.reported.add(index);self._clean_registered_owners.append(owner)
            self.host.trace('draftkv_gpu_route_arena',arena=index,**owner.metrics,allocation_scope='included_in_process_and_generation_wall',preallocated_chunks=owner.chunks)
        copy_begin=torch.cuda.Event(enable_timing=True);copy_end=torch.cuda.Event(enable_timing=True)
        # Attention completion is already a device dependency on the model stream.
        ready=torch.cuda.Event();ready.record();self.copy_stream.wait_event(ready)
        with torch.cuda.stream(self.copy_stream):
            copy_begin.record();slab.copy_(chunk.slab,non_blocking=True);copy_end.record();chunk.slab.record_stream(self.copy_stream)
        chunk._slot['release']=copy_end
        start=c.start_frame;stop=start+c.current_frames
        bank[:,:,start:stop].copy_(chunk.descriptors.permute(1,2,0,3,4),non_blocking=True)
        archived=PreparedChunk(slab,bank[:,:,start:stop],copy_end,c.current_frames,h,w,32,32,12,self.chrono,chunk.prepare_start,chunk.prepare_end,copy_begin,owner)
        framebytes=h*w*2*12*128*2
        host_addresses=torch.empty(c.current_frames,dtype=torch.int64,pin_memory=True)
        for frame in range(c.current_frames):host_addresses[frame]=slab.data_ptr()+frame*framebytes
        addresses[start:stop].copy_(host_addresses,non_blocking=True)
        # Retain the small H2D source until clean commit's event fence.
        rows={fid:(archived,rank) for rank,fid in enumerate(pending['current_ids'])}
        desc={fid:bank[:,:,fid].unsqueeze(1) for fid in pending['current_ids']}
        self.host.trace('draftkv_gpu_route_clean_submit',layer=layer,start_frame=start,arena_wait_ms=wait,
            submit_wall_ms=(time.perf_counter_ns()-begin)/1e6,reused_route_preparation='current_preparation' in pending,
            descriptor_source='native_post_rope',payload_bytes=slab.numel()*2)
        return dict(rows=rows,descriptors=desc,raw={},nbytes=slab.numel()*2,registered=True,completion=archived,address_source=host_addresses)

    def commit_clean(self,*,timestep):
        import torch
        if (self.active is None or not self.active['clean'] or timestep is not self.active['timestep']
            or self.done!=self.expected_layers or set(self.pending)!=self.done or any(not p['consumed'] for p in self.pending.values())):
            raise BackboneError('DIRTY_HISTORY','only whole successful clean forwards may publish')
        ids=[p['current_ids'] for p in self.pending.values()]
        if any(row!=ids[0] for row in ids[1:]):raise BackboneError('DIRTY_HISTORY','clean layer ID mismatch')
        # ONE boundary fence, after all layers have overlapped D2H with later compute.
        # Include descriptor/address bank updates in the publication dependency.
        ready=torch.cuda.Event();ready.record();self.copy_stream.wait_event(ready)
        fence=torch.cuda.Event();fence.record(self.copy_stream);begin=time.perf_counter_ns();fence.synchronize()
        wait_ms=(time.perf_counter_ns()-begin)/1e6;self._collect(wait=True)
        for layer,p in self.pending.items():
            row=p['cpu_stage'];self.committed.setdefault(layer,{}).update(row['rows']);self.descriptors.setdefault(layer,{}).update(row['descriptors'])
            self.reserved+=row['nbytes']
            chunk=row['completion']
            # The whole-forward fence already covers every timing event.
            # Do not issue one host synchronization per layer to read metrics.
            self.host.trace('draftkv_gpu_prepare_complete',layer=layer,
                prepare_ms=chunk.prepare_start.elapsed_time(chunk.prepare_end),
                d2h_ms=chunk.copy_start.elapsed_time(chunk.event),
                total_gpu_ms=chunk.prepare_start.elapsed_time(chunk.event),
                d2h_bytes=row['nbytes'],pinned_backend='cudaHostRegister_THP_arena',pinned=True)
        self.next_frame=ids[0][-1]+1;self.commit_generation+=1
        self.host.trace('draftkv_clean_commit',start_frame=self.active['start_frame'],committed_frontier=self.next_frame,
            generation=self.commit_generation,layers=len(self.pending),committed_ids=ids[0],whole_forward_wait_ms=wait_ms,
            pinned_history_bytes=self.reserved,persistent_gpu_draft_descriptor_bytes=sum(b.numel()*2 for b in self.device_banks.values()),
            descriptor_key_source='clean_context.key_native_post_rope',logging_mode=self.logging_mode)
        self.active,self.pending,self.done=None,{},set()
