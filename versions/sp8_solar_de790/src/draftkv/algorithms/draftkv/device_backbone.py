"""Spatial 8x16/full-head-group route: device selection, DMA, async clean archive."""
import time
from .block_backbone import BlockDraftKVHook
from .block_policy import applied_protection
from draftkv.backbones.contracts import AttentionSelection,BackboneError,FramePage
from draftkv.streaming.device_route import native,DevicePlan,scores,geometry,_kernel


class DeviceRouteBlockHook(BlockDraftKVHook):
    def _diagnostic_stage(self,c,stage):
        # Explicit opt-in diagnostic only; not used in production profiles.
        import os
        if self.h3 and os.environ.get('DRAFTKV_H3_STAGE_DIAG') == '1':
            import sys
            print(f'H3_STAGE layer={c.layer} start={c.start_frame} fw={c.forward_id} '
                  f'clean={c.clean} stage={stage} monotonic_ns={time.perf_counter_ns()}',
                  file=sys.stderr,flush=True)

    def __init__(self,host):
        import torch
        super().__init__(host)
        self.heads=host.spec.heads
        self.grouped=bool(self.settings.get('headgroup_pipeline',False))
        self.group=int(self.settings.get('prepack_head_group',self.heads))
        self.h3=host.spec.family=='solarwm'
        if (host.spec.head_dim!=128 or int(host.plan.get('sp',1))!=1 or
            (self.importance.pool_h,self.importance.pool_w)!=(8,16) or
            (self.group!=self.heads and not self.grouped) or self.group<1 or self.heads%self.group or self.importance.maxpool_weight!=.04):
            raise ValueError('device route requires D128/SP1/8x16/full head group/.04')
        if self.settings.get('prepack_chrono_placement','fetch')!='fetch':raise ValueError('B requires fetch rotation')
        self.capacity=int(self.settings.get('archive_capacity_frames',host.plan.get('history_plan',{}).get('capacity_frames',92)))
        self.batch_bytes=int(self.settings.get('prepack_dma_bytes',4194304))
        self.chrono=self.position_policy=='chrono_compact'
        if self.h3 and not self.chrono:
            raise ValueError('H3 device route currently supports Chrono-Compact B only')
        self.arena=None;self.plan_device=None;self.descriptor_banks={};self.addresses={}
        self.preparation_slots=[];self.slot_index=0;self.copy_stream=None
        self.metric_rows=[];self.clean_rows=[]
        self.resolved_budget=None
        if not host.plan.get('gpu_profile', False):
            native()
        # Profile builds happen in a disposable preparation process. Retain
        # this sampler's real first module load inside measured generation.

    def _bound_budget(self):
        # This owner's frozen plan binds the effective budget. Resolve and
        # validate once on the first real route, inside generation. On
        # NFS, reopening the same JSON twice per attention was ~1.5 ms/call.
        if self.resolved_budget is None:
            from draftkv.backbones.budgets import bound_budget
            self.resolved_budget=bound_budget(self.host,'draftkv')
        return self.resolved_budget

    def _initialize(self,c):
        import torch
        from .registered_pages import RegisteredArchiveArena
        from draftkv.streaming.prepacked import layout
        if self.arena is not None:return
        h,w=self.geometry[:2];ft=h*w
        self.arena=self._new_archive(c)
        self._clean_registered_owners.append(self.arena)
        if self.grouped:
            from draftkv.streaming.headgroup import HeadGroupPlan,PipelineConfig
            config=PipelineConfig(self.group,int(self.settings.get('pipeline_splits',1)),
                int(self.settings.get('native_fa_splits',1)),self.settings.get('pipeline_schedule','pipeline'),
                bool(self.settings.get('pipeline_reverse',True)),
                submission=self.settings.get('pipeline_submission','wave'),
                native_attention_version=int(self.host.plan.get('native_attention_version',3)))
            self.plan_device=self._new_group_plan(c,config)
        else:self.plan_device=DevicePlan(h,w,self.capacity,c.key.device,self.heads,self.host.spec.family,8,16)
        self.copy_stream=torch.cuda.Stream(device=c.key.device)
        r=self.plan_device.r
        for layer in range(self.host.spec.layers):
            base=self.arena.storage[layer].data_ptr()
            self.addresses[layer]=torch.tensor([base+(f*ft+s)*self.heads*512 for f in range(self.capacity)
                for s in self.plan_device.starts[:-1]],device=c.key.device,dtype=torch.int64)
            if self._store_draft_bank():
                self.descriptor_banks[layer]=torch.empty((2,1,self.heads,self.capacity*r,128),device=c.key.device,dtype=torch.float32)
        from .position import geometry_for
        geo=geometry_for(c,self.host.spec.family)
        self.phase=(torch.cat(geo.coefficients,-1).to(torch.bfloat16).float().contiguous() if self.h3 else
                    torch.view_as_real(geo.coefficients[0]).flatten(-2).contiguous()).to(c.key.device)
        if self.chrono:
            from draftkv.streaming.prepacked import _rotate_fetched
            from draftkv.streaming.device_route import rotate_query
            if self.h3:
                from draftkv.streaming.temporal_kernels import rotate_h3_fetched as _rotate_fetched, rotate_h3_query as rotate_query
            t=time.perf_counter_ns()
            # Pure module preload: no discarded attention/rotation execution.
            # Loading a new Triton module after cuStreamWaitValue32 can deadlock
            # CUDA's context-wide module synchronization with producer enqueue.
            dummy=torch.empty(1,device=c.key.device,dtype=torch.bfloat16)
            offset=torch.empty(1,device=c.key.device,dtype=torch.int64)
            slot=torch.empty(1,device=c.key.device,dtype=torch.int32)
            totals=sorted({(min(s,c.sink_frames)+min(max(0,s-c.sink_frames),self._bound_budget().retrieved)+c.current_frames)*ft
                           for s in range(0,self.capacity,c.current_frames)})
            for total in totals:
                if self.h3:_rotate_fetched(dummy,offset,slot,self.phase,total,self.heads,preload=True)
                else:_rotate_fetched(dummy,offset,slot,self.phase,total,self.heads,preload=True,heads=self.heads)
            rotate_query(c.query,c.raw_query,self.phase,ft,c.sink_frames+self._bound_budget().retrieved,preload=True)
            self.host.trace('draftkv_rotation_preload',totals=totals,host_ms=(time.perf_counter_ns()-t)/1e6,
                timing_scope='inside_generation_before_first_async_producer',executed_kernels=0)
        self.host.trace('draftkv_archive_arena',bytes=self.arena.bytes,capacity_frames=self.capacity,
            layers=self.host.spec.layers,touch_ms=self.arena.touch_ms,register_ms=self.arena.register_ms,
            timing_scope='inside_generation_first_route',logging_mode='bounded_summary')

    def _prepare(self,c):
        import torch
        h,w=self.geometry[:2];self._initialize(c)
        # Two slots have explicit GPU event ownership; no host wait for reuse.
        if len(self.preparation_slots)<2:
            self.preparation_slots.append({})
        slot=self.preparation_slots[self.slot_index%len(self.preparation_slots)];self.slot_index+=1
        if slot.get('release') is not None:torch.cuda.current_stream(c.key.device).wait_event(slot['release'])
        n=c.key.numel()*2;f=c.current_frames;r=self.plan_device.r
        if slot.get('packed') is None or slot['packed'].numel()<n:
            slot['packed']=torch.empty(n,device=c.key.device,dtype=torch.bfloat16)
        if slot.get('desc') is None or slot['desc'].shape[0]!=f:
            slot['desc']=torch.empty((f,2,self.heads,r,128),device=c.key.device,dtype=torch.float16)
        slot['view']=slot['packed'][:n];index,counts,starts,_=geometry(h,w,8,16,str(c.key.device))
        a,b=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
        a.record()
        raw=c.raw_key if self.chrono else c.key
        if self.grouped and not c.clean:
            from draftkv.streaming.headgroup import pack_current
            pack_current(c.key,c.value,raw,slot['view'],h,w,self.group,self.chrono,h3=self.h3)
        else:
            _kernel()[(f,r,self.heads)](c.key,c.value,raw,slot['view'],slot['desc'],index,counts,starts,
                f,h*w,self.heads,self.group,r,128,self.chrono,c.key.stride(1),c.key.stride(2),c.value.stride(1),c.value.stride(2),
                raw.stride(1),raw.stride(2),self.h3,num_warps=8)
        b.record();slot['prepared']=(a,b);slot['release']=b
        return slot

    def _new_archive(self, c):
        from .registered_pages import RegisteredArchiveArena
        return RegisteredArchiveArena(self.host.spec.layers,self.capacity,c.frame_tokens,self.heads)

    def _new_group_plan(self, c, config):
        from draftkv.streaming.headgroup import HeadGroupPlan
        h,w=self.geometry[:2]
        return HeadGroupPlan(h,w,self.capacity,c.key.device,self.heads,self.host.spec.family,config,8,16)

    def _store_draft_bank(self):
        return True

    def _selection_parameters(self, c, h, w):
        import torch
        sink=min(c.sink_frames,c.start_frame);budget=min(self._bound_budget().retrieved,c.start_frame-sink)
        protect=applied_protection(self.host.spec.backbone,budget) if budget else 0
        score=(scores(c.query,self.descriptor_banks[c.layer],h,w,8,16,sink,c.start_frame)
               if c.start_frame>sink else torch.empty(0,device=c.query.device,dtype=torch.float32))
        return sink,budget,protect,score

    def route(self,c):
        import torch
        begin=time.perf_counter_ns();h,w=self._verify_context(c);ft=c.frame_tokens
        if c.query.shape[2:]!=(self.heads,128) or c.sp!=1 or c.start_frame+c.current_frames>self.capacity:
            raise BackboneError('KV_CONSTRAINT','device route fixed scope/capacity exceeded')
        self.active['forward_id']=c.forward_id
        if self.next_frame!=c.start_frame:raise BackboneError('DIRTY_HISTORY','device clean frontier mismatch')
        self._diagnostic_stage(c,'before_prepare')
        slot=self._prepare(c)
        self._diagnostic_stage(c,'after_prepare')
        sink,budget,protect,score=self._selection_parameters(c,h,w)
        self._diagnostic_stage(c,'after_selection')
        p=self.plan_device
        prefix=getattr(c.native_geometry,'chrono_compact_prefix',None) if self.h3 else None
        if c.fixed_prefix is not None and prefix is None:
            raise BackboneError('KV_CONSTRAINT','missing native unshifted condition/audio prefix')
        self._diagnostic_stage(c,'before_group_route')
        route_events=p.route(score,slot['view'],self.addresses[c.layer],sink,c.start_frame,c.current_frames,
            budget,protect,self.chrono,self.batch_bytes,prefix_tokens=0 if prefix is None else prefix[0].shape[1])
        self._diagnostic_stage(c,'after_group_route')
        # Phase table is static for the scope, and remains on GPU.
        q=c.query
        if self.chrono and sink+budget!=c.start_frame:
            from draftkv.streaming.device_route import rotate_query
            if self.h3:
                from draftkv.streaming.temporal_kernels import rotate_h3_query as rotate_query
            q=rotate_query(c.query,c.raw_query,self.phase,ft,sink+budget)
        current=tuple(range(c.start_frame,c.start_frame+c.current_frames))
        pending=dict(context=c,current_ids=current,consumed=False,slot=slot,route_events=route_events,fixed_prefix=prefix,
            route_wall_ms=(time.perf_counter_ns()-begin)/1e6)
        self.pending[c.layer]=pending
        total=(sink+budget+c.current_frames)*ft
        k=c.key[:,:1].expand(1,total,self.heads,128);v=c.value[:,:1].expand_as(k)
        return AttentionSelection(q,k,v,tuple(range(sink)),tuple(range(c.start_frame-protect,c.start_frame)),(),current,(),
            dict(representation='device_blocks',draftkv_plan=pending,history_budget=budget,
                 device_offsets=p.offset,device_slots=p.slots,device_takes=p.take,
                 archive_frontier=self.next_frame,remote_ids_location='device',logging_mode='bounded_summary'))

    def consume(self,c,selection,native_fa2):
        p=self.pending[c.layer]
        if selection.metadata.get('draftkv_plan') is not p:raise BackboneError('DIRTY_HISTORY','device plan ownership')
        begin=time.perf_counter_ns()
        self._diagnostic_stage(c,'before_group_execute')
        out=self.plan_device.execute(selection.query,self.phase,p['route_events'],prefix=p['fixed_prefix']);p['consumed']=True
        self._diagnostic_stage(c,'after_group_execute')
        self.metric_rows.append(dict(layer=c.layer,start_frame=c.start_frame,clean=c.clean,
            route_wall_ms=p['route_wall_ms'],consume_submit_ms=(time.perf_counter_ns()-begin)/1e6))
        return out

    def _stage_clean_layer(self,layer,pending):
        import torch
        t=time.perf_counter_ns();c=pending['context']
        if not c.clean:raise BackboneError('DIRTY_HISTORY','no noisy archive')
        slot=pending.get('slot') or self._prepare(c)
        start=c.start_frame;stop=start+c.current_frames;r=self.plan_device.r
        # FP16 descriptors are converted exactly once, in place in persistent bank.
        desc=slot['desc'].permute(1,2,0,3,4).reshape(2,1,self.heads,c.current_frames*r,128)
        if self._store_draft_bank():
            self.descriptor_banks[layer][:,:,:,start*r:stop*r].copy_(desc)
        self.copy_stream.wait_event(slot['prepared'][1])
        a,b=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
        with torch.cuda.stream(self.copy_stream):
            a.record();self.arena.storage[layer,start:stop].view(-1).copy_(slot['view'],non_blocking=True);b.record()
        slot['release']=b
        row=dict(ids=pending['current_ids'],event=b,copy_start=a,prepare=slot['prepared'],
                 nbytes=c.current_frames*c.frame_tokens*self.heads*512,submit_ms=(time.perf_counter_ns()-t)/1e6,layer=layer)
        return row

    def _clean_transfer_metrics(self, row):
        return dict(d2h_ms=row['copy_start'].elapsed_time(row['event']))

    def _archive_storage_metrics(self):
        return dict(pinned_history_bytes=self.reserved)

    def commit_clean(self,*,timestep):
        if (self.active is None or not self.active['clean'] or timestep is not self.active['timestep'] or
            self.done!=self.expected_layers or set(self.pending)!=self.done or
            any(not p['consumed'] or 'cpu_stage' not in p for p in self.pending.values())):
            raise BackboneError('DIRTY_HISTORY','whole-model device clean commit incomplete')
        rows=[p['cpu_stage'] for p in self.pending.values()]
        if any(x['ids']!=rows[0]['ids'] for x in rows):raise BackboneError('DIRTY_HISTORY','clean ID mismatch')
        # All layer D2H writes share one stream; one final fence publishes all.
        rows[-1]['event'].synchronize()
        dma,gpu=self.plan_device.collect()
        for metric,device,d in zip(self.metric_rows,gpu,dma):
            self.host.trace('draftkv_device_attention',**metric,**device,dma_count=int(d[0]),h2d_bytes=int(d[1]),
                producer_wait_ms=d[2],producer_submit_ms=d[3],payload_dma_ms=d[4],
                source_identity=native().source_identity,position_policy=self.position_policy,logging_mode='bounded_summary')
        self.metric_rows=[]
        for x in rows:
            self.host.trace('draftkv_device_clean',layer=x['layer'],submit_ms=x['submit_ms'],bytes=x['nbytes'],
                prepare_ms=x['prepare'][0].elapsed_time(x['prepare'][1]),**self._clean_transfer_metrics(x))
        self.next_frame=rows[0]['ids'][-1]+1;self.reserved+=sum(x['nbytes'] for x in rows);self.commit_generation+=1
        self.host.trace('draftkv_clean_commit',committed_frontier=self.next_frame,generation=self.commit_generation,
            layers=len(rows),**self._archive_storage_metrics(),arena_bytes=self.arena.bytes,
            persistent_gpu_draft_descriptor_bytes=sum(x.numel()*x.element_size() for x in self.descriptor_banks.values()),
            position_policy=self.position_policy,logging_mode='bounded_summary')
        self.pending={};self.active=None;self.done=set()

    def history_ids(self,layer):return tuple(range(self.next_frame))
    def contains_history_ids(self,layer,ids):return all(0<=i<self.next_frame for i in ids)

    def frame_pages(self,layer,frame_id):
        if layer not in self.expected_layers or not 0<=frame_id<self.next_frame:
            raise BackboneError('KV_CONSTRAINT','frame is not clean committed in this device archive')
        return (FramePage(frame_id,frame_id,0,self.geometry[3]),)
