"""Ablation adapters for production GPU routing / prepared pinned archive.

Only small GPU-built DMA descriptions cross to CPU. Historical payloads remain
views of the original archive; no CPU gather, transpose or second packing slab.
The native head-group consumer and its FA3 TMA/warp-specialized kernel are reused.
"""
from dataclasses import replace
import time

from draftkv.algorithms.draftkv.device_backbone import DeviceRouteBlockHook
from draftkv.algorithms.draftkv.block_policy import applied_protection
from draftkv.streaming.headgroup import HeadGroupPlan
from .config import AblationError
from .instrumentation import Instrumentation
from .scheduler import Page, Route, make_scheduler
from .transport import TorchPageTransport, Ticket


class GPUArchive:
    def __init__(self, layers, frames, ft, heads, device):
        import torch
        self.storage = torch.empty((layers, frames, ft*heads*256), device=device, dtype=torch.bfloat16)
        self.bytes = self.storage.numel()*self.storage.element_size()
        self.touch_ms = self.register_ms = 0  # No host touch/register operation exists.


class DenseDescriptorTransport(TorchPageTransport):
    def submit(self, page, label, *, speculative=False):
        import torch
        from contextlib import nullcontext
        source, destination = page.payload
        with torch.cuda.stream(self.stream) if self.stream is not None else nullcontext():
            with self.ranges.range(label, stream=self.stream):
                target = torch.empty_like(source, device=self.device) if speculative else destination
                target.copy_(source, non_blocking=self.stream is not None)
                event = torch.cuda.Event() if self.stream is not None else None
                if event is not None:
                    event.record(self.stream)
        self.reap()
        return Ticket(source, (target,), event)


def descriptor_pages(table, archive, dense, *, layer, generation, branch):
    """Decode bounded DMA rows to zero-copy source/destination byte views.

    Pure tensor/view code is CPU-testable. Reject bad pointers/ranges instead of
    constructing arbitrary memory views. Preserve GPU-produced descriptor order.
    """
    import torch
    source = archive.view(torch.uint8).reshape(-1)
    target = dense.view(torch.uint8).reshape(-1)
    base, dst = source.data_ptr(), target.data_ptr()
    count = int(table[0])
    if count < 0 or 8+5*count > table.numel():
        raise AblationError('invalid GPU-generated descriptor count')
    pages, targets = [], []
    for src, dest, n, unit, publish in table[8:8+5*count].reshape(-1,5).tolist():
        if not n:
            continue
        a, b = src-base, dest-dst
        if n < 0 or a < 0 or a+n > source.numel() or b < 0 or b+n > target.numel():
            raise AblationError('DMA description outside immutable archive/receive allocation')
        # Raw prepared page has not yet been temporally rotated. Reuse across
        # compact slots is safe: rotate once AFTER placement into the actual map.
        identity = (branch, generation, layer, a, n)
        pages.append(Page(identity,n,(source[a:a+n],target[b:b+n])))
        targets.append(target[b:b+n])
    return Route(layer,tuple(pages)), targets


class ScheduledHeadGroupPlan(HeadGroupPlan):
    def route(self, *args, **kwargs):
        # Production route and group descriptor generation stay on GPU. Suppress
        # ONLY submission; scheduler controls WHEN these immutable spans move.
        kwargs['submit'] = False
        return super().route(*args, **kwargs)


class DeviceMechanismAblationHook(DeviceRouteBlockHook):
    def __init__(self, host, config):
        self.ablation = config
        self.ranges = Instrumentation(nvtx=bool(host.plan.get('ablation_nvtx', False)))
        self.scheduler = self.transport = None
        self.fullkeys = {}
        self.native_temporal = {}
        if not host.plan['algorithm_plugin']['config'].get('headgroup_pipeline'):
            raise AblationError('device mechanism adapter requires production headgroup_pipeline; do not silently replace the consumer')
        if host.spec.family == 'solarwm':
            raise AblationError('SolarWM fixed-prefix device headgroup is not supported by production; use canonical block provider')
        super().__init__(host)
        self.position_policy = config['retrieval']['position_mode']
        self.chrono = self.position_policy == 'chrono_compact'
        self.scheduled = config['attention']['prefetch_schedule'] != 'production' or config['history']['residency'] == 'gpu'

    def _store_draft_bank(self):
        return self.ablation['history']['searchable_index'] == 'draft_bank'

    def _new_archive(self, c):
        if self.ablation['history']['residency'] == 'gpu':
            return GPUArchive(self.host.spec.layers,self.capacity,c.frame_tokens,self.heads,c.key.device)
        return super()._new_archive(c)

    def _new_group_plan(self, c, config):
        if self.ablation['attention']['prefetch_schedule'] != 'production' or self.ablation['history']['residency'] == 'gpu':
            h,w=self.geometry[:2]
            return ScheduledHeadGroupPlan(h,w,self.capacity,c.key.device,self.heads,self.host.spec.family,config)
        return super()._new_group_plan(c,config)

    def _native_full_keys(self, c, sink):
        import torch
        from draftkv.algorithms.draftkv.position import geometry_for
        # Full-key search explicitly pays host→GPU key fetch, if host-resident.
        # Native post-RoPE spatial/camera components are archived unchanged.
        geo = geometry_for(c,self.host.spec.family)
        p=self.plan_device;frames=[];g=self.group
        for f in range(sink,c.start_frame):
            slab=self.arena.storage[c.layer,f]
            key=torch.empty((p.ft,self.heads,128),device=c.key.device,dtype=c.key.dtype)
            for b,tokens in enumerate(p.blocks):
                count=len(tokens);begin=p.starts[b]*self.heads*256
                for group in range(self.heads//g):
                    lo=begin+group*2*count*g*128
                    chunk=slab[lo:lo+count*g*128].reshape(count,g,128).to(c.key.device,non_blocking=True)
                    key[list(tokens),group*g:(group+1)*g]=chunk
            if self.chrono:
                saved = self.native_temporal.get(c.layer, {}).get(f)
                if saved is None:
                    raise AblationError('exact native scoring temporal K is missing; reconstruction/inverse fallback forbidden')
                geo.assign(key,saved.to(c.key.device,non_blocking=True))
            frames.append(key)
        return torch.cat(frames,0).unsqueeze(0)

    def _selection_parameters(self, c, h, w):
        import torch
        sink=min(c.sink_frames,c.start_frame)
        available=c.start_frame-sink
        budget=min(self._bound_budget().retrieved,available)
        mode=self.ablation['retrieval']['selector']
        protect=applied_protection(self.host.spec.backbone,budget) if budget else 0
        if mode=='all_history':
            horizon=self.ablation['history']['max_frames']
            budget=available if horizon is None else min(available,horizon)
            protect=budget
        elif mode=='local_only':
            budget=budget if self.ablation['history']['local_window_mode']=='capacity_fair' else protect
            protect=budget
        self.plan_device._ablation_skip_topk = mode in ('all_history','local_only')
        if mode in ('all_history','local_only') or not available:
            score=torch.zeros(available*self.plan_device.r,device=c.query.device,dtype=torch.float32)
        else:
            with self.ranges.range('draft_score'):
                if mode=='draft':
                    from draftkv.streaming.device_route import scores
                    score=scores(c.query,self.descriptor_banks[c.layer],h,w,32,32,sink,c.start_frame)
                else:
                    from .oracle import ExactRegionRelevanceOracle
                    keys=self.fullkeys[c.layer][sink:c.start_frame].flatten(0,1).unsqueeze(0) if self.ablation['history']['fullkey_residency']=='gpu' else self._native_full_keys(c,sink)
                    prob=ExactRegionRelevanceOracle().score(c.query,keys,h,w,config=self.importance)
                    score=(prob.float().sum(2).sum(1)/(prob.shape[1]*prob.shape[2])).flatten()
            if self.ablation['retrieval']['granularity']=='frame':
                mass=score.reshape(available,-1).sum(-1,keepdim=True)
                score=mass.expand(available,self.plan_device.r).reshape(-1).contiguous()
        return sink,budget,protect,score

    def begin_forward(self, **kwargs):
        super().begin_forward(**kwargs)
        if self.scheduler:
            self.scheduler.on_denoise_step_start(self.denoise_step,epoch=(self.branch,self.next_frame,self.commit_generation))

    def route(self,c):
        from .planning import check_memory
        check_memory(self.ablation,self.host.spec,history_frames=self.capacity,frame_tokens=c.frame_tokens)
        with self.ranges.range('route'):
            selection=super().route(c)
        from .reporting import record_route
        record_route(self.host,self.ablation,c,selection)
        if self.scheduled:
            p=self.plan_device
            if self.scheduler is None:
                self.transport=DenseDescriptorTransport(c.query.device,self.ranges)
                schedule=self.ablation['attention']['prefetch_schedule']
                self.scheduler=make_scheduler('serial' if schedule=='production' else schedule,self.transport,layers=self.host.spec.layers)
                self.scheduler.on_denoise_step_start(self.denoise_step,epoch=(self.branch,self.next_frame,self.commit_generation))
            self.scheduler.on_layer_start(c.layer)
            # Bounded descriptor D2H is part of route timing, never preparation
            # outside measurement. No per-token payload/source-map transfer.
            table=p.table.cpu()
            route,targets=descriptor_pages(table,self.arena.storage[c.layer],p.dense,
                layer=c.layer,generation=self.commit_generation,branch=self.branch)
            self.scheduler.on_route_ready(route)
            self.pending[c.layer]['ablation_targets']=targets
        return selection

    def validation_context(self,c):
        if self.ablation['retrieval']['selector']=='all_history':
            return replace(c,window_frames=max(c.window_frames,self.plan_device.total//c.frame_tokens))
        return c

    def consume(self,c,selection,native_fa2):
        if not self.scheduled:
            return super().consume(c,selection,native_fa2)
        import torch
        from draftkv.streaming.headgroup import _kernels
        pending=self.pending[c.layer];p=self.plan_device
        ready=self.scheduler.on_attention_start()
        with self.ranges.range('gpu_gather'):
            for (_,ticket),target in zip(ready,pending['ablation_targets']):
                if target.data_ptr()!=ticket.pair[0].data_ptr():
                    target.copy_(ticket.pair[0],non_blocking=True)
        # Data readiness now includes correction. Chrono is applied exactly once
        # to actual selected slots BEFORE native execute can enqueue TMA loads.
        if self.chrono:
            with self.ranges.range('rope',stream=torch.cuda.current_stream(c.query.device)):
                rotate,_=_kernels()
                for g,s,lo,hi in p._units():
                    rotate[((hi-lo+31)//32,self.group)](p.dense[g,0],p.slots,self.phase,
                        lo,hi-lo,self.group,False,enable_fp_fusion=False)
        with self.ranges.range('attention',stream=torch.cuda.current_stream(c.query.device)):
            out=p.execute(selection.query,self.phase,pending['route_events'],resident=True)
        pending['consumed']=True
        self.scheduler.on_attention_end()
        self.host.trace('ablation_prefetch',layer=c.layer,denoise_step=self.denoise_step,
            archive_residency=self.ablation['history']['residency'],**self.scheduler.metrics())
        return out

    def _prepare(self,c):
        # FullKV/M2 have no compact bank: use the existing pack-only current
        # kernel for clean KV as well; do not calculate unused descriptors.
        return super()._prepare(replace(c,clean=False) if c.clean and not self._store_draft_bank() else c)

    def _stage_clean_layer(self,layer,pending):
        row=super()._stage_clean_layer(layer,pending)
        if self.ablation['history']['searchable_index']=='full_keys' and self.ablation['history']['fullkey_residency']=='gpu':
            import torch
            c=pending['context']
            staged=self.fullkeys.get(layer)
            if staged is None:
                staged=torch.empty((self.capacity,c.frame_tokens,self.heads,128),device=c.key.device,dtype=c.key.dtype)
                self.fullkeys[layer]=staged
            staged[c.start_frame:c.start_frame+c.current_frames].copy_(c.key.reshape(c.current_frames,c.frame_tokens,self.heads,128))
        if self.ablation['retrieval']['selector']=='exact_region_relevance' and self.chrono and self.ablation['history']['fullkey_residency']=='host_fetch':
            import torch
            from draftkv.algorithms.draftkv.position import geometry_for
            c=pending['context'];geo=geometry_for(c,self.host.spec.family)
            bank=self.native_temporal.setdefault(layer,{})
            with torch.cuda.stream(self.copy_stream):
                for i,fid in enumerate(pending['current_ids']):
                    native=geo.extract(c.key[0,i*c.frame_tokens:(i+1)*c.frame_tokens])
                    target=torch.empty(native.shape,device='cpu',dtype=native.dtype,pin_memory=True)
                    target.copy_(native,non_blocking=True)
                    native.record_stream(self.copy_stream)
                    bank[fid]=target
                row['event']=torch.cuda.Event(enable_timing=True)
                row['event'].record(self.copy_stream)
        elif self.ablation['history']['fullkey_residency']=='gpu':
            # Include native search-key GPU readiness in the same whole-forward
            # publication fence as the prepared archive's D2H/D2D completion.
            import torch
            ready=torch.cuda.Event();ready.record()
            self.copy_stream.wait_event(ready)
            row['event']=torch.cuda.Event(enable_timing=True);row['event'].record(self.copy_stream)
        return row

    def _clean_transfer_metrics(self, row):
        if self.ablation['history']['residency'] == 'gpu':
            return dict(d2d_ms=row['copy_start'].elapsed_time(row['event']),d2h_ms=0)
        if self.ablation['retrieval']['selector']=='exact_region_relevance':
            return dict(archive_oracle_copy_envelope_ms=row['copy_start'].elapsed_time(row['event']),d2h_ms=None)
        return super()._clean_transfer_metrics(row)

    def _archive_storage_metrics(self):
        if self.ablation['history']['residency'] == 'gpu':
            return dict(pinned_history_bytes=0,gpu_history_bytes=self.reserved)
        fields=super()._archive_storage_metrics()
        fields['oracle_native_temporal_host_bytes']=sum(t.numel()*t.element_size() for bank in self.native_temporal.values() for t in bank.values())
        fields['fullkey_search_gpu_bytes']=sum(t.numel()*t.element_size() for t in self.fullkeys.values())
        return fields

    def commit_clean(self,*,timestep):
        super().commit_clean(timestep=timestep)
        if self.scheduler:
            self.scheduler.drain();self.scheduler.previous.clear();self.transport.close()
            self.host.trace('ablation_prefetch_summary', **self.scheduler.metrics())
        from .instrumentation import report_envelopes
        report_envelopes(self.host,self.ranges)

    def rollback(self):
        if self.scheduler:self.scheduler.drain()
        if self.transport:self.transport.close()
        super().rollback()
