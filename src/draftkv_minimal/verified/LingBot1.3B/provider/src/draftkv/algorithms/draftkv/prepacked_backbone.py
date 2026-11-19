"""Opt-in LingBot1.3B/SP1 GPU-prepared block cache, separate A/B position modes."""
import hashlib
import json
import time
from draftkv.backbones.contracts import AttentionSelection, BackboneError, FramePage
from draftkv.retrieval.block_topk_p3 import block_probabilities, select_blocks
from draftkv.streaming.prepacked import prepare, layout, BlockRef, execute, native
from .block_backbone import BlockDraftKVHook
from .block_policy import applied_protection
from .position import geometry_for


class GPUPreparedBlockHook(BlockDraftKVHook):
    def __init__(self,host):
        if host.spec.backbone!='LingBot1.3B' or int(host.plan.get('sp',1))!=1:
            raise BackboneError('KV_CONSTRAINT','GPU prepack experiment requires LingBot1.3B/SP1')
        super().__init__(host)
        self.group=int(self.settings.get('prepack_head_group',4))
        self.batch_bytes=int(self.settings.get('prepack_dma_bytes',2**20))
        if self.group not in (3,4,6,12) or self.batch_bytes<1: raise ValueError('invalid prepack configuration')
        self.chrono=self.position_policy=='chrono_compact'
        self.copy_stream=None
        self.execution_cache={}
        self.preparation_cache={}
        self.chrono_placement=self.settings.get("prepack_chrono_placement","consumer")
        if not host.plan.get('gpu_profile', False):
            native()
        # Profiling prepares binaries in another process, retaining this
        # sampler's first runtime module initialization inside generation.

    def _stage_clean_layer(self,layer,pending):
        import torch
        c=pending['context'];h,w=self.geometry[:2]
        if not c.clean: raise BackboneError('DIRTY_HISTORY','only clean KV may be archived')
        if self.copy_stream is None:self.copy_stream=torch.cuda.Stream(device=c.key.device)
        t=time.perf_counter_ns()
        from draftkv.streaming.archive_arena import prepacked_archive_slice
        chunk=prepare(c.key,c.value,c.raw_key,h,w,self.importance.pool_h,self.importance.pool_w,
                      self.group,self.chrono,copy_stream=self.copy_stream,workspace=self.preparation_cache,registered=True,
                      archive_view=prepacked_archive_slice(self,c))
        if not any(owner is chunk.pinned_owner for owner in self._clean_registered_owners):
            self._clean_registered_owners.append(chunk.pinned_owner)
        rows={fid:(chunk,rank) for rank,fid in enumerate(pending['current_ids'])}
        desc={fid:chunk.descriptors[rank].unsqueeze(1).detach() for rank,fid in enumerate(pending['current_ids'])}
        self.host.trace('draftkv_gpu_prepare_submit',layer=layer,start_frame=c.start_frame,
            submit_wall_ms=(time.perf_counter_ns()-t)/1e6,head_group=self.group,
            pool_hw=[self.importance.pool_h,self.importance.pool_w],position_policy=self.position_policy,
            payload='block_group_K_then_V_bf16',descriptor_source='native_post_rope',
            publish='private_until_whole_forward_clean_commit')
        return dict(rows=rows,descriptors=desc,raw={},nbytes=chunk.slab.numel()*2,
                    registered=True,completion=chunk,prepare_layer=layer)

    def route(self,c):
        import torch
        begin=time.perf_counter_ns()
        h,w=self._verify_context(c);ft=c.frame_tokens
        if c.query.shape[2:]!=(12,128) or c.sp!=1 or c.fixed_prefix is not None:
            raise BackboneError('KV_CONSTRAINT','prepack requires native B1 H12 D128/SP1 without prefix')
        self.active['forward_id']=c.forward_id
        bank=self.committed.get(c.layer,{})
        if set(bank)!=set(range(self.next_frame)):raise BackboneError('DIRTY_HISTORY','clean frontier mismatch')
        sink=tuple(range(min(c.sink_frames,c.start_frame)))
        candidates=tuple(f for f in bank if c.sink_frames<=f<c.start_frame)
        current=tuple(range(c.start_frame,c.start_frame+c.current_frames))
        from draftkv.backbones.budgets import bound_budget
        budget=min(bound_budget(self.host,'draftkv').retrieved,len(candidates))
        decision=None
        if candidates:
            prob=block_probabilities(c.query,self._scoring_descriptors(c.layer,candidates,c.query.device),h,w,config=self.importance)
            scores=prob.float().sum(2).sum(1)/(prob.shape[1]*prob.shape[2])
            decision=select_blocks(scores[0].detach().cpu().tolist(),candidates,budget,
                query_start=c.start_frame,grid_h=h,grid_w=w,
                protect_recent=applied_protection(self.host.spec.backbone,budget),
                pool_h=self.importance.pool_h,pool_w=self.importance.pool_w)
        selected=decision.token_indices if decision else ()
        history=tuple((fid,tuple(range(ft))) for fid in sink)+selected
        source=[fid*ft+t for fid,ts in history for t in ts]
        if source!=sorted(set(source)):raise BackboneError('KV_CONSTRAINT','source chronology mismatch')
        time_by_source={token:rank//ft for rank,token in enumerate(source)}
        blocks,inverse,_=layout(h,w,self.importance.pool_h,self.importance.pool_w,self.group)
        refs=[];pages=[];position=[];cursor=0
        for fid,tokens in history:
            chunk,frame=bank[fid]
            byblock={}
            for token in tokens:
                b,j=inverse[token];byblock.setdefault(b,[]).append((j,token))
            frame_tokens=[]
            for block,rows in sorted(byblock.items()):
                indices=tuple(j for j,_ in rows)
                slots=tuple(time_by_source[fid*ft+token] if self.chrono else fid for _,token in rows)
                refs.append(BlockRef(chunk,frame,block,indices,slots))
                frame_tokens.extend(token for _,token in rows)
                position.extend((slot,token) for slot,(_,token) in zip(slots,rows))
            pages.append(FramePage(fid,len(pages),cursor,cursor+len(tokens),tuple(frame_tokens)))
            cursor+=len(tokens)
        # Current KV stays on GPU, prepared into the same consumer layout.
        current_chunk=prepare(c.key,c.value,c.raw_key,h,w,self.importance.pool_h,self.importance.pool_w,
                              self.group,self.chrono,archive=False,workspace=self.preparation_cache)
        for frame,fid in enumerate(current):
            slot=len(sink)+budget+frame if self.chrono else fid
            physical=[]
            for block,tokens in enumerate(blocks):
                refs.append(BlockRef(current_chunk,frame,block,tuple(range(len(tokens))),tuple(slot for _ in tokens)))
                physical.extend(tokens);position.extend((slot,token) for token in tokens)
            pages.append(FramePage(fid,len(pages),cursor,cursor+ft,tuple(physical)));cursor+=ft
        geo=geometry_for(c,self.host.spec.family)
        phase=torch.view_as_real(geo.coefficients[0]).flatten(-2).contiguous()
        q=c.query
        if self.chrono:
            q=q.clone()
            for frame,fid in enumerate(current):
                slot=len(sink)+budget+frame
                lo,hi=frame*ft,(frame+1)*ft
                if slot!=fid:geo.assign(q[:,lo:hi],geo.rotate(geo.extract(c.raw_query[:,lo:hi]),slot,q.dtype))
        physical_sources=[fid*ft+token for page in pages for fid in (page.frame_id,) for token in page.source_token_ids]
        sha=hashlib.sha256(json.dumps(physical_sources,separators=(',',':')).encode()).hexdigest()
        pending=dict(context=c,current_ids=current,consumed=False,refs=refs,phase=phase,
            selected=tuple(f for f,_ in selected),source_sha=sha,route_wall_ms=(time.perf_counter_ns()-begin)/1e6,
            current_preparation=current_chunk)
        self.pending[c.layer]=pending
        self.host.trace('draftkv_gpu_prepack_route',layer=c.layer,start_frame=c.start_frame,
            clean=c.clean,route_wall_ms=pending['route_wall_ms'],source_token_map_sha256=sha,
            scoring_qk_space='native_post_rope',position_policy=self.position_policy,
            logical_time_rule='rank_in_sorted_selected_source_tokens//frame_tokens',
            pool_hw=[self.importance.pool_h,self.importance.pool_w],head_group=self.group,
            history_tokens=budget*ft,sink_tokens=len(sink)*ft,current_tokens=len(current)*ft,
            consumed_tokens=cursor,selected_block_count=len(refs),cpu_kv_gather=False)
        # ABI shape/provenance views only; the consumer uses refs and never reads
        # these placeholders. Avoid allocating a second full GPU KV bank.
        k=c.key[:,:1].expand(1,cursor,12,128);v=c.value[:,:1].expand_as(k)
        return AttentionSelection(q.contiguous(),k,v,sink,decision.protected if decision else (),
            decision.remote_ids if decision else (),current,tuple(pages),
            dict(representation='paged',position_map=tuple(position),draftkv_plan=pending,
                 gpu_prepacked=True,kv_materialized=False))

    def consume(self,c,selection,native_fa2):
        pending=self.pending[c.layer]
        if selection.metadata.get('draftkv_plan') is not pending:raise BackboneError('DIRTY_HISTORY','owner mismatch')
        begin=time.perf_counter_ns()
        # First native call has already compiled all template specializations;
        # no synchronizing discarded attention launch is needed per layer.
        out,metrics=execute(selection.query,pending['refs'],pending['phase'],batch_bytes=self.batch_bytes,warmup=False,cache=self.execution_cache,chrono_placement=self.chrono_placement)
        pending['consumed']=True
        metrics={k:v for k,v in metrics.items() if not k.startswith('_')}
        self.host.trace('draftkv_gpu_prepack_attention',layer=c.layer,start_frame=c.start_frame,
            clean=c.clean,consume_wall_ms=(time.perf_counter_ns()-begin)/1e6,
            route_wall_ms=pending['route_wall_ms'],source_token_map_sha256=pending['source_sha'],
            position_policy=self.position_policy,head_group=self.group,batch_bytes=self.batch_bytes,
            current_prepare_ms=pending['current_preparation'].times()['prepare_ms'],**metrics)
        return out
