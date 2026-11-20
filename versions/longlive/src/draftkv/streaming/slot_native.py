"""Bounded native-attention receive slots, with consumer-to-DMA backpressure.

Slot recycling is guarded by per-unit completion words, never a host estimate
of kernel duration. Selection, source order, temporal positions and native FA
math are shared with the dense path. Only the receive storage is bounded.
"""
from dataclasses import replace
import time

_REBASE = None


def slot_geometry(total, splits, slots, capacity):
    if any(type(v) is not int or v<1 for v in (total,splits,slots,capacity)) or slots>8:
        raise ValueError('invalid bounded slot geometry')
    splits=min(total,max(splits,(total+capacity-1)//capacity))
    return splits,min(slots,splits),capacity


def _preload_units(rotation, payload, units, start, history_tokens):
    """Preload each identical token interval once within this invocation.

    Every head group previously used this same slot-zero payload and rotation
    metadata. This removes duplicate warmup lookups, without caching across
    calls or omitting any first-use specialization before future waits.
    """
    if rotation is None:return
    seen=set()
    for unit in units:
        lo,hi=max(0,unit.begin-start),min(history_tokens,unit.end-start)
        if hi>lo and (lo,hi) not in seen:
            rotation.rotate_tokens_(payload,lo,hi,preload=True)
            seen.add((lo,hi))


def _rotation_offsets(plan, rotation, total, start, splits, group):
    global _REBASE,tl
    if rotation is None:return None
    import triton
    import triton.language as tl
    if _REBASE is None:
        @triton.jit
        def rebase(In,Out,N:tl.constexpr,T:tl.constexpr,A:tl.constexpr,S:tl.constexpr,G:tl.constexpr,B:tl.constexpr):
            i=tl.program_id(0)*B+tl.arange(0,B)
            off=tl.load(In+i,i<N,0)
            pos=off//(G*128)+A
            unit=((pos+1)*S-1)//T
            tl.store(Out+i,off+(A-T*unit//S)*G*128,i<N)
        _REBASE=rebase
    offsets=plan.reserve('slot_rotation_offsets',rotation.offsets.shape,rotation.offsets.dtype)
    n=offsets.numel()
    if n:_REBASE[((n+255)//256,)](rotation.offsets,offsets,n,total,start,splits,group,256)
    return replace(rotation,offsets=offsets)


def run_slot_native(q,k,v,*,source,rotation,binding,cache,config,family,resident_kv):
    import torch
    import triton
    from .table2_native import NativeSourceSpec,_native_plan,transfer_units
    from .headgroup import native_attention,_kernels
    from .device_route import native
    from .canonical_prepack import PackedArchiveSource
    from .archive_producer import submit_archive,submit_group_major
    started=time.perf_counter_ns()
    if (resident_kv is None or resident_kv.prepared is not None
            or config['schedule']!='frontier_streaming' or config.get('submission','unit')!='unit'):
        raise ValueError('bounded slots require unmaterialized resident ranges and unit Streaming')
    resident_kv.validate(q,k,v)
    total=resident_kv.total_tokens;g=config.get('head_group',source.head_group)
    spec=NativeSourceSpec(source.frame_ids,source.frame_tokens,source.history_start,total,q.shape[2],g)
    source.verify(spec)
    if rotation is not None:rotation.validate(source,spec,q.device)
    archived=isinstance(source,PackedArchiveSource)
    permutation=source.physical_to_logical if archived else None
    if rotation is not None and permutation is not None:rotation=rotation.in_physical_order(permutation)
    n=source.history_tokens if archived else source.key.shape[1]
    a,b=source.history_start,source.history_start+n
    if (resident_kv.history_start,resident_kv.history_tokens)!=(a,n):
        raise ValueError('resident/history slot layout mismatch')
    splits,slots,cap=slot_geometry(total,config.get('splits',12),config['kv_slots'],config['kv_slot_tokens'])
    cfg=dict(config,splits=splits)
    slot_cache=cache.setdefault('_bounded_slots_plans',{})
    p=_native_plan(q,total,g,binding=binding,cache=slot_cache,config=cfg,family=family)
    if not getattr(p,'slot_batch_resolved',False):
        from .slot_batch import make_producer
        producer=make_producer(q.device)
        if producer is not None:p.producer=producer
        p.slot_batch_active=producer is not None
        p.slot_batch_resolved=True
    signature=(tuple(q.shape),total,rotation is not None,slots,cap)
    cold=signature not in p.primed
    groups=source.groups;u=groups*splits
    p.total=total;p.chrono=False;p.prefix_tokens=0
    p.dense=p.reserve('kv_receive_slots',(slots,2,cap,g,128),q.dtype)
    p.dma_ready=p.reserve('slot_dma_ready',(u,),torch.int32)
    p.consume_ready=p.reserve('slot_prepare_ready',(u,),torch.int32)
    consumed=p.reserve('slot_consumed',(u,),torch.int32)
    for words in (p.dma_ready,p.consume_ready,consumed):words.zero_()
    origin=torch.cuda.Event(enable_timing=True);origin.record()
    out=p.reserve('group_out',q.shape,q.dtype)
    p.lse=p.reserve('group_lse',(1,q.shape[2],q.shape[1]),torch.float32)
    parts=p.reserve('group_partial',(splits,*q.shape),q.dtype) if splits>1 else out.unsqueeze(0)
    lses=p.reserve('group_partial_lse',(splits,1,q.shape[2],q.shape[1]),torch.float32)
    _,merge=_kernels()
    # Materialize Triton entry points before installing any future-unit waits.
    if cold:
        compiled=merge.warmup(parts,lses,out,p.lse,q.shape[1],q.shape[2],splits,
            triton.next_power_of_2(splits),grid=((q.shape[1]+15)//16,q.shape[2]))
        _=compiled.run
    units=transfer_units(groups,splits,total,a,n,reverse=cfg.get('reverse',True))
    local_rotation=_rotation_offsets(p,rotation,total,a,splits,g)
    _preload_units(local_rotation,p.dense[0,0],units,a,n)
    submit=submit_archive if archived else submit_group_major
    nbytes,table,host=submit(p,source,slot_count=slots,slot_tokens=cap,defer=True)
    expected=n*q.shape[2]*128*4
    if nbytes!=expected:raise RuntimeError('slot transfer byte coverage mismatch')
    m=native()
    p.ready=torch.cuda.Event(enable_timing=True);p.ready.record()
    p.fetch.wait_event(p.ready);p.prep.wait_event(p.ready);p.compute.wait_event(p.ready)
    stages=[];diag=cfg.get('diagnostics',False)
    def produce(first=0,stop=-1):
        p.producer.submit_slots(table,host,p.dma_ready,p.dense,consumed,p.fetch.cuda_stream,slots,first,stop)
    if not cold:
        # Stage the descriptor once. At most the free slots may be submitted
        # ahead of the host; a reused slot is admitted only after its previous
        # consumer-completed write exists on the compute stream.
        p.producer.submit_slots_windowed(table,host,p.dma_ready,p.dense,consumed,
                                        p.fetch.cuda_stream,slots)
    direct_units=0
    for ordinal,unit in enumerate(units):
        idx=unit.group*splits+unit.split;slot=ordinal%slots
        if cold:
            # Real first FA calls may synchronize CUDA. Never enqueue a wait
            # for a future consumer before those calls have initialized.
            produce(ordinal,ordinal+1);p.producer.wait_submitted()
        from .slot_resident import direct_resident_unit
        direct=direct_resident_unit(resident_kv,unit.group,unit.begin,unit.end,g,p.dense)
        direct_units+=direct is not None
        before,ready,rotated=[torch.cuda.Event(enable_timing=True) if diag else None for _ in range(3)]
        with torch.cuda.stream(p.prep):
            if diag:before.record()
            if direct is None:
                if ordinal>=slots:
                    previous=units[ordinal-slots]
                    m.group_ready(consumed,previous.group*splits+previous.split,False)
                m.group_ready(p.dma_ready,idx,False)
                if diag:ready.record()
                resident_kv.write_unit(p.dense[slot],unit.group,unit.begin,unit.end)
                lo,hi=max(0,unit.begin-a),min(n,unit.end-a)
                if local_rotation is not None and hi>lo:
                    local_rotation.rotate_tokens_(p.dense[slot,0],lo,hi)
                if diag:rotated.record()
                m.group_ready(p.consume_ready,idx,True)
            elif diag:
                ready.record();rotated.record()
        waiting,begin,end=[torch.cuda.Event(enable_timing=True) if diag else None for _ in range(3)]
        with torch.cuda.stream(p.compute):
            if diag:waiting.record()
            if direct is None:m.group_ready(p.consume_ready,idx,False)
            else:
                # This unit borrows native CUDA KV and never touches receive
                # storage. Slot-backed units still release only after LSE.
                m.group_ready(consumed,idx,True)
            if diag:begin.record()
            hs=slice(unit.group*g,(unit.group+1)*g);length=unit.end-unit.begin
            key,value=direct if direct is not None else (p.dense[slot,0,:length].unsqueeze(0),p.dense[slot,1,:length].unsqueeze(0))
            _,lse=native_attention(q[:,:,hs],key,value,out=parts[unit.split,:,:,hs],
                splits=cfg.get('native_splits',1),version=cfg.get('version',3))
            lses[unit.split,:,hs].copy_(lse)
            # After every native FA read, including internal split kernels.
            m.group_ready(consumed,idx,True)
            if not cold:p.producer.admit_slots(min(u,ordinal+slots+1))
            if diag:end.record()
            if cold:
                cold_done=torch.cuda.Event();cold_done.record()
        stages.append([idx,before,ready,rotated,waiting,begin,end])
        if cold:cold_done.synchronize()
    with torch.cuda.stream(p.compute):
        merge_begin=torch.cuda.Event(enable_timing=True) if diag else None
        if diag:merge_begin.record()
        if splits>1:
            merge[((q.shape[1]+15)//16,q.shape[2])](parts,lses,out,p.lse,
                q.shape[1],q.shape[2],splits,triton.next_power_of_2(splits))
        else:p.lse.copy_(lses[0])
        done=torch.cuda.Event(enable_timing=True);done.record()
    torch.cuda.current_stream(q.device).wait_event(done)
    p.end=done;done.synchronize()
    p.group_records.append(((origin,p.ready),stages,merge_begin,done,False))
    dma,rows=p.collect()
    if sum(int(r[1]) for r in dma)!=nbytes:raise RuntimeError('slot producer byte count mismatch')
    p.primed.add(signature)
    return out,dict(profile={'profile':{'sha256':binding.hardware_profile_binding.get('sha256')}},
        consumer_backend=f"FA{cfg.get('version',3)}",h2d_bytes=nbytes,h2d_copies=sum(int(r[0]) for r in dma),
        head_group=g,external_splits=splits,submission='unit',reverse=cfg.get('reverse',True),
        native_splits=cfg.get('native_splits',1),device_profile=cfg.get('device_profile','explicit_native'),
        producer_schedule=cfg['schedule'],producer='cpp_unit_batch' if p.slot_batch_active else 'cpp_async',cpu_kv_gather=not archived,
        source_layout='clean_block_group_archive' if archived else 'selected_group_major',
        physical_order='frame_block_token' if permutation is not None else 'chronological_token',
        resident_direct_write=True,direct_resident_units=direct_units,kv_slots=slots,kv_slot_tokens=cap,
        producer_host_window_units=slots if not cold else 1,
        receive_capacity_bytes=p.dense.numel()*p.dense.element_size(),
        partial_output_bytes=parts.numel()*parts.element_size()+lses.numel()*lses.element_size(),
        cold_serial_initialization=cold,pipeline_wall_ms=(time.perf_counter_ns()-started)/1e6,
        final_join=True,group_stage_envelopes=rows)
