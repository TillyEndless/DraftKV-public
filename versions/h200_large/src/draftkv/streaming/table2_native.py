"""Device-dispatched canonical A2 pinned source -> existing native FA tasks.

No selector, pooling, budget, source ordering or RoPE math is defined here.
Reuse HeadGroupPlan's group/split stream waits, native FA and LSE merge.
"""
from dataclasses import dataclass
import time


@dataclass(frozen=True)
class TransferUnit:
    group: int
    split: int
    begin: int
    end: int
    source_begin: int
    source_end: int


def transfer_units(groups, splits, total, history_start, history_tokens, *, reverse=True):
    if (not all(type(x) is int for x in (groups,splits,total,history_start,history_tokens))
            or groups<1 or not 1<=splits<=total
            or not 0<=history_start<=history_start+history_tokens<=total):
        raise ValueError('invalid direct native FA unit geometry')
    return tuple(TransferUnit(g,s,total*s//splits,total*(s+1)//splits,
        max(total*s//splits,history_start)-history_start,
        min(total*(s+1)//splits,history_start+history_tokens)-history_start)
        for g in range(groups) for s in (range(splits-1,-1,-1) if reverse else range(splits)))


def runtime_config(settings, *, device=None, native_version=None, heads=None):
    from .device_profiles import device_profile,legal_group
    profile=device_profile(device)
    cfg=settings.get('table2_native_runtime')
    if cfg is None:
        # The canonical Camera/Block hooks must actually enter the established
        # native-task executor on Hopper. Presence of headgroup.py alone did
        # not connect it. Device policy selects the native kernel.
        hopper=False
        ada=False
        blackwell=False
        if device is not None:
            import torch
            if torch.device(device).type=='cuda':
                capability=torch.cuda.get_device_capability(device)
                major=capability[0]
                hopper,blackwell,ada=major==9,major==12,capability==(8,9)
        if settings.get('headgroup_pipeline') is False:
            if profile:raise ValueError('device native pipeline cannot be disabled; use serial schedule for ablation')
            return None
        if not (hopper or blackwell or ada) and not settings.get('headgroup_pipeline',False):return None
        schedule=settings.get('pipeline_schedule','pipeline')
        if schedule not in ('pipeline','serial'):
            raise ValueError('invalid native head-group schedule')
        cfg=dict(schedule='serial_barrier' if schedule=='serial' else 'frontier_streaming',
                 splits=settings.get('pipeline_splits',profile.splits if profile else 12),
                 version=profile.version if profile else (native_version if native_version is not None else (3 if hopper else 2)),
                 native_splits=settings.get('native_fa_splits',profile.native_splits if profile else 1),
                 submission=settings.get('pipeline_submission',profile.submission if profile else 'unit'),
                 reverse=settings.get('pipeline_reverse',True),diagnostics=False)
        if 'prepack_head_group' in settings:cfg['head_group']=settings['prepack_head_group']
    if not isinstance(cfg,dict) or set(cfg)-{'schedule','splits','diagnostics','version','head_group','native_splits','submission','reverse'}:
        raise ValueError('unknown Table2 native FA runtime field')
    cfg=dict(cfg)
    cfg.setdefault('version',profile.version if profile else 2)
    cfg.setdefault('splits',profile.splits if profile else 12)
    cfg.setdefault('native_splits',1)
    cfg.setdefault('submission',profile.submission if profile else 'unit')
    cfg.setdefault('reverse',True)
    cfg.setdefault('diagnostics',False)
    if cfg.get('schedule') not in ('serial_barrier','frontier_streaming'):
        raise ValueError('explicit Table2 native FA schedule required')
    if type(cfg.get('splits',12)) is not int or cfg.get('splits',12)<1:
        raise ValueError('positive external split count required')
    if type(cfg.get('version',2)) is not int or cfg.get('version',2) not in (2,3):
        raise ValueError('native attention version must be 2 or 3')
    if type(cfg.get('diagnostics',False)) is not bool:
        raise ValueError('diagnostics must be bool')
    if profile and cfg['version']!=profile.version:
        raise ValueError(f'{profile.name} requires native FA{profile.version}; conflicting explicit version')
    if type(cfg['native_splits']) is not int or cfg['native_splits']<1 or (cfg['version']==2 and cfg['native_splits']!=1):
        raise ValueError('invalid native split count for attention backend')
    if cfg['submission'] not in ('wave','unit') or type(cfg['reverse']) is not bool:
        raise ValueError('invalid producer/consumer ordering')
    preferred=profile.group if profile else None
    if 'head_group' in cfg:
        group=cfg['head_group']
        if type(group) is not int or group<1 or (heads is not None and heads%group):
            raise ValueError('explicit head group must divide actual local heads')
    elif preferred is not None:
        cfg['head_group']=legal_group(heads,preferred) if heads is not None else preferred
    cfg['device_profile']=profile.name if profile else 'generic_native'
    cfg['preferred_head_group']=preferred
    cfg['group_adjusted']=preferred is not None and cfg.get('head_group')!=preferred
    return cfg


@dataclass(frozen=True)
class NativeSourceSpec:
    """Native transport contract; independent of legacy B01 FA2 tile profiles."""
    frame_ids: tuple
    frame_tokens: int
    history_start: int
    kv_tokens: int
    heads: int
    head_group: int
    consumer: str = 'native_headgroup'

    def __post_init__(self):
        if (not isinstance(self.frame_ids,tuple) or not self.frame_ids
                or any(type(f) is not int or f<0 for f in self.frame_ids)
                or len(set(self.frame_ids))!=len(self.frame_ids)
                or any(type(v) is not int for v in (self.frame_tokens,self.history_start,self.kv_tokens,self.heads,self.head_group))
                or self.frame_tokens<1 or self.history_start<0 or self.heads<1
                or self.head_group<1 or self.heads%self.head_group
                or self.history_start+len(self.frame_ids)*self.frame_tokens>self.kv_tokens):
            raise ValueError('invalid native group-major source contract')

    @property
    def groups(self):return self.heads//self.head_group


def run_direct_native(q,k,v,*,source,rotation,binding,cache,config,family):
    import torch
    from .headgroup import HeadGroupPlan,PipelineConfig,native_attention
    from .device_route import native
    from .pipeline import _check_qkv
    started=time.perf_counter_ns()
    host_phases={}
    _check_qkv(torch,q,k,v)
    version=config.get('version',2)
    if version==3 and torch.cuda.get_device_capability(q.device)[0]!=9:
        raise ValueError('Table2 native FA3 requires Hopper SM90')
    profile={'profile':{'sha256':binding.hardware_profile_binding.get('sha256')}}
    if source is None:
        if rotation is not None:raise ValueError('rotation without direct history')
        out,lse=native_attention(q,k,v,version=version,splits=1)
        torch.cuda.current_stream(q.device).synchronize()
        return out,dict(profile=profile,consumer_backend=f'FA{version}',h2d_bytes=0,
            producer_schedule=config['schedule'],device_profile=config.get('device_profile','explicit_native'),
            head_group=config.get('head_group',q.shape[2]),submission=config.get('submission','unit'),
            reverse=config.get('reverse',True),native_splits=config.get('native_splits',1),final_join=True,
            pipeline_wall_ms=(time.perf_counter_ns()-started)/1e6)
    # A legacy token-consumer PlanSpec forces all local heads into one group.
    # Native FA must verify the actual device-selected group/layout instead.
    g=config.get('head_group',source.head_group)
    spec=NativeSourceSpec(source.frame_ids,source.frame_tokens,source.history_start,
                          k.shape[1],q.shape[2],g)
    source.verify(spec)
    if rotation is not None:rotation.validate(source,spec,q.device)
    groups,g=source.groups,source.head_group;total=k.shape[1]
    from .canonical_prepack import PackedArchiveSource
    archived=isinstance(source,PackedArchiveSource)
    permutation=source.physical_to_logical if archived else None
    if rotation is not None and permutation is not None:
        rotation=rotation.in_physical_order(permutation)
    history_tokens=source.history_tokens if archived else source.key.shape[1]
    splits=min(config.get('splits',12),total)
    reverse=config.get('reverse',True);submission=config.get('submission','unit')
    native_splits=config.get('native_splits',1)
    units=transfer_units(groups,splits,total,source.history_start,history_tokens,reverse=reverse)
    key=(str(q.device),q.shape[2],g,splits,family,config['schedule'],config.get('diagnostics',False),version,
         reverse,submission,native_splits)
    entry=cache.get('_table2_direct_native')
    phase=time.perf_counter_ns()
    if entry is None or entry[0]!=key:
        c=PipelineConfig(g,splits=splits,native_splits=native_splits,
            schedule='serial' if config['schedule']=='serial_barrier' else 'pipeline',
            reverse=reverse,submission=submission,diagnostics=config.get('diagnostics',False),
            native_attention_version=version)
        # Keep the established native task executor, stream/readiness ABI and
        # C++ producer. Canonical selection owns its independent source map.
        c.validate(q.shape[2],total)
        p=HeadGroupPlan(1,1,1,q.device,q.shape[2],family,c)
        cache['_table2_direct_native']=(key,p)
    else:p=entry[1]
    host_phases['plan_create_or_lookup_ms']=(time.perf_counter_ns()-phase)/1e6
    phase=time.perf_counter_ns()
    cold=(tuple(q.shape),total,rotation is not None) not in p.primed
    p.total=total;p.chrono=False;p.prefix_tokens=0
    p.dense=p.reserve('group_dense',(groups,2,total,g,128),q.dtype)
    p.dma_ready=p.reserve('group_dma_ready',(groups*splits,),torch.int32)
    p.consume_ready=p.reserve('group_consume_ready',(groups*splits,),torch.int32)
    p.dma_ready.zero_();p.consume_ready.zero_()
    origin=torch.cuda.Event(enable_timing=True);origin.record()
    a=source.history_start;b=a+history_tokens
    for group in range(groups):
        hs=slice(group*g,(group+1)*g)
        for side,resident in enumerate((k,v)):
            p.dense[group,side,:a].copy_(resident[0,:a,hs])
            p.dense[group,side,b:].copy_(resident[0,b:,hs])
    p.ready=torch.cuda.Event(enable_timing=True);p.ready.record()
    host_phases['resident_buffers_ms']=(time.perf_counter_ns()-phase)/1e6
    phase=time.perf_counter_ns()
    # Compile existing RoPE specializations before any external native FA wait.
    if rotation is not None:
        for unit in units:
            lo,hi=max(0,unit.source_begin),min(history_tokens,unit.source_end)
            if hi>lo:rotation.rotate_tokens_(p.dense[unit.group,0,a:b],lo,hi,preload=True)
    host_phases['rope_preload_ms']=(time.perf_counter_ns()-phase)/1e6
    phase=time.perf_counter_ns()
    native()  # Resolve module entry points before installing external ready waits.
    host_phases['native_resolve_ms']=(time.perf_counter_ns()-phase)/1e6
    phase=time.perf_counter_ns()
    p.fetch.wait_event(p.ready)
    from .archive_producer import submit_archive,submit_group_major
    if archived:
        nbytes=submit_archive(p,source)
    else:
        nbytes=submit_group_major(p,source)
    # First-use native initialization can synchronize the CUDA context;
    # it must never wait on producer work whose API calls do not exist yet.
    # Warm streaming instead submits consumers while the producer is still
    # enqueueing DMA, for BOTH archive and CPU-RoPE compatibility sources.
    if cold or config['schedule']=='serial_barrier':
        p.producer.wait_submitted()
    host_phases['producer_submit_and_cold_wait_ms']=(time.perf_counter_ns()-phase)/1e6
    phase=time.perf_counter_ns()
    expected=history_tokens*q.shape[2]*128*4
    if nbytes!=expected:raise RuntimeError('direct native FA transfer coverage/bytes mismatch')
    def prepare_unit(group,split,lo,hi):
        if rotation is not None:
            first,last=max(0,lo-a),min(b-a,hi-a)
            if last>first:rotation.rotate_tokens_(p.dense[group,0,a:b],first,last)
    # Cold native FA initialization may synchronize a context. Supply genuinely
    # ready data and enqueue all preparation before that first call, as in the
    # historical implementation. The real result is retained and cost counted.
    actual_config=p.config
    if cold:
        from dataclasses import replace
        p.config=replace(actual_config,schedule='serial')
    try:
        out=p.execute(q,None,(origin,p.ready),prepare_unit=prepare_unit)
        p.end.synchronize()  # explicit final producer/prep/consumer join
        dma,stages=p.collect()
        copies=sum(int(row[0]) for row in dma)
        if sum(int(row[1]) for row in dma)!=nbytes:
            raise RuntimeError('background producer actual byte count mismatch')
    finally:p.config=actual_config
    p.primed.add((tuple(q.shape),total,rotation is not None))
    host_phases['consumer_and_final_join_ms']=(time.perf_counter_ns()-phase)/1e6
    return out,dict(profile=profile,consumer_backend=f'FA{version}',h2d_bytes=nbytes,
        h2d_copies=copies,head_group=g,external_splits=splits,
        device_profile=config.get('device_profile','explicit_native'),
        preferred_head_group=config.get('preferred_head_group'),
        group_adjusted=config.get('group_adjusted',False),
        submission=submission,reverse=reverse,native_splits=native_splits,
        physical_order='frame_block_token' if permutation is not None else 'chronological_token',
        source_layout='clean_block_group_archive' if archived else 'selected_group_major',
        cpu_kv_gather=not archived,producer='cpp_async',
        producer_schedule=config['schedule'],cold_serial_initialization=cold,
        pipeline_wall_ms=(time.perf_counter_ns()-started)/1e6,
        final_join=True,group_stage_envelopes=stages,host_initialization_phases_ms=host_phases,
        plan_constructor_phases_ms=p.constructor_host_phases_ms if cold else None)
