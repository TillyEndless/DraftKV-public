"""Unified evidence validator. Event interval intersection is only an upper bound."""
from __future__ import annotations
import math

METRICS = ('producer_plan_ms','cpu_prepare_pinned_ms','h2d_ms','frontier_init_ms',
           'frontier_publish_timestamps_ms','consumer_start_ms','consumer_end_ms',
           'attention_wall_ms','gpu_active_ms','native_fa2_gpu_ms','native_fa2_wall_ms','peak_gpu_memory_bytes')


def interval_overlap(h2d_intervals, attention_intervals):
    """Union intersection of actual GPU activity intervals in a shared clock domain."""
    def union(intervals):
        merged=[]
        for start,end in sorted(intervals):
            if not all(math.isfinite(v) for v in (start,end)) or start<0 or end<start:
                raise ValueError('Invalid activity interval')
            if merged and start<=merged[-1][1]: merged[-1][1]=max(merged[-1][1],end)
            else: merged.append([start,end])
        return merged
    dma,kernels=union(h2d_intervals),union(attention_intervals)
    return sum(max(0.,min(b,d)-max(a,c)) for a,b in dma for c,d in kernels)


def validate_record(record):
    missing=[k for k in ('frozen_commit','source_snapshot_sha256','gpu_uuid','workload_hash','metrics','activity_trace') if k not in record]
    if missing: raise ValueError('Missing evidence fields: '+', '.join(missing))
    absent=[k for k in METRICS if k not in record['metrics']]
    if absent: raise ValueError('Missing metrics (use null for unmeasured): '+', '.join(absent))
    for key,value in record['metrics'].items():
        if isinstance(value,(int,float)) and (not math.isfinite(value) or value<0): raise ValueError('Invalid metric '+key)
    trace=record.get('activity_trace')
    if not trace:
        return {'status':'unverified','true_overlap_verified':False,'overlap_ms':None,'reason':'No actual DMA/kernel activity trace'}
    if not all(trace.get(k) for k in ('path','sha256','clock_domain','attention_kernel_names')) or 'h2d_intervals_ms' not in trace or 'attention_intervals_ms' not in trace:
        raise ValueError('Trace provenance and common-clock GPU intervals required')
    from pathlib import Path
    import hashlib
    file=Path(trace['path'])
    if not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest()!=trace['sha256']:
        raise ValueError('Activity trace hash mismatch')
    overlap=interval_overlap(trace['h2d_intervals_ms'],trace['attention_intervals_ms'])
    # Kernel residency may contain frontier waits; SM-active overlap requires profiler counters.
    return {'status':'activity_intervals_observed','true_overlap_verified':False,'overlap_ms':overlap,
            'reason':'DMA/kernel residency intersection; inspect SM-active trace/counters to distinguish frontier stalls'}
