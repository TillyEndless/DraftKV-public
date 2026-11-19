# SPDX-License-Identifier: Apache-2.0
"""OFF/SUMMARY instrumentation. Never changes a model, kernel or selection recipe.

Correctness events are explicitly distinguished from optional timing events.
OFF does not import torch, read clocks/allocator diagnostics, or create outputs.
"""
from contextlib import contextmanager, nullcontext
from importlib.resources import files
from pathlib import Path
import json
import hashlib
import math
import os
import subprocess
import threading
import sys
from draftkv.runtime.io import sha
import time

from draftkv.runtime.io import require, read, write

_MODE = 'off'
_ACTIVE = None


def configure(mode='off'):
    global _MODE
    require(mode in ('off', 'summary'), 'Profile mode must be off or summary')
    _MODE = mode


def enabled(): return _MODE == 'summary'

def clock_ns(): return time.perf_counter_ns() if enabled() else 0

def event(torch, *, required=False):
    """Required events survive OFF with timing disabled; diagnostic events do not."""
    return torch.cuda.Event(enable_timing=enabled()) if required or enabled() else None


def record(value, stream=None):
    if value is not None:
        value.record() if stream is None else value.record(stream)


def diagnostic_sync(value):
    if enabled() and value is not None: value.synchronize()


def elapsed(begin, end):
    return begin.elapsed_time(end) if enabled() and begin is not None and end is not None else None


def relative(events):
    if not enabled(): return {}
    return {k: elapsed(events['origin'], v) for k, v in events.items() if k != 'origin'}


def peak_allocated(torch, device):
    return torch.cuda.max_memory_allocated(device) if enabled() else None


def observe(*, route, fetch_attention_ms=None, h2d_transfer_ms=None, h2d_bytes=None,
            receive_buffer_bytes=None, source=None):
    if not enabled() or _ACTIVE is None: return
    _ACTIVE.observe(route=route, fetch_attention_ms=fetch_attention_ms,
                    h2d_transfer_ms=h2d_transfer_ms, h2d_bytes=h2d_bytes,
                    receive_buffer_bytes=receive_buffer_bytes, source=source)


def diffusion_start():
    if not enabled() or _ACTIVE is None:return
    require(_ACTIVE.diffusion_started is None,'Duplicate native diffusion start')
    import torch
    torch.cuda.synchronize(_ACTIVE.device)
    _ACTIVE.diffusion_started=time.perf_counter()


def diffusion_end():
    if not enabled() or _ACTIVE is None:return
    require(_ACTIVE.diffusion_started is not None and _ACTIVE.diffusion_s is None,'Invalid native diffusion end')
    import torch
    torch.cuda.synchronize(_ACTIVE.device)
    _ACTIVE.diffusion_s=time.perf_counter()-_ACTIVE.diffusion_started


def protocol():
    return json.loads(files('draftkv').joinpath('data/paper_measurement_protocol.json').read_text())


class Totals:
    """Online sums/counts and buffer high-water mark; no per-tile trace retention."""
    def __init__(self):
        self.routes = {};self.device=None;self.diffusion_started=None;self.diffusion_s=None

    def observe(self, *, route, source=None, **values):
        row = self.routes.setdefault(route, {'calls': 0, 'metrics': {}, 'source': source})
        row['calls'] += 1
        for name, value in values.items():
            if value is None: continue
            require(isinstance(value, (int, float)) and math.isfinite(value) and value >= 0, 'Invalid measured metric: ' + name)
            metric = row['metrics'].setdefault(name, {'sum': 0, 'count': 0, 'max': 0})
            metric['sum'] += value; metric['count'] += 1; metric['max'] = max(metric['max'], value)


class _DeviceMemory:
    """Optional whole-device NVML sampling through nvidia-smi, scoped to the call."""
    def __init__(self, torch, device):
        self.peak = None; self.samples = 0; self.reason = None; self.proc = None; self.thread = None
        try:
            prop = torch.cuda.get_device_properties(device)
            uuid = getattr(prop, 'uuid', None)
            if isinstance(uuid,bytes): uuid=uuid.decode('ascii')
            if not uuid: raise ValueError('CUDA device UUID unavailable; no physical-index guessing')
            uuid=str(uuid)
            if not uuid.startswith(('GPU-','MIG-')): uuid='GPU-'+uuid
            self.proc = subprocess.Popen(['nvidia-smi','--id='+uuid,'--query-gpu=memory.used','--format=csv,noheader,nounits','--loop-ms=50'], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            self.thread = threading.Thread(target=self._read, daemon=True); self.thread.start()
        except (OSError, ValueError) as error: self.reason = str(error)

    def _read(self):
        for line in self.proc.stdout:
            try: value = float(line.strip()) * 2**20
            except ValueError:
                self.reason='NVML sampler: '+line.strip()[:240]
                continue
            self.peak = max(self.peak or 0, int(value)); self.samples += 1

    def stop(self):
        if self.proc:
            self.proc.terminate()
            try: self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired: self.proc.kill(); self.proc.wait()
            self.thread.join(timeout=3); self.proc.stdout.close()
        if not self.samples and not self.reason: self.reason = 'No valid NVML sample during the measured interval'


def run_profiled(plan, call):
    """Source-audited native generation seam, after model load, before caller encode.

    SUMMARY has two explicit device fences and allocator peak reset. OFF calls
    through directly. Internal first-use compilation/warmup remain in the call.
    """
    if not enabled(): return call()
    import torch
    global _ACTIVE
    require(_ACTIVE is None, 'Nested generation measurement is forbidden')
    device = int(os.environ.get('LOCAL_RANK', '0'))
    rank = int(os.environ.get('RANK', '0'))
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    memory = _DeviceMemory(torch, device)
    totals = Totals(); totals.device=device; _ACTIVE = totals
    watch=None
    if plan.get('_latency_build_preparation') is not None:
        from draftkv.backbones.latency_build import compilation_watch
        watch=compilation_watch(torch,plan['_latency_build_preparation'])
    started = time.perf_counter(); status = 'FAILED'; error = None
    try:
        with watch if watch is not None else nullcontext():
            started=time.perf_counter()
            if watch is not None:watch.attach_runtime_loader(torch)
            result = call()
            torch.cuda.synchronize(device)
            duration=time.perf_counter()-started
            status = 'COMPLETE'
        return result
    except BaseException as exc:
        error = type(exc).__name__ + ': ' + str(exc)
        raise
    finally:
        if status!='COMPLETE':duration = time.perf_counter() - started
        _ACTIVE = None; memory.stop()
        release = read(os.environ['DRAFTKV_PUBLIC_RELEASE_PLAN'])
        prop = torch.cuda.get_device_properties(device)
        row = {'case_id': plan['request']['case_id'], 'rank': rank, 'local_rank': device,
               'physical_device': release['physical_devices'][device], 'status': status,
               'error': error, 'generation_e2e_s': duration if status=='COMPLETE' else None,
               'failed_call_wall_s': duration if status!='COMPLETE' else None,
               'diffusion_s':totals.diffusion_s,
               'latency_admission':watch.report() if watch is not None else {'status':'PROTOCOL_UNVERIFIED','reason':'Historical admission observer unavailable for this source profile'},
               'build_preparation':plan.get('_latency_build_preparation'),
               'requested_frames': plan['request']['frames'], 'source_sha': release['binding']['runtime_source_revision'],
               'checkpoint_sha256': plan.get('checkpoint_sha256'),
               'kernel_hashes': {name:sha(module.__file__) for name,module in list(sys.modules.items()) if getattr(module,'__file__',None) and str(module.__file__).endswith('.so') and any(part in name.lower() for part in ('b01','producer','prepack','flash','route','consumer'))},
               'gpu_name': prop.name, 'cuda_version': torch.version.cuda, 'torch_version': str(torch.__version__),
               'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
               'peak_reserved_bytes': torch.cuda.max_memory_reserved(device),
               'nvml_device_sampled_peak_bytes': memory.peak, 'nvml_samples': memory.samples,
               'nvml_unavailable_reason': memory.reason, 'routes': totals.routes,
               'scope': 'native_generate_after_model_load_through_native_decode; Solar includes native video write',
               'warmup': 'internal first-use initialization/compilation included, no hidden prewarm',
               'compile_admission': 'Build preparation outside E2E; remaining runtime cold initialization included, never subtracted; source-verified observer rejects compilation for paper admission when available'}
        folder = Path(release['workspace'])/'profile'/'ranks'; target = folder/('rank'+str(rank)+'.json')
        record_doc = read(target) if target.exists() else {'profile_mode':'summary','backbone':release['backbone'],'sp':release['runtime_sp'],'rank':rank,'cases':[]}
        require(not any(c['case_id']==row['case_id'] for c in record_doc['cases']), 'Duplicate profile case/rank')
        record_doc['cases'].append(row); write(target, record_doc)


def _sum_complete(rows, field, *, maximum=False):
    values = []
    for row in rows:
        calls = sum(r['calls'] for r in row['routes'].values())
        metrics = [r['metrics'].get(field) for r in row['routes'].values()]
        if not calls or any(m is None or m['count']!=r['calls'] for r,m in zip(row['routes'].values(),metrics)):
            return None
        values.append(max(m['max'] for m in metrics) if maximum else sum(m['sum'] for m in metrics))
    return max(values)  # critical rank total, never sum SP ranks


def checkpoint_identity(release):
    """Describe pins already verified by preflight; never hash weights in OFF.

    Directory identity hashes the canonical relative-file SHA256 index, rather
    than pretending that a collection of checkpoint shards is one model file.
    """
    if not release.get('asset_config'):
        return {'sha256':None,'scheme':'UNAVAILABLE','reason':'No asset index supplied'}
    assets=read(release['asset_config'])
    index=assets.get('checkpoint_files')
    if index:
        raw=json.dumps(index,sort_keys=True,separators=(',',':')).encode()
        return {'sha256':hashlib.sha256(raw).hexdigest(),
                'scheme':'sha256-canonical-relative-file-sha256-index-v1',
                'files':index,'basis':'exact file pins verified by generation preflight'}
    return {'sha256':assets.get('checkpoint_sha256'),'scheme':'file-sha256',
            'basis':'exact file pin verified by generation preflight'}


def aggregate(release, video_rows):
    """Called once after real video validation. Missing ranks/metrics stay null."""
    if release.get('profile_mode','off') != 'summary': return None
    root = Path(release['workspace'])/'profile'
    ranks = []
    for rank in range(release['runtime_sp']):
        p = root/'ranks'/('rank'+str(rank)+'.json')
        if p.is_file(): ranks.append(read(p))
    case_ids = [v['case_id'] for v in video_rows]
    complete = len(ranks)==release['runtime_sp'] and bool(case_ids)
    cases = []; missing = []
    for video in video_rows:
        rows = [next((c for c in r['cases'] if c['case_id']==video['case_id']), None) for r in ranks]
        good = len(rows)==release['runtime_sp'] and all(r is not None and r['status']=='COMPLETE' for r in rows)
        if not good: complete=False; missing.append(video['case_id']); continue
        e2e = max(r['generation_e2e_s'] for r in rows)
        cases.append({'case_id':video['case_id'],'generation_e2e_s':e2e,
                      'generated_frames':video.get('generated_frames'), 'output_video_duration_s':video.get('output_video_duration_s'),
                      'fetch_attention_ms':_sum_complete(rows,'fetch_attention_ms'),
                      'h2d_transfer_ms':_sum_complete(rows,'h2d_transfer_ms'),
                      'h2d_bytes':_sum_complete(rows,'h2d_bytes'),
                      'diffusion_s':max(r['diffusion_s'] for r in rows) if all(r.get('diffusion_s') is not None for r in rows) else None,
                      'receive_buffer_bytes':_sum_complete(rows,'receive_buffer_bytes', maximum=True)})
    e2e = sum(c['generation_e2e_s'] for c in cases) if complete else None
    frames = sum(c['generated_frames'] for c in cases) if complete and all(c['generated_frames'] is not None for c in cases) else None
    all_rows = [c for r in ranks for c in r['cases'] if c['case_id'] in case_ids]
    spec = protocol()['backbones'][release['backbone']]
    memory_field = 'nvml_device_sampled_peak_bytes' if spec['paper_memory_scope']=='NVML_WHOLE_DEVICE_SAMPLED' else 'peak_allocated_bytes'
    peaks = [r.get(memory_field) for r in all_rows]
    peak = max(peaks)/2**30 if complete and peaks and all(p is not None for p in peaks) else None
    def case_metric(name, maximum=False):
        values=[c[name] for c in cases]
        return (max(values) if maximum else sum(values)) if complete and values and all(v is not None for v in values) else None
    report={'schema_version':1,'profile_mode':'summary','measurement_protocol':'paper-efficiency','protocol_version':protocol()['version'],
            'backbone':release['backbone'],'source_sha':release['binding']['runtime_source_revision'],'sp':release['runtime_sp'],
            'generation_e2e_s':e2e,'generated_frames':frames,'output_video_duration_s':sum(c['output_video_duration_s'] for c in cases) if complete and all(c['output_video_duration_s'] is not None for c in cases) else None,
            'throughput_fps':frames/e2e if frames is not None and e2e else None,
            'generation_throughput_fps':frames/e2e if frames is not None and e2e else None,
            'fetch_attention_ms':case_metric('fetch_attention_ms'), 'h2d_transfer_ms':case_metric('h2d_transfer_ms'),
            'h2d_bytes':case_metric('h2d_bytes'),'h2d_transfer_bytes':case_metric('h2d_bytes'),
            'receive_buffer_mib':case_metric('receive_buffer_bytes',True)/2**20 if case_metric('receive_buffer_bytes',True) is not None else None,
            'peak_gpu_memory_gib':peak,
            'peak_gpu_allocated_memory_gib':max(r['peak_allocated_bytes'] for r in all_rows)/2**30 if complete and all_rows else None,
            'peak_reserved_memory_gib':max(r['peak_reserved_bytes'] for r in all_rows)/2**30 if complete and all_rows else None,
            'diffusion_s':case_metric('diffusion_s'),
            'model_only_fps':frames/case_metric('diffusion_s') if frames is not None and case_metric('diffusion_s') else None,
            'paper_latency_admission':'PASS' if complete and all_rows and all(r.get('latency_admission',{}).get('status')=='PASS' for r in all_rows) else 'NOT_ADMITTED',
            'per_rank':ranks, 'cases':cases,'missing_or_failed_cases':missing,
            'streaming_overlap':{'status':'UNAVAILABLE','reason':'Event envelopes contain frontier/rotation/slot waits; no active-kernel/DMA intersection is observed. H2D and Fetch+Attn are not added.'},
            'metrics_provenance':protocol()['metrics'],
            'paper_protocol_status': 'PROTOCOL_UNVERIFIED' if release['runtime_sp']!=spec.get('paper_sp') or release['binding']['runtime_source_revision']!=spec.get('paper_source_sha') or spec['status']!='VERIFIED_BOUNDARY' else 'BOUNDARY_ALIGNED_NOT_NUMERICAL_REPRODUCTION',
            'peak_gpu_memory_scope':memory_field,'measurement_status':('MEASURED' if complete else 'INCOMPLETE') if all_rows else 'NOT_MEASURED'}
    checkpoint=checkpoint_identity(release)
    report.update(model_checkpoint_hash=checkpoint['sha256'],model_checkpoint_identity=checkpoint,
                  gpu_count=release['runtime_sp'],
                  kernel_hashes_by_rank={str(r['rank']):{c['case_id']:c.get('kernel_hashes',{}) for c in r['cases']} for r in ranks})
    for key in ('gpu_name','cuda_version','torch_version'):
        values=sorted({r[key] for r in all_rows if r.get(key) is not None})
        report[key]=values[0] if len(values)==1 else None
    required=['generation_e2e_s','throughput_fps','fetch_attention_ms','h2d_transfer_ms','h2d_bytes','receive_buffer_mib','peak_gpu_memory_gib']
    report['unavailable_metrics']={k:'UNAVAILABLE: missing source-supported measurement, complete rank coverage or final video metadata' for k in required if report[k] is None}
    if complete and report['unavailable_metrics']:report['measurement_status']='PARTIAL'
    write(root/'summary.json',report)
    return report
