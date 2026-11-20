"""Separate allocator counters and external NVIDIA sampled memory.

No CUDA synchronization, allocator reset, empty_cache, or tensor copy is added.
The external monitor starts before workers and uses two persistent nvidia-smi
queries (NVML), rather than spawning a process for every sample.
"""
import csv
from contextlib import contextmanager
from contextvars import ContextVar
import json
import os
from pathlib import Path
import selectors
import subprocess
import time

MIB = 1024 ** 2
GPU_PEAK_MEMORY_SCOPE = 'generation_nvml_whole_device_sampled_peak_max_per_gpu'

class MonitorInterrupted(RuntimeError):
    """Must not be swallowed by selectors' handling of InterruptedError."""

def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)

def memory_bytes(value):
    value = value.strip()
    if value in ('N/A', '[N/A]', 'Not Supported', '[Not Supported]', ''):
        return None
    return int(value) * MIB

def parse_sample(kind, line):
    fields = [x.strip() for x in next(csv.reader([line]))]
    if kind == 'gpu' and len(fields) == 4 and fields[1].isdigit():
        return dict(kind=kind, driver_timestamp=fields[0], index=int(fields[1]),
                    uuid=fields[2], used_bytes=memory_bytes(fields[3]))
    if kind == 'process' and len(fields) == 4 and fields[2].isdigit():
        return dict(kind=kind, driver_timestamp=fields[0], uuid=fields[1],
                    pid=int(fields[2]), used_bytes=memory_bytes(fields[3]))
    return None

def update_peak(bucket, field, value):
    if value is not None:
        bucket[field] = max(bucket.get(field, 0), value)

class MemoryAccumulator:
    def __init__(self):
        self.ranks = {}
        self.devices = {}
        self.other_processes = {}
        self.identity_mismatches = []
        self.cadence = {}

    def observe_cadence(self, sample):
        key = sample['kind'] + ':' + sample['uuid']
        if sample['kind'] == 'process': key += ':' + str(sample['pid'])
        item = self.cadence.setdefault(key, dict(samples=0, intervals=0, total_gap_ns=0, max_gap_ns=0))
        now = sample['monotonic_ns']
        if 'last_ns' in item:
            gap = now - item['last_ns']
            item['intervals'] += 1
            item['total_gap_ns'] += gap
            item['max_gap_ns'] = max(item['max_gap_ns'], gap)
        item['last_ns'] = now
        item['samples'] += 1

    def consume(self, sample, states):
        value = sample['used_bytes']
        if sample['kind'] == 'process':
            owned = next((s for s in states if s['pid'] == sample['pid'] or s['rank'] == sample.get('owner_rank')), None)
            if owned is None:
                key = f"{sample['uuid']}:{sample['pid']}"
                update_peak(self.other_processes.setdefault(key, {}), 'peak_used_bytes', value)
                return
            expected = owned.get('expected_gpu_uuid')
            if expected and expected != sample['uuid']:
                mismatch = dict(rank=owned['rank'], expected=expected, actual=sample['uuid'])
                if mismatch not in self.identity_mismatches:
                    self.identity_mismatches.append(mismatch)
            bucket = self.ranks.setdefault(str(owned['rank']), dict(pid=owned['pid'], uuid=sample['uuid'], phases={}))
            if sample['pid'] != owned['pid']:
                auxiliary = bucket.setdefault('auxiliary_processes', {}).setdefault(str(sample['pid']), {})
                update_peak(auxiliary, 'nvml_process_sampled_peak_bytes', value)
            else:
                update_peak(bucket, 'nvml_primary_process_sampled_peak_bytes', value)
            phase = bucket['phases'].setdefault(owned['phase'], {})
            for dst in (bucket, phase):
                update_peak(dst, 'nvml_process_sampled_peak_bytes', value)
                if owned.get('generation_active'):
                    update_peak(dst, 'generation_nvml_process_sampled_peak_bytes', value)
            return
        device = self.devices.setdefault(sample['uuid'], dict(index=sample['index'], phases={}))
        update_peak(device, 'nvml_device_sampled_peak_bytes', value)
        for owned in states:
            if owned.get('expected_gpu_uuid') != sample['uuid']:
                continue
            phase = device['phases'].setdefault(owned['phase'], {})
            update_peak(phase, 'nvml_device_sampled_peak_bytes', value)
            if owned.get('generation_active'):
                update_peak(device, 'generation_nvml_device_sampled_peak_bytes', value)

    def report(self):
        process = [x.get('generation_nvml_process_sampled_peak_bytes') for x in self.ranks.values()]
        device = [x.get('generation_nvml_device_sampled_peak_bytes') for x in self.devices.values()]
        cadence = {key: dict(samples=x['samples'],
            mean_received_interval_ms=x['total_gap_ns'] / x['intervals'] / 1e6 if x['intervals'] else None,
            max_received_gap_ms=x['max_gap_ns'] / 1e6 if x['intervals'] else None)
            for key, x in self.cadence.items()}
        return dict(ranks=self.ranks, devices=self.devices, other_processes=self.other_processes,
            observed_receipt_cadence=cadence,
            identity_mismatches=self.identity_mismatches,
            generation_max_rank_nvml_process_sampled_peak_bytes=max((x for x in process if x is not None), default=None),
            generation_max_device_nvml_sampled_peak_bytes=max((x for x in device if x is not None), default=None),
            sp_aggregation='maximum per GPU/rank; never sum ranks')

class RankMemoryRecorder:
    def __init__(self, run, rank, expected_gpu_uuid=None, torch_module=None):
        self.folder = Path(run) / 'memory'
        self.folder.mkdir(parents=True, exist_ok=True)
        self.rank = rank
        self.torch = torch_module
        self.state = dict(rank=rank, pid=os.getpid(), expected_gpu_uuid=expected_gpu_uuid,
                          phase='model_loading', generation_active=False)
        self.events = []
        self.status = 'RUNNING'
        self.generation_peaks = None
        self.finished = False
        self.journal = (self.folder / f'rank{rank}.events.jsonl').open('x', buffering=1)
        self.phase('model_loading')

    def snapshot(self):
        if self.torch is None or not self.torch.cuda.is_initialized():
            return None
        cuda = self.torch.cuda
        device = int(os.environ.get('LOCAL_RANK', '0'))
        return dict(allocated_bytes=int(cuda.memory_allocated(device)),
                    reserved_bytes=int(cuda.memory_reserved(device)),
                    cumulative_peak_allocated_bytes=int(cuda.max_memory_allocated(device)),
                    cumulative_peak_reserved_bytes=int(cuda.max_memory_reserved(device)))

    def phase(self, name):
        self.state.update(phase=name, monotonic_ns=time.monotonic_ns(), unix_ns=time.time_ns())
        event = dict(self.state, pytorch=self.snapshot())
        self.events.append(event)
        self.journal.write(json.dumps(event) + '\n')
        atomic_json(self.folder / f'rank{self.rank}.state.json', self.state)

    def finish(self, status, error=None):
        if self.finished:
            return
        self.status = status
        failed_phase = self.state['phase'] if status == 'FAILED' else None
        snapshot = self.snapshot()
        self.generation_peaks = snapshot if self.state.get('generation_active') else self.generation_peaks
        self.state['generation_active'] = False
        self.phase('failed' if status == 'FAILED' else 'finished')
        atomic_json(self.folder / f'torch_memory_rank{self.rank}.json', dict(
            schema_version=1, status=status, rank=self.rank, pid=self.state['pid'],
            failed_during_phase=failed_phase, error=error, generation_peaks=self.generation_peaks,
            events=self.events, phase_snapshots_are_not_phase_peaks=True,
            cumulative_peaks_reset_only_by_existing_generation_profiler=True,
            added_cuda_synchronizations=0, added_allocator_resets=0,
            worker_completed=status == 'COMPLETE', measurements_not_subtracted=True))
        self.finished = True
        self.journal.close()

    def begin_generation(self, torch_module):
        self.torch = torch_module
        self.state['generation_active'] = True
        self.phase('generation_setup')

    def end_generation(self):
        self.generation_peaks = self.snapshot()
        self.state['generation_active'] = False
        self.phase('generation_returned')

_CURRENT = ContextVar('draftkv_memory_recorder', default=None)

def current_recorder():
    return _CURRENT.get()

def memory_phase(name):
    recorder = current_recorder()
    if recorder is not None and not recorder.finished:
        recorder.phase(name)

@contextmanager
def record_worker(plan, torch_module=None):
    """Observe all algorithms before model loading and before hard CLI exit."""
    if not plan.get('gpu_profile', False):
        yield None
        return
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    uuids = os.environ.get('DRAFTKV_MEMORY_GPU_UUIDS', '').split(',')
    expected = uuids[local_rank] if local_rank < len(uuids) and uuids[local_rank] else None
    recorder = RankMemoryRecorder(plan['run_dir'], rank, expected, torch_module)
    token = _CURRENT.set(recorder)
    try:
        yield recorder
    except BaseException as error:
        try:
            recorder.finish('FAILED', f'{type(error).__name__}: {error}')
        except BaseException:
            pass  # A reporting error must not replace the original failure.
        raise
    else:
        recorder.finish('COMPLETE')
    finally:
        _CURRENT.reset(token)

def owner_rank(pid, states):
    """Attribute compiler/decode helpers to a worker, without unrelated PIDs."""
    owners = {s['pid']: s['rank'] for s in states}
    for _ in range(24):
        if pid in owners: return owners[pid]
        if pid <= 1: return None
        try:
            # comm can contain spaces; fields after the last ')' begin at state.
            tail = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
            pid = int(tail[1])
        except (OSError, ValueError, IndexError): return None
    return None

def finalize_memory_profile(run, expected_sp=None, worker_returncode=None, *, timing_profile=None):
    run = Path(run)
    ranks = [json.loads(p.read_text()) for p in sorted((run / 'memory').glob('torch_memory_rank*.json'))]
    nvml_path = run / 'memory/nvml_summary.json'
    nvml = json.loads(nvml_path.read_text()) if nvml_path.exists() else None
    profile_path = run / 'gpu_profile.json'
    # Aggregation supplies its prospective timing record before publishing the
    # canonical profile. This avoids reading a stale/half-written profile.
    profile = timing_profile if timing_profile is not None else (
        json.loads(profile_path.read_text()) if profile_path.exists() else None)
    eligible = bool(profile and profile.get('status') == 'COMPLETE'
                    and profile.get('formal_latency_admitted', True) and ranks
                    and {r['rank'] for r in ranks} == set(range(profile['sp']))
                    and len(ranks) == profile['sp']
                    and all(r['status'] == 'COMPLETE' and r.get('generation_peaks') for r in ranks))
    allocated = [r['generation_peaks']['cumulative_peak_allocated_bytes'] for r in ranks if r.get('generation_peaks')]
    reserved = [r['generation_peaks']['cumulative_peak_reserved_bytes'] for r in ranks if r.get('generation_peaks')]
    decode = None
    if (profile or {}).get('backbone') == 'SolarWM33B' or (run / 'sampling_receipt.json').exists():
        decode_path = run / 'decode_receipt.json'
        if not decode_path.exists(): decode_path = run / 'decode_memory_failure.json'
        if decode_path.exists():
            decode = json.loads(decode_path.read_text())
            if type(decode.get('peak_allocated_bytes')) is int: allocated.append(decode['peak_allocated_bytes'])
            if type(decode.get('peak_reserved_bytes')) is int: reserved.append(decode['peak_reserved_bytes'])
    expected_sp = expected_sp or (profile or {}).get('sp') or len(ranks)
    nvml_ranks = (nvml or {}).get('ranks', {})
    missing = [i for i in range(expected_sp) if not nvml_ranks.get(str(i), {}).get('generation_nvml_process_sampled_peak_bytes')]
    identity_path = run / 'memory_identity.json'
    identity = json.loads(identity_path.read_text()) if identity_path.exists() else {}
    expected_uuids = [d['uuid'] for d in identity.get('devices', [])]
    nvml_devices = (nvml or {}).get('devices', {})
    missing_devices = [uuid for uuid in expected_uuids if not
                       nvml_devices.get(uuid, {}).get('generation_nvml_device_sampled_peak_bytes')]
    process_peak = (nvml or {}).get('generation_max_rank_nvml_process_sampled_peak_bytes')
    device_peak = (nvml or {}).get('generation_max_device_nvml_sampled_peak_bytes')
    nvml_complete = bool(nvml and nvml['status'] == 'COMPLETE' and not missing
                         and not missing_devices and len(set(expected_uuids)) == expected_sp
                         and type(process_peak) is int and process_peak > 0
                         and type(device_peak) is int and device_peak > 0
                         and not nvml.get('identity_mismatches'))
    if worker_returncode not in (None, 0): eligible = False
    if nvml and nvml.get('identity_mismatches'): eligible = False
    result = dict(schema_version=2, status='COMPLETE' if eligible else 'FAILED_OR_INCOMPLETE',
        latency_eligible=eligible, nvml_complete=nvml_complete,
        memory_comparison_eligible=eligible and nvml_complete,
        worker_returncode=worker_returncode, expected_sp=expected_sp,
        ranks_without_generation_nvml=missing,
        devices_without_generation_nvml=missing_devices,
        e2e_latency_s=(profile or {}).get('e2e_latency_s'),
        diffusion_latency_s=(profile or {}).get('diffusion_latency_s'),
        peak_pytorch_allocated_bytes=max(allocated, default=None),
        peak_pytorch_reserved_bytes=max(reserved, default=None),
        generation_nvml_process_sampled_peak_bytes=process_peak,
        generation_nvml_device_sampled_peak_bytes=device_peak,
        peak_gpu_memory_bytes=device_peak if eligible and nvml_complete else None,
        peak_gpu_memory_gib=device_peak / (1024 ** 3) if eligible and nvml_complete else None,
        peak_memory_scope=GPU_PEAK_MEMORY_SCOPE,
        nvml_requested_interval_ms=identity.get('interval_ms'),
        source_allocated_profile_bytes=(profile or {}).get('peak_allocated_bytes'),
        per_rank=ranks, independent_decode_allocator=decode, nvml=nvml,
        definition='Allocated/reserved are allocator high-water counters for generation; NVML values are sampled lower bounds. Phase allocator entries are boundary snapshots, not independent phase peaks. No metric sums SP ranks.',
        partial_failure_memory_is_not_a_successful_E2E=True)
    if eligible:
        assert result['peak_pytorch_allocated_bytes'] == profile['peak_allocated_bytes']
    atomic_json(run / 'memory_profile.json', result)
    return result

def monitor(run, gpu_uuids, stop_file, ready_file, interval_ms=100):
    folder = Path(run) / 'memory'
    folder.mkdir(parents=True, exist_ok=True)
    selected = set(gpu_uuids)
    accumulator = MemoryAccumulator()
    queries = dict(gpu='timestamp,index,uuid,memory.used', process='timestamp,gpu_uuid,pid,used_memory')
    selector = selectors.DefaultSelector()
    children = []
    errors = []
    count = 0
    started = time.time()
    states = []
    last_summary = 0
    ready_devices = set()
    try:
        for kind, fields in queries.items():
            err = (folder / f'nvidia_smi_{kind}.stderr').open('x')
            argv = ['nvidia-smi', '--query-' + ('gpu' if kind == 'gpu' else 'compute-apps') + '=' + fields,
                    '--format=csv,noheader,nounits', f'--loop-ms={interval_ms}']
            if kind == 'gpu': argv.append('--id=' + ','.join(gpu_uuids))
            proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=err)
            err.close()
            children.append(proc)
            selector.register(proc.stdout, selectors.EVENT_READ, dict(kind=kind, pending=b''))
        with (folder / 'nvml_samples.jsonl').open('x', buffering=1) as raw:
            while not Path(stop_file).exists():
                states = []
                for path in folder.glob('rank*.state.json'):
                    try: states.append(json.loads(path.read_text()))
                    except (OSError, ValueError): pass
                for key, _ in selector.select(timeout=.1):
                    data = key.data
                    chunk = os.read(key.fileobj.fileno(), 65536)
                    if not chunk:
                        raise RuntimeError('Persistent nvidia-smi query ended unexpectedly')
                    lines = (data['pending'] + chunk).split(b'\n')
                    data['pending'] = lines.pop()
                    for line in lines:
                        try: sample = parse_sample(data['kind'], line.decode())
                        except (ValueError, UnicodeDecodeError): continue
                        if sample is None: continue
                        owner = owner_rank(sample['pid'], states) if sample['kind'] == 'process' else None
                        owned_pid = owner is not None
                        if sample['uuid'] not in selected and not owned_pid: continue
                        if owned_pid: sample['owner_rank'] = owner
                        sample.update(monotonic_ns=time.monotonic_ns(), received_unix_ns=time.time_ns())
                        sample['worker_markers'] = [dict(rank=s['rank'], pid=s['pid'], phase=s['phase'],
                            generation_active=s.get('generation_active', False), marker_monotonic_ns=s['monotonic_ns'])
                            for s in states if (s['rank'] == owner if owned_pid else s.get('expected_gpu_uuid') == sample['uuid'])]
                        accumulator.observe_cadence(sample)
                        accumulator.consume(sample, states)
                        raw.write(json.dumps(sample) + '\n')
                        count += 1
                        if sample['kind'] == 'gpu' and sample['used_bytes'] is not None:
                            ready_devices.add(sample['uuid'])
                        if ready_devices == selected and not Path(ready_file).exists():
                            atomic_json(ready_file, dict(status='READY', interval_ms=interval_ms,
                                gpu_uuids=sorted(ready_devices)))
                if time.monotonic() - last_summary >= 1:
                    atomic_json(folder / 'nvml_summary.json', dict(status='RUNNING', interval_ms=interval_ms,
                        samples=count, **accumulator.report()))
                    last_summary = time.monotonic()
    except BaseException as error:
        errors.append(f'{type(error).__name__}: {error}')
        raise
    finally:
        for child in children:
            if child.poll() is not None:
                errors.append(f'nvidia-smi query exited with code {child.returncode}')
        for child in children:
            if child.poll() is None: child.terminate()
        for child in children:
            try: child.wait(timeout=5)
            except subprocess.TimeoutExpired: child.kill(); child.wait()
        selector.close()
        atomic_json(folder / 'nvml_summary.json', dict(schema_version=1,
            status='FAILED' if errors or ready_devices != selected else 'COMPLETE', errors=errors,
            ready_devices=sorted(ready_devices), requested_devices=sorted(selected),
            interval_ms=interval_ms, samples=count, started_unix_s=started, ended_unix_s=time.time(),
            method='Two persistent nvidia-smi queries using NVML; MiB converted to bytes',
            sampled_peak_is_lower_bound=True, phase_assignment='latest worker host marker; resolution limited by sample interval and query latency',
            **accumulator.report()))

def resolve_gpu_uuids(selected_devices):
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,pci.bus_id',
                                   '--format=csv,noheader'], text=True)
    devices = {}
    for line in raw.splitlines():
        if not line.strip(): continue
        index, uuid, bus = (s.strip() for s in line.split(','))
        devices[int(index)] = dict(index=int(index), uuid=uuid, pci_bus_id=bus)
    if not selected_devices or len(selected_devices) != len(set(selected_devices)):
        raise ValueError('memory profiling requires distinct selected NVML GPU indices')
    return [devices[int(i)] for i in selected_devices]

@contextmanager
def memory_session(plan, environment):
    """Parent-owned collector; no torch import, synchronization or retries."""
    if not plan.get('gpu_profile', False):
        yield None
        return
    run = Path(plan['run_dir'])
    devices = resolve_gpu_uuids(plan['selected_devices'])
    if len(devices) != plan['sp']:
        raise ValueError('memory profiling requires one selected GPU per rank')
    uuids = [d['uuid'] for d in devices]
    environment['CUDA_VISIBLE_DEVICES'] = ','.join(uuids)
    environment['DRAFTKV_MEMORY_GPU_UUIDS'] = ','.join(uuids)
    atomic_json(run / 'memory_identity.json', dict(devices=devices, interval_ms=100,
                binding='CUDA_VISIBLE_DEVICES UUIDs in local rank order'))
    stop, ready = run / 'memory_monitor.stop', run / 'memory_monitor.ready.json'
    if stop.exists() or ready.exists():
        raise FileExistsError('refuse reuse of a memory profiling attempt')
    argv = [plan['python'], '-m', 'draftkv.backbones.memory_profile', '--run', str(run),
            '--gpu-uuids', *uuids, '--stop', str(stop), '--ready', str(ready), '--interval-ms', '100']
    with (run / 'memory_monitor.stdout.log').open('x') as out, (run / 'memory_monitor.stderr.log').open('x') as err:
        process = subprocess.Popen(argv, env=environment, stdout=out, stderr=err)
    failed = True
    monitor_failed = False
    try:
        deadline = time.monotonic() + 15
        while not ready.exists():
            if process.poll() is not None: raise RuntimeError('NVML collector failed before model loading')
            if time.monotonic() >= deadline: raise TimeoutError('NVML collector readiness timeout')
            time.sleep(.05)
        yield devices
        failed = False
    finally:
        stop.touch()
        try:
            code = process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.terminate()
            try: code = process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                code = process.wait(timeout=5)
            monitor_failed = True
        monitor_failed = monitor_failed or code != 0
        # The successful parent aggregates the immutable rank profiles later.
        # Failures still receive a partial memory artifact before re-raising.
        finalize_memory_profile(run, expected_sp=plan['sp'], worker_returncode=1 if failed else 0)
        if monitor_failed and not failed:
            raise RuntimeError('NVML collector failed; partial memory evidence retained')

if __name__ == '__main__':
    import argparse
    import signal
    def interrupted(signum, frame):
        raise MonitorInterrupted(f'memory monitor received signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--gpu-uuids', nargs='+', required=True)
    parser.add_argument('--stop', required=True)
    parser.add_argument('--ready', required=True)
    parser.add_argument('--interval-ms', type=int, default=100)
    args = parser.parse_args()
    monitor(args.run, args.gpu_uuids, args.stop, args.ready, args.interval_ms)
