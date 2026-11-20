"""Scheduler-issued diagnostic allocations backed by live shared-broker leases.

No queue dispatcher, model gate or CUDA code runs in this supervisor. Workers
retain their own ABI/shape/numerical gates. Receipts are issued only after the
atomic Resources acquisition and are revalidated against both authorities.
"""
from datetime import datetime, timezone
import os
from pathlib import Path
import signal
import subprocess
import time

from .common import atomic_json, digest, matches, process_identity, read_json
from .resources import Resources, ResourceBlocked
from .runner import owned_processes, stop_owned
from .store import Queue


def utc(epoch):
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat()


def _broker_expiry(record):
    value = record.get('expires_at', record.get('expires'))
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ResourceBlocked('Underlying Broker lease expiry missing or invalid')
    return float(value)


def verify_live(receipt):
    if receipt.get('bridge_schema') != 1 or not matches(receipt.get('supervisor')):
        raise ResourceBlocked('live Scheduler supervisor identity missing')
    if (len(receipt['device_uuids']) != len(set(receipt['device_uuids']))
            or set(receipt['device_uuids']) != {g['uuid'] for g in receipt['devices']}):
        raise ResourceBlocked('Receipt device set differs from actual allocation')
    broker = receipt['broker']
    if digest(broker['path']) != broker['sha256']:
        raise ResourceBlocked('Broker source identity changed')
    import sqlite3
    db = sqlite3.connect('file:' + receipt['scheduler_database'] + '?mode=ro', uri=True)
    try:
        rows = db.execute('SELECT gpu,owner,external_id,task_id FROM leases WHERE token=?',
                          (receipt['attempt_token'],)).fetchall()
    finally:
        db.close()
    expected = {(str(g['index']), broker['owner'], broker['lease_id'], receipt['job_id'])
                for g in receipt['devices']}
    if set(rows) != expected:
        raise ResourceBlocked('Scheduler allocation no longer owned')
    leases = read_json(broker['lease_file'])
    for gpu in receipt['devices']:
        current = leases.get(str(gpu['index']), {})
        recorded_uuid = current.get('uuid')
        if (current.get('owner') != broker['owner'] or current.get('lease_id') != broker['lease_id']
                or (recorded_uuid is not None and recorded_uuid != gpu['uuid'])
                or _broker_expiry(current) <= time.time()):
            raise ResourceBlocked('Underlying Broker lease expired or ownership/UUID changed')
    return True


def issue(queue, policy, job_id, token, devices, purposes, supervisor):
    rows = list(queue.db.execute('SELECT DISTINCT owner,external_id FROM leases WHERE token=?', (token,)))
    if len(rows) != 1:
        raise ResourceBlocked('One atomic Broker allocation required')
    leases = read_json(policy['legacy_leases'])
    expiration = min(_broker_expiry(leases[str(g['index'])]) for g in devices)
    receipt = dict(bridge_schema=1, issuer='unique-gpu-scheduler', status='allocated', job_id=job_id,
                   attempt_token=token, device_uuids=[g['uuid'] for g in devices], devices=devices,
                   purposes=purposes, expires_utc=utc(expiration), issued_utc=utc(time.time()),
                   supervisor=supervisor, scheduler_database=str(queue.root / 'queue.sqlite3'),
                   broker=dict(**policy['broker'], lease_file=policy['legacy_leases'],
                               owner=rows[0]['owner'], lease_id=rows[0]['external_id']))
    verify_live(receipt)
    queue.audit(job_id, 'DIAGNOSTIC_RECEIPT_ISSUED', receipt)
    return receipt


def execute(workspace, policy, spec):
    """Run one bounded, explicitly specified command, renewing until owned exit."""
    workspace = Path(workspace).resolve()
    if workspace.exists():
        raise ValueError('Use a new isolated allocation directory')
    queue = Queue(workspace)
    benchmark = spec.get('benchmark', 'bench1')
    if benchmark not in ('bench1', 'bench2'):
        raise ValueError('diagnostic benchmark must be bench1 or bench2')
    case_id = spec.get('case_id', 'PVB_004' if benchmark == 'bench1' else 'case_0051')
    config = dict(benchmark=benchmark, backbone=spec['backbone'], algorithm=spec.get('algorithm', 'draftkv'),
                  case_id=case_id, manifest_sha256=spec['manifest_sha256'], sp=spec['sp'],
                  memory=spec['memory'])
    if benchmark == 'bench2':
        selection_path = spec.get('selection')
        manifest_path = spec.get('manifest')
        if not selection_path or not manifest_path:
            raise ValueError('Bench2 diagnostic requires frozen manifest and selection paths')
        selection = read_json(selection_path)
        config.update(manifest=manifest_path, selection=selection_path,
                      selection_sha256=digest(selection_path),
                      expected_frames=357, fps=16,
                      pilot=case_id in selection.get('case_ids', ()),
                      seed=selection.get('generation_seed'))
    job = queue.submit(config)
    queue.transition(job, 'PREFLIGHT'); queue.transition(job, 'READY')
    token = queue.claim(job)
    resource = Resources(queue, policy)
    started = time.monotonic()
    receipt_path = workspace / 'lease.json'
    result = dict(status='BLOCKED', job_id=job, source_commit=spec['source_commit'])
    held = False
    child = None
    interrupted = [None]
    previous = {}
    def on_signal(signum, frame):
        interrupted[0] = signum
    for sig in (signal.SIGINT, signal.SIGTERM):
        previous[sig] = signal.signal(sig, on_signal)
    try:
        devices = resource.acquire(config, job, token)
        held = True
        receipt = issue(queue, policy, job, token, devices, spec['purposes'], process_identity(os.getpid()))
        atomic_json(receipt_path, receipt)
        atomic_json(workspace / 'allocation.json', receipt)
        argv = [arg.replace('{devices}', ','.join(g['uuid'] for g in devices)).replace('{lease}', str(receipt_path))
                for arg in spec['argv']]
        environment = dict(os.environ, DRAFTKV_ATTEMPT_TOKEN=token, PYTHONNOUSERSITE='1')
        atomic_json(workspace / 'command.json', dict(argv=argv, cwd=spec['cwd'], source_commit=spec['source_commit']))
        with (workspace / 'worker.log').open('wb') as log:
            child = subprocess.Popen(argv, cwd=spec['cwd'], env=environment, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            renewed = time.monotonic()
            while child.poll() is None:
                if interrupted[0] or time.monotonic() - started > spec['timeout_s']:
                    raise RuntimeError('Diagnostic interrupted or bounded timeout reached')
                # Host-memory floors are non-gating; runtime OOM is recorded instead.
                verify_live(read_json(receipt_path))
                if time.monotonic() - renewed > 60:
                    resource.renew(token)
                    receipt = issue(queue, policy, job, token, devices, spec['purposes'], process_identity(os.getpid()))
                    atomic_json(receipt_path, receipt)
                    renewed = time.monotonic()
                time.sleep(1)
        result.update(status='WORKER_EXITED', exit_code=child.wait(), devices=devices)
    except BaseException as exc:
        result['error'] = type(exc).__name__ + ': ' + str(exc)
    finally:
        try:
            # First ask the exact direct child to exit. torchrun propagates TERM
            # to its own ranks; this avoids invoking descendant cleanup at all
            # on the normal failure path.
            if child is not None and child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=5)
            if owned_processes(token):
                stop_owned(token)
            if child is not None:
                child.wait(timeout=15)
            if owned_processes(token):
                raise RuntimeError('owned workers remain; retain lease')
            if held:
                resource.release(token)
                if receipt_path.exists():
                    receipt = read_json(receipt_path)
                    receipt['status'] = 'released'
                    atomic_json(receipt_path, receipt)
                result['lease_released_after_owned_exit'] = True
        except BaseException as exc:
            result.update(cleanup_error=str(exc), status='CLEANUP_FAILED_LEASE_RETAINED')
        result['elapsed_s'] = time.monotonic() - started
        atomic_json(workspace / 'execution.json', result)
        queue.db.close()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return result
