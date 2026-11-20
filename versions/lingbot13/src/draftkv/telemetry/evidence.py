"""Hash- and sample-bound evidence checks. No success flag is trusted."""
import hashlib
import json
from pathlib import Path
import re

from .schema import Clock, normalize, structural_errors


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_artifact(reference, root):
    if root is None or not isinstance(reference, dict):
        raise ValueError('evidence requires an explicit artifact root and reference')
    name = reference.get('path')
    if (not isinstance(name, str) or not name or '\\' in name or ':' in name
            or Path(name).is_absolute() or '..' in Path(name).parts):
        raise ValueError('artifact path must be portable and root-relative')
    root = Path(root).resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError('artifact symlink escapes root')
    if sha256_file(path) != reference.get('sha256'):
        raise ValueError('artifact SHA-256 mismatch')
    return path


def union(intervals):
    out = []
    for begin, end in sorted(intervals):
        if out and begin <= out[-1][1]:
            out[-1] = (out[-1][0], max(end, out[-1][1]))
        elif end > begin:
            out.append((begin, end))
    return out


def duration(intervals):
    return sum((b - a for a, b in union(intervals)), 0)


def intersection(left, right):
    return union([(max(a, c), min(b, d)) for a, b in union(left)
                  for c, d in union(right) if min(b, d) > max(a, c)])


def _binding(data, record):
    expected = {key: record[key] for key in ('run_id', 'trace_id', 'git_sha', 'source_sha', 'kernel_sha')}
    expected.update(device_uuid=record['cuda_device']['uuid'], rank=record['rank'])
    if data.get('draftkv_binding') != expected:
        raise ValueError('evidence binding mismatch (run/source/kernel/device/rank)')


def _trace(record, root):
    reference = record.get('evidence', {}).get('nsight_trace')
    path = load_artifact(reference, root)
    if reference['sha256'] != record['trace_sha256']:
        raise ValueError('trace SHA does not match sample')
    data = json.loads(path.read_text())
    _binding(data, record)
    if data.get('trace_source') not in ('nsight_systems', 'nsight_compute_correlated'):
        raise ValueError('real Nsight trace source required')
    if data['trace_source'] != record['trace_source']:
        raise ValueError('trace source mismatch')
    if data.get('event_unit') not in ('ns', 'us', 'ms') or not data.get('clock_domain'):
        raise ValueError('trace clock metadata required')
    timeline = normalize(record)
    clock = Clock(record)
    events = data.get('traceEvents')
    if not isinstance(events, list):
        raise ValueError('traceEvents required; a summary is insufficient')
    dma, kernels, math, waits, pipeline, ready = [], [], [], [], [], []
    for event in events:
        if not isinstance(event, dict):
            raise ValueError('malformed trace event')
        args = event.get('args', {})
        if args.get('sample_id') != record['sample_id'] or event.get('ph') != 'X':
            continue
        cat = event.get('cat')
        if cat not in ('cuda_gpu_memcpy', 'cuda_gpu_kernel', 'cuda_gpu_mma_active', 'cuda_gpu_frontier_wait', 'cuda_gpu_pipeline', 'cuda_gpu_frontier_ready'):
            continue
        # Exported GPU intervals must retain their capture's domain, not host NVTX time.
        def stamp(value):
            return {'value': value, 'clock_domain': data['clock_domain'], 'event_unit': data['event_unit']}
        from .schema import number
        a = clock.ns(stamp(event['ts']))
        b = clock.ns(stamp(number(event['ts']) + number(event['dur'])))
        if b < a:
            raise ValueError('negative Nsight duration')
        if cat == 'cuda_gpu_pipeline':
            pipeline.append((a, b))
        elif cat == 'cuda_gpu_frontier_ready':
            ready.append(a)
        elif cat == 'cuda_gpu_memcpy' and args.get('copy_kind') == 'H2D':
            dma.append((a, b))
        elif cat == 'cuda_gpu_kernel' and event.get('name') == record['kernel_symbol']:
            kernels.append((a, b))
        elif cat in ('cuda_gpu_mma_active', 'cuda_gpu_frontier_wait'):
            if args.get('kernel_sha') != record['kernel_sha'] or args.get('kernel_symbol') != record['kernel_symbol']:
                raise ValueError('math/wait evidence kernel mismatch')
            if cat == 'cuda_gpu_frontier_wait' and args.get('scope') != 'whole_consumer_stalled':
                raise ValueError('frontier waiting fraction requires whole-consumer stall evidence')
            if args.get('observation') != 'sass_correlated_gpu_interval':
                raise ValueError('active math/waits need directly observed GPU intervals')
            (math if cat == 'cuda_gpu_mma_active' else waits).append((a, b))
    s = timeline['stamps']
    if pipeline != [(timeline['origin'], timeline['end'])] or ready != [s['frontier_ready']]:
        raise ValueError('pipeline origin/join or frontier not bound to GPU observations')
    if kernels != [(s['consumer_kernel_begin'], s['consumer_kernel_end'])]:
        raise ValueError('consumer interval not bound to exactly one real kernel')
    if not dma or union(dma) != union(timeline['dma']):
        raise ValueError('DMA intervals not bound to real copies')
    if timeline['active'] is not None and union(math) != union(timeline['active']):
        raise ValueError('active math intervals not present in GPU trace')
    if timeline['waits'] is not None and union(waits) != union(timeline['waits']):
        raise ValueError('frontier wait intervals not present in GPU trace')
    return bool(math), bool(waits), max(clock.error_ns, timeline['clock'].error_ns)


def _producer(record, root):
    path = load_artifact(record.get('evidence', {}).get('producer_plan'), root)
    data = json.loads(path.read_text())
    _binding(data, record)
    if data.get('kind') != 'b01_cpp_producer_plan' or data.get('schema_version') != 1:
        raise ValueError('original C++ ProducerPlan evidence required')
    for key in ('sample_id', 'producer_plan_id', 'producer_plan_begin', 'producer_plan_end', 'selected_frame_ids', 'local_ids', 'salient_ids',
                'history_frame_ids', 'input_sha'):
        if data.get(key) != record[key]:
            raise ValueError('ProducerPlan mismatch: ' + key)


def _sass(record, root):
    reference = record.get('evidence', {}).get('sass')
    path = load_artifact(reference, root)
    if reference.get('kernel_sha') != record['kernel_sha']:
        raise ValueError('SASS kernel binary SHA mismatch')
    text = path.read_text()
    # Limit the search to this function. A stock or unrelated kernel's HMMA is insufficient.
    sections = re.split(r'(?=Function\s*:\s*)', text)
    section = next((s for s in sections if s.startswith('Function') and
                    s.splitlines()[0].split(':', 1)[1].strip() == record['kernel_symbol']), '')
    if not re.search(r'\b(?:HMMA|MMA|WGMMA|HGMMA|TCGEN05\.MMA)(?:\.|\s)', section, re.I):
        raise ValueError('SASS lacks MMA instructions in the bound consumer symbol')


def validate(record, *, artifact_root=None, expected_run_id=None):
    """Structural validity and A/B observability are separate from release gates.

    Hash binding detects edits/reuse, not malicious capture authorship. Real
    attestation and provenance remain the release owner's responsibility.
    """
    errors = structural_errors(record)
    if not errors and expected_run_id is not None and record['run_id'] != expected_run_id:
        errors.append('cross-run record')
    result = {'valid': not errors, 'errors': errors, 'warnings': [], 'blockers': [],
              'interval_evidence_verified': False, 'math_evidence_verified': False,
              'frontier_wait_observed': False, 'overlap_verified': False,
              'evidence_level': 'COMPUTE_OVERLAP_UNPROVEN'}
    if errors:
        return result
    timeline = normalize(record)
    if len(timeline['clock'].used) > 1:
        result['warnings'].append('MIXED_CLOCK_CALIBRATED: affine mappings and error bounds retained')
    if record['consumer_kind'] != 'b01_streaming' and record['mode'] == 'streaming':
        result['blockers'].append('consumer is not original B01 streaming (Native FA2 forbidden)')
    if record['algorithm'] != 'draftkv':
        result['blockers'].append('overlap verification is scoped to DraftKV')
    try:
        math_present, waits_present, error_ns = _trace(record, artifact_root)
        _producer(record, artifact_root)
        result['interval_evidence_verified'] = True
        _sass(record, artifact_root)
        result['frontier_wait_observed'] = waits_present and timeline['waits'] is not None
        if timeline['active'] is None or not math_present:
            raise ValueError('consumer active math unavailable; A does not establish B')
        overlap = duration(intersection(timeline['dma'], timeline['active']))
        # Each intersection boundary can be displaced by the calibration error.
        pieces = len(intersection(timeline['dma'], timeline['active']))
        if overlap <= 2 * error_ns * pieces:
            raise ValueError('no positive active-math/DMA intersection above clock uncertainty')
        result['math_evidence_verified'] = True
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        result['blockers'].append(str(exc))
    if record['validation_stage'] == 'STATIC':
        result['blockers'].append('STATIC fixtures cannot verify real CUDA overlap')
    if record['mode'] != 'streaming':
        result['blockers'].append('serial reference does not verify streaming overlap')
    if record.get('historical', False):
        result['blockers'].append('historical evidence cannot accept a new release model')
    result['overlap_verified'] = not result['blockers'] and result['math_evidence_verified']
    if result['overlap_verified']:
        result['evidence_level'] = 'COMPUTE_OVERLAP_VERIFIED'
    return result
