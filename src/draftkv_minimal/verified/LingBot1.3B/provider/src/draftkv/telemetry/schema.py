"""Portable B01 observation schema. No torch import, clock reads, or GPU calls."""
from copy import deepcopy
from decimal import Decimal, InvalidOperation
import re

SCHEMA_VERSION = 1
GATES = ('STATIC', 'CUDA_GOLDEN', 'REAL_CHECKPOINT_FORWARD', 'VIDEO_E2E', 'OFFICIAL_EVAL')
UNITS = {'ns': Decimal(1), 'us': Decimal(1000), 'ms': Decimal(1000000), 's': Decimal(1000000000)}
STAMPS = ('producer_plan_begin', 'producer_plan_end', 'gpu_h2d_begin', 'gpu_h2d_end',
          'frontier_ready', 'consumer_kernel_begin', 'consumer_kernel_end', 'final_join')
REQUIRED = ('run_id', 'sample_id', 'case_id', 'benchmark', 'backbone', 'algorithm',
            'git_sha', 'source_sha', 'kernel_sha', 'input_sha', 'host_id', 'cuda_device',
            'rank', 'world_size', 'layer', 'chunk', 'denoise_step', 'selected_frame_ids',
            'local_ids', 'salient_ids', 'history_frame_ids', 'producer_plan_id',
            'consumer_kind', 'kernel_symbol', 'mode', 'pair_id', 'baseline_reference',
            'clock_domain', 'event_unit', 'trace_source', 'trace_id', 'trace_sha256',
            'validation_stage', 'consumer_active_math_intervals', 'active_math_unavailable',
            'critical_path', *STAMPS)
SHA256 = re.compile(r'^[0-9a-f]{64}$')
GIT_SHA = re.compile(r'^[0-9a-f]{40,64}$')


def number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float, str, Decimal)):
        raise ValueError('timestamp must be a finite number')
    try:
        out = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError('timestamp must be numeric') from exc
    if not out.is_finite():
        raise ValueError('timestamp must be finite')
    return out


class Clock:
    """Affine mappings are recorded, never inferred from CPU/CUDA ordering.

    Values remain Decimal until differences are taken; large absolute ns origins
    do not lose precision. Each calibration maps directly to one target domain.
    """
    def __init__(self, record):
        self.domain = record['clock_domain']
        self.unit = record['event_unit']
        if not isinstance(self.domain, str) or not self.domain or self.unit not in UNITS:
            raise ValueError('invalid clock_domain/event_unit')
        self.target = record.get('canonical_clock_domain', self.domain)
        if not isinstance(self.target, str) or not self.target:
            raise ValueError('invalid canonical_clock_domain')
        self.calibrations = record.get('clock_calibrations', {})
        if not isinstance(self.calibrations, dict):
            raise ValueError('clock_calibrations must be an object')
        self.used = set()
        self.error_ns = Decimal(0)

    def ns(self, stamp):
        if isinstance(stamp, dict):
            domain = stamp.get('clock_domain', self.domain)
            unit = stamp.get('event_unit', self.unit)
            value = stamp.get('value')
        else:
            domain, unit, value = self.domain, self.unit, stamp
        if unit not in UNITS or not isinstance(domain, str):
            raise ValueError('invalid timestamp domain/unit')
        value = number(value)
        self.used.add(domain)
        if domain == self.target:
            return value * UNITS[unit]
        cal = self.calibrations.get(domain)
        if not isinstance(cal, dict) or cal.get('target_domain') != self.target:
            raise ValueError('MIXED_CLOCK: missing direct clock calibration')
        if not cal.get('method') or not cal.get('source'):
            raise ValueError('MIXED_CLOCK: missing calibration provenance')
        if cal.get('event_unit') != unit:
            raise ValueError('MIXED_CLOCK: calibration unit mismatch')
        scale = number(cal['scale_ns_per_unit'])
        error = number(cal['max_error_ns'])
        if scale <= 0 or error < 0:
            raise ValueError('invalid calibration scale/error')
        self.error_ns = max(self.error_ns, error)
        return value * scale + number(cal['offset_ns'])

    def intervals(self, values):
        if not isinstance(values, list):
            raise ValueError('intervals must be a list')
        result = []
        for interval in values:
            if not isinstance(interval, (list, tuple)) or len(interval) != 2:
                raise ValueError('interval requires [begin, end]')
            begin, end = map(self.ns, interval)
            if end < begin:
                raise ValueError('negative interval duration')
            result.append((begin, end))
        return result


def normalize(record):
    clock = Clock(record)
    stamps = {key: clock.ns(record[key]) for key in STAMPS}
    for begin, end in (('producer_plan_begin', 'producer_plan_end'),
                       ('gpu_h2d_begin', 'gpu_h2d_end'),
                       ('consumer_kernel_begin', 'consumer_kernel_end')):
        if stamps[end] < stamps[begin]:
            raise ValueError('negative duration: ' + begin)
    if not stamps['gpu_h2d_begin'] <= stamps['frontier_ready'] <= stamps['gpu_h2d_end']:
        raise ValueError('frontier_ready outside H2D envelope')
    if record['mode'] == 'serial' and stamps['consumer_kernel_begin'] < stamps['gpu_h2d_end']:
        raise ValueError('serial consumer begins before producer completion')
    if stamps['final_join'] < max(stamps['gpu_h2d_end'], stamps['consumer_kernel_end'],
                                  stamps['producer_plan_end']):
        raise ValueError('final_join precedes completion')
    critical = record['critical_path']
    if not isinstance(critical, dict) or critical.get('definition') != 'pipeline_origin_to_final_join':
        raise ValueError('critical_path needs pipeline origin and definition')
    origin = clock.ns(critical['begin'])
    end = clock.ns(critical['end'])
    if end != stamps['final_join'] or origin > min(stamps.values()):
        raise ValueError('critical_path must enclose pipeline and end at final_join')
    dma = clock.intervals(record.get('gpu_h2d_intervals', [
        [record['gpu_h2d_begin'], record['gpu_h2d_end']]]))
    math_intervals = record['consumer_active_math_intervals']
    active = None if math_intervals is None else clock.intervals(math_intervals)
    wait = record.get('frontier_wait_intervals')
    waits = None if wait is None else clock.intervals(wait)
    if waits is not None and record.get('frontier_wait_observation_scope') != 'whole_consumer_stalled':
        raise ValueError('frontier waits require whole_consumer_stalled observation scope')
    for a, b in dma:
        if not stamps['gpu_h2d_begin'] <= a <= b <= stamps['gpu_h2d_end']:
            raise ValueError('DMA activity outside H2D envelope')
    for a, b in (active or []) + (waits or []):
        if not stamps['consumer_kernel_begin'] <= a <= b <= stamps['consumer_kernel_end']:
            raise ValueError('math/wait outside consumer')
    return {'stamps': stamps, 'dma': dma, 'active': active, 'waits': waits,
            'origin': origin, 'end': end, 'clock': clock}


def structural_errors(record):
    errors = []
    if not isinstance(record, dict):
        return ['record must be an object']
    if type(record.get('schema_version')) is not int or record['schema_version'] != SCHEMA_VERSION:
        errors.append('unsupported schema_version')
    errors += ['missing field: ' + name for name in REQUIRED if name not in record or
               (record[name] is None and name not in ('baseline_reference', 'consumer_active_math_intervals'))]
    if errors:
        return errors
    for name in ('run_id', 'sample_id', 'case_id', 'benchmark', 'backbone', 'algorithm',
                 'host_id', 'producer_plan_id', 'kernel_symbol', 'trace_id', 'trace_source', 'pair_id'):
        if not isinstance(record[name], str) or not record[name].strip():
            errors.append('invalid ' + name)
    for name in ('source_sha', 'kernel_sha', 'input_sha', 'trace_sha256'):
        if not isinstance(record[name], str) or not SHA256.fullmatch(record[name]):
            errors.append('invalid ' + name)
    if not isinstance(record['git_sha'], str) or not GIT_SHA.fullmatch(record['git_sha']):
        errors.append('invalid git_sha')
    if record['validation_stage'] not in GATES:
        errors.append('invalid validation_stage')
    if record['consumer_kind'] not in ('b01_streaming', 'b01_serial', 'native_fa2'):
        errors.append('invalid consumer_kind')
    if record['mode'] not in ('serial', 'streaming'):
        errors.append('invalid mode')
    for name in ('rank', 'world_size', 'layer', 'chunk', 'denoise_step'):
        if type(record[name]) is not int or record[name] < (1 if name == 'world_size' else 0):
            errors.append('invalid ' + name)
    if type(record['rank']) is int and type(record['world_size']) is int and record['rank'] >= record['world_size']:
        errors.append('rank outside world_size')
    device = record['cuda_device']
    if not isinstance(device, dict) or not all(isinstance(device.get(k), str) and device[k]
                                               for k in ('name', 'uuid')):
        errors.append('cuda_device needs name and UUID')
    lists = ('selected_frame_ids', 'local_ids', 'salient_ids', 'history_frame_ids')
    for name in lists:
        ids = record[name]
        if (not isinstance(ids, list) or any(type(x) is not int or x < 0 for x in ids)
                or len(ids) != len(set(ids))):
            errors.append('invalid/duplicate ' + name)
    if not any('ids' in e for e in errors):
        history = set(record['history_frame_ids'])
        if any(not set(record[name]) <= history for name in lists[:3]):
            errors.append('frame ID outside history')
        if not set(record['salient_ids']) <= set(record['selected_frame_ids']):
            errors.append('salient_ids must be selected')
    if not isinstance(record['active_math_unavailable'], str):
        errors.append('active_math_unavailable must be a reason string')
    if record['consumer_active_math_intervals'] is None and not record['active_math_unavailable']:
        errors.append('missing active_math_unavailable reason')
    if record['consumer_active_math_intervals'] is not None and record['active_math_unavailable']:
        errors.append('active math cannot also be unavailable')
    reference = record['baseline_reference']
    if reference is not None and (not isinstance(reference, dict)
            or any(not isinstance(reference.get(k), str) or not reference[k]
                   for k in ('run_id', 'sample_id', 'trace_sha256'))
            or not SHA256.fullmatch(reference['trace_sha256'])):
        errors.append('invalid baseline_reference')
    # Success is an analyzer output, never a producer assertion.
    for name in ('overlap_verified', 'success', 'passed', 'status', 'evidence_level'):
        if name in record:
            errors.append('producer success claim forbidden: ' + name)
    if not errors:
        try:
            normalize(record)
        except (ValueError, KeyError, TypeError, AttributeError, InvalidOperation) as exc:
            errors.append(str(exc))
    return errors


class TraceBuilder:
    """Callers resolve CUDA events externally, then supply their observed values.

    mark() never substitutes perf_counter for a CUDA event. One sample describes
    one layer/chunk/step/rank; unavailable observations stay explicit.
    """
    def __init__(self, metadata):
        self.record = deepcopy(metadata)
        self.record.setdefault('schema_version', SCHEMA_VERSION)
        self.record.setdefault('consumer_active_math_intervals', None)
        self.record.setdefault('active_math_unavailable', 'not observed')
        self.record.setdefault('baseline_reference', None)

    def mark(self, name, value, *, clock_domain=None, event_unit=None):
        if name not in STAMPS:
            raise ValueError('unknown event: ' + name)
        if name in self.record:
            raise ValueError('event already recorded: ' + name)
        self.record[name] = (value if clock_domain is None and event_unit is None else
                             {'value': value, 'clock_domain': clock_domain or self.record['clock_domain'],
                              'event_unit': event_unit or self.record['event_unit']})
        return self

    def finish(self, critical_path):
        record = deepcopy(self.record)
        record['critical_path'] = deepcopy(critical_path)
        errors = structural_errors(record)
        if errors:
            raise ValueError('; '.join(errors))
        return record
