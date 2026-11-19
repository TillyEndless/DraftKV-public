"""Offline descriptive statistics and conservative A/B/C evidence aggregation."""
from collections import defaultdict
from decimal import Decimal
import math
import statistics

from .evidence import duration, intersection, validate
from .schema import normalize


def ms(ns):
    return float(ns / Decimal(1000000))


def distribution(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return {'count': 0, 'median': None, 'p90': None}
    # Nearest-rank P90; no hidden interpolation or averaging of medians.
    return {'count': len(values), 'median': statistics.median(values),
            'p90': values[max(0, math.ceil(.9 * len(values)) - 1)]}


def metrics(record, validation):
    timeline = normalize(record)
    s, dma = timeline['stamps'], timeline['dma']
    consumer = [(s['consumer_kernel_begin'], s['consumer_kernel_end'])]
    overlap = duration(intersection(dma, consumer))
    active_overlap = (None if timeline['active'] is None else
                      ms(duration(intersection(dma, timeline['active']))))
    waits = (ms(duration(timeline['waits'])) / ms(duration(consumer))
             if validation['frontier_wait_observed'] and duration(consumer) else None)
    return {
        'interval_overlap_ms': ms(overlap),
        'overlap_basis': 'dma_activity_union' if 'gpu_h2d_intervals' in record else 'cuda_event_envelope',
        'h2d_ms': ms(duration(dma)),
        'h2d_envelope_ms': ms(s['gpu_h2d_end'] - s['gpu_h2d_begin']),
        'exposed_h2d_ms': ms(duration(dma) - overlap),
        'exposed_h2d_tail_ms': ms(duration(intersection(dma, [(s['consumer_kernel_end'], s['final_join'])]))),
        'consumer_ms': ms(duration(consumer)),
        'consumer_start_delay_ms': ms(s['consumer_kernel_begin'] - timeline['origin']),
        'producer_plan_ms': ms(s['producer_plan_end'] - s['producer_plan_begin']),
        'pipeline_critical_path_ms': ms(timeline['end'] - timeline['origin']),
        'active_math_overlap_ms': active_overlap,
        'frontier_waiting_fraction': waits,
        'clock_error_bound_ns': float(timeline['clock'].error_ns),
    }


PAIR_FIELDS = ('case_id', 'benchmark', 'backbone', 'algorithm', 'git_sha', 'source_sha',
               'kernel_sha', 'input_sha', 'host_id', 'cuda_device', 'rank', 'world_size',
               'layer', 'chunk', 'denoise_step', 'selected_frame_ids', 'local_ids',
               'salient_ids', 'history_frame_ids', 'pair_id', 'validation_stage')


def paired_savings(streaming, serial, stream_metrics, serial_metrics):
    if streaming['mode'] != 'streaming' or serial['mode'] != 'serial':
        raise ValueError('pair requires serial and streaming modes')
    if streaming['trace_sha256'] == serial['trace_sha256']:
        raise ValueError('same trace reused as serial and streaming')
    if streaming['baseline_reference'] != {key: serial[key] for key in ('run_id', 'sample_id', 'trace_sha256')}:
        raise ValueError('baseline reference mismatch')
    for key in PAIR_FIELDS:
        if streaming[key] != serial[key]:
            raise ValueError('unmatched paired field: ' + key)
    before, after = serial_metrics['pipeline_critical_path_ms'], stream_metrics['pipeline_critical_path_ms']
    return {'serial_ms': before, 'streaming_ms': after, 'saved_ms': before - after,
            'saved_fraction': (before - after) / before if before > 0 else None,
            'consumer_penalty_ms': stream_metrics['consumer_ms'] - serial_metrics['consumer_ms']}


def _statistics(rows):
    keys = [key for row in rows for key, value in row['metrics'].items()
            if isinstance(value, (float, int)) or value is None]
    return {key: distribution([row['metrics'].get(key) for row in rows]) for key in sorted(set(keys))}


def analyze(records, *, artifact_root=None):
    """Multiple runs are allowed only as explicitly identified paired captures.

    A repeated capture may hold many unique samples in one run. Capture SHA or
    trace ID reuse across runs, and duplicate sample ownership, invalidate ALL
    involved records, independent of order. No persistent global state is used.
    """
    records = list(records)
    validations = [validate(r, artifact_root=artifact_root) for r in records]
    ownership = defaultdict(list)
    for i, (record, validation) in enumerate(zip(records, validations)):
        if validation['valid']:
            ownership[('sample', record['run_id'], record['sample_id'])].append(i)
            ownership[('capture_hash', record['trace_sha256'])].append(i)
            ownership[('capture_id', record['trace_id'])].append(i)
            ownership[('run', record['run_id'])].append(i)
    for key, indices in ownership.items():
        if key[0] == 'sample':
            duplicate = len(indices) > 1
        elif key[0] == 'run':
            fields = ('benchmark', 'backbone', 'algorithm', 'git_sha', 'source_sha',
                      'kernel_sha', 'host_id', 'world_size', 'validation_stage', 'mode')
            duplicate = len({tuple(records[i][k] for k in fields) for i in indices}) > 1
        else:
            counterpart = 'trace_id' if key[0] == 'capture_hash' else 'trace_sha256'
            duplicate = (len({records[i]['run_id'] for i in indices}) > 1 or
                         len({records[i][counterpart] for i in indices}) > 1)
        if duplicate:
            for i in indices:
                validations[i]['valid'] = False
                validations[i]['errors'].append('trace/sample reuse: ' + key[0])
                validations[i]['overlap_verified'] = False
                validations[i]['evidence_level'] = 'COMPUTE_OVERLAP_UNPROVEN'
    rows = []
    index = {}
    for record, validation in zip(records, validations):
        record_view = record if isinstance(record, dict) else {}
        row = {key: record_view.get(key) for key in ('run_id', 'sample_id', 'benchmark', 'backbone',
               'case_id', 'layer', 'chunk', 'denoise_step', 'rank', 'world_size', 'mode', 'pair_id')}
        row.update(validation=validation, metrics={})
        if validation['valid']:
            row['metrics'] = metrics(record, validation)
            index[(record['run_id'], record['sample_id'])] = (record, row)
        rows.append(row)
    pairs = []
    for record, row in zip(records, rows):
        if not row['validation']['valid'] or record['mode'] != 'streaming':
            continue
        reference = record['baseline_reference']
        if reference is None:
            row['validation']['warnings'].append('no paired serial baseline; C unavailable')
            continue
        try:
            serial, serial_row = index[(reference['run_id'], reference['sample_id'])]
            pair = paired_savings(record, serial, row['metrics'], serial_row['metrics'])
            pair.update({k: row[k] for k in ('run_id', 'sample_id', 'case_id', 'benchmark', 'backbone',
                        'layer', 'chunk', 'denoise_step', 'rank', 'world_size', 'pair_id')})
            pair['serial_run_id'] = serial['run_id']
            pair['critical_path_verified'] = record['consumer_kind'] == 'b01_streaming' and serial['consumer_kind'] == 'b01_serial' and record['algorithm'] == 'draftkv' and all(r['validation']['interval_evidence_verified']
                                                for r in (row, serial_row)) and all(
                r['validation_stage'] != 'STATIC' and not r.get('historical', False) for r in (record, serial))
            pair['critical_path_shortened'] = pair['critical_path_verified'] and pair['saved_ms'] > 0
            pairs.append(pair)
        except (KeyError, ValueError, TypeError) as exc:
            row['validation']['warnings'].append('pair rejected: ' + str(exc))
    layers = defaultdict(list)
    for row in rows:
        if row['metrics']:
            layers[str(row['layer'])].append(row)
    # Single-machine SP: compare complete rank sets at identical logical scopes.
    groups = defaultdict(list)
    for record, row in zip(records, rows):
        if row['metrics']:
            key = tuple(record[k] for k in ('run_id', 'case_id', 'layer', 'chunk', 'denoise_step', 'mode'))
            groups[key].append((record, row))
    sp = []
    for key, group in groups.items():
        first = group[0][0]
        size = first['world_size']
        ranks = [record['rank'] for record, _ in group]
        complete = (len(group) == size and sorted(ranks) == list(range(size)) and
                    len({r['host_id'] for r, _ in group}) == 1 and
                    len({r['cuda_device']['uuid'] for r, _ in group}) == size and
                    all(r['world_size'] == size for r, _ in group))
        worst = max(group, key=lambda x: x[1]['metrics']['pipeline_critical_path_ms'])
        sp.append({'run_id': key[0], 'case_id': key[1], 'layer': key[2], 'chunk': key[3],
                   'denoise_step': key[4], 'mode': key[5], 'world_size': size,
                   'complete_single_host_sp': complete, 'observed_ranks': sorted(ranks),
                   'slowest_rank': worst[0]['rank'] if complete else None,
                   'critical_path_ms': worst[1]['metrics']['pipeline_critical_path_ms'] if complete else None})
    sp_pairs = []
    paired_groups = defaultdict(list)
    for pair in pairs:
        paired_groups[tuple(pair[k] for k in ('run_id', 'case_id', 'layer', 'chunk', 'denoise_step', 'pair_id'))].append(pair)
    for key, group in paired_groups.items():
        size = group[0]['world_size']
        complete = (len(group) == size and sorted(p['rank'] for p in group) == list(range(size)) and
                    all(p['world_size'] == size for p in group))
        # Require the single-host observations too; never substitute fastest-rank time.
        serial_runs = {p['serial_run_id'] for p in group}
        complete = complete and len(serial_runs) == 1 and all(any(
                    s['run_id'] == run and s['case_id'] == key[1] and
                    (s['layer'], s['chunk'], s['denoise_step']) == key[2:5] and s['complete_single_host_sp']
                    for s in sp) for run in {key[0], *serial_runs})
        before = max(p['serial_ms'] for p in group) if complete else None
        after = max(p['streaming_ms'] for p in group) if complete else None
        sp_pairs.append({'run_id': key[0], 'case_id': key[1], 'layer': key[2], 'chunk': key[3],
                         'denoise_step': key[4], 'pair_id': key[5], 'world_size': size,
                         'complete': complete, 'serial_ms': before, 'streaming_ms': after,
                         'saved_ms': before - after if complete else None,
                         'critical_path_shortened': complete and all(p['critical_path_verified'] for p in group)
                                                    and before > after})
    # A capture span is measured only when its sample origins share a calibrated
    # epoch. Summing layer durations is not a pipeline measurement.
    rank_groups = defaultdict(list)
    for record, row in zip(records, rows):
        if row['metrics']:
            rank_groups[(record['run_id'], record['rank'])].append(record)
    pipeline_ranks = []
    for (run, rank), group in rank_groups.items():
        timelines = [normalize(r) for r in group]
        domains = sorted({t['clock'].target for t in timelines})
        span = (ms(max(t['end'] for t in timelines) - min(t['origin'] for t in timelines))
                if len(domains) == 1 else None)
        pipeline_ranks.append({'run_id': run, 'rank': rank, 'world_size': group[0]['world_size'],
                               'clock_domains': domains, 'scope': 'observed_samples_only',
                               'pipeline_critical_path_ms': span,
                               'warning': None if span is not None else 'independent clock epochs; span unavailable'})
    streaming_rows = [row for row in rows if row['mode'] == 'streaming']
    verified = (bool(streaming_rows) and all(row['validation']['valid'] for row in rows)
                and all(row['validation']['overlap_verified'] for row in streaming_rows)
                and all(item['complete_single_host_sp'] for item in sp if item['mode'] == 'streaming'))
    return {'schema_version': 1, 'kind': 'overlap_summary', 'release_gates': {g: 'UNASSESSED' for g in
            ('STATIC', 'CUDA_GOLDEN', 'REAL_CHECKPOINT_FORWARD', 'VIDEO_E2E', 'OFFICIAL_EVAL')},
            'overlap_verified': verified,
            'statistics_scope': 'all structurally valid input samples; descriptive only; no release acceptance',
            'global': _statistics(rows), 'by_layer': {k: _statistics(v) for k, v in sorted(layers.items())},
            'paired_savings': pairs, 'paired_saved_ms': distribution([p['saved_ms'] for p in pairs]),
            'observed_pipeline_by_rank': pipeline_ranks,
            'single_host_sp': sp, 'single_host_sp_pairs': sp_pairs, 'samples': rows}
