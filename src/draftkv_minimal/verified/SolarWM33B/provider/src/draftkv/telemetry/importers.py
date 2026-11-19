"""Loss-preserving historical imports; no fabricated per-layer CUDA timestamps."""
import csv
import json
from pathlib import Path

from .evidence import duration, intersection, sha256_file
from .schema import number

SERIAL = 'serial_safe_prerope_dma'
STREAMING = 'grouped_frontier_prerope_dma_g4'
CSV_HASH = '27973caa8ca337faca234918764754bc07b53b0a713c7d73004aaf1c4ad41c59'


def import_h200_csv(path, *, source_revision, expected_sha256=CSV_HASH):
    """The actual two-row 2026-10-07 summary. Medians cannot become paired samples.

    C is historical run-level E2E/critical-sum evidence; B remains unproven. Even
    positive historical C is not a new-model release gate or per-sample pairing.
    """
    digest = sha256_file(path)
    if digest != expected_sha256:
        raise ValueError('historical CSV SHA-256 mismatch')
    from .schema import GIT_SHA
    if not isinstance(source_revision, str) or not GIT_SHA.fullmatch(source_revision):
        raise ValueError('source_revision must be a full source revision')
    with Path(path).open(newline='') as stream:
        reader = csv.DictReader(stream)
        headers = reader.fieldnames
        rows = list(reader)
    if len(rows) != 2 or {r['mode'] for r in rows} != {SERIAL, STREAMING}:
        raise ValueError('expected one actual serial and grouped historical row')
    numeric = []
    for row in rows:
        values = {k: number(v) for k, v in row.items() if k != 'mode'}
        if any(v < 0 for v in values.values()):
            raise ValueError('negative historical measurement')
        numeric.append((row['mode'], values))
    by_mode = dict(numeric)
    serial, streaming = by_mode[SERIAL], by_mode[STREAMING]
    for key in ('frames', 'mature_selected_chunks', 'mature_events', 'all_w1_events'):
        if serial[key] != streaming[key] or serial[key] <= 0:
            raise ValueError('unmatched historical workload/count')
    if min(serial['e2e_ms'], serial['all_w1_critical_sum_ms']) <= 0:
        raise ValueError('historical baseline duration must be positive')
    e2e_saved = serial['e2e_ms'] - streaming['e2e_ms']
    critical_saved = serial['all_w1_critical_sum_ms'] - streaming['all_w1_critical_sum_ms']
    historical_pass = digest == CSV_HASH and e2e_saved > 0 and critical_saved > 0
    return {'schema_version': 1, 'kind': 'historical_h200_summary', 'historical': True,
            'source': {'path': Path(path).name, 'sha256': digest, 'source_revision': source_revision, 'revision_kind': 'caller_supplied_archive_reference',
                       'revision_caveat': 'observed archived tree HEAD; tree was dirty; not a clean performance build ID'},
            'original_headers': headers, 'original_rows': rows,
            'evidence_levels': (['HISTORY_CRITICAL_PATH_PASS', 'COMPUTE_OVERLAP_UNPROVEN'] if historical_pass
                                else ['HISTORY_UNVERIFIED', 'COMPUTE_OVERLAP_UNPROVEN']),
            'overlap_verified': False, 'consumer_active_math_intervals': None,
            'active_math_unavailable': 'CSV contains medians and run totals, no observed MMA intervals',
            'historical_overlap_median_ms': float(streaming['overlap_median_ms']),
            'historical_e2e_saved_ms': float(e2e_saved),
            'historical_e2e_saved_fraction': float(e2e_saved / serial['e2e_ms']),
            'historical_critical_sum_saved_ms': float(critical_saved),
            'per_sample_paired_savings': None, 'new_release_acceptance': False}


def import_mac_nsight(path, *, expected_sha256, source_revision):
    """Import analyze_timeline.py's actual per-copy/per-kernel JSON export.

    This preserves range-relative origins. Different ranges are not concatenated
    into one invented clock, and resident pack kernels do not become attention.
    """
    digest = sha256_file(path)
    if digest != expected_sha256:
        raise ValueError('Mac Nsight JSON SHA-256 mismatch')
    from .schema import GIT_SHA
    if not isinstance(source_revision, str) or not GIT_SHA.fullmatch(source_revision):
        raise ValueError('source_revision must be full Git SHA or manifest SHA-256')
    data = json.loads(Path(path).read_text())
    if not isinstance(data, list) or not data:
        raise ValueError('expected nonempty Nsight timeline export list')
    ranges = []
    seen = set()
    for row in data:
        name = row['range']
        if name in seen:
            raise ValueError('duplicate replay range')
        seen.add(name)
        dma = [(number(c['start_ms']), number(c['end_ms'])) for c in row['dma']]
        attention = [k for k in row['kernels'] if any(token in k['name'] for token in
                     ('flash_fwd_kernel', 'b01_ready_m64', 'b01_frontier_opt'))]
        if len(attention) != 1 or not dma:
            raise ValueError('need exactly one consumer and actual DMA intervals')
        kernel = attention[0]
        consumer = [(number(kernel['start_ms']), number(kernel['end_ms']))]
        all_intervals = dma + [(number(k['start_ms']), number(k['end_ms'])) for k in row['kernels']]
        if any(a < 0 or b < a for a, b in all_intervals):
            raise ValueError('negative Nsight interval')
        overlap = duration(intersection(dma, consumer))
        if abs(overlap - number(row['dma_kernel_interval_intersection_ms'])) > number('.000001'):
            raise ValueError('Nsight summary overlap inconsistent with raw intervals')
        if row['dma_copy_count'] != len(dma) or row['attention_kernel_count'] != 1:
            raise ValueError('Nsight event count mismatch')
        ranges.append({'range': name, 'clock_domain': 'nsight_range:' + name, 'event_unit': 'ms',
                       'interval_overlap_ms': float(overlap), 'dma_active_union_ms': float(duration(dma)),
                       'consumer_ms': float(duration(consumer)),
                       'critical_device_activity_span_ms': float(max(b for _, b in all_intervals) -
                                                                  min(a for a, _ in all_intervals)),
                       'consumer_kind': 'native_fa2' if 'flash_fwd_kernel' in kernel['name'] else 'b01_streaming',
                       'consumer_active_math_intervals': None,
                       'frontier_waiting_fraction': None, 'original': row})
    return {'schema_version': 1, 'kind': 'historical_mac_nsight', 'historical': True,
            'source': {'path': Path(path).name, 'sha256': digest, 'source_revision': source_revision, 'revision_kind': 'caller_supplied_archive_reference'},
            'evidence_levels': ['INTERVAL_OVERLAP_OBSERVED', 'COMPUTE_OVERLAP_UNPROVEN'],
            'overlap_verified': False, 'new_release_acceptance': False, 'ranges': ranges}
