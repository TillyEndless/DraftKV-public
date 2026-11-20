"""CPU-only tables. Never import an evaluation runtime or load metric models."""
from __future__ import annotations
import csv
import io
import json
from pathlib import Path
from .quality import QUALITY_DIMENSIONS, QUALITY_LABEL, quality_fields, finite

COLUMNS = ('Backbone', 'Method', 'Subject', 'Background', 'Flickering', 'Motion', 'Dynamic',
           'Aesthetic', 'Imaging', 'Avg6', 'Quality Score')


def report_rows(groups):
    """Groups must share protocol+dataset to enter the same paired comparison."""
    # CLIP is independently authenticated; its protocol never repartitions or
    # changes the official seven-dimensional paired quality calculation.
    from .segment_clip import audit_attachment, COLUMNS as CLIP_COLUMNS
    from draftkv.benchmarks.worldattention import MANIFEST_SHA256
    clip_protocols = set()
    for group in groups:
        if any(k.startswith('clip_') for k in group):
            from .quality_pipeline import read_reference
            snapshot = read_reference(group['provenance_reference'])
            audit_attachment(group, Path(__file__).resolve().parents[3], snapshot['videos'])
            if group.get('clip_segment_scores') is not None:
                clip_protocols.add(group['clip_protocol_sha256'])
    clip_pairing_blocked = len(clip_protocols) > 1
    partitions = {}
    for group in groups:
        key = (group['backbone'], group['dataset_sha256'], group['evaluator_sha256'],
               group['method'] if not group.get('comparison_eligible',True) else None, group.get('scope'))
        partitions.setdefault(key, []).append(group)
    output = []
    for key, partition in partitions.items():
        complete = [{c['case_id']: c for c in g['cases'] if c.get('quality_eligible',True) and quality_fields(c['scores'])['quality_score'] is not None}
                    for g in partition]
        common = sorted(set.intersection(*(set(c) for c in complete))) if complete else []
        if any(not g.get('comparison_eligible',True) for g in partition): common = []
        native_index = next((i for i, g in enumerate(partition) if g.get('comparison_eligible',True) and g['method'] in ('native', 'Native Baseline', 'baseline')), None)
        for index, g in enumerate(partition):
            cases = g['cases']
            means = {d: (sum(c['scores'][d] for c in cases if finite(c['scores'].get(d))) /
                         sum(finite(c['scores'].get(d)) for c in cases))
                     if any(finite(c['scores'].get(d)) for c in cases) else None for d in QUALITY_DIMENSIONS}
            valid = complete[index]
            q = quality_fields(means)['quality_score'] if len(valid) == len(cases) else None
            paired = sum(quality_fields(valid[c]['scores'])['quality_score'] for c in common) / len(common) if common else None
            native = complete[native_index] if native_index is not None else {}
            pair_ids = sorted(set(valid) & set(native))
            delta = (sum(quality_fields(valid[c]['scores'])['quality_score'] - quality_fields(native[c]['scores'])['quality_score']
                         for c in pair_ids) / len(pair_ids)) if pair_ids else None
            # Avg6 is mean of per-case six-dimensional averages, never a mix of dimension subsets.
            avg6 = [quality_fields(c['scores'])['vbench_long_avg6'] for c in cases]
            avg6 = [v for v in avg6 if v is not None]
            row = dict(zip(COLUMNS, [g['backbone'], g['method'], *[means[d] for d in QUALITY_DIMENSIONS],
                                    sum(avg6) / len(avg6) if avg6 else None, q]))
            row.update(quality_score_scope=g.get('quality_score_scope', 'exploratory-subset'),
                formal_quality_score_percent=g.get('formal_quality_score_percent'),
                vbench_quality_score_normalized=q / 100 if q is not None else None,
                completed_cases=len(valid), total_cases=len(cases), quality_case_ids=sorted(valid),
                paired_common_case_ids=common, paired_common_quality_score=paired,
                native_pair_case_ids=pair_ids, paired_delta_vs_native=delta,
                failed_dimensions=sum(v=='failed' for c in cases for v in c.get('dimension_status',{}).values()),
                missing_dimensions=sum(v=='missing' for c in cases for v in c.get('dimension_status',{}).values()),
                failed_cases=sum(any(v == 'failed' for v in c.get('dimension_status', {}).values()) for c in cases),
                missing_cases=sum(any(not finite(c['scores'].get(d)) for d in QUALITY_DIMENSIONS) for c in cases),
                incompatible_cases=sum(not c.get('quality_eligible',True) for c in cases),
                protocol_verified=g.get('protocol_verified',True), comparison_eligible=g.get('comparison_eligible',True),
                provenance_note=g.get('provenance_note'), dimension_counts={d:sum(finite(c['scores'].get(d)) for c in cases) for d in QUALITY_DIMENSIONS},
                dimension_case_ids={d:sorted(c['case_id'] for c in cases if finite(c['scores'].get(d))) for d in QUALITY_DIMENSIONS},
                dataset_sha256=key[1], evaluator_sha256=key[2],
                quality_score_version=g.get('quality_score_version',quality_fields({})['quality_score_version']),
                code_commit=g.get('code_commit'), input_scope=g.get('scope'), label=QUALITY_LABEL)
            row.update(full_sample_dimension_means=means,
                       full_sample_avg6=sum(avg6)/len(avg6) if avg6 else None, full_sample_quality_score=q)
            if len(partition)>1 and all(g.get('comparison_eligible',True) for g in partition):
                paired_means={d:sum(valid[c]['scores'][d] for c in common)/len(common) if common else None for d in QUALITY_DIMENSIONS}
                for column,dimension in zip(COLUMNS[2:9],QUALITY_DIMENSIONS): row[column]=paired_means[dimension]
                row['Avg6']=quality_fields(paired_means)['vbench_long_avg6']
                row['Quality Score']=paired
                row['table_case_ids']=common
                row['paired_delta_vs_native']=(sum(quality_fields(valid[c]['scores'])['quality_score']-quality_fields(native[c]['scores'])['quality_score'] for c in common)/len(common)) if common and native else None
                row['pairwise_delta_vs_native']=delta
            else:
                row['table_case_ids']=sorted(valid)
            if g['backbone'].lower() == 'longlive1.3b' and g['dataset_sha256'] == MANIFEST_SHA256:
                values = g.get('clip_segment_scores')
                displayed = None if clip_pairing_blocked else values
                row['CLIP Status'] = g.get('clip_status', 'NOT_RUN')
                row['CLIP Scope'] = g.get('clip_scope', 'unavailable')
                row.update(dict(zip(CLIP_COLUMNS, displayed if displayed is not None else [None]*6)))
                row.update(clip_status=g.get('clip_status', 'NOT_RUN'), clip_blocker=g.get('clip_blocker'),
                    clip_scope=g.get('clip_scope', 'unavailable'), clip_completed_cases=g.get('clip_completed_cases', 0),
                    clip_completed_segments=g.get('clip_completed_segments', 0), clip_case_ids=g.get('clip_case_ids', []),
                    clip_protocol_sha256=g.get('clip_protocol_sha256'),
                    clip_exploratory_segment_scores=g.get('clip_exploratory_segment_scores'),
                    clip_comparison_eligible=displayed is not None,
                    clip_full_sample_segment_scores=values,
                    clip_comparison_blocker='Different CLIP protocol/dependency identities; paired values withheld' if clip_pairing_blocked else None,
                    clip_label='Custom OpenAI CLIP cosine × 100; no paper numeric equivalence')
            output.append(row)
    return output


def export_report(groups, output):
    from .vbench_long import write_new
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    rows = report_rows(groups)
    write_new(output / 'quality_table.json', dict(label=QUALITY_LABEL, rows=rows))
    from .segment_clip import COLUMNS as CLIP_COLUMNS
    columns = COLUMNS + CLIP_COLUMNS + ('CLIP Status', 'CLIP Scope') if any(c in row for row in rows for c in CLIP_COLUMNS) else COLUMNS
    stream = io.StringIO(); fields = list(columns) + [k for r in rows for k in r if k not in columns]
    fields = list(dict.fromkeys(fields))
    writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
    for row in rows:
        writer.writerow({k:json.dumps(v, ensure_ascii=False, allow_nan=False) if isinstance(v, (dict,list)) else 'NULL' if k in CLIP_COLUMNS and v is None else v for k,v in row.items()})
    from .vbench_long import atomic_text
    atomic_text(output / 'quality_table.csv', stream.getvalue())
    def render(value):
        if value is None: return '—'
        if isinstance(value, (int,float)): return f'{value:.6f}'
        return str(value).replace('|', '\\|').replace('\n', ' ')
    lines = [QUALITY_LABEL, '', '| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---']*len(columns)) + ' |']
    lines += ['| ' + ' | '.join(render(row.get(c)) for c in columns) + ' |' for row in rows]
    lines += ['', 'Certified multi-method table rows use the common complete seven-dimensional case set. Full sample dimension means, Avg6 and Quality Score are retained in JSON/CSV. A single method or an unverified historical observation shows its available sample means; historical observations have no paired claims. Raw dimensions and Avg6 are fractions; Quality Score and deltas are percentage points. See JSON/CSV for sample sets, common-case means, Native-paired deltas and provenance.', '', 'Long custom-input development protocol; no static filter. This is not the original standard VBench leaderboard protocol.']
    if columns != COLUMNS:
        lines += ['', *[render(row['Backbone']) + ' / ' + render(row['Method']) + ': CLIP blocker: ' + render(row['clip_blocker']) for row in rows if row.get('clip_blocker')]]
        if clip_pairing_blocked := any(row.get('clip_comparison_blocker') for row in rows):
            lines += ['', 'CLIP comparison BLOCKED: different protocol/dependency identities; six paired values are withheld. Independent full-sample CLIP means remain in JSON/CSV. Quality scoring is unchanged.']
        lines += ['', 'CLIP columns use the independent custom protocol (cosine × 100, no clamp). Six formal values require all 160 × 6 successful source-prompt segments. Missing values are null in JSON/CSV; exploratory means and CLIP status/blocker are separate fields. No WorldAttention paper numeric equivalence is claimed.']
    atomic_text(output / 'quality_table.md', '\n'.join(lines) + '\n')
    return rows


def publish_report(repo_root, summary):
    """Append a result event and refresh a shared paired table under a file lock.

    Independent methods/GPU workers cannot overwrite one another's observations.
    Datasets, Long protocols and scopes always get separate report directories.
    """
    import fcntl
    import os
    import uuid
    from datetime import datetime, timezone
    from .vbench_long import canonical_sha, write_new
    from .quality_pipeline import read_reference, reference
    partition = dict(dataset_sha256=summary['dataset_sha256'], evaluator_sha256=summary['evaluator_sha256'], scope=summary['scope'])
    root = Path(repo_root) / 'runs' / 'bench1' / '_quality_reports' / canonical_sha(partition)
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        events = root / 'events'; events.mkdir(exist_ok=True)
        event = dict(created_at=datetime.now(timezone.utc).isoformat(), backbone=summary['backbone'], method=summary['method'],
                     summary=reference(summary['summary_path']))
        write_new(events / (uuid.uuid4().hex + '.json'), event)
        latest = {}
        for path in events.glob('*.json'):
            value = json.loads(path.read_text())
            key = (value['backbone'], value['method'])
            if key not in latest or value['created_at'] > latest[key]['created_at']: latest[key] = value
        groups = []
        for key,value in sorted(latest.items()):
            group = read_reference(value['summary'])
            if any(group[k] != v for k,v in partition.items()): raise ValueError('Cross-protocol quality report event')
            groups.append(group)
        destination = root / ('report-' + uuid.uuid4().hex)
        export_report(groups, destination)
        pointer = root / ('.latest-' + uuid.uuid4().hex + '.json')
        write_new(pointer, dict(report=str(destination), partition=partition))
        os.replace(pointer, root / 'latest.json')
    return str(destination)
