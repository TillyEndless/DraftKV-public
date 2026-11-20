"""CPU-only historical observations, never promoted to authenticated metric cache."""
from pathlib import Path
from .quality import QUALITY_DIMENSIONS, quality_fields, finite, UPSTREAM_COMMIT

HISTORICAL_SCORE_VERSION = "vbench-normalization:" + UPSTREAM_COMMIT + ":quality-7d-percent/historical-metric-protocol-unverified"
from .quality_pipeline import read_reference, reference, canonical_sha, audit_quality_summary
from .quality_report import export_report
from .official import EvaluationError
from draftkv.benchmarks.datasets import sha256_file, unique_json


def export_inventory(inventory, output, *, sidecars=()):
    """Full seven-dimensional *single-run* records may be normalized as observations.

    Disjoint historical files and new sidecars are displayed but cannot supply a
    complete Quality Score without matching source/runtime/weight certificates.
    """
    supplemental={}
    for path in sidecars:
        summary=unique_json(Path(path).read_bytes())
        snapshot=read_reference(summary['provenance_reference'])
        audit_quality_summary(summary,snapshot['evaluation_config'],snapshot['videos'])
        for case in summary['cases']:
            if finite(case['scores'].get('temporal_flickering')):
                supplemental[case['video_sha256']]=dict(value=case['scores']['temporal_flickering'],summary=reference(path))
    groups={};occupied={};rejected=[]
    rows=sorted(inventory['rows'],key=lambda r:(-len(r.get('existing_dimensions',[])),str(r.get('case_id')),r['video_sha256']))
    for row in rows:
        if not row.get('case_id') or row.get('method')=='unclassified':continue
        available=[Path(p) for p in row['video_paths'] if Path(p).is_file()]
        if not available or sha256_file(available[0])!=row['video_sha256']:
            rejected.append(dict(video_sha256=row['video_sha256'],reason='source absent or changed'));continue
        records=[]
        for score in row.get('scores',[]):
            if score.get('input_video_sha256')!=row['video_sha256']:continue
            try:
                raw=read_reference(dict(path=score['path'],sha256=score['sha256']))
                values={d:raw[d][0] for d in QUALITY_DIMENSIONS if d in raw and isinstance(raw[d],list) and raw[d]}
                if not values or not all(finite(v) for v in values.values()):continue
                records.append((values,reference(score['path'])))
            except (OSError,ValueError,KeyError,TypeError) as exc:
                rejected.append(dict(path=score['path'],reason=str(exc)))
        if not records and row['video_sha256'] not in supplemental:continue
        full=next(((s,r) for s,r in records if set(s)==set(QUALITY_DIMENSIONS)),None)
        values=dict(full[0] if full else max((s for s,_ in records),key=len,default={}))
        if not full:
            # Preserve all already observed dimensions for display. This union
            # never establishes Quality eligibility or enters metric cache.
            for observed,_ in sorted(records,key=lambda item:len(item[0]),reverse=True):
                for dimension,value in observed.items(): values.setdefault(dimension,value)
        # Show a new TF sidecar, but do not pretend two protocols form one score.
        if row['video_sha256'] in supplemental and 'temporal_flickering' not in values:
            values['temporal_flickering']=supplemental[row['video_sha256']]['value']
        name=row['method'];key=(row['backbone'],name)
        old=occupied.setdefault(key,{})
        if row['case_id'] in old and old[row['case_id']]!=row['video_sha256']:
            name += ' [variant '+row['video_sha256'][:8]+']';key=(row['backbone'],name)
        else:old[row['case_id']]=row['video_sha256']
        group=groups.setdefault(key,dict(backbone=row['backbone'],method=name,
            dataset_sha256=canonical_sha(dict(historical_inventory=inventory.get('schema'),backbone=row['backbone'])),
            evaluator_sha256=canonical_sha(dict(historical_unverified_method=name,backbone=row['backbone'])),
            scope='historical-observation',protocol_verified=False,comparison_eligible=False,
            quality_score_version=HISTORICAL_SCORE_VERSION,
            provenance_note='Single-file 7D observations can be normalized. Source/runtime/weight identity across methods is unproven; no paired claim. Legacy 6D plus a new TF sidecar remains incompatible, Quality Score null.',cases=[]))
        case=dict(case_id=row['case_id'],video_sha256=row['video_sha256'],scores=values,
            quality_eligible=bool(full),dimension_status={d:'complete' if finite(values.get(d)) else 'missing' for d in QUALITY_DIMENSIONS},
            historical_raw_results=[r for _,r in records],temporal_sidecar=supplemental.get(row['video_sha256']))
        fields=quality_fields(values)
        fields['quality_score_version']=HISTORICAL_SCORE_VERSION
        if not full:fields.update(quality_score=None,quality_score_status='incomplete')
        case.update(fields);group['cases'].append(case)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    from .vbench_long import write_new
    write_new(output/'historical_observations.json',dict(groups=list(groups.values()),rejected=rejected))
    rows=export_report(list(groups.values()),output)
    return dict(status='COMPLETE',historical_only=True,protocol_verified=False,output=str(output),rows=rows,rejected=rejected)
