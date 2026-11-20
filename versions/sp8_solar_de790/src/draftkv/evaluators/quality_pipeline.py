"""Per-dimension immutable checkpoints for Bench1, including historical inputs.

No generation and no metric implementation lives here. Each missing dimension
calls the source-pinned official Long CLI through VBenchLong.
"""
from __future__ import annotations
import fcntl
import json
from pathlib import Path
import subprocess
import uuid
from .official import EvaluationError
from .vbench_long import canonical_sha, write_new, parse_batch_result
from .quality import QUALITY_DIMENSIONS, quality_fields, finite, full_benchmark_cohort, cohort_quality_fields
from draftkv.benchmarks.datasets import sha256_file, unique_json

SCHEMA = 'draftkv.bench1.quality.summary.v2'


def reference(path):
    return dict(path=str(Path(path).resolve()), sha256=sha256_file(Path(path)))


def read_reference(ref):
    path = Path(ref['path'])
    if sha256_file(path) != ref['sha256']: raise EvaluationError('Quality evidence SHA mismatch: ' + str(path))
    return unique_json(path.read_bytes())


def verify_record(ref, binding, dimension):
    receipt = read_reference(ref)
    if receipt.get('binding') != binding or receipt.get('dimension') != dimension:
        raise EvaluationError('Quality checkpoint input/protocol binding mismatch')
    if receipt.get('status') != 'complete' or not finite(receipt.get('score')):
        raise EvaluationError('Invalid quality checkpoint score/status')
    if 'legacy' in receipt:
        return _verify_legacy_record(receipt, binding, dimension)
    batch = read_reference(receipt['batch'])
    attempt = Path(receipt['batch']['path']).parent
    if batch['dimension'] != dimension or batch['evaluator_sha256'] != binding['evaluator_sha256']:
        raise EvaluationError('Batch protocol mismatch')
    if set(batch['evidence_sha256']) != {'invocation.json','process.json','stdout.log','stderr.log','video_probe.json'}:
        raise EvaluationError('Incomplete batch process evidence')
    for name, digest in batch['evidence_sha256'].items():
        if sha256_file(attempt / name) != digest: raise EvaluationError('Batch evidence changed')
    invocation = unique_json((attempt / 'invocation.json').read_bytes())
    snapshot = read_reference(batch['protocol_reference'])
    if (snapshot['identity'] != {k:v for k,v in binding.items() if k not in ('evaluator_sha256','video')}
            or canonical_sha(snapshot['preflight']) != binding['evaluator_sha256']
            or canonical_sha(snapshot['evaluation_config']) != binding['evaluation_config_sha256']):
        raise EvaluationError('Batch configuration/protocol binding differs')
    process = unique_json((attempt / 'process.json').read_bytes())
    if process.get('exit_code') != 0: raise EvaluationError('Batch process did not succeed')
    config = snapshot['evaluation_config']
    if config.get('backend') == 'official_vbench_long':
        from .vbench_long import VBenchLong, ENTRY
        backend = VBenchLong(batch['repo_root'], config)
        expected = [str(backend.path(config['python'], resolve_symlinks=False)), '-B',
            str(backend.path(config['official_root']) / ENTRY), '--videos_path', str(attempt / 'input'),
            '--output_path', str(attempt / 'raw'), '--dimension', dimension, '--mode', 'long_custom_input', '--dev_flag']
        environment = backend.environment()
        critical = ('PYTHONPATH','PYTHONNOUSERSITE','PYTHONDONTWRITEBYTECODE','HF_HUB_OFFLINE',
                    'TRANSFORMERS_OFFLINE','VBENCH_CACHE_DIR','TORCH_HOME','EVAL_COMPAT_ENABLE',
                    'EVAL_COMPAT_OVERLAY','DRAFTKV_VBENCH_OFFLINE_HUB')
        if (invocation.get('argv') != expected or invocation.get('cwd') != str(backend.path(config['official_root']))
                or any(invocation.get('environment',{}).get(k) != environment.get(k) for k in critical)):
            raise EvaluationError('Batch official invocation/environment differs')
    aliases = invocation['aliases']
    if [a for a,v in aliases.items() if v == binding['video']] != [receipt['alias']]:
        raise EvaluationError('Batch video alias mismatch')
    for alias, video in aliases.items():
        if sha256_file(attempt / 'input' / (alias + '.mp4')) != video['sha256']:
            raise EvaluationError('Batch copied input changed')
    raw_path = Path(batch['raw_result']['path']).resolve()
    if (not raw_path.is_relative_to((attempt / 'raw').resolve())
            or sha256_file(raw_path) != batch['raw_result']['sha256']):
        raise EvaluationError('Batch raw output changed')
    scores = parse_batch_result(raw_path, attempt / 'input', dimension, aliases)
    if scores[binding['video']['case_id']] != receipt['score']:
        raise EvaluationError('Checkpoint differs from original official result')
    return receipt


def validate_inputs(videos):
    ids = [v['case_id'] for v in videos]
    if not ids or len(set(ids)) != len(ids): raise EvaluationError('Unique nonempty quality case IDs required')
    for video in videos:
        path = Path(video['path'])
        if path.suffix.lower() != '.mp4' or not path.is_file() or path.stat().st_size == 0 or sha256_file(path) != video['sha256']:
            raise EvaluationError('Quality input MP4 absent or SHA mismatch')


def evaluate_quality(evaluator, videos, *, output_dir, dataset_sha256, backbone, method,
                     generation_sha256, metadata=None, scope='historical', mode='missing-only',
                     dimensions=QUALITY_DIMENSIONS, batch_size=16, case_ids=None):
    """All successful metrics survive any other metric's failure.

    full/eval-only also reuse valid checkpoints; no mode destroys prior results.
    The generation part of full is provided by the existing one-click scheduler.
    """
    validate_inputs(videos)
    if mode not in ('full', 'eval-only', 'missing-only', 'aggregate-only'): raise EvaluationError('Unknown evaluation mode')
    if not dimensions or len(set(dimensions)) != len(dimensions) or not set(dimensions) <= set(QUALITY_DIMENSIONS):
        raise EvaluationError('Invalid requested dimensions')
    if not 1 <= batch_size <= 256: raise EvaluationError('batch_size must be 1..256')
    if case_ids is not None and (not case_ids or not set(case_ids) <= {v['case_id'] for v in videos}):
        raise EvaluationError('Selected scoring cases are outside the bound cohort')
    config = evaluator.config
    identity = dict(generation_sha256=generation_sha256, dataset_sha256=dataset_sha256,
        backbone=backbone, method=method, evaluation_config_sha256=canonical_sha(config), metadata=metadata or {})
    base = Path(output_dir).resolve() / ('quality-v2-' + canonical_sha(identity))
    base.mkdir(parents=True, exist_ok=True)
    with (base / '.lock').open('a') as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc: raise EvaluationError('Quality scoring already active for this generation/method') from exc
        if mode == 'aggregate-only':
            # Deliberately no preflight/import probe/model construction. Choose
            # only an existing protocol snapshot for this exact config+generation.
            snapshots = sorted(base.glob('*/provenance.json'), key=lambda p:p.stat().st_mtime_ns, reverse=True)
            if not snapshots:
                provenance = _legacy_snapshot(evaluator, output_dir)
                protocol = canonical_sha(provenance)
            else:
                snapshot = unique_json(snapshots[0].read_bytes())
                if snapshot['identity'] != identity or canonical_sha(snapshot['preflight']) != snapshots[0].parent.name:
                    raise EvaluationError('Saved quality protocol binding changed')
                provenance = snapshot['preflight']; protocol = snapshots[0].parent.name
        else:
            provenance = evaluator.preflight()
            if provenance.get('status') != 'READY': return provenance
            protocol = canonical_sha(provenance)
        output = base / protocol; output.mkdir(exist_ok=True)
        snapshot = dict(identity=identity, preflight=provenance, evaluation_config=config, videos=videos, scope=scope)
        if not (output / 'provenance.json').exists(): write_new(output / 'provenance.json', snapshot)
        else:
            previous=unique_json((output / 'provenance.json').read_bytes())
            expected=dict(snapshot)
            if 'scope' not in previous and scope.startswith('historical:'): expected.pop('scope')
            if previous != expected: raise EvaluationError('Quality protocol snapshot changed')
        bindings = {v['case_id']:dict(**identity, evaluator_sha256=protocol, video=v) for v in videos}
        scores = {v['case_id']:{} for v in videos}
        statuses = {v['case_id']:{d:'missing' for d in QUALITY_DIMENSIONS} for v in videos}
        refs, failures, rejected = [], [], []
        for dimension in QUALITY_DIMENSIONS:
            for video in videos:
                cid = video['case_id']; key = canonical_sha(cid)
                folder = output / 'cases' / key / dimension
                for path in sorted(folder.glob('*.json'), key=lambda p:p.stat().st_mtime_ns, reverse=True):
                    try:
                        ref = reference(path)
                        receipt = verify_record(ref, bindings[cid], dimension)
                        scores[cid][dimension] = receipt['score']; statuses[cid][dimension] = 'complete'; refs.append(ref)
                        break
                    except (OSError, ValueError, KeyError, TypeError) as exc:
                        rejected.append(dict(case_id=cid, dimension=dimension, path=str(path), reason=str(exc)))
        # Retain failed/missing distinction in CPU-only aggregation after restart.
        for failure_path in output.glob('batches/*/failure-*.json'):
            failure = unique_json(failure_path.read_bytes())
            if len(failure.get('case_ids', [])) != 1: continue
            cid = failure['case_ids'][0]
            dimension = failure.get('dimension')
            if cid in scores and dimension in QUALITY_DIMENSIONS and dimension not in scores[cid]:
                statuses[cid][dimension] = 'failed'
        if provenance is not None:
            _adopt_legacy(evaluator, output_dir, output, provenance, bindings, scores, statuses, refs)
        if mode != 'aggregate-only':
            for dimension in dimensions:
                pending = [v for v in videos if dimension not in scores[v['case_id']]
                           and (case_ids is None or v['case_id'] in case_ids)]
                for start in range(0, len(pending), batch_size):
                    _score_batch(evaluator, pending[start:start+batch_size], dimension, output, protocol,
                                 bindings, scores, statuses, refs, failures)
            postflight = evaluator.preflight()
            if postflight != provenance: raise EvaluationError('Official source/runtime drifted during evaluation')
        validate_inputs(videos)
        cases = [dict(case_id=v['case_id'], video_sha256=v['sha256'], scores=scores[v['case_id']],
                      dimension_status=statuses[v['case_id']],
                      **quality_fields(scores[v['case_id']], failed='failed' in statuses[v['case_id']].values())) for v in videos]
        complete = [c for c in cases if c['quality_score_status'] == 'complete']
        means = {d:sum(c['scores'][d] for c in complete) / len(complete) for d in QUALITY_DIMENSIONS} if complete else {}
        all_complete = len(complete) == len(videos)
        # A partial sample mean is available for analysis but never substitutes
        # for a complete-run Quality Score.
        formal = full_benchmark_cohort(backbone, dataset_sha256, [v["case_id"] for v in videos], scope)
        overall_fields = cohort_quality_fields(means if all_complete else {}, formal=formal and all_complete, failed=any('failed' in s.values() for s in statuses.values()))
        six_cases = [c['vbench_long_avg6'] for c in cases if c['vbench_long_avg6'] is not None]
        overall_fields['vbench_long_avg6'] = sum(six_cases)/len(six_cases) if len(six_cases) == len(cases) else None
        summary = dict(schema=SCHEMA, status='COMPLETE' if all_complete else 'FAILED' if any('failed' in s.values() for s in statuses.values()) else 'INCOMPLETE',
            evaluator='vbench-long', long_video_mode=True, release_acceptance=False, overall=None,
            backbone=backbone, method=method, scope=scope, identity=identity,
            dataset_sha256=dataset_sha256, generation_sha256=generation_sha256, evaluator_sha256=protocol,
            evaluation_config_sha256=canonical_sha(config), provenance=provenance,
            provenance_reference=reference(output / 'provenance.json'), requested_case_ids=[v['case_id'] for v in videos],
            valid_scored_count=len(complete), quality_case_ids=[c['case_id'] for c in complete], cases=cases,
            scores=means if all_complete else None, formal_mean=means if all_complete and formal else None,
            partial_sample_mean=means or None, case_receipts=refs, failures=failures, rejected_cache=rejected,
            code_commit=_code_commit(evaluator.root), integration_source_sha256=_integration_sha(evaluator.root), **overall_fields)
        destination = output / ('summary-' + uuid.uuid4().hex)
        destination.mkdir()
        from .quality_report import export_report
        export_report([summary], destination)
        path = destination / 'summary.json'; write_new(path, summary)
        result = dict(summary, summary_path=str(path), summary_sha256=sha256_file(path))
        from .quality_report import publish_report
        publish_report(evaluator.root, result)
        return result


def _code_commit(root):
    p = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root, capture_output=True, text=True, check=False)
    if p.returncode == 0: return p.stdout.strip()
    revision=Path(root)/'CODE_REVISION'
    value=revision.read_text().strip() if revision.is_file() else None
    return value if value and len(value)==40 and all(c in '0123456789abcdef' for c in value) else None


def _score_batch(evaluator, videos, dimension, output, protocol, bindings, scores, statuses, refs, failures):
    videos=[v for v in videos if dimension not in scores[v['case_id']]]
    if not videos: return
    from .bench1 import probe_video
    attempt = output / 'batches' / (dimension + '-' + uuid.uuid4().hex)
    attempt.mkdir(parents=True)
    try:
        if hasattr(evaluator.backend, 'group_videos'):
            groups = evaluator.backend.group_videos(videos, dimension)
            if len(groups) > 1:
                for group in groups:
                    _score_batch(evaluator, group, dimension, output, protocol, bindings, scores, statuses, refs, failures)
                return
        probes = {v['case_id']:probe_video(v['path'], ffprobe=evaluator.config.get('ffprobe','ffprobe')) for v in videos}
        write_new(attempt / 'video_probe.json', probes)
        result = evaluator.backend.run_dimension_batch(videos, dimension, attempt)
        batch = dict(dimension=dimension, evaluator_sha256=protocol, raw_result=result['raw_result'],
            protocol_reference=reference(output / 'provenance.json'), repo_root=str(evaluator.root),
            evidence_sha256={n:sha256_file(attempt / n) for n in
                            ('invocation.json','process.json','stdout.log','stderr.log','video_probe.json')})
        write_new(attempt / 'batch.json', batch)
        batch_ref = reference(attempt / 'batch.json')
        invocation = unique_json((attempt / 'invocation.json').read_bytes())
        for video in videos:
            cid = video['case_id']
            alias, = [a for a,v in invocation['aliases'].items() if v == video]
            receipt = dict(status='complete', binding=bindings[cid], dimension=dimension,
                           score=result['scores'][cid], alias=alias, batch=batch_ref)
            folder = output / 'cases' / canonical_sha(cid) / dimension; folder.mkdir(parents=True, exist_ok=True)
            path = folder / (uuid.uuid4().hex + '.json'); write_new(path, receipt)
            ref = reference(path); verify_record(ref, bindings[cid], dimension)
            scores[cid][dimension] = receipt['score']; statuses[cid][dimension] = 'complete'; refs.append(ref)
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        write_new(attempt / ('failure-' + uuid.uuid4().hex + '.json'), dict(reason=str(exc), dimension=dimension, case_ids=[v['case_id'] for v in videos]))
        # Bisect batches to isolate one broken video without sacrificing other cases.
        if len(videos) > 1:
            middle = len(videos)//2
            for chunk in (videos[:middle], videos[middle:]):
                _score_batch(evaluator, chunk, dimension, output, protocol, bindings, scores, statuses, refs, failures)
        else:
            cid = videos[0]['case_id']; statuses[cid][dimension] = 'failed'
            failures.append(dict(case_id=cid, dimension=dimension, attempt=str(attempt), reason=str(exc)))


def audit_quality_summary(summary, config, videos):
    """Recompute all public values from raw pinned receipts; CPU-only."""
    if summary.get('schema') != SCHEMA or summary.get('evaluation_config_sha256') != canonical_sha(config):
        raise EvaluationError('Quality summary schema/config mismatch')
    validate_inputs(videos)
    for field in ('dataset_sha256','backbone','method','generation_sha256','evaluation_config_sha256'):
        if summary.get(field) != summary['identity'].get(field):
            raise EvaluationError('Quality summary identity differs: ' + field)
    snapshot = read_reference(summary['provenance_reference'])
    expected_snapshot=dict(identity=summary['identity'], preflight=summary['provenance'], evaluation_config=config, videos=videos, scope=summary['scope'])
    if 'scope' not in snapshot and summary['scope'].startswith('historical:') and summary['formal_mean'] is None:
        expected_snapshot.pop('scope')
    if snapshot != expected_snapshot:
        raise EvaluationError('Quality summary protocol snapshot mismatch')
    if canonical_sha(summary['provenance']) != summary['evaluator_sha256']:
        raise EvaluationError('Quality summary protocol digest mismatch')
    original = {v['case_id']:{} for v in videos}
    by_id = {v['case_id']:v for v in videos}
    for ref in summary['case_receipts']:
        receipt = read_reference(ref); cid = receipt['binding']['video']['case_id']; dimension = receipt['dimension']
        binding = dict(**summary['identity'], evaluator_sha256=summary['evaluator_sha256'], video=by_id[cid])
        receipt = verify_record(ref, binding, dimension)
        if dimension in original[cid]: raise EvaluationError('Duplicate quality checkpoint')
        original[cid][dimension] = receipt['score']
    if [c['case_id'] for c in summary['cases']] != list(original): raise EvaluationError('Quality case order changed')
    for case in summary['cases']:
        if case['scores'] != original[case['case_id']]: raise EvaluationError('Quality case score changed')
        expected = quality_fields(case['scores'], failed='failed' in case['dimension_status'].values())
        if any(case.get(k) != v for k,v in expected.items()): raise EvaluationError('Quality case fields changed')
    complete = [s for s in original.values() if quality_fields(s)['quality_score'] is not None]
    all_complete = len(complete) == len(videos)
    means = {d:sum(s[d] for s in complete)/len(complete) for d in QUALITY_DIMENSIONS} if complete else {}
    formal = full_benchmark_cohort(summary['backbone'], summary['dataset_sha256'], list(original), summary['scope'])
    expected = cohort_quality_fields(means if all_complete else {}, formal=formal and all_complete, failed=any('failed' in c['dimension_status'].values() for c in summary['cases']))
    six = [quality_fields(s)['vbench_long_avg6'] for s in original.values()]
    expected['vbench_long_avg6'] = sum(six)/len(six) if all(v is not None for v in six) else None
    if (any(summary.get(k) != v for k,v in expected.items()) or summary['scores'] != (means if all_complete else None)
            or summary['valid_scored_count'] != len(complete) or summary['partial_sample_mean'] != (means or None)
            or summary['quality_case_ids'] != [cid for cid,s in original.items() if quality_fields(s)['quality_score'] is not None]):
        raise EvaluationError('Quality aggregate fields changed')
    if (summary['requested_case_ids'] != list(original)
            or summary['formal_mean'] != (means if all_complete and formal else None)
            or summary['status'] != ('COMPLETE' if all_complete else 'FAILED' if any('failed' in c['dimension_status'].values() for c in summary['cases']) else 'INCOMPLETE')):
        raise EvaluationError('Quality formal scope/status changed')
    for case in summary['cases']:
        if set(case['dimension_status']) != set(QUALITY_DIMENSIONS): raise EvaluationError('Quality dimension status incomplete')
        if any((case['dimension_status'][d] != 'complete') if d in case['scores'] else case['dimension_status'][d] not in ('failed','missing') for d in QUALITY_DIMENSIONS):
            raise EvaluationError('Quality dimension status contradicts raw scores')
    if any(k.startswith('clip_') for k in summary):
        from .segment_clip import audit_attachment
        # Frozen prompt verification uses this checkout; no model is imported.
        root = Path(__file__).resolve().parents[3]
        audit_attachment(summary, root, videos)
    return all_complete


def _compatible_preflight(old, new):
    # Dimension order/list alone does not change compute semantics; every source,
    # weight, interpreter, dependency, environment and parameter pin must match.
    from .quality import AVG6_DIMENSIONS
    if set(old.get('dimensions', [])) not in (set(QUALITY_DIMENSIONS), set(AVG6_DIMENSIONS)):
        return False
    return {k:v for k,v in old.items() if k != 'dimensions'} == {k:v for k,v in new.items() if k != 'dimensions'}


def _verify_legacy_record(receipt, binding, dimension):
    from .bench1 import Bench1Evaluator
    summary = read_reference(receipt['legacy_summary'])
    snapshot = read_reference(receipt['target_provenance'])
    old = read_reference(receipt['legacy'])
    if (not _compatible_preflight(summary['provenance'], snapshot['preflight'])
            or canonical_sha(snapshot['preflight']) != binding['evaluator_sha256']
            or summary['generation_sha256'] != binding['generation_sha256']
            or summary['dataset_sha256'] != binding['dataset_sha256']
            or canonical_sha(summary['provenance']) != summary['evaluator_sha256']):
        raise EvaluationError('Legacy source/runtime/dataset differs from current protocol')
    previous = old['binding']
    if (previous['generation_sha256'] != binding['generation_sha256']
            or previous['dataset_sha256'] != binding['dataset_sha256']
            or previous['backbone'] != binding['backbone']
            or previous['evaluator_sha256'] != summary['evaluator_sha256']
            or previous['case_id'] != binding['video']['case_id']
            or previous['video_sha256'] != binding['video']['sha256']):
        raise EvaluationError('Legacy case/video does not bind current input')
    if not any(r['receipt_path'] == receipt['legacy']['path'] and r['receipt_sha256'] == receipt['legacy']['sha256']
               for r in summary['case_receipts']):
        raise EvaluationError('Legacy receipt absent from original summary')
    certified = Bench1Evaluator._receipt(Path(receipt['legacy']['path']), previous)
    if certified['scores'][dimension] != receipt['score']:
        raise EvaluationError('Legacy score differs from raw official output')
    invocation = unique_json((Path(receipt['legacy']['path']).parent / 'invocation.json').read_bytes())
    argv = invocation['argv']
    if ('--dimension' not in argv or argv[-3:] != ['--mode','long_custom_input','--dev_flag']
            or set(argv[argv.index('--dimension')+1:-3]) != set(certified['scores'])):
        raise EvaluationError('Legacy invocation has different Long protocol')
    attempt=Path(receipt['legacy']['path']).parent
    process=unique_json((attempt/'process.json').read_bytes())
    target=attempt/'input'/(binding['video']['case_id']+'.mp4')
    if (invocation.get('input_video')!=binding['video']
            or invocation.get('input_copy_sha256')!=binding['video']['sha256']
            or sha256_file(target)!=binding['video']['sha256']
            or process.get('exit_code')!=0 or process.get('failure_type') is not None):
        raise EvaluationError('Legacy copied video/process differs from current input')
    config=snapshot['evaluation_config']
    if config.get('backend')=='official_vbench_long':
        from .vbench_long import VBenchLong,ENTRY
        backend=VBenchLong(receipt['repo_root'],config)
        attempt=Path(receipt['legacy']['path']).parent
        # Runtime paths in reviewed production configs are absolute. Relative
        # paths must resolve identically to the original invocation's cwd.
        official=backend.path(config['official_root'])
        expected=[str(backend.path(config['python'],resolve_symlinks=False)),'-B',str(official/ENTRY),
                  '--videos_path',str(attempt/'input'),'--output_path',str(attempt/'raw'),
                  '--dimension',*certified['scores'],'--mode','long_custom_input','--dev_flag']
        environment=backend.environment()
        critical=('PYTHONPATH','VBENCH_CACHE_DIR','TORCH_HOME','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE',
                  'EVAL_COMPAT_ENABLE','EVAL_COMPAT_OVERLAY','DRAFTKV_VBENCH_OFFLINE_HUB')
        if (argv!=expected or invocation.get('cwd')!=str(official)
                or any(invocation.get('environment',{}).get(k)!=environment.get(k) for k in critical)):
            raise EvaluationError('Legacy invocation source/runtime configuration mismatch')
    return receipt


def _adopt_legacy(evaluator, output_dir, output, provenance, bindings, scores, statuses, refs):
    # Only authentic old Bench1 case receipts are auto-imported. Arbitrary CSVs
    # lack model/input evidence and stay historical observations, not cache hits.
    for path in sorted(Path(output_dir).glob('*/summary-*.json')):
        try:
            summary = unique_json(path.read_bytes())
            if summary.get('schema') != 'draftkv.bench1.vbench_long.summary.v1' or not _compatible_preflight(summary['provenance'], provenance):
                continue
            for old_ref in summary['case_receipts']:
                legacy_ref = dict(path=old_ref['receipt_path'], sha256=old_ref['receipt_sha256'])
                old = read_reference(legacy_ref); cid = old['binding']['case_id']
                if cid not in bindings: continue
                for dimension, value in old['scores'].items():
                    if dimension not in QUALITY_DIMENSIONS or dimension in scores[cid]: continue
                    record = dict(status='complete', binding=bindings[cid], dimension=dimension, score=value, repo_root=str(evaluator.root),
                        legacy=legacy_ref, legacy_summary=reference(path), target_provenance=reference(output / 'provenance.json'))
                    _verify_legacy_record(record, bindings[cid], dimension)
                    folder = output / 'cases' / canonical_sha(cid) / dimension; folder.mkdir(parents=True, exist_ok=True)
                    target = folder / (uuid.uuid4().hex + '.json'); write_new(target, record)
                    scores[cid][dimension] = value; statuses[cid][dimension] = 'complete'; refs.append(reference(target))
        except (OSError, ValueError, KeyError, TypeError):
            continue


def _legacy_snapshot(evaluator, output_dir):
    """Select an existing source/runtime certificate without an import/model probe."""
    from .vbench_long import SOURCE_SHA256, ENTRY_SHA256
    expected_lock = evaluator.config.get('runtime_lock', {}).get('sha256')
    if not expected_lock: return None
    for path in sorted(Path(output_dir).glob('*/summary-*.json'),reverse=True):
        try:
            summary=unique_json(path.read_bytes()); preflight=summary['provenance']
            if (summary.get('schema')=='draftkv.bench1.vbench_long.summary.v1'
                    and preflight.get('status')=='READY' and preflight.get('source_sha256')==SOURCE_SHA256
                    and preflight.get('entry_sha256')==ENTRY_SHA256
                    and preflight.get('runtime_lock_sha256')==expected_lock
                    and canonical_sha(preflight)==summary['evaluator_sha256']):
                return preflight
        except (OSError,ValueError,KeyError,TypeError):continue
    return None


def _integration_sha(root):
    folder=Path(root)/'src/draftkv/evaluators'
    return canonical_sha({p.name:sha256_file(p) for p in sorted(folder.glob('*.py'))})
