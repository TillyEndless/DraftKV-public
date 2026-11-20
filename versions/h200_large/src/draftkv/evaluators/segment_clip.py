"""Custom WorldAttention CLIP6 protocol and immutable CPU-verifiable receipts.

This module deliberately imports no tensor, decoder or metric-model packages.
It is independent of the official VBench dimensions and Quality Score.
"""
from __future__ import annotations

import fcntl
import hashlib
import math
from pathlib import Path
import uuid

from draftkv.benchmarks.datasets import sha256_file, unique_json
from draftkv.benchmarks.worldattention import CASE_IDS, MANIFEST_SHA256, verify_manifest
from .official import EvaluationError
from .vbench_long import canonical_sha, write_new

MODEL_SHA256 = '40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af'
SCHEMA = 'draftkv.bench1.segment_clip6.v1'
COLUMNS = tuple(f'CLIP {10*i}–{10*(i+1)}s' for i in range(6))
PROTOCOL = dict(version='worldattention-clip6-openai-vitb32-v1', model='OpenAI CLIP ViT-B/32',
    model_sha256=MODEL_SHA256, windows=[[10*i, 10*(i+1)] for i in range(6)],
    sampling='10 fixed centers per window: start+0.5,...,start+9.5; containing PTS interval',
    timeline='native 957 frames at 16fps, zero origin, CFR; reject other geometry',
    tokenizer=dict(implementation='openai-clip 1.0.1 BPE', context_length=77, truncate=True,
                   truncation='retain SOT, first 75 content tokens, force EOT in slot 76'),
    preprocessing='OpenAI Resize(shorter=224,BICUBIC), CenterCrop(224), RGB, ToTensor, Normalize',
    image_mean=[0.48145466, 0.4578275, 0.40821073], image_std=[0.26862954, 0.26130258, 0.27577711],
    precision='float32 model/encoder outputs; Python float64 L2/cosine/fsum; TF32 disabled; deterministic algorithms',
    formula='100 * dot(image / L2(image), text / L2(text)); no clamp; no logit_scale',
    reduction='10 frame cosines -> case segment mean -> equally weighted cohort segment mean; six-value mean separately',
    image_batch_size=32, official=False, paper_numeric_equivalence=False)
METHODS = ('native', 'draftkv', 'shadowkv', 'infinigen', 'infllm', 'freekv', 'deep-forcing', 'future-forcing', 'worldtrace-landmark')


def ref(path):
    return dict(path=str(Path(path).resolve()), sha256=sha256_file(Path(path)))


def read_ref(value):
    path = Path(value['path'])
    if sha256_file(path) != value['sha256']:
        raise EvaluationError('CLIP evidence SHA256 mismatch: ' + str(path))
    return unique_json(path.read_bytes())


def frozen_rows(root):
    path = Path(root) / 'configs/datasets/bench1/worldattention160.manifest.json'
    return {r['case_id']: r for r in verify_manifest(path)['cases']}


def prompt_hash(prompt):
    if not isinstance(prompt, str) or not prompt.strip():
        raise EvaluationError('CLIP requires each exact individual source prompt')
    return hashlib.sha256(prompt.encode('utf-8')).hexdigest()


def unit(vector):
    if (not isinstance(vector, (list, tuple)) or not vector
            or any(isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x) for x in vector)):
        raise EvaluationError('CLIP embedding must be a finite nonempty vector')
    norm = math.sqrt(math.fsum(x*x for x in vector))
    if norm <= 0 or not math.isfinite(norm):
        raise EvaluationError('CLIP embedding has invalid L2 norm')
    return [x / norm for x in vector]


def cosine100(image, text):
    """The actual scoring function, shared by the native worker and CPU audit."""
    image, text = unit(image), unit(text)
    if len(image) != len(text): raise EvaluationError('CLIP embedding sizes differ')
    return 100.0 * math.fsum(a*b for a, b in zip(image, text))


def score_embeddings(images, texts):
    if len(images) != 60 or len(texts) != 6:
        raise EvaluationError('CLIP requires 60 image and six individual text embeddings')
    per_frame = [[cosine100(images[10*s+i], texts[s]) for i in range(10)] for s in range(6)]
    segments = [math.fsum(values)/10 for values in per_frame]
    return dict(frame_scores=per_frame, segment_scores=segments, six_segment_mean=math.fsum(segments)/6)


def sampling_plan(pts, *, fps=16, duration=None):
    """Record actual full decoded timeline and fixed nominal sample selection.

    Fail closed on time scaling, a shifted origin, dropped/extra frames or VFR.
    Native timing is an admission rule, never a reason to move prompt windows.
    """
    if (fps != 16 or len(pts) != 957 or any(not math.isfinite(p) or abs(p-i/16) > 1e-7 for i, p in enumerate(pts))
            or duration is None or abs(duration - 957/16) > 1e-7):
        raise EvaluationError('CLIP requires unshifted native 957-frame 16fps timeline (59.8125s)')
    segments = []
    for s in range(6):
        start, end = 10*s, 10*(s+1)
        times = [start+i+.5 for i in range(10)]
        indices = [math.floor(t*fps) for t in times]
        valid = [i for i, p in enumerate(pts) if start <= p < end]
        if any(i >= len(pts) or not start <= pts[i] < end for i in indices):
            raise EvaluationError('CLIP sample is outside actual video/window')
        segments.append(dict(prompt_index=s, nominal_window=[start, end], sample_timestamps=times,
            frame_indices=indices, frame_pts=[pts[i] for i in indices], valid_frame_count=len(valid),
            valid_frame_index_range=[valid[0], valid[-1]], available_window=[start, min(end, duration)],
            missing_tail_seconds=max(0, end-duration)))
    return dict(fps=fps, frame_count=len(pts), pts=pts, duration_seconds=duration, segments=segments)


def validate_identity(identity, rows, videos):
    ids = [v['case_id'] for v in videos]
    if (identity.get('dataset_sha256') != MANIFEST_SHA256 or identity.get('backbone', '').lower() != 'longlive1.3b'
            or identity.get('method') not in METHODS or not ids or ids != [c for c in CASE_IDS if c in set(ids)]
            or identity.get('videos') != videos or identity.get('protocol') != PROTOCOL
            or identity.get('prompt_sha256') != {cid:[prompt_hash(p) for p in rows[cid]['prompts']] for cid in ids}):
        raise EvaluationError('CLIP frozen cohort/model/method/prompt/protocol binding mismatch')
    for v in videos:
        p = Path(v['path'])
        if p.suffix.lower() != '.mp4' or not p.is_file() or sha256_file(p) != v['sha256']:
            raise EvaluationError('CLIP input video SHA256 mismatch')
    if len({Path(v['path']).resolve() for v in videos}) != len(ids):
        raise EvaluationError('CLIP videos cannot share an artifact')


def verify_case(reference, identity, video, rows):
    receipt = read_ref(reference)
    cid = video['case_id']
    if (receipt.get('schema') != SCHEMA or receipt.get('status') != 'COMPLETE'
            or receipt.get('binding') != dict(identity_sha256=canonical_sha(identity), video=video,
                prompts=rows[cid]['prompts'], prompt_sha256=identity['prompt_sha256'][cid])):
        raise EvaluationError('CLIP case/video/prompt/provenance binding mismatch')
    result = read_ref(receipt['result'])
    plan = result['sampling']
    if plan != sampling_plan(plan['pts'], fps=plan['fps'], duration=plan['duration_seconds']):
        raise EvaluationError('CLIP frame/window mapping changed')
    if (result['case_id'] != cid or result['video_sha256'] != video['sha256']
            or result['prompts'] != rows[cid]['prompts'] or result['provenance'] != identity['provenance']):
        raise EvaluationError('CLIP raw case/model/provenance mismatch')
    images, texts = result['image_embeddings'], result['text_embeddings']
    if any(len(e) != 512 for e in [*images, *texts]): raise EvaluationError('CLIP ViT-B/32 requires 512 dimensions')
    scores = score_embeddings(images, texts)
    if any(result.get(k) != v or receipt.get(k) != v for k, v in scores.items()):
        raise EvaluationError('CLIP scores differ from actual cosine*100 embeddings')
    process = read_ref(receipt['process'])
    job = read_ref(receipt['job'])
    if (process.get('exit_code') != 0 or process.get('case_id') != cid or job['identity'] != identity
            or video not in job['videos'] or job['prompts'].get(cid) != rows[cid]['prompts']):
        raise EvaluationError('CLIP original worker process/job mismatch')
    return receipt


def aggregate(identity, receipts, requested_ids, *, blocker=None):
    ids = [r['binding']['video']['case_id'] for r in receipts]
    if len(set(ids)) != len(ids) or any(c not in requested_ids for c in ids):
        raise EvaluationError('Duplicate/unknown CLIP case')
    complete = len(ids) == len(requested_ids)
    full = complete and not blocker and requested_ids == list(CASE_IDS)
    means = [math.fsum(r['segment_scores'][s] for r in receipts)/len(receipts) for s in range(6)] if receipts else None
    return dict(clip_status='BLOCKED' if blocker else 'COMPLETE' if complete else 'INCOMPLETE',
        clip_blocker=blocker, clip_scope='FULL-160' if full else 'exploratory-subset',
        clip_completed_cases=len(ids), clip_completed_segments=6*len(ids), clip_case_ids=ids,
        clip_segment_scores=means if full else None, clip_exploratory_segment_scores=means,
        clip_six_segment_mean=math.fsum(means)/6 if full else None,
        clip_protocol_sha256=canonical_sha(dict(protocol=PROTOCOL, provenance=identity.get('provenance'))))


def audit_summary(summary, root, videos, *, generation_sha256, backbone, method):
    rows = frozen_rows(root)
    identity = summary['identity']
    validate_identity(identity, rows, videos)
    if (summary.get('schema') != SCHEMA or identity['generation_sha256'] != generation_sha256
            or identity['backbone'] != backbone or identity['method'] != method):
        raise EvaluationError('CLIP cross-generation/model/method summary')
    provenance = identity.get('provenance')
    if provenance is not None:
        from .segment_clip_backend import validate_provenance
        validate_provenance(provenance)
    if any(r.get('case_id') not in {v['case_id'] for v in videos} for r in summary['case_receipts']):
        raise EvaluationError('Unknown CLIP receipt case')
    receipts = [verify_case(r, identity, {v['case_id']:v for v in videos}[r['case_id']], rows)
                for r in summary['case_receipts']]
    expected = aggregate(identity, receipts, [v['case_id'] for v in videos], blocker=summary.get('clip_blocker'))
    if any(summary.get(k) != v for k, v in expected.items()): raise EvaluationError('CLIP summary aggregation changed')
    return expected


def evaluate_clip(root, quality, videos, output_dir, *, mode='missing-only', enabled=True, backend=None):
    """One worker/model load for all missing cases; resume from immutable cases.

    aggregate-only uses saved provenance and never invokes/probes an interpreter.
    CLIP exceptions are isolated by attach_clip; official quality stays persisted.
    """
    if mode not in ('full', 'eval-only', 'missing-only', 'aggregate-only'): raise EvaluationError('Invalid CLIP mode')
    from .segment_clip_backend import NativeCLIP
    rows = frozen_rows(root)
    base_identity = dict(dataset_sha256=quality['dataset_sha256'], backbone=quality['backbone'],
        method=quality['method'], generation_sha256=quality['generation_sha256'], videos=videos,
        prompt_sha256={v['case_id']:[prompt_hash(p) for p in rows[v['case_id']]['prompts']] for v in videos}, protocol=PROTOCOL)
    base = Path(output_dir).resolve() / ('clip6-' + canonical_sha(base_identity))
    base.mkdir(parents=True, exist_ok=True)
    backend = backend or NativeCLIP(root, quality['provenance'], quality_config=read_ref(quality['provenance_reference'])['evaluation_config'])
    with (base / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        blocker = None
        if not enabled:
            provenance = None; blocker = 'CLIP explicitly disabled (--clip-mode off)'
        elif mode == 'aggregate-only':
            snapshots = sorted(base.glob('*/identity.json'), key=lambda p:p.stat().st_mtime_ns, reverse=True)
            saved = None
            for snapshot_path in snapshots:
                candidate = read_ref(ref(snapshot_path))
                if (canonical_sha(candidate) != snapshot_path.parent.name
                        or {k:v for k, v in candidate.items() if k != 'provenance'} != base_identity):
                    raise EvaluationError('CLIP saved identity digest/cohort changed')
                if candidate['provenance'] is not None:
                    saved = candidate
                    break
            provenance = saved['provenance'] if saved else None
            if saved is None: blocker = 'No saved ready CLIP protocol; aggregate-only never probes or scores'
        else:
            try: provenance = backend.preflight()
            except Exception as exc:
                provenance = None; blocker = 'CLIP offline preflight blocked: ' + str(exc)
        identity = dict(base_identity, provenance=provenance)
        validate_identity(identity, rows, videos)
        output = base / canonical_sha(identity); output.mkdir(exist_ok=True)
        snapshot = output / 'identity.json'
        if not snapshot.exists(): write_new(snapshot, identity)
        elif read_ref(ref(snapshot)) != identity: raise EvaluationError('CLIP saved identity changed')
        receipts, references, rejected = [], [], []
        for v in videos:
            for p in sorted((output / 'cases' / v['case_id']).glob('*.json'), key=lambda p:p.stat().st_mtime_ns, reverse=True):
                try:
                    r = dict(ref(p), case_id=v['case_id'])
                    receipt = verify_case(r, identity, v, rows)
                    receipts.append(receipt); references.append(r); break
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    rejected.append(dict(path=str(p), reason=str(exc)))
        done = {r['binding']['video']['case_id'] for r in receipts}
        pending = [v for v in videos if v['case_id'] not in done]
        if pending and not blocker and mode != 'aggregate-only':
            attempt = output / ('batch-' + uuid.uuid4().hex); attempt.mkdir()
            job = dict(identity=identity, videos=pending, prompts={v['case_id']:rows[v['case_id']]['prompts'] for v in pending})
            write_new(attempt / 'job.json', job)
            try: backend.run(attempt)
            except Exception as exc: blocker = 'CLIP stage failed: ' + str(exc)
            # Even an interrupted batch can retain earlier successful cases if
            # the worker wrote an explicit successful completion marker for them.
            for v in pending:
                result_path = attempt / (v['case_id'] + '.json')
                try:
                    result = unique_json(result_path.read_bytes())
                    process_path = attempt / (v['case_id'] + '.process.json')
                    receipt = dict(schema=SCHEMA, status='COMPLETE',
                        binding=dict(identity_sha256=canonical_sha(identity), video=v, prompts=rows[v['case_id']]['prompts'],
                                     prompt_sha256=identity['prompt_sha256'][v['case_id']]),
                        result=ref(result_path), job=ref(attempt / 'job.json'), process=ref(process_path),
                        **{k:result[k] for k in ('frame_scores','segment_scores','six_segment_mean')})
                    folder = output / 'cases' / v['case_id']; folder.mkdir(parents=True, exist_ok=True)
                    path = folder / (uuid.uuid4().hex + '.json'); write_new(path, receipt)
                    reference = dict(ref(path), case_id=v['case_id'])
                    verify_case(reference, identity, v, rows)
                    receipts.append(receipt); references.append(reference)
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    rejected.append(dict(case_id=v['case_id'], reason=str(exc)))
                    blocker = blocker or 'CLIP case failed: ' + str(exc)
            if backend.preflight() != provenance: raise EvaluationError('CLIP runtime changed during scoring')
        order = {v['case_id']:i for i, v in enumerate(videos)}
        receipts.sort(key=lambda r:order[r['binding']['video']['case_id']]); references.sort(key=lambda r:order[r['case_id']])
        summary = dict(schema=SCHEMA, identity=identity, case_receipts=references, rejected_cache=rejected,
                       **aggregate(identity, receipts, [v['case_id'] for v in videos], blocker=blocker))
        audit_summary(summary, root, videos, generation_sha256=quality['generation_sha256'], backbone=quality['backbone'], method=quality['method'])
        path = output / ('summary-' + uuid.uuid4().hex + '.json'); write_new(path, summary)
        return summary, ref(path)


def attach_clip(evaluator, quality, videos, output_dir, *, mode, enabled=True):
    """Publish an appended immutable summary, never edit quality/cache history."""
    if quality.get('schema') != 'draftkv.bench1.quality.summary.v2': return quality
    try:
        clip, reference = evaluate_clip(evaluator.root, quality, videos, output_dir, mode=mode, enabled=enabled)
        fields = {k:v for k, v in clip.items() if k.startswith('clip_')}
        fields['clip_summary'] = reference
    except Exception as exc:
        fields = dict(clip_status='BLOCKED', clip_blocker='CLIP stage blocked: ' + str(exc),
            clip_scope='unavailable', clip_segment_scores=None, clip_exploratory_segment_scores=None,
            clip_completed_cases=0, clip_completed_segments=0, clip_case_ids=[], clip_summary=None,
            clip_six_segment_mean=None, clip_protocol_sha256=None)
    result = {k:v for k, v in quality.items() if k not in ('summary_path','summary_sha256')}
    result.update(fields, clip_quality_source=ref(quality['summary_path']))
    destination = Path(quality['summary_path']).parent.parent / ('summary-' + uuid.uuid4().hex)
    destination.mkdir()
    path = destination / 'summary.json'; write_new(path, result)
    from .quality_report import export_report, publish_report
    export_report([result], destination)
    result.update(summary_path=str(path), summary_sha256=sha256_file(Path(path)))
    publish_report(evaluator.root, result)
    return result


def audit_attachment(quality, root, videos):
    """Backward compatible: old quality summaries do not require a CLIP stage."""
    if 'clip_status' not in quality:
        if any(k.startswith('clip_') for k in quality):
            raise EvaluationError('CLIP fields require explicit status and authenticated evidence')
        return
    original = read_ref(quality['clip_quality_source'])
    if {k:v for k, v in quality.items() if not k.startswith('clip_') and k not in ('summary_path','summary_sha256','run_id')} != original:
        raise EvaluationError('CLIP attachment changed original quality fields')
    if quality.get('clip_summary'):
        clip = read_ref(quality['clip_summary'])
        fields = audit_summary(clip, root, videos, generation_sha256=quality['generation_sha256'],
                               backbone=quality['backbone'], method=quality['method'])
        if any(quality.get(k) != v for k, v in fields.items()): raise EvaluationError('CLIP attachment differs from summary')
    elif (quality['clip_status'] != 'BLOCKED' or not quality.get('clip_blocker')
          or quality.get('clip_segment_scores') is not None or quality.get('clip_exploratory_segment_scores') is not None
          or quality.get('clip_completed_cases') != 0 or quality.get('clip_completed_segments') != 0
          or quality.get('clip_case_ids') != [] or quality.get('clip_six_segment_mean') is not None):
        raise EvaluationError('CLIP missing evidence cannot be complete/scored')
