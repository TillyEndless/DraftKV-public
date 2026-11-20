"""Offline OpenAI CLIP worker. Tensor/decoder imports occur only in subprocess.

--probe imports dependencies with CUDA hidden but never constructs a model.
--metadata-only audits installed package records without importing torch.
Normal execution is explicit inference and is NOT part of CPU repository tests.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import sys

from draftkv.benchmarks.datasets import sha256_file, unique_json
from .official import EvaluationError
from .segment_clip import MODEL_SHA256, PROTOCOL, sampling_plan, score_embeddings, unit
from .vbench_long import canonical_sha, write_new

# Existing portable evaluator requirements; never pip install on an eval node.
VERSIONS = {'torch':'2.14.0', 'torchvision':'0.29.0', 'openai-clip':'1.0.1',
            'Pillow':'11.3.0', 'av':'15.1.0', 'numpy':'2.5.3', 'ftfy':'6.3.1'}
CLIP_FILES = ('clip/__init__.py', 'clip/clip.py', 'clip/model.py', 'clip/simple_tokenizer.py',
              'clip/bpe_simple_vocab_16e6.txt.gz')


def code_hashes():
    folder = Path(__file__).parent
    return {name:sha256_file(folder/name) for name in ('segment_clip.py','segment_clip_backend.py')}


def dependency_observation(*, import_only=False):
    versions = {name:metadata.version(name) for name in VERSIONS}
    if versions != VERSIONS: raise EvaluationError('CLIP exact portable dependency versions differ: ' + repr(versions))
    # regex affects tokenization; retain its exact version in comparison identity.
    versions['regex'] = metadata.version('regex')
    dist = metadata.distribution('openai-clip')
    files = {str(f):f for f in dist.files or []}
    hashes = {}
    for name in CLIP_FILES:
        entry = files.get(name)
        if entry is None or entry.hash is None or entry.hash.mode != 'sha256':
            raise EvaluationError('CLIP wheel RECORD lacks source/tokenizer hash: ' + name)
        path = Path(dist.locate_file(entry)).resolve()
        digest = sha256_file(path)
        encoded = base64.urlsafe_b64encode(bytes.fromhex(digest)).decode().rstrip('=')
        if encoded != entry.hash.value: raise EvaluationError('CLIP installed source/tokenizer differs from wheel RECORD')
        hashes[name] = digest
    if import_only:
        import clip
        import av
        if Path(clip.__file__).resolve() != Path(dist.locate_file('clip/__init__.py')).resolve():
            raise EvaluationError('CLIP import shadowed the recorded OpenAI distribution')
        if not callable(clip.load) or not callable(clip.tokenize) or not callable(av.open):
            raise EvaluationError('CLIP/AV adapter API missing')
    return dict(dependency_versions=versions, clip_distribution='openai-clip==1.0.1', clip_files_sha256=hashes)


def validate_provenance(value):
    if (value.get('model_sha256') != MODEL_SHA256 or value.get('protocol') != PROTOCOL
            or value.get('adapter_sha256') != code_hashes() or value.get('device') != 'cuda'
            or any(not isinstance(value.get(k), str) or len(value[k]) != 64 or any(c not in '0123456789abcdef' for c in value[k])
                   for k in ('quality_runtime_lock_sha256','interpreter_sha256'))
            or value.get('clip_distribution') != 'openai-clip==1.0.1'
            or any(value.get('dependency_versions', {}).get(k) != v for k, v in VERSIONS.items())
            or not value.get('dependency_versions', {}).get('regex')
            or set(value.get('clip_files_sha256', {})) != set(CLIP_FILES)
            or any(not isinstance(h, str) or len(h) != 64 or any(c not in '0123456789abcdef' for c in h)
                   for h in value['clip_files_sha256'].values())):
        raise EvaluationError('CLIP model/dependency/protocol provenance mismatch')


class NativeCLIP:
    def __init__(self, root, quality_provenance, *, quality_config):
        from .vbench_long import VBenchLong
        self.root = Path(root).resolve()
        self.runtime = VBenchLong(root, quality_config)
        self.quality_provenance = quality_provenance
        self.config = quality_config

    def environment(self, *, probe=False):
        env = self.runtime.environment(probe=probe)
        # Retain vetted site overlays (e.g. observed PyAV); the import probe
        # independently rejects a CLIP import shadowing its recorded wheel.
        env['PYTHONPATH'] = os.pathsep.join([str(self.root / 'src'), env['PYTHONPATH']])
        env['EVAL_COMPAT_ENABLE'] = '0'  # CLIP never calls torchvision.write_video.
        env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                   PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
        return env

    def command(self, *args):
        return [str(self.runtime.path(self.config.get('python'), resolve_symlinks=False)), '-B',
                '-m', 'draftkv.evaluators.segment_clip_backend', *map(str, args)]

    def assets(self):
        model = self.runtime.path(self.config.get('cache_root')) / 'clip_model/ViT-B-32.pt'
        if not model.is_file() or sha256_file(model) != MODEL_SHA256:
            raise EvaluationError('CLIP ViT-B/32 missing/changed: requires local pinned asset; downloads prohibited')
        lock_spec = self.config['runtime_lock']
        lock_path = self.runtime.path(lock_spec['path'])
        if sha256_file(lock_path) != lock_spec['sha256']:
            raise EvaluationError('CLIP evaluator runtime lock changed')
        lock = unique_json(lock_path.read_bytes())
        python = self.runtime.path(self.config['python'], resolve_symlinks=False)
        if (self.quality_provenance.get('status') != 'READY' or not lock.get('verified')
                or self.quality_provenance.get('runtime_lock_sha256') != lock_spec['sha256']
                or sha256_file(python) != lock['interpreter_sha256']):
            raise EvaluationError('CLIP requires the existing vetted evaluator interpreter/runtime')
        return model, lock

    def preflight(self):
        self.assets()
        p = subprocess.run(self.command('--probe'), env=self.environment(probe=True), cwd=self.root,
                           capture_output=True, text=True, timeout=120, check=False)
        if p.returncode: raise EvaluationError('CLIP CPU import probe failed: ' + p.stderr[-1500:])
        observation = json.loads(p.stdout.strip().splitlines()[-1])
        provenance = dict(observation, model_sha256=MODEL_SHA256, protocol=PROTOCOL, adapter_sha256=code_hashes(),
            quality_runtime_lock_sha256=self.config['runtime_lock']['sha256'],
            interpreter_sha256=sha256_file(self.runtime.path(self.config['python'], resolve_symlinks=False)), device='cuda')
        validate_provenance(provenance)
        return provenance

    def run(self, attempt):
        model, _ = self.assets()
        job_path = attempt / 'job.json'
        expected = unique_json(job_path.read_bytes())['identity']['provenance']
        if self.preflight() != expected: raise EvaluationError('CLIP preflight drift before worker')
        command = self.command('--job', job_path, '--model', model)
        write_new(attempt / 'invocation.json', dict(argv=command, cwd=str(self.root),
            environment={k:v for k, v in self.environment().items() if k in
                         ('PYTHONPATH','CUDA_VISIBLE_DEVICES','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','CUBLAS_WORKSPACE_CONFIG')}))
        # One process for all missing cases, one model load. Stream logs to disk.
        with (attempt/'stdout.log').open('x') as out, (attempt/'stderr.log').open('x') as err:
            p = subprocess.run(command, env=self.environment(), cwd=self.root, stdout=out, stderr=err,
                               timeout=self.config.get('timeout_seconds', 3600)*max(1, len(unique_json(job_path.read_bytes())['videos'])), check=False)
        write_new(attempt/'process.json', dict(exit_code=p.returncode))
        if p.returncode: raise EvaluationError('CLIP worker failed; see ' + str(attempt/'stderr.log'))


def decode_samples(path, preprocess):
    """One sequential decode, bounded at sixty preprocessed CPU images per case."""
    import av
    targets = {8+16*i for i in range(60)}
    pts, images = [], []
    with av.open(str(path), mode='r') as container:
        if len(container.streams.video) != 1: raise EvaluationError('CLIP requires one video stream')
        stream = container.streams.video[0]
        fps = float(stream.average_rate)
        if stream.duration is None: raise EvaluationError('CLIP requires measured stream duration')
        duration = float(stream.duration * stream.time_base)
        for i, frame in enumerate(container.decode(stream)):
            if frame.pts is None: raise EvaluationError('CLIP requires actual decoded PTS')
            pts.append(float(frame.pts * frame.time_base))
            if i in targets: images.append(preprocess(frame.to_image().convert('RGB')))
            if i >= 957: raise EvaluationError('CLIP native video has extra frames')
    plan = sampling_plan(pts, fps=fps, duration=duration)
    if len(images) != 60: raise EvaluationError('CLIP could not decode all fixed samples')
    return images, plan


def execute_job(job_path, model_path):
    """Real model execution. Never used by the CPU test suite."""
    from .segment_clip import validate_identity, frozen_rows
    job = unique_json(Path(job_path).read_bytes())
    identity = job['identity']
    rows = frozen_rows(Path(__file__).resolve().parents[3])
    validate_identity(identity, rows, identity['videos'])
    validate_provenance(identity['provenance'])
    if dependency_observation(import_only=True) != {k:identity['provenance'][k] for k in
                                                  ('dependency_versions','clip_distribution','clip_files_sha256')}:
        raise EvaluationError('CLIP worker dependency provenance changed')
    if sha256_file(model_path) != MODEL_SHA256: raise EvaluationError('CLIP worker weight SHA256 mismatch')
    # Explicit local filename prevents clip.load from consulting its URL registry.
    import clip
    import torch
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    device = identity['provenance']['device']
    model, preprocess = clip.load(str(Path(model_path).resolve()), device=device, jit=False)
    model = model.float().eval()
    if model.visual.input_resolution != 224: raise EvaluationError('CLIP ViT-B/32 resolution mismatch')
    folder = Path(job_path).parent
    for video in job['videos']:
        cid = video['case_id']
        try:
            if video not in identity['videos'] or job['prompts'][cid] != rows[cid]['prompts']:
                raise EvaluationError('CLIP worker video/prompt changed')
            if sha256_file(Path(video['path'])) != video['sha256']: raise EvaluationError('CLIP video changed before decode')
            images, plan = decode_samples(video['path'], preprocess)
            with torch.inference_mode():
                tokens = clip.tokenize(job['prompts'][cid], context_length=77, truncate=True).to(device)
                texts = model.encode_text(tokens).float().cpu().tolist()
                embeddings = []
                for start in range(0, 60, PROTOCOL['image_batch_size']):
                    batch = torch.stack(images[start:start+PROTOCOL['image_batch_size']]).to(device)
                    embeddings.extend(model.encode_image(batch).float().cpu().tolist())
            images_unit, texts_unit = [unit(e) for e in embeddings], [unit(e) for e in texts]
            if sha256_file(Path(video['path'])) != video['sha256']: raise EvaluationError('CLIP video changed during decode')
            result = dict(case_id=cid, video_sha256=video['sha256'], prompts=job['prompts'][cid],
                          provenance=identity['provenance'], sampling=plan,
                          image_embeddings=images_unit, text_embeddings=texts_unit,
                          **score_embeddings(images_unit, texts_unit))
            write_new(folder/(cid+'.json'), result)
            write_new(folder/(cid+'.process.json'), dict(exit_code=0, case_id=cid))
        except Exception as exc:
            write_new(folder/(cid+'.failure.json'), dict(case_id=cid, reason=str(exc)))
    if sha256_file(model_path) != MODEL_SHA256: raise EvaluationError('CLIP model changed during batch')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--probe', action='store_true')
    p.add_argument('--metadata-only', action='store_true')
    p.add_argument('--job', type=Path)
    p.add_argument('--model', type=Path)
    args = p.parse_args(argv)
    if args.probe or args.metadata_only:
        print(json.dumps(dependency_observation(import_only=args.probe), allow_nan=False)); return
    if args.job is None or args.model is None: p.error('--job and --model are required for explicit scoring')
    execute_job(args.job, args.model)


if __name__ == '__main__': main()
