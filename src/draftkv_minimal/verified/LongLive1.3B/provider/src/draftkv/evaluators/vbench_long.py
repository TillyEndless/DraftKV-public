"""Pinned upstream VBench-Long CLI, without a replacement metric implementation."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from datetime import datetime, timezone

from draftkv.benchmarks.datasets import sha256_file, unique_json
from .official import EvaluationError


from .quality import QUALITY_DIMENSIONS

DIMENSIONS = QUALITY_DIMENSIONS
ENTRY = "vbench2_beta_long/eval_long.py"
SOURCE_COUNT = 239
SOURCE_SHA256 = "f794912d3e91e67b61dfe2d9593c1b75e2d0a7224acccd08f047e354981d96c5"
ENTRY_SHA256 = "a4804167eab96e0911cf1fbdc54fb9e850e91195edae5dacf256b8b2fcb4869e"
REQUIRED_WEIGHTS = {
    ("cache_root", "clip_model/ViT-B-32.pt"): "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af",
    ("cache_root", "clip_model/ViT-L-14.pt"): "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836",
    ("cache_root", "dino_model/dino_vitbase16_pretrain.pth"): "bf34ad0f424b9029b593e8dc3ed553bf26e88bcba0d32bf3e62a6209cb64c85e",
    ("cache_root", "amt_model/amt-s.pth"): "07e7e03405c213fe4405678db9b42d05671e48c56faf6a441b99bb0047d3cf77",
    ("cache_root", "raft_model/models/raft-things.pth"): "fcfa4125d6418f4de95d84aec20a3c5f4e205101715a79f193243c186ac9a7e1",
    ("cache_root", "pyiqa_model/musiq_spaq_ckpt-358bb6af.pth"): "358bb6af275e28ea56821d44fb55c6cb83645db11f394d3ad65b2d149965ab50",
    ("cache_root", "aesthetic_model/emb_reader/sa_0_4_vit_l_14_linear.pth"): "2cd4e60f4f24ae3bcd57b847b13c1f3ba27edc28cc1a7f9ce74ee9f421243cba",
    ("torch_home", "hub/checkpoints/dinov2_vitb14_pretrain.pth"): "0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73",
}
REQUIRED_HUB_TREES = {
    "hub/facebookresearch_dino_main": (18, "8be2f81ff85a730562d063056bb6fe84749c5244c35a597ae7841c88eec80611"),
    "hub/facebookresearch_dinov2_main": (203, "f43fb107d7d79cb277cd4c73ac679017a7aff1cfde599e2b4bcce6b6023eca87"),
}


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def atomic_text(path, text):
    """Publish a complete immutable file; interrupted writes never become cache hits."""
    import tempfile
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            stream.write(text); stream.flush(); os.fsync(stream.fileno())
        # link, unlike replace, also refuses concurrent overwrite.
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        os.unlink(temporary)


def write_new(path, value):
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def source_inventory(root):
    root = Path(root)
    files = sorted((p for folder in ("vbench", "vbench2_beta_long")
        for p in (root / folder).rglob("*") if p.is_file()
        and p.suffix in (".py", ".json", ".yaml") and "__pycache__" not in p.parts),
        key=lambda p: str(p.relative_to(root)))
    rows = {str(p.relative_to(root)): sha256_file(p) for p in files}
    digest = hashlib.sha256("".join(f"{sha}  {name}\n" for name, sha in rows.items()).encode()).hexdigest()
    return dict(count=len(rows), sha256=digest, files=rows)


def parse_raw_result(path, input_root, dimensions=DIMENSIONS):
    """Read upstream aggregates verbatim; keep the original per-clip result file."""
    data = unique_json(Path(path).read_bytes())
    if set(data) != set(dimensions):
        raise EvaluationError("Official VBench-Long result must contain exactly the seven frozen dimensions")
    scores = {}
    for dimension in dimensions:
        value = data[dimension]
        if (not isinstance(value, list) or len(value) < 2 or type(value[0]) not in (int, float)
                or not math.isfinite(value[0]) or not isinstance(value[1], list) or not value[1]):
            raise EvaluationError(f"Official VBench-Long dimension has no finite aggregate/details: {dimension}")
        for detail in value[1]:
            if not isinstance(detail, dict) or not isinstance(detail.get("video_path"), str):
                raise EvaluationError("Official per-clip video provenance missing")
            if not Path(detail["video_path"]).resolve().is_relative_to(Path(input_root).resolve()):
                raise EvaluationError("Official result refers to a video outside this case attempt")
            score = detail.get("video_results")
            # Upstream RAFT dynamic_degree emits a per-clip boolean decision;
            # its aggregate remains numeric. Preserve that official schema.
            valid_type = type(score) in (int, float) or (dimension == "dynamic_degree" and type(score) is bool)
            if not valid_type or not math.isfinite(score):
                raise EvaluationError("Official per-clip score missing/nonfinite")
        scores[dimension] = value[0]
    return scores


class VBenchLong:
    def __init__(self, repo_root, config):
        self.root = Path(repo_root).resolve()
        self.config = dict(config)

    def path(self, value, *, resolve_symlinks=True):
        if not isinstance(value, str) or not value or "${" in os.path.expandvars(value):
            raise EvaluationError("VBench-Long requires explicit resolved external paths")
        path = Path(os.path.expandvars(value))
        path = path if path.is_absolute() else self.root / path
        # Invoking a venv through its symlink preserves pyvenv.cfg discovery.
        return path.resolve() if resolve_symlinks else Path(os.path.abspath(path))

    def environment(self, *, probe=False):
        env = os.environ.copy()
        source = str(self.path(self.config.get("official_root")))
        # Explicit overlays are locked in runtime_lock; no site-package fallback precedes source.
        additions = [str(self.path(p)) for p in self.config.get("pythonpath", [])]
        env.update(PYTHONPATH=os.pathsep.join([source, *additions]), PYTHONNOUSERSITE="1",
                   PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        env["VBENCH_CACHE_DIR"] = str(self.path(self.config.get("cache_root")))
        env["TORCH_HOME"] = str(self.path(self.config.get("torch_home")))
        if self.config.get("home"):
            env["HOME"] = str(self.path(self.config["home"]))
        if self.config.get("compatibility_bridge") == "torchvision_write_video":
            env["EVAL_COMPAT_ENABLE"] = "1"
            if self.config.get("compatibility_overlay"):
                env["EVAL_COMPAT_OVERLAY"] = str(self.path(self.config["compatibility_overlay"]))
        if self.config.get("offline_hub") is True:
            env["DRAFTKV_VBENCH_OFFLINE_HUB"] = "1"
        if probe:
            env["CUDA_VISIBLE_DEVICES"] = ""
        return env

    def preflight(self):
        """Only hashes files and runs an import probe with CUDA hidden; never scores."""
        try:
            official = self.path(self.config.get("official_root"))
            python = self.path(self.config.get("python"), resolve_symlinks=False)
            inventory = source_inventory(official)
            if (inventory["count"] != SOURCE_COUNT or inventory["sha256"] != SOURCE_SHA256
                    or inventory["files"].get(ENTRY) != ENTRY_SHA256):
                raise EvaluationError("VBench-Long original 239-file source/entry pin mismatch")
            spec = self.config.get("runtime_lock", {})
            lock_path = self.path(spec.get("path"))
            lock_sha = sha256_file(lock_path)
            if lock_sha != spec.get("sha256"):
                raise EvaluationError("VBench-Long runtime lock SHA256 mismatch")
            lock = unique_json(lock_path.read_bytes())
            if (lock.get("schema") != "draftkv.vbench_long.runtime.v1" or lock.get("verified") is not True
                    or not lock.get("verification_reference") or sha256_file(python) != lock.get("interpreter_sha256")):
                raise EvaluationError("VBench-Long reviewed interpreter/runtime lock required")
            versions = lock.get("dependency_versions")
            if (not isinstance(versions, dict) or not versions
                    or not {"torch", "torchvision", "dreamsim", "moviepy", "decord"}.issubset(versions)
                    or any(not isinstance(v, str) or not v for v in versions.values())):
                raise EvaluationError("VBench-Long exact dependency versions, including DreamSim, required")
            artifacts = lock.get("artifacts")
            if (not isinstance(artifacts, list) or not artifacts or not all(isinstance(a, dict) for a in artifacts)
                    or not any(a.get("role") == "dreamsim_weights" for a in artifacts)):
                raise EvaluationError("VBench-Long runtime/weight closure including DreamSim is missing")
            measured = []
            for asset in artifacts:
                path = self.path(asset.get("path"))
                if sha256_file(path) != asset.get("sha256"):
                    raise EvaluationError(f"VBench-Long runtime/weight artifact SHA mismatch: {path}")
                measured.append(dict(path=str(path), sha256=asset["sha256"], role=asset.get("role")))
            covered = {Path(a["path"]).resolve() for a in measured}
            by_path = {Path(a["path"]).resolve(): a["sha256"] for a in measured}
            for (key, name), expected in REQUIRED_WEIGHTS.items():
                if by_path.get((self.path(self.config.get(key)) / name).resolve()) != expected:
                    raise EvaluationError(f"Original VBench-Long metric weight pin missing: {key}/{name}")
            overlays = [self.path(p) for p in self.config.get("pythonpath", [])]
            trees = []
            for tree in lock.get("trees", []):
                folder = self.path(tree.get("path"))
                names = sorted(p for p in folder.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc")
                raw = "".join(f"{sha256_file(p)}  {p.relative_to(folder)}\n" for p in names).encode()
                if len(names) != tree.get("count") or hashlib.sha256(raw).hexdigest() != tree.get("sha256"):
                    raise EvaluationError("VBench-Long runtime source tree SHA/count mismatch")
                trees.append(dict(path=str(folder), count=len(names), sha256=tree["sha256"]))
            for name, expected in REQUIRED_HUB_TREES.items():
                folder = (self.path(self.config.get("torch_home")) / name).resolve()
                if not any(Path(t["path"]) == folder and (t["count"], t["sha256"]) == expected for t in trees):
                    raise EvaluationError(f"Original offline VBench-Long Hub source tree missing: {name}")
            if any(not any(p == self.path(t["path"]) for t in trees) for p in overlays):
                raise EvaluationError("Every VBench-Long PYTHONPATH overlay requires a complete locked tree")
            if self.config.get("compatibility_overlay"):
                compatibility = self.path(self.config["compatibility_overlay"])
                if not any(compatibility == self.path(t["path"]) for t in trees):
                    raise EvaluationError("VBench-Long compatibility overlay requires a complete locked tree")
            for asset in lock.get("required_cache_files", []):
                path = self.path(asset)
                if path not in covered:
                    raise EvaluationError("VBench-Long cache file is not hash-locked")
            script = ("import importlib.metadata as m,json,pathlib,vbench2_beta_long; "
                "print(json.dumps({'versions':{n:m.version(n) for n in " + repr(list(versions)) + "},"
                "'module':str(pathlib.Path(vbench2_beta_long.__file__).resolve())}))")
            probe = subprocess.run([str(python), "-B", "-c", script], cwd=official,
                env=self.environment(probe=True), capture_output=True, text=True, timeout=120, check=False)
            if probe.returncode:
                raise EvaluationError("VBench-Long CPU import probe failed: " + probe.stderr[-1200:])
            observation = json.loads(probe.stdout.strip().splitlines()[-1])
            if observation["versions"] != versions or not Path(observation["module"]).is_relative_to(official / "vbench2_beta_long"):
                raise EvaluationError("VBench-Long dependency versions/import origin mismatch")
            return dict(status="READY", evaluator="vbench-long", long_video_mode=True,
                dimensions=list(DIMENSIONS), source_sha256=inventory["sha256"], entry_sha256=ENTRY_SHA256,
                runtime_lock_sha256=lock_sha, python_sha256=lock["interpreter_sha256"],
                dependency_versions=versions, artifacts=measured, trees=trees,
                environment={k: self.environment()[k] for k in ("PYTHONPATH", "VBENCH_CACHE_DIR", "TORCH_HOME", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "EVAL_COMPAT_ENABLE", "EVAL_COMPAT_OVERLAY", "DRAFTKV_VBENCH_OFFLINE_HUB") if k in self.environment()},
                scores=None)
        except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
            return dict(status="BLOCKED_DEPENDENCY", evaluator="vbench-long", reasons=[str(exc)], scores=None)

    def run_case(self, video, attempt):
        """Invoke the unmodified original CLI in an isolated copied-input directory."""
        import shutil
        attempt = Path(attempt)
        inputs, raw = attempt / "input", attempt / "raw"
        inputs.mkdir(); raw.mkdir()
        target = inputs / (video["case_id"] + ".mp4")
        shutil.copyfile(video["path"], target)
        if sha256_file(target) != video["sha256"]:
            raise EvaluationError("Copied scoring input differs from the accepted generation MP4")
        command = [str(self.path(self.config["python"], resolve_symlinks=False)), "-B", str(self.path(self.config["official_root"]) / ENTRY),
                   "--videos_path", str(inputs), "--output_path", str(raw), "--dimension", *DIMENSIONS,
                   "--mode", "long_custom_input", "--dev_flag"]
        env = self.environment()
        write_new(attempt / "invocation.json", dict(argv=command, cwd=str(self.path(self.config["official_root"])),
            environment=env, input_video=dict(video), input_copy_sha256=sha256_file(target)))
        started = datetime.now(timezone.utc).isoformat()
        try:
            with (attempt / "stdout.log").open("x") as stdout, (attempt / "stderr.log").open("x") as stderr:
                process = subprocess.run(command, cwd=self.path(self.config["official_root"]), env=env,
                    stdout=stdout, stderr=stderr, timeout=float(self.config.get("timeout_seconds", 3600)), check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            write_new(attempt / "process.json", dict(exit_code=None, started_at=started,
                ended_at=datetime.now(timezone.utc).isoformat(), failure_type="TIMEOUT" if isinstance(exc,subprocess.TimeoutExpired) else "LAUNCH_ERROR",
                error=str(exc)))
            raise
        error_text = (attempt / "stderr.log").read_text(errors="replace")
        failure = ("CUDA_OOM" if "out of memory" in error_text.lower() else
                   "ABI_ERROR" if "undefined symbol" in error_text.lower() else
                   "ENV_MISSING" if "ModuleNotFoundError" in error_text else "EVALUATOR_EXIT") if process.returncode else None
        write_new(attempt / "process.json", dict(exit_code=process.returncode, started_at=started,
            ended_at=datetime.now(timezone.utc).isoformat(), failure_type=failure))
        if process.returncode:
            raise EvaluationError(f"Official VBench-Long exited {process.returncode}; see {attempt}")
        if sha256_file(target) != video["sha256"] or sha256_file(Path(video["path"])) != video["sha256"]:
            raise EvaluationError("Original/copied MP4 changed while scoring")
        results = list(raw.glob("results_*_eval_results.json"))
        if len(results) != 1:
            raise EvaluationError("Official VBench-Long must emit exactly one original result JSON")
        scores = parse_raw_result(results[0], inputs)
        return dict(scores=scores, exit_code=0, raw_result=dict(path=str(results[0]), sha256=sha256_file(results[0])))


    def group_videos(self, videos, dimension):
        # The official crossclip encoder uses the first video's integer FPS for
        # every video in the invocation. Partition using the SAME decord probe.
        if dimension not in ('subject_consistency', 'background_consistency') or len(videos) < 2:
            return [videos]
        script = ("import json,sys; from decord import VideoReader; "
                  "print(json.dumps([int(VideoReader(p).get_avg_fps()) for p in json.load(sys.stdin)]))")
        result = subprocess.run([str(self.path(self.config['python'], resolve_symlinks=False)), '-B', '-c', script],
            input=json.dumps([v['path'] for v in videos]), text=True, capture_output=True,
            env=self.environment(probe=True), cwd=self.path(self.config['official_root']), check=False, timeout=120)
        if result.returncode: raise EvaluationError('Official decord FPS probe failed: ' + result.stderr[-600:])
        fps = json.loads(result.stdout.strip().splitlines()[-1])
        if len(fps) != len(videos) or any(type(f) is not int or f < 1 for f in fps):
            raise EvaluationError('Invalid official decord FPS')
        groups = {}
        for video, rate in zip(videos, fps): groups.setdefault(rate, []).append(video)
        return list(groups.values())

    def run_dimension_batch(self, videos, dimension, attempt):
        """Load a metric once for a bounded batch, retaining unmodified upstream output.

        Safe opaque aliases avoid upstream's underscore-based case grouping. The
        extraction uses the same official entries averaged by the single-case CLI.
        Each subprocess exits before the next metric, releasing CUDA allocations.
        """
        import shutil
        attempt = Path(attempt)
        inputs, raw = attempt / 'input', attempt / 'raw'
        inputs.mkdir(); raw.mkdir()
        aliases = {}
        for video in videos:
            alias = 'c' + canonical_sha(video['case_id']) + '-0'
            if alias in aliases: raise EvaluationError('Duplicate batch case')
            aliases[alias] = video
            target = inputs / (alias + '.mp4')
            shutil.copyfile(video['path'], target)
            if sha256_file(target) != video['sha256']:
                raise EvaluationError('Copied scoring input changed')
        command = [str(self.path(self.config['python'], resolve_symlinks=False)), '-B',
                   str(self.path(self.config['official_root']) / ENTRY), '--videos_path', str(inputs),
                   '--output_path', str(raw), '--dimension', dimension, '--mode', 'long_custom_input', '--dev_flag']
        env = self.environment()
        write_new(attempt / 'invocation.json', dict(argv=command, cwd=str(self.path(self.config['official_root'])),
            environment=env, aliases=aliases, extraction='official-single-case-aggregate-v1'))
        started = datetime.now(timezone.utc).isoformat()
        try:
            with (attempt / 'stdout.log').open('x') as stdout, (attempt / 'stderr.log').open('x') as stderr:
                process = subprocess.run(command, cwd=self.path(self.config['official_root']), env=env,
                    stdout=stdout, stderr=stderr, timeout=float(self.config.get('timeout_seconds', 3600)), check=False)
            write_new(attempt / 'process.json', dict(exit_code=process.returncode, started_at=started,
                ended_at=datetime.now(timezone.utc).isoformat()))
            if process.returncode: raise EvaluationError(f'Official dimension batch exited {process.returncode}: {attempt}')
        except (OSError, subprocess.TimeoutExpired) as exc:
            write_new(attempt / 'process.json', dict(exit_code=None, started_at=started, error=str(exc)))
            raise
        for alias, video in aliases.items():
            if (sha256_file(inputs / (alias + '.mp4')) != video['sha256']
                    or sha256_file(Path(video['path'])) != video['sha256']):
                raise EvaluationError('Original/copied batch MP4 changed during evaluation')
        results = list(raw.glob('results_*_eval_results.json'))
        if len(results) != 1: raise EvaluationError('Expected exactly one original dimension result')
        scores = parse_batch_result(results[0], inputs, dimension, aliases)
        return dict(scores=scores, raw_result=dict(path=str(results[0]), sha256=sha256_file(results[0])))


def parse_batch_result(path, inputs, dimension, aliases):
    """Reproduce the pinned CLI aggregate for each isolated original video.

    Long returns clip averages at index 2, except fused consistency at index 1.
    Imaging is in 0..100 there, and the upstream aggregate divides it by 100.
    Do not average index 1 for the other dimensions: upstream index 2 is decisive.
    """
    from .quality import finite
    if dimension not in DIMENSIONS: raise EvaluationError('Unknown dimension')
    data = unique_json(Path(path).read_bytes())
    if set(data) != {dimension}: raise EvaluationError('Unexpected batch dimensions')
    value = data[dimension]
    expected_length = 2 if dimension in ('subject_consistency', 'background_consistency') else 3
    if not isinstance(value, list) or len(value) != expected_length or not finite(value[0]):
        raise EvaluationError('Invalid official batch aggregate')
    entries = value[1] if expected_length == 2 else value[2]
    if not isinstance(entries, list) or not entries: raise EvaluationError('Empty batch details')
    grouped = {alias:[] for alias in aliases}
    seen = set()
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get('video_path'), str) or not finite(entry.get('video_results')):
            raise EvaluationError('Invalid official batch video result')
        video_path = Path(entry['video_path']).resolve()
        # Upstream synthesizes a nonexistent path below the clip pathname. It
        # still must lie under this input tree and its exact alias folder.
        if not video_path.is_relative_to(Path(inputs).resolve()) or str(video_path) in seen:
            raise EvaluationError('Foreign or duplicate batch detail')
        seen.add(str(video_path))
        matching = [a for a in aliases if a in video_path.parts or video_path.name == a + '.mp4']
        if len(matching) != 1: raise EvaluationError('Ambiguous official batch case mapping')
        grouped[matching[0]].append(entry['video_results'])
    divisor = 100 if dimension == 'imaging_quality' else 1
    expected = sum(e['video_results'] for e in entries) / len(entries) / divisor
    if not math.isclose(expected, value[0], rel_tol=1e-12, abs_tol=1e-12):
        raise EvaluationError('Official batch aggregate does not match original entries')
    if any(not values for values in grouped.values()): raise EvaluationError('Missing batch video')
    return {aliases[a]['case_id']:sum(values) / len(values) / divisor for a,values in grouped.items()}
