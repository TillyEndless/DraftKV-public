"""Runtime hooks delegate to pinned official evaluators and signed evidence gates."""
from __future__ import annotations

import hashlib
import fcntl
import json
import sys
from pathlib import Path

from draftkv.benchmarks.datasets import DatasetError, require_sha, sha256_file, unique_json
from draftkv.benchmarks.orbia import ORBIADatasetResolver, ORBIASpec
from draftkv.benchmarks.orbia_source import verify_source
from .official import DependencyPin, EvaluationError, EvaluationInputs, HookSpec, OfficialEvaluator
from .orbia import ORBIAEvaluatorSpec, ORBIAOfficialEvaluator, ORBIAVideos


def _json_bytes(doc):
    return (json.dumps(doc, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _pinned(context, spec, label):
    if not isinstance(spec, dict):
        raise EvaluationError(f"{label} requires a path and externally reviewed SHA256")
    path = Path(spec.get("path", ""))
    path = (path if path.is_absolute() else context.root / path).resolve()
    if sha256_file(path) != require_sha(spec.get("sha256"), label):
        raise EvaluationError(f"{label} SHA256 mismatch")
    return path


def _reference(context):
    # A separate, reproducible envelope binds the actual native frozen source.
    # It never edits the original manifest or claims original-leaderboard input.
    return dict(schema_version=1, benchmark="bench1", dataset_family="bench1",
                backbone=context.config["backbone"].lower(),
                frozen_case_ids=context.dataset["case_ids"],
                original_manifest_sha256=context.dataset["manifest_sha256"],
                input_mode=context.dataset["protocol"]["input_mode"])


def _bench1_evaluator(context, spec):
    protocol = spec.get("protocol", {})
    hook = spec.get("hook")
    adapter = None
    if hook:
        adapter = HookSpec(_pinned(context, hook, "official adapter"), hook["sha256"],
                           tuple(DependencyPin(**d) for d in hook.get("dependencies", [])))
    return OfficialEvaluator(context.root, "vbench-long",
        protocol=_pinned(context, protocol, "official protocol") if protocol else None,
        protocol_sha256=protocol.get("sha256"), hook=adapter)


def _orbia(context, spec):
    config = context.config.get("dataset_config", {})
    root = context.dataset["root"]
    dataset = ORBIADatasetResolver(context.root).resolve(ORBIASpec(
        root, config.get("index"), config.get("index_sha256"), "references"))
    if dataset.manifest_sha256 != context.state.dataset_sha:
        raise EvaluationError("Official ORBIA dataset differs from runtime generation")
    if not spec.get("official_root") or not spec.get("config"):
        raise EvaluationError("Official ORBIA checkout and pinned evaluation.local.yaml required")
    evaluation_config = _pinned(context, spec["config"], "official ORBIA config")
    mapping = context.config.get("device_mapping", {})
    devices = ",".join(map(str, context.state.selected_devices))
    if mapping:
        ranks = mapping.get("ranks", [])
        if ([r.get("physical_id") for r in ranks] != context.state.selected_devices
                or len({r.get("uuid") for r in ranks}) != len(ranks)
                or mapping.get("cuda_visible_devices") != ",".join(r.get("uuid", "") for r in ranks)):
            raise EvaluationError("ORBIA GPU UUID inventory differs from explicit --devices")
        # Upstream sets CUDA_VISIBLE_DEVICES separately for every worker. Pass
        # inventory UUIDs, rather than mistaking nvidia-smi IDs for PCI ordinals
        # or indices inside the runtime's already restricted UUID mask.
        devices = mapping["cuda_visible_devices"]
    evaluator = ORBIAOfficialEvaluator(ORBIAEvaluatorSpec(
        official_root=spec["official_root"], config=evaluation_config,
        python=spec.get("python", sys.executable),
        devices=devices, threads=spec.get("threads", 4),
        tiers=tuple(spec.get("tiers", ("T1", "T2", "T3", "T4", "T5", "T6"))),
        stage=spec.get("stage", "full"),
        installation_receipt=_pinned(context, spec["installation_receipt"], "ORBIA installation receipt") if spec.get("installation_receipt") else None,
        installation_receipt_sha256=spec.get("installation_receipt", {}).get("sha256")), ffprobe=spec.get("ffprobe", "ffprobe"),
        ffmpeg=spec.get("ffmpeg", "ffmpeg"))
    return dataset, evaluator


def preflight(context):
    spec = context.config.get("evaluation_config", {})
    if not spec:
        return dict(status="BLOCKED_DEPENDENCY" if context.config["benchmark"] == "bench2" else "BLOCKED_PROTOCOL",
                    reasons=["No reviewed evaluation_config: official protocol, adapter/dependencies or ORBIA CUDA environment required"], scores=None)
    if context.config["benchmark"] == "bench1" and spec.get("backend") == "official_vbench_long":
        from .bench1 import Bench1Evaluator
        return Bench1Evaluator(context.root, spec).preflight()
    if context.config["benchmark"] == "bench1":
        evaluator = _bench1_evaluator(context, spec)
        digest = hashlib.sha256(_json_bytes(_reference(context))).hexdigest()
        return evaluator.protocol_preflight(digest).as_dict()
    try:
        dataset, evaluator = _orbia(context, spec)
        source = verify_source(evaluator.spec.official_root, evaluator.lock)
        installation = evaluator.verify_installation()
        # Dependency-only preflight needs no generated video. It executes the
        # official-environment probe in its configured interpreter, never a model.
        probe = evaluator._dependency_probe(source)
        return dict(status="BLOCKED_DEPENDENCY" if probe["reasons"] else "READY",
                    reasons=probe["reasons"], provenance=dict(probe, **installation), scores=None)
    except (EvaluationError, DatasetError, OSError) as exc:
        return dict(status="BLOCKED_DEPENDENCY", reasons=[str(exc)], scores=None)


def evaluate(context):
    from draftkv.runtime.config import json_sha256
    from draftkv.runtime.engine import verify_generation_files
    spec = context.config.get("evaluation_config", {})
    resumable_bench1 = context.config["benchmark"] == "bench1" and spec.get("backend") == "official_vbench_long"
    if not resumable_bench1:
        report = preflight(context)
        if report["status"] != "READY":
            return report
    verify_generation_files(context)
    generation = context.generation
    if generation.get("dataset_sha256") != context.state.dataset_sha:
        raise EvaluationError("Generation dataset provenance differs from evaluator inputs")
    artifacts = generation.get("artifacts", [])
    ids = [row["case_id"] for row in artifacts]
    if ids != context.state.case_ids:
        raise EvaluationError("Generation artifact case order differs from runtime selection")
    spec = context.config["evaluation_config"]
    if context.config["benchmark"] == "bench1":
        if spec.get("backend") == "official_vbench_long":
            from .bench1 import Bench1Evaluator
            # Store attempts with the generation, so a new evaluate run can reuse
            # valid per-case scores after a previous pipeline scoring failure.
            output = (context.root / "runs" / "bench1" / context.config["backbone"] /
                      context.config["algorithm"] / generation["run_id"] / "bench1_evaluation")
            result = Bench1Evaluator(context.root, spec).evaluate(context.dataset, generation,
                context.config["backbone"], output, metadata=dict(
                    git_sha=generation.get("git_sha"), weight_hashes=generation.get("weight_hashes"),
                    request_sha=generation.get("request_sha"), config_sha=context.state.config_sha,
                    method=context.config["algorithm"]))
            return dict(result, run_id=context.state.run_id, generation_sha256=json_sha256(generation))
        if ids != context.dataset["case_ids"]:
            return dict(status="BLOCKED_PROTOCOL", reasons=["VBench-Long official receipt requires all frozen backbone cases"], scores=None)
        reference = context.run_dir / "official_reference.json"
        videos = context.run_dir / "official_videos.json"
        reference.write_bytes(_json_bytes(_reference(context)))
        # Paths are relative to one declared common root, preserving exact bytes.
        import os
        root = Path(os.path.commonpath([str(Path(row["path"]).resolve().parent) for row in artifacts]))
        doc = dict(schema_version=1, benchmark="bench1", dataset_family="bench1",
                   reference_manifest_sha256=sha256_file(reference), videos=[
                       dict(case_id=row["case_id"], path=str(Path(row["path"]).resolve().relative_to(root)),
                            sha256=row["sha256"]) for row in artifacts])
        videos.write_bytes(_json_bytes(doc))
        inputs = EvaluationInputs(reference, sha256_file(reference), videos, sha256_file(videos), root)
        result = _bench1_evaluator(context, spec).evaluate(inputs, context.run_dir / "official_evaluation")
    else:
        # A legacy explicit manifest is optional, but an incorrect user-supplied
        # pin must fail rather than being silently replaced by auto assembly.
        if spec.get("videos"):
            supplied = _pinned(context, spec["videos"], "ORBIA native video manifest")
            supplied_doc = unique_json(supplied.read_bytes())
            supplied_root = Path(spec.get("video_root", ""))
            supplied_root = (supplied_root if supplied_root.is_absolute() else context.root / supplied_root).resolve()
            actual = [(r.get("case_id"), str((supplied_root / r.get("path", "")).resolve()), r.get("sha256"))
                      for r in supplied_doc.get("videos", [])]
            expected = [(r["case_id"], str(Path(r["path"]).resolve()), r["sha256"]) for r in artifacts]
            if actual != expected:
                raise EvaluationError("ORBIA video manifest does not bind this generation's exact artifacts")
        dataset, evaluator = _orbia(context, spec)
        from .orbia_video import build_generation_manifest
        # The generation run owns scoring state even when `evaluate` is invoked
        # in a new runtime run. Official worker resume never invokes generation.
        identity = json_sha256(dict(generation_sha256=json_sha256(generation),
            evaluation_config=spec, devices=context.state.selected_devices))
        output = (context.root / "runs/bench2" / context.config["backbone"] /
                  context.config["algorithm"] / generation["run_id"] / "bench2_evaluation" / identity)
        output.mkdir(parents=True, exist_ok=True)
        with (output / "scoring.lock").open("a+") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            binding = build_generation_manifest(context, dataset, output, ffprobe=evaluator.ffprobe, ffmpeg=evaluator.ffmpeg)
            videos = ORBIAVideos(binding["path"], binding["sha256"], binding["root"])
            attempt = output / "official_evaluation"
            result = evaluator.evaluate(dataset, videos, attempt, resume=attempt.exists())
            derived_fields = {}
            if result.status.value == "COMPLETE" and evaluator.spec.stage == "full":
                from .orbia_short import calculate_from_file, verify_short_video_scores
                raw_scores = attempt / "official/scores/scores.json"
                sidecar = attempt / "short_video_scores.json"
                # Older immutable COMPLETE archives can be read and derived
                # without rewriting any archived original receipt.
                derived = (verify_short_video_scores(raw_scores, attempt)
                           if sidecar.is_file() else calculate_from_file(raw_scores))
                derived_fields = dict(short_video_scores=derived,
                                      short_video_scores_path=str(sidecar) if sidecar.is_file() else None,
                                      short_video_scores_sha256=sha256_file(sidecar) if sidecar.is_file() else None)
            return dict(**result.as_dict(), **derived_fields, run_id=context.state.run_id,
                        generation_sha256=json_sha256(generation),
                        video_manifest=binding, evaluation_identity=identity,
                        report_path=str(attempt / "evaluation_report.json"),
                        report_sha256=sha256_file(attempt / "evaluation_report.json"))
    return dict(**result.as_dict(), run_id=context.state.run_id,
                generation_sha256=json_sha256(generation))


def audit(context):
    """A runtime success cannot replace independently collected signed evidence."""
    from draftkv.runtime.config import json_sha256
    spec = context.config.get("audit_config", {}).get("record")
    if not spec:
        return dict(status="BLOCKED_PROVENANCE", reasons=[
            "Signed current-source tools.audit.v2 official run record and collector policy required"], scores=None)
    from tools.audit.audit import Auditor
    from tools.audit.provenance import git
    from tools.audit.release import POLICY, verify_isolation, verify_signature
    path = _pinned(context, spec, "signed release record")
    record = unique_json(path.read_bytes())
    policy = json.loads(git(context.root, "show", f"HEAD:{POLICY}"))
    if policy != unique_json((context.root / POLICY).read_bytes()):
        raise EvaluationError("Release trust policy differs from the committed policy")
    verify_signature(context.root, path, record, policy)
    checker = Auditor(context.root, revision=context.state.git_sha, evaluator_pins=policy.get("evaluators", {}))
    verify_isolation(checker, record, path.parent)
    checker.validate(record, "OFFICIAL_BENCHMARK_PASS")
    expected = dict(runtime_run_id=context.state.run_id,
                    generation_sha256=json_sha256(context.generation),
                    evaluation_sha256=json_sha256(context.results["evaluate"]))
    if record.get("runtime_lineage") != expected or any(record.get(k) != context.config[k]
            for k in ("benchmark", "backbone", "algorithm")):
        raise EvaluationError("Signed official record does not bind this runtime generation/evaluation/group")
    return dict(status="PASS", run_id=context.state.run_id,
                generation_sha256=expected["generation_sha256"],
                evaluation_sha256=expected["evaluation_sha256"], signed_record_sha256=spec["sha256"])


def audit_bench2_workflow(context):
    """Post-scoring byte/source integrity; independent signed audit stays above."""
    from draftkv.runtime.config import json_sha256
    from draftkv.runtime.engine import verify_generation_files
    from .orbia_video import build_generation_manifest, execution_scope

    if context.config.get("benchmark") != "bench2":
        raise EvaluationError("Bench2 workflow audit requires ORBIA")
    verify_generation_files(context)
    generation, result = context.generation, context.results["evaluate"]
    if (result.get("status") != "COMPLETE" or result.get("run_id") != context.state.run_id
            or result.get("generation_sha256") != json_sha256(generation)):
        raise EvaluationError("ORBIA evaluation does not bind this runtime generation")
    spec = context.config["evaluation_config"]
    dataset, evaluator = _orbia(context, spec)
    if evaluator.spec.stage != "full" or evaluator.spec.tiers != ("T1", "T2", "T3", "T4", "T5", "T6"):
        raise EvaluationError("Workflow requires the complete official tier evaluation")
    identity = json_sha256(dict(generation_sha256=json_sha256(generation),
        evaluation_config=spec, devices=context.state.selected_devices))
    output = (context.root / "runs/bench2" / context.config["backbone"] / context.config["algorithm"] /
              generation["run_id"] / "bench2_evaluation" / identity).resolve()
    path = output / "official_evaluation/evaluation_report.json"
    if (result.get("evaluation_identity") != identity or result.get("report_path") != str(path)
            or result.get("report_sha256") != sha256_file(path)):
        raise EvaluationError("Workflow official report path/SHA mismatch")
    binding = build_generation_manifest(context, dataset, output, ffprobe=evaluator.ffprobe, ffmpeg=evaluator.ffmpeg)
    if result.get("video_manifest") != binding:
        raise EvaluationError("Workflow generation video manifest changed")
    videos = ORBIAVideos(binding["path"], binding["sha256"], binding["root"])
    saved = evaluator.archived_report(videos, path.parent)
    if saved != {k: result[k] for k in ("status", "reasons", "provenance", "scores")}:
        raise EvaluationError("Workflow result differs from original official report")
    expected_command = evaluator.command(dataset, path.parent / "official_videos.jsonl", path.parent / "official")
    expected_command[expected_command.index("--config") + 1] = str(path.parent / "evaluation.local.yaml")
    if (saved["provenance"].get("command") != expected_command
            or saved["provenance"].get("cwd") != str(Path(evaluator.spec.official_root).resolve())):
        raise EvaluationError("Workflow original official evaluator command differs")
    checked, doc = evaluator._check(dataset, videos)
    if checked.status.value != "READY":
        raise EvaluationError("Workflow official dependency/reference postflight failed: " + "; ".join(checked.reasons))
    for key, value in checked.provenance.items():
        if saved["provenance"].get(key) != value:
            raise EvaluationError("Workflow official provenance changed: " + key)
    if (doc.get("generation_artifacts") != generation["artifacts"]
            or doc.get("generation_sha256") != json_sha256(generation)):
        raise EvaluationError("Workflow native MP4 manifest is not bound to generation")
    scope = execution_scope(generation["case_ids"], dataset.case_ids)
    scores = saved["scores"]
    if (scores["rows"][0]["T6"] is not None or scores["rows"][0]["Overall"] is not None):
        raise EvaluationError("Short-only ORBIA scores must preserve null T6/Overall")
    from .orbia_short import calculate_from_file, verify_short_video_scores
    official_score_path = path.parent / "official/scores/scores.json"
    derived_path = path.parent / "short_video_scores.json"
    expected_short = (verify_short_video_scores(official_score_path, path.parent)
                      if derived_path.is_file() else calculate_from_file(official_score_path))
    if (result.get("short_video_scores") != expected_short
            or result.get("short_video_scores_path") != (str(derived_path) if derived_path.is_file() else None)
            or result.get("short_video_scores_sha256") != (
                sha256_file(derived_path) if derived_path.is_file() else None)):
        raise EvaluationError("Workflow derived short-video T6/Overall mismatch")
    verify_generation_files(context)
    return dict(status="PASS", scope="runtime_workflow_integrity", execution_scope=scope,
                release_acceptance=False, official_720_case_leaderboard=False,
                run_id=context.state.run_id, generation_sha256=json_sha256(generation),
                evaluation_sha256=json_sha256(result), report_sha256=sha256_file(path),
                valid_scored_count=len(generation["case_ids"]), formal_benchmark_complete=scope == "full480")


def audit_bench1_workflow(context):
    """Verify this pipeline's bytes and official receipts, without release claims.

    The builder selects this explicitly for official_vbench_long. The signed
    tools.audit.v2 release path above retains its separate trust/signature gates.
    """
    import math
    from draftkv.runtime.config import json_sha256
    from draftkv.runtime.engine import verify_generation_files
    from .bench1 import Bench1Evaluator, verified_inputs
    from .vbench_long import VBenchLong, DIMENSIONS, ENTRY, canonical_sha

    spec = context.config.get("evaluation_config", {})
    if context.config.get("benchmark") != "bench1" or spec.get("backend") != "official_vbench_long":
        raise EvaluationError("Bench1 workflow audit requires the official VBench-Long backend")
    verify_generation_files(context)
    generation, evaluation = context.generation, context.results["evaluate"]
    generation_sha = json_sha256(generation)
    frozen_ids, rows = verified_inputs(context.dataset, generation, context.config["backbone"])
    if evaluation.get("run_id") != context.state.run_id or evaluation.get("generation_sha256") != generation_sha:
        raise EvaluationError("Workflow evaluation does not bind this runtime/generation")
    path = Path(evaluation.get("summary_path", "")).resolve()
    if sha256_file(path) != evaluation.get("summary_sha256"):
        raise EvaluationError("Workflow evaluation summary changed")
    summary = unique_json(path.read_bytes())
    if {k:v for k,v in evaluation.items() if k not in ("summary_path", "summary_sha256", "run_id")} != summary:
        raise EvaluationError("Workflow evaluation differs from the original summary")
    if summary.get("schema") == "draftkv.bench1.quality.summary.v2":
        from .quality_pipeline import audit_quality_summary
        if (summary.get("generation_sha256") != generation_sha
                or summary.get("dataset_sha256") != generation["dataset_sha256"]
                or summary.get("backbone") != context.config["backbone"]
                or summary.get("method") != context.config["algorithm"]
                or summary.get("requested_case_ids") != generation["case_ids"]
                or summary.get("status") != "COMPLETE" or summary.get("failures") != []
                or summary.get("release_acceptance") is not False or summary.get("overall") is not None):
            raise EvaluationError("Quality workflow generation/protocol mismatch")
        expected_metadata = dict(git_sha=generation.get("git_sha"), weight_hashes=generation.get("weight_hashes"),
            request_sha=generation.get("request_sha"), config_sha=context.state.config_sha,
            method=context.config["algorithm"], source_rows_sha256=canonical_sha(rows))
        expected_identity = dict(generation_sha256=generation_sha, dataset_sha256=generation["dataset_sha256"],
            backbone=context.config["backbone"], method=context.config["algorithm"],
            evaluation_config_sha256=canonical_sha(spec), metadata=expected_metadata)
        expected_base = (context.root / "runs" / "bench1" / context.config["backbone"] / context.config["algorithm"] /
            generation["run_id"] / "bench1_evaluation" / ("quality-v2-" + canonical_sha(expected_identity)) /
            summary["evaluator_sha256"])
        from draftkv.benchmarks.bench1 import formal_scope
        expected_scope = formal_scope(context.config["backbone"]) if generation["case_ids"] == frozen_ids else "exploratory-subset"
        if (summary["identity"] != expected_identity or path.parent.parent != expected_base.resolve()
                or summary["scope"] != expected_scope):
            raise EvaluationError("Quality summary is outside the native generation/protocol scope")
        complete = audit_quality_summary(summary, spec, generation["artifacts"])
        postflight = Bench1Evaluator(context.root, spec).preflight()
        if postflight.get("status") != "READY" or postflight != summary["provenance"]:
            raise EvaluationError("Quality official runtime changed after scoring")
        verify_generation_files(context)
        if not complete or sha256_file(path) != evaluation["summary_sha256"]:
            raise EvaluationError("Quality workflow incomplete or summary changed")
        return dict(status="PASS", scope="runtime_workflow_integrity", release_acceptance=False,
            run_id=context.state.run_id, generation_sha256=generation_sha,
            evaluation_sha256=json_sha256(evaluation), summary_sha256=evaluation["summary_sha256"],
            valid_scored_count=summary["valid_scored_count"],
            formal_benchmark_complete=generation["case_ids"] == frozen_ids)
    if (summary.get("schema") != "draftkv.bench1.vbench_long.summary.v1"
            or summary.get("status") != "COMPLETE" or summary.get("failures") != []
            or summary.get("evaluator") != "vbench-long" or summary.get("long_video_mode") is not True
            or summary.get("generation_run_id") != generation["run_id"]
            or summary.get("dataset_sha256") != generation["dataset_sha256"]
            or summary.get("frozen_case_ids") != frozen_ids
            or summary.get("requested_case_ids") != generation["case_ids"]
            or summary.get("release_acceptance") is not False or summary.get("overall") is not None):
        raise EvaluationError("Workflow summary has incomplete or different benchmark provenance")
    identity = dict(generation_sha256=generation_sha, evaluator_sha256=summary["evaluator_sha256"],
                    dataset_sha256=generation["dataset_sha256"], backbone=context.config["backbone"])
    output = (context.root / "runs" / "bench1" / context.config["backbone"] / context.config["algorithm"] /
              generation["run_id"] / "bench1_evaluation" / canonical_sha(identity)).resolve()
    if path.parent != output:
        raise EvaluationError("Workflow summary is outside the original generation's scoring directory")
    references = summary.get("case_receipts")
    if not isinstance(references, list) or len(references) != len(generation["case_ids"]):
        raise EvaluationError("Workflow requires one original receipt per generated case")
    backend, receipts = VBenchLong(context.root, spec), []
    metadata = dict(git_sha=generation.get("git_sha"), weight_hashes=generation.get("weight_hashes"),
                    request_sha=generation.get("request_sha"), config_sha=context.state.config_sha)
    for video, reference in zip(generation["artifacts"], references):
        cid = video["case_id"]
        receipt_path = Path(reference["receipt_path"]).resolve()
        case_root = output / "cases" / cid / "attempts"
        if (not receipt_path.is_relative_to(case_root) or receipt_path.name != "receipt.json"
                or receipt_path.parent.parent != case_root or sha256_file(receipt_path) != reference["receipt_sha256"]):
            raise EvaluationError("Workflow case receipt path/SHA differs from the original attempt")
        binding = dict(**identity, case_id=cid, video_sha256=video["sha256"],
                       native_case_row_sha256=canonical_sha(rows[cid]), metadata=metadata)
        receipt = Bench1Evaluator._receipt(receipt_path, binding)
        if (receipt.get("schema") != "draftkv.bench1.vbench_long.case.v1" or receipt.get("evaluator") != "vbench-long"
                or receipt.get("long_video_mode") is not True or receipt.get("original_prompt") != rows[cid]["prompt"]
                or receipt.get("original_prompts") != rows[cid].get("prompts", [rows[cid]["prompt"]])
                or receipt.get("generation_run_id") != generation["run_id"]):
            raise EvaluationError("Workflow case receipt has different native case provenance")
        attempt = receipt_path.parent
        invocation = unique_json((attempt / "invocation.json").read_bytes())
        process = unique_json((attempt / "process.json").read_bytes())
        expected_argv = [str(backend.path(spec["python"], resolve_symlinks=False)), "-B", str(backend.path(spec["official_root"]) / ENTRY),
            "--videos_path", str(attempt / "input"), "--output_path", str(attempt / "raw"),
            "--dimension", *receipt["scores"], "--mode", "long_custom_input", "--dev_flag"]
        environment = backend.environment()
        critical = ("PYTHONPATH", "PYTHONNOUSERSITE", "PYTHONDONTWRITEBYTECODE", "HF_HUB_OFFLINE",
                    "TRANSFORMERS_OFFLINE", "VBENCH_CACHE_DIR", "TORCH_HOME")
        if (invocation.get("argv") != expected_argv or invocation.get("cwd") != str(backend.path(spec["official_root"]))
                or invocation.get("input_video") != video or invocation.get("input_copy_sha256") != video["sha256"]
                or sha256_file(attempt / "input" / (cid + ".mp4")) != video["sha256"]
                or any(invocation.get("environment", {}).get(k) != environment[k] for k in critical)
                or process.get("exit_code") != 0 or process.get("failure_type") is not None):
            raise EvaluationError("Workflow original VBench-Long invocation/input/process differs")
        receipts.append(receipt)
    full = generation["case_ids"] == frozen_ids and len(receipts) == len(frozen_ids)
    means = {d: math.fsum(r["scores"][d] for r in receipts) / len(frozen_ids) for d in DIMENSIONS} if full else None
    from .quality import cohort_quality_fields
    from draftkv.benchmarks.bench1 import formal_scope
    expected_quality = cohort_quality_fields(means, formal=full)
    if any(summary.get(key) != value for key, value in expected_quality.items()):
        raise EvaluationError("Workflow Quality Score differs from original seven benchmark means")
    if (summary.get("valid_scored_count") != len(receipts) or summary.get("formal_mean") != means
            or summary.get("scores") != means or summary.get("scope") != (formal_scope(context.config["backbone"]) if full else "exploratory-subset")):
        raise EvaluationError("Workflow mean/scope differs from all original case results")
    # A postflight rehashes the original source, runtime, dependencies and weights.
    postflight = Bench1Evaluator(context.root, spec).preflight()
    original_provenance = summary.get("provenance", {})
    # The historical v1 CLI listed flickering last. Dimension order does not
    # change any metric, and must not invalidate an otherwise identical lock.
    without_dimensions = lambda value: {k:v for k,v in value.items() if k != "dimensions"}
    if (postflight.get("status") != "READY"
            or without_dimensions(postflight) != without_dimensions(original_provenance)
            or ("dimensions" in original_provenance and set(original_provenance["dimensions"]) != set(DIMENSIONS))
            or canonical_sha(original_provenance) != summary["evaluator_sha256"]):
        raise EvaluationError("Workflow official evaluator source/runtime changed after scoring")
    verify_generation_files(context)
    if (json_sha256(generation) != generation_sha or sha256_file(path) != evaluation["summary_sha256"]):
        raise EvaluationError("Workflow generation/summary changed during integrity audit")
    return dict(status="PASS", scope="runtime_workflow_integrity", release_acceptance=False,
        run_id=context.state.run_id, generation_sha256=generation_sha,
        evaluation_sha256=json_sha256(evaluation), summary_sha256=evaluation["summary_sha256"],
        valid_scored_count=len(receipts), formal_benchmark_complete=full)
