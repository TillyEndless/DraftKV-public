"""Persistent evaluation-only queues reuse verified generation without a rerun."""

import json
from pathlib import Path

from .common import atomic_json, canonical, digest, object_hash, read_json
from .gates import pinned
from .receipts import evaluator_preflight, official_case_verified, verified_success
from .store import Queue


def source_task(config):
    source = read_json(pinned(config["reuse"]["source_task"]))
    if not verified_success(source):
        raise ValueError("evaluation source is not a hash-verified VIDEO_E2E_PASS")
    original = json.loads(source["config"])
    for name in ("benchmark", "backbone", "algorithm", "case_id", "manifest_sha256", "selection_sha256", "seed"):
        if config.get(name) != original.get(name):
            raise ValueError("evaluation cannot change generation identity/conditioning")
    if source["id"] != config["reuse"]["task_id"]:
        raise ValueError("evaluation source Task ID mismatch")
    return source


def preflight_scoring(config):
    from .manifests import validate_manifest
    validate_manifest(config)
    source_task(config)
    evaluator_preflight(config)
    return {"source_verified": True, "official_evaluator_verified": True}


def enqueue_scoring(parent, task_id, evaluator_config, evaluator_sha):
    source = parent.task(task_id)
    if not verified_success(source):
        raise ValueError("evaluation-only submission requires verified successful generation")
    if digest(evaluator_config) != evaluator_sha:
        raise ValueError("evaluation config pin mismatch")
    spec = read_json(evaluator_config)
    identity = object_hash([task_id, evaluator_sha])[:24]
    workspace = parent.root / "scoring" / identity
    workspace.mkdir(parents=True, exist_ok=True)
    snapshot = workspace / "source_task.json"
    if snapshot.exists():
        old = read_json(snapshot)
        if old["config_sha"] != source["config_sha"] or json.loads(old["result"])["video"] != json.loads(source["result"])["video"]:
            raise ValueError("evaluation source changed")
    else:
        atomic_json(snapshot, source)
    config = json.loads(source["config"])
    memory = spec["memory"]
    config.update(mode="evaluation_only", evaluator=spec, blocked_configuration=False,
                  sp=memory["sp"], memory={"peak_mib": memory["peak_mib"], "margin_mib": memory["margin_mib"]},
                  reuse={"parent_workspace": str(parent.root), "task_id": task_id,
                         "source_task": {"path": str(snapshot), "sha256": digest(snapshot)},
                         "evaluator_config_sha256": evaluator_sha})
    queue = Queue(workspace)
    identifier = queue.submit(config)
    parent.audit(task_id, "EVALUATION_ONLY_SUBMITTED", {"workspace": str(workspace), "task_id": identifier, "evaluator_sha256": evaluator_sha})
    return queue, identifier


def publish_scoring(queue, config, result):
    """Preserve the generation config/receipt and attach separately pinned scores."""
    source = source_task(config)
    if result["evaluation"]["state"] != "OFFICIAL_CASE_EVALUATION_PASS":
        return
    own = queue.task(source["id"])
    if not official_case_verified(own) or json.loads(own["result"])["evaluation"] != result["evaluation"]:
        raise ValueError("evaluation-only result is not a verified, persisted official score")
    parent = Queue(config["reuse"]["parent_workspace"])
    with parent.transaction():
        current = parent.task(source["id"])
        if not verified_success(current) or current["config_sha"] != source["config_sha"]:
            raise ValueError("parent generation changed during evaluation")
        original = json.loads(current["result"])
        if original["video"] != json.loads(source["result"])["video"]:
            raise ValueError("parent video changed during evaluation")
        original["evaluation"] = result["evaluation"]
        original["evaluation_attempt"] = {"workspace": str(queue.root), "token": result["attempt_token"],
                                          "config_sha256": result["config_sha256"]}
        parent.db.execute("UPDATE tasks SET result=?,reason=? WHERE id=?",
                          (canonical(original), "OFFICIAL_CASE_EVALUATION_PASS", source["id"]))
        parent.audit(source["id"], "EVALUATION_ONLY_COMPLETE", original["evaluation_attempt"])
    parent.db.close()
