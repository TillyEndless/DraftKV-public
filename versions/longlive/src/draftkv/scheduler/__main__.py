"""python -m draftkv.scheduler --help (all commands use an explicit workspace)."""

import argparse
import json
from pathlib import Path
import time

from .common import atomic_json, read_json
from .manifests import freeze_orbia
from .report import status
from .runner import dispatch, preflight_task, resume, worker
from .store import Queue


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--status", action="store_true")
    actions.add_argument("--dry-run", action="store_true")
    actions.add_argument("--resume", action="store_true")
    actions.add_argument("--retry-failed", metavar="TASK_ID")
    actions.add_argument("--submit", type=Path, metavar="FROZEN_CONFIG")
    actions.add_argument("--intent", type=Path, metavar="BLOCKED_GROUP_CONFIG")
    actions.add_argument("--revise", metavar="TASK_ID")
    actions.add_argument("--score-video", metavar="VERIFIED_TASK_ID")
    actions.add_argument("--run-once", action="store_true")
    actions.add_argument("--daemon", action="store_true")
    actions.add_argument("--worker", metavar="TASK_ID", help=argparse.SUPPRESS)
    actions.add_argument("--freeze-orbia", type=Path, metavar="OFFICIAL_RAW_MANIFEST")
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--token", help=argparse.SUPPRESS)
    parser.add_argument("--repair", type=Path)
    parser.add_argument("--repair-sha256")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--config-sha256")
    parser.add_argument("--previous-sha256")
    parser.add_argument("--reference")
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--provenance", type=Path)
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--interval", type=float, default=30)
    args = parser.parse_args(argv)
    if args.freeze_orbia:
        if not (args.output and args.provenance and args.manifest_sha256):
            parser.error("freeze requires --output, --provenance, --manifest-sha256")
        _, selection = freeze_orbia(args.freeze_orbia, args.manifest_sha256, args.output, args.seed, read_json(args.provenance))
        print(json.dumps(selection, indent=2))
        return 0
    queue = Queue(args.workspace)
    if args.status:
        result = status(queue)
    elif args.submit:
        config = read_json(args.submit)
        task_id = queue.submit(config)
        task = queue.task(task_id)
        if not task["attempt"] and task["state"] not in {"COMPLETE", "FAILED_OOM", "FAILED_RUNTIME", "STALLED"}:
            preflight_task(queue, task)
        result = {"task_id": task_id, "state": queue.task(task_id)["state"]}
    elif args.intent:
        config = read_json(args.intent)
        queue.intent(config, config["blocked_reason"])
        result = {"state": "BLOCKED_INPUT", "reason": config["blocked_reason"]}
    elif args.score_video:
        if not args.config or not args.config_sha256:
            parser.error("--score-video requires pinned evaluator --config and --config-sha256")
        from .scoring import enqueue_scoring
        scoring_queue, identifier = enqueue_scoring(queue, args.score_video, args.config, args.config_sha256)
        result = {"task_id": identifier, "scoring_workspace": str(scoring_queue.root), "generation_reused": True}
    elif args.revise:
        if not args.config or not args.previous_sha256 or not args.reference:
            parser.error("revision requires --config, --previous-sha256 and --reference")
        sha = queue.revise(args.revise, read_json(args.config), args.previous_sha256, args.reference)
        result = {"task_id": args.revise, "config_sha256": sha, "state": queue.task(args.revise)["state"]}
    elif args.retry_failed:
        if not args.repair or not args.repair_sha256:
            parser.error("retry requires a pinned --repair receipt and --repair-sha256")
        queue.retry(args.retry_failed, {"path": str(args.repair.resolve()), "sha256": args.repair_sha256})
        result = {"task_id": args.retry_failed, "state": "PENDING"}
    else:
        if not args.policy:
            parser.error("dispatch/recovery requires --policy")
        policy = read_json(args.policy)
        if args.worker:
            worker(queue, policy, args.worker, args.token)
            return 0 if queue.task(args.worker)["state"] == "COMPLETE" else 1
        if args.resume:
            result = {"recovered": resume(queue, policy)}
        elif args.daemon:
            if not 1 <= args.interval <= 60:
                parser.error("--interval must be between 1 and 60 seconds")
            # A second daemon cannot dispatch duplicate tasks; flock/SQLite
            # serialize ticks. Each live worker remains independently persistent.
            while True:
                resume(queue, policy)
                dispatch(queue, policy)
                atomic_json(queue.root / "release_matrix.json", status(queue))
                time.sleep(args.interval)
        else:
            result = {"started": dispatch(queue, policy, dry_run=args.dry_run)}
    if args.output:
        atomic_json(args.output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
