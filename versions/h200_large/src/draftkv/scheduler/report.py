"""Detailed five-case receipts and paired quality; missing evidence stays null."""

from collections import Counter
import json

from draftkv.registry import Registry
from .receipts import official_case_verified, verified_success


def status(queue):
    tasks = queue.tasks()
    matrix = []
    for group in Registry().groups:
        members = [t for t in tasks if (t["benchmark"], t["backbone"], t["algorithm"]) == group.key]
        cases = []
        for task in members:
            result = json.loads(task["result"]) if task["result"] else None
            valid = verified_success(task)
            evaluation = result.get("evaluation", {}) if result else {}
            cases.append(dict(case_id=task["case_id"], task_id=task["id"], state=task["state"], reason=task["reason"],
                              attempt_token=task["attempt"], config_sha256=task["config_sha"],
                              video_e2e_pass=valid, evaluation_state=evaluation.get("state"),
                              metrics=evaluation.get("metrics") if official_case_verified(task) else None,
                              video=result.get("video") if result else None, trace=result.get("trace") if result else None,
                              exit_code=result.get("exit_code") if result else None))
        intent = queue.db.execute("SELECT * FROM intents WHERE group_id=?", ("/".join(group.key),)).fetchone()
        matrix.append(dict(**group.to_dict(), cases=cases, count_by_state=dict(Counter(t["state"] for t in members)),
                           intent=dict(intent) if intent else None, official_benchmark_pass=False,
                           note="Full official benchmark acceptance requires full-scope official aggregation"))
    paired = []
    for row in matrix:
        native = next(r for r in matrix if r["benchmark"] == row["benchmark"] and r["backbone"] == row["backbone"] and r["algorithm"] == "native")
        baseline = {c["case_id"]: c for c in native["cases"]}
        for case in row["cases"]:
            reference = baseline.get(case["case_id"], {})
            metrics, native_metrics = case["metrics"], reference.get("metrics")
            paired.append(dict(benchmark=row["benchmark"], backbone=row["backbone"], algorithm=row["algorithm"],
                               case_id=case["case_id"], task_id=case["task_id"], native_task_id=reference.get("task_id"),
                               quality=metrics, native_quality=native_metrics,
                               delta={k: metrics[k] - native_metrics[k] for k in metrics.keys() & native_metrics.keys()}
                               if metrics and native_metrics else None))
    return dict(schema="draftkv.scheduler.release_matrix.v1", groups=matrix, paired_quality=paired,
                task_count=len(tasks), counts=dict(Counter(t["state"] for t in tasks)),
                started_task_ids=[r[0] for r in queue.db.execute("SELECT DISTINCT task_id FROM attempts WHERE child IS NOT NULL")],
                not_started_task_ids=[t["id"] for t in tasks if not queue.db.execute("SELECT 1 FROM attempts WHERE task_id=? AND child IS NOT NULL", (t["id"],)).fetchone()],
                leases=[dict(r) for r in queue.db.execute("SELECT * FROM leases")])
