"""Detached workers survive dispatcher restarts; reconciliation never redispatches live work."""

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .common import atomic_json, canonical, digest, matches, object_hash, process_identity, read_json
from .gates import preflight
from .receipts import evaluate, media_receipt, trace_receipt, trace_rows, verified_success
from .resources import ResourceBlocked, Resources
from .store import ACTIVE, BLOCKED, FAILURES, Queue


@contextmanager
def dispatch_lock(queue):
    with (queue.root / "dispatch.lock").open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def owned_processes(token, *, exclude_self=True):
    result = []
    marker = ("DRAFTKV_ATTEMPT_TOKEN=" + token).encode()
    for path in Path("/proc").iterdir():
        if not path.name.isdigit() or exclude_self and int(path.name) == os.getpid():
            continue
        try:
            if path.stat().st_uid != os.getuid() or marker not in (path / "environ").read_bytes().split(b"\0"):
                continue
            identity = process_identity(int(path.name))
            if identity:
                result.append(identity)
        except (OSError, ValueError):
            continue
    return result


def stop_owned(token, grace=10):
    """Stop only exact UID+token+start-ticks+boot-ID processes.

    Prefer pidfds. On older Python/Linux without pidfd_open/pidfd_send_signal,
    fall back to PID signaling only after re-reading the exact process identity
    immediately before the signal. Never signal a process group or a PID whose
    frozen identity/token no longer matches.
    """
    have_pidfd = callable(getattr(os, "pidfd_open", None)) and callable(
        getattr(signal, "pidfd_send_signal", None))
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for identity in owned_processes(token):
            if not matches(identity) or identity not in owned_processes(token):
                continue
            if have_pidfd:
                try:
                    fd = os.pidfd_open(identity["pid"])
                except ProcessLookupError:
                    continue
                try:
                    if matches(identity) and identity in owned_processes(token):
                        signal.pidfd_send_signal(fd, sig)
                except ProcessLookupError:
                    pass
                finally:
                    os.close(fd)
            else:
                # Last-moment /proc identity check closes the normal PID-reuse
                # window; token ownership is independently re-read above.
                current = process_identity(identity["pid"])
                if current != identity or identity not in owned_processes(token):
                    continue
                try:
                    os.kill(identity["pid"], sig)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + grace
        while owned_processes(token) and time.monotonic() < deadline:
            time.sleep(0.2)
        if not owned_processes(token):
            return
    raise RuntimeError("owned process cleanup incomplete; retain leases")


def classify(log, stalled=False):
    lower = log.lower()
    if any(s in lower for s in ("cuda out of memory", "cuda error: out of memory", "outofmemoryerror",
                                "cuda failure 'out of memory'")):
        return "FAILED_OOM", "OOM"
    if any(s in lower for s in ("ncclerror", "nccl error", "nccl watchdog", "nccl timeout")):
        return "FAILED_RUNTIME", "NCCL_ERROR"
    if any(s in lower for s in ("modulenotfounderror", "undefined symbol", "importerror", "no kernel image", "glibcxx_")):
        return "BLOCKED_ENVIRONMENT", "ENVIRONMENT_ERROR"
    return ("STALLED", "STALLED") if stalled else ("FAILED_RUNTIME", "RUNTIME_ERROR")


def folder_for(queue, task_id, token):
    """Group scheduler logs by benchmark/backbone/method/case before attempt token."""
    task = queue.task(task_id)
    components = [task["benchmark"], task["backbone"], task["algorithm"], task["case_id"], token]
    if any(not isinstance(part, str) or not part or "/" in part or "\\" in part or part in {".", ".."}
           for part in components):
        raise ValueError("unsafe scheduler log path component")
    return (queue.root / "groups" / task["benchmark"] / task["backbone"] / task["algorithm"] /
            task["case_id"] / "attempts" / token)


def set_attempt(queue, token, **fields):
    if not set(fields) <= {"worker", "child", "ended", "exit_code", "diagnostic"}:
        raise ValueError("invalid attempt field")
    queue.db.execute("UPDATE attempts SET " + ",".join(k + "=?" for k in fields) + " WHERE token=?",
                     (*fields.values(), token))


def preflight_task(queue, task):
    config = json.loads(task["config"])
    report = preflight(config, probe=True)
    with queue.transaction():
        queue.transition(task["id"], "PREFLIGHT")
        queue.audit(task["id"], "PREFLIGHT", report)
        queue.transition(task["id"], report["state"], report["reason"])
        if task["state"] == "BLOCKED_ENVIRONMENT" and report["state"] == "READY" and queue.db.execute("SELECT 1 FROM attempts WHERE task_id=?", (task["id"],)).fetchone():
            position = queue.db.execute("SELECT max(position) FROM tasks").fetchone()[0] + 1
            queue.db.execute("UPDATE tasks SET position=? WHERE id=?", (position, task["id"]))
            queue.audit(task["id"], "ENVIRONMENT_REPAIRED_TO_TAIL", report)
    return report


def dispatch(queue, policy, *, dry_run=False):
    with dispatch_lock(queue):
        candidates = []
        for task in queue.tasks():
            if task["state"] == "COMPLETE":
                if not verified_success(task):
                    queue.audit(task["id"], "SUCCESS_HASH_DRIFT", {"action": "BLOCK_REUSE_REQUIRE_REVIEW"})
                continue
            if task["state"] in ACTIVE or task["state"] in FAILURES or task["attempt"]:
                continue
            report = preflight_task(queue, task)
            if report["state"] == "READY":
                candidates.append(queue.task(task["id"]))
        # Smaller SP first, stable FIFO within a device-count tier.
        candidates.sort(key=lambda task: (json.loads(task["config"])["sp"], task["position"]))
        started = []
        for task in candidates:
            config = json.loads(task["config"])
            try:
                from .resources import choose_devices
                choose_devices(Resources(queue, policy).snapshot(dict(config, task_id=task["id"])),
                               config["sp"], 0)
            except Exception as exc:
                with queue.transaction():
                    queue.transition(task["id"], "BLOCKED_GPU", str(exc))
                continue
            if dry_run:
                queue.audit(task["id"], "DRY_RUN_READY", {"no_dispatch": True, "cpu_probe": "deferred"})
                continue
            # One worker per tick; it acquires the real shared lease, rechecks all
            # gates and devices, then launches. Subsequent ticks may use GPU2.
            token = queue.claim(task["id"])
            environment = dict(os.environ, DRAFTKV_ATTEMPT_TOKEN=token)
            folder = folder_for(queue, task["id"], token)
            folder.mkdir(parents=True, exist_ok=False)
            atomic_json(folder / "policy.json", policy)
            with (folder / "worker.stdout.log").open("ab") as out, (folder / "worker.stderr.log").open("ab") as err:
                child = subprocess.Popen([sys.executable, "-m", "draftkv.scheduler", "--workspace", str(queue.root),
                                          "--worker", task["id"], "--token", token, "--policy", str(folder / "policy.json")],
                                         env=environment, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                         start_new_session=True, close_fds=True)
            queue.audit(task["id"], "WORKER_SPAWNED", {"token": token, "pid": child.pid})
            started.append(task["id"])
            break
        return started


def worker(queue, policy, task_id, token):
    if os.environ.get("DRAFTKV_ATTEMPT_TOKEN") != token or process_identity(os.getpid()) is None:
        raise RuntimeError("Linux owned worker identity required")
    folder = folder_for(queue, task_id, token)
    folder.mkdir(parents=True, exist_ok=True)
    task_locks = Path(policy["task_lock_root"]).resolve()
    task_locks.mkdir(parents=True, exist_ok=True)
    with (folder / "worker.lock").open("a+") as lock, (task_locks / (task_id + ".lock")).open("a+") as task_lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            fcntl.flock(task_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            with queue.transaction():
                queue.transition(task_id, "BLOCKED_GPU", "same Task ID already executing in another scheduler queue", token=token)
                set_attempt(queue, token, ended=time.time(), exit_code=1)
                queue.db.execute("UPDATE tasks SET attempt=NULL WHERE id=?", (task_id,))
            return
        return _worker(queue, policy, task_id, token, folder)


def _worker(queue, policy, task_id, token, folder):
    resource = Resources(queue, policy)
    with queue.transaction():
        task = queue.task(task_id)
        if task["attempt"] != token or task["state"] != "READY":
            raise ValueError("worker does not own a fresh ready task")
        set_attempt(queue, token, worker=canonical(process_identity(os.getpid())))
    config = json.loads(task["config"])
    atomic_json(folder / "frozen_config.json", config)
    request = dict(task_id=task_id, attempt_token=token, config_sha256=task["config_sha"], config=config,
                   output=str(folder / "video.mp4"), trace=str(folder / "native_trace.jsonl"))
    atomic_json(folder / "request.json", request)
    held, result, last_exit = False, None, None
    phase = "PREFLIGHT"
    try:
        report = preflight(config, probe=True)
        atomic_json(folder / "preflight.json", report)
        if report["state"] != "READY":
            with queue.transaction():
                queue.transition(task_id, report["state"], report["reason"], token=token)
            return
        devices = resource.acquire(config, task_id, token)
        held = True
        request["gpus"] = devices
        atomic_json(folder / "request.json", request)
        with queue.transaction():
            queue.transition(task_id, "RUNNING", token=token)

        def run(argv, stage):
            nonlocal last_exit, phase
            phase = stage
            with queue.transaction():
                queue.transition(task_id, stage, token=token)
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(str(g["index"]) for g in devices),
                               DRAFTKV_ATTEMPT_TOKEN=token, DRAFTKV_TASK_ID=task_id)
            # Adapter executes the frozen request; no shell or command fallback.
            with (folder / (stage.lower() + ".stdout.log")).open("ab") as out, (folder / (stage.lower() + ".stderr.log")).open("ab") as err:
                process = subprocess.Popen(argv, cwd=config["repo_root"], env=environment, stdin=subprocess.DEVNULL,
                                           stdout=out, stderr=err, start_new_session=True, close_fds=True)
            identity = process_identity(process.pid)
            if identity is None:
                process.wait()
                raise RuntimeError("child exited before process identity could be frozen")
            with queue.transaction():
                set_attempt(queue, token, child=canonical(identity))
                queue.audit(task_id, "PROCESS_STARTED", {"token": token, "identity": identity, "argv": argv, "stage": stage})
            progress_at = time.monotonic()
            maximum_seq, last_progress, renewed = -1, None, time.monotonic()
            timeout = config["timeouts"]["generation_no_progress_s" if stage == "GENERATING" else "evaluation_max_s"]
            if type(timeout) not in (int, float) or timeout <= 0:
                raise ValueError("positive explicit timeout required")
            while process.poll() is None:
                if stage == "GENERATING":
                    previous_seq = maximum_seq
                    for row in trace_rows(request["trace"], task_id, task["config_sha"], token, partial=True):
                        seq = row.get("progress_seq")
                        if row.get("event") in ("kernel", "chunk", "forward") and type(seq) is int and seq > maximum_seq:
                            maximum_seq, last_progress, progress_at = seq, row, time.monotonic()
                    if maximum_seq > previous_seq:
                        queue.audit(task_id, "PROGRESS", {"token": token, "last_actual_kernel_or_chunk": last_progress})
                if time.monotonic() - progress_at > timeout:
                    diagnostic = dict(category="STALLED", stage=stage, last_actual_kernel_or_chunk=last_progress,
                                      seconds_without_progress=time.monotonic() - progress_at,
                                      resources=resource.snapshot(dict(config, task_id=task_id)),
                                      stderr=(folder / (stage.lower() + ".stderr.log")).read_text(errors="replace")[-24000:])
                    atomic_json(folder / "stall_diagnostic.json", diagnostic)
                    queue.audit(task_id, "NO_PROGRESS", diagnostic)
                    raise TimeoutError("no verified kernel/chunk progress; full diagnostic retained")
                if time.monotonic() - renewed > 60:
                    resource.renew(token)
                    renewed = time.monotonic()
                time.sleep(min(1, timeout / 10))
            last_exit = process.wait()
            if owned_processes(token):
                raise RuntimeError("adapter left live descendant processes")
            return last_exit

        if config.get("mode") == "evaluation_only":
            from .scoring import source_task
            original = json.loads(source_task(config)["result"])
            trace = original["trace"]
            video = media_receipt(original["video"]["path"], config)
            if video["sha256"] != original["video"]["sha256"]:
                raise ValueError("evaluation video hash drift")
            with queue.transaction():
                queue.transition(task_id, "GENERATING", "REUSED_VERIFIED_GENERATION_NO_FORWARD", token=token)
            request["reused_generation"] = original
        else:
            values = {"request": str(folder / "request.json"), "output": request["output"], "trace": request["trace"],
                      "python": config["environment"]["python"], "adapter": config["runtime"]["adapter"]["path"]}
            from draftkv.runtime.adapters import render_command
            argv = render_command(config["runtime"]["argv"], values)
            if run(argv, "GENERATING"):
                raise RuntimeError(f"native generator exit {last_exit}")
            trace = trace_receipt(request["trace"], config, task_id, token)
            video = media_receipt(request["output"], config)
        atomic_json(folder / "video_receipt.json", video)
        evaluation = evaluate(config, folder, request, run)
        result = dict(task_id=task_id, attempt_token=token, config_sha256=task["config_sha"],
                      acceptance="VIDEO_E2E_PASS", official_benchmark_pass=False,
                      video=video, trace=trace, exit_code=0, evaluation=evaluation)
        atomic_json(folder / "receipt.json", result)
        with queue.transaction():
            queue.db.execute("UPDATE tasks SET result=? WHERE id=?", (canonical(result), task_id))
            queue.transition(task_id, "COMPLETE", evaluation["state"], token=token)
        if config.get("mode") == "evaluation_only":
            from .scoring import publish_scoring
            try:
                publish_scoring(queue, config, result)
            except Exception as exc:
                queue.audit(task_id, "SCORE_PUBLICATION_PENDING", {"error": str(exc), "token": token})
    except Exception as exc:
        if isinstance(exc, ResourceBlocked):
            state, category = "BLOCKED_GPU", "GPU_RESOURCE_ERROR"
        else:
            logs = "\n".join(p.read_text(errors="replace")[-24000:] for p in folder.glob("*.stderr.log"))
            state, category = classify(logs + "\n" + str(exc), stalled=isinstance(exc, TimeoutError))
        result = dict(task_id=task_id, attempt_token=token, state=state, category=category, error=str(exc),
                      exit_code=last_exit, phase=phase, acceptance=None, official_benchmark_pass=False)
        atomic_json(folder / "failure.json", result)
        # Stop only this token's descendants before releasing any owned lease.
        try:
            stop_owned(token)
        except Exception as cleanup:
            result["cleanup_error"] = str(cleanup)
            atomic_json(folder / "failure.json", result)
        with queue.transaction():
            current = queue.task(task_id)["state"]
            if current == "READY" and state in FAILURES:
                queue.transition(task_id, "RUNNING", "worker failure before native launch", token=token)
            queue.transition(task_id, state, f"{category}: {exc}", token=token)
            queue.audit(task_id, "FAILURE", result)
    finally:
        set_attempt(queue, token, ended=time.time(), exit_code=last_exit, diagnostic=canonical(result) if result else None)
        if not owned_processes(token):
            if held:
                try:
                    resource.release(token)
                except Exception as exc:
                    queue.audit(task_id, "LEASE_RELEASE_FAILED", {"token": token, "error": str(exc)})
            # Keep attempt association on terminal/failure states for provenance;
            # blocked tasks can be re-preflighted only after safe cleanup.
            if queue.task(task_id)["state"] in BLOCKED and not list(queue.db.execute("SELECT 1 FROM leases WHERE token=?", (token,))):
                queue.db.execute("UPDATE tasks SET attempt=NULL WHERE id=?", (task_id,))


def resume(queue, policy):
    actions = []
    with dispatch_lock(queue):
        for task in queue.tasks():
            token = task["attempt"]
            if not token:
                continue
            attempt = queue.db.execute("SELECT * FROM attempts WHERE token=?", (token,)).fetchone()
            worker_id = json.loads(attempt["worker"]) if attempt["worker"] else None
            if matches(worker_id):
                actions.append(dict(task_id=task["id"], action="LIVE_WORKER_UNTOUCHED"))
                continue
            # The spawn/claim interval is durable but has no child yet. Avoid a
            # second dispatch during that interval, including scheduler restart.
            if not worker_id and time.time() - attempt["started"] < 60:
                actions.append(dict(task_id=task["id"], action="WAIT_WORKER_BOOTSTRAP"))
                continue
            resource = Resources(queue, policy)
            orphan_failure = None
            # If there is no worker and no existing local lease, adoption is
            # needed only when a real broker policy was supplied.
            if policy.get("legacy_leases"):
                resource.recover_owned(task["id"], token)
            if owned_processes(token):
                config = json.loads(task["config"])
                rows = trace_rows(folder_for(queue, task["id"], token) / "native_trace.jsonl",
                                  task["id"], task["config_sha"], token, partial=True)
                progress = [r for r in rows if r.get("event") in ("kernel", "chunk", "forward")
                            and type(r.get("progress_seq")) is int]
                old = next((r for r in queue.db.execute("SELECT timestamp,payload FROM events WHERE task_id=? AND kind='PROGRESS' ORDER BY seq DESC", (task["id"],))
                            if json.loads(r["payload"]).get("token") == token), None)
                maximum = max(progress, key=lambda r: r["progress_seq"]) if progress else None
                previous = json.loads(old["payload"]).get("last_actual_kernel_or_chunk") if old else None
                observed = old["timestamp"] if old else attempt["started"]
                if task["state"] == "EVALUATING":
                    start = next((r for r in queue.db.execute("SELECT timestamp,payload FROM events WHERE task_id=? AND kind='PROCESS_STARTED' ORDER BY seq DESC", (task["id"],))
                                  if json.loads(r["payload"]).get("token") == token and json.loads(r["payload"]).get("stage") == "EVALUATING"), None)
                    observed = max(observed, start["timestamp"] if start else observed)
                if maximum and (not previous or maximum["progress_seq"] > previous.get("progress_seq", -1)):
                    queue.audit(task["id"], "PROGRESS", {"token": token, "last_actual_kernel_or_chunk": maximum})
                    observed = time.time()
                timeout = config["timeouts"]["evaluation_max_s" if task["state"] == "EVALUATING" else "generation_no_progress_s"]
                resources = Resources(queue, policy)
                resources.renew(token)
                if time.time() - observed <= timeout:
                    actions.append(dict(task_id=task["id"], action="ORPHAN_PROGRESS_RETAINED", last_progress=maximum))
                    continue
                stderr = "\n".join(p.read_text(errors="replace")[-24000:] for p in folder_for(queue, task["id"], token).glob("*.stderr.log"))
                orphan_failure = classify(stderr, stalled=True)
                diagnostic = {"token": token, "last_actual_kernel_or_chunk": maximum, "category": orphan_failure[1], "stderr": stderr,
                              "resources": resources.snapshot(dict(config, task_id=task["id"])),
                              "reason": "owned orphan exceeded explicit no-progress timeout"}
                queue.audit(task["id"], "ORPHAN_STALLED", diagnostic)
                atomic_json(folder_for(queue, task["id"], token) / "orphan_diagnostic.json", diagnostic)
                stop_owned(token)
            if list(queue.db.execute("SELECT 1 FROM leases WHERE token=?", (token,))):
                try:
                    resource.release(token)
                except Exception as exc:
                    queue.audit(task["id"], "RECOVERY_LEASE_BLOCKED", {"error": str(exc)})
                    continue
            if task["state"] == "COMPLETE":
                config = json.loads(task["config"])
                if config.get("mode") == "evaluation_only" and verified_success(task):
                    from .scoring import publish_scoring
                    try:
                        publish_scoring(queue, config, json.loads(task["result"]))
                    except Exception as exc:
                        queue.audit(task["id"], "SCORE_PUBLICATION_PENDING", {"error": str(exc), "token": token})
                actions.append(dict(task_id=task["id"], action="REUSE_VERIFIED" if verified_success(task) else "HASH_DRIFT_BLOCK_REUSE"))
                continue
            folder = folder_for(queue, task["id"], token)
            # Crash between receipt fsync and DB commit: recover the receipt
            # only after independently checking exit code, attempt and hashes.
            receipt = folder / "receipt.json"
            recovered = False
            if receipt.is_file() and task["state"] in ACTIVE:
                candidate = read_json(receipt)
                restored = dict(task, state="COMPLETE", result=canonical(candidate))
                if candidate.get("attempt_token") == token and candidate.get("task_id") == task["id"] and verified_success(restored):
                    with queue.transaction():
                        queue.db.execute("UPDATE tasks SET result=? WHERE id=?", (canonical(candidate), task["id"]))
                        queue.transition(task["id"], "COMPLETE", "RECOVERED_VERIFIED_RECEIPT", token=token)
                    recovered = True
            if not recovered and task["state"] in ACTIVE | {"READY"}:
                with queue.transaction():
                    if task["state"] == "READY":
                        queue.transition(task["id"], "RUNNING", "bootstrap interrupted", token=token)
                    queue.transition(task["id"], orphan_failure[0] if orphan_failure else "STALLED",
                                     orphan_failure[1] if orphan_failure else "worker exited without verified receipt; no owned process alive", token=token)
                actions.append(dict(task_id=task["id"], action="STALLED_PRESERVE_LOGS_REQUIRE_REPAIR"))
            elif task["state"] in BLOCKED:
                queue.db.execute("UPDATE tasks SET attempt=NULL WHERE id=?", (task["id"],))
    return actions
