"""Live PID/memory/queue checks and atomic shared-broker leases. Never preempt."""

import csv
import io
import json
import os
from pathlib import Path
import pwd
import time
import xml.etree.ElementTree as ET

from .common import command, process_identity, read_json
from .gates import pinned


class ResourceBlocked(RuntimeError):
    pass


PORTABLE_BENCHMARK_SCOPES = frozenset({'bench1', 'bench2'})


def legacy_h200_only(policy):
    """Only unspecialized legacy release policies retain the historical 2×H200 host contract."""
    return policy.get('scope') not in PORTABLE_BENCHMARK_SCOPES


def lease_owner(scope, token):
    if scope == 'bench1':
        return 'draftkv-bench1/' + token
    if scope == 'bench2':
        return 'draftkv-bench2/' + token
    return 'draftkv-h200-release/' + token


def choose_devices(snapshot, sp, required_mib, *, ignore_owner=None):
    safe = []
    for gpu in snapshot["gpus"]:
        leases = [x for x in snapshot["leases"] if str(x["gpu"]) == str(gpu["index"])
                  and x.get("owner") != ignore_owner]
        workloads = [p for p in gpu["processes"] if p.get("allowed_display") is not True]
        if not leases and not workloads:
            safe.append(gpu)
    safe.sort(key=lambda gpu: (-gpu["free_mib"], gpu["index"]))
    if len(safe) < sp:
        raise ResourceBlocked(f"SP{sp} needs {sp} idle unleased devices")
    return sorted(safe[:sp], key=lambda gpu: gpu["index"])


class Resources:
    def __init__(self, queue, policy):
        self.queue, self.policy = queue, policy

    def snapshot(self, config, *, ignore_owner=None):
        raw = command(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu",
                       "--format=csv,noheader,nounits"])
        gpus = []
        for line in csv.reader(io.StringIO(raw), skipinitialspace=True):
            index, uuid, name, total, used, free, util = [x.strip() for x in line]
            if self.policy.get("allowed_gpu_indices") is not None and int(index) not in self.policy["allowed_gpu_indices"]:
                continue
            if legacy_h200_only(self.policy) and "H200" not in name:
                raise ResourceBlocked("GPU host is not H200")
            gpus.append(dict(index=int(index), uuid=uuid, name=name, total_mib=int(total),
                             used_mib=int(used), free_mib=int(free), utilization=int(util), processes=[]))
        if legacy_h200_only(self.policy) and len(gpus) != 2:
            raise ResourceBlocked("expected two H200 devices")
        process_rows = command(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                                "--format=csv,noheader,nounits"])
        for fields in csv.reader(io.StringIO(process_rows), skipinitialspace=True):
            uuid, pid, name, used = [x.strip() for x in fields]
            ident = process_identity(int(pid))
            try:
                user = pwd.getpwuid(ident["uid"]).pw_name if ident else "UNKNOWN"
            except KeyError:
                user = "UNKNOWN"
            for gpu in gpus:
                if gpu["uuid"] == uuid:
                    gpu["processes"].append(dict(pid=int(pid), user=user, identity=ident, name=name, used_mib=used))
        # Include graphics contexts as well as compute processes. A low-memory
        # foreign context is still a context and cannot be treated as idle.
        xml = ET.fromstring(command(["nvidia-smi", "-q", "-x"]))
        for node in xml.findall("gpu"):
            gpu = next((g for g in gpus if g["uuid"] == node.findtext("uuid")), None)
            if gpu is None:
                continue
            if node.find("processes") is None:
                raise ResourceBlocked("incomplete GPU process inventory")
            known = {p["pid"] for p in gpu["processes"]}
            for process in node.findall("processes/process_info"):
                pid = int(process.findtext("pid"))
                if pid in known:
                    continue
                identity = process_identity(pid)
                try:
                    user = pwd.getpwuid(identity["uid"]).pw_name if identity else "UNKNOWN"
                except KeyError:
                    user = "UNKNOWN"
                gpu["processes"].append(dict(pid=pid, identity=identity, user=user,
                                             name=process.findtext("process_name"), type=process.findtext("type")))
        for gpu in gpus:
            for process in gpu["processes"]:
                for allowed in self.policy.get("allowed_display_contexts", []):
                    if (process.get("type") == "G" and process.get("name") == allowed["name"]
                            and process.get("identity") == allowed["identity"]):
                        process["allowed_display"] = True
                        process["display_baseline_reference"] = allowed["verification_reference"]
        # Read the legacy lease file directly. Some legacy `status` commands prune
        # leases as a side effect, which is inappropriate during read-only audit.
        legacy = read_json(self.policy["legacy_leases"])
        leases = [dict(gpu=int(k), **v) for k, v in legacy.items()]
        leases.extend(dict(row) for row in self.queue.db.execute("SELECT * FROM leases"))
        states = []
        aliases = {config.get("task_id"), *config.get("legacy_task_ids", [])} - {None}
        for path in self.policy["legacy_queues"]:
            payload = read_json(path)
            states.append({"path": path, "running": payload.get("running"), "phase": payload.get("phase")})
            if payload.get("running") is not None and not any(j.get("id") == payload["running"] for j in payload.get("jobs", [])):
                raise ResourceBlocked("unknown legacy running-record format")
            for job in payload.get("jobs", []):
                active = str(job.get("status", job.get("state", ""))).upper() in {
                    "RUNNING", "GENERATING", "EVALUATING", "READY", "QUEUED", "PENDING"}
                if active and job.get("id", job.get("task_id")) in aliases:
                    raise ResourceBlocked("same Task ID already active in legacy queue")
                running = job.get("id") == payload.get("running") or str(job.get("status", "")).upper() in {"RUNNING", "GENERATING", "EVALUATING"}
                if running:
                    devices = job.get("gpus", job.get("gpu_candidate"))
                    if devices is None:
                        raise ResourceBlocked("legacy running task has unknown GPU allocation")
                    for gpu in devices:
                        leases.append(dict(gpu=gpu, owner="legacy-running", task_id=job.get("id")))
            if payload.get("running") and not payload.get("jobs"):
                raise ResourceBlocked("uninterpretable active legacy queue")
        snapshot = dict(time=time.time(), gpus=gpus, leases=leases, legacy_queues=states,
                        process_tree=command(["ps", "-eo", "user:22,pid,ppid,pgid,lstart,etimes,args", "--forest"]))
        # Always retain actual process ownership and the full tree locally.
        self.queue.audit(config.get("task_id"), "RESOURCE_SNAPSHOT", snapshot)
        return snapshot

    def broker(self, action, **values):
        path = pinned(self.policy["broker"])
        argv = [self.policy["broker_python"], str(path), action]
        if self.policy.get("broker_state"):
            argv.extend(["--state", str(self.policy["broker_state"])])
        for key, value in values.items():
            argv.extend(["--" + key.replace("_", "-"), str(value)])
        result = json.loads(command(argv))
        if result.get("ok") is not True:
            raise ResourceBlocked(f"shared broker {action}: {result}")
        return result

    def acquire(self, config, task_id, token):
        owner = lease_owner(self.policy.get("scope"), token)
        required = 0  # capacity/OOM never blocks dispatch
        with self.queue.transaction():
            selected = choose_devices(self.snapshot(dict(config, task_id=task_id)), config["sp"], required)
            # Atomic multi-GPU acquisition by the existing broker; SP2 never
            # holds one device while waiting for another.
            result = self.broker("lease", gpus=",".join(str(g["index"]) for g in selected), owner=owner, ttl=3600)
            try:
                for gpu in selected:
                    self.queue.db.execute("INSERT INTO leases VALUES(?,?,?,?,?,?)",
                                          (str(gpu["index"]), task_id, token, result["lease_id"], owner, time.time()))
                current = self.snapshot(dict(config, task_id=task_id))
                safe = choose_devices(current, config["sp"], required, ignore_owner=owner)
                if {g["index"] for g in safe} != {g["index"] for g in selected}:
                    raise ResourceBlocked("GPU occupancy changed after lease")
                self.queue.audit(task_id, "LEASE_ACQUIRED", {"owner": owner, "gpus": selected, **result})
            except BaseException:
                self.broker("release", owner=owner, lease_id=result["lease_id"])
                raise
        return selected

    def renew(self, token):
        rows = list(self.queue.db.execute("SELECT DISTINCT owner,external_id FROM leases WHERE token=?", (token,)))
        if not rows:
            raise ResourceBlocked("owned lease missing")
        for row in rows:
            self.broker("renew", owner=row["owner"], lease_id=row["external_id"], ttl=3600)

    def recover_owned(self, task_id, token):
        """Recover a crash between external broker acquisition and SQLite commit."""
        owner = lease_owner(self.policy.get("scope"), token)
        legacy = read_json(self.policy["legacy_leases"])
        with self.queue.transaction():
            for gpu, lease in legacy.items():
                if lease.get("owner") != owner:
                    continue
                old = self.queue.db.execute("SELECT token FROM leases WHERE gpu=?", (gpu,)).fetchone()
                if old and old[0] != token:
                    raise ResourceBlocked("recovery found conflicting local device lease")
                self.queue.db.execute("INSERT OR IGNORE INTO leases VALUES(?,?,?,?,?,?)",
                                      (gpu, task_id, token, lease["lease_id"], owner, time.time()))
                self.queue.audit(task_id, "EXTERNAL_LEASE_RECOVERED", {"gpu": gpu, "token": token})

    def release(self, token):
        # Caller must first prove that every process of the owned attempt exited.
        rows = list(self.queue.db.execute("SELECT DISTINCT owner,external_id FROM leases WHERE token=?", (token,)))
        legacy = read_json(self.policy["legacy_leases"]) if rows else {}
        for row in rows:
            if any(v.get("owner") == row["owner"] and v.get("lease_id") == row["external_id"] for v in legacy.values()):
                self.broker("release", owner=row["owner"], lease_id=row["external_id"])
            else:
                self.queue.audit(None, "OWNED_EXTERNAL_LEASE_ALREADY_ABSENT", dict(row))
        with self.queue.transaction():
            self.queue.db.execute("DELETE FROM leases WHERE token=?", (token,))
            self.queue.audit(None, "LEASE_RELEASED", {"token": token})
