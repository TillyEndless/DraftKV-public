"""SQLite is the dispatch authority; configs and attempts are append-only."""

from contextlib import contextmanager
from enum import Enum
import json
from pathlib import Path
import sqlite3
import time
import uuid

from .common import canonical, object_hash


class State(str, Enum):
    PENDING = "PENDING"
    PREFLIGHT = "PREFLIGHT"
    READY = "READY"
    RUNNING = "RUNNING"
    GENERATING = "GENERATING"
    EVALUATING = "EVALUATING"
    COMPLETE = "COMPLETE"
    BLOCKED_INPUT = "BLOCKED_INPUT"
    BLOCKED_ENVIRONMENT = "BLOCKED_ENVIRONMENT"
    BLOCKED_SEMANTICS = "BLOCKED_SEMANTICS"
    BLOCKED_GPU = "BLOCKED_GPU"
    FAILED_OOM = "FAILED_OOM"
    FAILED_RUNTIME = "FAILED_RUNTIME"
    STALLED = "STALLED"


ACTIVE = {"RUNNING", "GENERATING", "EVALUATING"}
FAILURES = {"FAILED_OOM", "FAILED_RUNTIME", "STALLED"}
BLOCKED = {s.value for s in State if s.value.startswith("BLOCKED_")}
EDGES = {
    "PENDING": {"PREFLIGHT"},
    "PREFLIGHT": {"READY"} | BLOCKED,
    "READY": {"RUNNING", "PREFLIGHT"} | BLOCKED,
    "RUNNING": {"GENERATING"} | BLOCKED | FAILURES,
    "GENERATING": {"EVALUATING", "COMPLETE"} | BLOCKED | FAILURES,
    "EVALUATING": {"COMPLETE"} | FAILURES,
    "COMPLETE": set(),
    **{s: {"PREFLIGHT"} for s in BLOCKED},
    **{s: {"PENDING"} for s in FAILURES},
}


class Queue:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.root / "queue.sqlite3", timeout=30, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS tasks (
          id TEXT PRIMARY KEY, benchmark TEXT NOT NULL, backbone TEXT NOT NULL,
          algorithm TEXT NOT NULL, case_id TEXT NOT NULL, manifest_sha TEXT NOT NULL,
          config TEXT NOT NULL, config_sha TEXT NOT NULL, state TEXT NOT NULL,
          reason TEXT NOT NULL DEFAULT '', position REAL NOT NULL, updated REAL NOT NULL,
          attempt TEXT, result TEXT,
          UNIQUE(benchmark, backbone, algorithm, case_id));
        CREATE TABLE IF NOT EXISTS attempts (
          token TEXT PRIMARY KEY, task_id TEXT NOT NULL REFERENCES tasks(id),
          number INTEGER NOT NULL, started REAL NOT NULL, worker TEXT, child TEXT,
          ended REAL, exit_code INTEGER, diagnostic TEXT, UNIQUE(task_id, number));
        CREATE TABLE IF NOT EXISTS leases (
          gpu TEXT PRIMARY KEY, task_id TEXT NOT NULL REFERENCES tasks(id),
          token TEXT NOT NULL, external_id TEXT, owner TEXT NOT NULL, created REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS events (
          seq INTEGER PRIMARY KEY, task_id TEXT, timestamp REAL NOT NULL,
          kind TEXT NOT NULL, payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS intents (
          group_id TEXT PRIMARY KEY, config TEXT NOT NULL, config_sha TEXT NOT NULL,
          state TEXT NOT NULL, reason TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS configs (
          task_id TEXT NOT NULL REFERENCES tasks(id), sha256 TEXT NOT NULL,
          config TEXT NOT NULL, created REAL NOT NULL, reference TEXT NOT NULL,
          PRIMARY KEY(task_id,sha256));
        CREATE TRIGGER IF NOT EXISTS frozen_config BEFORE UPDATE ON configs
          BEGIN SELECT RAISE(ABORT, 'frozen configuration revision'); END;
        CREATE TRIGGER IF NOT EXISTS retained_config BEFORE DELETE ON configs
          BEGIN SELECT RAISE(ABORT, 'retain original configuration'); END;
        CREATE TRIGGER IF NOT EXISTS frozen_task BEFORE UPDATE OF
          id,benchmark,backbone,algorithm,case_id,manifest_sha ON tasks
          BEGIN SELECT RAISE(ABORT, 'frozen task identity/config'); END;
        """)

    @contextmanager
    def transaction(self):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            yield
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    def audit(self, task_id, kind, payload):
        self.db.execute("INSERT INTO events(task_id,timestamp,kind,payload) VALUES(?,?,?,?)",
                        (task_id, time.time(), kind, canonical(payload)))

    def submit(self, config):
        from draftkv.registry import Registry
        group = Registry().select(config["benchmark"], config["backbone"], config["algorithm"])[0]
        if (config["benchmark"], config["backbone"], config["algorithm"]) != group.key:
            raise ValueError("use canonical release group names")
        case = config["case_id"]
        if not isinstance(case, str) or not case or "/" in case or ".." in case:
            raise ValueError("unsafe case ID")
        sha = object_hash(config)
        task_id = "h200-" + object_hash([*group.key, case, config["manifest_sha256"]])[:24]
        with self.transaction():
            if config["benchmark"] == "bench2":
                from .manifests import validate_manifest, validate_release_group
                if config.get("workflow") == "generate_then_official_orbia":
                    validate_release_group(config)
                else:
                    validate_manifest(config)
                for existing in self.db.execute("SELECT config FROM tasks WHERE benchmark='bench2'"):
                    previous = json.loads(existing[0])
                    if previous["selection_sha256"] != config["selection_sha256"] or previous["manifest_sha256"] != config["manifest_sha256"]:
                        raise ValueError("all Bench2 groups must use one frozen manifest/selection")
            old = self.db.execute("SELECT * FROM tasks WHERE benchmark=? AND backbone=? AND algorithm=? AND case_id=?",
                                  (*group.key, case)).fetchone()
            if old:
                if old["config_sha"] != sha:
                    raise ValueError(f"frozen task config collision: {old['id']}; use a separate release workspace")
                return old["id"]
            self.db.execute("INSERT INTO tasks(id,benchmark,backbone,algorithm,case_id,manifest_sha,config,config_sha,state,position,updated) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                            (task_id, *group.key, case, config["manifest_sha256"], canonical(config), sha,
                             "PENDING", time.time(), time.time()))
            self.db.execute("INSERT INTO configs VALUES(?,?,?,?,?)", (task_id, sha, canonical(config), time.time(), "ORIGINAL_SUBMISSION"))
            self.audit(task_id, "SUBMITTED", {"config_sha256": sha})
        return task_id

    def revise(self, task_id, config, previous_sha, reference):
        """Append an explicitly audited config revision; retain original bytes.

        Only blocked, inactive tasks may be revised. Successful/failed attempts
        cannot be rewritten; failed tasks first need the repair/retry workflow.
        """
        if not reference:
            raise ValueError("revision requires an evidence reference")
        with self.transaction():
            task = self.task(task_id)
            if task["state"] not in BLOCKED | {"PENDING"} or task["attempt"]:
                raise ValueError("revision requires inactive blocked/pending task")
            if task["config_sha"] != previous_sha:
                raise ValueError("previous config pin mismatch")
            if self.db.execute("SELECT 1 FROM leases WHERE task_id=?", (task_id,)).fetchone():
                raise ValueError("task still holds a lease")
            original = json.loads(task["config"])
            for key in ("benchmark", "backbone", "algorithm", "case_id", "manifest_sha256", "selection_sha256", "seed", "pilot"):
                if config.get(key) != original.get(key):
                    raise ValueError("config revision cannot change frozen task/selection/seed identity")
            if config["benchmark"] == "bench2":
                from .manifests import validate_manifest, validate_release_group
                if config.get("workflow") == "generate_then_official_orbia":
                    validate_release_group(config)
                else:
                    validate_manifest(config)
            sha = object_hash(config)
            self.db.execute("INSERT OR IGNORE INTO configs VALUES(?,?,?,?,?)", (task_id, sha, canonical(config), time.time(), reference))
            self.db.execute("UPDATE tasks SET config=?,config_sha=? WHERE id=?", (canonical(config), sha, task_id))
            self.audit(task_id, "CONFIG_REVISION", {"previous_sha256": previous_sha, "sha256": sha, "reference": reference})
        return sha

    def intent(self, config, reason):
        key = "/".join(config[k] for k in ("benchmark", "backbone", "algorithm"))
        with self.transaction():
            old = self.db.execute("SELECT config_sha FROM intents WHERE group_id=?", (key,)).fetchone()
            if old and old[0] != object_hash(config):
                raise ValueError("frozen group intent changed")
            self.db.execute("INSERT OR IGNORE INTO intents VALUES(?,?,?,?,?)",
                            (key, canonical(config), object_hash(config), "BLOCKED_INPUT", reason))

    def task(self, task_id):
        row = self.db.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        if row is None:
            raise KeyError(task_id)
        return dict(row)

    def tasks(self):
        return [dict(row) for row in self.db.execute("SELECT * FROM tasks ORDER BY position,id")]

    def transition(self, task_id, state, reason="", *, token=None):
        state = State(state).value
        old = self.task(task_id)
        if token is not None and old["attempt"] != token:
            raise ValueError("attempt ownership mismatch")
        if state != old["state"] and state not in EDGES[old["state"]]:
            raise ValueError(f"illegal transition {old['state']} -> {state}")
        self.db.execute("UPDATE tasks SET state=?,reason=?,updated=? WHERE id=?",
                        (state, reason, time.time(), task_id))
        self.audit(task_id, "STATE", {"from": old["state"], "to": state, "reason": reason})

    def claim(self, task_id):
        with self.transaction():
            task = self.task(task_id)
            if task["state"] != "READY" or task["attempt"]:
                raise ValueError("task already claimed or not ready")
            token = uuid.uuid4().hex
            number = self.db.execute("SELECT count(*)+1 FROM attempts WHERE task_id=?", (task_id,)).fetchone()[0]
            self.db.execute("INSERT INTO attempts(token,task_id,number,started) VALUES(?,?,?,?)",
                            (token, task_id, number, time.time()))
            self.db.execute("UPDATE tasks SET attempt=? WHERE id=?", (token, task_id))
            self.audit(task_id, "CLAIMED", {"token": token, "number": number})
        return token

    def retry(self, task_id, repair):
        """A hash-pinned repair receipt is mandatory; retry always goes to the tail."""
        from .common import digest, read_json
        receipt = read_json(repair["path"])
        if digest(repair["path"]) != repair["sha256"] or receipt.get("task_id") != task_id:
            raise ValueError("repair receipt pin/task mismatch")
        if not receipt.get("root_cause") or not receipt.get("fix") or not receipt.get("verification"):
            raise ValueError("repair receipt needs root_cause, fix, verification")
        with self.transaction():
            task = self.task(task_id)
            if task["state"] not in FAILURES:
                raise ValueError("retry requires a failed/stalled task")
            if self.db.execute("SELECT 1 FROM leases WHERE task_id=?", (task_id,)).fetchone():
                raise ValueError("owned processes/resources must be reconciled before retry")
            attempt = self.db.execute("SELECT * FROM attempts WHERE token=?", (task["attempt"],)).fetchone()
            from .common import matches
            if attempt and any(matches(json.loads(attempt[k])) for k in ("worker", "child") if attempt[k]):
                raise ValueError("attempt still alive")
            position = self.db.execute("SELECT max(position) FROM tasks").fetchone()[0] + 1
            self.transition(task_id, "PENDING", "REPAIR_VERIFIED")
            self.db.execute("UPDATE tasks SET attempt=NULL,position=? WHERE id=?", (position, task_id))
            self.audit(task_id, "RETRY_REPAIR", repair)
