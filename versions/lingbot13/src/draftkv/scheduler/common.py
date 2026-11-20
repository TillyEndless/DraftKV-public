"""Small, dependency-free provenance helpers."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def object_hash(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def command(argv, timeout=30, **kwargs):
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout,
                          check=True, **kwargs).stdout


def process_identity(pid):
    """Linux start ticks and boot ID defeat PID reuse; absence is not ownership."""
    try:
        base = Path(f"/proc/{int(pid)}")
        stat = (base / "stat").read_text().rsplit(")", 1)[1].split()
        if stat[0] == "Z":
            return None
        return {"pid": int(pid), "uid": base.stat().st_uid,
                "start_ticks": stat[19], "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                "pgid": int(stat[2])}
    except (OSError, ValueError, IndexError):
        return None


def matches(identity):
    return bool(identity and process_identity(identity["pid"]) == identity)


def event(path, **values):
    with Path(path).open("a") as stream:
        stream.write(canonical(dict(time=time.time(), **values)) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
