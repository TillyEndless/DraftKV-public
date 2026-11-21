"""Thin startup-only mapping to original, externally prepared raw-plan commands."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ROWS = {
    ("longlive1.3b", 1): ("longlive", "LongLive1.3B"),
    ("lingbot1.3b", 1): ("lingbot13", "LingBot1.3B"),
    ("lingbot14b", 1): ("h200_large", "LingBot14B"),
    ("solarwm33b", 1): ("h200_large", "SolarWM33B"),
    ("lingbot14b", 8): ("sp8_lingbot14_d546", "LingBot14B"),
    ("solarwm33b", 8): ("sp8_solar_de790", "SolarWM33B"),
}

# Recovered original campaign bytes, not the later profiling staging framework.
# Controllers remain external: their host-specific paths are not redistributed.
PROFILE_CONTROLLERS = {
    "longlive1.3b": {
        "name": "controller_user_repair2_draft_admission_retry.py",
        "sha256": "f6a26bc19ab26e57a112ea62f0cb0a5df49e79391336f1a64cdcf800d8075dd6",
    },
    "lingbot1.3b": {
        "name": "controller.py",
        "sha256": "16fd7e0f51f926178d960b64fb448c52591f5977e2cddde1b677081e75df69e5",
    },
    "lingbot14b": {
        "name": "run_campaign.py",
        "sha256": "112d5a5c2920009c81fc4ec6d7f1fe051df98bdb74e24c14f94fa51072de3477",
        "entry": "run_one(row)",
    },
    "solarwm33b": {
        "name": "controller.py",
        "sha256": "fe94e35d08106aed71f63f6053b9ee76f56d80d693ffd64dae5c105774ca9ce6",
        "entry": "run_one(row, tools)",
    },
}
LINGBOT13_TOOLS = {
    "controller.py": PROFILE_CONTROLLERS["lingbot1.3b"]["sha256"],
    "worker.py": "33ea5645ce3dfc408b7da8fce6b12bb5304509858dd1e4ee33da69068f693684",
    "execute.py": "5e7ed9902f9706c0422e5dec92cad84cbc1249331976ada2796ded5ed890047e",
    "prepare.py": "cf70b054681c64f55e16715c7b32bb84f4e11b2f27f1efb14abb8c230c98e1ba",
}
LINGBOT13_ENV_SHA256 = "52493dc7023a107765aa73ff2ceda934bbc8d91286a61e38dd220e404b676a38"
REFERENCE_RESULT_SHA256 = {
    "lingbot14b": "719a67e0b27dd105024fc48d02b6d8271d3d89c0c36db1b33f768e06acf3f905",
    "solarwm33b": "9cccb355fa3013d796ed4f40fe683ec2c247f51bd8c4142fb3b1e03094a81888",
}
MEASUREMENT_ADMISSION_SHA256 = "ffc75d45d404be8dcaa3bf893060fc5a17015e9ca3da57ae688e2a23ce611c19"
L14_TEST_OBSERVER_SHA256 = "49687494648724cff9716b5dc4df866142f6b38f4508c30b1c9971596c911ca9"

# Thin single-row selector, following the existing private launch_one.py API.
# Only startup B/S/COMMIT/REF bindings change. No function is replaced and no
# scientific plan, profiler, CUDA callback, cache setting or timer is edited.
SCOPED_CONTROLLER_ENTRY = """import importlib.util, json, sys
from pathlib import Path
binding = json.loads(sys.argv[2])
controller = Path(sys.argv[1])
import hashlib
if hashlib.sha256(controller.read_bytes()).hexdigest() != binding['controller_sha256']:
    raise RuntimeError('controller changed after CLI validation')
spec = importlib.util.spec_from_file_location('original_controller', controller)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.B = Path(binding['base'])
module.S = Path(binding['source'])
module.COMMIT = binding['commit']
sys.path.insert(0, str(module.B))
if binding['backbone'] == 'lingbot14b':
    module.REF = Path(binding['reference'])
    module.run_one(binding['row'])
else:
    module.run_one(binding['row'], binding['tools'])
"""

def select_controller(a):
    require(a.sp == 1, "summary: no original Table 1/2 SP8 controller admitted")
    original = PROFILE_CONTROLLERS[a.backbone]
    require(a.controller is not None and a.controller.is_file() and outside(a.controller),
            "summary requires separately installed original --controller: " + original["name"])
    require(sha(a.controller) == original["sha256"], "summary controller SHA is not the recovered original for this backbone")
    return a.controller.resolve()

def literal_assignment(tree, name):
    values = [node.value.value for node in ast.walk(tree)
              if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
              and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)]
    require(len(values) == 1, "ambiguous original controller literal: " + name)
    return values[0]

def large_summary_command(a, controller, root, plan, record, tree):
    """Select existing metadata for the unchanged original single-row API."""
    large14 = a.backbone == "lingbot14b"
    base = a.worker.resolve().parent.parent
    expected_controller = base / "tools/run_campaign.py" if large14 else base / "controller.py"
    require(controller == expected_controller, "summary controller/worker campaign layout mismatch")
    require((base / "source").resolve() == root, "summary controller B/source must bind this release tree")
    index = 0 if large14 else literal_assignment(tree, "GPU")
    require(a.controller_device_index in (None, index) and plan.get("selected_devices") == [index],
            "summary retains the original controller physical GPU index")
    python = literal_assignment(tree, "PY")
    require(a.python == python and Path(python).is_file(), "--python must name the original controller's provisioned interpreter")
    row_file = base / "selected_row.json"
    row = json.loads(row_file.read_text())
    inv = json.loads((base / "inventory.json").read_text())
    run = Path(plan["run_dir"]).resolve()
    require(run == base / "runs" / row["name"], "summary selected row/run_dir mismatch")
    require(inv.get("rows") == [row] and Path(inv["source"]).resolve() == root and inv.get("gpu") == index,
            "summary requires one prepared inventory row bound to this source/GPU")
    require(inv.get("commit", record["candidate_commit"]) == record["candidate_commit"], "summary inventory commit mismatch")
    require(row.get("schedule") == "streaming" and plan.get("measurement_mode") == "formal",
            "summary requires the original formal streaming row")
    require(a.controller_reference is not None and outside(a.controller_reference), "summary requires --controller-reference with original reference result")
    reference = a.controller_reference.resolve()
    result = reference / "result.json"
    require(result.is_file() and sha(result) == REFERENCE_RESULT_SHA256[a.backbone], "original reference result missing/changed")
    if large14:
        require(row.get("mode") == "formal" and row.get("plan") == str(a.plan.resolve()), "summary original 14B row/plan mismatch")
    else:
        require(row.get("backbone") == "SolarWM33B" and row.get("reused") is False
                and row.get("physical_gpu") == index and row.get("run_dir") == str(run)
                and Path(row["reference_dir"]).resolve() == reference, "summary original Solar row/plan/reference mismatch")
        helper = base / "measurement_admission.py"
        require(helper.is_file() and sha(helper) == MEASUREMENT_ADMISSION_SHA256, "original measurement_admission.py missing/changed")
        require(not any((p / "measurement_admission.py").exists() or (p / "measurement_admission").exists()
                        for p in (root / "src", root)), "original measurement admission would be shadowed")
    identity = json.loads((ROOT / "ORIGINAL_COMMAND_MAPPING.json").read_text())[a.backbone + ":sp1"]
    expected = {**identity["external_companion_sha256"], "worker.py": sha(a.worker)}
    if large14:
        expected.update({"run_campaign.py": sha(controller), "test_observers.py": L14_TEST_OBSERVER_SHA256})
    actual = {f.name: sha(f) for f in (base / "tools").glob("*.py")}
    require(inv.get("tools") == actual == expected and plan.get("campaign_tools") == actual,
            "original campaign tool inventory missing/changed")
    require((base / "weights-identity.json").is_file(), "original worker weight identity receipt missing")
    for folder in (base / f"tmp_gpu{index}", root.parent / "build"):
        require(folder.is_dir(), "original controller dependency missing: " + str(folder))
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg in ("TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR"):
            require(isinstance(node.value, ast.Constant) and Path(node.value.value).is_dir(), "original controller cache path missing; CLI never remaps caches")
    binding = dict(backbone=a.backbone, controller_sha256=sha(controller), base=str(base), source=str(root),
                   commit=record["candidate_commit"], reference=str(reference), row=row, tools=actual)
    record.update(profile="summary", controller_sha256=sha(controller), controller_entry=PROFILE_CONTROLLERS[a.backbone]["entry"],
                  controller_gpu_index=index, selected_row_sha256=sha(row_file), reference_result_sha256=sha(result),
                  startup_bindings={k: binding[k] for k in ("base", "source", "commit", "reference")},
                  argv=[python, "-B", "-u", "-c", SCOPED_CONTROLLER_ENTRY, str(controller), json.dumps(binding, sort_keys=True)],
                  scope="existing single-row selector API; original run_one, guards, timers, admission and output checks",
                  gpu_test=False, device_binding_checked=False, cli_status="CLI_STATIC_VERIFIED")
    return record

def summary_command(a, controller, root, plan, record):
    """Read-only startup checks; preserve each original controller's measurement API."""
    require(plan.get("gpu_profile") is True, "summary requires an already-profile-enabled original plan; plans are never rewritten")
    run = Path(plan["run_dir"]).resolve()
    require(a.plan.resolve() == run / "plan.json", "summary requires original run_dir/plan.json layout")
    require(not any((run / x).exists() for x in (
        "command.json", "result.json", "failure.json", "stdout.log", "stderr.log", "traces",
        "gpu_profile.json", "memory_profile.json")), "summary refuses an existing attempt; use separately prepared fresh metadata")
    for name, digest in plan.get("source_files", {}).items():
        source = Path(plan["source_root"]) / name
        require(source.is_file() and sha(source) == digest, "summary original source dependency missing/changed: " + name)
    require(bool(plan.get("source_files")), "summary requires the original source dependency inventory")
    base = controller.parent
    tree = ast.parse(controller.read_bytes())  # Never import a top-level generation controller.
    if a.backbone in ("lingbot14b", "solarwm33b"):
        return large_summary_command(a, controller, root, plan, record, tree)
    require(a.controller_reference is None, "--controller-reference applies only to 14B/Solar original controllers")
    require((base / "source").resolve() == root, "summary controller B/source must bind this release tree")
    if a.backbone == "longlive1.3b":
        require(a.controller_device_index is None, "LongLive controller fixes GPU index 1; it cannot be remapped")
        index = 1
        require(a.devices == literal_assignment(tree, "UUID"), "original LongLive controller fixes a different physical GPU UUID")
        queue = base / "user_requested_repair2/draft_admission_retry_queue.json"
        rows = json.loads(queue.read_text())
        require(len(rows) == 1 and rows[0].get("algorithm") == "draftkv"
                and rows[0].get("plan") == str(a.plan.resolve()), "summary requires one prepared Full DraftKV row bound to --plan")
        require(not (queue.parent / "draft_admission_retry_run_status.json").exists(), "summary refuses an existing LongLive campaign")
        python = literal_assignment(tree, "PY")
        for folder in (base / "build", base / "tmp"):
            require(folder.is_dir(), "original controller dependency missing: " + str(folder))
        override = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant)
                    and isinstance(n.value, str) and n.value.startswith("/") and n.value.endswith("/fa3_torch28")]
        require(len(override) == 1 and Path(override[0]).is_dir(), "original LongLive FA3 environment missing")
    else:
        index = a.controller_device_index
        require(index is not None and index >= 0, "LingBot1.3B summary requires --controller-device-index for original main(g)")
        require(a.worker.resolve() == base / "worker.py", "summary must use the original controller's own worker")
        inv = json.loads((base / "inventory.json").read_text())
        require(inv["commit"] == record["candidate_commit"] and Path(inv["source"]).resolve() == root,
                "summary inventory must bind this release commit/source")
        require(inv["tools"] == LINGBOT13_TOOLS == {f.name: sha(f) for f in base.glob("*.py")}, "original campaign tools missing/changed")
        rows = inv["rows"]
        require(len(rows) == 1, "summary requires a single-row original inventory; no extra experiments")
        row = rows[0]
        require(row.get("method") == "draftkv" and row.get("model") == "LingBot1.3B"
                and row.get("gpu") == index and Path(row["run_dir"]).resolve() == run
                and row.get("plan_sha256") == record["plan_sha256"], "summary inventory row/plan identity mismatch")
        require(not (base / f"queue_gpu{index}.json").exists() and not (base / "STOP_AFTER_CURRENT").exists(),
                "summary refuses an existing or stopped LingBot campaign")
        tools = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant)
                 and isinstance(n.value, str) and n.value.startswith("/") and n.value.endswith("/tools")]
        require(len(tools) == 1, "ambiguous original environment module lookup")
        environment = Path(tools[0]) / "environment.py"
        require(environment.is_file() and sha(environment) == LINGBOT13_ENV_SHA256, "original controller environment.py missing/changed")
        require(not any((p / "environment.py").exists() or (p / "environment").exists()
                        for p in (root / "src", root, base)), "original environment module would be shadowed")
        python = literal_assignment(ast.parse(environment.read_bytes()), "PY")
        builds = [n.value.value for n in ast.walk(tree) if isinstance(n, ast.keyword)
                  and n.arg == "DRAFTKV_PREPACK_BUILD" and isinstance(n.value, ast.Constant)]
        require(len(builds) == 1 and Path(builds[0]).is_dir(), "original controller native build dependency missing")
    require(a.python == python and Path(python).is_file(), "--python must name the original controller's provisioned interpreter")
    record.update(profile="summary", controller_sha256=sha(controller), controller_gpu_index=index,
                  argv=[python, "-B", "-u", str(controller)] + ([str(index)] if a.backbone == "lingbot1.3b" else []),
                  scope="unchanged original parent campaign; original result/admission artifacts, no new summary schema",
                  gpu_test=False, device_binding_checked=False, cli_status="CLI_STATIC_VERIFIED")
    return record

def require(condition, message):
    if not condition:
        raise ValueError(message)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def verify(version=None):
    records = json.loads((ROOT / "RETAINED_SOURCE_SHA256.json").read_text())
    selected = [x for x in records if version is None or x["version"] == version]
    require(bool(selected), "empty retained-source inventory")
    for x in selected:
        p = ROOT / "versions" / x["version"] / x["path"]
        require(p.is_file() and not p.is_symlink(), "missing/nonregular retained source: " + str(p))
        require(sha(p) == x["sha256"], "source/config bytes changed: " + str(p))
    return len(selected)

def outside(path):
    return not Path(path).resolve().is_relative_to(ROOT)

def scientific_identity(plan, keys):
    config = json.loads(json.dumps(plan["algorithm_plugin"]["config"]))
    for key in ("cache_directory", "build_spec_path"):
        config.get("b01", {}).pop(key, None)
    value = {"original_algorithm_plugin_config_except_external_b01_paths": config,
             **{key: plan[key] for key in keys if key in plan}}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", nargs="?", choices=("run", "plan", "list", "check"), default="run")
    p.add_argument("--backbone", choices=sorted({x[0] for x in ROWS}))
    p.add_argument("--variant", choices=("draftkv",), default="draftkv")
    p.add_argument("--sp", type=int, choices=(1, 8), default=1)
    p.add_argument("--devices", help="exclusive physical GPU UUIDs, comma separated")
    p.add_argument("--plan", type=Path, help="external original-schema execution metadata; never rewritten")
    p.add_argument("--python", default=sys.executable, help="original provisioned Python interpreter")
    p.add_argument("--worker", type=Path, help="original external campaign worker, where required")
    p.add_argument("--original-worker", type=Path, help="original verified worker; required to audit a startup-only source-path mapping")
    p.add_argument("--worker-sha256", help="SHA from the verified original command evidence")
    p.add_argument("--profile", choices=("off", "summary"), default="off",
                   help="off: unchanged worker command; summary: original external profiling controller")
    p.add_argument("--controller", type=Path, help="separately installed original profiling controller, checked against known SHA")
    p.add_argument("--controller-device-index", type=int, help="original LingBot1.3B controller main(g) physical index; UUID checked before run")
    p.add_argument("--controller-reference", type=Path, help="original 14B/Solar reference result directory; known SHA checked")
    a = p.parse_args(argv)
    require(a.profile == "summary" or (a.controller is None and a.controller_device_index is None and a.controller_reference is None),
            "controller options require --profile summary")
    if a.action == "list":
        for (name, sp), (version, _) in ROWS.items():
            print(f"{name} draftkv SP={sp} versions/{version}")
        return 0
    if a.action == "check":
        print(json.dumps({"retained_files_verified": verify(), "gpu_test": False}))
        return 0
    require(a.backbone is not None, "--backbone is required")
    require((a.backbone, a.sp) in ROWS, "unsupported original backbone/topology")
    controller = select_controller(a) if a.profile == "summary" else None
    version, original_name = ROWS[a.backbone, a.sp]
    root = (ROOT / "versions" / version).resolve()
    count = verify(version)  # startup only; no layer/tile/timestep callback is installed
    require(a.plan is not None and a.plan.is_file(), "--plan must name prepared external execution metadata")
    require(outside(a.plan), "execution plans must remain outside the clean source repository")
    plan = json.loads(a.plan.read_text())
    require(plan.get("algorithm") == "draftkv", "only original Full DraftKV is supported")
    require(plan.get("backbone") == original_name and plan.get("sp") == a.sp, "plan backbone/SP mismatch")
    require(Path(plan.get("repo_root", "")).resolve() == root, "plan must bind this exact retained source tree")
    require(Path(plan.get("algorithm_plugin", {}).get("root", "")).resolve() == root, "algorithm plugin must bind the candidate, not an old source tree")
    require(plan.get("algorithm_plugin", {}).get("entry") == "src/draftkv/algorithms/draftkv/entry.py", "original DraftKV plugin entry required")
    retained = {x["path"]: x["sha256"] for x in json.loads((ROOT / "RETAINED_SOURCE_SHA256.json").read_text()) if x["version"] == version}
    external = {x["path"]: x["sha256"] for x in json.loads((ROOT / "EXTERNAL_DEPENDENCY_SHA256.json").read_text()) if x["version"] == version}
    allowed = {**external, **retained}
    for inventory in (plan.get("adapter_source_files", {}), plan.get("algorithm_plugin", {}).get("files", {})):
        require(bool(inventory), "nonempty current source inventory required")
        require(all(allowed.get(name) == digest for name, digest in inventory.items()), "execution source inventory differs from retained/original external identities")
        for name, digest in inventory.items():
            if name in external:
                dependency = root / name
                require(dependency.is_file() and sha(dependency) == digest, "external original dependency missing/changed: " + name)
    identity = json.loads((ROOT / "ORIGINAL_COMMAND_MAPPING.json").read_text())[a.backbone + ":sp" + str(a.sp)]
    require(scientific_identity(plan, identity["scientific_keys"]) == identity["scientific_sections_sha256"], "original algorithm/budget/kernel/profile semantics mismatch")
    actual_commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    require(plan.get("repo_commit") == actual_commit, "plan must truthfully record current candidate commit")
    require(not subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"], text=True), "source Git tree must be clean")
    for key in ("run_dir", "output"):
        require(key in plan and outside(plan[key]), key + " must stay outside the source repository")
    require(not Path(plan["output"]).exists(), "refusing to overwrite existing output")
    require(a.devices is not None, "--devices must list exclusively allocated physical GPU UUIDs")
    devices = a.devices.split(",")
    require(len(devices) == a.sp and len(set(devices)) == a.sp and all(x.startswith("GPU-") for x in devices), "one distinct physical UUID per rank is required")
    argv = [a.python, "-u", "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={a.sp}"]
    if a.backbone == "longlive1.3b":
        require(a.worker is None, "LongLive uses its original module worker")
        argv += ["-m", "draftkv.backbones.worker", "--plan", str(a.plan.resolve())]
    else:
        require(a.worker is not None and a.worker.is_file() and outside(a.worker), "original external worker and its companion tools must be installed separately")
        require(a.worker_sha256 is not None and sha(a.worker) == a.worker_sha256, "external worker SHA must match original verified evidence")
        known = json.loads((ROOT / "ORIGINAL_COMMAND_MAPPING.json").read_text())[a.backbone + ":sp" + str(a.sp)]["worker_sha256"]
        if a.backbone in ("lingbot14b", "solarwm33b") and a.sp == 1:
            require(a.original_worker is not None and sha(a.original_worker) == known, "verified original external worker is required for auditing the source-path-only mapping")
            original = a.original_worker.read_bytes()
            paths = [node.args[0].value for node in ast.walk(ast.parse(original))
                     if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "Path"
                     and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)
                     and node.args[0].value.startswith("/") and node.args[0].value.endswith("/source")]
            require(len(paths) == 1, "ambiguous original startup source binding")
            mapped = original.replace(paths[0].encode(), str(root).encode(), 1)
            require(a.worker.read_bytes() == mapped, "only the approved outer worker source-root literal mapping is allowed")
        else:
            require(a.worker_sha256 == known, "worker SHA is not the verified worker for this row")
        for name, digest in identity.get("external_companion_sha256", {}).items():
            companion = a.worker.resolve().parent / name
            require(companion.is_file() and sha(companion) == digest, "original external companion missing/changed: " + name)
        if a.sp == 8:
            require((a.worker.resolve().parent / "source").resolve() == root, "SP8 original worker's B/source must resolve to selected candidate source")
        argv += [str(a.worker.resolve()), str(a.plan.resolve())]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = a.devices
    env["PYTHONPATH"] = os.pathsep.join([str(root / "src"), str(root), env.get("PYTHONPATH", "")])
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if env.get("PYTHONPYCACHEPREFIX"):
        require(outside(env["PYTHONPYCACHEPREFIX"]), "pycache must remain external")
    record = {"candidate_commit": actual_commit, "version": version, "retained_files_verified": count,
              "plan_sha256": sha(a.plan), "cwd": str(root), "argv": argv,
              "CUDA_VISIBLE_DEVICES": a.devices, "scope": "original native worker; parent campaign NVML protocol is separate"}
    record["profile"] = a.profile
    if controller is not None:
        record = summary_command(a, controller, root, plan, record)
        argv = record["argv"]
        env = os.environ.copy()  # The original controller supplies its own worker environment.
    if a.action == "plan":
        print(json.dumps(record, indent=2))
        return 0
    if controller is not None:
        uuid = subprocess.check_output(["nvidia-smi", "-i", str(record["controller_gpu_index"]),
                                        "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
        require(uuid == a.devices, "original controller physical index does not match --devices UUID")
        record["device_binding_checked"] = True
    print(json.dumps(record), flush=True)
    os.chdir(root)
    os.execvpe(a.python, argv, env)
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as error:
        sys.exit("BLOCKED: " + str(error))
