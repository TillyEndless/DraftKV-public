"""CPU dispatch tests with synthetic external metadata, never a GPU/controller run.

Only these fixtures replace controller hashes. Production admits only the archived
hashes in launch_original.py; fixture dispatch is not historical/GPU acceptance.
"""
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from unittest.mock import Mock
from types import SimpleNamespace

SPEC = importlib.util.spec_from_file_location("launch_original", Path(__file__).resolve().parents[1] / "tools/launch_original.py")
cli = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cli)


class ProfileCLI(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name).resolve()
        self.root = self.base / "release"
        self.root.mkdir()
        self.commit = "1" * 40
        self.patch = patch.object(cli, "ROOT", self.root)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.git = patch.object(cli.subprocess, "check_output", side_effect=self.command_output)
        self.git.start()
        self.addCleanup(self.git.stop)

    def command_output(self, args, **kwargs):
        if args[0] == "git":
            return self.commit + "\n" if "rev-parse" in args else ""
        if args[0] == "nvidia-smi":
            return "GPU-fixture\n"
        self.fail("unexpected subprocess: " + repr(args))

    def write_json(self, file, value):
        file.write_text(json.dumps(value))

    def setup_row(self, name):
        version, original = cli.ROWS[name, 1]
        root = self.root / "versions" / version
        root.mkdir(parents=True)
        (root / "original.py").write_text("# original fixture, never imported\n")
        digest = cli.sha(root / "original.py")
        self.write_json(self.root / "RETAINED_SOURCE_SHA256.json", [dict(version=version, path="original.py", sha256=digest)])
        self.write_json(self.root / "EXTERNAL_DEPENDENCY_SHA256.json", [])
        run = self.base / "run"
        run.mkdir()
        dependency = self.base / "original-dependency.py"
        dependency.write_text("# external fixture\n")
        plan = dict(algorithm="draftkv", backbone=original, sp=1, repo_root=str(root),
                    repo_commit=self.commit, run_dir=str(run), output=str(run / "native.mp4"),
                    source_root=str(self.base), source_files={dependency.name: cli.sha(dependency)},
                    gpu_profile=True, adapter_source_files={"original.py": digest},
                    algorithm_plugin=dict(root=str(root), entry="src/draftkv/algorithms/draftkv/entry.py",
                                          config={"original_budget": [3, 0, 11, 4]}, files={"original.py": digest}))
        metadata = dict(scientific_keys=["gpu_profile"], scientific_sections_sha256=cli.scientific_identity(plan, ["gpu_profile"]))
        args = ["--backbone", name, "--plan", str(run / "plan.json"), "--devices", "GPU-fixture", "--python", sys.executable]
        if name != "longlive1.3b":
            original_worker = self.base / "original-worker.py"
            original_worker.write_text("from pathlib import Path\nS=Path('/historical/source')\n")
            worker = self.base / "worker.py"
            worker.write_bytes(original_worker.read_bytes().replace(b"/historical/source", str(root).encode())
                               if name in ("lingbot14b", "solarwm33b") else original_worker.read_bytes())
            metadata["worker_sha256"] = cli.sha(original_worker)
            args += ["--worker", str(worker), "--worker-sha256", cli.sha(worker)]
            if name in ("lingbot14b", "solarwm33b"):
                args += ["--original-worker", str(original_worker)]
        self.write_json(self.root / "ORIGINAL_COMMAND_MAPPING.json", {name + ":sp1": metadata})
        self.write_json(run / "plan.json", plan)
        return args, root, run, plan

    def invoke(self, args):
        text = io.StringIO()
        with redirect_stdout(text):
            self.assertEqual(cli.main(args), 0)
        return json.loads(text.getvalue())

    def install_fixture_controller(self, name, root, run):
        # Synthetic scripts model only original startup literals. They must never execute.
        code = "PY=" + repr(sys.executable) + "\nUUID='GPU-fixture'\n"
        if name == "longlive1.3b":
            override = self.base / "fa3_torch28"
            override.mkdir()
            code += "override=" + repr(str(override)) + "\n"
            for folder in ("build", "tmp", "user_requested_repair2"):
                (self.base / folder).mkdir()
            self.write_json(self.base / "user_requested_repair2/draft_admission_retry_queue.json",
                            [dict(algorithm="draftkv", plan=str(run / "plan.json"))])
        else:
            tools = self.base / "tools"
            tools.mkdir()
            environment = tools / "environment.py"
            environment.write_text("PY=" + repr(sys.executable) + "\n")
            self.env_patch = patch.object(cli, "LINGBOT13_ENV_SHA256", cli.sha(environment))
            self.env_patch.start()
            self.addCleanup(self.env_patch.stop)
            build = self.base / "build"
            build.mkdir()
            code += "sys.path[:0]=[" + repr(str(tools)) + "]\nrenv=dict(DRAFTKV_PREPACK_BUILD=" + repr(str(build)) + ")\n"
        controller = self.base / "controller.py"
        controller.write_text(code)
        (self.base / "source").symlink_to(root, target_is_directory=True)
        registry = {**cli.PROFILE_CONTROLLERS, name: dict(name="controller.py", sha256=cli.sha(controller))}
        self.registry_patch = patch.object(cli, "PROFILE_CONTROLLERS", registry)
        self.registry_patch.start()
        self.addCleanup(self.registry_patch.stop)
        if name == "lingbot1.3b":
            tool_hashes = {f.name: cli.sha(f) for f in self.base.glob("*.py")}
            self.tools_patch = patch.object(cli, "LINGBOT13_TOOLS", tool_hashes)
            self.tools_patch.start()
            self.addCleanup(self.tools_patch.stop)
            self.write_json(self.base / "inventory.json", dict(commit=self.commit, source=str(root), tools=tool_hashes,
                rows=[dict(method="draftkv", model="LingBot1.3B", gpu=1, run_dir=str(run), plan_sha256=cli.sha(run / "plan.json"))]))
        return controller

    def snapshot(self):
        return {str(p.relative_to(self.base)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in self.base.rglob("*") if p.is_file() and not p.is_symlink()}

    def test_default_and_explicit_off_use_original_argv_and_environment(self):
        for name in cli.PROFILE_CONTROLLERS:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as folder:
                # Each case has independent prepared fixtures.
                previous = self.base
                self.base = Path(folder).resolve()
                root_patch = patch.object(cli, "ROOT", self.base / "release")
                self.root = self.base / "release"
                self.root.mkdir()
                with root_patch:
                    args, root, run, _ = self.setup_row(name)
                    before = self.snapshot()
                    omitted = self.invoke(["plan", *args])
                    explicit = self.invoke(["plan", *args, "--profile", "off"])
                    self.assertEqual(omitted, explicit)
                    self.assertEqual(omitted["profile"], "off")
                    expected = [sys.executable, "-u", "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=1"]
                    expected += (["-m", "draftkv.backbones.worker", "--plan", str(run / "plan.json")]
                                 if name == "longlive1.3b" else [str(self.base / "worker.py"), str(run / "plan.json")])
                    self.assertEqual(omitted["argv"], expected)
                    with patch.object(cli.os, "execvpe") as execute, patch.object(cli.os, "chdir"):
                        self.invoke(["run", *args])
                    first = execute.call_args
                    with patch.object(cli.os, "execvpe") as execute, patch.object(cli.os, "chdir"):
                        self.invoke(["run", *args, "--profile", "off"])
                    self.assertEqual(first, execute.call_args)
                    self.assertEqual(before, self.snapshot())
                self.base = previous

    def test_summary_missing_controller_and_unknown_modes_fail_closed(self):
        for name in cli.PROFILE_CONTROLLERS:
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "--controller"):
                cli.main(["plan", "--backbone", name, "--profile", "summary"])
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            cli.main(["--profile", "trace"])

    def test_controller_identity_is_backbone_specific(self):
        controller = self.base / "controller.py"
        controller.write_text("# not a recovered original\n")
        for name in cli.PROFILE_CONTROLLERS:
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "controller SHA"):
                cli.main(["plan", "--backbone", name, "--profile", "summary", "--controller", str(controller)])

    def test_large_original_controllers_select_single_row_api_after_matching_sha(self):
        for name in ("lingbot14b", "solarwm33b"):
            expected = cli.PROFILE_CONTROLLERS[name]["sha256"]
            controller = self.base / "controller.py"
            controller.write_text("# fixture selection only\n")
            self.assertIn("run_one", cli.PROFILE_CONTROLLERS[name]["entry"])
            with patch.object(cli, "sha", return_value=expected):
                selected = cli.select_controller(SimpleNamespace(backbone=name, sp=1, controller=controller))
            self.assertEqual(selected, controller.resolve())

    def large_fixture(self, name):
        args, root, run, plan = self.setup_row(name)
        base = self.base / "campaign"
        tools = base / "tools"
        tools.mkdir(parents=True)
        worker = tools / "worker.py"
        worker.write_bytes((self.base / "worker.py").read_bytes())
        args[args.index("--worker") + 1] = str(worker)
        (base / "runs").mkdir()
        target = base / "runs/one-original-row"
        run.rename(target)
        run = target
        args[args.index("--plan") + 1] = str(run / "plan.json")
        plan.update(run_dir=str(run), output=str(run / "native.mp4"), measurement_mode="formal")
        (base / "source").symlink_to(root, target_is_directory=True)
        index = 0 if name == "lingbot14b" else 1
        plan["selected_devices"] = [index]
        (base / f"tmp_gpu{index}").mkdir()
        (root.parent / "build").mkdir()
        reference = self.base / "reference"
        reference.mkdir()
        self.write_json(reference / "result.json", dict(status="COMPLETE", correctness={}))
        controller = tools / "run_campaign.py" if name == "lingbot14b" else base / "controller.py"
        caches = [self.base / "triton", self.base / "torchinductor"]
        for cache in caches:
            cache.mkdir()
        controller.write_text("PY=" + repr(sys.executable) + "\nGPU=1\nsettings=dict(TRITON_CACHE_DIR="
                              + repr(str(caches[0])) + ",TORCHINDUCTOR_CACHE_DIR=" + repr(str(caches[1])) + ")\n")
        if name == "lingbot14b":
            row = dict(name=run.name, mode="formal", schedule="streaming", plan=str(run / "plan.json"))
            observer = tools / "test_observers.py"
            observer.write_text("# test-only original-tool fixture\n")
            pin = patch.object(cli, "L14_TEST_OBSERVER_SHA256", cli.sha(observer))
        else:
            row = dict(name=run.name, backbone="SolarWM33B", reused=False, schedule="streaming", physical_gpu=1,
                       run_dir=str(run), reference_dir=str(reference))
            helper = base / "measurement_admission.py"
            helper.write_text("# test-only original-helper fixture\n")
            pin = patch.object(cli, "MEASUREMENT_ADMISSION_SHA256", cli.sha(helper))
        pin.start()
        self.addCleanup(pin.stop)
        known = {**cli.PROFILE_CONTROLLERS[name], "sha256": cli.sha(controller)}
        pin = patch.dict(cli.PROFILE_CONTROLLERS, {name: known})
        pin.start()
        self.addCleanup(pin.stop)
        pin = patch.dict(cli.REFERENCE_RESULT_SHA256, {name: cli.sha(reference / "result.json")})
        pin.start()
        self.addCleanup(pin.stop)
        inventory = {f.name: cli.sha(f) for f in tools.glob("*.py")}
        plan["campaign_tools"] = inventory
        mapping = json.loads((self.root / "ORIGINAL_COMMAND_MAPPING.json").read_text())
        mapping[name + ":sp1"]["external_companion_sha256"] = {}
        self.write_json(self.root / "ORIGINAL_COMMAND_MAPPING.json", mapping)
        self.write_json(run / "plan.json", plan)
        self.write_json(base / "weights-identity.json", {})
        self.write_json(base / "selected_row.json", row)
        self.write_json(base / "inventory.json", dict(rows=[row], source=str(root), commit=self.commit, gpu=index, tools=inventory))
        args += ["--profile", "summary", "--controller", str(controller), "--controller-reference", str(reference)]
        return args, base, row, root, controller

    def check_large_summary(self, name):
        args, base, row, root, controller = self.large_fixture(name)
        before = self.snapshot()
        record = self.invoke(["plan", *args])
        self.assertEqual(record["cli_status"], "CLI_STATIC_VERIFIED")
        self.assertEqual(record["argv"][3:5], ["-c", cli.SCOPED_CONTROLLER_ENTRY])
        binding = json.loads(record["argv"][-1])
        self.assertEqual(binding["row"], row)
        self.assertEqual(binding["source"], str(root))
        self.assertEqual(before, self.snapshot())
        # Execute only the stdlib selector with a mocked original module/API.
        module = SimpleNamespace(run_one=Mock(), main=Mock(), prepare=Mock())
        spec = SimpleNamespace(loader=SimpleNamespace(exec_module=Mock()))
        import importlib.util
        with patch.object(importlib.util, "spec_from_file_location", return_value=spec), \
             patch.object(importlib.util, "module_from_spec", return_value=module), \
             patch.object(sys, "argv", ["-c", str(controller), record["argv"][-1]]), patch.object(sys, "path", list(sys.path)):
            exec(cli.SCOPED_CONTROLLER_ENTRY, {})
        expected = (row,) if name == "lingbot14b" else (row, binding["tools"])
        module.run_one.assert_called_once_with(*expected)
        module.main.assert_not_called()
        module.prepare.assert_not_called()
        self.assertEqual(module.B, base)
        self.assertEqual(module.S, root)
        self.assertEqual(module.COMMIT, self.commit)
        self.assertEqual(before, self.snapshot())
        return args, base

    def test_lingbot14_original_single_row_dispatch_is_cpu_verified(self):
        args, base = self.check_large_summary("lingbot14b")
        (self.base / "reference/result.json").write_text("{}")
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, "reference result missing/changed"):
            self.invoke(["plan", *args])
        self.assertEqual(before, self.snapshot())

    def test_solar_original_single_row_dispatch_and_helper_are_cpu_verified(self):
        args, base = self.check_large_summary("solarwm33b")
        (base / "measurement_admission.py").unlink()
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, "measurement_admission.py missing/changed"):
            self.invoke(["plan", *args])
        self.assertEqual(before, self.snapshot())

    def test_large_summary_refuses_extra_rows_or_changed_tool_inventory(self):
        args, base, _, _, _ = self.large_fixture("solarwm33b")
        file = base / "inventory.json"
        inv = json.loads(file.read_text())
        inv["rows"].append(inv["rows"][0])
        self.write_json(file, inv)
        with self.assertRaisesRegex(ValueError, "one prepared inventory row"):
            self.invoke(["plan", *args])
        inv["rows"].pop()
        self.write_json(file, inv)
        (base / "tools/unverified.py").write_text("# must not be accepted\n")
        with self.assertRaisesRegex(ValueError, "tool inventory missing/changed"):
            self.invoke(["plan", *args])

    def check_summary(self, name):
        args, root, run, plan = self.setup_row(name)
        controller = self.install_fixture_controller(name, root, run)
        summary = [*args, "--profile", "summary", "--controller", str(controller)]
        if name == "lingbot1.3b":
            summary += ["--controller-device-index", "1"]
        before = self.snapshot()
        record = self.invoke(["plan", *summary])
        self.assertEqual(record["argv"], [sys.executable, "-B", "-u", str(controller)] + (["1"] if name == "lingbot1.3b" else []))
        self.assertEqual(record["profile"], "summary")
        self.assertFalse(record["gpu_test"])
        self.assertFalse(record["device_binding_checked"])
        with patch.object(cli.os, "execvpe") as execute, patch.object(cli.os, "chdir"):
            self.invoke(["run", *summary])
        self.assertEqual(execute.call_args.args[1], record["argv"])
        self.assertEqual(before, self.snapshot())
        with patch.object(cli.subprocess, "check_output", side_effect=lambda argv, **kw: "GPU-other\n" if argv[0] == "nvidia-smi" else self.command_output(argv, **kw)), patch.object(cli.os, "execvpe") as execute:
            with self.assertRaisesRegex(ValueError, "physical index"):
                self.invoke(["run", *summary])
            execute.assert_not_called()
        (self.base / "original-dependency.py").unlink()
        with self.assertRaisesRegex(ValueError, "dependency missing/changed"):
            self.invoke(["plan", *summary])
        (self.base / "original-dependency.py").write_text("# external fixture\n")
        if name == "longlive1.3b":
            queue = self.base / "user_requested_repair2/draft_admission_retry_queue.json"
            row = json.loads(queue.read_text())[0]
            self.write_json(queue, [row, row])
        else:
            inv = self.base / "inventory.json"
            value = json.loads(inv.read_text())
            value["rows"].append(value["rows"][0])
            self.write_json(inv, value)
        with self.assertRaisesRegex(ValueError, "(one prepared|single-row)"):
            self.invoke(["plan", *summary])
        return run, plan, summary

    def test_longlive_summary_selects_original_parent_without_mutation(self):
        self.check_summary("longlive1.3b")

    def test_lingbot13_summary_selects_original_main_without_mutation(self):
        self.check_summary("lingbot1.3b")

    def test_summary_rejects_missing_environment_and_companion(self):
        args, root, run, _ = self.setup_row("lingbot1.3b")
        controller = self.install_fixture_controller("lingbot1.3b", root, run)
        summary = ["plan", *args, "--profile", "summary", "--controller", str(controller), "--controller-device-index", "1"]
        (self.base / "tools/environment.py").unlink()
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, "environment.py missing/changed"):
            self.invoke(summary)
        self.assertEqual(before, self.snapshot())
        (self.base / "worker.py").write_text("# altered original worker\n")
        with self.assertRaisesRegex(ValueError, "external worker SHA"):
            self.invoke(summary)

    def test_summary_refuses_old_attempt_or_wrong_source_binding(self):
        args, root, run, _ = self.setup_row("longlive1.3b")
        controller = self.install_fixture_controller("longlive1.3b", root, run)
        summary = ["plan", *args, "--profile", "summary", "--controller", str(controller)]
        (run / "command.json").write_text("{}")
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, "existing attempt"):
            self.invoke(summary)
        self.assertEqual(before, self.snapshot())
        (run / "command.json").unlink()
        (self.base / "source").unlink()
        (self.base / "source").symlink_to(self.base / "old-research-tree")
        with self.assertRaisesRegex(ValueError, "bind this release tree"):
            self.invoke(summary)

    def test_scientific_profile_switch_is_never_rewritten(self):
        args, _, run, plan = self.setup_row("longlive1.3b")
        plan["gpu_profile"] = False
        self.write_json(run / "plan.json", plan)
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, "profile semantics mismatch"):
            self.invoke(["plan", *args])
        self.assertEqual(before, self.snapshot())

    def test_summary_never_falls_back_to_sp8_worker(self):
        with self.assertRaisesRegex(ValueError, "SP8 controller"):
            cli.main(["plan", "--backbone", "lingbot14b", "--sp", "8", "--profile", "summary"])


if __name__ == "__main__":
    unittest.main()
