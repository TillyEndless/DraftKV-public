"""CPU acceptance: original-byte integrity and fail-closed public selection."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
LAUNCH = ROOT / "tools/launch_original.py"

class OuterAcceptance(unittest.TestCase):
    def cli(self, *args):
        return subprocess.run([sys.executable, "-B", str(LAUNCH), *args], capture_output=True, text=True)

    def test_retained_original_source_bytes(self):
        result = self.cli("check")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_four_backbones_and_two_distinct_original_sp8_paths(self):
        result = self.cli("list")
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = result.stdout.splitlines()
        self.assertEqual(len(lines), 6)
        self.assertEqual(sum("SP=8" in line for line in lines), 2)
        self.assertTrue(all("draftkv" in line for line in lines))

    def test_reject_baseline_and_ablation_cli(self):
        for variant in ("worldkv", "native", "camera-only", "draft-only"):
            with self.subTest(variant=variant):
                result = self.cli("plan", "--backbone", "lingbot14b", "--variant", variant)
                self.assertNotEqual(result.returncode, 0)

    def test_missing_original_plan_fails_before_inference(self):
        result = self.cli("plan", "--backbone", "lingbot14b")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--plan", result.stderr)

    def test_no_single_card_sp8_substitute(self):
        result = self.cli("plan", "--backbone", "longlive1.3b", "--sp", "8")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unsupported", result.stderr)

if __name__ == "__main__":
    unittest.main()
