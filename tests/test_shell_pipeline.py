"""Shell pipeline ordering, argument handling, and failure propagation."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


PIPELINE = Path(__file__).resolve().parents[1] / "scripts" / "RunScript.sh"


class ShellPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="kuang shell ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.calls = self.root / "calls.jsonl"
        self.output = self.root / "run with spaces"
        self.python = self.root / "fake python"
        self.python.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys\n"
            "from pathlib import Path\n"
            "script = Path(sys.argv[2]).name\n"
            "with open(os.environ['PIPELINE_TEST_CALLS'], 'a') as file:\n"
            "    file.write(json.dumps([script, *sys.argv[3:]]) + '\\n')\n"
            "print('stage output: ' + script, flush=True)\n"
            "if script == os.environ.get('PIPELINE_TEST_FAIL'):\n"
            "    print('deliberate stage failure', file=sys.stderr)\n"
            "    sys.exit(7)\n"
        )
        self.python.chmod(0o755)
        self.env = dict(os.environ, PIPELINE_TEST_CALLS=str(self.calls))

    def run_pipeline(self, *args, reuse=False):
        return subprocess.run(
            ["bash", str(PIPELINE), "--python", str(self.python),
             "--run-dir" if reuse else "--output-dir", str(self.output), *args],
            cwd=self.root, env=self.env, text=True, capture_output=True,
        )

    def recorded_calls(self):
        return [json.loads(line) for line in self.calls.read_text().splitlines()]

    def test_full_stage_order_and_shared_arguments(self):
        result = self.run_pipeline("--members", "2", "--seed", "42", "--rad-scaling", "0.0",
                                   "--zonal-wavenumbers", "0.5", "1", "--reconstruct")
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.recorded_calls()
        self.assertEqual([call[0] for call in calls],
                         ["Simulation.py", "EigenAnalysis.py", "Reconstruct.py"])
        self.assertEqual(calls[0][1:], calls[1][1:])
        run = Path(calls[0][-1])
        self.assertEqual(run.parent, self.output)
        self.assertRegex(run.name, r"^\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-\d+-\d+$")
        self.assertTrue(run.is_dir())
        self.assertEqual(calls[0][-2:], ["--output-dir", str(run)])
        self.assertEqual(calls[2][1:], [str(run), "--x-block-size", "64", "--workers", "4"])
        logs = list((run / "logs").glob("*/*.log"))
        self.assertEqual(len(logs), 3)
        for log in logs:
            self.assertIn("stage output:", log.read_text())

    def test_failure_stops_later_stages_and_preserves_exit_code(self):
        self.env["PIPELINE_TEST_FAIL"] = "EigenAnalysis.py"
        result = self.run_pipeline("--reconstruct")
        self.assertEqual(result.returncode, 7)
        self.assertEqual([call[0] for call in self.recorded_calls()], ["Simulation.py", "EigenAnalysis.py"])
        self.assertIn("Stage eigen_analysis failed", result.stderr)
        run = Path(self.recorded_calls()[0][-1])
        log = next((run / "logs").glob("*/eigen_analysis.log"))
        self.assertIn("deliberate stage failure", log.read_text())

    def test_reconstruction_only_uses_existing_output_and_worker_override(self):
        self.output.mkdir()
        (self.output / "Metadata.h5").touch()
        (self.output / "State.h5").touch()
        result = self.run_pipeline("--skip-simulation", "--skip-analysis", "--reconstruction-workers", "1", reuse=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.recorded_calls(), [["Reconstruct.py", str(self.output),
                                               "--x-block-size", "64", "--workers", "1"]])

    def test_preflight_collision_stops_before_simulation(self):
        self.output.mkdir()
        (self.output / "EigenAnalysis.h5").write_bytes(b"existing output")
        result = self.run_pipeline(reuse=True)
        self.assertEqual(result.returncode, 2)
        self.assertFalse(self.calls.exists())
        self.assertEqual((self.output / "EigenAnalysis.h5").read_bytes(), b"existing output")

    def test_repeated_runs_create_separate_directories_and_preserve_parent_files(self):
        self.output.mkdir()
        existing = self.output / "State.h5"
        existing.write_bytes(b"previous run")
        for _ in range(2):
            result = self.run_pipeline()
            self.assertEqual(result.returncode, 0, result.stderr)
        runs = [Path(call[-1]) for call in self.recorded_calls() if call[0] == "Simulation.py"]
        self.assertEqual(len(set(runs)), 2)
        for run in runs:
            self.assertEqual(run.parent, self.output)
            self.assertTrue((run / "logs").is_dir())
        self.assertEqual(existing.read_bytes(), b"previous run")
        self.assertFalse((self.output / "logs").exists())

    def test_existing_run_directory_must_exist(self):
        result = self.run_pipeline(reuse=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("Run directory does not exist", result.stderr)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.calls.exists())

    def test_invalid_options_and_missing_inputs(self):
        for arguments in (("--seed",), ("--reconstruct", "--x-block-size", "0"),
                          ("--reconstruction-workers", "0"), ("--reconstruction-workers", "-1"),
                          ("--reconstruction-workers", "1.5"), ("--reconstruction-workers",),
                          ("--skip-simulation", "--skip-analysis"),
                          ("--skip-simulation", "--skip-analysis", "--no-plot"),
                          ("--unknown-option",)):
            with self.subTest(arguments=arguments):
                result = self.run_pipeline(*arguments)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertFalse(self.calls.exists())
        result = self.run_pipeline("--run-dir", str(self.root))
        self.assertEqual(result.returncode, 2)
        self.assertIn("not both", result.stderr)
        self.assertFalse(self.calls.exists())


if __name__ == "__main__":
    unittest.main()
