"""Integration checks for the Python execution workflow."""

import contextlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.linalg import expm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from RunConfig import validate_domain
from Configuration.Domain import Domain
from ModelCalculation.Operators import Operator
from Simulation import main as simulate
from EigenAnalysis import main as eigen_analysis
from Reconstruct import main as reconstruct


class RunScriptTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name)
        self.domain = Domain()
        self.domain.Ens = 2
        self.domain.Tmax = 1.0
        self.domain.dt = 0.25
        self.domain.dx = self.domain.Xmax / 4
        self.domain.dz = self.domain.Zmax / 2
        self.domain.nondim_k = [0.5, 1.0]

    def test_python_launcher_records_only_successful_completion(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "RunScript.py"
        run = self.output / "completed run"
        command = [sys.executable, str(script), "--output-dir", str(run),
                   "--members", "1", "--tmax", "0.25", "--dt", "0.25",
                   "--dx", "40000000", "--dz", "7000", "--zonal-wavenumbers", "1",
                   "--seed", "7", "--no-plot"]
        result = subprocess.run(command, cwd=self.output, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        history_path = self.output / "RunHistory.jsonl"
        record = json.loads(history_path.read_text())
        self.assertEqual(record["argv"], command)
        self.assertEqual(record["output_directory"], str(run))
        self.assertEqual(record["run_id"], run.name)
        self.assertEqual(record["working_directory"], str(self.output))
        self.assertTrue(record["completed_at"].endswith("+08:00"))
        self.assertEqual(json.loads(Path(record["record_path"]).read_text()), record)
        self.assertIn(str(history_path), result.stdout)
        self.assertIn(f"Model run completed. Results: {run}", result.stdout)
        # An output collision must not publish a new successful-run record.
        result = subprocess.run(command, cwd=self.output, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(json.loads(history_path.read_text()), record)
        self.assertEqual(len(list((run / "logs").glob("*/RunRecord.json"))), 1)

    def test_simulation_eigenvalues_and_heating(self):
        for model in ("one-way", "two-way"):
            for scaling in (0.0, 1.0):
                with self.subTest(model=model, scaling=scaling), contextlib.redirect_stdout(io.StringIO()):
                    folder = self.output / f"{model}_{scaling}"
                    simulate(model, scaling, folder, self.domain, seed=7)
                    eigen_analysis(model, scaling, folder, self.domain)
                    op = Operator(model)
                    names = ("w1", "w2", "T1", "T2", "q", "L") if model == "two-way" else ("T1", "T2", "q", "L")
                    with h5py.File(folder / "Metadata.h5") as metadata, h5py.File(folder / "State.h5") as state, h5py.File(folder / "EigenAnalysis.h5") as eigen:
                        self.assertTrue(state.attrs["complete"])
                        assert_array_equal(metadata["time"][:], self.domain.T)
                        for index, k in enumerate(self.domain.k):
                            matrix = op.operator(k, scaling)
                            initial = metadata["InitState"][:, :, index]
                            for time_index, time in enumerate(self.domain.T):
                                actual = np.stack([state[name][:, time_index, index] for name in names])
                                assert_allclose(actual, expm(matrix * time) @ initial, atol=1e-11)
                            values = eigen["Eigenvalues"][:, index]
                            assert_allclose(np.sort_complex(values), np.sort_complex(np.linalg.eigvals(matrix)))
                            assert_allclose(eigen["PhaseSpeeds"][:, index], -values.imag / k * 50)
                        if model == "two-way":
                            upper = op.r0 * state["L"][:] + op.rq * (state["q"][:] - 1.5 * state["T1"][:])
                            assert_allclose(state["J1"][:], state["L"][:] + upper)
                            assert_allclose(state["J2"][:], state["L"][:] - upper)
                        else:
                            assert_allclose(state["J1"][:], state["L"][:])
                            assert_allclose(state["J2"][:], op.g0() * state["L"][:] + op.gq() * (state["q"][:] - 1.5 * state["T1"][:]))
                    config = json.loads((folder / "Config.json").read_text())
                    self.assertEqual(config["seed"], 7)
                    self.assertEqual(config["rad_scaling"], scaling)

    def test_reconstruction_matches_direct_formula(self):
        with contextlib.redirect_stdout(io.StringIO()):
            simulate("two-way", 1.0, self.output, self.domain, seed=3)
            reconstruct(self.output, x_block_size=2)
        with h5py.File(self.output / "Metadata.h5") as metadata, h5py.File(self.output / "State.h5") as state, h5py.File(self.output / "FourierState.h5") as fourier, h5py.File(self.output / "GalerkinState.h5") as galerkin:
            x, z, k = (metadata[name][:] for name in ("x", "z", "k"))
            density = metadata["rho"][:]
            self.assertEqual(galerkin["T1"].shape, (2, 1, 5, 3, 2))
            assert_allclose(fourier["time"][:], [0.375])
            for name, mode in (("w1", 1), ("w2", 2), ("T1", 1), ("T2", 2), ("J1", 1), ("J2", 2)):
                basis = (np.pi / 2) * np.sin(mode * np.pi * z / z[-1]) / density
                if not name.startswith("w"):
                    basis *= metadata.attrs["g"] / metadata.attrs["cp"] - metadata.attrs["GammaT"]
                expected = state[name][:, :4, :].mean(axis=1)[:, None, :] * np.exp(1j * x[:, None] * k / 4_320_000)[None, :, :]
                assert_allclose(fourier[name][:, 0], expected)
                assert_allclose(galerkin[name][:, 0], expected.real[:, :, None, :] * basis[None, None, :, None])

    def test_reproducibility_and_overwrite_guard(self):
        with contextlib.redirect_stdout(io.StringIO()):
            simulate("two-way", 0.0, self.output / "first", self.domain, seed=42)
            simulate("two-way", 0.0, self.output / "second", self.domain, seed=42)
            with self.assertRaises(FileExistsError):
                simulate("two-way", 0.0, self.output / "first", self.domain, seed=99)
        with h5py.File(self.output / "first/State.h5") as first, h5py.File(self.output / "second/State.h5") as second:
            for name in first:
                assert_array_equal(first[name][:], second[name][:])

    def test_four_workers_match_serial_reconstruction(self):
        self.domain.Tmax = 2.0
        self.domain.nondim_k = [0.5, 1.0, 2.0, 3.0, 5.0]
        for model in ("one-way", "two-way"):
            with self.subTest(model=model), contextlib.redirect_stdout(io.StringIO()):
                serial = self.output / f"{model}_serial"
                parallel = self.output / f"{model}_parallel"
                simulate(model, 1.0, serial, self.domain, seed=3)
                shutil.copytree(serial, parallel)
                reconstruct(serial, x_block_size=2, workers=1)
                reconstruct(parallel, x_block_size=2, workers=4)
                for filename in ("FourierState.h5", "GalerkinState.h5"):
                    with h5py.File(serial / filename) as expected, h5py.File(parallel / filename) as actual:
                        self.assertEqual(dict(actual.attrs), dict(expected.attrs))
                        self.assertTrue(actual.attrs["complete"])
                        self.assertEqual(set(actual), set(expected))
                        for name in actual:
                            assert_array_equal(actual[name][:], expected[name][:])
                            self.assertEqual(dict(actual[name].attrs), dict(expected[name].attrs))
                            self.assertEqual(actual[name].chunks, expected[name].chunks)
                        self.assertTrue(actual["T1"].is_virtual)
                        self.assertEqual(len(actual["T1"].virtual_sources()), 4)
                for part in (parallel / "ReconstructionParts").glob("run-*/*part*.h5"):
                    with h5py.File(part) as file:
                        self.assertTrue(file.attrs["complete"])
                        self.assertEqual(file["T1"].compression, "gzip")
                with self.assertRaises(FileExistsError):
                    reconstruct(parallel, workers=4)

    def test_invalid_worker_count_does_not_create_outputs(self):
        for workers in (0, -1, True, 1.5):
            with self.subTest(workers=workers), self.assertRaisesRegex(ValueError, "workers"):
                reconstruct(self.output, workers=workers)
        self.assertEqual(list(self.output.iterdir()), [])

    def test_worker_failure_does_not_publish_outputs(self):
        with contextlib.redirect_stdout(io.StringIO()):
            simulate("two-way", 1.0, self.output, self.domain, seed=3)
            # A picklable callable that fails inside the child, exercising
            # exception propagation and process-pool cleanup.
            with patch("SimulationAnalysis.Reconstruction._reconstruct_part", new=int):
                with self.assertRaises(TypeError):
                    reconstruct(self.output, workers=4)
        for filename in ("FourierState.h5", "GalerkinState.h5"):
            self.assertFalse((self.output / filename).exists())
        self.assertEqual(list((self.output / "ReconstructionParts").iterdir()), [])

    def test_daily_means_are_read_once_and_discard_tail(self):
        from SimulationAnalysis.Reconstruction import _daily_means
        history = np.arange(2 * 11 * 3).reshape(2, 11, 3).astype(complex)
        with h5py.File(self.output / "sample.h5", "w") as file:
            data = file.create_dataset("state", data=history)
            tracked = MagicMock()
            tracked.__getitem__.side_effect = data.__getitem__
            actual = _daily_means(tracked, 1, 2, 4)
            self.assertEqual(tracked.__getitem__.call_count, 1)
        expected = np.column_stack([history[:, day * 4:(day + 1) * 4, 1].mean(axis=1)
                                    for day in range(2)])
        assert_array_equal(actual, expected)

    def test_virtual_outputs_survive_move_and_overwrite(self):
        with contextlib.redirect_stdout(io.StringIO()):
            simulate("two-way", 1.0, self.output / "original", self.domain, seed=3)
            reconstruct(self.output / "original", x_block_size=2, workers=4)
            moved = self.output / "moved run"
            (self.output / "original").rename(moved)
            with h5py.File(moved / "GalerkinState.h5") as file:
                expected = file["T1"][:]
                self.assertTrue(np.isfinite(expected).all())
                self.assertGreater(np.abs(expected).max(), 0)
                self.assertEqual(len(file["T1"].virtual_sources()), 2)
            old_parts = set((moved / "ReconstructionParts").glob("run-*"))
            unrelated = moved / "ReconstructionParts" / "notes.txt"
            unrelated.write_text("keep this")
            # A failed replacement preserves the previous, readable generation.
            with patch("SimulationAnalysis.Reconstruction._reconstruct_part", new=int):
                with self.assertRaises(TypeError):
                    reconstruct(moved, workers=4, overwrite=True)
            with h5py.File(moved / "GalerkinState.h5") as file:
                assert_array_equal(file["T1"][:], expected)
            reconstruct(moved, x_block_size=3, workers=1, overwrite=True)
            with h5py.File(moved / "GalerkinState.h5") as file:
                assert_array_equal(file["T1"][:], expected)
                self.assertEqual(len(file["T1"].virtual_sources()), 1)
            self.assertTrue(all(not path.exists() for path in old_parts))
            self.assertEqual(unrelated.read_text(), "keep this")

    def test_one_way_reconstruction_with_no_complete_day(self):
        self.domain.Tmax = 0.25
        with contextlib.redirect_stdout(io.StringIO()):
            simulate("one-way", 1.0, self.output, self.domain, seed=1)
            reconstruct(self.output)
        with h5py.File(self.output / "GalerkinState.h5") as file:
            self.assertNotIn("w1", file)
            self.assertEqual(file["T1"].shape, (2, 0, 5, 3, 2))

    def test_incompatible_time_settings(self):
        self.domain.Tmax = 0.35
        with self.assertRaisesRegex(ValueError, "multiple"):
            validate_domain(self.domain)


if __name__ == "__main__":
    unittest.main()
