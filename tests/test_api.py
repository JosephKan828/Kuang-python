"""Checks for the reusable model API and ensemble initialization."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.linalg import expm

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from API import (Domain, ModelConfig, Operator, analyze_eigenmodes, iter_simulation,
                 run_model, run_simulation, simulate_wavenumber)
from Output.ModelIO import SimulationWriter
from SimulationAnalysis.SimulationRun import SimulationRun, TimeSeries


class APITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        domain = Domain()
        domain.Tmax, domain.dt, domain.Ens = 1.0, 0.25, 3
        domain.dx, domain.dz = domain.Xmax / 4, domain.Zmax / 2
        domain.nondim_k = [0.5, 1.0]
        self.config = ModelConfig(domain=domain, seed=17)

    def test_import_without_scripts_or_side_effects(self):
        code = (
            f"import sys; sys.path.insert(0, {str(SRC)!r}); import API; "
            "assert 'RunConfig' not in sys.modules; "
            "assert 'matplotlib' not in sys.modules"
        )
        subprocess.run([sys.executable, "-B", "-c", code], cwd=self.output, check=True)
        self.assertEqual(list(self.output.iterdir()), [])

    def test_in_memory_results_and_seeded_equivalence(self):
        for model in ("one-way", "two-way"):
            with self.subTest(model=model):
                self.config.model_type = model
                k = self.config.domain.k[0]
                matrix = Operator(model).operator(k, self.config.rad_scaling)
                values, vectors = np.linalg.eig(matrix)
                mode = vectors[:, np.argmax(values.real)]
                mode = mode / np.linalg.norm(mode)
                legacy_rng = np.random.RandomState(self.config.seed)
                initial = np.column_stack([
                    mode + (legacy_rng.normal(0, 0.1, mode.shape)
                            + 1j * legacy_rng.normal(0, 0.1, mode.shape))
                    for _ in range(self.config.domain.Ens)
                ])
                with patch("numpy.linalg.eig", wraps=np.linalg.eig) as eig:
                    result = simulate_wavenumber(self.config, k)
                self.assertEqual(eig.call_count, 1)
                assert_array_equal(result.states[0], initial)
                for index, time in enumerate(result.times):
                    assert_allclose(result.states[index], expm(matrix * time) @ initial, atol=1e-11)
                explicit = simulate_wavenumber(self.config, k, initial[:, 0])
                assert_allclose(explicit.states, result.states[:, :, 0], atol=1e-11)
        self.assertEqual(list(self.output.iterdir()), [])

    def test_api_preserves_global_rng(self):
        before = np.random.get_state()
        simulate_wavenumber(self.config, self.config.domain.k[0])
        run_simulation(self.config, self.output)
        after = np.random.get_state()
        self.assertEqual(before[0], after[0])
        assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])

    def test_eigen_arrays_match_optional_output(self):
        result = analyze_eigenmodes(self.config)
        self.assertEqual(list(self.output.iterdir()), [])
        saved = analyze_eigenmodes(self.config, self.output)
        assert_array_equal(result.eigenvalues, saved.eigenvalues)
        assert_allclose(result.dominant_growth, result.growth_rates.max(axis=0), atol=1e-12)
        with h5py.File(self.output / "EigenAnalysis.h5") as file:
            assert_array_equal(file["GrowthRates"][:], result.growth_rates)
            assert_array_equal(file["PhaseSpeeds"][:], result.phase_speeds)

    def test_combined_workflow_and_preflight(self):
        result = run_model(self.config, self.output, reconstruct=True, x_block_size=2)
        for file in (result.simulation.state, result.simulation.metadata,
                     result.simulation.configuration, result.eigen_file,
                     result.reconstruction.fourier, result.reconstruction.galerkin):
            self.assertTrue(file.is_file())
        with self.assertRaises(FileExistsError):
            run_model(self.config, self.output)
        invalid = self.output / "invalid"
        with self.assertRaises(ValueError):
            run_model(self.config, invalid, reconstruct=True, x_block_size=0)
        self.assertFalse(invalid.exists())
        with self.assertRaisesRegex(ValueError, "workers"):
            run_model(self.config, invalid, reconstruct=True, reconstruction_workers=0)
        self.assertFalse(invalid.exists())

    def test_config_owns_domain_and_validates_mutations(self):
        original = Domain()
        config = ModelConfig(domain=original)
        config.domain.nondim_k.append(99)
        self.assertNotIn(99, original.nondim_k)
        self.assertNotIn(99, ModelConfig().domain.nondim_k)
        for seed in (-1, 2**32, 0.5, True):
            with self.assertRaises(ValueError):
                ModelConfig(seed=seed)
        self.config.rad_scaling = np.nan
        with self.assertRaises(ValueError):
            analyze_eigenmodes(self.config)

    def test_iterator_is_lazy_and_matches_saved_run(self):
        with patch("numpy.linalg.eig", wraps=np.linalg.eig) as eig:
            histories = iter_simulation(self.config)
            self.assertEqual(eig.call_count, 0)
            first = next(histories)
            self.assertEqual(eig.call_count, 1)
            second = next(histories)
            self.assertEqual(eig.call_count, 2)
            with self.assertRaises(StopIteration):
                next(histories)
        self.assertEqual(list(self.output.iterdir()), [])
        files = run_simulation(self.config, self.output)
        with h5py.File(files.state) as file:
            for index, history in enumerate((first, second)):
                for position, name in enumerate(history.state_names):
                    assert_array_equal(file[name][:, :, index], history.states[:, position, :].T)
                for name in ("J1", "J2"):
                    assert_array_equal(file[name][:, :, index], history.heating[name].T)

    def test_writer_only_saves_supplied_arrays(self):
        metadata = SimulationRun(self.config).metadata()
        times = metadata.coordinates["time"]
        shape = (times.size, len(metadata.state_names), self.config.domain.Ens)
        # Supply deliberately synthetic states/heating, without any model solve.
        histories = [TimeSeries(
            k, times, np.full(shape, index + 3j), metadata.state_names,
            {"J1": np.full((times.size, shape[2]), 101 + index + 4j),
             "J2": np.full((times.size, shape[2]), 201 + index + 5j)},
        ) for index, k in enumerate(metadata.coordinates["k"])]
        with patch(f"{SimulationRun.__module__}.TimeStepper", side_effect=AssertionError("Writer ran the model")), \
             patch(f"{SimulationRun.__module__}.MeanState", side_effect=AssertionError("Writer computed profiles")):
            with SimulationWriter(metadata, self.output) as writer:
                for history in histories:
                    writer.write_wavenumber(history)
        with h5py.File(self.output / "State.h5") as file:
            self.assertTrue(file.attrs["complete"])
            for index, history in enumerate(histories):
                assert_array_equal(file["T1"][:, :, index], history.states[:, 2, :].T)
                assert_array_equal(file["J1"][:, :, index], history.heating["J1"].T)

    def test_incomplete_or_out_of_order_output_is_not_complete(self):
        run = SimulationRun(self.config)
        metadata = run.metadata()
        histories = list(run)
        for case in ("missing", "wrong_order"):
            folder = self.output / case
            with self.subTest(case=case), self.assertRaises(ValueError):
                with SimulationWriter(metadata, folder) as writer:
                    writer.write_wavenumber(histories[0] if case == "missing" else histories[1])
            with h5py.File(folder / "State.h5") as state, h5py.File(folder / "Metadata.h5") as meta:
                self.assertFalse(state.attrs["complete"])
                self.assertFalse(meta.attrs["complete"])


if __name__ == "__main__":
    unittest.main()
