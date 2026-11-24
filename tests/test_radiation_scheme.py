"""Radiation choices must survive configuration, calculation, and output."""

import json
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from API import (Domain, ModelConfig, Operator, TimeStepper, analyze_eigenmodes,
                 reconstruct_fields, run_model, simulate_wavenumber)
from Configuration.Params import Radiation
from ModelCalculation.Radiation import RadiationOperator


class RadiationSchemeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        self.domain = Domain()
        self.domain.Tmax, self.domain.dt, self.domain.Ens = 1.0, 0.25, 2
        self.domain.dx, self.domain.dz = self.domain.Xmax / 4, self.domain.Zmax / 2
        self.domain.nondim_k = [0.5, 1.0]

    def config(self, **kwargs):
        return ModelConfig(domain=self.domain, seed=17, rad_scheme="q",
                           lw_weight=0.35, sw_weight=1.2, **kwargs)

    def moisture_matrix(self, k, config):
        """Evaluate moisture heating and its convective response independently."""
        operator = Operator("two-way")
        params = Radiation()
        heating = config.rad_scaling * (
            config.lw_weight * np.array([params.beta1_LW, params.beta2_LW])
            + config.sw_weight * np.array([params.beta1_SW, params.beta2_SW])
        )
        matrix = operator.operator(k, 0.0)
        matrix[2:4, 4] += heating
        matrix[5, 4] -= np.dot([operator.f, 1 - operator.f], heating) / (operator.tau * operator.B())
        return matrix

    def test_q_initialization_propagation_and_eigenanalysis(self):
        config = self.config(rad_scaling=0.7)
        k = config.domain.k[0]
        matrix = self.moisture_matrix(k, config)
        values, vectors = np.linalg.eig(matrix)
        mode = vectors[:, np.argmax(values.real)]
        mode /= np.linalg.norm(mode)
        noise = np.random.RandomState(config.seed).normal(
            0, 0.1 / np.sqrt(2), (config.domain.Ens, 2, 6),
        )
        initial = (mode[None, :] * (noise[:, 0] + 1j * noise[:, 1])).T
        history = simulate_wavenumber(config, k)
        assert_allclose(history.states[0], initial, atol=1e-14)
        for index, time in enumerate(history.times):
            assert_allclose(history.states[index], expm(matrix * time) @ initial, atol=1e-12)

        stepper = TimeStepper("two-way", config.rad_scaling, domain=config.domain,
                              rad_scheme=config.rad_scheme, lw_weight=config.lw_weight,
                              sw_weight=config.sw_weight)
        assert_allclose(stepper.operator(k, config.rad_scaling), matrix)
        assert_allclose(stepper.propagator(k, config.domain.dt), expm(matrix * config.domain.dt))
        assert_allclose(stepper.radiation(k, two_way=True),
                        (matrix - Operator("two-way").operator(k, 0)) / config.rad_scaling,
                        atol=1e-14)
        result = analyze_eigenmodes(config)
        assert_allclose(np.sort_complex(result.eigenvalues[:, 0]), np.sort_complex(values), atol=1e-12)
        self.assertEqual((result.rad_scheme, result.lw_weight, result.sw_weight), ("q", 0.35, 1.2))
        circulation = ModelConfig(domain=self.domain, seed=17, rad_scaling=0.7,
                                  lw_weight=0.35, sw_weight=1.2)
        other = simulate_wavenumber(circulation, k, initial)
        self.assertFalse(np.allclose(history.states[-1], other.states[-1]))

    def test_zero_scaling_and_zero_weights_remove_radiation(self):
        k = self.domain.k[0]
        initial = np.arange(1, 7, dtype=complex)
        reference = simulate_wavenumber(ModelConfig(domain=self.domain, rad_scaling=0), k, initial)
        for config in (self.config(rad_scaling=0),
                       ModelConfig(domain=self.domain, rad_scheme="q", lw_weight=0, sw_weight=0)):
            with self.subTest(config=config):
                assert_array_equal(simulate_wavenumber(config, k, initial).states, reference.states)

    def test_weighted_components_add_for_each_supported_scheme(self):
        k = self.domain.k[0]
        lw = RadiationOperator(lw_weight=1, sw_weight=0)
        sw = RadiationOperator(lw_weight=0, sw_weight=1)
        mixed = RadiationOperator(lw_weight=0.35, sw_weight=1.2)
        for two_way, scheme in ((False, "w"), (True, "w"), (True, "q")):
            with self.subTest(two_way=two_way, scheme=scheme):
                assert_allclose(mixed.radiation(k, two_way, scheme),
                                0.35 * lw.radiation(k, two_way, scheme)
                                + 1.2 * sw.radiation(k, two_way, scheme), atol=1e-14)

    def test_invalid_and_unsupported_settings_fail_before_output(self):
        for scheme in ("invalid", "q"):
            for factory in (
                lambda: ModelConfig("one-way", domain=self.domain, rad_scheme=scheme),
                lambda: TimeStepper("one-way", domain=self.domain, rad_scheme=scheme),
                lambda: RadiationOperator().radiation(1, rad_scheme=scheme),
                lambda: Operator("one-way").operator(1, 0, rad_scheme=scheme),
                lambda: Operator("one-way", rad_scheme=scheme),
            ):
                with self.subTest(scheme=scheme, factory=factory), self.assertRaisesRegex(ValueError, "rad_scheme"):
                    factory()
        for name in ("lw_weight", "sw_weight"):
            for value in (np.nan, np.inf, -np.inf):
                for factory in (ModelConfig, RadiationOperator):
                    with self.subTest(name=name, value=value, factory=factory), self.assertRaisesRegex(ValueError, name):
                        factory(**{name: value})
        config = ModelConfig(domain=self.domain)
        config.rad_scheme = "invalid"
        with self.assertRaisesRegex(ValueError, "rad_scheme"):
            run_model(config, self.output / "invalid")
        self.assertEqual(list(self.output.iterdir()), [])

    def test_scheme_and_weights_are_saved_and_reconstructed(self):
        config = self.config(rad_scaling=0.7)
        result = run_model(config, self.output, reconstruct=True, reconstruction_workers=1)
        snapshot = json.loads(result.simulation.configuration.read_text())
        for name in ("rad_scheme", "lw_weight", "sw_weight"):
            self.assertEqual(snapshot[name], getattr(config, name))
        files = [result.simulation.metadata, result.simulation.state, result.eigen_file,
                 result.reconstruction.fourier, result.reconstruction.galerkin]
        files.extend((self.output / "ReconstructionParts").glob("run-*/*part*.h5"))
        for path in files:
            with self.subTest(path=path), h5py.File(path) as file:
                for name in ("rad_scheme", "lw_weight", "sw_weight"):
                    self.assertEqual(file.attrs[name], getattr(config, name))
        with h5py.File(result.simulation.metadata) as file:
            self.assertEqual(json.loads(file.attrs["configuration"]), snapshot)
            self.assertEqual(file["RadqScale"][()], 0.7)
            self.assertEqual(file["RadwScale"][()], 0.0)

    def test_old_outputs_keep_working_and_plot_labels_distinguish_schemes(self):
        from matplotlib.axes import Axes
        from Plot_Eigen import main as plot

        legacy = self.output / "legacy"
        modern = self.output / "modern"
        run_model(ModelConfig(domain=self.domain, seed=17), legacy)
        run_model(self.config(), modern)
        for filename in ("Metadata.h5", "State.h5", "EigenAnalysis.h5"):
            with h5py.File(legacy / filename, "r+") as file:
                for name in ("rad_scheme", "lw_weight", "sw_weight"):
                    del file.attrs[name]
        reconstructed = reconstruct_fields(legacy, workers=1)
        with h5py.File(reconstructed.galerkin) as file:
            self.assertEqual(file.attrs["rad_scheme"], "w")
            self.assertEqual(file.attrs["lw_weight"], 1)
            self.assertEqual(file.attrs["sw_weight"], 1)
        with patch.object(Axes, "plot", autospec=True, side_effect=Axes.plot) as plotted:
            figures = plot([legacy, modern])
        labels = {call.kwargs["label"] for call in plotted.call_args_list}
        self.assertEqual(labels, {"Scheme w, scaling = 1\nLW = 1, SW = 1",
                                  "Scheme q, scaling = 1\nLW = 0.35, SW = 1.2"})
        self.assertTrue((figures / "EigenAnalysis.png").is_file())

    def test_shell_pipeline_accepts_q_and_nondefault_weights(self):
        parent = self.output / "pipeline"
        command = ["bash", str(ROOT / "scripts" / "RunScript.sh"), "--python", sys.executable,
                   "--output-dir", str(parent), "--model", "two-way", "--rad-scheme", "q",
                   "--lw-weight", "0.35", "--sw-weight", "1.2", "--rad-scaling", "0.7",
                   "--members", "2", "--tmax", "1", "--dt", "0.25", "--dx", "10000000",
                   "--dz", "7000", "--zonal-wavenumbers", "0.5", "1", "--seed", "17",
                   "--reconstruction-workers", "1", "--plot", "--no-email"]
        completed = subprocess.run(command, cwd=self.output, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        record = json.loads((parent / "RunHistory.jsonl").read_text())
        directory = Path(record["output_directory"])
        snapshot = json.loads((directory / "Config.json").read_text())
        self.assertEqual((snapshot["rad_scheme"], snapshot["lw_weight"], snapshot["sw_weight"]),
                         ("q", 0.35, 1.2))
        config = self.config(rad_scaling=0.7)
        with h5py.File(directory / "Metadata.h5") as metadata, \
             h5py.File(directory / "State.h5") as state, \
             h5py.File(directory / "EigenAnalysis.h5") as eigen:
            names = ("w1", "w2", "T1", "T2", "q", "L")
            for index, k in enumerate(metadata["k"][:]):
                matrix = self.moisture_matrix(k, config)
                initial = metadata["InitState"][:, :, index]
                for t_index, time in enumerate(metadata["time"][:]):
                    actual = np.stack([state[name][:, t_index, index] for name in names])
                    assert_allclose(actual, expm(matrix * time) @ initial, atol=1e-12)
                assert_allclose(np.sort_complex(eigen["Eigenvalues"][:, index]),
                                np.sort_complex(np.linalg.eigvals(matrix)), atol=1e-12)
        with h5py.File(directory / "GalerkinState.h5") as file:
            self.assertEqual(file.attrs["rad_scheme"], "q")
        self.assertTrue((directory / "Figures" / "EigenAnalysis.png").is_file())


if __name__ == "__main__":
    unittest.main()
