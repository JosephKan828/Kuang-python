"""Run with: python -m unittest discover -s tests -v."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from Configuration.Domain import Domain
from Configuration.ModelConfig import ModelConfig
from ModelCalculation.TimeStepper import TimeStepper


class TimeStepperTests(unittest.TestCase):
    def solver(self, model="one-way", radiation=1.0, dt=0.25, Tmax=0.5):
        domain = Domain()
        domain.dt, domain.Tmax = dt, Tmax
        return TimeStepper(model, radiation, domain=domain)

    def test_model_eigenmode_evolution(self):
        for model in ("one-way", "two-way"):
            for radiation in (0.0, 1.0):
                with self.subTest(model=model, radiation=radiation):
                    solver = self.solver(model, radiation)
                    k = solver.domain.k[0]
                    matrix = solver.operator(k, radiation)
                    values, vectors = np.linalg.eig(matrix)
                    # All eigenvectors evolve independently, also testing ensembles.
                    times, states = solver.simulate(k, vectors)
                    expected = vectors[None, :, :] * np.exp(
                        times[:, None, None] * values[None, None, :]
                    )
                    assert_allclose(states, expected, rtol=1e-10, atol=1e-12)
                    assert_allclose(times, [0.0, 0.25, 0.5])

    def test_defective_matrix_and_complex_state(self):
        # A Jordan block has no complete eigenvector basis: exp(A*t) = I + A*t.
        solver = self.solver()
        matrix = np.zeros((4, 4))
        matrix[0, 1] = 1.0
        initial = np.array([1j, 2 - 1j, 3, 4j])
        original = initial.copy()
        with patch.object(solver, "operator", return_value=matrix) as operator:
            times, states = solver.simulate(0.0, initial)
        expected = initial[None, :] + times[:, None] * (matrix @ initial)[None, :]
        assert_allclose(states, expected, atol=1e-14)
        assert_array_equal(initial, original)
        operator.assert_called_once()

    def test_time_boundaries(self):
        for duration, interval, expected in (
            (0.25, 0.25, [0.0, 0.25]),
            (0.5, 0.25, [0.0, 0.25, 0.5]),
            (1.0, 0.25, [0.0, 0.25, 0.5, 0.75, 1.0]),
        ):
            with self.subTest(duration=duration):
                solver = self.solver(dt=interval, Tmax=duration)
                times, states = solver.simulate(0.0, np.zeros(4))
                assert_allclose(times, expected)
                self.assertEqual(times[-1], duration)
                self.assertTrue(np.all(np.diff(times) > 0))
                assert_array_equal(states, np.zeros((len(expected), 4)))

    def test_domain_settings_and_generated_ensemble(self):
        domain = Domain()
        domain.dt, domain.Tmax, domain.Ens = 0.25, 0.5, 3
        solver = TimeStepper("two-way", domain=domain)
        initial_states = [np.arange(6) + index * 1j for index in range(domain.Ens)]
        with patch.object(solver, "initial_ensemble", return_value=np.column_stack(initial_states)) as ic:
            times, states = solver.simulate_ensemble(domain.k[0])
        ic.assert_called_once_with(domain.k[0], domain.Ens, rng=None)
        self.assertEqual(states.shape, (3, 6, 3))
        for index, initial in enumerate(initial_states):
            _, individual = solver.simulate(domain.k[0], initial)
            assert_allclose(states[:, :, index], individual, atol=1e-12)
        self.assertEqual(domain.dt, 0.25)
        self.assertEqual(domain.Tmax, 0.5)

    def test_default_initial_condition(self):
        solver = self.solver()
        initial = np.arange(4) + 1j
        with patch.object(solver, "initial_condition", return_value=initial) as ic:
            _, states = solver.simulate(0.2)
        ic.assert_called_once_with(0.2)
        assert_array_equal(states[0], initial)

    def test_propagator(self):
        solver = self.solver("two-way", dt=0.25, Tmax=0.25)
        assert_allclose(solver.propagator(0.2, dt=0), np.eye(6))
        initial = np.arange(6) + 1j
        _, states = solver.simulate(0.2, initial)
        assert_allclose(solver.propagator(0.2, solver.domain.dt) @ initial, states[-1])

    def test_invalid_inputs(self):
        for interval in (0, -1, np.nan, np.inf):
            domain = Domain()
            domain.dt = interval
            with self.assertRaises(ValueError):
                ModelConfig(domain=domain)
        for duration in (0, -1, np.nan, np.inf):
            domain = Domain()
            domain.Tmax = duration
            with self.assertRaises(ValueError):
                ModelConfig(domain=domain)
        solver = TimeStepper("one-way")
        for initial in (1, np.zeros(6), np.zeros((4, 0)), np.zeros((4, 1, 1)), [np.nan] * 4):
            with self.assertRaises(ValueError):
                solver.simulate(0.2, initial)
        for members in (0, -1, 2.5, True):
            with self.assertRaises(ValueError):
                solver.simulate_ensemble(0.2, members)
        for interval in (-1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                solver.propagator(0.2, interval)


if __name__ == "__main__":
    unittest.main()
