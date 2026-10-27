"""
    This script contains the matrix-exponential time-stepping methods
    for the one-way and two-way systems
"""

# ====================================================
# Import package
# ====================================================

import numpy as np

from scipy.linalg import expm

from Configuration.Domain import Domain
from ModelCalculation.InitCondition import InitialCondition
# ====================================================
# Classes
# ====================================================

class TimeStepper(InitialCondition):
    """
        This class advances the linear system dx/dt = A x at a given wavenumber
        using the matrix-exponential propagator exp(A * dt).

        Simulation configuration
        ---------------------
        type       : one-way or two-way system
        rad_scaling: scaling factor for radiative feedback
        domain     : domain settings and default ensemble size
        dt, Tmax   : time interval and simulation duration; default to Domain

        Time uses the same units as the operator's inverse rates. Model
        parameters and radiative feedback remain fixed during each simulation.
    """

    def __init__(
        self,
        type: str,
        rad_scaling: float = 1.0,
        *,
        domain: Domain | None = None,
    ) -> None:

        super().__init__(type, rad_scaling)

        self.domain: Domain = Domain() if domain is None else domain

        self._validate_time_settings()

    def _validate_time_settings(self) -> None:
        """Check the time interval and simulation duration"""

        time_coords: np.ndarray = self.domain.T

        if time_coords.size < 2:
            raise ValueError("The time axis must have at least two points.")

    def propagator(self, k: float, dt: float) -> np.ndarray:
        """Matrix-exponential propagator; a zero interval gives the identity"""

        interval: float = dt
        if not np.isfinite(interval) or interval < 0:
            raise ValueError("The propagation interval must be finite and nonnegative.")

        op: np.ndarray = self.operator(k, self.rad_scaling)

        return expm(op * interval)

    def simulate(
        self, k: float, initial_state: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
            Return the time axis and complex state history, including time zero.

            State dimensions
            ---------------------
            single state : (n_state,) -> (n_time, n_state)
            ensemble     : (n_state, n_members) -> (n_time, n_state, n_members)

            n_state is four for one-way and six for two-way systems. If no
            initial state is supplied, use zero-mean complex random states
            weighted component-wise by the most unstable eigenmode from
            InitialCondition. Randomization occurs only at initialization.
        """

        # introduce the time axis and evolution operator
        times: np.ndarray = self.domain.T
        op: np.ndarray = self.operator(k, self.rad_scaling)

        # introduce and validate the initial condition
        if initial_state is None:
            initial_state = self.initial_condition(k)
        state: np.ndarray = np.asarray(initial_state, dtype=complex)

        if state.ndim not in (1, 2) or state.shape[0] != op.shape[0]:
            raise ValueError(
                f"initial_state must have shape ({op.shape[0]},) or "
                f"({op.shape[0]}, n_members)."
            )
        if state.ndim == 2 and state.shape[1] == 0:
            raise ValueError("The ensemble must contain at least one member.")
        if not np.all(np.isfinite(state)):
            raise ValueError("initial_state must contain only finite values.")

        # allocate the state history and store the initial condition
        states: np.ndarray = np.empty((times.size,) + state.shape, dtype=complex)
        states[0] = state

        if times.size == 1:
            return times, states

        # reuse the full-step propagator throughout the simulation
        full_step: np.ndarray = expm(op * self.domain.dt)

        for index in range(1, times.size):
            propagator = full_step

            # adjust the propagator for a shorter final interval
            if index == times.size - 1:
                interval = times[index] - times[index - 1]
                if not np.isclose(interval, self.domain.dt, rtol=1e-12, atol=0):
                    propagator = expm(op * interval)

            states[index] = propagator @ states[index - 1]

        return times, states

    def simulate_ensemble(
        self, k: float, members: int | None = None, *, rng=None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
            Simulate an ensemble with independent noisy initial conditions.
            The default number of members is Domain.Ens. For a prescribed
            initial ensemble, pass its state array directly to simulate.
            Pass rng to generate noise without changing the global random state.
        """

        members = self.domain.Ens if members is None else members
        if (
            isinstance(members, (bool, np.bool_))
            or not isinstance(members, (int, np.integer))
            or members <= 0
        ):
            raise ValueError("members must be a positive integer.")
        # share the eigenmode calculation across all ensemble members
        initial_states: np.ndarray = self.initial_ensemble(k, members, rng=rng)

        return self.simulate(k, initial_states)
