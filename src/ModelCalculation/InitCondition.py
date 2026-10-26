"""
    Initial condition configuration and functions
"""

# ====================================================
# import necessary module
# ====================================================

import numpy as np

from ModelCalculation.Operators import Operator

# ====================================================
# Classes
# ====================================================
class InitialCondition(Operator):

    def __init__(self, type: str, rad_scaling: float) -> None:
        """scaling factor for initial condition"""

        super().__init__(type)
        self.scale: np.ndarray | float = 1.0
        self.rad_scaling: float = rad_scaling

    def initial_condition(self, k):

        return self.initial_ensemble(k, 1)[:, 0]

    def initial_ensemble(self, k: float, members: int, rng=None) -> np.ndarray:
        """
            Generate independent noisy states from one eigenmode calculation.
            Return (n_state, members). An optional NumPy random generator keeps
            API runs independent of the global random state.
        """

        if (
            isinstance(members, (bool, np.bool_))
            or not isinstance(members, (int, np.integer))
            or members <= 0
        ):
            raise ValueError("members must be a positive integer.")

        # introduce the operator
        op = self.operator(k, self.rad_scaling)

        # solve for the most unstable mode
        eigvals, eigvecs = np.linalg.eig(op)
        max_index = np.argmax(np.real(eigvals))

        most_unstable_mode = eigvecs[:, max_index] / np.linalg.norm(eigvecs[:, max_index])

        # add random noise on the initial condition
        rng = np.random if rng is None else rng
        # draw real then imaginary noise per member, preserving seeded ordering
        noise = rng.normal(0, 0.1, (members, 2, most_unstable_mode.size))
        initial_states = most_unstable_mode[None, :] + (noise[:, 0, :] + 1j * noise[:, 1, :])

        return initial_states.T


