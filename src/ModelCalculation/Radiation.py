"""
    This script contains the operator responsible for 
    radiation processes
"""

# ====================================================
# Import package
# ====================================================

import numpy as np

from Configuration.Params import Convection, Dynamics, Radiation

# ====================================================
# Define radiation operator
# ====================================================

class RadiationOperator(Dynamics, Convection, Radiation):
    def __init__(self):
        super().__init__()

        self.alpha11: float = self.alpha11_LW + self.alpha11_SW
        self.alpha12: float = self.alpha12_LW + self.alpha12_SW
        self.alpha21: float = self.alpha21_LW + self.alpha21_SW
        self.alpha22: float = self.alpha22_LW + self.alpha22_SW

    def _one_way_radiation(self, k: float) -> np.ndarray:

        operator: np.ndarray = np.array([
            [-1j * k * self.c1 * self.alpha11, -1j * k * self.c2 * self.alpha12, 0.0, 0.0],
            [-1j * k * self.c1 * self.alpha21, -1j * k * self.c2 * self.alpha22, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0]
        ])

        return operator 

    def _two_way_radiation(self, k: float) -> np.ndarray:

        operator: np.ndarray = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [self.alpha11, self.alpha12, 0.0, 0.0, 0.0, 0.0],
            [self.alpha21, self.alpha22, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-self.f*self.alpha11-(1-self.f)*self.alpha21, -self.f*self.alpha12-(1-self.f)*self.alpha22, 0.0, 0.0, 0.0, 0.0]
        ])

        return operator

    def radiation(self, k: float, two_way: bool = False) -> np.ndarray:
        if two_way:
            return self._two_way_radiation(k)
        else:
            return self._one_way_radiation(k)


