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
    def __init__(self, *, lw_weight=1.0, sw_weight=1.0):
        super().__init__()
        self.lw_weight = lw_weight
        self.sw_weight = sw_weight

    @property
    def alpha11(self) -> float:
        return self.lw_weight * self.alpha11_LW + self.sw_weight * self.alpha11_SW

    @property
    def alpha12(self) -> float:
        return self.lw_weight * self.alpha12_LW + self.sw_weight * self.alpha12_SW

    @property
    def alpha21(self) -> float:
        return self.lw_weight * self.alpha21_LW + self.sw_weight * self.alpha21_SW

    @property
    def alpha22(self) -> float:
        return self.lw_weight * self.alpha22_LW + self.sw_weight * self.alpha22_SW

    @property
    def beta1(self) -> float:
        return self.lw_weight * self.beta1_LW + self.sw_weight * self.beta1_SW

    @property
    def beta2(self) -> float:
        return self.lw_weight * self.beta2_LW + self.sw_weight * self.beta2_SW


    def _one_way_radiation(self, k: float) -> np.ndarray:

        operator: np.ndarray = np.array([
            [-1j * k * self.c1 * self.alpha11, -1j * k * self.c2 * self.alpha12, 0.0, 0.0],
            [-1j * k * self.c1 * self.alpha21, -1j * k * self.c2 * self.alpha22, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0]
        ])

        return operator 

    def _two_way_radiation(self, k: float, rad_scheme: str) -> np.ndarray:

        if rad_scheme not in ["w", "q"]:
            raise ValueError("Invalid rad_scheme. Must be 'w' or 'q'.")

        tauB: float = self.tau * self.B()

        if rad_scheme == "w":

            operator: np.ndarray = np.array([
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [self.alpha11, self.alpha12, 0.0, 0.0, 0.0, 0.0],
                [self.alpha21, self.alpha22, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [(-self.f*self.alpha11-(1-self.f)*self.alpha21)/tauB, (-self.f*self.alpha12-(1-self.f)*self.alpha22)/tauB, 0.0, 0.0, 0.0, 0.0]
            ])

        elif rad_scheme == "q":
            operator: np.ndarray = np.array([
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, self.beta1, 0.0],
                [0.0, 0.0, 0.0, 0.0, self.beta2, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, (-self.f*self.beta1 - (1-self.f)*self.beta2) / tauB, 0.0]
            ])

        return operator

    def radiation(self, k: float, two_way: bool = False, rad_scheme: str = "w") -> np.ndarray:
        if two_way:
            return self._two_way_radiation(k, rad_scheme)
        else:
            return self._one_way_radiation(k)


