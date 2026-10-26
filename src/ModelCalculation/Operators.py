"""
    This script contains the operator responsible for 
    different physical processes
"""

# ====================================================
# Import package
# ====================================================

import numpy as np

from Configuration.Params import Convection, Moisture, Dynamics, Radiation
from ModelCalculation.Radiation import RadiationOperator

# ====================================================
# Classes
# ====================================================

class Operator(RadiationOperator, Moisture):
    """
        This class represents the operator for the linear system associated with
        one-way and two-way system
    """
    
    def __init__(self, type: str) -> None:
        super().__init__()
        self.type = type

        if self.type not in ["one-way", "two-way"]:
            raise ValueError("Invalid operator type. Must be 'one-way' or 'two-way'.")

    def _one_way_operator(self, k:float, rad_scaling: float):

        D: float = self.tau*(self.b1+self.b2*self.g0())

        M: np.ndarray = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [-self.F*self.f/D, -self.F*(1-self.f)/D, 0.0, 1.0]
        ])

        K: np.ndarray = np.array([
            [1j*k*self.c1, 0.0, 0.0, 1.0],
            [1.5*self.gq(), 1j*k*self.c2, -self.gq(), self.g0()],
            [1.5*self.gq(), 0.0, -self.gq(), self.m1+self.m2*self.g0()],
            [-1.5*self.b2*self.gq()/D, 0.0, self.b2*self.gq()/D, -1/self.tau]
        ]) + rad_scaling*self.radiation(k, False)

        return np.linalg.solve(M, K)

    def _two_way_operator(self, k:float, rad_scaling: float):

        delta: float = 1.5*self.rq*(self.d1-self.d2)
        zeta: float = self.rq*(self.d2 - self.d1)
        psi: float = (-self.d1*(1+self.r0) - self.d2*(1-self.r0))
        tauB: float = self.tau * self.B()

        K: np.ndarray = np.array([
            [-self.epsilon, 0.0, (k*self.c1)**2, 0.0, 0.0, 0.0],
            [0.0, -self.epsilon, 0.0, (k*self.c2)**2, 0.0, 0.0],
            [-1.0, 0.0, -1.5*self.rq, 0.0, self.rq, 1+self.r0],
            [0.0, -1.0, 1.5*self.rq, 0.0, -self.rq, 1-self.r0],
            [self.a1, self.a2, delta, 0.0, zeta, psi],
            [self.f/tauB, (1-self.f)/tauB, -1.5*(self.A()*self.rq)/tauB, 0.0, self.A()*self.rq/tauB, -1/self.tau]
        ]) + rad_scaling*self.radiation(k, True)

        return np.linalg.solve(np.eye(6), K)

    def operator(self, k:float, rad_scaling: float):
        if self.type == "one-way":
            return self._one_way_operator(k, rad_scaling)
        else:
            return self._two_way_operator(k, rad_scaling)

