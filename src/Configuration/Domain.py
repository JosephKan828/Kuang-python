"""
    Classes documenting the information of domain setup,
    simulation configuration, and also experiment design
"""

# ====================================================
# Import package
# ====================================================

import numpy as np

from dataclasses import dataclass
from Configuration.Params import Constant

# ====================================================
# classes
# ====================================================
class Domain(Constant):
    """
        This class is to define the domain setting,
        including domain size, spacing, wavenumber sampling,
        and ensemble size

        Tmax , dt : simulation duration and its interval
        Xmax , dx : zonal domain size and its interval
        Zmax , dz : vertical domain size and its interval
        nondim_k  : list for assigned non-dimensional wavenumber
        k         : list for assigned non-dimensional angular wavenumber
        Ens       : number of ensemble members
    """

    Tmax: float = 40.0
    Zmax: float = 14_000.0

    dt: float = 0.25
    dx: float = 100_000.0
    dz: float = 200.0

    nondim_k: list[float] = [0.1] + [i / 2 for i in range(1, 61)]

    Ens: int = 200

    @property
    def k(self) -> list[float]:
        # return list of non-dimensional angular wavenumber
        return [2*np.pi*4320_000/(self.Xmax / nk) for nk in self.nondim_k]

    @property
    def Xmax(self) -> float:
        return 2*np.pi*self.a

    @property
    def X(self) -> np.ndarray:
        """return the zonal axis"""
        half_X = self.Xmax / 2
        return np.linspace(-half_X, half_X, int(self.Xmax / self.dx) + 1)
       
    @property
    def Z(self) -> np.ndarray:
        """return the vertical axis"""
        return np.linspace(0, self.Zmax, int(self.Zmax / self.dz) + 1)

    @property
    def T(self) -> np.ndarray:
        """return the time axis"""
        return np.linspace(0, self.Tmax, int(self.Tmax / self.dt) + 1)