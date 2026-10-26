"""
    This classes is used to generate mean-state profiles of 
    temperature, pressure, and density
"""

# ====================================================
# Import package
# ====================================================

import numpy as np

from Configuration.Domain import Domain
from Configuration.Params import Constant
# ====================================================
# Classes
# ====================================================

class MeanState(Domain, Constant):

    def __init__(self) -> None:

        super().__init__()

    def Tmean(self) -> np.ndarray:
        """Mean-state temperature profile"""
        return self.T0 - self.GammaT * self.Z
    
    def pmean(self) -> np.ndarray:
        """Mean-state pressure profile corresponding to temperature profile through hydrostatic balance"""
        return self.p0 * (self.Tmean() / self.T0) ** (9.81 / (self.GammaT * 287.0))
    
    def rhomean(self) -> np.ndarray:
        """Mean-state density profile corresponding to pressure and temperature profiles"""
        return self.pmean() / (287.0 * self.Tmean())
