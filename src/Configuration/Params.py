"""
    This script is to assign model parameter to
    dataclass so that it can be used with the simulation
    scripts.
"""

# ====================================================
# Import package
# ====================================================

import numpy as np

from dataclasses import dataclass

# ====================================================
# dataclass for different parameters
# ====================================================

@dataclass
class Constant:
    """
        Model constants used throughout the simulation.

        Parameters:
        ---------------------
        - g : gravitational acceleration
        - R : gas constant for dry air
        - cp: specific heat at constant pressure
        - Lv: latent heat of vaporization
        - a : radius of the Earth
        - GammaT: lapse rate of temperature with height
        - T0: reference temperature at the surface
        - p0: reference pressure at the surface
    """
    g     : float = 9.81
    R     : float = 287.0
    cp    : float = 1004.0
    Lv    : float = 2.5e6
    a     : float = 6.37e6
    GammaT: float = 0.0065
    T0    : float = 300.0
    p0    : float = 1e5

@dataclass
class Dynamics:
    """
        Dynamics parameters.

        Parameters:
        ---------------------
        - c1, c2 : Phase speed
        - epsilon: dynamical damping coefficient.
    """

    c1      : float = 1.0
    c2      : float = 0.5

    epsilon : float = 0.1

@dataclass
class Moisture:
    """
        Parameters associated with the moisture budget

        Stratification factors
        ---------------------
        - a1, a2: regression coefficient of vertical moisture advection

        Interaction factors
        ---------------------
        - d1, d2: drying effect associated with convection
        - m1, m2: combining effect of convective drying and vertical moisture supply

        """

    a1: float = 1.4
    a2: float = 0.0

    d1: float = 1.1
    d2: float = -1.0

    m1: float = 0.3
    m2: float = 1.0

@dataclass
class Convection:
    """
        Parameters associated with convection parameterization

        CQE
        ---------------------
        - F     : Clausius-Clapeyron coefficient
        - b1, b2: Consumption rate of PBL MSE by convection 
        - f     : Weighting of 1st and 2nd mode in CQE
        - tau   : relaxation time scale

        Heating Configuration
        ---------------------
        - r0: ratio between upper- and lower-level heating
        - rq: relations between moisture and upper-level heating
        - g0: converted r0 in one-way system
        - gq: converted rq in one-way system 
    """

    F  : float = 4.0
    b1 : float = 1.0
    b2 : float = 2.0
    f  : float = 0.5
    tau: float = 1/12

    r0 : float = 1.0
    rq : float = 0.7

    def g0(self):
        return (1-self.r0)/(1+self.r0)

    def gq(self):
        return (2*self.rq)/(1+self.r0)

    def A(self):
        return 1 - 2*self.f + (self.b2 - self.b1) / self.F

    def B(self):
        return 1 + (self.b1 + self.b2)/self.F - self.A()*self.r0

@dataclass
class Radiation:
    """
        Parameter used for radiation parameterization
        Only circulation-based regression coefficient is preserved
        in current model

        scaling: The magnitude of radiative effect

        LW parameter
        ---------------------
        alpha11_LW, alpha21_LW: LW heating correlated with w1
        alpha12_LW, alpha22_LW: LW heating correlated with w2

        SW parameter
        ---------------------
        alpha11_SW, alpha21_SW: SW heating correlated with w1
        alpha12_SW, alpha22_SW: SW heating correlated with w2 
    """

    alpha11_LW: float = 0.0335
    alpha21_LW: float = 0.0455
    alpha12_LW: float = -0.0414
    alpha22_LW: float = -0.0291

    alpha11_SW: float = 0.0130
    alpha21_SW: float = -0.0371
    alpha12_SW: float = -0.00508
    alpha22_SW: float = 0.0378
