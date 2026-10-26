"""
    Eigenmode analysis of the one-way and two-way operators
"""

# ====================================================
# Import package
# ====================================================

from dataclasses import dataclass

import numpy as np

from Configuration.ModelConfig import ModelConfig
from ModelCalculation.Operators import Operator

# ====================================================
# Analysis result and calculation
# ====================================================

@dataclass
class EigenResult:
    """Eigenmode arrays have shape (mode, wavenumber); mode indices are zero-based."""

    model_type: str
    rad_scaling: float
    k: np.ndarray
    nondim_k: np.ndarray
    eigenvalues: np.ndarray
    growth_rates: np.ndarray
    phase_speeds: np.ndarray
    dominant_mode: np.ndarray

    @property
    def dominant_growth(self) -> np.ndarray:
        return self.growth_rates[self.dominant_mode, np.arange(self.k.size)]

    @property
    def dominant_speed(self) -> np.ndarray:
        return self.phase_speeds[self.dominant_mode, np.arange(self.k.size)]


def analyze(config: ModelConfig) -> EigenResult:
    """Analyze the configured grid without writing files or running a simulation."""

    config.validate()
    model_type, rad_scaling, domain = config.model_type, config.rad_scaling, config.domain
    operator = Operator(model_type)
    k: np.ndarray = np.asarray(domain.k)
    values = []
    for kn in k:
        eigenvalues = np.linalg.eigvals(operator.operator(float(kn), rad_scaling))
        values.append(eigenvalues[np.argsort(-eigenvalues.real, kind="stable")])
    eigenvalues = np.stack(values, axis=1)
    growth_rates = eigenvalues.real
    phase_speeds = np.full(growth_rates.shape, np.nan)
    np.divide(-eigenvalues.imag * 50.0, k[None, :], out=phase_speeds, where=k[None, :] != 0)

    # use the positive-phase-speed member when conjugate modes tie in growth
    selected = np.empty(k.size, dtype=int)
    for index in range(k.size):
        tied = np.flatnonzero(np.isclose(growth_rates[:, index], growth_rates[0, index], rtol=1e-10, atol=1e-12))
        speeds = phase_speeds[tied, index]
        selected[index] = tied[np.nanargmax(speeds)] if np.any(np.isfinite(speeds)) else tied[0]

    return EigenResult(model_type, rad_scaling, k, np.asarray(domain.nondim_k).copy(),
                       eigenvalues, growth_rates, phase_speeds, selected)
