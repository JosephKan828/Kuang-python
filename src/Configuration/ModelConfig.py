"""
    Configuration shared by the model API and command-line scripts
"""

# ====================================================
# Import package
# ====================================================

from copy import deepcopy
from dataclasses import dataclass, field, fields

import numpy as np

from Configuration.Domain import Domain
from Configuration.Params import Constant, Dynamics, Moisture, Convection, Radiation

# ====================================================
# Configuration
# ====================================================

@dataclass
class ModelConfig:
    """Settings for one run; each instance receives its own default Domain.

    Physical parameters use the current Params classes. No source files or
    global random state are changed by this configuration.
    """

    model_type: str = "two-way"
    rad_scaling: float = 1.0
    domain: Domain = field(default_factory=Domain)
    seed: int | None = None

    def __post_init__(self) -> None:
        self.domain = deepcopy(self.domain)
        self.domain.nondim_k = list(self.domain.nondim_k)
        self.validate()

    def validate(self) -> None:
        """
            This function is to validate the input of model configuration, including:
            - model_type must be 'one-way' or 'two-way'
            - rad_scaling must be finite
            - seed must be an integer in [0, 2**32) if specified
            - domain must satisfy the constraints checked by validate_domain
        """

        if self.model_type not in ("one-way", "two-way"):
            raise ValueError("model_type must be 'one-way' or 'two-way'.")

        if not np.isfinite(self.rad_scaling):
            raise ValueError("rad_scaling must be finite.")

        if self.seed is not None and (
            isinstance(self.seed, (bool, np.bool_))
            or not isinstance(self.seed, (int, np.integer))
            or not 0 <= self.seed < 2**32
        ):
            raise ValueError("seed must be an integer in [0, 2**32).")

        validate_domain(self.domain)


def validate_domain(domain: Domain) -> None:
    """Require a grid consistent with the current Domain.T and TimeStepper"""

    # Check that basic domain parameters are finite and positive.
    for name in ("dt", "Tmax", "dx", "dz", "Zmax", "Xmax"):
        value = getattr(domain, name)
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive.")
    
    if not isinstance(domain.Ens, (int, np.integer)) or isinstance(domain.Ens, bool) or domain.Ens < 1:
        raise ValueError("Ens must be a positive integer.")
    ratio = domain.Tmax / domain.dt
    if ratio < 1 or not np.isclose(ratio, round(ratio), rtol=1e-12, atol=0):
        raise ValueError("Tmax must be a positive integer multiple of dt for the current Domain.T.")
    
    # Domain.T uses int(Tmax / dt); check its actual spacing as well.
    if not np.allclose(np.diff(domain.T), domain.dt, rtol=1e-12, atol=0):
        raise ValueError("Domain.T spacing differs from dt; choose exactly compatible Tmax and dt.")
    k = np.asarray(domain.k)
    
    if k.ndim != 1 or k.size == 0 or not np.all(np.isfinite(k)):
        raise ValueError("The wavenumber grid must be nonempty, one-dimensional, and finite.")
    if domain.dz > domain.Zmax or domain.dx > domain.Xmax:
        raise ValueError("dx and dz must not exceed their domain sizes.")


def configuration(operator, domain: Domain, seed: int | None) -> dict:
    """Snapshot the active Python values, rather than unused TOML settings"""

    groups = {}
    for group in (Dynamics, Moisture, Convection, Radiation):
        groups[group.__name__] = {field.name: float(getattr(operator, field.name)) for field in fields(group)}
    groups["Constant"] = {field.name: float(getattr(domain, field.name)) for field in fields(Constant)}
    return {
        "model_type": operator.type, "rad_scaling": float(operator.rad_scaling), "seed": None if seed is None else int(seed),
        "parameters": groups,
        "domain": {name: float(getattr(domain, name)) for name in ("Tmax", "dt", "Xmax", "Zmax", "dx", "dz")},
        "members": int(domain.Ens), "nondim_k": list(map(float, domain.nondim_k)),
        "reference_length_m": 4_320_000.0, "reference_time_s": 86_400.0,
    }
