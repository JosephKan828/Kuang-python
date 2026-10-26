"""
    Calculate simulation histories, convective heating, and run metadata
"""

# ====================================================
# Import package
# ====================================================

from copy import deepcopy
from dataclasses import dataclass, field
from collections.abc import Iterator

import numpy as np

from ModelCalculation.MeanState import MeanState
from Configuration.ModelConfig import ModelConfig, configuration
from ModelCalculation.TimeStepper import TimeStepper

# ====================================================
# Simulation results
# ====================================================

@dataclass
class TimeSeries:
    """History: (time, state) or (time, state, ensemble).

    Heating arrays have shape (time,) or (time, ensemble), respectively.
    """

    k: float
    times: np.ndarray
    states: np.ndarray
    state_names: tuple[str, ...]
    heating: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class SimulationMetadata:
    """Precomputed configuration, coordinates, and background profiles."""

    configuration: dict
    state_names: tuple[str, ...]
    coordinates: dict[str, np.ndarray]
    background: dict

# ====================================================
# Simulation calculation
# ====================================================

class SimulationRun:
    """Compute a configured run without opening or writing output files."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = deepcopy(config)
        self.config.validate()
        self.stepper = TimeStepper(self.config.model_type, self.config.rad_scaling,
                                   domain=self.config.domain)
        self.state_names = (("w1", "w2", "T1", "T2", "q", "L")
                            if self.config.model_type == "two-way" else ("T1", "T2", "q", "L"))

    def simulate_wavenumber(self, k: float, initial_state: np.ndarray | None = None,
                            *, rng=None) -> TimeSeries:
        """Compute one history and its convective heating diagnostics."""

        if not np.isfinite(k):
            raise ValueError("k must be finite.")
        if initial_state is None:
            rng = np.random.RandomState(self.config.seed) if rng is None else rng
            times, states = self.stepper.simulate_ensemble(k, rng=rng)
        else:
            times, states = self.stepper.simulate(k, initial_state)
        if not np.all(np.isfinite(states)):
            raise FloatingPointError(f"Nonfinite simulation state at k={k}.")

        components = {name: states[:, index, ...] for index, name in enumerate(self.state_names)}
        if self.config.model_type == "two-way":
            upper = self.stepper.r0 * components["L"] + self.stepper.rq * (components["q"] - 1.5 * components["T1"])
            heating = {"J1": components["L"] + upper, "J2": components["L"] - upper}
        else:
            heating = {
                "J1": components["L"],
                "J2": self.stepper.g0() * components["L"] + self.stepper.gq() * (components["q"] - 1.5 * components["T1"]),
            }
        return TimeSeries(k, times, states, self.state_names, heating)

    def __iter__(self) -> Iterator[TimeSeries]:
        """Yield one wavenumber at a time, using one random stream per iteration."""

        rng = np.random.RandomState(self.config.seed)
        for k in self.config.domain.k:
            yield self.simulate_wavenumber(float(k), rng=rng)

    def metadata(self) -> SimulationMetadata:
        """Calculate the background and collect the settings used by this run."""

        domain = self.config.domain
        background = MeanState()
        background.Zmax, background.dz = domain.Zmax, domain.dz
        coordinates = {
            "time": domain.T, "k": np.asarray(domain.k),
            "nondim_k": np.asarray(domain.nondim_k), "x": domain.X, "z": domain.Z,
        }
        profiles = {
            "Tmean": background.Tmean(), "pmean": background.pmean(), "rho": background.rhomean(),
            "GammaT": background.GammaT, "g": background.g, "cp": background.cp,
        }
        return SimulationMetadata(configuration(self.stepper, domain, self.config.seed),
                                  self.state_names, coordinates, profiles)
