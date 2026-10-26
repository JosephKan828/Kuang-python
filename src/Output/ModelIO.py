"""
    Save precomputed simulation results and eigen-analysis arrays
"""

from __future__ import annotations

# ====================================================
# Import package
# ====================================================

from contextlib import ExitStack
import json
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np

from Output.Output import prepare_output

if TYPE_CHECKING:
    from SimulationAnalysis.SimulationRun import SimulationMetadata, TimeSeries

# ====================================================
# Simulation output
# ====================================================

class SimulationWriter:
    """Write supplied histories without initializing or advancing the model.

    Use as a context manager and supply one ensemble history per configured
    wavenumber, in grid order. Files are marked complete only when all histories
    have been written and the context exits without an exception.
    """

    def __init__(self, metadata: SimulationMetadata, output_path: str | Path,
                 overwrite: bool = False) -> None:
        self.metadata = metadata
        self.output_path = Path(output_path)
        self.overwrite = overwrite
        self.next_index = 0

    def __enter__(self) -> SimulationWriter:
        snapshot = self.metadata.configuration
        self.times = self.metadata.coordinates["time"]
        self.k = self.metadata.coordinates["k"]
        self.names = self.metadata.state_names
        self.members = snapshot["members"]
        serialized = json.dumps(snapshot)
        prepare_output(self.output_path, ("Metadata.h5", "State.h5", "Config.json"), self.overwrite)
        (self.output_path / "Config.json").write_text(json.dumps(snapshot, indent=2) + "\n")

        self.files = ExitStack()
        try:
            self.metadata_file = self.files.enter_context(h5py.File(self.output_path / "Metadata.h5", "w"))
            self.state_file = self.files.enter_context(h5py.File(self.output_path / "State.h5", "w"))
            for file in (self.metadata_file, self.state_file):
                file.attrs["complete"] = False
                file.attrs["model_type"] = snapshot["model_type"]
                file.attrs["rad_scaling"] = snapshot["rad_scaling"]
            self.metadata_file.attrs["configuration"] = serialized
            self.metadata_file.attrs["dt"] = snapshot["domain"]["dt"]
            self.metadata_file.attrs["reference_length_m"] = snapshot["reference_length_m"]
            for name in ("GammaT", "g", "cp"):
                self.metadata_file.attrs[name] = self.metadata.background[name]
            for name, value in (("RadqScale", 0.0), ("RadtScale", 0.0),
                                ("RadwScale", snapshot["rad_scaling"]), ("nEnsemble", self.members)):
                self.metadata_file.create_dataset(name, data=value)
            for name, values in self.metadata.coordinates.items():
                self.metadata_file.create_dataset(name, data=values)
            for name in ("Tmean", "pmean", "rho"):
                self.metadata_file.create_dataset(name, data=self.metadata.background[name])
            self.initial = self.metadata_file.create_dataset(
                "InitState", (len(self.names), self.members, self.k.size), dtype=complex,
            )
            self.initial.attrs["axes"] = "state, ensemble, wavenumber"
            self.initial.attrs["state_names"] = ", ".join(self.names)
            for name in (*self.names, "J1", "J2"):
                dataset = self.state_file.create_dataset(
                    name, (self.members, self.times.size, self.k.size), dtype=complex,
                    chunks=(min(self.members, 32), min(self.times.size, 32), 1),
                )
                dataset.attrs["axes"] = "ensemble, time, wavenumber"
        except BaseException:
            self.files.close()
            raise
        return self

    def write_wavenumber(self, result: TimeSeries) -> None:
        """Validate and save an already computed ensemble history and heating."""

        index = self.next_index
        if index >= self.k.size:
            raise ValueError("All configured wavenumbers have already been written.")
        if result.k != self.k[index]:
            raise ValueError("Histories must be supplied in the configured wavenumber order.")
        if result.state_names != self.names or not np.array_equal(result.times, self.times):
            raise ValueError("History state names or times do not match the output metadata.")
        if result.states.shape != (self.times.size, len(self.names), self.members):
            raise ValueError("History must have shape (time, state, ensemble) matching the metadata.")
        if not np.all(np.isfinite(result.states)):
            raise ValueError("History contains nonfinite states.")
        for name in ("J1", "J2"):
            values = result.heating.get(name)
            if values is None or values.shape != (self.times.size, self.members):
                raise ValueError(f"Supply precomputed {name} with shape (time, ensemble).")
            if not np.all(np.isfinite(values)):
                raise ValueError(f"History contains nonfinite {name} values.")

        self.initial[:, :, index] = result.states[0]
        for position, name in enumerate(self.names):
            self.state_file[name][:, :, index] = result.states[:, position, :].T
        for name in ("J1", "J2"):
            self.state_file[name][:, :, index] = result.heating[name].T
        self.next_index += 1

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        try:
            if exc_type is None:
                if self.next_index != self.k.size:
                    raise ValueError("Simulation output is incomplete: not all wavenumbers were supplied.")
                self.state_file.attrs["complete"] = True
                self.metadata_file.attrs["complete"] = True
        finally:
            self.files.close()


def save_eigen_analysis(result, output_path: Path, overwrite: bool = False,
                        verbose: bool = False) -> Path:
    """Write an EigenResult using the existing EigenAnalysis.h5 schema."""

    model_type, rad_scaling = result.model_type, result.rad_scaling
    k, eigenvalues = result.k, result.eigenvalues
    growth_rates, phase_speeds = result.growth_rates, result.phase_speeds
    selected = result.dominant_mode
    output_path = prepare_output(output_path, ("EigenAnalysis.h5",), overwrite)
    with h5py.File(output_path / "EigenAnalysis.h5", "w") as file:
        file.attrs["model_type"] = model_type
        file.attrs["rad_scaling"] = rad_scaling
        file.attrs["reference_length_m"] = 4_320_000.0
        file.attrs["reference_time_s"] = 86_400.0
        file.create_dataset("k", data=k)
        file.create_dataset("nondim_k", data=result.nondim_k)
        for name, data, units in (("Eigenvalues", eigenvalues, "day^-1"),
                                  ("GrowthRates", growth_rates, "day^-1"),
                                  ("PhaseSpeeds", phase_speeds, "m s^-1")):
            dataset = file.create_dataset(name, data=data)
            dataset.attrs["axes"] = "mode, wavenumber"
            dataset.attrs["units"] = units
        file.create_dataset("DominantMode", data=selected).attrs["index_base"] = 0
    if verbose:
        print(f"Eigen-analysis saved to {output_path / 'EigenAnalysis.h5'}", flush=True)
    return output_path / "EigenAnalysis.h5"

