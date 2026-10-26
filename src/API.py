"""
    Public interface for simulation, eigen-analysis, and reconstruction
"""

# ====================================================
# Import package
# ====================================================

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from Configuration.Domain import Domain
from SimulationAnalysis.EigenModes import EigenResult, analyze
from ModelCalculation.InitCondition import InitialCondition
from ModelCalculation.MeanState import MeanState
from Configuration.ModelConfig import ModelConfig
from Output.ModelIO import SimulationWriter, save_eigen_analysis
from ModelCalculation.Operators import Operator
from Output.Output import prepare_output
from SimulationAnalysis.Reconstruction import reconstruct, validate_reconstruction_options
from SimulationAnalysis.SimulationRun import SimulationRun, TimeSeries
from ModelCalculation.TimeStepper import TimeStepper

__all__ = [
    "Domain", "Operator", "InitialCondition", "MeanState", "TimeStepper",
    "ModelConfig", "EigenResult", "TimeSeries", "SimulationFiles",
    "ReconstructionFiles", "RunResult", "simulate_wavenumber", "iter_simulation", "run_simulation",
    "analyze_eigenmodes", "reconstruct_fields", "run_model",
]

# ====================================================
# Result containers
# ====================================================

@dataclass
class SimulationFiles:
    directory: Path
    metadata: Path
    state: Path
    configuration: Path


@dataclass
class ReconstructionFiles:
    """Combined HDF5 paths; their ReconstructionParts/ sources must stay nearby."""

    fourier: Path
    galerkin: Path


@dataclass
class RunResult:
    simulation: SimulationFiles
    eigen_analysis: EigenResult
    eigen_file: Path
    reconstruction: ReconstructionFiles | None

# ====================================================
# Public functions
# ====================================================

def simulate_wavenumber(config: ModelConfig, k: float,
                        initial_state: np.ndarray | None = None) -> TimeSeries:
    """Simulate one angular wavenumber in memory without creating files.

    With no initial_state, generate config.domain.Ens noisy members. Otherwise
    accept a state vector or an ensemble with members in columns. Each call
    starts its own seeded random stream; run_simulation uses one stream across
    the entire configured wavenumber grid.
    """

    return SimulationRun(config).simulate_wavenumber(k, initial_state)


def iter_simulation(config: ModelConfig):
    """Yield histories and heating across the full grid without writing files.

    The configuration is copied when this iterator is created. One random stream
    is used across the grid, matching run_simulation's initial ensembles.
    """

    return iter(SimulationRun(config))


def run_simulation(config: ModelConfig, output_path: str | Path, *,
                   overwrite: bool = False, verbose: bool = False) -> SimulationFiles:
    """Stream the full ensemble to HDF5, retaining one wavenumber in memory."""

    simulation = SimulationRun(config)
    metadata = simulation.metadata()
    directory = Path(output_path)
    with SimulationWriter(metadata, directory, overwrite) as writer:
        for index, result in enumerate(simulation):
            writer.write_wavenumber(result)
            count = metadata.coordinates["k"].size
            if verbose and (index == 0 or (index + 1) % 10 == 0 or index == count - 1):
                print(f"Simulation: {index + 1}/{count} wavenumbers", flush=True)
    if verbose:
        print(f"Simulation saved to {directory}", flush=True)
    return SimulationFiles(directory, directory / "Metadata.h5", directory / "State.h5",
                           directory / "Config.json")


def analyze_eigenmodes(config: ModelConfig, output_path: str | Path | None = None, *,
                       overwrite: bool = False, verbose: bool = False) -> EigenResult:
    """Return eigenmode arrays; optionally also write EigenAnalysis.h5."""

    result = analyze(config)
    if output_path is not None:
        save_eigen_analysis(result, Path(output_path), overwrite, verbose)
    return result


def reconstruct_fields(input_path: str | Path, *, x_block_size: int = 64,
                       overwrite: bool = False, verbose: bool = False,
                       workers: int = 4) -> ReconstructionFiles:
    """Reconstruct with parallel compression and bounded spatial blocks.

    Returned files are virtual datasets backed by ReconstructionParts/ in the
    same directory. Keep that directory when moving or copying the outputs.
    """

    directory = Path(input_path)
    reconstruct(directory, x_block_size, overwrite, verbose, workers=workers)
    return ReconstructionFiles(directory / "FourierState.h5", directory / "GalerkinState.h5")


def run_model(config: ModelConfig, output_path: str | Path, *,
              reconstruct: bool = False, x_block_size: int = 64,
              overwrite: bool = False, verbose: bool = False,
              reconstruction_workers: int = 4) -> RunResult:
    """Run simulation, eigen-analysis, and optional reconstruction together.

    Plotting remains in scripts/Plot_Eigen.py. Existing numerical outputs require
    overwrite=True. No work is performed merely by importing this module.
    """

    config.validate()
    filenames = ("Metadata.h5", "State.h5", "Config.json", "EigenAnalysis.h5")
    if reconstruct:
        dt = config.domain.dt
        if dt > 1 or not np.isclose(round(1 / dt) * dt, 1, rtol=1e-12, atol=0):
            raise ValueError("Reconstruction requires dt to divide one day into whole samples.")
        validate_reconstruction_options(x_block_size, reconstruction_workers)
        filenames += ("FourierState.h5", "GalerkinState.h5")
    directory = prepare_output(Path(output_path), filenames, overwrite)
    simulation = run_simulation(config, directory, overwrite=overwrite, verbose=verbose)
    eigen = analyze_eigenmodes(config, directory, overwrite=overwrite, verbose=verbose)
    fields = None
    if reconstruct:
        fields = reconstruct_fields(directory, x_block_size=x_block_size, overwrite=overwrite,
                                    verbose=verbose, workers=reconstruction_workers)
    return RunResult(simulation, eigen, directory / "EigenAnalysis.h5", fields)
