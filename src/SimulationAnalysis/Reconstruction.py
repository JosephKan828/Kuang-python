"""
    Reconstruct daily spectral states using Fourier and vertical Galerkin bases
"""

# ====================================================
# Import package
# ====================================================

from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
import shutil
from tempfile import mkdtemp

import h5py
import numpy as np

from Output.Output import prepare_output

# ====================================================
# Configuration and output layout
# ====================================================

def validate_reconstruction_options(x_block_size: int, workers: int) -> None:
    for name, value in (("x_block_size", x_block_size), ("workers", workers)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")


def _read_configuration(input_path, x_block_size):
    with h5py.File(input_path / "Metadata.h5", "r") as metadata, \
         h5py.File(input_path / "State.h5", "r") as state:
        if not metadata.attrs.get("complete", False) or not state.attrs.get("complete", False):
            raise ValueError("Simulation files are incomplete.")
        dt = float(metadata.attrs.get("dt"))
        if not np.isfinite(dt) or not 0 < dt <= 1:
            raise ValueError("Daily reconstruction requires dt in (0, 1] days.")
        samples = int(round(1 / dt))
        if not np.isclose(samples * dt, 1, rtol=1e-12, atol=0):
            raise ValueError("dt must divide one day into a whole number of samples.")
        x, z, k, times, density = (metadata[name][:] for name in ("x", "z", "k", "time", "rho"))
        if not np.all(np.isfinite(density)) or np.any(density <= 0) or z[-1] <= 0:
            raise ValueError("Reconstruction requires positive finite density and vertical extent.")
        if k.size == 0:
            raise ValueError("Reconstruction requires at least one wavenumber.")
        members = int(metadata["nEnsemble"][()])
        ndays = times.size // samples
        variables = [name for name in ("w1", "w2", "T1", "T2", "J1", "J2") if name in state]
        if not variables:
            raise ValueError("No modal variables are available for reconstruction.")
        for name in variables:
            if state[name].shape != (members, times.size, k.size):
                raise ValueError(f"Unexpected dimensions for {name}.")

        g1 = (np.pi / 2) * np.sin(np.pi * z / z[-1])
        g2 = (np.pi / 2) * np.sin(2 * np.pi * z / z[-1])
        stratification = metadata.attrs["g"] / metadata.attrs["cp"] - metadata.attrs["GammaT"]
        bases = {"w1": g1, "w2": g2, "T1": g1 * stratification,
                 "T2": g2 * stratification, "J1": g1 * stratification, "J2": g2 * stratification}
        return dict(x=x, z=z, k=k, members=members, ndays=ndays, samples=samples,
                    time=times[:ndays * samples].reshape(ndays, samples).mean(axis=1),
                    variables=variables, vertical={name: bases[name] / density for name in variables},
                    reference_length=metadata.attrs["reference_length_m"],
                    model_type=metadata.attrs["model_type"], rad_scaling=metadata.attrs["rad_scaling"],
                    block=min(int(x_block_size), x.size))


def _write_header(file, config, first, last, galerkin=False):
    file.attrs["complete"] = False
    for name in ("x", "time"):
        file.create_dataset(name, data=config[name])
    file.create_dataset("k", data=config["k"][first:last])
    if galerkin:
        file.create_dataset("z", data=config["z"])
    file.attrs["averaging"] = "consecutive sample means starting at t=0; incomplete tail omitted"
    file.attrs["model_type"] = config["model_type"]
    file.attrs["rad_scaling"] = config["rad_scaling"]


def _shape(config, count, galerkin=False):
    shape = (config["members"], config["ndays"], config["x"].size)
    return shape + ((config["z"].size,) if galerkin else ()) + (count,)


def _axes(galerkin=False):
    return "ensemble, day, x, z, wavenumber" if galerkin else "ensemble, day, x, wavenumber"


# ====================================================
# Independent worker calculation and compression
# ====================================================

def _daily_means(dataset, index, ndays, samples):
    """Read one modal history once; reuse its daily means for every x block."""

    history = dataset[:, :ndays * samples, index]
    return history.reshape(history.shape[0], ndays, samples).mean(axis=2)


def _reconstruct_part(job):
    input_path, directory, number, first, last, config = job
    with h5py.File(input_path / "State.h5", "r") as state, \
         h5py.File(directory / f"Fourier-part-{number:04d}.h5", "w") as fourier, \
         h5py.File(directory / f"Galerkin-part-{number:04d}.h5", "w") as galerkin:
        _write_header(fourier, config, first, last)
        _write_header(galerkin, config, first, last, galerkin=True)
        horizontal = np.exp(1j * config["x"][:, None] * config["k"][None, first:last]
                            / config["reference_length"])
        for name in config["variables"]:
            for file, is_galerkin in ((fourier, False), (galerkin, True)):
                chunks = (min(config["members"], 16), 1, config["block"])
                chunks += ((config["z"].size,) if is_galerkin else ()) + (1,)
                dataset = file.create_dataset(name, _shape(config, last - first, is_galerkin),
                                              dtype=float if is_galerkin else complex,
                                              chunks=chunks, compression="gzip", compression_opts=1)
                dataset.attrs["axes"] = _axes(is_galerkin)
            fourier_field, galerkin_field = fourier[name], galerkin[name]
            vertical = config["vertical"][name]
            for local_index, index in enumerate(range(first, last)):
                means = _daily_means(state[name], index, config["ndays"], config["samples"])
                for day in range(config["ndays"]):
                    for start in range(0, config["x"].size, config["block"]):
                        stop = min(start + config["block"], config["x"].size)
                        # Outer products preserve ensemble (e), horizontal (x),
                        # and vertical (z) axes; no modes are summed.
                        values = np.einsum("e,x->ex", means[:, day],
                                           horizontal[start:stop, local_index],
                                           optimize=False, order="C")
                        fourier_field[:, day, start:stop, local_index] = values
                        galerkin_field[:, day, start:stop, :, local_index] = np.einsum(
                            "ex,z->exz", values.real, vertical, optimize=False, order="C")
        fourier.attrs["complete"] = True
        galerkin.attrs["complete"] = True
    return number


# ====================================================
# Combined virtual datasets and run lifecycle
# ====================================================

def _combine_parts(directory, config, ranges):
    """Create small combined files; relative sources remain valid after moving a run."""

    for prefix, is_galerkin in (("Fourier", False), ("Galerkin", True)):
        for number, (first, last) in enumerate(ranges):
            with h5py.File(directory / f"{prefix}-part-{number:04d}.h5", "r") as part:
                if not part.attrs.get("complete", False):
                    raise ValueError("Reconstruction part is incomplete.")
                for name in config["variables"]:
                    if part[name].shape != _shape(config, last - first, is_galerkin):
                        raise ValueError(f"Unexpected reconstruction part dimensions for {name}.")
        with h5py.File(directory / f"{prefix}State.h5", "w", libver="latest") as file:
            _write_header(file, config, 0, config["k"].size, is_galerkin)
            for name in config["variables"]:
                shape = _shape(config, config["k"].size, is_galerkin)
                dtype = float if is_galerkin else complex
                if not ranges:
                    dataset = file.create_dataset(name, shape, dtype=dtype)
                else:
                    layout = h5py.VirtualLayout(shape=shape, dtype=dtype)
                    for number, (first, last) in enumerate(ranges):
                        # These paths are relative to the final combined files in the run root.
                        source_path = Path("ReconstructionParts") / directory.name / f"{prefix}-part-{number:04d}.h5"
                        source = h5py.VirtualSource(str(source_path), name,
                                                   shape=_shape(config, last - first, is_galerkin))
                        layout[..., first:last] = source
                    fillvalue = np.nan if is_galerkin else complex(np.nan, np.nan)
                    dataset = file.create_virtual_dataset(name, layout, fillvalue=fillvalue)
                dataset.attrs["axes"] = _axes(is_galerkin)
            file.attrs["complete"] = True


def _previous_sources(input_path):
    """Collect only old virtual sources inside our reserved parts directory."""

    sources = set()
    root = (input_path / "ReconstructionParts").resolve()
    for filename in ("FourierState.h5", "GalerkinState.h5"):
        path = input_path / filename
        if not path.exists():
            continue
        with h5py.File(path, "r") as file:
            for dataset in file.values():
                if isinstance(dataset, h5py.Dataset) and dataset.is_virtual:
                    for source in dataset.virtual_sources():
                        path = (input_path / source.file_name).resolve()
                        if path.parent.parent == root and path.parent.name.startswith("run-"):
                            sources.add(path)
    return sources


def reconstruct(input_path: Path, x_block_size: int = 64, overwrite: bool = False,
                verbose: bool = False, *, workers: int = 4) -> None:
    """Reconstruct independent wavenumber groups, including parallel compression.

    Daily means use consecutive groups of 1/dt samples starting at zero; discard
    incomplete tails. Keep wavenumbers separate. Combined HDF5 files are virtual
    datasets backed by ReconstructionParts/, which must stay with the run.
    Use workers=1 to avoid a process pool while retaining the same output layout.
    """

    input_path = Path(input_path).resolve()
    validate_reconstruction_options(x_block_size, workers)
    config = _read_configuration(input_path, x_block_size)
    prepare_output(input_path, ("FourierState.h5", "GalerkinState.h5"), overwrite)
    previous = _previous_sources(input_path) if overwrite else set()
    count = min(int(workers), config["k"].size) if config["ndays"] else 0
    edges = np.linspace(0, config["k"].size, count + 1, dtype=int) if count else []
    ranges = [(int(first), int(last)) for first, last in zip(edges[:-1], edges[1:])]
    logical_bytes = (len(config["variables"]) * config["members"] * config["ndays"]
                     * config["x"].size * config["k"].size * (16 + 8 * config["z"].size))
    if verbose:
        print(f"Reconstruction logical output size before compression: {logical_bytes / 1024**3:.3f} GiB", flush=True)
        print(f"Reconstruction workers: {count}; each calculates, compresses, and writes its own wavenumbers", flush=True)

    parts_root = input_path / "ReconstructionParts"
    parts_root.mkdir(exist_ok=True)
    directory = Path(mkdtemp(prefix="run-", dir=parts_root))
    published = False
    try:
        jobs = [(input_path, directory, number, first, last, config)
                for number, (first, last) in enumerate(ranges)]
        if count == 1:
            _reconstruct_part(jobs[0])
        elif count > 1:
            with ProcessPoolExecutor(max_workers=count, mp_context=get_context("spawn")) as pool:
                futures = [pool.submit(_reconstruct_part, job) for job in jobs]
                for future in as_completed(futures):
                    number = future.result()
                    if verbose:
                        print(f"Reconstructed wavenumber group {number + 1}/{count}", flush=True)
        _combine_parts(directory, config, ranges)
        for filename in ("FourierState.h5", "GalerkinState.h5"):
            (directory / filename).replace(input_path / filename)
            published = True
    finally:
        # Keep any already-published sources even if publishing the second file fails.
        if not published:
            shutil.rmtree(directory)

    # Only discard previously referenced parts after both replacements succeeded.
    for path in previous:
        path.unlink(missing_ok=True)
    for parent in {path.parent for path in previous} | {directory}:
        if parent.is_dir() and not any(parent.iterdir()):
            parent.rmdir()
    if verbose:
        print("Combined reconstruction files saved; keep ReconstructionParts/ with them.", flush=True)
