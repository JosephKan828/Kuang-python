# Running the Python model

These scripts adapt the previous version's `script/` workflow to the current
`src` classes. Run them with Python 3.10 or newer and the packages in
`requirements.txt`; the existing `atmos-env` environment has these packages.
Paths to `src` are resolved relative to the scripts, so execution does not depend
on the working directory. Explicit relative output paths use the working directory.

## Python API

`src/API.py` is the shared entry point for notebooks and Python programs. The
command-line scripts call this API; algorithms do not import anything from
`scripts`. Importing the API does not start a run or create files.

From the project root:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path("src").resolve()))
from API import Domain, ModelConfig, analyze_eigenmodes, simulate_wavenumber, run_model

domain = Domain()
domain.Ens = 4
domain.Tmax = 2.0
domain.nondim_k = [1.0, 5.0, 10.0]
config = ModelConfig(model_type="two-way", rad_scaling=1.0, domain=domain, seed=42)

# In-memory calculations; no output files are created.
eigen = analyze_eigenmodes(config)
print(eigen.dominant_growth, eigen.dominant_speed)
history = simulate_wavenumber(config, config.domain.k[0])
print(history.times.shape, history.states.shape)  # (9,), (9, 6, 4)

# Stream all configured wavenumbers to disk and save eigen-analysis.
run = run_model(config, "output/api_run")
print(run.simulation.state, run.eigen_file)
```

In a notebook inside `scripts`, add `../src` to the import path instead. The
API uses angular `domain.k` values for individual wavenumbers. `ModelConfig`
copies the supplied domain; subsequent adjustments should use `config.domain`.
Configurations are validated again when calculations start.

| API function | Return value |
|---|---|
| `simulate_wavenumber(config, k, initial_state=None)` | `TimeSeries` with times, states, and state names; accepts an explicit vector or ensemble |
| `iter_simulation(config)` | Iterator over computed `TimeSeries` objects, including heating, without saving files |
| `run_simulation(config, output_path)` | `SimulationFiles` containing paths to state, metadata, and configuration files |
| `analyze_eigenmodes(config, output_path=None)` | `EigenResult` arrays; optionally also saves `EigenAnalysis.h5` |
| `reconstruct_fields(input_path)` | `ReconstructionFiles` paths, using saved run metadata |
| `run_model(config, output_path, reconstruct=False)` | `RunResult` combining simulation paths, eigenmode arrays, and optional reconstruction paths |

File-writing functions accept `overwrite=True`; workflow functions also accept
`verbose=True` for progress messages. Plotting remains in `Plot_Eigen.py`.
The API also exports the existing `Operator`, `InitialCondition`, `MeanState`,
and `TimeStepper` classes for direct use.

Ensemble initialization computes the eigenmode once per wavenumber and generates
all members' noise together. Seeded API runs use a local NumPy `RandomState` to
preserve the previous seed/draw ordering without changing the caller's global
random state. A single-wavenumber API call starts a new random stream; a full run
uses one stream across wavenumbers. No persistent matrix cache is used.

The simulation and output responsibilities are separate:

- `SimulationRun.py` computes histories, convective heating, and background profiles.
- `ModelIO.SimulationWriter` writes supplied metadata and completed histories;
  it does not initialize, advance, or diagnose the model.
- `API.run_simulation()` iterates over the calculation and passes each history
  to the writer. It retains bounded memory rather than collecting the full grid.

`TimeSeries.heating` contains `J1` and `J2` arrays with shape `(time, ensemble)`
or `(time,)` for a single state. To process a full run without saving it:

```python
from API import iter_simulation

for history in iter_simulation(config):
    print(history.k, history.states[-1])
```

This iterator uses the same random-number sequence as a saved run. Output files
are marked complete only after all configured histories have been saved.

## Run an experiment

The Bash pipeline controls the individual Python stages:

```bash
conda activate atmos-env
bash scripts/RunScript.sh --model two-way --rad-scaling 1.0 --seed 42
```

The current shell defaults run simulation, eigen-analysis, and reconstruction,
with plotting disabled. Reconstruction uses four worker processes by default;
set `--reconstruction-workers 1` for serial execution. Other controls are `--skip-simulation`,
`--skip-analysis`, and `--no-plot`. Run `bash scripts/RunScript.sh --help` for all
options. Use Bash rather than `sh`, because the launcher uses argument arrays.

The shell pipeline forwards the same model arguments and output directory to
simulation and eigen-analysis. It uses the active Python environment, or an
explicit `--python /path/to/python` (also configurable through `$PYTHON`). It
works from any working directory when invoked by its absolute path.

A small complete run:

```bash
bash scripts/RunScript.sh --members 2 --tmax 2 --dt 0.25 \
  --zonal-wavenumbers 1 5 --dx 10000000 --dz 7000 --reconstruct \
  --seed 42 --output-dir output/shell_run
```

Rerun only plotting on saved eigen-analysis:

```bash
python scripts/Plot_Eigen.py output/shell_run/TIMESTAMP
```

Run only reconstruction on a saved simulation:

```bash
bash scripts/RunScript.sh --skip-simulation --skip-analysis --no-plot \
  --reconstruct --run-dir output/shell_run/TIMESTAMP --overwrite
```

The pipeline stops when a stage fails and preserves its exit status. Each stage
has a log, including its exact command, under the resolved run directory's
`logs/TIMESTAMP/` folder.
Required existing inputs and output collisions are checked before stages start.
`--overwrite` applies to the selected numerical outputs; plotting replaces its
generated figures. In the shell launcher, `--output-dir PATH` is the parent:
each invocation creates `PATH/TIMESTAMP/` and all stages write there. Without
that option, the parent is `output/pipeline/` under the project root. Timestamps
use Asia/Taipei and include fractional seconds and the process ID to separate
closely spaced runs. The resolved directory is printed when the pipeline starts.

After all requested stages succeed, both `RunScript.sh` and `RunScript.py` save
a completion record under `RUN_DIRECTORY/logs/INVOCATION_ID/RunRecord.json` and
append the same record to `RunHistory.jsonl` in the run directory's parent.
The console prints both record locations and the completed output directory.
For example, `--output-dir output/shell_run` in the shell launcher produces:

```text
output/shell_run/RunHistory.jsonl
output/shell_run/TIMESTAMP/logs/INVOCATION_ID/RunRecord.json
```

Each record includes the absolute output directory, its folder name as `run_id`,
the invocation ID, start and completion timestamps in Asia/Taipei (`+08:00`),
the working directory, and the exact launch command as both a quoted string and
an argument array. The command records supplied options; `Config.json` contains
the resolved simulation settings. Use the recorded working directory when
interpreting relative command arguments. Identical random seeds alone do not
make results reproducible if the source or configuration changes.

`RunHistory.jsonl` contains one JSON object per successful launcher invocation.
Failed pipelines do not receive a completion entry. Processing an existing run
with `--run-dir` retains its run ID and adds a separate invocation record;
previous entries are preserved. These automatic records apply to the two
launchers; direct API calls and individual stage scripts retain their existing
output behavior.

To process an existing run, use `--run-dir PATH/TIMESTAMP` instead; replace
`TIMESTAMP` in these examples with the actual folder name. This uses the directory
directly, without creating another timestamped child. `--output-dir` and
`--run-dir` cannot be combined. The individual Python scripts still interpret
their output paths as exact directories so the shell pipeline does not create
an extra level for each stage.

When skipping simulation but running eigen-analysis, supply the original model,
radiation, and wavenumber options: eigen-analysis uses the current Python
configuration. Reconstruction and plotting read the saved files. The pipeline
does not automatically restore physical parameters from an older run.

The Python launcher remains available for running the same workflow through the
API in one process.

From the project root:

```bash
conda activate atmos-env
python scripts/RunScript.py --model two-way --rad-scaling 1.0 --seed 42
```

This runs the ensemble, writes eigen-analysis, and saves a two-panel eigenmode
figure. Defaults come from `src/Domain.py` and `src/Params.py`: 200 members, 61
wavenumbers, 40 days, and a 0.25-day interval. A timestamped directory is created
under `output/two-way/Rad(1)/`. Timestamps use Asia/Taipei.

For a named directory and smaller run:

```bash
python scripts/RunScript.py --model two-way --rad-scaling 1.0 \
  --members 4 --tmax 2 --dt 0.25 --zonal-wavenumbers 1 5 10 \
  --seed 42 --output-dir output/small_run
```

`--model one-way` selects the four-component model. `--no-plot` skips figures.
Existing numerical output files are protected; use `--overwrite` to replace them.
A failed stage stops the workflow instead of continuing with incomplete results.

The current `Domain.T` uses `linspace`, while `TimeStepper` uses `domain.dt`.
The launcher therefore requires a positive `Tmax` that is an integer multiple of
`dt`, and checks that the actual time spacing agrees. It does not edit source settings.
Edit Python parameters for model configuration: the current classes do not load
the TOML file. CLI settings affect only that run. `Config.json` records the active
parameter values, domain settings, radiation scaling, and random seed.

## Compare radiation factors

```bash
python scripts/RunScript.py --rad-scaling 0.0 --seed 42 --output-dir output/rad0
python scripts/RunScript.py --rad-scaling 1.0 --seed 42 --output-dir output/rad1
python scripts/Plot_Eigen.py output/rad0 output/rad1 --figure-dir output/comparison
```

The comparison uses solid black for the lower scaling and dashed black for the
higher scaling. Growth rate is on the left and corresponding phase speed on the
right. It chooses the fastest-growing mode at each wavenumber, preferring the
positive-speed member when conjugates tie. Physical branches may switch as k
changes. Negative growth rates remain visible. The figures are PNG and PDF.

## Individual stages

```bash
python scripts/Simulation.py --model two-way --rad-scaling 1.0 --seed 42 --output-dir output/run1
python scripts/EigenAnalysis.py --model two-way --rad-scaling 1.0 --output-dir output/run1
python scripts/Plot_Eigen.py output/run1
python scripts/Reconstruct.py output/run1 --x-block-size 64
```

For separate simulation and eigen-analysis calls, supply the same model,
radiation scaling, and wavenumber settings. `RunScript.py` shares one configuration
automatically. `Reconstruct.py` reads the saved metadata rather than the live
configuration, so later source edits do not change its grids or background.

## Reconstruction

Reconstruction assigns independent wavenumber groups to four processes by
default (or fewer when there are fewer wavenumbers). Each worker opens the saved
state file, reads one variable's history at one wavenumber, computes its daily
means once, and reuses those means across all horizontal blocks. It calculates,
gzip-compresses, and writes its own Fourier and Galerkin part files. Large
reconstructed arrays are never sent to the parent process or collected in memory.
Fourier and Galerkin blocks use `np.einsum` outer products (`e,x->ex` and
`ex,z->exz`), retaining each wavenumber and vertical-mode variable separately.
Contraction-path optimization is disabled for these simple outer products.

The parent creates small HDF5 virtual datasets named `FourierState.h5` and
`GalerkinState.h5`. Reading them gives the same arrays, axes, daily means, and
wavenumber order as before, without copying the part files into a combined array.
The compressed data are stored under `ReconstructionParts/run-.../`.
**Keep that directory with both combined files when copying or moving a run.**
Source paths are relative, so moving the whole run directory is supported.
Virtual datasets require HDF5 1.10 or newer. The top-level datasets are virtual
rather than chunked; gzip compression and chunks belong to the part datasets.
Missing part files yield NaN values rather than valid reconstruction data.

```bash
# Reconstruct an existing simulation using four processes.
python scripts/Reconstruct.py output/run1 --workers 4 --x-block-size 64

# Reconstruct only, through the shell pipeline.
bash scripts/RunScript.sh --skip-simulation --skip-analysis \
  --run-dir output/run1 --reconstruction-workers 4
```

These are alternative commands; add `--overwrite` when replacing existing
reconstruction files. Use `--workers 1` for a serial comparison. The Python
launcher also accepts `--reconstruction-workers`; the API accepts
`reconstruct_fields(path, workers=4)` and
`run_model(config, path, reconstruct=True, reconstruction_workers=4)`.
When calling the parallel API from a Python script, place the call inside
`if __name__ == "__main__":` because workers use the spawn start method.

Each worker holds one modal history, its daily means, and spatial blocks.
Smaller `--x-block-size` values reduce memory per block. Compression and writes
now run in the workers; shared disk bandwidth can still limit speedup, and tiny
runs can be faster with one worker. One-worker execution uses the same part-file
and virtual-dataset layout without launching a process pool.

Worker results are checked before publishing the combined files. A worker failure
removes the new temporary parts and leaves any existing reconstruction unchanged.
With `--overwrite`, new parts are built before replacing the combined files, so
disk space must accommodate old and new output temporarily. Previously referenced
parts are removed after successful replacement. The two combined files are
replaced individually, so avoid reading them during replacement. A run with no
complete daily groups writes empty ordinary datasets without starting workers.

Add `--reconstruct` to the launcher for the full workflow. A small complete example:

```bash
python scripts/RunScript.py --members 2 --tmax 2 --dt 0.25 \
  --zonal-wavenumbers 1 5 --dx 10000000 --dz 7000 --reconstruct \
  --seed 42 --output-dir output/small_reconstruction
```

The implementation follows the Julia formulas: consecutive daily sample means,
`exp(1j*k*x/4320000)` horizontally, `(pi/2)*sin(j*pi*z/Zmax)` vertically,
stratification scaling for temperature/heating, and inverse background density.
It keeps each wavenumber separate and takes the real Fourier component for the
Galerkin output. Incomplete sample groups are discarded; for the default time
axis this omits the isolated day-40 endpoint. This follows the Julia code's
sample grouping, rather than its outdated comment about dropping a complete day.

Daily reconstruction requires `dt` to divide one day into whole samples.
Available fields are `w1`, `w2`, `T1`, `T2`, `J1`, `J2`; the one-way system has no
`w1` or `w2` state and reconstructs the other four only. Python uses `Domain.X`
(`-Xmax/2` to `Xmax/2`), rather than the Julia reconstruction's `-X` to `X` grid.

Reconstruction is optional because the full default fields have a logical size
of several terabytes before compression. Computation writes spatial blocks to
HDF5 instead of allocating the full field. Smaller `--dx`, `--dz`, or more
ensemble members increase output size; `--x-block-size` controls working memory.

## Saved data

| File | Contents / axis order as read by h5py |
|---|---|
| `Config.json` | Active Python parameter and run configuration snapshot |
| `Metadata.h5` | Coordinates, background profiles, scales, ensemble size, and `InitState (state, ensemble, k)` |
| `State.h5` | Each state/heating variable has shape `(ensemble, time, k)` |
| `EigenAnalysis.h5` | `k`, `nondim_k`, `Eigenvalues`, `GrowthRates`, `PhaseSpeeds` with shape `(mode, k)`; zero-based `DominantMode` |
| `FourierState.h5` | Virtual complex fields `(ensemble, day, x, k)` |
| `GalerkinState.h5` | Virtual real fields `(ensemble, day, x, z, k)` |
| `ReconstructionParts/run-.../` | Required compressed Fourier/Galerkin data for the combined virtual datasets |
| `Figures/EigenAnalysis.png`, `.pdf` | Growth rate and phase speed |

The two-way state order is `(w1, w2, T1, T2, q, L)`, and one-way is `(T1, T2, q, L)`.
Two-way heating uses `U = r0*L + rq*(q - 1.5*T1)`, `J1 = L + U`, `J2 = L - U`.
One-way heating follows its operator: `J1 = L`, `J2 = g0*L + gq*(q - 1.5*T1)`.
Initial conditions are zero-mean complex random states weighted component-wise
by the normalized maximum-growth eigenvector: `q_j = 0.1 * v_j * xi_j`, where
each independent `xi_j` has real and imaginary variance `1/2`. This matches
Julia's multiplicative initialization with unit component scale factors.
Component-wise randomization mixes eigenmodes and does not enforce eastward
propagation. The ensemble mean is zero in expectation, not exactly zero for a
finite ensemble. Previously generated runs used additive noise around `v`.

Filenames and core variable names follow Julia, but Python HDF5 axis order is
explicit above. Julia/HDF5 dimension conventions and complex storage can differ;
the old Julia reconstruction and plotting scripts are not drop-in readers for
these files. Use the supplied Python readers. This workflow does not change the
physical differences between the two model versions, including radiation closure.
