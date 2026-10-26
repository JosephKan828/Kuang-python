"""
    Plot growth rate and phase speed for one run or compare two radiation cases
"""

# ====================================================
# Import package
# ====================================================

import argparse
from pathlib import Path

import h5py
import numpy as np
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

# ====================================================
# Plotting function
# ====================================================

def main(input_paths: list[Path], figure_path: Path | None = None) -> Path:
    """Plot the fastest-growing mode, with consistent conjugate-pair selection"""

    if not 1 <= len(input_paths) <= 2:
        raise ValueError("Supply one or two run directories.")
    cases = []
    for path in input_paths:
        with h5py.File(Path(path) / "EigenAnalysis.h5", "r") as file:
            growth, speed = file["GrowthRates"][:], file["PhaseSpeeds"][:]
            selected = file["DominantMode"][:]
            columns = np.arange(selected.size)
            cases.append((file.attrs["rad_scaling"], file.attrs["model_type"], file["nondim_k"][:],
                          growth[selected, columns], speed[selected, columns]))
    cases.sort(key=lambda case: case[0])
    if len(cases) == 2 and (cases[0][1] != cases[1][1] or not np.array_equal(cases[0][2], cases[1][2])):
        raise ValueError("Comparison runs must use the same model and wavenumber grid.")

    figure_path = Path(input_paths[0]) / "Figures" if figure_path is None else Path(figure_path)
    figure_path.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for (scaling, model, wavenumbers, growth, speed), style in zip(cases, ("-", "--")):
        options = dict(color="black", linestyle=style, linewidth=2, label=f"Radiation scaling = {scaling:.1f}")
        axes[0].plot(wavenumbers, growth, **options)
        axes[1].plot(wavenumbers, speed, **options)
    axes[0].set_title("(a) Growth rate")
    axes[1].set_title("(b) Phase speed")
    axes[0].set_ylabel(r"Growth rate (day$^{-1}$)")
    axes[1].set_ylabel(r"Phase speed (m s$^{-1}$)")
    for ax in axes:
        ax.set_xlabel("Zonal wavenumber")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
    fig.suptitle(f"{cases[0][1]}: fastest-growing mode")
    for extension in ("png", "pdf"):
        fig.savefig(figure_path / f"EigenAnalysis.{extension}", dpi=200)
    plt.close(fig)
    print(f"Figures saved to {figure_path}", flush=True)
    return figure_path

# ====================================================
# Run main function
# ====================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dirs", type=Path, nargs="+")
    parser.add_argument("--figure-dir", type=Path)
    args = parser.parse_args()
    main(args.input_dirs, args.figure_dir)
