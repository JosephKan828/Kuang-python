"""
    Execute the simulation, eigen-analysis, optional reconstruction, and plotting
"""

# ====================================================
# Import package
# ====================================================

import argparse

from RunConfig import add_run_arguments, config_from_arguments, output_from_arguments
from API import run_model

# ====================================================
# Run main function
# ====================================================

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_run_arguments(parser)
    parser.add_argument("--reconstruct", action="store_true", help="Also write Fourier and Galerkin fields")
    parser.add_argument("--x-block-size", type=int, default=64)
    parser.add_argument("--reconstruction-workers", type=int, default=4,
                        help="Reconstruction processes (default: 4)")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    config = config_from_arguments(args)
    output_path = output_from_arguments(args)
    print(f"Output directory: {output_path}", flush=True)
    run_model(config, output_path, reconstruct=args.reconstruct,
              x_block_size=args.x_block_size, overwrite=args.overwrite, verbose=True,
              reconstruction_workers=args.reconstruction_workers)
    if not args.no_plot:
        from Plot_Eigen import main as plot_eigen
        plot_eigen([output_path])
    print("Model run completed.", flush=True)


if __name__ == "__main__":
    main()
