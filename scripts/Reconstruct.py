"""
    Reconstruct daily spectral states using Fourier and vertical Galerkin bases
"""

# ====================================================
# Import package
# ====================================================

import argparse
from pathlib import Path

# RunConfig makes src importable when this file is executed directly.
from RunConfig import PROJECT_ROOT
from API import reconstruct_fields

# ====================================================
# Reconstruction entry point
# ====================================================

def main(input_path: Path, x_block_size: int = 64, overwrite: bool = False,
         workers: int = 4) -> None:
    reconstruct_fields(input_path, x_block_size=x_block_size, overwrite=overwrite,
                       verbose=True, workers=workers)

# ====================================================
# Run main function
# ====================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--x-block-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4,
                        help="Reconstruction processes (default: 4; use 1 for serial execution)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    main(args.input_dir, args.x_block_size, args.overwrite, args.workers)
