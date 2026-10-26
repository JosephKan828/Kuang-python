"""
    Run the spectral ensemble and save state histories and configuration
"""

# ====================================================
# Import package
# ====================================================

import argparse
from pathlib import Path

from RunConfig import add_run_arguments, domain_from_arguments, output_from_arguments
from API import Domain, ModelConfig, run_simulation

# ====================================================
# Simulation entry point
# ====================================================

def main(model_type: str, rad_scaling: float, output_path: Path,
         domain: Domain | None = None, seed: int | None = None,
         overwrite: bool = False) -> Path:
    config = ModelConfig(model_type, rad_scaling, Domain() if domain is None else domain, seed)
    return run_simulation(config, output_path, overwrite=overwrite, verbose=True).directory

# ====================================================
# Run main function
# ====================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_run_arguments(parser)
    args = parser.parse_args()
    main(args.model, args.rad_scaling, output_from_arguments(args),
         domain_from_arguments(args), args.seed, args.overwrite)
