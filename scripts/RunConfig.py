"""
    Shared configuration and output helpers for the model execution scripts
"""

# ====================================================
# Import package
# ====================================================

import argparse
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from Configuration.Domain import Domain
from Configuration.ModelConfig import ModelConfig, validate_domain, configuration
from Output.Output import prepare_output

# ====================================================
# Configuration functions
# ====================================================

def add_run_arguments(parser: argparse.ArgumentParser) -> None:
    """Common command-line settings; omitted values use the src classes"""

    parser.add_argument("--model", choices=("one-way", "two-way"), default="two-way")
    parser.add_argument("--rad-scaling", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--members", type=int)
    parser.add_argument("--tmax", type=float)
    parser.add_argument("--dt", type=float)
    parser.add_argument("--dx", type=float)
    parser.add_argument("--dz", type=float)
    parser.add_argument("--zonal-wavenumbers", type=float, nargs="+")
    parser.add_argument("--seed", type=int, help="Seed for the existing NumPy initial-condition generator")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output files")


def domain_from_arguments(args: argparse.Namespace) -> Domain:
    """Apply run-specific settings without changing the source configuration"""

    domain = Domain()
    for argument, attribute in (("members", "Ens"), ("tmax", "Tmax"),
                                ("dt", "dt"), ("dx", "dx"), ("dz", "dz")):
        value = getattr(args, argument)
        if value is not None:
            setattr(domain, attribute, value)
    if args.zonal_wavenumbers is not None:
        domain.nondim_k = list(args.zonal_wavenumbers)
    validate_domain(domain)
    if not np.isfinite(args.rad_scaling):
        raise ValueError("rad-scaling must be finite.")
    if args.seed is not None and not 0 <= args.seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32).")
    return domain


def output_from_arguments(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir.resolve()
    stamp = datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y-%m-%d-%H-%M-%S-%f")
    return PROJECT_ROOT / "output" / args.model / f"Rad({args.rad_scaling:g})" / stamp


def config_from_arguments(args: argparse.Namespace) -> ModelConfig:
    return ModelConfig(args.model, args.rad_scaling, domain_from_arguments(args), args.seed)
