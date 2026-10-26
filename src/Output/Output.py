"""
    Shared output-directory handling for model results
"""

from pathlib import Path


def prepare_output(output_path: Path, filenames: tuple[str, ...], overwrite: bool) -> Path:
    output_path = Path(output_path)
    for name in filenames:
        if (output_path / name).exists() and not overwrite:
            raise FileExistsError(f"{output_path / name} exists; use --overwrite to replace it.")
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path

