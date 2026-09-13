"""Command-line validation for a RailBlock synthetic data directory."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data.validation import DatasetValidationError, validate_dataset_directory


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate RailBlock synthetic CSV datasets.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "synthetic",
        help="Directory containing the five RailBlock CSV files.",
    )
    args = parser.parse_args()
    try:
        datasets = validate_dataset_directory(args.data_dir)
    except DatasetValidationError as error:
        parser.error(str(error))
    sizes = ", ".join(f"{name}={len(frame)}" for name, frame in datasets.items())
    print(f"Validation passed for {args.data_dir} ({sizes}).")


if __name__ == "__main__":
    main()
