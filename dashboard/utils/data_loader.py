"""Load validated synthetic source data for the RailBlock dashboard."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.validation import DATASET_FILES, DatasetValidationError, validate_dataset_directory


class DashboardDataError(RuntimeError):
    """Raised when the dashboard's required synthetic inputs are unavailable or invalid."""


def project_root() -> Path:
    """Return the repository root without relying on the current working directory."""

    return Path(__file__).resolve().parents[2]


def default_data_dir() -> Path:
    """Return the synthetic source-data directory."""

    return project_root() / "data" / "synthetic"


def _prepare_frames(datasets: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Normalize display-friendly dates while preserving validated source records."""

    prepared = {name: frame.copy() for name, frame in datasets.items()}
    prepared["stations"] = prepared["stations"].sort_values("sequence").reset_index(drop=True)
    prepared["sections"] = prepared["sections"].reset_index(drop=True)

    tasks = prepared["maintenance_tasks"]
    tasks["earliest_start"] = pd.to_datetime(tasks["earliest_start"])
    tasks["deadline"] = pd.to_datetime(tasks["deadline"])

    movements = prepared["train_movements"]
    movements["service_date"] = pd.to_datetime(movements["service_date"])

    blocks = prepared["block_availability"]
    blocks["date"] = pd.to_datetime(blocks["date"])
    return prepared


def load_dashboard_data(data_dir: str | Path | None = None) -> dict[str, pd.DataFrame]:
    """Load the existing CSVs, checking file presence and the shared data contract."""

    directory = Path(data_dir) if data_dir is not None else default_data_dir()
    missing = [filename for filename in DATASET_FILES.values() if not (directory / filename).is_file()]
    if missing:
        names = ", ".join(missing)
        raise DashboardDataError(
            f"Expected synthetic source data is missing from {directory}: {names}. "
            "Run the synthetic-data generator before starting the dashboard."
        )

    try:
        datasets = validate_dataset_directory(directory)
    except DatasetValidationError as error:
        raise DashboardDataError(
            "Synthetic source data failed validation. Resolve the dataset errors before "
            f"starting the dashboard.\n{error}"
        ) from error
    return _prepare_frames(datasets)
