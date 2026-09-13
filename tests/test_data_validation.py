"""Tests for RailBlock's Phase 0 synthetic data contract."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.generate_synthetic_data import generate_synthetic_data
from src.data.validation import DatasetValidationError, validate_dataset_directory


@pytest.fixture()
def synthetic_data_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "synthetic"
    generate_synthetic_data(directory)
    return directory


def test_generated_datasets_pass_schema_and_integrity_validation(
    synthetic_data_dir: Path,
) -> None:
    datasets = validate_dataset_directory(synthetic_data_dir)

    assert len(datasets["stations"]) == 18
    assert len(datasets["sections"]) == 17
    assert len(datasets["maintenance_tasks"]) == 108
    assert len(datasets["train_movements"]) == 560
    assert len(datasets["block_availability"]) == 357


def test_schema_validation_reports_missing_column(synthetic_data_dir: Path) -> None:
    path = synthetic_data_dir / "maintenance_tasks.csv"
    tasks = pd.read_csv(path)
    tasks.drop(columns="status").to_csv(path, index=False)

    with pytest.raises(DatasetValidationError, match="missing columns: status"):
        validate_dataset_directory(synthetic_data_dir)


def test_referential_integrity_rejects_unknown_section(synthetic_data_dir: Path) -> None:
    path = synthetic_data_dir / "maintenance_tasks.csv"
    tasks = pd.read_csv(path)
    tasks.loc[0, "section_id"] = "SEC999"
    tasks.to_csv(path, index=False)

    with pytest.raises(DatasetValidationError, match="unknown section_id value\\(s\\): SEC999"):
        validate_dataset_directory(synthetic_data_dir)


def test_time_validation_rejects_non_positive_train_occupancy(
    synthetic_data_dir: Path,
) -> None:
    path = synthetic_data_dir / "train_movements.csv"
    movements = pd.read_csv(path)
    movements.loc[0, "exit_time"] = movements.loc[0, "entry_time"]
    movements.to_csv(path, index=False)

    with pytest.raises(DatasetValidationError, match="exit_time must be after entry_time"):
        validate_dataset_directory(synthetic_data_dir)


def test_generation_is_deterministic(tmp_path: Path) -> None:
    first = generate_synthetic_data(tmp_path / "first", seed=77)
    second = generate_synthetic_data(tmp_path / "second", seed=77)

    for filename in (
        "stations.csv",
        "sections.csv",
        "maintenance_tasks.csv",
        "train_movements.csv",
        "block_availability.csv",
    ):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()


def test_generation_changes_with_a_different_seed(tmp_path: Path) -> None:
    first = generate_synthetic_data(tmp_path / "first", seed=77)
    second = generate_synthetic_data(tmp_path / "second", seed=78)

    assert (first / "sections.csv").read_bytes() != (second / "sections.csv").read_bytes()
