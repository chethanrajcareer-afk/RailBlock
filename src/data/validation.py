"""Validation and loading for RailBlock's synthetic CSV data contract.

The checks in this module are data-quality checks only. They are not a
substitute for railway operating, safety, or dispatching validation.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping
from datetime import date, datetime, time
from pathlib import Path
from typing import Annotated, Any, Literal, Self

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


NonEmptyText = Annotated[str, Field(min_length=1)]
PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveFloat = Annotated[float, Field(gt=0)]
NonNegativeFloat = Annotated[float, Field(ge=0)]

Department = Literal["Engineering", "S&T", "Traction"]
Criticality = Literal["low", "medium", "high", "critical"]
Urgency = Literal["low", "medium", "high"]
TaskStatus = Literal["requested", "approved", "deferred"]
TrainType = Literal["passenger", "goods"]
Direction = Literal["UP", "DOWN"]
Priority = Literal["low", "medium", "high"]
AvailabilityStatus = Literal["available", "unavailable"]


class DatasetValidationError(ValueError):
    """Raised when one or more datasets violate the RailBlock data contract."""

    def __init__(self, errors: Iterable[str]) -> None:
        self.errors = list(errors)
        message = "Dataset validation failed:\n- " + "\n- ".join(self.errors)
        super().__init__(message)


class RailBlockRecord(BaseModel):
    """Common strict configuration for CSV row models."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class StationRecord(RailBlockRecord):
    station_id: NonEmptyText
    station_name: NonEmptyText
    sequence: PositiveInt
    km_from_origin: NonNegativeFloat


class SectionRecord(RailBlockRecord):
    section_id: NonEmptyText
    from_station: NonEmptyText
    to_station: NonEmptyText
    distance_km: PositiveFloat
    normal_running_time_min: PositiveInt
    minimum_block_duration_min: PositiveInt
    notes: NonEmptyText

    @model_validator(mode="after")
    def stations_must_differ(self) -> Self:
        if self.from_station == self.to_station:
            raise ValueError("from_station and to_station must differ")
        return self


class MaintenanceTaskRecord(RailBlockRecord):
    task_id: NonEmptyText
    department: Department
    asset_type: NonEmptyText
    section_id: NonEmptyText
    asset_id: NonEmptyText
    task_type: NonEmptyText
    criticality: Criticality
    urgency: Urgency
    estimated_duration_min: PositiveInt
    earliest_start: datetime
    deadline: datetime
    required_resources: NonEmptyText
    status: TaskStatus

    @model_validator(mode="after")
    def deadline_must_follow_earliest_start(self) -> Self:
        if self.deadline <= self.earliest_start:
            raise ValueError("deadline must be after earliest_start")
        return self


class TrainMovementRecord(RailBlockRecord):
    movement_id: NonEmptyText
    train_id: NonEmptyText
    train_type: TrainType
    service_date: date
    section_id: NonEmptyText
    entry_time: time
    exit_time: time
    direction: Direction
    priority: Priority
    expected_delay_min: NonNegativeInt

    @model_validator(mode="after")
    def exit_must_follow_entry(self) -> Self:
        if self.exit_time <= self.entry_time:
            raise ValueError("exit_time must be after entry_time")
        return self


class BlockAvailabilityRecord(RailBlockRecord):
    block_window_id: NonEmptyText
    section_id: NonEmptyText
    date: date
    start_time: time
    end_time: time
    availability_status: AvailabilityStatus
    reason: NonEmptyText

    @model_validator(mode="after")
    def end_must_follow_start(self) -> Self:
        if self.end_time <= self.start_time:
            raise ValueError("end_time must be after start_time")
        return self


DATASET_FILES: Mapping[str, str] = {
    "stations": "stations.csv",
    "sections": "sections.csv",
    "maintenance_tasks": "maintenance_tasks.csv",
    "train_movements": "train_movements.csv",
    "block_availability": "block_availability.csv",
}

RECORD_MODELS: Mapping[str, type[RailBlockRecord]] = {
    "stations": StationRecord,
    "sections": SectionRecord,
    "maintenance_tasks": MaintenanceTaskRecord,
    "train_movements": TrainMovementRecord,
    "block_availability": BlockAvailabilityRecord,
}


def _expected_columns(model: type[RailBlockRecord]) -> list[str]:
    return list(model.model_fields)


def _read_and_validate_rows(path: Path, dataset_name: str) -> pd.DataFrame:
    """Read one CSV and validate its headers and each row against its model."""

    if not path.is_file():
        raise DatasetValidationError([f"{dataset_name}: missing file {path}"])

    model = RECORD_MODELS[dataset_name]
    try:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    except (pd.errors.ParserError, UnicodeDecodeError) as error:
        raise DatasetValidationError([f"{dataset_name}: cannot parse CSV: {error}"]) from error

    expected = _expected_columns(model)
    actual = list(frame.columns)
    missing = [column for column in expected if column not in actual]
    unexpected = [column for column in actual if column not in expected]
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append(f"missing columns: {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected columns: {', '.join(unexpected)}")
        raise DatasetValidationError([f"{dataset_name}: {'; '.join(details)}"])

    row_errors: list[str] = []
    validated_rows: list[dict[str, Any]] = []
    for index, row in frame.iterrows():
        try:
            record = model.model_validate(row.to_dict())
        except ValidationError as error:
            for detail in error.errors(include_url=False):
                field = ".".join(str(item) for item in detail["loc"]) or "row"
                row_errors.append(
                    f"{dataset_name}: row {index + 2}, {field}: {detail['msg']}"
                )
        else:
            validated_rows.append(record.model_dump())

    if row_errors:
        raise DatasetValidationError(row_errors)

    return pd.DataFrame(validated_rows, columns=expected)


def _require_unique(frame: pd.DataFrame, dataset_name: str, column: str) -> list[str]:
    duplicates = frame.loc[frame[column].duplicated(), column].tolist()
    if not duplicates:
        return []
    values = ", ".join(str(value) for value in sorted(set(duplicates)))
    return [f"{dataset_name}: duplicate {column} value(s): {values}"]


def _validate_referential_integrity(datasets: Mapping[str, pd.DataFrame]) -> list[str]:
    """Validate cross-file IDs and linear-corridor ordering rules."""

    errors: list[str] = []
    stations = datasets["stations"]
    sections = datasets["sections"]
    tasks = datasets["maintenance_tasks"]
    movements = datasets["train_movements"]
    blocks = datasets["block_availability"]

    errors.extend(_require_unique(stations, "stations", "station_id"))
    errors.extend(_require_unique(stations, "stations", "sequence"))
    errors.extend(_require_unique(sections, "sections", "section_id"))
    errors.extend(_require_unique(tasks, "maintenance_tasks", "task_id"))
    errors.extend(_require_unique(movements, "train_movements", "movement_id"))
    errors.extend(_require_unique(blocks, "block_availability", "block_window_id"))
    if errors:
        return errors

    station_sequences = dict(zip(stations["station_id"], stations["sequence"], strict=True))
    station_ids = set(station_sequences)
    section_ids = set(sections["section_id"])

    for row in sections.itertuples(index=False):
        if row.from_station not in station_ids:
            errors.append(f"sections: {row.section_id} references unknown from_station {row.from_station}")
        if row.to_station not in station_ids:
            errors.append(f"sections: {row.section_id} references unknown to_station {row.to_station}")
        if (
            row.from_station in station_sequences
            and row.to_station in station_sequences
            and station_sequences[row.from_station] >= station_sequences[row.to_station]
        ):
            errors.append(f"sections: {row.section_id} has invalid station order")

    duplicate_pairs = sections.duplicated(["from_station", "to_station"])
    if duplicate_pairs.any():
        duplicate_ids = ", ".join(sections.loc[duplicate_pairs, "section_id"].tolist())
        errors.append(f"sections: duplicate station-pair section(s): {duplicate_ids}")

    for dataset_name, frame in (
        ("maintenance_tasks", tasks),
        ("train_movements", movements),
        ("block_availability", blocks),
    ):
        unknown_sections = sorted(set(frame.loc[~frame["section_id"].isin(section_ids), "section_id"]))
        if unknown_sections:
            errors.append(
                f"{dataset_name}: unknown section_id value(s): {', '.join(unknown_sections)}"
            )

    if errors:
        return errors

    errors.extend(_validate_train_paths(movements, sections))
    return errors


def _validate_train_paths(movements: pd.DataFrame, sections: pd.DataFrame) -> list[str]:
    """Check contiguous, time-ordered paths for this synthetic linear corridor."""

    section_order = {
        section_id: position
        for position, section_id in enumerate(sections["section_id"].tolist(), start=1)
    }
    errors: list[str] = []
    grouped = movements.groupby(["service_date", "train_id", "direction"], sort=False)
    for (service_date, train_id, direction), journey in grouped:
        ordered = journey.sort_values("entry_time")
        section_positions = [section_order[section_id] for section_id in ordered["section_id"]]
        expected_step = 1 if direction == "UP" else -1
        for previous, current in zip(section_positions, section_positions[1:], strict=False):
            if current - previous != expected_step:
                errors.append(
                    f"train_movements: {train_id} on {service_date} has non-contiguous {direction} sections"
                )
                break
        previous_exit: time | None = None
        for row in ordered.itertuples(index=False):
            if previous_exit is not None and row.entry_time < previous_exit:
                errors.append(
                    f"train_movements: {train_id} on {service_date} has overlapping section times"
                )
                break
            previous_exit = row.exit_time
    return errors


def validate_dataset_directory(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load, validate, and return every dataset in a synthetic data directory."""

    directory = Path(data_dir)
    datasets = {
        name: _read_and_validate_rows(directory / filename, name)
        for name, filename in DATASET_FILES.items()
    }
    errors = _validate_referential_integrity(datasets)
    if errors:
        raise DatasetValidationError(errors)
    return datasets


def load_datasets(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Alias for the public load-and-validate entry point."""

    return validate_dataset_directory(data_dir)


def write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    """Write a deterministic UTF-8 CSV file for the synthetic generator."""

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
