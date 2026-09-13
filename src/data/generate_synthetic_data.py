"""Deterministically generate RailBlock's fictional Phase 0 corridor data."""

from __future__ import annotations

import argparse
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from src.data.validation import DATASET_FILES, validate_dataset_directory, write_csv_rows


DEFAULT_SEED = 20260105
PLANNING_START = date(2026, 1, 5)
PLANNING_DAYS = 7

STATION_NAMES = (
    "Aster Terminal",
    "Birchfield",
    "Copper Grove",
    "Dawn Ridge",
    "Elm Crossing",
    "Fern Valley",
    "Garnet Halt",
    "Harbor View",
    "Indigo Park",
    "Juniper Junction",
    "Kestrel Point",
    "Lumen Gate",
    "Meadowbridge",
    "Northstar",
    "Orchid Plain",
    "Pinecrest",
    "Quartz Bay",
    "Rivermark",
)

DEPARTMENT_WORK: dict[str, dict[str, tuple[str, ...]]] = {
    "Engineering": {
        "asset_types": ("track", "rail", "sleeper", "turnout"),
        "task_types": (
            "track inspection",
            "rail repair",
            "sleeper replacement",
            "turnout maintenance",
        ),
        "resources": (
            "track_team|inspection_trolley",
            "track_team|rail_grinder",
            "track_team|lifting_equipment",
            "track_team|turnout_specialist",
        ),
    },
    "S&T": {
        "asset_types": ("signal", "interlocking", "cable", "telecom"),
        "task_types": (
            "signal inspection",
            "signalling equipment maintenance",
            "cable inspection",
            "telecom equipment maintenance",
        ),
        "resources": (
            "signal_team|test_equipment",
            "signal_team|interlocking_specialist",
            "signal_team|cable_team",
            "telecom_team|test_equipment",
        ),
    },
    "Traction": {
        "asset_types": ("OHE", "mast", "isolator", "substation_equipment"),
        "task_types": (
            "OHE inspection",
            "OHE maintenance",
            "electrical equipment inspection",
        ),
        "resources": (
            "traction_team|OHE_vehicle",
            "traction_team|electrical_test_kit",
            "traction_team|isolator_specialist",
        ),
    },
}


def _time_text(value: datetime) -> str:
    return value.strftime("%H:%M:%S")


def _timestamp_text(value: datetime) -> str:
    return value.isoformat(timespec="seconds")


def _make_stations(rng: np.random.Generator) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    km_from_origin = 0.0
    for index, station_name in enumerate(STATION_NAMES, start=1):
        if index > 1:
            km_from_origin += float(rng.integers(8, 23))
        rows.append(
            {
                "station_id": f"ST{index:03d}",
                "station_name": station_name,
                "sequence": index,
                "km_from_origin": f"{km_from_origin:.1f}",
            }
        )
    return rows


def _make_sections(
    stations: list[dict[str, Any]], rng: np.random.Generator
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    notes = (
        "Fictional open-line segment.",
        "Fictional segment with a passing loop nearby.",
        "Fictional segment with a turnout approach.",
        "Fictional segment with an equipment cabin nearby.",
    )
    for index, (origin, destination) in enumerate(zip(stations, stations[1:]), start=1):
        distance = float(destination["km_from_origin"]) - float(origin["km_from_origin"])
        rows.append(
            {
                "section_id": f"SEC{index:03d}",
                "from_station": origin["station_id"],
                "to_station": destination["station_id"],
                "distance_km": f"{distance:.1f}",
                "normal_running_time_min": int(max(8, round(distance * rng.uniform(1.25, 1.8)))),
                "minimum_block_duration_min": int(rng.choice([45, 60, 75])),
                "notes": notes[(index - 1) % len(notes)],
            }
        )
    return rows


def _make_maintenance_tasks(
    sections: list[dict[str, Any]], rng: np.random.Generator
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    departments = ("Engineering", "S&T", "Traction")
    base_time = datetime.combine(PLANNING_START, time(7, 0))
    deadline_offsets = {
        "critical": (1, 3),
        "high": (2, 5),
        "medium": (4, 7),
        "low": (6, 10),
    }
    for index in range(1, 109):
        department = departments[(index - 1) % len(departments)]
        work = DEPARTMENT_WORK[department]
        section = sections[int(rng.integers(0, len(sections)))]
        asset_type = str(rng.choice(work["asset_types"]))
        task_type = str(rng.choice(work["task_types"]))
        resources = str(rng.choice(work["resources"]))
        criticality = str(
            rng.choice(["low", "medium", "high", "critical"], p=[0.25, 0.4, 0.25, 0.1])
        )
        urgency = str(rng.choice(["low", "medium", "high"], p=[0.25, 0.5, 0.25]))
        earliest_start = base_time + timedelta(
            days=int(rng.integers(0, PLANNING_DAYS)),
            minutes=int(rng.integers(0, 11 * 60) // 15 * 15),
        )
        minimum_days, maximum_days = deadline_offsets[criticality]
        deadline = earliest_start + timedelta(
            days=int(rng.integers(minimum_days, maximum_days + 1)),
            hours=int(rng.integers(0, 8)),
        )
        abbreviation = {"Engineering": "ENG", "S&T": "SNT", "Traction": "TRC"}[department]
        rows.append(
            {
                "task_id": f"MT{index:04d}",
                "department": department,
                "asset_type": asset_type,
                "section_id": section["section_id"],
                "asset_id": f"{abbreviation}-{section['section_id']}-{index:03d}",
                "task_type": task_type,
                "criticality": criticality,
                "urgency": urgency,
                "estimated_duration_min": int(rng.choice([45, 60, 75, 90, 120, 150, 180])),
                "earliest_start": _timestamp_text(earliest_start),
                "deadline": _timestamp_text(deadline),
                "required_resources": resources,
                "status": str(rng.choice(["requested", "approved", "deferred"], p=[0.6, 0.3, 0.1])),
            }
        )
    return rows


def _make_train_movements(
    sections: list[dict[str, Any]], rng: np.random.Generator
) -> list[dict[str, Any]]:
    """Create 20 four-section fictional journeys per day (80 records/day)."""

    rows: list[dict[str, Any]] = []
    section_count_per_journey = 4
    for day_offset in range(PLANNING_DAYS):
        service_date = PLANNING_START + timedelta(days=day_offset)
        for journey_index in range(1, 21):
            direction = "UP" if journey_index % 2 else "DOWN"
            train_type = "goods" if journey_index % 4 == 0 else "passenger"
            starting_position = int(
                rng.integers(0, len(sections) - section_count_per_journey + 1)
            )
            positions = list(range(starting_position, starting_position + section_count_per_journey))
            if direction == "DOWN":
                positions.reverse()
            start_time = datetime.combine(service_date, time(5, 30)) + timedelta(
                minutes=int(rng.integers(0, 14 * 60))
            )
            current_time = start_time
            train_id = f"{'GDS' if train_type == 'goods' else 'PAX'}{journey_index:03d}"
            expected_delay = int(rng.integers(0, 31 if train_type == "goods" else 16))
            priority = (
                str(rng.choice(["medium", "high"], p=[0.3, 0.7]))
                if train_type == "passenger"
                else str(rng.choice(["low", "medium"], p=[0.7, 0.3]))
            )
            for leg_number, section_position in enumerate(positions, start=1):
                section = sections[section_position]
                running_minutes = int(section["normal_running_time_min"]) + int(rng.integers(0, 4))
                exit_time = current_time + timedelta(minutes=running_minutes)
                rows.append(
                    {
                        "movement_id": (
                            f"MV{service_date.strftime('%Y%m%d')}{journey_index:03d}{leg_number:02d}"
                        ),
                        "train_id": train_id,
                        "train_type": train_type,
                        "service_date": service_date.isoformat(),
                        "section_id": section["section_id"],
                        "entry_time": _time_text(current_time),
                        "exit_time": _time_text(exit_time),
                        "direction": direction,
                        "priority": priority,
                        "expected_delay_min": expected_delay,
                    }
                )
                current_time = exit_time + timedelta(minutes=int(rng.integers(2, 7)))
    return rows


def _make_block_availability(
    sections: list[dict[str, Any]], rng: np.random.Generator
) -> list[dict[str, Any]]:
    """Create two available night windows and one protected daytime period per section/day."""

    rows: list[dict[str, Any]] = []
    window_number = 1
    for day_offset in range(PLANNING_DAYS):
        window_date = PLANNING_START + timedelta(days=day_offset)
        for section_index, section in enumerate(sections):
            minimum_duration = int(section["minimum_block_duration_min"])
            early_start = datetime.combine(window_date, time(1, 0)) + timedelta(
                minutes=(section_index * 5 + day_offset * 3) % 30
            )
            early_end = early_start + timedelta(
                minutes=max(minimum_duration + 30, int(rng.choice([90, 105, 120])))
            )
            midday_start = datetime.combine(window_date, time(10, 0)) + timedelta(
                minutes=(section_index % 3) * 10
            )
            midday_end = midday_start + timedelta(minutes=120)
            late_start = datetime.combine(window_date, time(21, 15)) + timedelta(
                minutes=(section_index * 4 + day_offset * 2) % 25
            )
            late_end = late_start + timedelta(minutes=max(minimum_duration + 15, 90))
            windows = (
                (early_start, early_end, "available", "Fictional night maintenance allowance."),
                (midday_start, midday_end, "unavailable", "Fictional passenger peak protection period."),
                (late_start, late_end, "available", "Fictional late-night maintenance allowance."),
            )
            for start, end, availability_status, reason in windows:
                rows.append(
                    {
                        "block_window_id": f"BW{window_number:05d}",
                        "section_id": section["section_id"],
                        "date": window_date.isoformat(),
                        "start_time": _time_text(start),
                        "end_time": _time_text(end),
                        "availability_status": availability_status,
                        "reason": reason,
                    }
                )
                window_number += 1
    return rows


def generate_synthetic_data(
    output_dir: str | Path | None = None, seed: int = DEFAULT_SEED
) -> Path:
    """Generate and validate the fictional corridor dataset using ``seed``."""

    directory = (
        Path(output_dir)
        if output_dir is not None
        else Path(__file__).resolve().parents[2] / "data" / "synthetic"
    )
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    stations = _make_stations(rng)
    sections = _make_sections(stations, rng)
    datasets: dict[str, list[dict[str, Any]]] = {
        "stations": stations,
        "sections": sections,
        "maintenance_tasks": _make_maintenance_tasks(sections, rng),
        "train_movements": _make_train_movements(sections, rng),
        "block_availability": _make_block_availability(sections, rng),
    }
    for name, filename in DATASET_FILES.items():
        write_csv_rows(directory / filename, list(datasets[name][0]), datasets[name])

    validate_dataset_directory(directory)
    return directory


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate fictional RailBlock Phase 0 datasets.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Deterministic random seed.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Destination directory (default: data/synthetic).",
    )
    return parser.parse_args()


def main() -> None:
    """Run the generator as ``python -m src.data.generate_synthetic_data``."""

    args = _parse_args()
    directory = generate_synthetic_data(args.output_dir, args.seed)
    datasets = validate_dataset_directory(directory)
    sizes = ", ".join(f"{name}={len(frame)}" for name, frame in datasets.items())
    print(f"Generated fictional synthetic RailBlock data in {directory} ({sizes}).")


if __name__ == "__main__":
    main()
