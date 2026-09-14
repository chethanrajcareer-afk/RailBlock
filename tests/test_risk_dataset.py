"""Tests for RailBlock Phase 2.3.0 supervised risk dataset."""

from __future__ import annotations

import pandas as pd
import pytest

from src.prediction.build_risk_dataset import (
    TARGET_HORIZON_DAYS,
    build_risk_training_dataset,
    validate_risk_training_dataset,
)


def _assets() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "asset_id": "A001",
                "department": "Engineering",
                "asset_type": "rail",
                "section_id": "SEC001",
                "criticality": "high",
                "installation_year": 2018,
                "age_years": 8,
            },
            {
                "asset_id": "A002",
                "department": "S&T",
                "asset_type": "signal",
                "section_id": "SEC002",
                "criticality": "medium",
                "installation_year": 2020,
                "age_years": 6,
            },
        ]
    )


def _history() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "history_id": "H001",
                "asset_id": "A001",
                "section_id": "SEC001",
                "department": "Engineering",
                "maintenance_date": "2025-01-15",
                "maintenance_type": "repair",
                "defect_found": 1,
                "failure_occurred": 0,
                "downtime_min": 120,
                "severity": "high",
            },
            {
                "history_id": "H002",
                "asset_id": "A001",
                "section_id": "SEC001",
                "department": "Engineering",
                "maintenance_date": "2025-03-15",
                "maintenance_type": "repair",
                "defect_found": 1,
                "failure_occurred": 1,
                "downtime_min": 240,
                "severity": "critical",
            },
            {
                "history_id": "H003",
                "asset_id": "A002",
                "section_id": "SEC002",
                "department": "S&T",
                "maintenance_date": "2025-03-20",
                "maintenance_type": "inspection",
                "defect_found": 0,
                "failure_occurred": 0,
                "downtime_min": 60,
                "severity": "low",
            },
            {
                "history_id": "H004",
                "asset_id": "A001",
                "section_id": "SEC001",
                "department": "Engineering",
                "maintenance_date": "2025-05-15",
                "maintenance_type": "emergency_repair",
                "defect_found": 1,
                "failure_occurred": 1,
                "downtime_min": 300,
                "severity": "critical",
            },
            {
                "history_id": "H005",
                "asset_id": "A002",
                "section_id": "SEC002",
                "department": "S&T",
                "maintenance_date": "2025-05-20",
                "maintenance_type": "repair",
                "defect_found": 1,
                "failure_occurred": 0,
                "downtime_min": 90,
                "severity": "medium",
            },
            {
                "history_id": "H006",
                "asset_id": "A002",
                "section_id": "SEC002",
                "department": "S&T",
                # Deliberately after the 2025-05-20 snapshot's
                # 90-day target horizon. This lets us test that
                # future events are not incorrectly included.
                "maintenance_date": "2025-08-20",
                "maintenance_type": "emergency_repair",
                "defect_found": 1,
                "failure_occurred": 1,
                "downtime_min": 240,
                "severity": "critical",
            },
            {
                "history_id": "H007",
                "asset_id": "A002",
                "section_id": "SEC002",
                "department": "S&T",
                "maintenance_date": "2025-09-01",
                "maintenance_type": "inspection",
                "defect_found": 0,
                "failure_occurred": 0,
                "downtime_min": 30,
                "severity": "low",
            },
        ]
    )


def test_dataset_has_one_row_per_asset_per_snapshot() -> None:
    dataset = build_risk_training_dataset(
        _assets(),
        _history(),
        [pd.Timestamp("2025-04-01")],
    )

    assert len(dataset) == 2
    assert not dataset[
        ["asset_id", "snapshot_date"]
    ].duplicated().any()


def test_future_failure_becomes_target() -> None:
    dataset = build_risk_training_dataset(
        _assets(),
        _history(),
        [pd.Timestamp("2025-04-01")],
    ).set_index("asset_id")

    # A001 fails on 2025-05-15, inside the 90-day horizon.
    assert dataset.loc[
        "A001",
        "failure_in_next_90d",
    ] == 1

    # A002 has no failure between 2025-04-01 and 2025-06-30.
    # Its next failure occurs on 2025-08-20.
    assert dataset.loc[
        "A002",
        "failure_in_next_90d",
    ] == 0


def test_snapshot_features_do_not_use_future_events() -> None:
    dataset = build_risk_training_dataset(
        _assets(),
        _history(),
        [pd.Timestamp("2025-04-01")],
    ).set_index("asset_id")

    # A001's May failure must not affect its April features.
    assert dataset.loc[
        "A001",
        "failure_count_180d",
    ] == 1

    assert dataset.loc[
        "A001",
        "defect_count_180d",
    ] == 2


def test_events_on_snapshot_date_are_not_features() -> None:
    dataset = build_risk_training_dataset(
        _assets(),
        _history(),
        [pd.Timestamp("2025-05-20")],
    ).set_index("asset_id")

    # H003 on 2025-03-20 is inside the 90-day
    # historical feature window.
    assert dataset.loc[
        "A002",
        "maintenance_count_90d",
    ] == 1

    # H005 occurs exactly on the snapshot date and must
    # therefore NOT be included in the feature window.
    assert dataset.loc[
        "A002",
        "defect_count_90d",
    ] == 0


def test_incomplete_future_horizon_is_skipped() -> None:
    dataset = build_risk_training_dataset(
        _assets(),
        _history(),
        [pd.Timestamp("2025-08-01")],
    )

    # A complete 90-day target would require history through
    # 2025-10-30, but the fixture ends on 2025-09-01.
    assert dataset.empty


def test_validation_passes() -> None:
    dataset = build_risk_training_dataset(
        _assets(),
        _history(),
        [pd.Timestamp("2025-04-01")],
    )

    validate_risk_training_dataset(
        dataset,
        _assets(),
    )


def test_invalid_target_is_rejected() -> None:
    dataset = build_risk_training_dataset(
        _assets(),
        _history(),
        [pd.Timestamp("2025-04-01")],
    )

    dataset.loc[
        0,
        "failure_in_next_90d",
    ] = 2

    with pytest.raises(
        ValueError,
        match="Target",
    ):
        validate_risk_training_dataset(
            dataset,
            _assets(),
        )