"""Tests for RailBlock Phase 2.3.1 risk model."""

from __future__ import annotations

import pandas as pd
import pytest

from src.prediction.train_risk_model import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    temporal_train_test_split,
)


def _dataset() -> pd.DataFrame:
    rows = []

    snapshots = pd.date_range(
        "2025-04-01",
        periods=3,
        freq="MS",
    )

    for snapshot_index, snapshot in enumerate(
        snapshots
    ):
        for asset_index in range(4):
            rows.append(
                {
                    "asset_id": f"A{asset_index:03d}",
                    "snapshot_date": snapshot,
                    "department": (
                        "Engineering"
                        if asset_index % 2 == 0
                        else "S&T"
                    ),
                    "asset_type": "rail",
                    "section_id": f"SEC{asset_index + 1:03d}",
                    "criticality": (
                        "high"
                        if asset_index == 0
                        else "medium"
                    ),
                    "installation_year": 2018,
                    "age_years": 8,
                    "days_since_last_maintenance": 30,
                    "has_maintenance_history": 1,
                    "maintenance_count_90d": 2,
                    "maintenance_count_180d": 3,
                    "maintenance_count_365d": 5,
                    "defect_count_90d": 1,
                    "defect_count_180d": 1,
                    "defect_count_365d": 2,
                    "failure_count_180d": 1,
                    "failure_count_365d": 1,
                    "average_downtime_min_180d": 120.0,
                    "average_downtime_min_365d": 100.0,
                    "total_downtime_min_365d": 500.0,
                    "average_severity_score_180d": 2.0,
                    "max_severity_score_180d": 4.0,
                    "failure_rate_365d": 0.2,
                    "defect_rate_365d": 0.4,
                    "maintenance_frequency_90d": 2.0,
                    TARGET_COLUMN: (
    1
    if (
        (
            snapshot_index == 1
            and asset_index == 1
        )
        or (
            snapshot_index == 2
            and asset_index == 0
        )
    )
    else 0
),
                }
            )

    return pd.DataFrame(rows)


def test_feature_columns_are_defined() -> None:
    assert CATEGORICAL_FEATURES
    assert NUMERIC_FEATURES
    assert FEATURE_COLUMNS

    assert set(FEATURE_COLUMNS) == (
        set(CATEGORICAL_FEATURES)
        | set(NUMERIC_FEATURES)
    )


def test_temporal_split_uses_latest_snapshot_as_test() -> None:
    dataset = _dataset()

    train, test = temporal_train_test_split(
        dataset
    )

    assert (
        train["snapshot_date"].max()
        < test["snapshot_date"].min()
    )

    assert (
        test["snapshot_date"].nunique()
        == 1
    )

    assert (
        test["snapshot_date"].iloc[0]
        == pd.Timestamp("2025-06-01")
    )


def test_temporal_split_contains_no_future_training_rows() -> None:
    dataset = _dataset()

    train, test = temporal_train_test_split(
        dataset
    )

    assert (
        train["snapshot_date"].max()
        < test["snapshot_date"].min()
    )


def test_temporal_split_preserves_all_rows() -> None:
    dataset = _dataset()

    train, test = temporal_train_test_split(
        dataset
    )

    assert len(train) + len(test) == len(dataset)


def test_target_is_not_a_feature() -> None:
    assert TARGET_COLUMN not in FEATURE_COLUMNS


def test_snapshot_date_is_not_a_feature() -> None:
    assert "snapshot_date" not in FEATURE_COLUMNS


def test_asset_id_is_not_a_feature() -> None:
    assert "asset_id" not in FEATURE_COLUMNS


def test_single_snapshot_is_rejected() -> None:
    dataset = _dataset()

    dataset = dataset[
        dataset["snapshot_date"]
        == pd.Timestamp("2025-04-01")
    ]

    with pytest.raises(
        ValueError,
        match="At least two snapshot dates",
    ):
        temporal_train_test_split(
            dataset
        )