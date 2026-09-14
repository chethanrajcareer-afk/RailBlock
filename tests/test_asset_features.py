"""Tests for RailBlock Phase 2.1 feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.build_asset_features import (
    REFERENCE_DATE,
    build_asset_features,
    validate_features,
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
                "last_major_maintenance": "2025-10-01",
                "age_years": 8,
            },
            {
                "asset_id": "A002",
                "department": "S&T",
                "asset_type": "signal",
                "section_id": "SEC002",
                "criticality": "critical",
                "installation_year": 2015,
                "last_major_maintenance": "2025-11-01",
                "age_years": 11,
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
                "maintenance_date": "2025-12-20",
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
                "maintenance_date": "2025-08-01",
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
                "maintenance_date": "2025-12-01",
                "maintenance_type": "inspection",
                "defect_found": 0,
                "failure_occurred": 0,
                "downtime_min": 60,
                "severity": "low",
            },
            {
                "history_id": "H004",
                "asset_id": "A002",
                "section_id": "SEC002",
                "department": "S&T",
                "maintenance_date": "2026-01-05",
                "maintenance_type": "repair",
                "defect_found": 1,
                "failure_occurred": 1,
                "downtime_min": 300,
                "severity": "critical",
            },
        ]
    )


def test_features_have_one_row_per_asset() -> None:
    features = build_asset_features(_assets(), _history())

    assert len(features) == 2
    assert features["asset_id"].is_unique


def test_reference_date_prevents_future_leakage() -> None:
    features = build_asset_features(_assets(), _history())

    a002 = features.loc[features["asset_id"] == "A002"].iloc[0]

    # H004 occurs exactly on the reference date and must not enter the features.
    assert a002["maintenance_count_365d"] == 1
    assert a002["failure_count_365d"] == 0
    assert a002["defect_count_365d"] == 0


def test_window_features_are_calculated_correctly() -> None:
    features = build_asset_features(_assets(), _history())

    a001 = features.loc[features["asset_id"] == "A001"].iloc[0]

    assert a001["maintenance_count_90d"] == 1
    assert a001["maintenance_count_180d"] == 2
    assert a001["defect_count_180d"] == 2
    assert a001["failure_count_180d"] == 1
    assert a001["days_since_last_maintenance"] == 16


def test_rates_and_downtime_are_correct() -> None:
    features = build_asset_features(_assets(), _history())

    a001 = features.loc[features["asset_id"] == "A001"].iloc[0]

    assert a001["failure_rate_365d"] == 0.5
    assert a001["defect_rate_365d"] == 1.0
    assert a001["average_downtime_min_365d"] == 180.0
    assert a001["total_downtime_min_365d"] == 360


def test_validation_passes() -> None:
    assets = _assets()
    features = build_asset_features(assets, _history())

    validate_features(features, assets)


def test_no_missing_feature_values() -> None:
    features = build_asset_features(_assets(), _history())

    assert not features.isna().any().any()
