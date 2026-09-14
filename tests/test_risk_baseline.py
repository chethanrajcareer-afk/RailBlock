"""Tests for RailBlock Phase 2.2 baseline risk scoring."""

from __future__ import annotations

import pandas as pd
import pytest

from src.prediction.risk_baseline import (
    calculate_baseline_risk,
    validate_baseline_risk,
)


def _features() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "asset_id": "A001",
                "department": "Engineering",
                "asset_type": "rail",
                "section_id": "SEC001",
                "criticality": "critical",
                "days_since_last_maintenance": 100,
                "defect_count_180d": 5,
                "failure_count_180d": 2,
                "failure_rate_365d": 0.50,
                "average_downtime_min_180d": 240,
            },
            {
                "asset_id": "A002",
                "department": "Engineering",
                "asset_type": "rail",
                "section_id": "SEC002",
                "criticality": "low",
                "days_since_last_maintenance": 10,
                "defect_count_180d": 0,
                "failure_count_180d": 0,
                "failure_rate_365d": 0.00,
                "average_downtime_min_180d": 30,
            },
            {
                "asset_id": "A003",
                "department": "S&T",
                "asset_type": "signal",
                "section_id": "SEC003",
                "criticality": "high",
                "days_since_last_maintenance": 50,
                "defect_count_180d": 2,
                "failure_count_180d": 1,
                "failure_rate_365d": 0.20,
                "average_downtime_min_180d": 120,
            },
        ]
    )


def test_one_risk_row_per_asset() -> None:
    features = _features()
    risk = calculate_baseline_risk(features)

    assert len(risk) == len(features)
    assert risk["asset_id"].is_unique
    assert set(risk["asset_id"]) == set(features["asset_id"])


def test_scores_are_normalized() -> None:
    risk = calculate_baseline_risk(_features())

    assert risk["baseline_risk_score"].between(0, 1).all()


def test_higher_risk_profile_scores_higher() -> None:
    risk = calculate_baseline_risk(_features()).set_index("asset_id")

    assert risk.loc["A001", "baseline_risk_score"] > risk.loc[
        "A003", "baseline_risk_score"
    ]
    assert risk.loc["A003", "baseline_risk_score"] > risk.loc[
        "A002", "baseline_risk_score"
    ]


def test_risk_levels_are_valid() -> None:
    risk = calculate_baseline_risk(_features())

    assert set(risk["risk_level"]).issubset({"low", "medium", "high"})


def test_constant_numeric_feature_does_not_fail() -> None:
    features = _features()
    features["defect_count_180d"] = 1

    risk = calculate_baseline_risk(features)

    assert risk["baseline_risk_score"].between(0, 1).all()


def test_validation_passes() -> None:
    features = _features()
    risk = calculate_baseline_risk(features)

    validate_baseline_risk(risk, features)


def test_missing_required_column_is_rejected() -> None:
    features = _features().drop(columns=["failure_rate_365d"])

    with pytest.raises(ValueError, match="missing required columns"):
        calculate_baseline_risk(features)
