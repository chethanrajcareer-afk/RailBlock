from __future__ import annotations

import pandas as pd
import pytest

from src.prediction.generate_risk_scores import (
    classify_risk,
    generate_risk_scores,
    select_latest_snapshot,
)


def test_classify_risk_critical():
    assert classify_risk(0.75) == "Critical"
    assert classify_risk(0.95) == "Critical"


def test_classify_risk_high():
    assert classify_risk(0.50) == "High"
    assert classify_risk(0.74) == "High"


def test_classify_risk_medium():
    assert classify_risk(0.25) == "Medium"
    assert classify_risk(0.49) == "Medium"


def test_classify_risk_low():
    assert classify_risk(0.00) == "Low"
    assert classify_risk(0.24) == "Low"


def test_select_latest_snapshot():
    dataset = pd.DataFrame(
        {
            "asset_id": [
                "A1",
                "A2",
                "A1",
            ],
            "snapshot_date": [
                "2025-08-01",
                "2025-10-01",
                "2025-10-01",
            ],
        }
    )

    result = select_latest_snapshot(dataset)

    assert len(result) == 2
    assert result["snapshot_date"].min() == pd.Timestamp(
        "2025-10-01"
    )


def test_generate_risk_scores_structure():
    dataset = pd.DataFrame(
        {
            "asset_id": ["A1", "A2"],
            "department": ["Engineering", "S&T"],
            "asset_type": ["Track", "Signal"],
            "section_id": ["SEC001", "SEC002"],
            "criticality": ["high", "critical"],
            "installation_year": [2015, 2012],
            "age_years": [10, 13],
            "days_since_last_maintenance": [30, 90],
            "has_maintenance_history": [1, 1],
            "maintenance_count_90d": [2, 1],
            "maintenance_count_180d": [4, 2],
            "maintenance_count_365d": [8, 5],
            "defect_count_90d": [1, 0],
            "defect_count_180d": [2, 1],
            "defect_count_365d": [3, 2],
            "failure_count_180d": [0, 1],
            "failure_count_365d": [1, 1],
            "average_downtime_min_180d": [30.0, 45.0],
            "average_downtime_min_365d": [35.0, 50.0],
            "average_severity_score_180d": [2.0, 3.0],
            "max_severity_score_180d": [3.0, 4.0],
            "total_downtime_min_365d": [280.0, 400.0],
            "failure_rate_365d": [0.1, 0.2],
            "defect_rate_365d": [0.3, 0.4],
            "maintenance_frequency_90d": [0.0222, 0.0111],
            "snapshot_date": [
                "2025-10-01",
                "2025-10-01",
            ],
            "failure_in_next_90d": [0, 1],
        }
    )

    class DummyModel:
        def predict_proba(self, X):
            return [
                [0.10, 0.90],
                [0.80, 0.20],
            ]

    result = generate_risk_scores(
        dataset,
        DummyModel(),
    )

    expected_columns = [
        "asset_id",
        "department",
        "asset_type",
        "section_id",
        "criticality",
        "snapshot_date",
        "predicted_failure_probability",
        "risk_level",
    ]

    assert list(result.columns) == expected_columns
    assert len(result) == 2
    assert result["predicted_failure_probability"].between(
        0, 1
    ).all()


def test_generate_risk_scores_sorted_by_probability():
    dataset = pd.DataFrame(
        {
            "asset_id": ["A1", "A2"],
            "department": ["Engineering", "S&T"],
            "asset_type": ["Track", "Signal"],
            "section_id": ["SEC001", "SEC002"],
            "criticality": ["high", "critical"],
            "installation_year": [2015, 2012],
            "age_years": [10, 13],
            "days_since_last_maintenance": [30, 90],
            "has_maintenance_history": [1, 1],
            "maintenance_count_90d": [2, 1],
            "maintenance_count_180d": [4, 2],
            "maintenance_count_365d": [8, 5],
            "defect_count_90d": [1, 0],
            "defect_count_180d": [2, 1],
            "defect_count_365d": [3, 2],
            "failure_count_180d": [0, 1],
            "failure_count_365d": [1, 1],
            "average_downtime_min_180d": [30.0, 45.0],
            "average_downtime_min_365d": [35.0, 50.0],
            "average_severity_score_180d": [2.0, 3.0],
            "max_severity_score_180d": [3.0, 4.0],
            "total_downtime_min_365d": [280.0, 400.0],
            "failure_rate_365d": [0.1, 0.2],
            "defect_rate_365d": [0.3, 0.4],
            "maintenance_frequency_90d": [0.0222, 0.0111],
            "snapshot_date": [
                "2025-10-01",
                "2025-10-01",
            ],
            "failure_in_next_90d": [0, 1],
        }
    )

    class DummyModel:
        def predict_proba(self, X):
            return [
                [0.20, 0.80],
                [0.60, 0.40],
            ]

    result = generate_risk_scores(
        dataset,
        DummyModel(),
    )

    probabilities = (
        result["predicted_failure_probability"]
        .tolist()
    )

    assert probabilities == sorted(
        probabilities,
        reverse=True,
    )