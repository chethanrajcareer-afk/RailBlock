"""Interpretable baseline asset-risk scoring for RailBlock Phase 2.2.

This is a transparent benchmark, not an ML model. It converts selected historical
reliability signals into a normalized 0-1 risk score so that a later ML model can
be compared against a simple, explainable baseline.

All data is synthetic and does not represent operational Indian Railways data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC_DIR = PROJECT_ROOT / "data" / "synthetic"

FEATURE_FILE = SYNTHETIC_DIR / "asset_features.csv"
OUTPUT_FILE = SYNTHETIC_DIR / "asset_risk_baseline.csv"

CRITICALITY_SCORE = {
    "low": 0.25,
    "medium": 0.50,
    "high": 0.75,
    "critical": 1.00,
}

# Transparent weights. They intentionally sum to 1.
RISK_WEIGHTS = {
    "criticality": 0.25,
    "defect_activity": 0.20,
    "failure_activity": 0.25,
    "maintenance_recency": 0.10,
    "failure_rate": 0.10,
    "downtime": 0.10,
}


def _require_columns(features: pd.DataFrame) -> None:
    required = {
        "asset_id",
        "criticality",
        "days_since_last_maintenance",
        "defect_count_180d",
        "failure_count_180d",
        "failure_rate_365d",
        "average_downtime_min_180d",
    }

    missing = required - set(features.columns)
    if missing:
        raise ValueError(
            f"asset_features.csv is missing required columns: {sorted(missing)}"
        )


def _min_max_scale(series: pd.Series) -> pd.Series:
    """Scale a numeric series to [0, 1] without failing on constant data."""
    minimum = float(series.min())
    maximum = float(series.max())

    if maximum == minimum:
        return pd.Series(
            np.zeros(len(series)),
            index=series.index,
            dtype=float,
        )

    return ((series - minimum) / (maximum - minimum)).clip(0, 1)


def calculate_baseline_risk(features: pd.DataFrame) -> pd.DataFrame:
    """Calculate one transparent baseline risk score per asset."""
    _require_columns(features)

    result = features.copy()

    result["criticality_risk"] = (
        result["criticality"]
        .map(CRITICALITY_SCORE)
        .fillna(CRITICALITY_SCORE["medium"])
    )

    result["defect_activity_risk"] = _min_max_scale(
        result["defect_count_180d"]
    )

    result["failure_activity_risk"] = _min_max_scale(
        result["failure_count_180d"]
    )

    result["maintenance_recency_risk"] = _min_max_scale(
        result["days_since_last_maintenance"]
    )

    result["failure_rate_risk"] = result["failure_rate_365d"].clip(0, 1)

    result["downtime_risk"] = _min_max_scale(
        result["average_downtime_min_180d"]
    )

    result["baseline_risk_score"] = (
        result["criticality_risk"] * RISK_WEIGHTS["criticality"]
        + result["defect_activity_risk"] * RISK_WEIGHTS["defect_activity"]
        + result["failure_activity_risk"] * RISK_WEIGHTS["failure_activity"]
        + result["maintenance_recency_risk"] * RISK_WEIGHTS["maintenance_recency"]
        + result["failure_rate_risk"] * RISK_WEIGHTS["failure_rate"]
        + result["downtime_risk"] * RISK_WEIGHTS["downtime"]
    ).clip(0, 1).round(4)

    result["risk_level"] = pd.cut(
        result["baseline_risk_score"],
        bins=[-np.inf, 0.34, 0.69, np.inf],
        labels=["low", "medium", "high"],
    ).astype(str)

    # Put the most useful decision-support columns first.
    preferred = [
        "asset_id",
        "department",
        "asset_type",
        "section_id",
        "criticality",
        "baseline_risk_score",
        "risk_level",
    ]

    remaining = [
        column
        for column in result.columns
        if column not in preferred
    ]

    return result[preferred + remaining].sort_values(
        ["baseline_risk_score", "asset_id"],
        ascending=[False, True],
    ).reset_index(drop=True)


def validate_baseline_risk(
    risk_df: pd.DataFrame,
    source_features: pd.DataFrame,
) -> None:
    """Validate baseline risk output."""
    if len(risk_df) != len(source_features):
        raise ValueError("Risk output row count does not match feature rows.")

    if not risk_df["asset_id"].is_unique:
        raise ValueError("Risk output contains duplicate asset IDs.")

    if set(risk_df["asset_id"]) != set(source_features["asset_id"]):
        raise ValueError("Risk output asset IDs do not match source features.")

    if not risk_df["baseline_risk_score"].between(0, 1).all():
        raise ValueError("Baseline risk scores are outside [0, 1].")

    if not set(risk_df["risk_level"]).issubset(
        {"low", "medium", "high"}
    ):
        raise ValueError("Risk output contains invalid risk levels.")

    if risk_df["baseline_risk_score"].isna().any():
        raise ValueError("Baseline risk output contains missing scores.")


def main() -> None:
    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Missing feature dataset: {FEATURE_FILE}"
        )

    features = pd.read_csv(FEATURE_FILE)

    risk_df = calculate_baseline_risk(features)
    validate_baseline_risk(risk_df, features)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    risk_df.to_csv(OUTPUT_FILE, index=False)

    print("RailBlock Phase 2.2 baseline risk generated")
    print(f"Assets scored          : {len(risk_df)}")
    print(f"Risk levels            :")
    print(risk_df["risk_level"].value_counts().sort_index().to_string())
    print(
        f"Risk score range       : "
        f"{risk_df['baseline_risk_score'].min():.4f} - "
        f"{risk_df['baseline_risk_score'].max():.4f}"
    )
    print(f"Output                 : {OUTPUT_FILE}")
    print("\nTop 10 highest-risk assets:")
    print(
        risk_df[
            [
                "asset_id",
                "department",
                "section_id",
                "criticality",
                "baseline_risk_score",
                "risk_level",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
