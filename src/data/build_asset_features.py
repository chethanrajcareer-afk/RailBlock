"""Build ML-ready asset features for RailBlock Phase 2.1.

Features are calculated using maintenance history strictly before the reference date.
The data is fictional synthetic railway data and does not represent operational
Indian Railways data or rules.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


REFERENCE_DATE = pd.Timestamp("2026-01-05")
WINDOW_90 = 90
WINDOW_180 = 180
WINDOW_365 = 365

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC_DIR = PROJECT_ROOT / "data" / "synthetic"

ASSET_MASTER_FILE = SYNTHETIC_DIR / "asset_master.csv"
HISTORY_FILE = SYNTHETIC_DIR / "maintenance_history.csv"
FEATURE_FILE = SYNTHETIC_DIR / "asset_features.csv"

SEVERITY_SCORE = {
    "low": 1,
    "medium": 2,
    "high": 3,
    "critical": 4,
}


def _validate_inputs(
    asset_master: pd.DataFrame,
    history: pd.DataFrame,
) -> None:
    """Validate the minimum schema required for feature engineering."""
    asset_required = {
        "asset_id",
        "department",
        "asset_type",
        "section_id",
        "criticality",
        "installation_year",
        "last_major_maintenance",
        "age_years",
    }

    history_required = {
        "history_id",
        "asset_id",
        "maintenance_date",
        "maintenance_type",
        "defect_found",
        "failure_occurred",
        "downtime_min",
        "severity",
    }

    missing_asset = asset_required - set(asset_master.columns)
    missing_history = history_required - set(history.columns)

    if missing_asset:
        raise ValueError(
            f"asset_master.csv is missing required columns: {sorted(missing_asset)}"
        )

    if missing_history:
        raise ValueError(
            "maintenance_history.csv is missing required columns: "
            f"{sorted(missing_history)}"
        )


def _count_events(
    history: pd.DataFrame,
    cutoff: pd.Timestamp,
    days: int,
    column: str | None = None,
    value: int | None = None,
) -> pd.Series:
    """Count events per asset inside a backward-looking time window."""
    start = cutoff - pd.Timedelta(days=days)

    window = history[
        (history["maintenance_date"] >= start)
        & (history["maintenance_date"] < cutoff)
    ]

    if column is not None:
        window = window[window[column] == value]

    return window.groupby("asset_id").size()


def _sum_events(
    history: pd.DataFrame,
    cutoff: pd.Timestamp,
    days: int,
    column: str,
) -> pd.Series:
    """Sum a numeric event field per asset inside a time window."""
    start = cutoff - pd.Timedelta(days=days)

    window = history[
        (history["maintenance_date"] >= start)
        & (history["maintenance_date"] < cutoff)
    ]

    return window.groupby("asset_id")[column].sum()


def _mean_events(
    history: pd.DataFrame,
    cutoff: pd.Timestamp,
    days: int,
    column: str,
) -> pd.Series:
    """Calculate a mean event field per asset inside a time window."""
    start = cutoff - pd.Timedelta(days=days)

    window = history[
        (history["maintenance_date"] >= start)
        & (history["maintenance_date"] < cutoff)
    ]

    return window.groupby("asset_id")[column].mean()


def build_asset_features(
    asset_master: pd.DataFrame,
    history: pd.DataFrame,
    reference_date: pd.Timestamp = REFERENCE_DATE,
) -> pd.DataFrame:
    """Create one ML-ready feature row per asset.

    Events on or after reference_date are excluded. This prevents future
    information from leaking into the feature set.
    """
    _validate_inputs(asset_master, history)

    assets = asset_master.copy()
    events = history.copy()

    assets["last_major_maintenance"] = pd.to_datetime(
        assets["last_major_maintenance"]
    )
    events["maintenance_date"] = pd.to_datetime(events["maintenance_date"])

    # Exclude boundary/future events rather than failing. This makes the
    # feature builder safe when the history contains the prediction-date event.
    events = events[
        events["maintenance_date"] < reference_date
    ].copy()

    events = events[
        events["asset_id"].isin(assets["asset_id"])
    ].copy()

    events["severity_score"] = (
        events["severity"]
        .map(SEVERITY_SCORE)
        .fillna(0)
        .astype(float)
    )

    result = assets[
        [
            "asset_id",
            "department",
            "asset_type",
            "section_id",
            "criticality",
            "installation_year",
            "age_years",
        ]
    ].copy()

    # Recency.
    last_event = events.groupby("asset_id")["maintenance_date"].max()

    result["days_since_last_maintenance"] = (
        reference_date - result["asset_id"].map(last_event)
    ).dt.days

    result["has_maintenance_history"] = (
        result["days_since_last_maintenance"]
        .notna()
        .astype(int)
    )

    result["days_since_last_maintenance"] = (
        result["days_since_last_maintenance"]
        .fillna(WINDOW_365 + 1)
    )

    # Maintenance activity.
    for days, column in [
        (WINDOW_90, "maintenance_count_90d"),
        (WINDOW_180, "maintenance_count_180d"),
        (WINDOW_365, "maintenance_count_365d"),
    ]:
        values = (
            _count_events(events, reference_date, days)
            .reindex(result["asset_id"])
            .fillna(0)
            .astype(int)
            .to_numpy()
        )
        result[column] = values

    # Defect activity.
    for days, column in [
        (WINDOW_90, "defect_count_90d"),
        (WINDOW_180, "defect_count_180d"),
        (WINDOW_365, "defect_count_365d"),
    ]:
        values = (
            _count_events(
                events,
                reference_date,
                days,
                "defect_found",
                1,
            )
            .reindex(result["asset_id"])
            .fillna(0)
            .astype(int)
            .to_numpy()
        )
        result[column] = values

    # Failure activity.
    for days, column in [
        (WINDOW_180, "failure_count_180d"),
        (WINDOW_365, "failure_count_365d"),
    ]:
        values = (
            _count_events(
                events,
                reference_date,
                days,
                "failure_occurred",
                1,
            )
            .reindex(result["asset_id"])
            .fillna(0)
            .astype(int)
            .to_numpy()
        )
        result[column] = values

    # Downtime.
    for days, column in [
        (WINDOW_180, "average_downtime_min_180d"),
        (WINDOW_365, "average_downtime_min_365d"),
    ]:
        values = (
            _mean_events(
                events,
                reference_date,
                days,
                "downtime_min",
            )
            .reindex(result["asset_id"])
            .fillna(0)
            .round(2)
            .to_numpy()
        )
        result[column] = values

    result["total_downtime_min_365d"] = (
        _sum_events(
            events,
            reference_date,
            WINDOW_365,
            "downtime_min",
        )
        .reindex(result["asset_id"])
        .fillna(0)
        .round(2)
        .to_numpy()
    )

    # Severity.
    result["average_severity_score_180d"] = (
        _mean_events(
            events,
            reference_date,
            WINDOW_180,
            "severity_score",
        )
        .reindex(result["asset_id"])
        .fillna(0)
        .round(2)
        .to_numpy()
    )

    recent_90_start = reference_date - pd.Timedelta(days=WINDOW_90)

    recent_90 = events[
        (events["maintenance_date"] >= recent_90_start)
        & (events["maintenance_date"] < reference_date)
    ]

    max_recent_severity = recent_90.groupby("asset_id")[
        "severity_score"
    ].max()

    result["max_severity_score_90d"] = (
        max_recent_severity
        .reindex(result["asset_id"])
        .fillna(0)
        .to_numpy()
    )

    # Interpretable reliability rates.
    result["failure_rate_365d"] = np.where(
        result["maintenance_count_365d"] > 0,
        result["failure_count_365d"]
        / result["maintenance_count_365d"],
        0.0,
    ).round(4)

    result["defect_rate_365d"] = np.where(
        result["maintenance_count_365d"] > 0,
        result["defect_count_365d"]
        / result["maintenance_count_365d"],
        0.0,
    ).round(4)

    result["maintenance_frequency_90d"] = (
        result["maintenance_count_365d"]
        / (WINDOW_365 / WINDOW_90)
    ).round(2)

    return result.sort_values("asset_id").reset_index(drop=True)


def validate_features(
    features: pd.DataFrame,
    asset_master: pd.DataFrame,
) -> None:
    """Validate the generated feature table."""
    if len(features) != len(asset_master):
        raise ValueError("Feature row count does not match asset count.")

    if not features["asset_id"].is_unique:
        raise ValueError("Feature table contains duplicate asset IDs.")

    if set(features["asset_id"]) != set(asset_master["asset_id"]):
        raise ValueError(
            "Feature asset IDs do not match asset_master asset IDs."
        )

    numeric_nonnegative = [
        "age_years",
        "days_since_last_maintenance",
        "maintenance_count_90d",
        "maintenance_count_180d",
        "maintenance_count_365d",
        "defect_count_90d",
        "defect_count_180d",
        "defect_count_365d",
        "failure_count_180d",
        "failure_count_365d",
        "average_downtime_min_180d",
        "average_downtime_min_365d",
        "total_downtime_min_365d",
        "average_severity_score_180d",
        "max_severity_score_90d",
        "maintenance_frequency_90d",
    ]

    for column in numeric_nonnegative:
        if (features[column] < 0).any():
            raise ValueError(
                f"Feature {column} contains negative values."
            )

    for rate_column in [
        "failure_rate_365d",
        "defect_rate_365d",
    ]:
        if not features[rate_column].between(0, 1).all():
            raise ValueError(
                f"Feature {rate_column} is outside [0, 1]."
            )

    if features.isna().any().any():
        raise ValueError(
            "Feature table contains unexpected missing values."
        )


def main() -> None:
    if not ASSET_MASTER_FILE.exists():
        raise FileNotFoundError(
            f"Missing asset master: {ASSET_MASTER_FILE}"
        )

    if not HISTORY_FILE.exists():
        raise FileNotFoundError(
            f"Missing maintenance history: {HISTORY_FILE}"
        )

    assets = pd.read_csv(ASSET_MASTER_FILE)
    history = pd.read_csv(HISTORY_FILE)

    features = build_asset_features(
        assets,
        history,
        REFERENCE_DATE,
    )

    validate_features(features, assets)

    FEATURE_FILE.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(FEATURE_FILE, index=False)

    print("RailBlock Phase 2.1 asset features generated")
    print(f"Assets / feature rows : {len(features)}")
    print(f"Feature columns        : {len(features.columns)}")
    print(f"Reference date         : {REFERENCE_DATE.date()}")
    print(f"Output                 : {FEATURE_FILE}")
    print("\nFeature preview:")
    print(features.head().to_string(index=False))


if __name__ == "__main__":
    main()
