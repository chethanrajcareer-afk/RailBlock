"""Build a leakage-safe supervised dataset for RailBlock Phase 2.3.0.

Each row represents an asset at a historical snapshot date. Features use only
maintenance events strictly before that snapshot. The target records whether the
asset experiences a failure during the following 90 days.

All data is fictional synthetic railway data and does not represent operational
Indian Railways data or rules.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC_DIR = PROJECT_ROOT / "data" / "synthetic"

ASSET_MASTER_FILE = SYNTHETIC_DIR / "asset_master.csv"
HISTORY_FILE = SYNTHETIC_DIR / "maintenance_history.csv"
OUTPUT_FILE = SYNTHETIC_DIR / "risk_training_dataset.csv"

SNAPSHOT_DATES = pd.date_range(
    "2025-04-01",
    "2025-10-01",
    freq="MS",
)

TARGET_HORIZON_DAYS = 90

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
    asset_required = {
        "asset_id",
        "department",
        "asset_type",
        "section_id",
        "criticality",
        "installation_year",
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


def _window(
    history: pd.DataFrame,
    snapshot: pd.Timestamp,
    days: int,
) -> pd.DataFrame:
    start = snapshot - pd.Timedelta(days=days)
    return history[
        (history["maintenance_date"] >= start)
        & (history["maintenance_date"] < snapshot)
    ]


def _build_snapshot_features(
    assets: pd.DataFrame,
    history: pd.DataFrame,
    snapshot: pd.Timestamp,
) -> pd.DataFrame:
    """Build features using only information available before snapshot."""
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

    historical = history[
        history["maintenance_date"] < snapshot
    ].copy()

    historical["severity_score"] = (
        historical["severity"]
        .map(SEVERITY_SCORE)
        .fillna(0)
        .astype(float)
    )

    last_event = historical.groupby("asset_id")["maintenance_date"].max()

    result["days_since_last_maintenance"] = (
        snapshot - result["asset_id"].map(last_event)
    ).dt.days

    result["has_maintenance_history"] = (
        result["days_since_last_maintenance"].notna().astype(int)
    )

    result["days_since_last_maintenance"] = (
        result["days_since_last_maintenance"]
        .fillna(366)
    )

    for days in (90, 180, 365):
        window = _window(historical, snapshot, days)

        counts = window.groupby("asset_id").size()
        defects = window[window["defect_found"] == 1].groupby("asset_id").size()

        result[f"maintenance_count_{days}d"] = (
            counts.reindex(result["asset_id"]).fillna(0).astype(int).to_numpy()
        )
        result[f"defect_count_{days}d"] = (
            defects.reindex(result["asset_id"]).fillna(0).astype(int).to_numpy()
        )

    for days in (180, 365):
        window = _window(historical, snapshot, days)

        failures = (
            window[window["failure_occurred"] == 1]
            .groupby("asset_id")
            .size()
        )

        result[f"failure_count_{days}d"] = (
            failures.reindex(result["asset_id"]).fillna(0).astype(int).to_numpy()
        )

    window_180 = _window(historical, snapshot, 180)

    average_downtime = window_180.groupby("asset_id")["downtime_min"].mean()
    average_severity = window_180.groupby("asset_id")["severity_score"].mean()

    result["average_downtime_min_180d"] = (
        average_downtime.reindex(result["asset_id"])
        .fillna(0)
        .round(2)
        .to_numpy()
    )

    result["average_severity_score_180d"] = (
        average_severity.reindex(result["asset_id"])
        .fillna(0)
        .round(2)
        .to_numpy()
    )
    window_365 = _window(historical, snapshot, 365)

    average_downtime_365 = (
        window_365.groupby("asset_id")["downtime_min"].mean()
    )

    total_downtime_365 = (
        window_365.groupby("asset_id")["downtime_min"].sum()
    )

    result["average_downtime_min_365d"] = (
        average_downtime_365.reindex(result["asset_id"])
        .fillna(0)
        .round(2)
        .to_numpy()
    )

    result["total_downtime_min_365d"] = (
        total_downtime_365.reindex(result["asset_id"])
        .fillna(0)
        .round(2)
        .to_numpy()
    )

    result["max_severity_score_180d"] = (
        window_180.groupby("asset_id")["severity_score"]
        .max()
        .reindex(result["asset_id"])
        .fillna(0)
        .to_numpy()
    )

    result["maintenance_frequency_90d"] = (
        result["maintenance_count_90d"] / 90.0
    ).round(4)

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

    result["snapshot_date"] = snapshot

    return result


def _build_target(
    assets: pd.DataFrame,
    history: pd.DataFrame,
    snapshot: pd.Timestamp,
) -> pd.DataFrame:
    """Create the future 90-day failure target without feature leakage."""
    horizon_end = snapshot + pd.Timedelta(days=TARGET_HORIZON_DAYS)

    future = history[
        (history["maintenance_date"] >= snapshot)
        & (history["maintenance_date"] < horizon_end)
        & (history["failure_occurred"] == 1)
    ]

    failed_assets = set(future["asset_id"])

    return pd.DataFrame(
        {
            "asset_id": assets["asset_id"],
            "failure_in_next_90d": assets["asset_id"]
            .isin(failed_assets)
            .astype(int),
        }
    )


def build_risk_training_dataset(
    asset_master: pd.DataFrame,
    history: pd.DataFrame,
    snapshot_dates: pd.DatetimeIndex | list[pd.Timestamp] = SNAPSHOT_DATES,
) -> pd.DataFrame:
    """Create historical asset snapshots and future-failure labels."""
    _validate_inputs(asset_master, history)

    assets = asset_master.copy()
    events = history.copy()

    events["maintenance_date"] = pd.to_datetime(events["maintenance_date"])

    if events.empty:
        raise ValueError("Maintenance history is empty.")

    latest_history_date = events["maintenance_date"].max()

    frames: list[pd.DataFrame] = []

    for raw_snapshot in snapshot_dates:
        snapshot = pd.Timestamp(raw_snapshot).normalize()
        horizon_end = snapshot + pd.Timedelta(days=TARGET_HORIZON_DAYS)

        # A complete future horizon is mandatory for an unbiased target.
        if horizon_end > latest_history_date + pd.Timedelta(days=1):
            continue

        features = _build_snapshot_features(
            assets,
            events,
            snapshot,
        )

        target = _build_target(
            assets,
            events,
            snapshot,
        )

        frame = features.merge(
            target,
            on="asset_id",
            how="left",
            validate="one_to_one",
        )

        frames.append(frame)

    if not frames:
     return pd.DataFrame()
    dataset = pd.concat(frames, ignore_index=True)

    return dataset.sort_values(
        ["snapshot_date", "asset_id"]
    ).reset_index(drop=True)


def validate_risk_training_dataset(
    dataset: pd.DataFrame,
    asset_master: pd.DataFrame,
) -> None:
    """Validate dataset structure and basic leakage constraints."""
    required = {
        "asset_id",
        "snapshot_date",
        "failure_in_next_90d",
        "days_since_last_maintenance",
        "defect_count_180d",
        "failure_count_180d",
        "failure_rate_365d",
    }

    missing = required - set(dataset.columns)
    if missing:
        raise ValueError(
            f"Risk dataset is missing required columns: {sorted(missing)}"
        )

    if dataset.empty:
        raise ValueError("Risk dataset is empty.")

    if dataset["asset_id"].isna().any():
        raise ValueError("Risk dataset contains missing asset IDs.")

    if not dataset["asset_id"].isin(asset_master["asset_id"]).all():
        raise ValueError("Risk dataset contains unknown asset IDs.")

    if not dataset["failure_in_next_90d"].isin([0, 1]).all():
        raise ValueError("Target must contain only 0/1 values.")

    if dataset[["asset_id", "snapshot_date"]].duplicated().any():
        raise ValueError("Duplicate asset/snapshot rows found.")

    snapshot_dates = pd.to_datetime(dataset["snapshot_date"])
    if not snapshot_dates.is_monotonic_increasing:
        raise ValueError("Dataset is not sorted by snapshot_date.")

    numeric_columns = [
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
        "average_severity_score_180d",
        "failure_rate_365d",
        "defect_rate_365d",
        "average_downtime_min_365d",
        "total_downtime_min_365d",
        "max_severity_score_180d",
        "maintenance_frequency_90d",
    ]

    if dataset[numeric_columns].isna().any().any():
        raise ValueError("Feature dataset contains missing numeric values.")

    if (dataset[numeric_columns] < 0).any().any():
        raise ValueError("Feature dataset contains negative values.")

    if not dataset["failure_rate_365d"].between(0, 1).all():
        raise ValueError("Failure rate is outside [0, 1].")

    if not dataset["defect_rate_365d"].between(0, 1).all():
        raise ValueError("Defect rate is outside [0, 1].")


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

    dataset = build_risk_training_dataset(
        assets,
        history,
        SNAPSHOT_DATES,
    )

    validate_risk_training_dataset(dataset, assets)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(OUTPUT_FILE, index=False)

    print("RailBlock Phase 2.3.0 supervised dataset generated")
    print(f"Rows                   : {len(dataset)}")
    print(f"Assets                 : {dataset['asset_id'].nunique()}")
    print(
        f"Snapshots              : "
        f"{dataset['snapshot_date'].nunique()}"
    )
    print(
        f"Failure-positive rows  : "
        f"{dataset['failure_in_next_90d'].sum()}"
    )
    print(
        f"Positive rate           : "
        f"{dataset['failure_in_next_90d'].mean():.2%}"
    )
    print(
        f"Snapshot range         : "
        f"{dataset['snapshot_date'].min().date()} to "
        f"{dataset['snapshot_date'].max().date()}"
    )
    print(f"Output                 : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
