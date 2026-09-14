"""Generate RailBlock asset risk scores from the trained Random Forest.

Phase 2.3.3 converts predicted failure probabilities into interpretable
risk bands that can later be consumed by the maintenance optimizer.

All data is fictional synthetic railway data and does not represent
operational Indian Railways data or rules.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_FILE = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "risk_training_dataset.csv"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "models"
    / "risk_random_forest.joblib"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "asset_risk_scores.csv"
)


TARGET_COLUMN = "failure_in_next_90d"

CATEGORICAL_FEATURES = [
    "department",
    "asset_type",
    "section_id",
    "criticality",
]

NUMERIC_FEATURES = [
    "installation_year",
    "age_years",
    "days_since_last_maintenance",
    "has_maintenance_history",
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
    "average_severity_score_180d",
    "max_severity_score_180d",
    "total_downtime_min_365d",
    "failure_rate_365d",
    "defect_rate_365d",
    "maintenance_frequency_90d",
]

FEATURE_COLUMNS = (
    CATEGORICAL_FEATURES
    + NUMERIC_FEATURES
)


def load_inputs() -> tuple[pd.DataFrame, object]:
    """Load the latest risk dataset and trained model."""

    if not DATASET_FILE.exists():
        raise FileNotFoundError(
            f"Risk dataset not found: {DATASET_FILE}"
        )

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Random Forest model not found: {MODEL_FILE}"
        )

    dataset = pd.read_csv(DATASET_FILE)

    model = joblib.load(MODEL_FILE)

    return dataset, model


def select_latest_snapshot(
    dataset: pd.DataFrame,
) -> pd.DataFrame:
    """Return the most recent asset snapshot."""

    dataset = dataset.copy()

    dataset["snapshot_date"] = pd.to_datetime(
        dataset["snapshot_date"]
    )

    latest_snapshot = dataset[
        "snapshot_date"
    ].max()

    latest = dataset.loc[
        dataset["snapshot_date"] == latest_snapshot
    ].copy()

    if latest.empty:
        raise ValueError(
            "No rows found for the latest snapshot."
        )

    return latest


def classify_risk(
    probability: float,
) -> str:
    """Convert failure probability into a human-readable risk band."""

    if probability >= 0.75:
        return "Critical"

    if probability >= 0.50:
        return "High"

    if probability >= 0.25:
        return "Medium"

    return "Low"


def generate_risk_scores(
    dataset: pd.DataFrame,
    model: object,
) -> pd.DataFrame:
    """Generate risk probabilities and risk bands."""

    latest = select_latest_snapshot(dataset)

    X = latest[FEATURE_COLUMNS]

    probabilities = np.asarray(
    model.predict_proba(X)
)[:, 1]

    result = latest[
        [
            "asset_id",
            "department",
            "asset_type",
            "section_id",
            "criticality",
            "snapshot_date",
        ]
    ].copy()

    result[
        "predicted_failure_probability"
    ] = probabilities

    result[
        "risk_level"
    ] = result[
        "predicted_failure_probability"
    ].apply(classify_risk)

    result[
        "predicted_failure_probability"
    ] = result[
        "predicted_failure_probability"
    ].round(4)

    risk_order = {
        "Critical": 0,
        "High": 1,
        "Medium": 2,
        "Low": 3,
    }

    result["risk_rank"] = (
        result["risk_level"]
        .map(risk_order)
    )

    result = result.sort_values(
        [
            "risk_rank",
            "predicted_failure_probability",
        ],
        ascending=[
            True,
            False,
        ],
    ).reset_index(drop=True)

    result = result.drop(
        columns=["risk_rank"]
    )

    return result


def main() -> None:
    """Generate and save the RailBlock asset risk table."""

    dataset, model = load_inputs()

    risk_scores = generate_risk_scores(
        dataset,
        model,
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    risk_scores.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        "RailBlock Phase 2.3.3 risk scores generated"
    )

    print(
        f"Assets scored          : {len(risk_scores)}"
    )

    print(
        f"Snapshot               : "
        f"{risk_scores['snapshot_date'].iloc[0].date()}"
    )

    print()

    print("RISK DISTRIBUTION")
    print("----------------------------------------")

    print(
        risk_scores[
            "risk_level"
        ].value_counts()
        .reindex(
            ["Critical", "High", "Medium", "Low"],
            fill_value=0,
        )
        .to_string()
    )

    print()

    print("TOP 10 HIGHEST-RISK ASSETS")
    print("----------------------------------------")

    display_columns = [
        "asset_id",
        "department",
        "section_id",
        "criticality",
        "predicted_failure_probability",
        "risk_level",
    ]

    print(
        risk_scores[
            display_columns
        ]
        .head(10)
        .to_string(index=False)
    )

    print()

    print(
        f"Output                 : {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()