"""Train the first ML risk model for RailBlock Phase 2.3.1.

This module trains a leakage-safe supervised model for predicting whether a
railway asset will experience a failure within the next 90 days.

The training data is fictional synthetic railway data and does not represent
operational Indian Railways data or rules.

Model strategy:
    1. Use historical asset snapshots.
    2. Split chronologically by snapshot date.
    3. Encode categorical asset attributes.
    4. Scale numeric features.
    5. Train class-balanced Logistic Regression.
    6. Evaluate using recall, precision, F1, ROC-AUC and PR-AUC.

The latest snapshot is reserved as the test set so future observations are not
used to train the model.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "risk_training_dataset.csv"
)

MODEL_DIR = PROJECT_ROOT / "outputs" / "models"

MODEL_FILE = MODEL_DIR / "risk_logistic_regression.joblib"

PREDICTION_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "risk_model_predictions.csv"
)


TARGET_COLUMN = "failure_in_next_90d"
SNAPSHOT_COLUMN = "snapshot_date"
ID_COLUMN = "asset_id"


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
    "total_downtime_min_365d",
    "average_severity_score_180d",
    "max_severity_score_180d",
    "failure_rate_365d",
    "defect_rate_365d",
    "maintenance_frequency_90d",
]


FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def load_training_dataset() -> pd.DataFrame:
    """Load and validate the supervised risk dataset."""

    if not DATA_FILE.is_file():
        raise FileNotFoundError(
            f"Risk training dataset not found: {DATA_FILE}"
        )

    dataset = pd.read_csv(DATA_FILE)

    required = {
        ID_COLUMN,
        SNAPSHOT_COLUMN,
        TARGET_COLUMN,
        *FEATURE_COLUMNS,
    }

    missing = required - set(dataset.columns)

    if missing:
        raise ValueError(
            f"Risk training dataset is missing columns: "
            f"{sorted(missing)}"
        )

    dataset[SNAPSHOT_COLUMN] = pd.to_datetime(
        dataset[SNAPSHOT_COLUMN]
    )

    if dataset.empty:
        raise ValueError("Risk training dataset is empty.")

    if not dataset[TARGET_COLUMN].isin([0, 1]).all():
        raise ValueError(
            "Target must contain only 0/1 values."
        )

    if dataset[FEATURE_COLUMNS].isna().any().any():
        raise ValueError(
            "Training features contain missing values."
        )

    return dataset.sort_values(
        [SNAPSHOT_COLUMN, ID_COLUMN]
    ).reset_index(drop=True)


def temporal_train_test_split(
    dataset: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data chronologically using the latest snapshot as test data."""

    snapshots = sorted(
        dataset[SNAPSHOT_COLUMN].unique()
    )

    if len(snapshots) < 2:
        raise ValueError(
            "At least two snapshot dates are required "
            "for a temporal train/test split."
        )

    test_snapshot = snapshots[-1]

    train = dataset[
        dataset[SNAPSHOT_COLUMN] < test_snapshot
    ].copy()

    test = dataset[
        dataset[SNAPSHOT_COLUMN] == test_snapshot
    ].copy()

    if train.empty:
        raise ValueError("Temporal training split is empty.")

    if test.empty:
        raise ValueError("Temporal test split is empty.")

    if train[TARGET_COLUMN].nunique() < 2:
        raise ValueError(
            "Training split contains only one target class."
        )

    return train, test


def build_model() -> Pipeline:
    """Build the preprocessing + Logistic Regression pipeline."""

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )

    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        solver="liblinear",
        random_state=42,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def evaluate_model(
    model: Pipeline,
    test: pd.DataFrame,
) -> dict[str, float]:
    """Evaluate the model on the held-out latest snapshot."""

    x_test = test[FEATURE_COLUMNS]
    y_test = test[TARGET_COLUMN]

    probabilities = model.predict_proba(x_test)[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    metrics = {
        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_test,
            predictions,
            zero_division=0,
        ),
    }

    if y_test.nunique() == 2:
        metrics["roc_auc"] = roc_auc_score(
            y_test,
            probabilities,
        )

        metrics["pr_auc"] = average_precision_score(
            y_test,
            probabilities,
        )
    else:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")

    return metrics


def save_predictions(
    model: Pipeline,
    test: pd.DataFrame,
) -> pd.DataFrame:
    """Generate and save predictions for the test snapshot."""

    result = test[
        [
            ID_COLUMN,
            SNAPSHOT_COLUMN,
            TARGET_COLUMN,
            "department",
            "asset_type",
            "section_id",
            "criticality",
        ]
    ].copy()

    result["predicted_failure_probability"] = (
        model.predict_proba(
            test[FEATURE_COLUMNS]
        )[:, 1]
    )

    result["predicted_failure"] = (
        result["predicted_failure_probability"] >= 0.5
    ).astype(int)

    result = result.sort_values(
        "predicted_failure_probability",
        ascending=False,
    ).reset_index(drop=True)

    PREDICTION_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        PREDICTION_FILE,
        index=False,
    )

    return result


def main() -> None:
    """Train, evaluate and save the RailBlock risk model."""

    dataset = load_training_dataset()

    train, test = temporal_train_test_split(
        dataset
    )

    model = build_model()

    x_train = train[FEATURE_COLUMNS]
    y_train = train[TARGET_COLUMN]

    model.fit(
        x_train,
        y_train,
    )

    metrics = evaluate_model(
        model,
        test,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_FILE,
    )

    predictions = save_predictions(
        model,
        test,
    )

    print(
        "RailBlock Phase 2.3.1 Logistic Regression model trained"
    )

    print(
        f"Training rows         : {len(train)}"
    )

    print(
        f"Test rows             : {len(test)}"
    )

    print(
        f"Training snapshots    : "
        f"{train[SNAPSHOT_COLUMN].min().date()} "
        f"to "
        f"{train[SNAPSHOT_COLUMN].max().date()}"
    )

    print(
        f"Test snapshot         : "
        f"{test[SNAPSHOT_COLUMN].iloc[0].date()}"
    )

    print(
        f"Training failures     : "
        f"{int(y_train.sum())}"
    )

    print(
        f"Test failures         : "
        f"{int(test[TARGET_COLUMN].sum())}"
    )

    print()
    print("MODEL METRICS")
    print("-" * 40)

    print(
        f"Precision             : "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Recall                : "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"F1                    : "
        f"{metrics['f1']:.4f}"
    )

    print(
        f"ROC-AUC               : "
        f"{metrics['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC                : "
        f"{metrics['pr_auc']:.4f}"
    )

    matrix = confusion_matrix(
        test[TARGET_COLUMN],
        predictions["predicted_failure"],
    )

    print()
    print("CONFUSION MATRIX")
    print("-" * 40)
    print(matrix)

    print()
    print("TOP 10 PREDICTED HIGH-RISK ASSETS")
    print("-" * 40)

    print(
        predictions[
            [
                ID_COLUMN,
                "department",
                "section_id",
                "criticality",
                TARGET_COLUMN,
                "predicted_failure_probability",
                "predicted_failure",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    print()
    print(f"Model output          : {MODEL_FILE}")
    print(f"Predictions output    : {PREDICTION_FILE}")


if __name__ == "__main__":
    main()