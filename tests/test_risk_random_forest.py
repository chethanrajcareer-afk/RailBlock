"""Train a Random Forest failure-risk model for RailBlock Phase 2.3.2.

This model uses the same leakage-safe supervised dataset and temporal
evaluation strategy as the Logistic Regression baseline.

All data is fictional synthetic railway data and does not represent
operational Indian Railways data or rules.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_FILE = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "risk_training_dataset.csv"
)

MODEL_DIR = PROJECT_ROOT / "outputs" / "models"

MODEL_FILE = (
    MODEL_DIR
    / "risk_random_forest.joblib"
)

PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "risk_random_forest_predictions.csv"
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

FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def load_dataset() -> pd.DataFrame:
    """Load and validate the supervised risk dataset."""

    if not DATASET_FILE.exists():
        raise FileNotFoundError(
            f"Risk training dataset not found: {DATASET_FILE}"
        )

    dataset = pd.read_csv(DATASET_FILE)

    required_columns = set(FEATURE_COLUMNS + [TARGET_COLUMN])

    missing = sorted(
        required_columns - set(dataset.columns)
    )

    if missing:
        raise ValueError(
            "Risk training dataset is missing columns: "
            f"{missing}"
        )

    if dataset.empty:
        raise ValueError(
            "Risk training dataset is empty."
        )

    return dataset


def build_pipeline() -> Pipeline:
    """Build the Random Forest preprocessing and model pipeline."""

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                CATEGORICAL_FEATURES,
            ),
            (
                "numeric",
                "passthrough",
                NUMERIC_FEATURES,
            ),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=8,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def temporal_split(
    dataset: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use the latest snapshot as the test set."""

    dataset = dataset.copy()

    dataset["snapshot_date"] = pd.to_datetime(
        dataset["snapshot_date"]
    )

    snapshots = sorted(
        dataset["snapshot_date"].unique()
    )

    if len(snapshots) < 2:
        raise ValueError(
            "At least two snapshots are required "
            "for temporal evaluation."
        )

    test_snapshot = snapshots[-1]

    train = dataset.loc[
        dataset["snapshot_date"] < test_snapshot
    ].copy()

    test = dataset.loc[
        dataset["snapshot_date"] == test_snapshot
    ].copy()

    if train.empty or test.empty:
        raise ValueError(
            "Temporal split produced an empty train or test set."
        )

    return train, test


def evaluate_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Evaluate the model using probability-based risk metrics."""

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

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
        "roc_auc": roc_auc_score(
            y_test,
            probabilities,
        ),
        "pr_auc": average_precision_score(
            y_test,
            probabilities,
        ),
    }

    predictions_df = X_test[
        [
            "asset_id",
            "department",
            "section_id",
            "criticality",
        ]
    ].copy()

    predictions_df[
        TARGET_COLUMN
    ] = y_test.to_numpy()

    predictions_df[
        "predicted_failure_probability"
    ] = probabilities.round(6)

    predictions_df[
        "predicted_failure"
    ] = predictions

    predictions_df = predictions_df.sort_values(
        "predicted_failure_probability",
        ascending=False,
    ).reset_index(drop=True)

    return metrics, predictions_df


def main() -> None:
    """Train, evaluate, and save the Random Forest model."""

    dataset = load_dataset()

    train, test = temporal_split(dataset)

    X_train = train[
        ["asset_id"] + FEATURE_COLUMNS
    ]

    X_test = test[
        ["asset_id"] + FEATURE_COLUMNS
    ]

    y_train = train[TARGET_COLUMN]
    y_test = test[TARGET_COLUMN]

    # asset_id is an identifier, not a predictive feature.
    X_train_features = X_train[
        FEATURE_COLUMNS
    ]

    X_test_features = X_test[
        FEATURE_COLUMNS
    ]

    model = build_pipeline()

    model.fit(
        X_train_features,
        y_train,
    )

    metrics, predictions = evaluate_model(
        model,
        X_test.assign(
            asset_id=X_test["asset_id"]
        )[["asset_id"] + FEATURE_COLUMNS],
        y_test,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PREDICTIONS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_FILE,
    )

    predictions.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    print(
        "RailBlock Phase 2.3.2 Random Forest model trained"
    )

    print(
        f"Training rows         : {len(train)}"
    )

    print(
        f"Test rows             : {len(test)}"
    )

    print(
        f"Training snapshots    : "
        f"{train['snapshot_date'].min().date()} "
        f"to "
        f"{train['snapshot_date'].max().date()}"
    )

    print(
        f"Test snapshot         : "
        f"{test['snapshot_date'].iloc[0].date()}"
    )

    print(
        f"Training failures     : "
        f"{int(y_train.sum())}"
    )

    print(
        f"Test failures         : "
        f"{int(y_test.sum())}"
    )

    print()
    print("MODEL METRICS")
    print("----------------------------------------")

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

    print()
    print("CONFUSION MATRIX")
    print("----------------------------------------")

    probabilities = model.predict_proba(
        X_test_features
    )[:, 1]

    predictions_binary = (
        probabilities >= 0.5
    ).astype(int)

    print(
        confusion_matrix(
            y_test,
            predictions_binary,
        )
    )

    print()
    print("TOP 10 PREDICTED HIGH-RISK ASSETS")
    print("----------------------------------------")

    print(
        predictions.head(10).to_string(
            index=False
        )
    )

    print()
    print(
        f"Model output          : {MODEL_FILE}"
    )

    print(
        f"Predictions output    : {PREDICTIONS_FILE}"
    )


if __name__ == "__main__":
    main()