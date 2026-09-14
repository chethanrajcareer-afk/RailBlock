from __future__ import annotations

from pathlib import Path

import pandas as pd


# =========================================================
# Project paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs"


BASELINE_FILE = OUTPUT_DIR / "optimized_schedule.csv"
RISK_AWARE_FILE = OUTPUT_DIR / "risk_aware_schedule.csv"
RISK_FILE = OUTPUT_DIR / "asset_risk_scores.csv"
MAINTENANCE_FILE = (
    PROJECT_ROOT / "data" / "synthetic" / "maintenance_tasks.csv"
)


# =========================================================
# Risk classification
# =========================================================

def classify_risk(probability: float) -> str:
    """Convert failure probability into a readable risk band."""

    if probability >= 0.75:
        return "Critical"

    if probability >= 0.50:
        return "High"

    if probability >= 0.25:
        return "Medium"

    return "Low"


# =========================================================
# Data loading
# =========================================================

def load_data():
    """Load schedules, risk scores and maintenance data."""

    required_files = [
        BASELINE_FILE,
        RISK_AWARE_FILE,
        RISK_FILE,
        MAINTENANCE_FILE,
    ]

    for file_path in required_files:

        if not file_path.exists():
            raise FileNotFoundError(
                f"Required file not found: {file_path}"
            )

    baseline = pd.read_csv(BASELINE_FILE)
    risk_aware = pd.read_csv(RISK_AWARE_FILE)
    risk = pd.read_csv(RISK_FILE)
    maintenance = pd.read_csv(MAINTENANCE_FILE)

    return baseline, risk_aware, risk, maintenance


# =========================================================
# Add risk information
# =========================================================

def add_risk_information(
    schedule: pd.DataFrame,
    risk: pd.DataFrame,
    maintenance: pd.DataFrame,
) -> pd.DataFrame:
    """Ensure schedule rows contain asset and risk information."""

    result = schedule.copy()

    # -----------------------------------------------------
    # Recover asset_id from maintenance data when required
    # -----------------------------------------------------

    if "asset_id" not in result.columns:

        asset_lookup = maintenance[
            ["task_id", "asset_id"]
        ].drop_duplicates("task_id")

        result = result.merge(
            asset_lookup,
            on="task_id",
            how="left",
        )

    # -----------------------------------------------------
    # Risk information may already exist in the schedule
    # -----------------------------------------------------

    if "predicted_failure_probability" in result.columns:

        result["predicted_failure_probability"] = pd.to_numeric(
            result["predicted_failure_probability"],
            errors="coerce",
        )

        if "risk_level" not in result.columns:

            result["risk_level"] = result[
                "predicted_failure_probability"
            ].apply(classify_risk)

        return result

    # -----------------------------------------------------
    # Otherwise merge risk scores
    # -----------------------------------------------------

    risk_columns = [
        "asset_id",
        "predicted_failure_probability",
    ]

    available_columns = [
        column
        for column in risk_columns
        if column in risk.columns
    ]

    result = result.merge(
        risk[available_columns].drop_duplicates("asset_id"),
        on="asset_id",
        how="left",
    )

    result["predicted_failure_probability"] = pd.to_numeric(
        result["predicted_failure_probability"],
        errors="coerce",
    )

    result["risk_level"] = result[
        "predicted_failure_probability"
    ].apply(classify_risk)

    return result


# =========================================================
# Task-level decision analysis
# =========================================================

def compare_task_decisions(
    baseline: pd.DataFrame,
    risk_aware: pd.DataFrame,
) -> pd.DataFrame:
    """Compare task scheduling decisions between models."""

    baseline_ids = set(baseline["task_id"])
    risk_aware_ids = set(risk_aware["task_id"])

    all_task_ids = sorted(
        baseline_ids | risk_aware_ids
    )

    rows = []

    for task_id in all_task_ids:

        if task_id in baseline_ids and task_id in risk_aware_ids:
            decision = "Both scheduled"

        elif task_id in risk_aware_ids:
            decision = "Risk-aware only"

        else:
            decision = "Baseline only"

        rows.append(
            {
                "task_id": task_id,
                "decision": decision,
            }
        )

    return pd.DataFrame(rows)


# =========================================================
# Risk-aware decision report
# =========================================================

def print_decision_report(
    baseline: pd.DataFrame,
    risk_aware: pd.DataFrame,
):
    """Print risk-focused scheduling analysis."""

    print("\n" + "=" * 65)
    print("RAILBLOCK RISK DECISION ANALYSIS")
    print("=" * 65)

    # -----------------------------------------------------
    # Overall decision changes
    # -----------------------------------------------------

    baseline_ids = set(baseline["task_id"])
    risk_aware_ids = set(risk_aware["task_id"])

    common = baseline_ids & risk_aware_ids
    risk_aware_only = risk_aware_ids - baseline_ids
    baseline_only = baseline_ids - risk_aware_ids

    print("\nSCHEDULE DECISIONS")
    print("-" * 65)

    print(
        f"Common scheduled tasks       : {len(common)}"
    )

    print(
        f"Risk-aware only              : {len(risk_aware_only)}"
    )

    print(
        f"Baseline only                : {len(baseline_only)}"
    )

    # -----------------------------------------------------
    # Risk-aware-only tasks
    # -----------------------------------------------------

    print("\nRISK-AWARE ONLY TASKS")
    print("-" * 65)

    if risk_aware_only:

        columns = [
            "task_id",
            "asset_id",
            "department",
            "section_id",
            "criticality",
            "predicted_failure_probability",
            "risk_level",
            "block_window_id",
        ]

        available = [
            column
            for column in columns
            if column in risk_aware.columns
        ]

        print(
            risk_aware[
                risk_aware["task_id"].isin(
                    risk_aware_only
                )
            ][available].to_string(index=False)
        )

    else:
        print("None")

    # -----------------------------------------------------
    # Baseline-only tasks
    # -----------------------------------------------------

    print("\nBASELINE ONLY TASKS")
    print("-" * 65)

    if baseline_only:

        columns = [
            "task_id",
            "asset_id",
            "department",
            "section_id",
            "criticality",
            "predicted_failure_probability",
            "risk_level",
            "block_window_id",
        ]

        available = [
            column
            for column in columns
            if column in baseline.columns
        ]

        print(
            baseline[
                baseline["task_id"].isin(
                    baseline_only
                )
            ][available].to_string(index=False)
        )

    else:
        print("None")

    # -----------------------------------------------------
    # Risk distribution of scheduled work
    # -----------------------------------------------------

    print("\nSCHEDULED RISK DISTRIBUTION")
    print("-" * 65)

    print("Baseline:")

    baseline_counts = (
        baseline["risk_level"]
        .value_counts()
        .reindex(
            ["Critical", "High", "Medium", "Low"],
            fill_value=0,
        )
    )

    print(baseline_counts.to_string())

    print("\nRisk-aware:")

    risk_counts = (
        risk_aware["risk_level"]
        .value_counts()
        .reindex(
            ["Critical", "High", "Medium", "Low"],
            fill_value=0,
        )
    )

    print(risk_counts.to_string())

    # -----------------------------------------------------
    # High-risk scheduled work
    # -----------------------------------------------------

    high_risk = risk_aware[
        risk_aware["predicted_failure_probability"] >= 0.50
    ].copy()

    print("\nHIGH-RISK ASSETS IN RISK-AWARE SCHEDULE")
    print("-" * 65)

    if high_risk.empty:
        print("None")
    else:
        columns = [
            "task_id",
            "asset_id",
            "department",
            "section_id",
            "criticality",
            "predicted_failure_probability",
            "risk_level",
            "block_window_id",
        ]

        available = [
            column
            for column in columns
            if column in high_risk.columns
        ]

        print(
            high_risk[available]
            .sort_values(
                "predicted_failure_probability",
                ascending=False,
            )
            .to_string(index=False)
        )

    # -----------------------------------------------------
    # Highest-risk scheduled tasks
    # -----------------------------------------------------

    print("\nTOP SCHEDULED RISKS")
    print("-" * 65)

    top_risk = (
        risk_aware
        .sort_values(
            "predicted_failure_probability",
            ascending=False,
        )
        .head(10)
    )

    columns = [
        "task_id",
        "asset_id",
        "predicted_failure_probability",
        "risk_level",
        "criticality",
        "department",
        "section_id",
    ]

    available = [
        column
        for column in columns
        if column in top_risk.columns
    ]

    print(
        top_risk[available].to_string(index=False)
    )

    print("\n" + "=" * 65)


# =========================================================
# Main
# =========================================================

def main():

    baseline, risk_aware, risk, maintenance = load_data()

    baseline = add_risk_information(
        baseline,
        risk,
        maintenance,
    )

    risk_aware = add_risk_information(
        risk_aware,
        risk,
        maintenance,
    )

    print_decision_report(
        baseline,
        risk_aware,
    )

    # -----------------------------------------------------
    # Save task-level comparison
    # -----------------------------------------------------

    comparison = compare_task_decisions(
        baseline,
        risk_aware,
    )

    comparison = comparison.merge(
        risk_aware[
            [
                "task_id",
                "asset_id",
                "predicted_failure_probability",
                "risk_level",
            ]
        ],
        on="task_id",
        how="left",
    )

    output_file = (
        OUTPUT_DIR / "risk_decision_analysis.csv"
    )

    comparison.to_csv(
        output_file,
        index=False,
    )

    print(
        f"\nDecision analysis output: {output_file}"
    )


if __name__ == "__main__":
    main()