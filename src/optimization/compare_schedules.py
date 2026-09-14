"""Compare RailBlock baseline and risk-aware schedules.

Phase 2.4.2 evaluates whether adding predicted asset failure risk
changes the maintenance plan in a meaningful way.

All data is fictional synthetic railway data and does not represent
operational Indian Railways data or rules.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


# =========================================================
# Project paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = PROJECT_ROOT / "outputs"

BASELINE_FILE = (
    OUTPUT_DIR / "optimized_schedule.csv"
)

RISK_AWARE_FILE = (
    OUTPUT_DIR / "risk_aware_schedule.csv"
)

RISK_FILE = (
    OUTPUT_DIR / "asset_risk_scores.csv"
)

MAINTENANCE_FILE = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "maintenance_tasks.csv"
)

COMPARISON_FILE = (
    OUTPUT_DIR / "schedule_comparison.csv"
)


# =========================================================
# File loading
# =========================================================

def load_files():
    """Load schedules, risk scores, and maintenance tasks."""

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

    baseline = pd.read_csv(
        BASELINE_FILE
    )

    risk_aware = pd.read_csv(
        RISK_AWARE_FILE
    )

    risk = pd.read_csv(
        RISK_FILE
    )

    maintenance = pd.read_csv(
        MAINTENANCE_FILE
    )

    return (
        baseline,
        risk_aware,
        risk,
        maintenance,
    )


# =========================================================
# Risk information
# =========================================================

def add_risk_information(
    schedule: pd.DataFrame,
    risk: pd.DataFrame,
    maintenance: pd.DataFrame,
) -> pd.DataFrame:
    """Attach asset and risk information to a schedule."""

    result = schedule.copy()

    # ---------------------------------------------------------
    # 1. Recover asset_id when the schedule doesn't contain it
    # ---------------------------------------------------------

    if "asset_id" not in result.columns:

        task_asset_map = maintenance[
            [
                "task_id",
                "asset_id",
            ]
        ].drop_duplicates(
            subset=["task_id"]
        )

        result = result.merge(
            task_asset_map,
            on="task_id",
            how="left",
            validate="many_to_one",
        )

    # ---------------------------------------------------------
    # 2. If risk information is already present, keep it
    # ---------------------------------------------------------

    if (
        "predicted_failure_probability"
        in result.columns
    ):

        result[
            "predicted_failure_probability"
        ] = pd.to_numeric(
            result[
                "predicted_failure_probability"
            ],
            errors="coerce",
        ).fillna(0.0)

        if "risk_level" not in result.columns:

            result["risk_level"] = (
                result[
                    "predicted_failure_probability"
                ].apply(
                    classify_risk
                )
            )

        return result

    # ---------------------------------------------------------
    # 3. Attach risk information to schedules that don't
    #    already contain it.
    # ---------------------------------------------------------

    risk_subset = risk[
        [
            "asset_id",
            "predicted_failure_probability",
            "risk_level",
        ]
    ].drop_duplicates(
        subset=["asset_id"],
        keep="last",
    )

    result = result.merge(
        risk_subset,
        on="asset_id",
        how="left",
        validate="many_to_one",
    )

    result[
        "predicted_failure_probability"
    ] = pd.to_numeric(
        result[
            "predicted_failure_probability"
        ],
        errors="coerce",
    ).fillna(0.0)

    result["risk_level"] = result[
        "risk_level"
    ].fillna(
        result[
            "predicted_failure_probability"
        ].apply(
            classify_risk
        )
    )

    return result


# =========================================================
# Risk classification
# =========================================================

def classify_risk(
    probability: float,
) -> str:
    """Convert failure probability into a risk level."""

    if probability >= 0.75:
        return "Critical"

    if probability >= 0.50:
        return "High"

    if probability >= 0.25:
        return "Medium"

    return "Low"


# =========================================================
# Metrics
# =========================================================

def calculate_metrics(
    schedule: pd.DataFrame,
    name: str,
    total_tasks: int,
) -> dict:
    """Calculate schedule-level comparison metrics."""

    scheduled_tasks = len(
        schedule
    )

    blocks_used = (
        schedule[
            "block_window_id"
        ].nunique()
        if not schedule.empty
        else 0
    )

    coverage = (
        scheduled_tasks
        / total_tasks
        * 100
        if total_tasks > 0
        else 0.0
    )

    average_risk = (
        schedule[
            "predicted_failure_probability"
        ].mean()
        if not schedule.empty
        else 0.0
    )

    critical_risk = int(
        (
            schedule[
                "predicted_failure_probability"
            ]
            >= 0.75
        ).sum()
    )

    high_risk = int(
        (
            schedule[
                "predicted_failure_probability"
            ]
            >= 0.50
        ).sum()
    )

    return {
        "schedule": name,
        "tasks_scheduled": scheduled_tasks,
        "tasks_unscheduled": (
            total_tasks
            - scheduled_tasks
        ),
        "coverage_percent": round(
            coverage,
            2,
        ),
        "blocks_used": blocks_used,
        "average_scheduled_risk": round(
            average_risk,
            4,
        ),
        "critical_risk_assets": critical_risk,
        "high_risk_assets": high_risk,
    }


# =========================================================
# Task comparison
# =========================================================

def compare_task_changes(
    baseline: pd.DataFrame,
    risk_aware: pd.DataFrame,
):
    """Identify task-level schedule changes."""

    baseline_tasks = set(
        baseline["task_id"]
    )

    risk_aware_tasks = set(
        risk_aware["task_id"]
    )

    newly_scheduled = sorted(
        risk_aware_tasks
        - baseline_tasks
    )

    removed = sorted(
        baseline_tasks
        - risk_aware_tasks
    )

    common = sorted(
        baseline_tasks
        & risk_aware_tasks
    )

    return (
        newly_scheduled,
        removed,
        common,
    )


# =========================================================
# Risk exposure
# =========================================================

def calculate_risk_exposure(
    maintenance: pd.DataFrame,
    schedule: pd.DataFrame,
):
    """Calculate average risk of scheduled and unscheduled work."""

    scheduled_task_ids = set(
        schedule["task_id"]
    )

    unscheduled = maintenance[
        ~maintenance["task_id"].isin(
            scheduled_task_ids
        )
    ].copy()

    scheduled = maintenance[
        maintenance["task_id"].isin(
            scheduled_task_ids
        )
    ].copy()

    return (
        scheduled,
        unscheduled,
    )


# =========================================================
# Main comparison
# =========================================================

def main() -> None:
    """Run the baseline vs risk-aware comparison."""

    print("=" * 65)
    print(
        "RAILBLOCK BASELINE vs RISK-AWARE"
    )
    print("=" * 65)

    (
        baseline,
        risk_aware,
        risk,
        maintenance,
    ) = load_files()

    total_tasks = len(
        maintenance
    )

    # ---------------------------------------------------------
    # Add risk information
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # Calculate metrics
    # ---------------------------------------------------------

    baseline_metrics = calculate_metrics(
        baseline,
        "Baseline",
        total_tasks,
    )

    risk_aware_metrics = calculate_metrics(
        risk_aware,
        "Risk-aware",
        total_tasks,
    )

    comparison = pd.DataFrame(
        [
            baseline_metrics,
            risk_aware_metrics,
        ]
    )

    comparison.to_csv(
        COMPARISON_FILE,
        index=False,
    )

    # ---------------------------------------------------------
    # Print metrics
    # ---------------------------------------------------------

    print()
    print(
        f"{'METRIC':<35}"
        f"{'BASELINE':>15}"
        f"{'RISK-AWARE':>15}"
    )

    print("-" * 65)

    print(
        f"{'Tasks scheduled':<35}"
        f"{baseline_metrics['tasks_scheduled']:>15}"
        f"{risk_aware_metrics['tasks_scheduled']:>15}"
    )

    print(
        f"{'Tasks unscheduled':<35}"
        f"{baseline_metrics['tasks_unscheduled']:>15}"
        f"{risk_aware_metrics['tasks_unscheduled']:>15}"
    )

    print(
        f"{'Coverage (%)':<35}"
        f"{baseline_metrics['coverage_percent']:>15.2f}"
        f"{risk_aware_metrics['coverage_percent']:>15.2f}"
    )

    print(
        f"{'Blocks used':<35}"
        f"{baseline_metrics['blocks_used']:>15}"
        f"{risk_aware_metrics['blocks_used']:>15}"
    )

    print(
        f"{'Average scheduled risk':<35}"
        f"{baseline_metrics['average_scheduled_risk']:>15.4f}"
        f"{risk_aware_metrics['average_scheduled_risk']:>15.4f}"
    )

    print(
        f"{'Critical-risk assets (>=75%)':<35}"
        f"{baseline_metrics['critical_risk_assets']:>15}"
        f"{risk_aware_metrics['critical_risk_assets']:>15}"
    )

    print(
        f"{'High-risk assets (>=50%)':<35}"
        f"{baseline_metrics['high_risk_assets']:>15}"
        f"{risk_aware_metrics['high_risk_assets']:>15}"
    )

    # ---------------------------------------------------------
    # Schedule changes
    # ---------------------------------------------------------

    (
        newly_scheduled,
        removed,
        common,
    ) = compare_task_changes(
        baseline,
        risk_aware,
    )

    print()
    print("=" * 65)
    print("SCHEDULE CHANGES")
    print("=" * 65)

    print(
        f"\nCommon tasks           : "
        f"{len(common)}"
    )

    print(
        f"Newly scheduled       : "
        f"{len(newly_scheduled)}"
    )

    print(
        f"Removed by risk-aware : "
        f"{len(removed)}"
    )

    # ---------------------------------------------------------
    # Newly scheduled tasks
    # ---------------------------------------------------------

    if newly_scheduled:

        print()
        print(
            "NEWLY SCHEDULED BY RISK-AWARE"
        )

        print("-" * 65)

        new_rows = (
            risk_aware[
                risk_aware["task_id"].isin(
                    newly_scheduled
                )
            ]
            .sort_values(
                "predicted_failure_probability",
                ascending=False,
            )
        )

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

        available_columns = [
            column
            for column in columns
            if column in new_rows.columns
        ]

        print(
            new_rows[
                available_columns
            ].to_string(
                index=False
            )
        )

    # ---------------------------------------------------------
    # Removed tasks
    # ---------------------------------------------------------

    if removed:

        print()
        print(
            "REMOVED BY RISK-AWARE"
        )

        print("-" * 65)

        removed_rows = (
            baseline[
                baseline["task_id"].isin(
                    removed
                )
            ]
            .sort_values(
                "predicted_failure_probability",
                ascending=False,
            )
        )

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

        available_columns = [
            column
            for column in columns
            if column in removed_rows.columns
        ]

        print(
            removed_rows[
                available_columns
            ].to_string(
                index=False
            )
        )

    # ---------------------------------------------------------
    # Risk exposure comparison
    # ---------------------------------------------------------

    print()
    print("=" * 65)
    print("RISK EXPOSURE")
    print("=" * 65)

    baseline_scheduled, baseline_unscheduled = (
        calculate_risk_exposure(
            maintenance,
            baseline,
        )
    )

    risk_aware_scheduled, risk_aware_unscheduled = (
        calculate_risk_exposure(
            maintenance,
            risk_aware,
        )
    )

    # Attach risk to complete maintenance task dataset
    maintenance_with_risk = maintenance.merge(
        risk[
            [
                "asset_id",
                "predicted_failure_probability",
            ]
        ].drop_duplicates(
            subset=["asset_id"]
        ),
        on="asset_id",
        how="left",
        validate="many_to_one",
    )

    maintenance_with_risk[
        "predicted_failure_probability"
    ] = pd.to_numeric(
        maintenance_with_risk[
            "predicted_failure_probability"
        ],
        errors="coerce",
    ).fillna(0.0)

    baseline_scheduled = (
        maintenance_with_risk[
            maintenance_with_risk["task_id"].isin(
                set(baseline["task_id"])
            )
        ]
    )

    baseline_unscheduled = (
        maintenance_with_risk[
            ~maintenance_with_risk["task_id"].isin(
                set(baseline["task_id"])
            )
        ]
    )

    risk_aware_scheduled = (
        maintenance_with_risk[
            maintenance_with_risk["task_id"].isin(
                set(risk_aware["task_id"])
            )
        ]
    )

    risk_aware_unscheduled = (
        maintenance_with_risk[
            ~maintenance_with_risk["task_id"].isin(
                set(risk_aware["task_id"])
            )
        ]
    )

    baseline_scheduled_avg = (
        baseline_scheduled[
            "predicted_failure_probability"
        ].mean()
    )

    baseline_unscheduled_avg = (
        baseline_unscheduled[
            "predicted_failure_probability"
        ].mean()
    )

    risk_aware_scheduled_avg = (
        risk_aware_scheduled[
            "predicted_failure_probability"
        ].mean()
    )

    risk_aware_unscheduled_avg = (
        risk_aware_unscheduled[
            "predicted_failure_probability"
        ].mean()
    )

    print(
        "Baseline scheduled avg risk   : "
        f"{baseline_scheduled_avg:.4f}"
    )

    print(
        "Baseline unscheduled avg risk : "
        f"{baseline_unscheduled_avg:.4f}"
    )

    print(
        "Risk-aware scheduled avg risk : "
        f"{risk_aware_scheduled_avg:.4f}"
    )

    print(
        "Risk-aware unscheduled avg risk: "
        f"{risk_aware_unscheduled_avg:.4f}"
    )

    # ---------------------------------------------------------
    # Risk exposure reduction
    # ---------------------------------------------------------

    risk_reduction = (
        baseline_unscheduled_avg
        - risk_aware_unscheduled_avg
    )

    print(
        "Unscheduled risk reduction    : "
        f"{risk_reduction:+.4f}"
    )

    # ---------------------------------------------------------
    # Output
    # ---------------------------------------------------------

    print()
    print(
        f"Comparison output: "
        f"{COMPARISON_FILE}"
    )


if __name__ == "__main__":
    main()