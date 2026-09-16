"""RailBlock risk-weight sensitivity analysis.

Phase 2.4.7

Runs the same risk-aware CP-SAT scheduling model under multiple risk-weight
settings while keeping the synthetic data, constraints, candidate generation,
and all non-risk objective terms unchanged.

The experiment measures whether and when the ML-derived failure probability
actually changes scheduling decisions.

All data is fictional synthetic railway data and does not represent
operational Indian Railways data or rules.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from ortools.sat.python import cp_model

from src.optimization import model_risk_aware
from src.optimization.validator import validate_schedule

from src.optimization.model_risk_aware import (
    build_risk_aware_model,
    load_asset_risk_scores,
    load_optimization_data,
)


# =========================================================
# Configuration
# =========================================================

RISK_WEIGHTS = [0, 25, 50, 100, 150, 200]

OUTPUT_DIR = (
    Path(__file__).resolve().parents[2]
    / "outputs"
)

SUMMARY_FILE = OUTPUT_DIR / "risk_weight_sensitivity.csv"
CHANGES_FILE = OUTPUT_DIR / "risk_weight_schedule_changes.csv"


# =========================================================
# Timestamp conversion
# =========================================================

def solver_minutes_to_timestamp(
    minutes: int,
) -> pd.Timestamp:
    """Convert CP-SAT epoch minutes to pandas timestamp."""

    return pd.Timestamp(
        datetime(1970, 1, 1)
        + timedelta(minutes=int(minutes))
    )


# =========================================================
# Solve one risk-weight configuration
# =========================================================

def solve_for_weight(
    data,
    risk_df: pd.DataFrame,
    risk_weight: int,
) -> dict:
    """Build and solve the unchanged model with one risk weight."""

    # The current model reads RISK_WEIGHT as a module-level constant.
    # Temporarily override it only for this experiment, then restore it.
    original_weight = model_risk_aware.RISK_WEIGHT
    model_risk_aware.RISK_WEIGHT = risk_weight

    try:
        (
            model,
            assignment_vars,
            interval_vars,
            block_used_vars,
            _block_used_minutes_vars,
        ) = build_risk_aware_model(
            data,
            risk_df,
        )
    finally:
        model_risk_aware.RISK_WEIGHT = original_weight

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 30
    solver.parameters.num_workers = 8

    status = solver.Solve(model)
    status_name = solver.StatusName(status)

    if status not in (
        cp_model.OPTIMAL,
        cp_model.FEASIBLE,
    ):
        return {
            "risk_weight": risk_weight,
            "status": status_name,
            "scheduled_task_ids": set(),
            "schedule_rows": [],
            "objective_value": None,
            "tasks_scheduled": 0,
            "tasks_total": len(data.tasks),
            "blocks_used": 0,
        }

    risk_by_asset = dict(
        zip(
            risk_df["asset_id"],
            risk_df["predicted_failure_probability"],
        )
    )

    scheduled_rows = []

    for (
        task_index,
        block_index,
    ), assignment_var in assignment_vars.items():

        if solver.Value(assignment_var) != 1:
            continue

        task = data.tasks.loc[task_index]
        block = data.blocks.loc[block_index]

        interval = interval_vars[
            (
                task_index,
                block_index,
            )
        ]

        start_minutes = solver.Value(
            interval.StartExpr()
        )
        end_minutes = solver.Value(
            interval.EndExpr()
        )

        scheduled_start = solver_minutes_to_timestamp(
            start_minutes
        )
        scheduled_end = solver_minutes_to_timestamp(
            end_minutes
        )

        asset_id = task["asset_id"]
        failure_probability = float(
            risk_by_asset.get(
                asset_id,
                0.0,
            )
        )

        scheduled_rows.append(
            {
                "task_id": task["task_id"],
                "asset_id": asset_id,
                "department": task["department"],
                "section_id": task["section_id"],
                "criticality": task["criticality"],
                "urgency": task["urgency"],
                "estimated_duration_min": int(
                    task["estimated_duration_min"]
                ),
                "block_window_id": block[
                    "block_window_id"
                ],
                "scheduled_start": scheduled_start,
                "scheduled_end": scheduled_end,
                "predicted_failure_probability": round(
                    failure_probability,
                    4,
                ),
                "risk_weight": risk_weight,
            }
        )

    schedule_df = pd.DataFrame(
        scheduled_rows
    )

    # Independently validate this schedule against the same RailBlock
    # hard-constraint checks used by the normal validation pipeline.
    validation = validate_schedule(
        data.tasks,
        data.blocks,
        data.trains,
        schedule_df,
    )

    validation_metrics = {
        "validation_status": "VALID" if validation.is_valid else "INVALID",
        "timing_violations": len(validation.timing_violations),
        "block_violations": len(validation.block_violations),
        "duration_violations": len(validation.duration_violations),
        "section_violations": len(validation.section_violations),
        "duplicate_tasks": len(validation.duplicate_tasks),
        "maintenance_conflicts": len(validation.maintenance_conflicts),
        "train_conflicts": len(validation.train_conflicts),
        "total_violations": validation.total_violations,
    }

    scheduled_ids = set(
        schedule_df["task_id"]
    ) if not schedule_df.empty else set()

    used_blocks = sum(
        solver.Value(variable)
        for variable in block_used_vars.values()
    )

    return {
        "risk_weight": risk_weight,
        "status": status_name,
        "scheduled_task_ids": scheduled_ids,
        "schedule_rows": scheduled_rows,
        "objective_value": solver.ObjectiveValue(),
        "tasks_scheduled": len(scheduled_ids),
        "tasks_total": len(data.tasks),
        "blocks_used": used_blocks,
        **validation_metrics,
    }


# =========================================================
# Risk metrics
# =========================================================

def calculate_schedule_metrics(
    result: dict,
    risk_lookup: dict,
) -> dict:
    """Calculate risk exposure metrics for one solved schedule."""

    scheduled_ids = result["scheduled_task_ids"]

    scheduled_probabilities = [
        float(risk_lookup.get(task_id, 0.0))
        for task_id in scheduled_ids
    ]

    all_task_ids = set(risk_lookup)

    unscheduled_ids = (
        all_task_ids - scheduled_ids
    )

    unscheduled_probabilities = [
        float(risk_lookup.get(task_id, 0.0))
        for task_id in unscheduled_ids
    ]

    critical_scheduled = sum(
        probability >= 0.75
        for probability in scheduled_probabilities
    )

    high_scheduled = sum(
        probability >= 0.50
        for probability in scheduled_probabilities
    )

    medium_or_higher_scheduled = sum(
        probability >= 0.25
        for probability in scheduled_probabilities
    )

    average_scheduled_risk = (
        sum(scheduled_probabilities)
        / len(scheduled_probabilities)
        if scheduled_probabilities
        else 0.0
    )

    average_unscheduled_risk = (
        sum(unscheduled_probabilities)
        / len(unscheduled_probabilities)
        if unscheduled_probabilities
        else 0.0
    )

    return {
        "critical_risk_scheduled": critical_scheduled,
        "high_risk_scheduled": high_scheduled,
        "medium_or_higher_risk_scheduled": (
            medium_or_higher_scheduled
        ),
        "average_scheduled_risk": (
            average_scheduled_risk
        ),
        "average_unscheduled_risk": (
            average_unscheduled_risk
        ),
        "scheduled_risk_sum": sum(
            scheduled_probabilities
        ),
        "unscheduled_risk_sum": sum(
            unscheduled_probabilities
        ),
    }


# =========================================================
# Main experiment
# =========================================================

def main() -> None:
    print("\n" + "=" * 80)
    print("RAILBLOCK RISK-WEIGHT SENSITIVITY ANALYSIS")
    print("=" * 80)

    data = load_optimization_data()
    risk_df = load_asset_risk_scores()

    print(
        f"\nMaintenance tasks : {len(data.tasks)}"
    )
    print(
        f"Risk scores       : {len(risk_df)}"
    )
    print(
        f"Risk weights      : {RISK_WEIGHTS}"
    )

    risk_lookup_by_asset = dict(
        zip(
            risk_df["asset_id"],
            risk_df["predicted_failure_probability"],
        )
    )

    task_risk_lookup = {
        row["task_id"]: float(
            risk_lookup_by_asset.get(
                row["asset_id"],
                0.0,
            )
        )
        for _, row in data.tasks.iterrows()
    }

    results = []

    solved_runs = {}

    for risk_weight in RISK_WEIGHTS:

        print("\n" + "-" * 80)
        print(
            f"Testing RISK_WEIGHT = {risk_weight}"
        )

        result = solve_for_weight(
            data,
            risk_df,
            risk_weight,
        )

        solved_runs[risk_weight] = result

        metrics = calculate_schedule_metrics(
            result,
            task_risk_lookup,
        )

        tasks_scheduled = result[
            "tasks_scheduled"
        ]

        tasks_total = result[
            "tasks_total"
        ]

        coverage = (
            tasks_scheduled / tasks_total * 100
            if tasks_total
            else 0.0
        )

        print(
            f"Status                 : "
            f"{result['status']}"
        )
        print(
            f"Tasks scheduled        : "
            f"{tasks_scheduled}"
        )
        print(
            f"Schedule coverage      : "
            f"{coverage:.1f}%"
        )
        print(
            f"Blocks used            : "
            f"{result['blocks_used']}"
        )
        print(
            f"Critical-risk scheduled : "
            f"{metrics['critical_risk_scheduled']}"
        )
        print(
            f"High-risk scheduled     : "
            f"{metrics['high_risk_scheduled']}"
        )
        print(
            f"Avg scheduled risk      : "
            f"{metrics['average_scheduled_risk']:.4f}"
        )
        print(
            f"Avg unscheduled risk    : "
            f"{metrics['average_unscheduled_risk']:.4f}"
        )

        print(
            f"Validation              : "
            f"{result['validation_status']} "
            f"(violations: {result['total_violations']})"
        )

        results.append(
            {
                "risk_weight": risk_weight,
                "status": result["status"],
                "tasks_scheduled": tasks_scheduled,
                "tasks_total": tasks_total,
                "schedule_coverage_pct": round(
                    coverage,
                    2,
                ),
                "blocks_used": result[
                    "blocks_used"
                ],
                "critical_risk_scheduled": metrics[
                    "critical_risk_scheduled"
                ],
                "high_risk_scheduled": metrics[
                    "high_risk_scheduled"
                ],
                "medium_or_higher_risk_scheduled": (
                    metrics[
                        "medium_or_higher_risk_scheduled"
                    ]
                ),
                "average_scheduled_risk": round(
                    metrics[
                        "average_scheduled_risk"
                    ],
                    6,
                ),
                "average_unscheduled_risk": round(
                    metrics[
                        "average_unscheduled_risk"
                    ],
                    6,
                ),
                "scheduled_risk_sum": round(
                    metrics["scheduled_risk_sum"],
                    6,
                ),
                "unscheduled_risk_sum": round(
                    metrics["unscheduled_risk_sum"],
                    6,
                ),
                "validation_status": result[
                    "validation_status"
                ],
                "timing_violations": result[
                    "timing_violations"
                ],
                "block_violations": result[
                    "block_violations"
                ],
                "duration_violations": result[
                    "duration_violations"
                ],
                "section_violations": result[
                    "section_violations"
                ],
                "duplicate_tasks": result[
                    "duplicate_tasks"
                ],
                "maintenance_conflicts": result[
                    "maintenance_conflicts"
                ],
                "train_conflicts": result[
                    "train_conflicts"
                ],
                "total_violations": result[
                    "total_violations"
                ],
                "objective_value": (
                    round(
                        result["objective_value"],
                        2,
                    )
                    if result[
                        "objective_value"
                    ] is not None
                    else None
                ),
            }
        )

    # =====================================================
    # Schedule-change analysis
    # =====================================================

    print("\n" + "=" * 80)
    print("SCHEDULE CHANGE ANALYSIS")
    print("=" * 80)

    change_rows = []

    for index in range(
        1,
        len(RISK_WEIGHTS),
    ):
        previous_weight = RISK_WEIGHTS[
            index - 1
        ]
        current_weight = RISK_WEIGHTS[
            index
        ]

        previous_ids = solved_runs[
            previous_weight
        ]["scheduled_task_ids"]

        current_ids = solved_runs[
            current_weight
        ]["scheduled_task_ids"]

        added = sorted(
            current_ids - previous_ids
        )

        removed = sorted(
            previous_ids - current_ids
        )

        common = (
            current_ids & previous_ids
        )

        print(
            f"\n{previous_weight} -> "
            f"{current_weight}"
        )
        print(
            f"Tasks added   : {len(added)}"
        )
        print(
            f"Tasks removed : {len(removed)}"
        )
        print(
            f"Tasks common  : {len(common)}"
        )

        for task_id in added:
            change_rows.append(
                {
                    "from_risk_weight": (
                        previous_weight
                    ),
                    "to_risk_weight": (
                        current_weight
                    ),
                    "change_type": "ADDED",
                    "task_id": task_id,
                    "risk_probability": round(
                        task_risk_lookup.get(
                            task_id,
                            0.0,
                        ),
                        4,
                    ),
                }
            )

        for task_id in removed:
            change_rows.append(
                {
                    "from_risk_weight": (
                        previous_weight
                    ),
                    "to_risk_weight": (
                        current_weight
                    ),
                    "change_type": "REMOVED",
                    "task_id": task_id,
                    "risk_probability": round(
                        task_risk_lookup.get(
                            task_id,
                            0.0,
                        ),
                        4,
                    ),
                }
            )

    summary_df = pd.DataFrame(results)
    changes_df = pd.DataFrame(change_rows)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_df.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    changes_df.to_csv(
        CHANGES_FILE,
        index=False,
    )

    print("\n" + "=" * 80)
    print("SENSITIVITY SUMMARY")
    print("=" * 80)

    display_columns = [
        "risk_weight",
        "tasks_scheduled",
        "schedule_coverage_pct",
        "blocks_used",
        "critical_risk_scheduled",
        "high_risk_scheduled",
        "average_scheduled_risk",
        "average_unscheduled_risk",
    ]

    print(
        summary_df[
            display_columns
        ].to_string(index=False)
    )

    print("\n" + "-" * 80)
    print("VALIDATION SUMMARY")
    print("-" * 80)
    print(
        summary_df[
            [
                "risk_weight",
                "validation_status",
                "total_violations",
            ]
        ].to_string(index=False)
    )

    invalid_runs = summary_df[
        summary_df["validation_status"] != "VALID"
    ]

    print(
        f"\nValid configurations   : "
        f"{len(summary_df) - len(invalid_runs)}/{len(summary_df)}"
    )
    print(
        f"Invalid configurations : "
        f"{len(invalid_runs)}/{len(summary_df)}"
    )

    if not invalid_runs.empty:
        raise RuntimeError(
            "Phase 2.4.7 validation failed for one or more "
            "risk-weight configurations."
        )

    print(
        f"\nSummary output : {SUMMARY_FILE}"
    )
    print(
        f"Change output  : {CHANGES_FILE}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()