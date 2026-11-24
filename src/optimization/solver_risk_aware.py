"""Solve the RailBlock risk-aware maintenance planning model.

Phase 2.4.1 uses predicted asset failure risk as an additional
maintenance-priority signal while preserving the existing scheduling
constraints.

The baseline solver.py remains untouched.

All data is fictional synthetic railway data and does not represent
operational Indian Railways data or rules.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

# Make the project root importable when this file is
# executed directly with Python.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from ortools.sat.python import cp_model

from src.optimization import model_risk_aware
from src.optimization.experiment_report import write_experiment_report
from src.optimization.model_risk_aware import (
    build_risk_aware_model,
    load_asset_risk_scores,
    load_optimization_data,
)
from src.optimization.validator import validate_schedule


# =========================================================
# Project paths
# =========================================================


OUTPUT_DIR = PROJECT_ROOT / "outputs"

OUTPUT_FILE = (
    OUTPUT_DIR
    / "risk_aware_schedule.csv"
)
REPORT_FILE = OUTPUT_DIR / "risk_aware_experiment_report.json"

MAX_SOLVE_TIME_SECONDS = 30
NUM_SOLVER_WORKERS = 8


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
# Solve
# =========================================================

def solve() -> None:
    """Build and solve the risk-aware RailBlock model."""

    print("=" * 60)
    print("RAILBLOCK RISK-AWARE OPTIMIZATION ENGINE")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Load data
    # ---------------------------------------------------------

    data = load_optimization_data()

    risk_df = load_asset_risk_scores()

    print(
        f"Maintenance tasks : {len(data.tasks)}"
    )

    print(
        f"Block windows     : {len(data.blocks)}"
    )

    print(
        f"Train movements   : {len(data.trains)}"
    )

    print(
        f"Risk scores       : {len(risk_df)}"
    )

    # ---------------------------------------------------------
    # 2. Build model
    # ---------------------------------------------------------

    print(
        "\nBuilding risk-aware optimization model..."
    )

    (
        model,
        assignment_vars,
        interval_vars,
        block_used_vars,
        block_used_minutes_vars,
    ) = build_risk_aware_model(
        data,
        risk_df,
    )

    print(
        f"Candidate assignments : "
        f"{len(assignment_vars)}"
    )

    print(
        f"Candidate intervals   : "
        f"{len(interval_vars)}"
    )

    print(
        f"Candidate blocks      : "
        f"{len(block_used_vars)}"
    )

    # ---------------------------------------------------------
    # 3. Solve
    # ---------------------------------------------------------

    print("\nSolving risk-aware model...")

    solver = cp_model.CpSolver()

    solver.parameters.max_time_in_seconds = MAX_SOLVE_TIME_SECONDS

    solver.parameters.num_workers = NUM_SOLVER_WORKERS

    status = solver.Solve(model)

    # ---------------------------------------------------------
    # 4. Solver result
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("RISK-AWARE SOLVER RESULT")
    print("=" * 60)

    print(
        "Status:",
        solver.StatusName(status),
    )

    if status not in (
        cp_model.OPTIMAL,
        cp_model.FEASIBLE,
    ):
        print(
            "No feasible risk-aware schedule "
            "was found."
        )
        empty_schedule = pd.DataFrame(
            columns=[
                "task_id",
                "asset_id",
                "department",
                "section_id",
                "task_type",
                "criticality",
                "urgency",
                "estimated_duration_min",
                "block_window_id",
                "block_date",
                "scheduled_start",
                "scheduled_end",
                "predicted_failure_probability",
            ]
        )
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        empty_schedule.to_csv(OUTPUT_FILE, index=False)

        write_experiment_report(
            project_root=PROJECT_ROOT,
            report_path=REPORT_FILE,
            input_paths={
                "maintenance_tasks": (
                    PROJECT_ROOT / "data" / "synthetic" / "maintenance_tasks.csv"
                ),
                "block_availability": (
                    PROJECT_ROOT / "data" / "synthetic" / "block_availability.csv"
                ),
                "train_movements": (
                    PROJECT_ROOT / "data" / "synthetic" / "train_movements.csv"
                ),
                "asset_risk_scores": OUTPUT_DIR / "asset_risk_scores.csv",
            },
            source_paths={
                "solver": Path(__file__),
                "model": Path(model_risk_aware.__file__),
                "report": PROJECT_ROOT / "src" / "optimization" / "experiment_report.py",
                "validator": PROJECT_ROOT / "src" / "optimization" / "validator.py",
            },
            tasks=data.tasks,
            blocks=data.blocks,
            trains=data.trains,
            risk_scores=risk_df,
            schedule=empty_schedule,
            schedule_path=OUTPUT_FILE,
            risk_weight=model_risk_aware.RISK_WEIGHT,
            max_time_seconds=MAX_SOLVE_TIME_SECONDS,
            num_workers=NUM_SOLVER_WORKERS,
            solver_status=solver.StatusName(status),
            objective_value=None,
            wall_time_seconds=solver.WallTime(),
            num_conflicts=solver.NumConflicts(),
            num_branches=solver.NumBranches(),
            candidate_assignment_count=len(assignment_vars),
            tasks_with_candidates=len(
                {task_index for task_index, _ in assignment_vars}
            ),
            blocks_used=0,
            validation=None,
        )
        print(f"Experiment report : {REPORT_FILE}")
        return

    # ---------------------------------------------------------
    # 5. Risk lookup
    # ---------------------------------------------------------

    risk_by_asset = dict(
        zip(
            risk_df["asset_id"],
            risk_df[
                "predicted_failure_probability"
            ],
        )
    )

    # ---------------------------------------------------------
    # 6. Extract selected assignments
    # ---------------------------------------------------------

    scheduled_rows = []

    for (
        task_index,
        block_index,
    ), assignment_var in assignment_vars.items():

        if solver.Value(
            assignment_var
        ) != 1:
            continue

        task = data.tasks.loc[
            task_index
        ]

        block = data.blocks.loc[
            block_index
        ]

        interval = interval_vars[
            (
                task_index,
                block_index,
            )
        ]

        # -----------------------------------------------------
        # Retrieve optimized interval
        # -----------------------------------------------------

        start_minutes = solver.Value(
            interval.StartExpr()
        )

        end_minutes = solver.Value(
            interval.EndExpr()
        )

        scheduled_start = (
            solver_minutes_to_timestamp(
                start_minutes
            )
        )

        scheduled_end = (
            solver_minutes_to_timestamp(
                end_minutes
            )
        )

        # -----------------------------------------------------
        # Asset risk
        # -----------------------------------------------------

        asset_id = task[
            "asset_id"
        ]

        failure_probability = float(
            risk_by_asset.get(
                asset_id,
                0.0,
            )
        )

        # -----------------------------------------------------
        # Build output row
        # -----------------------------------------------------

        scheduled_rows.append(
            {
                "task_id": task[
                    "task_id"
                ],

                "asset_id": asset_id,

                "department": task[
                    "department"
                ],

                "section_id": task[
                    "section_id"
                ],

                "task_type": task[
                    "task_type"
                ],

                "criticality": task[
                    "criticality"
                ],

                "urgency": task[
                    "urgency"
                ],

                "estimated_duration_min": task[
                    "estimated_duration_min"
                ],

                "block_window_id": block[
                    "block_window_id"
                ],

                "block_date": block[
                    "date"
                ],

                "scheduled_start": (
                    scheduled_start.isoformat(
                        timespec="minutes"
                    )
                ),

                "scheduled_end": (
                    scheduled_end.isoformat(
                        timespec="minutes"
                    )
                ),

                "predicted_failure_probability": (
                    round(
                        failure_probability,
                        4,
                    )
                ),
            }
        )

    # ---------------------------------------------------------
    # 7. Result dataframe
    # ---------------------------------------------------------

    result = pd.DataFrame(
        scheduled_rows
    )

    # ---------------------------------------------------------
    # 8. Save
    # ---------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # ---------------------------------------------------------
    # 9. Summary
    # ---------------------------------------------------------

    scheduled_count = len(
        result
    )

    total_tasks = len(
        data.tasks
    )

    used_blocks = sum(
        solver.Value(variable)
        for variable
        in block_used_vars.values()
    )

    validation = validate_schedule(
        data.tasks,
        data.blocks,
        data.trains,
        result,
    )

    write_experiment_report(
        project_root=PROJECT_ROOT,
        report_path=REPORT_FILE,
        input_paths={
            "maintenance_tasks": (
                PROJECT_ROOT / "data" / "synthetic" / "maintenance_tasks.csv"
            ),
            "block_availability": (
                PROJECT_ROOT / "data" / "synthetic" / "block_availability.csv"
            ),
            "train_movements": (
                PROJECT_ROOT / "data" / "synthetic" / "train_movements.csv"
            ),
            "asset_risk_scores": OUTPUT_DIR / "asset_risk_scores.csv",
        },
        source_paths={
            "solver": Path(__file__),
            "model": Path(model_risk_aware.__file__),
            "report": PROJECT_ROOT / "src" / "optimization" / "experiment_report.py",
            "validator": PROJECT_ROOT / "src" / "optimization" / "validator.py",
        },
        tasks=data.tasks,
        blocks=data.blocks,
        trains=data.trains,
        risk_scores=risk_df,
        schedule=result,
        schedule_path=OUTPUT_FILE,
        risk_weight=model_risk_aware.RISK_WEIGHT,
        max_time_seconds=MAX_SOLVE_TIME_SECONDS,
        num_workers=NUM_SOLVER_WORKERS,
        solver_status=solver.StatusName(status),
        objective_value=solver.ObjectiveValue(),
        wall_time_seconds=solver.WallTime(),
        num_conflicts=solver.NumConflicts(),
        num_branches=solver.NumBranches(),
        candidate_assignment_count=len(assignment_vars),
        tasks_with_candidates=len(
            {task_index for task_index, _ in assignment_vars}
        ),
        blocks_used=used_blocks,
        validation=validation,
    )

    print()

    print(
        "Tasks scheduled :",
        scheduled_count,
    )

    print(
        "Tasks total     :",
        total_tasks,
    )

    print(
        "Tasks unscheduled:",
        total_tasks
        - scheduled_count,
    )

    print(
        "Blocks used     :",
        used_blocks,
    )

    if total_tasks:
        coverage = (
            scheduled_count
            / total_tasks
        ) * 100

        print(
            f"Schedule coverage: "
            f"{coverage:.1f}%"
        )

    # ---------------------------------------------------------
    # 10. Risk-aware statistics
    # ---------------------------------------------------------

    if not result.empty:

        print()
        print(
            "RISK-AWARE SCHEDULE"
        )

        print(
            "-" * 40
        )

        risk_counts = (
            result[
                "predicted_failure_probability"
            ]
            .ge(0.75)
            .sum()
        )

        high_risk_count = (
            result[
                "predicted_failure_probability"
            ]
            .ge(0.50)
            .sum()
        )

        print(
            "Critical-risk assets scheduled "
            f"(>=75%): {risk_counts}"
        )

        print(
            "High-risk assets scheduled "
            f"(>=50%): {high_risk_count}"
        )

        print(
            "Average scheduled asset risk: "
            f"{result['predicted_failure_probability'].mean():.4f}"
        )

        print()
        print(
            "TOP 10 RISK-AWARE SCHEDULED ASSETS"
        )

        print(
            "-" * 40
        )

        display_columns = [
            "task_id",
            "asset_id",
            "department",
            "section_id",
            "criticality",
            "predicted_failure_probability",
            "block_window_id",
        ]

        print(
            result[
                display_columns
            ]
            .sort_values(
                "predicted_failure_probability",
                ascending=False,
            )
            .head(10)
            .to_string(
                index=False
            )
        )

    print()

    print(
        f"Output            : "
        f"{OUTPUT_FILE}"
    )

    print(
        f"Experiment report : "
        f"{REPORT_FILE}"
    )


if __name__ == "__main__":
    solve()
