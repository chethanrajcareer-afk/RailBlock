from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from ortools.sat.python import cp_model

from src.optimization.model import (
    build_model,
    load_optimization_data,
)


# =========================================================
# Project paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_FILE = OUTPUT_DIR / "optimized_schedule.csv"


# =========================================================
# Timestamp conversion
# =========================================================

def solver_minutes_to_timestamp(minutes: int) -> pd.Timestamp:
    """
    Convert the integer-minute representation used by CP-SAT
    back into a pandas timestamp.

    The optimization model represents timestamps as minutes
    from the Unix epoch.

    Using pandas Timestamp keeps the conversion consistent
    with the timestamps used during model construction.
    """

    return pd.Timestamp(
        datetime(1970, 1, 1)
        + timedelta(minutes=int(minutes))
    )


# =========================================================
# Solver
# =========================================================

def solve():
    """Build and solve the RailBlock optimization model."""

    print("=" * 60)
    print("RAILBLOCK OPTIMIZATION ENGINE")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Load data
    # ---------------------------------------------------------

    data = load_optimization_data()

    print(f"Maintenance tasks : {len(data.tasks)}")
    print(f"Block windows     : {len(data.blocks)}")
    print(f"Train movements   : {len(data.trains)}")

    # ---------------------------------------------------------
    # 2. Build optimization model
    # ---------------------------------------------------------

    print("\nBuilding optimization model...")

    (
        model,
        assignment_vars,
        interval_vars,
        block_used_vars,
    ) = build_model(data)

    print(
        f"Candidate assignments : {len(assignment_vars)}"
    )

    print(
        f"Candidate intervals   : {len(interval_vars)}"
    )

    print(
        f"Candidate blocks      : {len(block_used_vars)}"
    )

    # ---------------------------------------------------------
    # 3. Solve
    # ---------------------------------------------------------

    print("\nSolving...")

    solver = cp_model.CpSolver()

    # Reasonable time limit for the current prototype.
    solver.parameters.max_time_in_seconds = 30

    # Allow CP-SAT to search in parallel.
    solver.parameters.num_workers = 8

    status = solver.Solve(model)

    # ---------------------------------------------------------
    # 4. Solver result
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("SOLVER RESULT")
    print("=" * 60)

    print("Status:", solver.StatusName(status))

    if status not in (
        cp_model.OPTIMAL,
        cp_model.FEASIBLE,
    ):
        print("No feasible schedule was found.")
        return

    # ---------------------------------------------------------
    # 5. Extract solution
    # ---------------------------------------------------------

    scheduled_rows = []

    for (
        task_index,
        block_index,
    ), assignment_var in assignment_vars.items():

        # Only extract selected assignments.

        if solver.Value(assignment_var) != 1:
            continue

        task = data.tasks.loc[task_index]
        block = data.blocks.loc[block_index]

        interval = interval_vars[
            (task_index, block_index)
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

        scheduled_start = solver_minutes_to_timestamp(
            start_minutes
        )

        scheduled_end = solver_minutes_to_timestamp(
            end_minutes
        )

        # -----------------------------------------------------
        # Build output row
        # -----------------------------------------------------

        scheduled_rows.append(
            {
                "task_id": task["task_id"],
                "department": task["department"],
                "section_id": task["section_id"],
                "task_type": task["task_type"],
                "criticality": task["criticality"],
                "urgency": task["urgency"],
                "estimated_duration_min": task[
                    "estimated_duration_min"
                ],
                "block_window_id": block[
                    "block_window_id"
                ],
                "block_date": block["date"],
                "scheduled_start": scheduled_start.isoformat(
                    timespec="minutes"
                ),
                "scheduled_end": scheduled_end.isoformat(
                    timespec="minutes"
                ),
            }
        )

    # ---------------------------------------------------------
    # 6. Create result dataframe
    # ---------------------------------------------------------

    result = pd.DataFrame(
        scheduled_rows
    )

    # ---------------------------------------------------------
    # 7. Save result
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
    # 8. Summary statistics
    # ---------------------------------------------------------

    scheduled_count = len(result)
    total_tasks = len(data.tasks)

    used_blocks = sum(
        solver.Value(variable)
        for variable in block_used_vars.values()
    )

    print("\nTasks scheduled :", scheduled_count)
    print("Tasks total     :", total_tasks)
    print(
        "Tasks unscheduled:",
        total_tasks - scheduled_count,
    )
    print("Blocks used     :", used_blocks)

    if total_tasks:

        percentage = (
            scheduled_count / total_tasks
        ) * 100

        print(
            f"Schedule coverage: {percentage:.1f}%"
        )

    print(
        "\nObjective value:",
        solver.ObjectiveValue(),
    )

    print("\nOutput saved to:")
    print(OUTPUT_FILE)

    # ---------------------------------------------------------
    # 9. Display first scheduled tasks
    # ---------------------------------------------------------

    if not result.empty:

        print("\nFIRST SCHEDULED TASKS")
        print("-" * 60)

        print(
            result.head(15).to_string(
                index=False
            )
        )

    # ---------------------------------------------------------
    # 10. Completion message
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("RAILBLOCK SOLVE COMPLETE")
    print("=" * 60)


# =========================================================
# Entry point
# =========================================================

if __name__ == "__main__":
    solve()