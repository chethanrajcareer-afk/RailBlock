"""RailBlock risk-aware CP-SAT optimization model.

Phase 2.4.0 adds predicted asset failure risk to the maintenance
prioritization objective while preserving the existing scheduling
constraints.

The baseline optimizer in model.py remains untouched.

All data is fictional synthetic railway data and does not represent
operational Indian Railways data or rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from ortools.sat.python import cp_model

from src.optimization.candidates import generate_candidates


# =========================================================
# Objective weights
# =========================================================

CRITICALITY_WEIGHT = {
    "critical": 100,
    "high": 70,
    "medium": 40,
    "low": 20,
}

URGENCY_WEIGHT = {
    "high": 50,
    "medium": 25,
    "low": 10,
}

BLOCK_USAGE_PENALTY = 5
UNUSED_CAPACITY_PENALTY = 1

MULTI_DEPARTMENT_BONUS = 20

# ---------------------------------------------------------
# NEW: Risk weight
# ---------------------------------------------------------
#
# A probability of:
#
#     0.84 -> 84 risk points
#     0.50 -> 50 risk points
#     0.20 -> 20 risk points
#
# This keeps the Random Forest probability on a scale
# comparable with the existing maintenance priority terms.
#

RISK_WEIGHT = 100


# =========================================================
# Optimization data container
# =========================================================

@dataclass
class OptimizationData:
    """Datasets required by the risk-aware optimizer."""

    tasks: pd.DataFrame
    blocks: pd.DataFrame
    trains: pd.DataFrame


# =========================================================
# Data loading
# =========================================================

def load_optimization_data() -> OptimizationData:
    """Load synthetic RailBlock optimization datasets."""

    project_root = Path(__file__).resolve().parents[2]

    data_dir = (
        project_root
        / "data"
        / "synthetic"
    )

    tasks_df = pd.read_csv(
        data_dir / "maintenance_tasks.csv"
    )

    blocks_df = pd.read_csv(
        data_dir / "block_availability.csv"
    )

    trains_df = pd.read_csv(
        data_dir / "train_movements.csv"
    )

    return OptimizationData(
        tasks=tasks_df,
        blocks=blocks_df,
        trains=trains_df,
    )


# =========================================================
# Risk loading
# =========================================================

def load_asset_risk_scores() -> pd.DataFrame:
    """Load the latest Random Forest asset-risk scores."""

    project_root = Path(__file__).resolve().parents[2]

    risk_file = (
        project_root
        / "outputs"
        / "asset_risk_scores.csv"
    )

    if not risk_file.exists():
        raise FileNotFoundError(
            f"Asset risk scores not found: {risk_file}"
        )

    risk_df = pd.read_csv(risk_file)

    required_columns = {
        "asset_id",
        "predicted_failure_probability",
    }

    missing = (
        required_columns
        - set(risk_df.columns)
    )

    if missing:
        raise ValueError(
            "Asset risk scores are missing required "
            f"columns: {sorted(missing)}"
        )

    risk_df = risk_df[
        [
            "asset_id",
            "predicted_failure_probability",
        ]
    ].copy()

    risk_df["predicted_failure_probability"] = (
        pd.to_numeric(
            risk_df[
                "predicted_failure_probability"
            ],
            errors="coerce",
        )
        .fillna(0.0)
        .clip(0.0, 1.0)
    )

    risk_df = risk_df.drop_duplicates(
        subset=["asset_id"],
        keep="last",
    )

    return risk_df


# =========================================================
# Optimization model
# =========================================================

def build_risk_aware_model(
    data: OptimizationData,
    risk_df: pd.DataFrame,
):
    """Build the RailBlock risk-aware CP-SAT model."""

    tasks_df = data.tasks
    blocks_df = data.blocks
    trains_df = data.trains

    model = cp_model.CpModel()

    # ---------------------------------------------------------
    # 1. Generate feasible candidates
    # ---------------------------------------------------------

    candidates = generate_candidates(
        tasks_df,
        blocks_df,
    )

    # ---------------------------------------------------------
    # 2. ID -> DataFrame index mappings
    # ---------------------------------------------------------

    task_id_to_index = {
        row["task_id"]: index
        for index, row in tasks_df.iterrows()
    }

    block_id_to_index = {
        row["block_window_id"]: index
        for index, row in blocks_df.iterrows()
    }

    tasks_by_id = {
        row["task_id"]: row
        for _, row in tasks_df.iterrows()
    }

    # ---------------------------------------------------------
    # 3. Risk lookup
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
    # 4. Decision variables
    # ---------------------------------------------------------

    assignment_vars = {}
    interval_vars = {}
    block_used_vars = {}
    block_used_minutes_vars = {}

    # ---------------------------------------------------------
    # 5. Candidate blocks
    # ---------------------------------------------------------

    candidate_block_ids = {
        candidate["block_window_id"]
        for candidate in candidates
    }

    candidate_block_indices = {
        block_id_to_index[block_id]
        for block_id in candidate_block_ids
    }

    # ---------------------------------------------------------
    # 6. Block variables
    # ---------------------------------------------------------

    for block_index in candidate_block_indices:

        block_used_vars[block_index] = (
            model.NewBoolVar(
                f"use_block_{block_index}"
            )
        )

        block = blocks_df.loc[block_index]

        block_start = pd.Timestamp(
            f"{block['date']} {block['start_time']}"
        )

        block_end = pd.Timestamp(
            f"{block['date']} {block['end_time']}"
        )

        block_duration_min = int(
            (
                block_end - block_start
            ).total_seconds()
            // 60
        )

        block_used_minutes_vars[
            block_index
        ] = model.NewIntVar(
            0,
            block_duration_min,
            f"used_minutes_{block_index}",
        )

    # ---------------------------------------------------------
    # 7. Maintenance intervals grouped by block
    # ---------------------------------------------------------

    maintenance_intervals_by_block = {
        block_index: []
        for block_index in candidate_block_indices
    }

    # ---------------------------------------------------------
    # 8. Maintenance decision variables
    # ---------------------------------------------------------

    for candidate in candidates:

        task_id = candidate["task_id"]
        block_id = candidate["block_window_id"]

        task_index = task_id_to_index[
            task_id
        ]

        block_index = block_id_to_index[
            block_id
        ]

        task = tasks_by_id[task_id]

        task_duration = int(
            candidate["task_duration_min"]
        )

        block_start = candidate["block_start"]
        block_end = candidate["block_end"]

        # -----------------------------------------------------
        # Convert timestamps to integer minutes
        # -----------------------------------------------------

        block_start_min = int(
            block_start.timestamp() // 60
        )

        block_end_min = int(
            block_end.timestamp() // 60
        )

        task_earliest = pd.Timestamp(
            task["earliest_start"]
        )

        task_deadline = pd.Timestamp(
            task["deadline"]
        )

        task_earliest_min = int(
            task_earliest.timestamp() // 60
        )

        task_deadline_min = int(
            task_deadline.timestamp() // 60
        )

        feasible_start_min = max(
            block_start_min,
            task_earliest_min,
        )

        feasible_end_min = min(
            block_end_min,
            task_deadline_min,
        )

        # -----------------------------------------------------
        # Assignment variable
        # -----------------------------------------------------

        assignment_var = model.NewBoolVar(
            f"assign_{task_id}_{block_id}"
        )

        assignment_vars[
            (task_index, block_index)
        ] = assignment_var

        # -----------------------------------------------------
        # Start/end variables
        # -----------------------------------------------------

        start_var = model.NewIntVar(
            feasible_start_min,
            feasible_end_min - task_duration,
            f"start_{task_id}_{block_id}",
        )

        end_var = model.NewIntVar(
            feasible_start_min + task_duration,
            feasible_end_min,
            f"end_{task_id}_{block_id}",
        )

        model.Add(
            end_var
            == start_var + task_duration
        ).OnlyEnforceIf(
            assignment_var
        )

        # -----------------------------------------------------
        # Optional interval
        # -----------------------------------------------------

        interval = (
            model.NewOptionalIntervalVar(
                start_var,
                task_duration,
                end_var,
                assignment_var,
                f"maintenance_{task_id}_{block_id}",
            )
        )

        interval_vars[
            (task_index, block_index)
        ] = interval

        maintenance_intervals_by_block[
            block_index
        ].append(interval)

        model.AddImplication(
            assignment_var,
            block_used_vars[
                block_index
            ],
        )

    # ---------------------------------------------------------
    # 9. Used minutes
    # ---------------------------------------------------------

    for block_index in block_used_minutes_vars:

        block_assignments = []

        for (
            task_index,
            candidate_block_index,
        ), assignment_var in assignment_vars.items():

            if candidate_block_index != block_index:
                continue

            task_duration = int(
                tasks_df.loc[
                    task_index,
                    "estimated_duration_min",
                ]
            )

            block_assignments.append(
                task_duration
                * assignment_var
            )

        if block_assignments:
            model.Add(
                block_used_minutes_vars[
                    block_index
                ]
                == sum(block_assignments)
            )
        else:
            model.Add(
                block_used_minutes_vars[
                    block_index
                ]
                == 0
            )

    # ---------------------------------------------------------
    # 10. Each task at most once
    # ---------------------------------------------------------

    for task_index in tasks_df.index:

        task_assignment_vars = [
            assignment_var
            for (
                candidate_task_index,
                _,
            ), assignment_var
            in assignment_vars.items()
            if candidate_task_index
            == task_index
        ]

        if task_assignment_vars:
            model.AddAtMostOne(
                task_assignment_vars
            )

    # ---------------------------------------------------------
    # 11. Maintenance no-overlap
    # ---------------------------------------------------------

    for (
        block_index,
        maintenance_intervals,
    ) in maintenance_intervals_by_block.items():

        if maintenance_intervals:
            model.AddNoOverlap(
                maintenance_intervals
            )

    # ---------------------------------------------------------
    # 12. Train movement conflicts
    # ---------------------------------------------------------

    for (
        block_index,
        maintenance_intervals,
    ) in maintenance_intervals_by_block.items():

        if not maintenance_intervals:
            continue

        block = blocks_df.loc[
            block_index
        ]

        block_section = block[
            "section_id"
        ]

        block_window_start = pd.Timestamp(
            f"{block['date']} {block['start_time']}"
        )

        block_window_end = pd.Timestamp(
            f"{block['date']} {block['end_time']}"
        )

        train_intervals = []

        for _, train in trains_df.iterrows():

            if train["section_id"] != block_section:
                continue

            train_entry = pd.Timestamp(
                train["entry_time"]
            )

            train_exit = pd.Timestamp(
                train["exit_time"]
            )

            if train_exit <= block_window_start:
                continue

            if train_entry >= block_window_end:
                continue

            train_start_min = int(
                train_entry.timestamp() // 60
            )

            train_end_min = int(
                train_exit.timestamp() // 60
            )

            train_interval = (
                model.NewIntervalVar(
                    train_start_min,
                    train_end_min
                    - train_start_min,
                    train_end_min,
                    f"train_"
                    f"{train['movement_id']}_"
                    f"{block_index}",
                )
            )

            train_intervals.append(
                train_interval
            )

        if train_intervals:
            model.AddNoOverlap(
                train_intervals
                + maintenance_intervals
            )

    # =========================================================
    # 13. RISK-AWARE OBJECTIVE
    # =========================================================

    objective_terms = []

    # ---------------------------------------------------------
    # Deadline reference
    # ---------------------------------------------------------

    task_deadlines = pd.to_datetime(
        tasks_df["deadline"]
    )

    latest_deadline = (
        task_deadlines.max()
    )

    DEADLINE_PRESSURE_PER_DAY = 5

    # ---------------------------------------------------------
    # Maintenance + risk priority
    # ---------------------------------------------------------

    for (
        task_index,
        block_index,
    ), assignment_var in assignment_vars.items():

        task = tasks_df.loc[
            task_index
        ]

        # -----------------------------------------------------
        # Criticality
        # -----------------------------------------------------

        criticality = str(
            task["criticality"]
        ).lower()

        criticality_score = (
            CRITICALITY_WEIGHT.get(
                criticality,
                0,
            )
        )

        # -----------------------------------------------------
        # Urgency
        # -----------------------------------------------------

        urgency = str(
            task["urgency"]
        ).lower()

        urgency_score = (
            URGENCY_WEIGHT.get(
                urgency,
                0,
            )
        )

        # -----------------------------------------------------
        # Deadline pressure
        # -----------------------------------------------------

        task_deadline = pd.Timestamp(
            task["deadline"]
        )

        days_until_reference = (
            latest_deadline
            - task_deadline
        ).days

        deadline_pressure_score = (
            max(
                0,
                days_until_reference,
            )
            * DEADLINE_PRESSURE_PER_DAY
        )

        # -----------------------------------------------------
        # Asset failure risk
        # -----------------------------------------------------

        asset_id = task["asset_id"]

        failure_probability = float(
            risk_by_asset.get(
                asset_id,
                0.0,
            )
        )

        risk_score = round(
            failure_probability
            * RISK_WEIGHT
        )

        # -----------------------------------------------------
        # Combined priority
        # -----------------------------------------------------

        priority_score = (
            criticality_score
            + urgency_score
            + deadline_pressure_score
            + risk_score
        )

        objective_terms.append(
            priority_score
            * assignment_var
        )

    # ---------------------------------------------------------
    # 14. Block usage + utilization
    # ---------------------------------------------------------

    for (
        block_index,
        block_used_var,
    ) in block_used_vars.items():

        objective_terms.append(
            -BLOCK_USAGE_PENALTY
            * block_used_var
        )

        block = blocks_df.loc[
            block_index
        ]

        block_start = pd.Timestamp(
            f"{block['date']} {block['start_time']}"
        )

        block_end = pd.Timestamp(
            f"{block['date']} {block['end_time']}"
        )

        block_duration_min = int(
            (
                block_end
                - block_start
            ).total_seconds()
            // 60
        )

        unused_capacity = (
            model.NewIntVar(
                0,
                block_duration_min,
                f"unused_capacity_"
                f"{block_index}",
            )
        )

        model.Add(
            unused_capacity
            == (
                block_duration_min
                * block_used_var
                - block_used_minutes_vars[
                    block_index
                ]
            )
        )

        objective_terms.append(
            -UNUSED_CAPACITY_PENALTY
            * unused_capacity
        )

    # ---------------------------------------------------------
    # 15. Multi-department coordination
    # ---------------------------------------------------------

    department_used_vars = {}

    for block_index in candidate_block_indices:

        departments = sorted(
            tasks_df[
                tasks_df["task_id"].isin(
                    [
                        candidate["task_id"]
                        for candidate in candidates
                        if block_id_to_index[
                            candidate[
                                "block_window_id"
                            ]
                        ]
                        == block_index
                    ]
                )
            ]["department"]
            .dropna()
            .unique()
        )

        department_vars = []

        for department in departments:

            department_assignments = []

            for (
                task_index,
                candidate_block_index,
            ), assignment_var in assignment_vars.items():

                if candidate_block_index != block_index:
                    continue

                if (
                    tasks_df.loc[
                        task_index,
                        "department",
                    ]
                    == department
                ):
                    department_assignments.append(
                        assignment_var
                    )

            if not department_assignments:
                continue

            department_var = (
                model.NewBoolVar(
                    f"dept_{department}"
                    f"_block_{block_index}"
                )
            )

            model.AddMaxEquality(
                department_var,
                department_assignments,
            )

            department_used_vars[
                (block_index, department)
            ] = department_var

            department_vars.append(
                department_var
            )

        if department_vars:

            department_count = sum(
                department_vars
            )

            multi_department_var = (
                model.NewBoolVar(
                    f"multi_department_"
                    f"block_{block_index}"
                )
            )

            model.Add(
                department_count
                <= (
                    1
                    + 2
                    * multi_department_var
                )
            )

            model.Add(
                department_count
                >= (
                    2
                    * multi_department_var
                )
            )

            objective_terms.append(
                MULTI_DEPARTMENT_BONUS
                * multi_department_var
            )

    # =========================================================
    # 16. Maximize
    # =========================================================

    model.Maximize(
        sum(objective_terms)
    )

    return (
        model,
        assignment_vars,
        interval_vars,
        block_used_vars,
        block_used_minutes_vars,
    )


if __name__ == "__main__":
    print(
        "Risk-aware model module loaded."
    )