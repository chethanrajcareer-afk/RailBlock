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


# =========================================================
# Optimization data container
# =========================================================

@dataclass
class OptimizationData:
    """
    Container for the datasets required by RailBlock.

    Attributes:
        tasks:
            Maintenance task dataframe.

        blocks:
            Maintenance block availability dataframe.

        trains:
            Train movement dataframe.
    """

    tasks: pd.DataFrame
    blocks: pd.DataFrame
    trains: pd.DataFrame


# =========================================================
# Data loading
# =========================================================

def load_optimization_data():
    """
    Load the synthetic datasets required by the RailBlock
    optimization engine.

    Returns:
        OptimizationData:
            tasks  -> maintenance task dataframe
            blocks -> maintenance block availability dataframe
            trains -> train movement dataframe
    """

    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data" / "synthetic"

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
# Optimization model
# =========================================================

def build_model(data):
    """
    Build the RailBlock CP-SAT optimization model.

    The model determines which maintenance tasks should be
    assigned to which available maintenance blocks.

    Constraints:
        1. A task can be scheduled at most once.
        2. A task can only use a feasible maintenance block.
        3. Maintenance tasks within the same block cannot overlap.
        4. Maintenance cannot overlap with train movements.
        5. Maintenance must fit completely inside its block.
        6. Maintenance must respect the task's earliest start.
        7. Maintenance must finish before the task deadline.
        8. A block is marked as used when a task is assigned.

    Objective:
        Maximize maintenance priority based on criticality
        and urgency while penalizing unnecessary block usage.

    Returns:
        tuple:
            model
            assignment_vars
            interval_vars
            block_used_vars

    Notes:
        Decision-variable dictionaries use DataFrame integer
        indices because solver.py retrieves selected rows using:

            data.tasks.loc[task_index]
            data.blocks.loc[block_index]
    """

    # ---------------------------------------------------------
    # 1. Extract datasets
    # ---------------------------------------------------------

    tasks_df = data.tasks
    blocks_df = data.blocks
    trains_df = data.trains

    # ---------------------------------------------------------
    # 2. Create CP-SAT model
    # ---------------------------------------------------------

    model = cp_model.CpModel()

    # ---------------------------------------------------------
    # 3. Generate feasible task -> block candidates
    # ---------------------------------------------------------

    # candidates.py is the single source of truth for
    # basic task/block feasibility.

    candidates = generate_candidates(
        tasks_df,
        blocks_df,
    )

    # ---------------------------------------------------------
    # 4. Create ID -> DataFrame index mappings
    # ---------------------------------------------------------

    # candidates.py works with meaningful IDs such as:
    #
    #   MT0001
    #   BW00351
    #
    # solver.py works with DataFrame indices.
    #
    # Therefore we maintain both representations.

    task_id_to_index = {
        row["task_id"]: index
        for index, row in tasks_df.iterrows()
    }

    block_id_to_index = {
        row["block_window_id"]: index
        for index, row in blocks_df.iterrows()
    }

    # Lookup dictionaries using IDs.

    tasks_by_id = {
        row["task_id"]: row
        for _, row in tasks_df.iterrows()
    }

    # ---------------------------------------------------------
    # 5. Decision variable containers
    # ---------------------------------------------------------

    assignment_vars = {}
    interval_vars = {}
    block_used_vars = {}
    block_used_minutes_vars = {}
    department_used_vars = {}
    multi_department_block_vars = {}

    # ---------------------------------------------------------
    # 6. Identify blocks with feasible candidates
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
    # 7. Create block usage variables
    # ---------------------------------------------------------

    for block_index in candidate_block_indices:

        block_used_vars[block_index] = model.NewBoolVar(
            f"use_block_{block_index}"
        )

        block = blocks_df.loc[block_index]

        block_start = pd.Timestamp(
            f"{block['date']} {block['start_time']}"
        )

        block_end = pd.Timestamp(
            f"{block['date']} {block['end_time']}"
        )

        block_duration_min = int(
            (block_end - block_start).total_seconds() // 60
        )

        block_used_minutes_vars[block_index] = model.NewIntVar(
            0,
            block_duration_min,
            f"used_minutes_{block_index}",
        )

    # ---------------------------------------------------------
    # 7A. Create department coordination variables
    # ---------------------------------------------------------
    #
    # A department is marked as active in a block when at least
    # one task from that department is assigned to the block.
    #
    # This lets the objective explicitly reward useful
    # multi-department coordination without weakening any
    # feasibility constraint.
    #
    departments = sorted(
        tasks_df["department"]
        .dropna()
        .astype(str)
        .unique()
    )

    for block_index in candidate_block_indices:
        department_used_vars[block_index] = {}

        for department in departments:
            department_used_vars[
                block_index
            ][department] = model.NewBoolVar(
                f"dept_{department}_block_{block_index}"
            )

        multi_department_block_vars[
            block_index
        ] = model.NewBoolVar(
            f"multi_department_block_{block_index}"
        )
    # ---------------------------------------------------------
    # 8. Maintenance intervals grouped by block
    # ---------------------------------------------------------

    maintenance_intervals_by_block = {
        block_index: []
        for block_index in candidate_block_indices
    }

    # ---------------------------------------------------------
    # 9. Create maintenance decision variables
    # ---------------------------------------------------------

    for candidate in candidates:

        task_id = candidate["task_id"]
        block_id = candidate["block_window_id"]

        # Convert IDs to DataFrame indices.

        task_index = task_id_to_index[task_id]
        block_index = block_id_to_index[block_id]

        task = tasks_by_id[task_id]

        task_duration = int(
            candidate["task_duration_min"]
        )

        block_start = candidate["block_start"]
        block_end = candidate["block_end"]

        # -----------------------------------------------------
        # Convert block timestamps to integer minutes
        # -----------------------------------------------------

        block_start_min = int(
            block_start.timestamp() // 60
        )

        block_end_min = int(
            block_end.timestamp() // 60
        )

        # -----------------------------------------------------
        # Convert task temporal boundaries
        # -----------------------------------------------------

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

        # -----------------------------------------------------
        # Calculate actual feasible scheduling window
        # -----------------------------------------------------

        # The task cannot start before either:
        #
        #   - the block starts
        #   - the task's earliest allowed start
        #
        # Therefore:

        feasible_start_min = max(
            block_start_min,
            task_earliest_min,
        )

        # The task must finish before either:
        #
        #   - the block ends
        #   - the task deadline
        #
        # Therefore:

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
        # Maintenance start variable
        # -----------------------------------------------------

        start_var = model.NewIntVar(
            feasible_start_min,
            feasible_end_min - task_duration,
            f"start_{task_id}_{block_id}",
        )

        # -----------------------------------------------------
        # Maintenance end variable
        # -----------------------------------------------------

        end_var = model.NewIntVar(
            feasible_start_min + task_duration,
            feasible_end_min,
            f"end_{task_id}_{block_id}",
        )

        # -----------------------------------------------------
        # Link start and end when assignment is selected
        # -----------------------------------------------------

        model.Add(
            end_var == start_var + task_duration
        ).OnlyEnforceIf(
            assignment_var
        )

        # -----------------------------------------------------
        # Optional maintenance interval
        # -----------------------------------------------------

        interval = model.NewOptionalIntervalVar(
            start_var,
            task_duration,
            end_var,
            assignment_var,
            f"maintenance_{task_id}_{block_id}",
        )

        interval_vars[
            (task_index, block_index)
        ] = interval

        maintenance_intervals_by_block[
            block_index
        ].append(interval)

        # -----------------------------------------------------
        # Link task assignment to block usage
        # -----------------------------------------------------

        model.AddImplication(
            assignment_var,
            block_used_vars[block_index],
        )

    # ---------------------------------------------------------
    # 9A. Link department usage to assignments
    # ---------------------------------------------------------
    #
    # For each block and department:
    #
    #   department_used = 1
    #       iff at least one task from that department
    #       is assigned to the block.
    #
    # Then multi_department_block = 1 iff at least two
    # departments are active in the same block.
    #

    for block_index in candidate_block_indices:

        department_used_for_block = []

        for department in departments:

            department_assignments = [
                assignment_var
                for (
                    task_index,
                    candidate_block_index,
                ), assignment_var in assignment_vars.items()
                if (
                    candidate_block_index == block_index
                    and str(
                        tasks_df.loc[
                            task_index,
                            "department",
                        ]
                    ) == department
                )
            ]

            department_used_var = department_used_vars[
                block_index
            ][department]

            if department_assignments:
                model.AddMaxEquality(
                    department_used_var,
                    department_assignments,
                )
            else:
                model.Add(
                    department_used_var == 0
                )

            department_used_for_block.append(
                department_used_var
            )

        multi_department_var = (
            multi_department_block_vars[block_index]
        )

        department_count = sum(
            department_used_for_block
        )

        # If multi_department_var == 0:
        #     department_count <= 1
        #
        # If multi_department_var == 1:
        #     department_count >= 2
        #
        model.Add(
            department_count
            <= 1 + 2 * multi_department_var
        )

        model.Add(
            department_count
            >= 2 * multi_department_var
        )

    # ---------------------------------------------------------
    # 9B. Calculate used minutes for each block
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
                task_duration * assignment_var
            )

        if block_assignments:
            model.Add(
                block_used_minutes_vars[block_index]
                == sum(block_assignments)
            )
        else:
            model.Add(
                block_used_minutes_vars[block_index] == 0
            )

    # ---------------------------------------------------------
    # 10. Each task can be scheduled at most once
    # ---------------------------------------------------------

    for task_index in tasks_df.index:

        task_assignment_vars = [
            assignment_var
            for (
                candidate_task_index,
                _,
            ), assignment_var in assignment_vars.items()
            if candidate_task_index == task_index
        ]

        if task_assignment_vars:

            model.AddAtMostOne(
                task_assignment_vars
            )

    # ---------------------------------------------------------
    # 11. Maintenance tasks cannot overlap
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

        # Retrieve the actual block using its DataFrame index.

        block = blocks_df.loc[block_index]

        block_section = block["section_id"]

        block_window_start = pd.Timestamp(
            f"{block['date']} {block['start_time']}"
        )

        block_window_end = pd.Timestamp(
            f"{block['date']} {block['end_time']}"
        )

        train_intervals = []

        # -----------------------------------------------------
        # Find trains conflicting with this block
        # -----------------------------------------------------

        for _, train in trains_df.iterrows():

            # Train must operate on the same section.

            if train["section_id"] != block_section:
                continue

            train_entry = pd.Timestamp(
                train["entry_time"]
            )

            train_exit = pd.Timestamp(
                train["exit_time"]
            )

            # Train completely before block.

            if train_exit <= block_window_start:
                continue

            # Train completely after block.

            if train_entry >= block_window_end:
                continue

            # -------------------------------------------------
            # Convert train timestamps to integer minutes
            # -------------------------------------------------

            train_start_min = int(
                train_entry.timestamp() // 60
            )

            train_end_min = int(
                train_exit.timestamp() // 60
            )

            # -------------------------------------------------
            # Fixed train interval
            # -------------------------------------------------

            train_interval = model.NewIntervalVar(
                train_start_min,
                train_end_min - train_start_min,
                train_end_min,
                f"train_{train['movement_id']}_{block_index}",
            )

            train_intervals.append(
                train_interval
            )

        # -----------------------------------------------------
        # Prevent maintenance/train overlap
        # -----------------------------------------------------

        if train_intervals:

            model.AddNoOverlap(
                train_intervals
                + maintenance_intervals
            )

        # ---------------------------------------------------------
    # 13. Objective function
    # ---------------------------------------------------------

    objective_terms = []

    # ---------------------------------------------------------
    # Deadline reference
    # ---------------------------------------------------------

    # Deadline pressure is calculated relative to the latest
    # deadline in the current task dataset.
    #
    # Earlier deadlines receive a higher score.
    #
    # Example:
    #
    # Latest deadline       -> 0 points
    # 1 day earlier         -> 5 points
    # 5 days earlier       -> 25 points
    #
    # This keeps deadline pressure proportional to the
    # planning dataset rather than using arbitrary calendar
    # dates.

    task_deadlines = pd.to_datetime(
        tasks_df["deadline"]
    )

    latest_deadline = task_deadlines.max()

    DEADLINE_PRESSURE_PER_DAY = 5

    # ---------------------------------------------------------
    # Calculate score for every candidate assignment
    # ---------------------------------------------------------

    for (
        task_index,
        block_index,
    ), assignment_var in assignment_vars.items():

        task = tasks_df.loc[task_index]

        # -----------------------------------------------------
        # Criticality score
        # -----------------------------------------------------

        criticality = str(
            task["criticality"]
        ).lower()

        criticality_score = CRITICALITY_WEIGHT.get(
            criticality,
            0,
        )

        # -----------------------------------------------------
        # Urgency score
        # -----------------------------------------------------

        urgency = str(
            task["urgency"]
        ).lower()

        urgency_score = URGENCY_WEIGHT.get(
            urgency,
            0,
        )

        # -----------------------------------------------------
        # Deadline pressure score
        # -----------------------------------------------------

        task_deadline = pd.Timestamp(
            task["deadline"]
        )

        days_until_reference = (
            latest_deadline - task_deadline
        ).days

        deadline_pressure_score = max(
            0,
            days_until_reference,
        ) * DEADLINE_PRESSURE_PER_DAY

        # -----------------------------------------------------
        # Combined maintenance priority
        # -----------------------------------------------------

        priority_score = (
            criticality_score
            + urgency_score
            + deadline_pressure_score
        )

        objective_terms.append(
            priority_score * assignment_var
        )

    # ---------------------------------------------------------
    # 14. Block usage and utilization
    # ---------------------------------------------------------

    # Encourage consolidation of maintenance work.

    for block_index, block_used_var in block_used_vars.items():

        # Penalty for opening a maintenance block.
        objective_terms.append(
            -BLOCK_USAGE_PENALTY * block_used_var
        )

        block = blocks_df.loc[block_index]

        block_start = pd.Timestamp(
            f"{block['date']} {block['start_time']}"
        )

        block_end = pd.Timestamp(
            f"{block['date']} {block['end_time']}"
        )

        block_duration_min = int(
            (block_end - block_start).total_seconds() // 60
        )

        unused_capacity = model.NewIntVar(
            0,
            block_duration_min,
            f"unused_capacity_{block_index}",
        )

        model.Add(
            unused_capacity
            == (
                block_duration_min * block_used_var
                - block_used_minutes_vars[block_index]
            )
        )

        # Penalize unused capacity inside an opened block.
        objective_terms.append(
            -UNUSED_CAPACITY_PENALTY * unused_capacity
        )

        # -----------------------------------------------------
        # Reward multi-department coordination
        # -----------------------------------------------------
        #
        # This is a soft objective preference, not a hard
        # requirement. A block earns the bonus only when tasks
        # from at least two departments are actually scheduled
        # in that same block.
        #
        objective_terms.append(
            MULTI_DEPARTMENT_BONUS
            * multi_department_block_vars[block_index]
        )

    # ---------------------------------------------------------
    # 15. Maximize objective
    # ---------------------------------------------------------

    model.Maximize(
        sum(objective_terms)
    )

        # ---------------------------------------------------------
    # 16. Return model objects
    # ---------------------------------------------------------

    return (
        model,
        assignment_vars,
        interval_vars,
        block_used_vars,
        block_used_minutes_vars,
    )