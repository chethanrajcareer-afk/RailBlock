from pathlib import Path

import pandas as pd

from src.optimization.candidates import generate_candidates


# =========================================================
# Configuration
# =========================================================

OUTPUT_DIR = Path("outputs")

SCHEDULE_FILE = OUTPUT_DIR / "optimized_schedule.csv"

DIAGNOSTICS_FILE = OUTPUT_DIR / "schedule_diagnostics.csv"

UTILIZATION_FILE = OUTPUT_DIR / "block_utilization.csv"

INFEASIBILITY_FILE = (
    OUTPUT_DIR / "infeasibility_analysis.csv"
)


# =========================================================
# Helper functions
# =========================================================

def priority_score(task):
    """
    Calculate the same priority score used by the optimizer.
    """

    criticality_weight = {
        "critical": 100,
        "high": 70,
        "medium": 40,
        "low": 20,
    }

    urgency_weight = {
        "high": 50,
        "medium": 25,
        "low": 10,
    }

    criticality = str(
        task["criticality"]
    ).lower()

    urgency = str(
        task["urgency"]
    ).lower()

    return (
        criticality_weight.get(
            criticality,
            0,
        )
        + urgency_weight.get(
            urgency,
            0,
        )
    )


def load_data():
    """
    Load RailBlock optimization datasets.
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

    schedule_df = pd.read_csv(
        SCHEDULE_FILE
    )

    return (
        tasks_df,
        blocks_df,
        trains_df,
        schedule_df,
    )


# =========================================================
# Candidate analysis
# =========================================================

def build_candidate_lookup(
    tasks_df,
    blocks_df,
):
    """
    Generate the same basic feasible task -> block
    candidates used by the optimizer.

    Returns:
        Dictionary:
            task_id -> list of candidate blocks
    """

    candidates = generate_candidates(
        tasks_df,
        blocks_df,
    )

    lookup = {}

    for candidate in candidates:

        task_id = candidate["task_id"]

        lookup.setdefault(
            task_id,
            [],
        ).append(candidate)

    return lookup


# =========================================================
# Detailed infeasibility analysis
# =========================================================

def analyze_infeasibility(
    task,
    blocks_df,
):
    """
    Analyze why a task has no basic feasible block.

    This diagnostic layer mirrors the basic task/block
    feasibility logic used by candidate generation:
        - matching section
        - available block
        - sufficient duration
        - earliest-start constraint
        - deadline constraint

    It does not modify the optimizer and does not claim
    train-conflict infeasibility, because train conflicts are
    handled later by the optimization model.
    """

    task_section = str(task["section_id"])
    required_duration = int(
        task["estimated_duration_min"]
    )
    earliest_start = pd.Timestamp(
        task["earliest_start"]
    )
    deadline = pd.Timestamp(
        task["deadline"]
    )

    total_blocks = len(blocks_df)
    section_blocks = 0
    unavailable_blocks = 0
    too_short_blocks = 0
    starts_too_late = 0
    deadline_violations = 0
    feasible_basic_blocks = 0

    # Track the strongest diagnostic evidence.
    longest_available_block = 0.0
    latest_possible_start = None
    earliest_block_end = None

    for _, block in blocks_df.iterrows():

        if str(block["section_id"]) != task_section:
            continue

        section_blocks += 1

        if str(
            block["availability_status"]
        ).lower() != "available":
            unavailable_blocks += 1
            continue

        block_start = pd.Timestamp(
            f"{block['date']} {block['start_time']}"
        )

        block_end = pd.Timestamp(
            f"{block['date']} {block['end_time']}"
        )

        block_minutes = (
            block_end - block_start
        ).total_seconds() / 60

        longest_available_block = max(
            longest_available_block,
            block_minutes,
        )

        if earliest_block_end is None:
            earliest_block_end = block_end
        else:
            earliest_block_end = min(
                earliest_block_end,
                block_end,
            )

        if block_minutes < required_duration:
            too_short_blocks += 1
            continue

        possible_start = max(
            block_start,
            earliest_start,
        )

        possible_end = (
            possible_start
            + pd.Timedelta(
                minutes=required_duration
            )
        )

        if possible_start > block_end:
            starts_too_late += 1
            latest_possible_start = max(
                latest_possible_start,
                possible_start,
            ) if latest_possible_start is not None else possible_start
            continue

        if possible_end > block_end:
            starts_too_late += 1
            latest_possible_start = max(
                latest_possible_start,
                possible_start,
            ) if latest_possible_start is not None else possible_start
            continue

        if possible_end > deadline:
            deadline_violations += 1
            continue

        feasible_basic_blocks += 1

    # Determine a concise primary reason for the task.
    if total_blocks == 0:
        primary_reason = "NO_BLOCKS_AVAILABLE"
    elif section_blocks == 0:
        primary_reason = "NO_SECTION_BLOCK"
    elif section_blocks == unavailable_blocks:
        primary_reason = "NO_AVAILABLE_BLOCK"
    elif feasible_basic_blocks > 0:
        primary_reason = "BASIC_FEASIBLE"
    elif too_short_blocks > 0 and (
        too_short_blocks
        == section_blocks - unavailable_blocks
    ):
        primary_reason = "BLOCK_TOO_SHORT"
    elif deadline_violations > 0:
        primary_reason = "DEADLINE_CONFLICT"
    elif starts_too_late > 0:
        primary_reason = "TIMING_CONFLICT"
    else:
        primary_reason = "NO_FEASIBLE_BLOCK"

    if primary_reason == "NO_SECTION_BLOCK":
        explanation = (
            f"No block exists for section {task_section} "
            "within the planning horizon."
        )
    elif primary_reason == "NO_AVAILABLE_BLOCK":
        explanation = (
            f"Blocks exist for section {task_section}, "
            "but none are marked available."
        )
    elif primary_reason == "BLOCK_TOO_SHORT":
        explanation = (
            f"Task requires {required_duration} minutes, "
            f"while all available section blocks are shorter."
        )
    elif primary_reason == "DEADLINE_CONFLICT":
        explanation = (
            "Available blocks exist, but the task cannot "
            "finish within its deadline."
        )
    elif primary_reason == "TIMING_CONFLICT":
        explanation = (
            "Available blocks exist, but the task cannot "
            "fit after its earliest allowed start."
        )
    elif primary_reason == "BASIC_FEASIBLE":
        explanation = (
            "At least one basic task/block combination is feasible."
        )
    else:
        explanation = (
            "No basic task/block combination satisfies "
            "the current feasibility rules."
        )

    return {
        "infeasibility_primary_reason": primary_reason,
        "section_block_count": section_blocks,
        "unavailable_block_count": unavailable_blocks,
        "too_short_block_count": too_short_blocks,
        "timing_conflict_count": starts_too_late,
        "deadline_conflict_count": deadline_violations,
        "basic_feasible_block_count": feasible_basic_blocks,
        "longest_available_block_minutes": round(
            longest_available_block,
            1,
        ),
        "infeasibility_explanation": explanation,
    }


# =========================================================
# Block utilization
# =========================================================

def calculate_block_utilization(
    blocks_df,
    schedule_df,
):
    """
    Calculate utilization for every block used by the
    optimizer.

    Utilization is:

        scheduled maintenance minutes
        --------------------------------
        available block minutes

    expressed as a percentage.
    """

    rows = []

    for _, block in blocks_df.iterrows():

        block_id = block["block_window_id"]

        block_start = pd.Timestamp(
            f"{block['date']} {block['start_time']}"
        )

        block_end = pd.Timestamp(
            f"{block['date']} {block['end_time']}"
        )

        block_minutes = (
            block_end - block_start
        ).total_seconds() / 60

        block_tasks = schedule_df[
            schedule_df["block_window_id"]
            == block_id
        ]

        scheduled_minutes = 0.0

        if not block_tasks.empty:

            for _, task in block_tasks.iterrows():

                scheduled_start = pd.Timestamp(
                    task["scheduled_start"]
                )

                scheduled_end = pd.Timestamp(
                    task["scheduled_end"]
                )

                scheduled_minutes += (
                    scheduled_end
                    - scheduled_start
                ).total_seconds() / 60

        utilization = 0.0

        if block_minutes > 0:

            utilization = (
                scheduled_minutes
                / block_minutes
            ) * 100

        rows.append(
            {
                "block_window_id": block_id,
                "section_id": block["section_id"],
                "date": block["date"],
                "start_time": block["start_time"],
                "end_time": block["end_time"],
                "availability_status": block[
                    "availability_status"
                ],
                "block_minutes": round(
                    block_minutes,
                    1,
                ),
                "scheduled_minutes": round(
                    scheduled_minutes,
                    1,
                ),
                "unused_minutes": round(
                    max(
                        block_minutes
                        - scheduled_minutes,
                        0,
                    ),
                    1,
                ),
                "utilization_pct": round(
                    utilization,
                    1,
                ),
                "tasks_scheduled": len(
                    block_tasks
                ),
            }
        )

    return pd.DataFrame(rows)


# =========================================================
# Human-readable explanations
# =========================================================

def build_priority_explanation(task):
    """Explain task priority without exposing optimizer terminology."""
    criticality = str(task["criticality"]).lower()
    urgency = str(task["urgency"]).lower()

    criticality_text = {
        "critical": "This work is critical and was given very high priority.",
        "high": "This work has high importance and was given higher priority.",
        "medium": "This work has moderate importance.",
        "low": "This work has lower priority compared with more urgent maintenance work.",
    }.get(criticality, "This work was considered as part of the maintenance plan.")

    urgency_text = {
        "high": "It also needs attention soon.",
        "medium": "It needs attention within the planned maintenance period.",
        "low": "There is less immediate urgency for this work.",
    }.get(urgency, "")

    return f"{criticality_text} {urgency_text}".strip()


def build_deadline_explanation(task, block_end):
    """Explain deadline pressure in plain language."""
    deadline = pd.Timestamp(task["deadline"])
    block_end = pd.Timestamp(block_end)
    days_remaining = (deadline.normalize() - block_end.normalize()).days

    if days_remaining < 0:
        return "This work is being completed before its required completion date."
    if days_remaining <= 1:
        return "The required completion date is very close, so completing this work now helps avoid missing the deadline."
    if days_remaining <= 3:
        return "The required completion date is approaching, so completing this work now reduces deadline risk."
    return "The work can be completed comfortably within its required completion period."


def build_block_explanation(block_minutes, maintenance_minutes, utilization):
    """Explain why the selected maintenance window is a good fit."""
    unused = max(block_minutes - maintenance_minutes, 0)

    if utilization >= 90:
        fit = "The maintenance work makes very good use of the available block time."
    elif utilization >= 70:
        fit = "The maintenance work makes good use of the available block time."
    elif utilization >= 50:
        fit = "The maintenance work fits comfortably within the available block time."
    else:
        fit = "The work fits within the available block, although some time remains unused."

    return f"{fit} {int(round(unused))} minutes remain unused in this block."


def build_coordination_explanation(block_id, schedule_df):
    """Explain whether multiple maintenance teams share the same block."""
    block_tasks = schedule_df[
        schedule_df["block_window_id"] == block_id
    ]

    departments = list(dict.fromkeys(
        str(value) for value in block_tasks["department"].tolist()
    ))

    if len(departments) > 1:
        if len(departments) == 2:
            teams = f"{departments[0]} and {departments[1]}"
        else:
            teams = ", ".join(departments[:-1]) + f", and {departments[-1]}"
        return (
            f"Maintenance work from {teams} can be completed during the same block, "
            "reducing the need for separate maintenance windows."
        )

    return "This block is suitable for the maintenance work required by the team."


def count_train_conflicts(block, schedule_df, trains_df):
    """Count train movements that overlap scheduled maintenance in a block."""
    block_id = block["block_window_id"]
    block_tasks = schedule_df[schedule_df["block_window_id"] == block_id]

    if block_tasks.empty or trains_df.empty:
        return 0

    section_id = str(block["section_id"])
    section_trains = trains_df[
        trains_df["section_id"].astype(str) == section_id
    ].copy()

    if section_trains.empty:
        return 0

    conflicts = 0
    for _, maintenance in block_tasks.iterrows():
        maintenance_start = pd.Timestamp(maintenance["scheduled_start"])
        maintenance_end = pd.Timestamp(maintenance["scheduled_end"])

        for _, train in section_trains.iterrows():
            train_start = pd.Timestamp(train["entry_time"])
            train_end = pd.Timestamp(train["exit_time"])

            if maintenance_start < train_end and train_start < maintenance_end:
                conflicts += 1

    return conflicts


def build_train_protection_explanation(conflict_count):
    """Explain the actual train-conflict check in plain language."""
    if conflict_count == 0:
        return "No scheduled train movement overlaps the maintenance work in this block."

    return (
        f"{conflict_count} scheduled train movement(s) overlap this maintenance window; "
        "this should be reviewed before the schedule is used operationally."
    )


def build_scheduled_explanation(
    task,
    block,
    block_minutes,
    maintenance_minutes,
    utilization,
    schedule_df,
    trains_df,
):
    """Build short, deterministic, human-readable reasons for a scheduled task."""
    conflict_count = count_train_conflicts(block, schedule_df, trains_df)

    reasons = [
        build_priority_explanation(task),
        build_deadline_explanation(
            task,
            block["date"] + " " + block["end_time"],
        ),
        build_block_explanation(
            block_minutes,
            maintenance_minutes,
            utilization,
        ),
        build_coordination_explanation(
            block["block_window_id"],
            schedule_df,
        ),
        build_train_protection_explanation(conflict_count),
    ]

    # Keep only useful reasons and avoid exposing internal scoring language.
    reasons = [reason for reason in reasons if reason]

    block_tasks = schedule_df[
        schedule_df["block_window_id"] == block["block_window_id"]
    ]
    multi_department = block_tasks["department"].nunique() > 1

    if multi_department:
        summary = (
            "This block was selected because maintenance work from multiple teams can "
            "be completed together while the available time is used efficiently."
        )
    elif conflict_count == 0:
        summary = (
            "This maintenance work fits the available window, matches the task's "
            "priority and timing needs, and does not overlap scheduled train movements."
        )
    else:
        summary = (
            "This maintenance work fits the available window, but the train-movement "
            "check found an overlap that should be reviewed."
        )

    return reasons, summary, conflict_count


# =========================================================
# Explain unscheduled task
# =========================================================

def explain_unscheduled_task(
    task,
    candidates,
    schedule_df,
    blocks_df,
):
    """
    Explain why a task was not scheduled.

    This function distinguishes between:

        1. No feasible candidate block exists.

        2. Feasible blocks exist but the optimizer
           selected another combination of tasks.
    """

    task_id = task["task_id"]

    required_duration = int(
        task["estimated_duration_min"]
    )

    task_candidates = candidates.get(
        task_id,
        [],
    )

    detailed_analysis = analyze_infeasibility(
        task,
        blocks_df,
    )

    # -----------------------------------------------------
    # No feasible candidates
    # -----------------------------------------------------

    if not task_candidates:

        return {
            "reason_category":
                "NO_FEASIBLE_BLOCK",

            "reason":
                "No available block satisfies the "
                "basic task feasibility constraints.",

            "candidate_count":
                0,

            "longest_candidate_minutes":
                0,

            "required_duration_minutes":
                required_duration,

            **detailed_analysis,

            "explanation":
                (
                    f"No available maintenance window matches the requirements "
                    f"for this work. It needs {required_duration} minutes and must "
                    "also fit the section, available time, start-time and deadline requirements."
                ),
        }

    # -----------------------------------------------------
    # Candidate blocks exist
    # -----------------------------------------------------

    candidate_count = len(
        task_candidates
    )

    longest_candidate = max(
        int(
            (
                candidate["block_end"]
                - candidate["block_start"]
            ).total_seconds()
            / 60
        )
        for candidate in task_candidates
    )

    # -----------------------------------------------------
    # Determine whether candidate blocks were used
    # -----------------------------------------------------

    candidate_block_ids = {
        candidate["block_window_id"]
        for candidate in task_candidates
    }

    used_candidate_blocks = schedule_df[
        schedule_df["block_window_id"].isin(
            candidate_block_ids
        )
    ]

    # -----------------------------------------------------
    # Duration issue
    # -----------------------------------------------------

    if longest_candidate < required_duration:

        return {
            "reason_category":
                "BLOCK_TOO_SHORT",

            "reason":
                "No feasible block is long enough.",

            "candidate_count":
                candidate_count,

            "longest_candidate_minutes":
                longest_candidate,

            "required_duration_minutes":
                required_duration,

            **detailed_analysis,

            "explanation":
                (
                    f"This work could not be scheduled because it requires {required_duration} "
                    f"minutes, but the longest suitable maintenance window is only "
                    f"{longest_candidate} minutes. A longer maintenance window would be needed."
                ),
        }

    # -----------------------------------------------------
    # Candidate existed and was used
    # -----------------------------------------------------

    if not used_candidate_blocks.empty:

        return {
            "reason_category":
                "OPTIMIZER_SELECTION",

            "reason":
                "Feasible blocks existed, but the "
                "optimizer selected another task "
                "combination.",

            "candidate_count":
                candidate_count,

            "longest_candidate_minutes":
                longest_candidate,

            "required_duration_minutes":
                required_duration,

            "explanation":
                (
                    "This work could have been scheduled, but another combination of "
                    "maintenance activities was chosen to make better use of the available blocks."
                ),
        }

    # -----------------------------------------------------
    # Candidate existed but no longer appears in schedule
    # -----------------------------------------------------

    return {
        "reason_category":
            "OPTIMIZER_SELECTION",

        "reason":
            "Feasible blocks existed, but the "
            "optimizer did not select this task.",

        "candidate_count":
            candidate_count,

        "longest_candidate_minutes":
            longest_candidate,

        "required_duration_minutes":
            required_duration,

        "explanation":
            (
                "Suitable maintenance windows existed, but this work was not included in "
                "the final schedule because other maintenance activities were selected instead."
            ),
    }


# =========================================================
# Main diagnostics engine
# =========================================================

def generate_diagnostics():
    """
    Generate detailed explanations for every maintenance task.

    The diagnostics include task-level explainability,
    block utilization, schedule coverage, criticality,
    urgency, department distribution, block consolidation,
    and multi-department coordination metrics.
    """

    (
        tasks_df,
        blocks_df,
        trains_df,
        schedule_df,
    ) = load_data()

    # Silence unused-variable warning for now.
    _ = trains_df

    # -----------------------------------------------------
    # Candidate lookup
    # -----------------------------------------------------

    candidate_lookup = build_candidate_lookup(
        tasks_df,
        blocks_df,
    )

    # -----------------------------------------------------
    # Scheduled task IDs
    # -----------------------------------------------------

    scheduled_task_ids = set(
        schedule_df["task_id"]
    )

    # -----------------------------------------------------
    # Build task diagnostics
    # -----------------------------------------------------

    diagnostics = []

    for _, task in tasks_df.iterrows():

        task_id = task["task_id"]

        scheduled = (
            task_id
            in scheduled_task_ids
        )

        task_priority = priority_score(
            task
        )

        task_candidates = candidate_lookup.get(
            task_id,
            [],
        )

        # -------------------------------------------------
        # Scheduled task
        # -------------------------------------------------

        if scheduled:

            scheduled_row = schedule_df[
                schedule_df["task_id"]
                == task_id
            ].iloc[0]

            block_id = (
                scheduled_row[
                    "block_window_id"
                ]
            )

            block_rows = blocks_df[
                blocks_df["block_window_id"]
                == block_id
            ]

            if not block_rows.empty:

                block = block_rows.iloc[0]

                block_start = pd.Timestamp(
                    f"{block['date']} "
                    f"{block['start_time']}"
                )

                block_end = pd.Timestamp(
                    f"{block['date']} "
                    f"{block['end_time']}"
                )

                block_minutes = (
                    block_end
                    - block_start
                ).total_seconds() / 60

                scheduled_start = pd.Timestamp(
                    scheduled_row[
                        "scheduled_start"
                    ]
                )

                scheduled_end = pd.Timestamp(
                    scheduled_row[
                        "scheduled_end"
                    ]
                )

                maintenance_minutes = (
                    scheduled_end
                    - scheduled_start
                ).total_seconds() / 60

                utilization = (
                    maintenance_minutes
                    / block_minutes
                    * 100
                    if block_minutes > 0
                    else 0
                )

            else:

                block_minutes = 0
                maintenance_minutes = 0
                utilization = 0

            human_reasons, human_summary, train_conflict_count = build_scheduled_explanation(
                task,
                block,
                block_minutes,
                maintenance_minutes,
                utilization,
                schedule_df,
                trains_df,
            )

            diagnostics.append(
                {
                    "task_id": task_id,
                    "status": "SCHEDULED",
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
                    "priority_score":
                        task_priority,
                    "required_duration_minutes":
                        int(
                            task[
                                "estimated_duration_min"
                            ]
                        ),
                    "candidate_count":
                        len(task_candidates),
                    "selected_block_id":
                        block_id,
                    "block_minutes":
                        round(
                            block_minutes,
                            1,
                        ),
                    "maintenance_minutes":
                        round(
                            maintenance_minutes,
                            1,
                        ),
                    "block_utilization_pct":
                        round(
                            utilization,
                            1,
                        ),
                    "reason_category":
                        "SCHEDULED",
                    "reason":
                        "Scheduled because the work fits the available window and is a suitable part of the overall maintenance plan.",
                    "explanation":
                        human_summary,
                    "human_reasons":
                        " | ".join(human_reasons),
                    "train_conflict_count":
                        train_conflict_count,
                }
            )

        # -------------------------------------------------
        # Unscheduled task
        # -------------------------------------------------

        else:

            explanation = explain_unscheduled_task(
                task,
                candidate_lookup,
                schedule_df,
                blocks_df,
            )

            diagnostics.append(
                {
                    "task_id": task_id,
                    "status": "UNSCHEDULED",
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
                    "priority_score":
                        task_priority,
                    "required_duration_minutes":
                        int(
                            task[
                                "estimated_duration_min"
                            ]
                        ),
                    "candidate_count":
                        explanation[
                            "candidate_count"
                        ],
                    "selected_block_id":
                        "",
                    "block_minutes":
                        0,
                    "maintenance_minutes":
                        0,
                    "block_utilization_pct":
                        0,
                    "reason_category":
                        explanation[
                            "reason_category"
                        ],
                    "reason":
                        explanation[
                            "reason"
                        ],
                    "infeasibility_primary_reason":
                        explanation.get(
                            "infeasibility_primary_reason",
                            "",
                        ),
                    "section_block_count":
                        explanation.get(
                            "section_block_count",
                            0,
                        ),
                    "unavailable_block_count":
                        explanation.get(
                            "unavailable_block_count",
                            0,
                        ),
                    "too_short_block_count":
                        explanation.get(
                            "too_short_block_count",
                            0,
                        ),
                    "timing_conflict_count":
                        explanation.get(
                            "timing_conflict_count",
                            0,
                        ),
                    "deadline_conflict_count":
                        explanation.get(
                            "deadline_conflict_count",
                            0,
                        ),
                    "basic_feasible_block_count":
                        explanation.get(
                            "basic_feasible_block_count",
                            0,
                        ),
                    "longest_available_block_minutes":
                        explanation.get(
                            "longest_available_block_minutes",
                            0,
                        ),
                    "infeasibility_explanation":
                        explanation.get(
                            "infeasibility_explanation",
                            "",
                        ),
                    "explanation":
                        explanation[
                            "explanation"
                        ],
                    "human_reasons":
                        explanation[
                            "explanation"
                        ],
                }
            )

    diagnostics_df = pd.DataFrame(
        diagnostics
    )

    # -----------------------------------------------------
    # Save diagnostics
    # -----------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    diagnostics_df.to_csv(
        DIAGNOSTICS_FILE,
        index=False,
    )

    infeasibility_df = diagnostics_df[
        (
            diagnostics_df["status"]
            == "UNSCHEDULED"
        )
        & (
            diagnostics_df["reason_category"]
            == "NO_FEASIBLE_BLOCK"
        )
    ].copy()

    infeasibility_df.to_csv(
        INFEASIBILITY_FILE,
        index=False,
    )

    # -----------------------------------------------------
    # Block utilization
    # -----------------------------------------------------

    utilization_df = calculate_block_utilization(
        blocks_df,
        schedule_df,
    )

    utilization_df.to_csv(
        UTILIZATION_FILE,
        index=False,
    )

    # -----------------------------------------------------
    # Console summary
    # -----------------------------------------------------

    total_tasks = len(
        diagnostics_df
    )

    scheduled_count = (
        diagnostics_df["status"]
        == "SCHEDULED"
    ).sum()

    unscheduled_count = (
        diagnostics_df["status"]
        == "UNSCHEDULED"
    ).sum()

    print()
    print("=" * 60)
    print("RAILBLOCK PHASE 1.5")
    print("SCHEDULE EXPLAINABILITY ENGINE")
    print("=" * 60)

    print(
        f"Total tasks       : {total_tasks}"
    )

    print(
        f"Scheduled         : {scheduled_count}"
    )

    print(
        f"Unscheduled       : {unscheduled_count}"
    )

    print()
    print("UNSCHEDULED REASONS")
    print("-" * 60)

    reason_counts = (
        diagnostics_df[
            diagnostics_df["status"]
            == "UNSCHEDULED"
        ]["reason_category"]
        .value_counts()
    )

    for reason, count in (
        reason_counts.items()
    ):

        print(
            f"{reason:<30} : {count}"
        )

    # -----------------------------------------------------
    # Detailed infeasibility analysis
    # -----------------------------------------------------

    print()
    print("DETAILED INFEASIBILITY ANALYSIS")
    print("-" * 60)

    infeasible_rows = diagnostics_df[
        (
            diagnostics_df["status"]
            == "UNSCHEDULED"
        )
        & (
            diagnostics_df["reason_category"]
            == "NO_FEASIBLE_BLOCK"
        )
    ]

    if infeasible_rows.empty:

        print("No basic infeasibility cases found.")

    else:

        reason_counts = (
            infeasible_rows[
                "infeasibility_primary_reason"
            ]
            .value_counts()
        )

        for reason, count in reason_counts.items():

            print(
                f"{reason:<30} : {count}"
            )

        print()
        print("INFEASIBLE TASK DETAILS")
        print("-" * 60)

        for _, row in infeasible_rows.iterrows():

            print(
                f"{row['task_id']} | "
                f"{row['department']} | "
                f"{row['section_id']} | "
                f"{row['required_duration_minutes']} min"
            )

            print(
                f"  Primary reason : "
                f"{row['infeasibility_primary_reason']}"
            )

            print(
                f"  Section blocks : "
                f"{row['section_block_count']}"
            )

            print(
                f"  Unavailable    : "
                f"{row['unavailable_block_count']}"
            )

            print(
                f"  Too short      : "
                f"{row['too_short_block_count']}"
            )

            print(
                f"  Timing conflict: "
                f"{row['timing_conflict_count']}"
            )

            print(
                f"  Deadline conflict: "
                f"{row['deadline_conflict_count']}"
            )

            print(
                f"  Explanation    : "
                f"{row['infeasibility_explanation']}"
            )

    # -----------------------------------------------------
    # Critical unscheduled tasks
    # -----------------------------------------------------

    critical_unscheduled = diagnostics_df[
        (
            diagnostics_df["status"]
            == "UNSCHEDULED"
        )
        & (
            diagnostics_df["criticality"]
            == "critical"
        )
    ]

    print()
    print("CRITICAL UNSCHEDULED TASKS")
    print("-" * 60)

    if critical_unscheduled.empty:

        print(
            "None"
        )

    else:

        for _, row in (
            critical_unscheduled.iterrows()
        ):

            print(
                f"{row['task_id']} | "
                f"{row['department']} | "
                f"{row['section_id']} | "
                f"{row['required_duration_minutes']} min"
            )

            print(
                f"  Reason: {row['reason']}"
            )

    # -----------------------------------------------------
    # Block utilization summary
    # -----------------------------------------------------

    used_blocks = utilization_df[
        utilization_df["tasks_scheduled"]
        > 0
    ]

    print()
    print("BLOCK UTILIZATION")
    print("-" * 60)

    print(
        f"Used blocks        : "
        f"{len(used_blocks)}"
    )

    if not used_blocks.empty:

        print(
            f"Average utilization: "
            f"{used_blocks['utilization_pct'].mean():.1f}%"
        )

        print(
            f"Minimum utilization: "
            f"{used_blocks['utilization_pct'].min():.1f}%"
        )

        print(
            f"Maximum utilization: "
            f"{used_blocks['utilization_pct'].max():.1f}%"
        )

    # -----------------------------------------------------
    # V1.3 Optimization performance metrics
    # -----------------------------------------------------

    print()
    print("V1.3 OPTIMIZATION PERFORMANCE")
    print("-" * 60)

    total_block_capacity = utilization_df[
        "block_minutes"
    ].sum()

    total_scheduled_minutes = utilization_df[
        "scheduled_minutes"
    ].sum()

    total_unused_minutes = utilization_df[
        "unused_minutes"
    ].sum()

    overall_utilization = (
        total_scheduled_minutes
        / total_block_capacity
        * 100
        if total_block_capacity > 0
        else 0
    )

    print(
        f"Total block capacity : "
        f"{total_block_capacity:.1f} min"
    )

    print(
        f"Maintenance scheduled: "
        f"{total_scheduled_minutes:.1f} min"
    )

    print(
        f"Unused block capacity: "
        f"{total_unused_minutes:.1f} min"
    )

    print(
        f"Overall utilization  : "
        f"{overall_utilization:.1f}%"
    )

    # -----------------------------------------------------
    # Criticality distribution
    # -----------------------------------------------------

    print()
    print("SCHEDULED TASK CRITICALITY")
    print("-" * 60)

    scheduled_tasks_df = diagnostics_df[
        diagnostics_df["status"] == "SCHEDULED"
    ]

    criticality_counts = (
        scheduled_tasks_df["criticality"]
        .value_counts()
    )

    for level in (
        "critical",
        "high",
        "medium",
        "low",
    ):
        print(
            f"{level:<20} : "
            f"{criticality_counts.get(level, 0)}"
        )

    # -----------------------------------------------------
    # Urgency distribution
    # -----------------------------------------------------

    print()
    print("SCHEDULED TASK URGENCY")
    print("-" * 60)

    urgency_counts = (
        scheduled_tasks_df["urgency"]
        .value_counts()
    )

    for level in (
        "high",
        "medium",
        "low",
    ):
        print(
            f"{level:<20} : "
            f"{urgency_counts.get(level, 0)}"
        )

    # -----------------------------------------------------
    # Department distribution
    # -----------------------------------------------------

    print()
    print("SCHEDULED TASKS BY DEPARTMENT")
    print("-" * 60)

    department_counts = (
        scheduled_tasks_df["department"]
        .value_counts()
    )

    for department, count in department_counts.items():
        print(
            f"{department:<20} : {count}"
        )

    # -----------------------------------------------------
    # Block consolidation
    # -----------------------------------------------------

    print()
    print("BLOCK CONSOLIDATION")
    print("-" * 60)

    task_count_distribution = (
        used_blocks["tasks_scheduled"]
        .value_counts()
        .sort_index()
    )

    one_task_blocks = int(
        task_count_distribution.get(1, 0)
    )

    two_task_blocks = int(
        task_count_distribution.get(2, 0)
    )

    multi_task_blocks = int(
        used_blocks[
            used_blocks["tasks_scheduled"] >= 3
        ].shape[0]
    )

    print(
        f"Blocks with 1 task  : {one_task_blocks}"
    )

    print(
        f"Blocks with 2 tasks : {two_task_blocks}"
    )

    print(
        f"Blocks with 3+ tasks: {multi_task_blocks}"
    )

    # -----------------------------------------------------
    # Multi-department coordination
    # -----------------------------------------------------

    print()
    print("MULTI-DEPARTMENT COORDINATION")
    print("-" * 60)

    multi_department_blocks = 0

    if not schedule_df.empty:

        for block_id, block_tasks in (
            schedule_df
            .groupby("block_window_id")
        ):

            department_count = (
                block_tasks["department"]
                .nunique()
            )

            if department_count > 1:
                multi_department_blocks += 1

    print(
        f"Multi-department blocks: "
        f"{multi_department_blocks}"
    )

    # -----------------------------------------------------
    # Human-readable explanation preview
    # -----------------------------------------------------

    print()
    print("HUMAN-READABLE EXPLANATION PREVIEW")
    print("-" * 60)

    preview_rows = diagnostics_df[
        diagnostics_df["status"] == "SCHEDULED"
    ].head(5)

    for _, row in preview_rows.iterrows():
        print(f"{row['task_id']} -> {row['selected_block_id']}")
        print(f"  Why: {row['explanation']}")
        for reason in str(row.get("human_reasons", "")).split(" | "):
            if reason:
                print(f"  - {reason}")

    print()
    print("OUTPUT FILES")
    print("-" * 60)

    print(
        f"Diagnostics : {DIAGNOSTICS_FILE}"
    )

    print(
        f"Utilization : {UTILIZATION_FILE}"
    )

    print(
        f"Infeasibility: {INFEASIBILITY_FILE}"
    )

    print()
    print("=" * 60)
    print("PHASE 1.5 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    generate_diagnostics()