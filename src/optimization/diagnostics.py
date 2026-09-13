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
# Explain unscheduled task
# =========================================================

def explain_unscheduled_task(
    task,
    candidates,
    schedule_df,
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

            "explanation":
                (
                    "No available block matches the "
                    "task section, availability, "
                    "duration, earliest-start and "
                    "deadline constraints."
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

            "explanation":
                (
                    f"Task requires {required_duration} "
                    f"minutes, but the longest feasible "
                    f"candidate block is only "
                    f"{longest_candidate} minutes."
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
                    "At least one feasible block was "
                    "available, but the optimizer chose "
                    "a different combination of tasks "
                    "to maximize the overall objective."
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
                "The task had feasible block options, "
                "but scheduling the task was not part "
                "of the optimal combination under the "
                "current objective and constraints."
            ),
    }


# =========================================================
# Main diagnostics engine
# =========================================================

def generate_diagnostics():
    """
    Generate detailed explanations for every maintenance task.
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
                        "Task selected by optimizer.",
                    "explanation":
                        (
                            "Task was selected as part "
                            "of the optimal schedule "
                            "based on priority and "
                            "available block capacity."
                        ),
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
                    "explanation":
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

    print()
    print("OUTPUT FILES")
    print("-" * 60)

    print(
        f"Diagnostics : {DIAGNOSTICS_FILE}"
    )

    print(
        f"Utilization : {UTILIZATION_FILE}"
    )

    print()
    print("=" * 60)
    print("PHASE 1.5 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    generate_diagnostics()