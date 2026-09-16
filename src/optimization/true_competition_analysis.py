"""RailBlock true temporal competition analysis.

Phase 2.4.6.1

This module performs counterfactual attribution for feasible-but-unselected
maintenance tasks. It distinguishes whether loss of continuous capacity in
the final risk-aware schedule is attributable to scheduled maintenance,
train movements, both, or the intrinsic block/timing limits.

All data is fictional synthetic railway data and does not represent
operational Indian Railways data or rules.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.optimization.candidates import generate_candidates


# =========================================================
# Project paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "synthetic"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

TASK_FILE = DATA_DIR / "maintenance_tasks.csv"
BLOCK_FILE = DATA_DIR / "block_availability.csv"
TRAIN_FILE = DATA_DIR / "train_movements.csv"
RISK_FILE = OUTPUT_DIR / "asset_risk_scores.csv"
SCHEDULE_FILE = OUTPUT_DIR / "risk_aware_schedule.csv"


# =========================================================
# Risk classification
# =========================================================

def classify_risk(probability: float) -> str:
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
    required_files = [
        TASK_FILE,
        BLOCK_FILE,
        TRAIN_FILE,
        RISK_FILE,
        SCHEDULE_FILE,
    ]

    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Required file not found: {file_path}"
            )

    tasks = pd.read_csv(TASK_FILE)
    blocks = pd.read_csv(BLOCK_FILE)
    trains = pd.read_csv(TRAIN_FILE)
    risk = pd.read_csv(RISK_FILE)
    schedule = pd.read_csv(SCHEDULE_FILE)

    tasks["earliest_start"] = pd.to_datetime(
        tasks["earliest_start"],
        format="mixed",
    )
    tasks["deadline"] = pd.to_datetime(
        tasks["deadline"],
        format="mixed",
    )

    blocks["block_start"] = pd.to_datetime(
        blocks["date"].astype(str) + " " + blocks["start_time"],
        format="mixed",
    )
    blocks["block_end"] = pd.to_datetime(
        blocks["date"].astype(str) + " " + blocks["end_time"],
        format="mixed",
    )

    trains["entry_time"] = pd.to_datetime(
        trains["entry_time"],
        format="mixed",
    )
    trains["exit_time"] = pd.to_datetime(
        trains["exit_time"],
        format="mixed",
    )

    schedule["scheduled_start"] = pd.to_datetime(
        schedule["scheduled_start"],
        format="mixed",
    )
    schedule["scheduled_end"] = pd.to_datetime(
        schedule["scheduled_end"],
        format="mixed",
    )

    return tasks, blocks, trains, risk, schedule


# =========================================================
# Candidate mapping
# =========================================================

def build_candidate_map(
    candidates: list[dict],
) -> dict[str, list[dict]]:
    candidate_map: dict[str, list[dict]] = {}

    for candidate in candidates:
        candidate_map.setdefault(
            candidate["task_id"],
            [],
        ).append(candidate)

    return candidate_map


# =========================================================
# Interval helpers
# =========================================================

def merge_intervals(
    intervals: list[tuple[pd.Timestamp, pd.Timestamp]],
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if not intervals:
        return []

    intervals = sorted(
        intervals,
        key=lambda x: x[0],
    )

    merged = [intervals[0]]

    for start, end in intervals[1:]:
        previous_start, previous_end = merged[-1]

        if start <= previous_end:
            merged[-1] = (
                previous_start,
                max(previous_end, end),
            )
        else:
            merged.append((start, end))

    return merged


def find_free_intervals(
    block_start: pd.Timestamp,
    block_end: pd.Timestamp,
    occupied_intervals: list[
        tuple[pd.Timestamp, pd.Timestamp]
    ],
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    clipped = []

    for start, end in occupied_intervals:
        start = max(start, block_start)
        end = min(end, block_end)

        if start < end:
            clipped.append((start, end))

    merged = merge_intervals(clipped)

    free = []
    cursor = block_start

    for start, end in merged:
        if cursor < start:
            free.append((cursor, start))

        cursor = max(cursor, end)

    if cursor < block_end:
        free.append((cursor, block_end))

    return free


def max_free_minutes(
    free_intervals: list[
        tuple[pd.Timestamp, pd.Timestamp]
    ],
) -> int:
    if not free_intervals:
        return 0

    return max(
        int((end - start).total_seconds() // 60)
        for start, end in free_intervals
    )


# =========================================================
# Occupancy helpers
# =========================================================

def get_train_intervals(
    block: pd.Series,
    trains: pd.DataFrame,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    block_start = block["block_start"]
    block_end = block["block_end"]
    section_id = block["section_id"]

    occupied = []

    section_trains = trains[
        trains["section_id"] == section_id
    ]

    for _, train in section_trains.iterrows():
        train_start = train["entry_time"]
        train_end = train["exit_time"]

        if train_end <= block_start:
            continue

        if train_start >= block_end:
            continue

        occupied.append(
            (
                max(train_start, block_start),
                min(train_end, block_end),
            )
        )

    return occupied


def get_maintenance_intervals(
    block: pd.Series,
    schedule: pd.DataFrame,
) -> list[dict]:
    block_start = block["block_start"]
    block_end = block["block_end"]

    block_schedule = schedule[
        schedule["block_window_id"]
        == block["block_window_id"]
    ]

    occupied = []

    for _, task in block_schedule.iterrows():
        task_start = task["scheduled_start"]
        task_end = task["scheduled_end"]

        if task_end <= block_start:
            continue

        if task_start >= block_end:
            continue

        occupied.append(
            {
                "task_id": task["task_id"],
                "start": max(task_start, block_start),
                "end": min(task_end, block_end),
            }
        )

    return occupied


def get_maintenance_tuples(
    maintenance_intervals: list[dict],
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    return [
        (item["start"], item["end"])
        for item in maintenance_intervals
    ]


def get_free_minutes_for_scenario(
    usable_start: pd.Timestamp,
    usable_end: pd.Timestamp,
    train_intervals: list[
        tuple[pd.Timestamp, pd.Timestamp]
    ],
    maintenance_intervals: list[
        tuple[pd.Timestamp, pd.Timestamp]
    ],
) -> int:
    occupied = train_intervals + maintenance_intervals

    free_intervals = find_free_intervals(
        usable_start,
        usable_end,
        occupied,
    )

    return max_free_minutes(free_intervals)


# =========================================================
# Test one candidate block
# =========================================================

def evaluate_candidate_block(
    candidate: dict,
    block: pd.Series,
    trains: pd.DataFrame,
    schedule: pd.DataFrame,
    task_duration: int,
    task_earliest: pd.Timestamp,
    task_deadline: pd.Timestamp,
) -> dict:
    del candidate  # Candidate identity is represented by block_window_id.

    block_start = block["block_start"]
    block_end = block["block_end"]

    earliest_start = pd.Timestamp(task_earliest)
    deadline = pd.Timestamp(task_deadline)

    usable_start = max(
        block_start,
        earliest_start,
    )

    usable_end = min(
        block_end,
        deadline,
    )

    if usable_start >= usable_end:
        return {
            "feasible": False,
            "reason": "TIMING_CONFLICT",
            "max_free_minutes": 0,
            "train_only_free_minutes": 0,
            "empty_block_free_minutes": 0,
            "maintenance_removed_free_minutes": 0,
            "maintenance_blockers": [],
            "maintenance_competition_tasks": [],
        }

    # ---------------------------------------------------------
    # Separate train and maintenance occupancy.
    # ---------------------------------------------------------

    train_intervals = get_train_intervals(
        block,
        trains,
    )

    maintenance_intervals = get_maintenance_intervals(
        block,
        schedule,
    )

    maintenance_tuples = get_maintenance_tuples(
        maintenance_intervals,
    )

    # ---------------------------------------------------------
    # Scenario A:
    # Actual final schedule.
    # Trains + scheduled maintenance.
    # ---------------------------------------------------------

    actual_max_free = get_free_minutes_for_scenario(
        usable_start,
        usable_end,
        train_intervals,
        maintenance_tuples,
    )

    # ---------------------------------------------------------
    # Scenario B:
    # Remove all scheduled maintenance, keep trains fixed.
    # ---------------------------------------------------------

    train_only_max_free = get_free_minutes_for_scenario(
        usable_start,
        usable_end,
        train_intervals,
        [],
    )

    # ---------------------------------------------------------
    # Scenario C:
    # Remove both trains and maintenance.
    # ---------------------------------------------------------

    empty_block_max_free = get_free_minutes_for_scenario(
        usable_start,
        usable_end,
        [],
        [],
    )

    actual_feasible = actual_max_free >= task_duration
    train_only_feasible = train_only_max_free >= task_duration
    empty_feasible = empty_block_max_free >= task_duration

    maintenance_blockers = [
        item["task_id"]
        for item in maintenance_intervals
    ]

    # ---------------------------------------------------------
    # Counterfactual attribution.
    #
    # If removing ALL maintenance restores feasibility while
    # trains remain fixed, maintenance contributed to the block.
    #
    # If trains alone still prevent feasibility, test each
    # maintenance task individually. If removing one or more
    # maintenance intervals restores feasibility, the conflict
    # is mixed rather than train-only.
    # ---------------------------------------------------------

    maintenance_competition_tasks = []

    if not actual_feasible and maintenance_intervals:

        for maintenance_item in maintenance_intervals:
            remaining_maintenance = [
                (
                    item["start"],
                    item["end"],
                )
                for item in maintenance_intervals
                if item["task_id"] != maintenance_item["task_id"]
            ]

            remaining_free = get_free_minutes_for_scenario(
                usable_start,
                usable_end,
                train_intervals,
                remaining_maintenance,
            )

            if remaining_free >= task_duration:
                maintenance_competition_tasks.append(
                    maintenance_item["task_id"]
                )

    if actual_feasible:
        reason = "STILL_FEASIBLE_AFTER_SCHEDULE"

    elif train_only_feasible:
        # Removing scheduled maintenance restores feasibility
        # while train movements remain unchanged.
        reason = "MAINTENANCE_COMPETITION"

    elif empty_feasible:
        if maintenance_competition_tasks:
            # Trains alone are insufficient, but removing
            # maintenance also changes feasibility. Therefore
            # both sources contribute to the conflict.
            reason = "MIXED_CONFLICT"
        else:
            # Trains alone prevent the task from fitting and
            # removing maintenance individually does not restore
            # feasibility.
            reason = "TRAIN_CONFLICT"

    else:
        # Even with trains and maintenance removed, the task
        # cannot fit inside its usable block/timing interval.
        reason = "BLOCK_CAPACITY_LIMITATION"

    return {
        "feasible": actual_feasible,
        "reason": reason,
        "max_free_minutes": actual_max_free,
        "train_only_free_minutes": train_only_max_free,
        "empty_block_free_minutes": empty_block_max_free,
        "maintenance_removed_free_minutes": (
            train_only_max_free
        ),
        "maintenance_blockers": maintenance_blockers,
        "maintenance_competition_tasks": (
            maintenance_competition_tasks
        ),
    }


# =========================================================
# Priority reconstruction
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

RISK_WEIGHT = 100
DEADLINE_PRESSURE_PER_DAY = 5


def calculate_priority_scores(
    tasks: pd.DataFrame,
    risk: pd.DataFrame,
) -> pd.DataFrame:
    result = tasks.copy()

    risk_lookup = dict(
        zip(
            risk["asset_id"],
            risk["predicted_failure_probability"],
        )
    )

    latest_deadline = result["deadline"].max()

    priority_scores = []
    risk_scores = []

    for _, task in result.iterrows():
        criticality = str(
            task["criticality"]
        ).lower()

        urgency = str(
            task["urgency"]
        ).lower()

        criticality_score = CRITICALITY_WEIGHT.get(
            criticality,
            0,
        )

        urgency_score = URGENCY_WEIGHT.get(
            urgency,
            0,
        )

        deadline = pd.Timestamp(
            task["deadline"]
        )

        days_until_reference = (
            latest_deadline - deadline
        ).days

        deadline_score = (
            max(
                0,
                days_until_reference,
            )
            * DEADLINE_PRESSURE_PER_DAY
        )

        probability = float(
            risk_lookup.get(
                task["asset_id"],
                0.0,
            )
        )

        risk_score = round(
            probability * RISK_WEIGHT
        )

        priority = (
            criticality_score
            + urgency_score
            + deadline_score
            + risk_score
        )

        priority_scores.append(priority)
        risk_scores.append(risk_score)

    result["priority_score"] = priority_scores
    result["risk_score"] = risk_scores

    result["predicted_failure_probability"] = (
        result["risk_score"]
        / RISK_WEIGHT
    )

    result["risk_level"] = (
        result["predicted_failure_probability"]
        .apply(classify_risk)
    )

    return result


# =========================================================
# Main analysis
# =========================================================

def main():
    print("\n" + "=" * 80)
    print("RAILBLOCK TRUE TEMPORAL COMPETITION ANALYSIS")
    print("=" * 80)

    tasks, blocks, trains, risk, schedule = load_data()

    candidates = generate_candidates(
        tasks,
        blocks,
    )

    print(
        f"\nCandidate assignments : "
        f"{len(candidates)}"
    )

    candidate_map = build_candidate_map(
        candidates
    )

    scored_tasks = calculate_priority_scores(
        tasks,
        risk,
    )

    task_info = scored_tasks.set_index(
        "task_id"
    )

    scheduled_ids = set(
        schedule["task_id"]
    )

    candidate_task_ids = set(
        candidate_map
    )

    feasible_not_selected = sorted(
        candidate_task_ids - scheduled_ids
    )

    print(
        f"Scheduled tasks        : "
        f"{len(scheduled_ids)}"
    )

    print(
        f"Feasible not selected  : "
        f"{len(feasible_not_selected)}"
    )

    results = []

    print("\n" + "=" * 80)
    print("TASK-LEVEL TEMPORAL COMPETITION")
    print("=" * 80)

    for task_id in feasible_not_selected:
        task = task_info.loc[task_id]

        task_duration = int(
            task["estimated_duration_min"]
        )

        task_candidates = candidate_map[
            task_id
        ]

        candidate_results = []

        for candidate in task_candidates:
            block_id = candidate[
                "block_window_id"
            ]

            block_rows = blocks[
                blocks["block_window_id"]
                == block_id
            ]

            if block_rows.empty:
                continue

            block = block_rows.iloc[0]

            evaluation = evaluate_candidate_block(
                candidate,
                block,
                trains,
                schedule,
                task_duration,
                task["earliest_start"],
                task["deadline"],
            )

            candidate_results.append(
                {
                    "block_window_id": block_id,
                    **evaluation,
                }
            )

        if not candidate_results:
            classification = "NO_FEASIBLE_BLOCK"

        else:
            reasons = {
                result["reason"]
                for result in candidate_results
            }

            if "STILL_FEASIBLE_AFTER_SCHEDULE" in reasons:
                classification = "STILL_FEASIBLE_AFTER_SCHEDULE"
            elif "MAINTENANCE_COMPETITION" in reasons:
                classification = "MAINTENANCE_COMPETITION"
            elif "MIXED_CONFLICT" in reasons:
                classification = "MIXED_CONFLICT"
            elif "TRAIN_CONFLICT" in reasons:
                classification = "TRAIN_CONFLICT"
            elif "BLOCK_CAPACITY_LIMITATION" in reasons:
                classification = "BLOCK_CAPACITY_LIMITATION"
            elif "TIMING_CONFLICT" in reasons:
                classification = "TIMING_CONFLICT"
            else:
                classification = "UNRESOLVED"

        max_available = max(
            (
                result["max_free_minutes"]
                for result in candidate_results
            ),
            default=0,
        )

        max_train_only = max(
            (
                result["train_only_free_minutes"]
                for result in candidate_results
            ),
            default=0,
        )

        max_empty_block = max(
            (
                result["empty_block_free_minutes"]
                for result in candidate_results
            ),
            default=0,
        )

        all_maintenance_blockers = sorted(
            {
                blocker
                for result in candidate_results
                for blocker in result["maintenance_blockers"]
            }
        )

        all_competition_tasks = sorted(
            {
                blocker
                for result in candidate_results
                for blocker in result[
                    "maintenance_competition_tasks"
                ]
            }
        )

        print("\n" + "-" * 80)

        print(
            f"Task: {task_id}"
        )

        print(
            f"Asset: {task['asset_id']}"
        )

        print(
            f"Department: {task['department']}"
        )

        print(
            f"Risk: "
            f"{float(task['predicted_failure_probability']):.4f} "
            f"({task['risk_level']})"
        )

        print(
            f"Duration: "
            f"{task_duration} min"
        )

        print(
            f"Priority: "
            f"{task['priority_score']}"
        )

        print(
            f"Candidate blocks: "
            f"{len(candidate_results)}"
        )

        print(
            f"Maximum continuous free time: "
            f"{max_available} min"
        )

        print(
            f"Maximum train-only free time: "
            f"{max_train_only} min"
        )

        print(
            f"Maximum empty-block free time: "
            f"{max_empty_block} min"
        )

        print(
            f"Classification: "
            f"{classification}"
        )

        for result in candidate_results:
            print(
                f"  {result['block_window_id']} "
                f"-> {result['reason']} "
                f"| Actual free: "
                f"{result['max_free_minutes']} min "
                f"| Train-only free: "
                f"{result['train_only_free_minutes']} min"
            )

            blockers = result[
                "maintenance_blockers"
            ]

            if blockers:
                print(
                    f"     Scheduled maintenance: "
                    f"{', '.join(blockers)}"
                )

            competition_tasks = result[
                "maintenance_competition_tasks"
            ]

            if competition_tasks:
                print(
                    f"     Maintenance causing competition: "
                    f"{', '.join(competition_tasks)}"
                )

        results.append(
            {
                "task_id": task_id,
                "asset_id": task["asset_id"],
                "department": task["department"],
                "section_id": task["section_id"],
                "criticality": task["criticality"],
                "urgency": task["urgency"],
                "risk_probability": float(
                    task[
                        "predicted_failure_probability"
                    ]
                ),
                "risk_level": task["risk_level"],
                "task_duration_min": task_duration,
                "priority_score": int(
                    task["priority_score"]
                ),
                "candidate_block_count": len(
                    candidate_results
                ),
                "max_continuous_free_min": max_available,
                "max_train_only_free_min": max_train_only,
                "max_empty_block_free_min": max_empty_block,
                "maintenance_blockers": ";".join(
                    all_maintenance_blockers
                ),
                "maintenance_competition_tasks": ";".join(
                    all_competition_tasks
                ),
                "classification": classification,
            }
        )

    # =====================================================
    # Summary
    # =====================================================

    result_df = pd.DataFrame(results)

    print("\n" + "=" * 80)
    print("TRUE COMPETITION SUMMARY")
    print("=" * 80)

    if result_df.empty:
        print(
            "No feasible-but-unselected tasks "
            "were found."
        )
        return

    print("\nClassification counts:")
    print(
        result_df[
            "classification"
        ].value_counts()
    )

    print(
        "\nRisk distribution by classification:"
    )

    print(
        result_df.groupby(
            [
                "classification",
                "risk_level",
            ]
        ).size()
    )

    print(
        "\nHigh-risk tasks:"
    )

    high_risk = result_df[
        result_df["risk_probability"] >= 0.50
    ].sort_values(
        "risk_probability",
        ascending=False,
    )

    if high_risk.empty:
        print("None")
    else:
        print(
            high_risk[
                [
                    "task_id",
                    "risk_probability",
                    "risk_level",
                    "candidate_block_count",
                    "max_continuous_free_min",
                    "max_train_only_free_min",
                    "classification",
                    "maintenance_competition_tasks",
                ]
            ].to_string(index=False)
        )

    output_file = (
        OUTPUT_DIR
        / "true_competition_analysis.csv"
    )

    result_df.to_csv(
        output_file,
        index=False,
    )

    print(
        f"\nOutput: {output_file}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()
