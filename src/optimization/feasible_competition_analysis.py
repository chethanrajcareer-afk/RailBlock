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
# Objective weights
# Keep these identical to model_risk_aware.py
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

    return tasks, blocks, trains, risk, schedule


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
# Calculate the same priority score used by optimizer
# =========================================================

def calculate_priority_scores(
    tasks: pd.DataFrame,
    risk: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reconstruct the maintenance priority calculation used
    by the risk-aware optimizer.

    This is for analysis only.
    """

    result = tasks.copy()

    risk_lookup = dict(
        zip(
            risk["asset_id"],
            risk["predicted_failure_probability"],
        )
    )

    latest_deadline = pd.to_datetime(
        result["deadline"]
    ).max()

    priority_scores = []
    risk_scores = []
    criticality_scores = []
    urgency_scores = []
    deadline_scores = []

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

        task_deadline = pd.Timestamp(
            task["deadline"]
        )

        days_until_reference = (
            latest_deadline - task_deadline
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

        priority_score = (
            criticality_score
            + urgency_score
            + deadline_score
            + risk_score
        )

        criticality_scores.append(
            criticality_score
        )

        urgency_scores.append(
            urgency_score
        )

        deadline_scores.append(
            deadline_score
        )

        risk_scores.append(
            risk_score
        )

        priority_scores.append(
            priority_score
        )

    result["criticality_score"] = criticality_scores
    result["urgency_score"] = urgency_scores
    result["deadline_score"] = deadline_scores
    result["risk_score"] = risk_scores
    result["priority_score"] = priority_scores

    result["predicted_failure_probability"] = (
        result["risk_score"] / RISK_WEIGHT
    )

    result["risk_level"] = (
        result["predicted_failure_probability"]
        .apply(classify_risk)
    )

    return result


# =========================================================
# Candidate competition
# =========================================================

def build_candidate_map(
    candidates: list[dict],
) -> dict[str, set[str]]:
    """
    Map each task to the block windows it can potentially use.
    """

    candidate_map = {}

    for candidate in candidates:

        task_id = candidate["task_id"]
        block_id = candidate["block_window_id"]

        candidate_map.setdefault(
            task_id,
            set(),
        ).add(block_id)

    return candidate_map


# =========================================================
# Find competing scheduled tasks
# =========================================================

def find_competing_tasks(
    task_id: str,
    candidate_map: dict[str, set[str]],
    scheduled_ids: set[str],
) -> dict[str, list[str]]:
    """
    Find scheduled tasks that share candidate blocks with
    an unselected task.

    Returns:
        block_id -> scheduled task IDs
    """

    target_blocks = candidate_map.get(
        task_id,
        set(),
    )

    competitions = {}

    for other_task_id in scheduled_ids:

        if other_task_id == task_id:
            continue

        other_blocks = candidate_map.get(
            other_task_id,
            set(),
        )

        shared_blocks = (
            target_blocks & other_blocks
        )

        for block_id in shared_blocks:

            competitions.setdefault(
                block_id,
                [],
            ).append(
                other_task_id
            )

    return competitions


# =========================================================
# Main analysis
# =========================================================

def main():

    print("\n" + "=" * 75)
    print("RAILBLOCK FEASIBLE COMPETITION ANALYSIS")
    print("=" * 75)

    tasks, blocks, trains, risk, schedule = (
        load_data()
    )

    # -----------------------------------------------------
    # Generate the same candidate set used by optimizer
    # -----------------------------------------------------

    candidates = generate_candidates(
        tasks,
        blocks,
    )

    print(
        f"\nCandidate assignments : {len(candidates)}"
    )

    # -----------------------------------------------------
    # Reconstruct objective scores
    # -----------------------------------------------------

    scored_tasks = calculate_priority_scores(
        tasks,
        risk,
    )

    task_info = scored_tasks.set_index(
        "task_id"
    )

    # -----------------------------------------------------
    # Candidate mapping
    # -----------------------------------------------------

    candidate_map = build_candidate_map(
        candidates
    )

    scheduled_ids = set(
        schedule["task_id"]
    )

    candidate_task_ids = set(
        candidate_map
    )

    # Tasks that had candidate blocks but weren't scheduled
    feasible_not_selected = sorted(
        candidate_task_ids - scheduled_ids
    )

    print(
        f"Feasible but not selected : "
        f"{len(feasible_not_selected)}"
    )

    # -----------------------------------------------------
    # Detailed analysis
    # -----------------------------------------------------

    competition_rows = []

    print("\n" + "=" * 75)
    print("FEASIBLE BUT NOT SELECTED TASKS")
    print("=" * 75)

    for task_id in feasible_not_selected:

        task = task_info.loc[
            task_id
        ]

        probability = float(
            task["predicted_failure_probability"]
        )

        print("\n" + "-" * 75)

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
            f"Section: {task['section_id']}"
        )

        print(
            f"Criticality: {task['criticality']}"
        )

        print(
            f"Urgency: {task['urgency']}"
        )

        print(
            f"Risk: {probability:.4f} "
            f"({task['risk_level']})"
        )

        print(
            f"Criticality score: "
            f"{task['criticality_score']}"
        )

        print(
            f"Urgency score: "
            f"{task['urgency_score']}"
        )

        print(
            f"Deadline score: "
            f"{task['deadline_score']}"
        )

        print(
            f"Risk score: "
            f"{task['risk_score']}"
        )

        print(
            f"Combined priority score: "
            f"{task['priority_score']}"
        )

        task_blocks = sorted(
            candidate_map.get(
                task_id,
                set(),
            )
        )

        print(
            f"Candidate blocks: "
            f"{len(task_blocks)}"
        )

        # -------------------------------------------------
        # Find competing scheduled tasks
        # -------------------------------------------------

        competitions = find_competing_tasks(
            task_id,
            candidate_map,
            scheduled_ids,
        )

        if not competitions:

            print(
                "Competing scheduled tasks: None identified"
            )

            continue

        print(
            "Competing scheduled tasks:"
        )

        for block_id, competing_tasks in (
            competitions.items()
        ):

            for competing_task_id in competing_tasks:

                competitor = task_info.loc[
                    competing_task_id
                ]

                print(
                    f"\n  Block: {block_id}"
                )

                print(
                    f"  Selected task: "
                    f"{competing_task_id}"
                )

                print(
                    f"  Risk: "
                    f"{float(competitor['predicted_failure_probability']):.4f}"
                )

                print(
                    f"  Criticality: "
                    f"{competitor['criticality']}"
                )

                print(
                    f"  Urgency: "
                    f"{competitor['urgency']}"
                )

                print(
                    f"  Priority score: "
                    f"{competitor['priority_score']}"
                )

                print(
                    f"  Risk score: "
                    f"{competitor['risk_score']}"
                )

                competition_rows.append(
                    {
                        "unselected_task_id": task_id,
                        "selected_task_id": (
                            competing_task_id
                        ),
                        "block_window_id": block_id,
                        "unselected_risk": probability,
                        "selected_risk": float(
                            competitor[
                                "predicted_failure_probability"
                            ]
                        ),
                        "unselected_priority": (
                            task["priority_score"]
                        ),
                        "selected_priority": (
                            competitor[
                                "priority_score"
                            ]
                        ),
                        "unselected_risk_score": (
                            task["risk_score"]
                        ),
                        "selected_risk_score": (
                            competitor[
                                "risk_score"
                            ]
                        ),
                    }
                )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    print("\n" + "=" * 75)
    print("COMPETITION SUMMARY")
    print("=" * 75)

    if competition_rows:

        competition_df = pd.DataFrame(
            competition_rows
        )

        print(
            f"Competition pairs identified: "
            f"{len(competition_df)}"
        )

        print(
            "\nPairs where unselected task has "
            "higher risk than selected task:"
        )

        higher_risk = competition_df[
            competition_df["unselected_risk"]
            > competition_df["selected_risk"]
        ]

        print(
            len(higher_risk)
        )

        print(
            "\nPairs where unselected task has "
            "higher combined priority:"
        )

        higher_priority = competition_df[
            competition_df["unselected_priority"]
            > competition_df["selected_priority"]
        ]

        print(
            len(higher_priority)
        )

        output_file = (
            OUTPUT_DIR
            / "feasible_competition_analysis.csv"
        )

        competition_df.to_csv(
            output_file,
            index=False,
        )

        print(
            f"\nCompetition output: "
            f"{output_file}"
        )

    else:

        print(
            "No direct candidate-block competitions "
            "were identified."
        )

    print("=" * 75)


if __name__ == "__main__":
    main()