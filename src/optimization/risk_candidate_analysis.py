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
# Risk thresholds
# =========================================================

HIGH_RISK_THRESHOLD = 0.50
CRITICAL_RISK_THRESHOLD = 0.75


# =========================================================
# Data loading
# =========================================================

def load_data():
    """Load all datasets required for candidate analysis."""

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
# Train conflict helper
# =========================================================

def has_train_conflict(
    candidate: dict,
    trains: pd.DataFrame,
) -> bool:
    """Return True when the maintenance interval overlaps a train."""

    section_id = candidate["section_id"]
    start = pd.Timestamp(candidate["block_start"])
    duration = int(candidate["task_duration_min"])
    end = start + pd.Timedelta(minutes=duration)

    for _, train in trains.iterrows():

        if train["section_id"] != section_id:
            continue

        train_start = pd.Timestamp(train["entry_time"])
        train_end = pd.Timestamp(train["exit_time"])

        if (
            start < train_end
            and end > train_start
        ):
            return True

    return False
    """Return True when a candidate overlaps a train movement."""

    section_id = candidate["section_id"]
    block_start = pd.Timestamp(candidate["block_start"])
    block_end = pd.Timestamp(candidate["block_end"])

    for _, train in trains.iterrows():

        if train["section_id"] != section_id:
            continue

        train_start = pd.Timestamp(train["entry_time"])
        train_end = pd.Timestamp(train["exit_time"])

        # Candidate and train intervals overlap when:
        #
        # candidate_start < train_end
        # AND
        # candidate_end > train_start

        if (
            block_start < train_end
            and block_end > train_start
        ):
            return True

    return False


# =========================================================
# Candidate-level analysis
# =========================================================

def analyse_task(
    task_id: str,
    candidates: list[dict],
    trains: pd.DataFrame,
    scheduled_task_ids: set[str],
) -> dict:
    """Analyse candidate availability for one task."""

    task_candidates = [
        candidate
        for candidate in candidates
        if candidate["task_id"] == task_id
    ]

    if not task_candidates:

        return {
            "task_id": task_id,
            "candidate_blocks": 0,
            "train_conflict_candidates": 0,
            "feasible_candidates": 0,
            "status": "NO_FEASIBLE_BLOCK",
            "reason": (
                "No candidate block can accommodate "
                "the task within its timing requirements."
            ),
        }

    train_conflicts = 0
    feasible_candidates = 0

    for candidate in task_candidates:

        if has_train_conflict(candidate, trains):
            train_conflicts += 1
        else:
            feasible_candidates += 1

    if task_id in scheduled_task_ids:

        status = "SCHEDULED"
        reason = (
            "The task was scheduled using a feasible "
            "candidate block."
        )

    elif feasible_candidates > 0:

        status = "FEASIBLE_NOT_SELECTED"
        reason = (
            "At least one feasible candidate block exists, "
            "but the optimizer did not select the task."
        )

    elif train_conflicts == len(task_candidates):

        status = "TRAIN_CONFLICT"
        reason = (
            "Candidate blocks exist, but all candidate "
            "windows overlap scheduled train movements."
        )

    else:

        status = "NO_FEASIBLE_BLOCK"
        reason = (
            "Candidate blocks exist, but none remain "
            "feasible after operational constraints."
        )

    return {
        "task_id": task_id,
        "candidate_blocks": len(task_candidates),
        "train_conflict_candidates": train_conflicts,
        "feasible_candidates": feasible_candidates,
        "status": status,
        "reason": reason,
    }


# =========================================================
# Main analysis
# =========================================================

def main():

    print("\n" + "=" * 70)
    print("RAILBLOCK RISK CANDIDATE FEASIBILITY ANALYSIS")
    print("=" * 70)

    tasks, blocks, trains, risk, schedule = load_data()

    # -----------------------------------------------------
    # Generate the same candidate set used by the optimizer
    # -----------------------------------------------------

    candidates = generate_candidates(
        tasks,
        blocks,
    )

    print(f"\nMaintenance tasks       : {len(tasks)}")
    print(f"Candidate assignments   : {len(candidates)}")
    print(f"Risk scores             : {len(risk)}")
    print(f"Scheduled tasks         : {len(schedule)}")

    # -----------------------------------------------------
    # Risk lookup
    # -----------------------------------------------------

    risk_by_asset = dict(
        zip(
            risk["asset_id"],
            risk["predicted_failure_probability"],
        )
    )

    # -----------------------------------------------------
    # Scheduled task IDs
    # -----------------------------------------------------

    scheduled_task_ids = set(
        schedule["task_id"]
    )

    # -----------------------------------------------------
    # Analyse every task
    # -----------------------------------------------------

    analysis_rows = []

    for _, task in tasks.iterrows():

        task_id = task["task_id"]
        asset_id = task["asset_id"]

        probability = float(
            risk_by_asset.get(
                asset_id,
                0.0,
            )
        )

        if probability >= CRITICAL_RISK_THRESHOLD:
            risk_level = "Critical"

        elif probability >= HIGH_RISK_THRESHOLD:
            risk_level = "High"

        elif probability >= 0.25:
            risk_level = "Medium"

        else:
            risk_level = "Low"

        result = analyse_task(
            task_id,
            candidates,
            trains,
            scheduled_task_ids,
        )

        result.update(
            {
                "asset_id": asset_id,
                "department": task["department"],
                "section_id": task["section_id"],
                "criticality": task["criticality"],
                "estimated_duration_min": task[
                    "estimated_duration_min"
                ],
                "predicted_failure_probability": probability,
                "risk_level": risk_level,
            }
        )

        analysis_rows.append(result)

    analysis = pd.DataFrame(
        analysis_rows
    )

    # -----------------------------------------------------
    # Print high-risk analysis
    # -----------------------------------------------------

    high_risk = analysis[
        analysis["predicted_failure_probability"]
        >= HIGH_RISK_THRESHOLD
    ].copy()

    high_risk = high_risk.sort_values(
        "predicted_failure_probability",
        ascending=False,
    )

    print("\n" + "=" * 70)
    print("HIGH-RISK TASK FEASIBILITY")
    print("=" * 70)

    columns = [
        "task_id",
        "asset_id",
        "department",
        "section_id",
        "criticality",
        "estimated_duration_min",
        "predicted_failure_probability",
        "risk_level",
        "candidate_blocks",
        "train_conflict_candidates",
        "feasible_candidates",
        "status",
    ]

    print(
        high_risk[columns].to_string(
            index=False
        )
    )

    # -----------------------------------------------------
    # Detailed critical-risk analysis
    # -----------------------------------------------------

    critical_risk = analysis[
        analysis["predicted_failure_probability"]
        >= CRITICAL_RISK_THRESHOLD
    ].copy()

    print("\n" + "=" * 70)
    print("CRITICAL-RISK TASKS")
    print("=" * 70)

    if critical_risk.empty:

        print("No critical-risk tasks found.")

    else:

        for _, row in critical_risk.iterrows():

            print(
                f"\nTask: {row['task_id']}"
            )

            print(
                f"Asset: {row['asset_id']}"
            )

            print(
                f"Risk: "
                f"{row['predicted_failure_probability']:.4f}"
            )

            print(
                f"Risk level: {row['risk_level']}"
            )

            print(
                f"Candidate blocks: "
                f"{row['candidate_blocks']}"
            )

            print(
                f"Train-conflicted candidates: "
                f"{row['train_conflict_candidates']}"
            )

            print(
                f"Feasible candidates: "
                f"{row['feasible_candidates']}"
            )

            print(
                f"Decision: {row['status']}"
            )

            print(
                f"Reason: {row['reason']}"
            )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("FEASIBILITY SUMMARY")
    print("=" * 70)

    print(
        analysis["status"]
        .value_counts()
        .to_string()
    )

    print("\nHigh-risk tasks:")

    print(
        high_risk["status"]
        .value_counts()
        .to_string()
    )

    # -----------------------------------------------------
    # Save analysis
    # -----------------------------------------------------

    output_file = (
        OUTPUT_DIR
        / "risk_candidate_feasibility.csv"
    )

    analysis.to_csv(
        output_file,
        index=False,
    )

    print(
        f"\nAnalysis output: {output_file}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()