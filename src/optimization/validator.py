from __future__ import annotations

from pathlib import Path

import pandas as pd


# =========================================================
# Project paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "synthetic"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

BASELINE_SCHEDULE_FILE = OUTPUT_DIR / "optimized_schedule.csv"
RISK_AWARE_SCHEDULE_FILE = OUTPUT_DIR / "risk_aware_schedule.csv"


# =========================================================
# Validation result
# =========================================================

class ValidationResult:
    """Container for schedule validation results."""

    def __init__(self):
        self.timing_violations = []
        self.block_violations = []
        self.duration_violations = []
        self.section_violations = []
        self.duplicate_tasks = []
        self.maintenance_conflicts = []
        self.train_conflicts = []

    @property
    def is_valid(self):
        """Return True when no validation errors exist."""

        return not any(
            [
                self.timing_violations,
                self.block_violations,
                self.duration_violations,
                self.section_violations,
                self.duplicate_tasks,
                self.maintenance_conflicts,
                self.train_conflicts,
            ]
        )

    @property
    def total_violations(self):
        """Return the total number of detected violations."""

        return sum(
            [
                len(self.timing_violations),
                len(self.block_violations),
                len(self.duration_violations),
                len(self.section_violations),
                len(self.duplicate_tasks),
                len(self.maintenance_conflicts),
                len(self.train_conflicts),
            ]
        )


# =========================================================
# Data loading
# =========================================================

def load_validation_data(schedule_file):
    """Load datasets required for schedule validation."""

    schedule_file = Path(schedule_file)

    if not schedule_file.exists():
        raise FileNotFoundError(
            f"Schedule file not found: {schedule_file}"
        )

    tasks = pd.read_csv(
        DATA_DIR / "maintenance_tasks.csv"
    )

    blocks = pd.read_csv(
        DATA_DIR / "block_availability.csv"
    )

    trains = pd.read_csv(
        DATA_DIR / "train_movements.csv"
    )

    schedule = pd.read_csv(schedule_file)

    return tasks, blocks, trains, schedule


# =========================================================
# Schedule validator
# =========================================================

def validate_schedule(
    tasks: pd.DataFrame,
    blocks: pd.DataFrame,
    trains: pd.DataFrame,
    schedule: pd.DataFrame,
) -> ValidationResult:
    """
    Independently validate an optimized RailBlock schedule.

    Checks:

        1. Task timing
        2. Block timing
        3. Task duration
        4. Section consistency
        5. Duplicate task assignments
        6. Maintenance conflicts
        7. Train conflicts
    """

    result = ValidationResult()

    # ---------------------------------------------------------
    # Create lookup dictionaries
    # ---------------------------------------------------------

    tasks_by_id = {
        row["task_id"]: row
        for _, row in tasks.iterrows()
    }

    blocks_by_id = {
        row["block_window_id"]: row
        for _, row in blocks.iterrows()
    }

    # ---------------------------------------------------------
    # 1. Duplicate task check
    # ---------------------------------------------------------

    duplicate_mask = schedule["task_id"].duplicated(
        keep=False
    )

    for task_id in schedule.loc[
        duplicate_mask,
        "task_id",
    ]:

        if task_id not in result.duplicate_tasks:

            result.duplicate_tasks.append(
                task_id
            )

    # ---------------------------------------------------------
    # 2. Validate individual schedule rows
    # ---------------------------------------------------------

    for _, scheduled in schedule.iterrows():

        task_id = scheduled["task_id"]
        block_id = scheduled["block_window_id"]

        # -----------------------------------------------------
        # Verify task exists
        # -----------------------------------------------------

        if task_id not in tasks_by_id:

            result.timing_violations.append(
                f"{task_id}: task not found"
            )

            continue

        # -----------------------------------------------------
        # Verify block exists
        # -----------------------------------------------------

        if block_id not in blocks_by_id:

            result.block_violations.append(
                f"{task_id}: block {block_id} not found"
            )

            continue

        task = tasks_by_id[task_id]
        block = blocks_by_id[block_id]

        scheduled_start = pd.Timestamp(
            scheduled["scheduled_start"]
        )

        scheduled_end = pd.Timestamp(
            scheduled["scheduled_end"]
        )

        # -----------------------------------------------------
        # Task temporal boundaries
        # -----------------------------------------------------

        task_earliest = pd.Timestamp(
            task["earliest_start"]
        )

        task_deadline = pd.Timestamp(
            task["deadline"]
        )

        if scheduled_start < task_earliest:

            result.timing_violations.append(
                f"{task_id}: starts before earliest_start"
            )

        if scheduled_end > task_deadline:

            result.timing_violations.append(
                f"{task_id}: finishes after deadline"
            )

        # -----------------------------------------------------
        # Block temporal boundaries
        # -----------------------------------------------------

        block_start = pd.Timestamp(
            f"{block['date']} {block['start_time']}"
        )

        block_end = pd.Timestamp(
            f"{block['date']} {block['end_time']}"
        )

        if scheduled_start < block_start:

            result.block_violations.append(
                f"{task_id}: starts before block"
            )

        if scheduled_end > block_end:

            result.block_violations.append(
                f"{task_id}: finishes after block"
            )

        # -----------------------------------------------------
        # Duration check
        # -----------------------------------------------------

        actual_duration = (
            scheduled_end - scheduled_start
        ).total_seconds() / 60

        expected_duration = float(
            task["estimated_duration_min"]
        )

        if actual_duration != expected_duration:

            result.duration_violations.append(
                f"{task_id}: expected "
                f"{expected_duration} min, got "
                f"{actual_duration} min"
            )

        # -----------------------------------------------------
        # Section consistency
        # -----------------------------------------------------

        if task["section_id"] != block["section_id"]:

            result.section_violations.append(
                f"{task_id}: task section "
                f"{task['section_id']} != "
                f"block section "
                f"{block['section_id']}"
            )

    # ---------------------------------------------------------
    # 3. Maintenance overlap check
    # ---------------------------------------------------------

    for block_id, group in schedule.groupby(
        "block_window_id"
    ):

        intervals = []

        for _, row in group.iterrows():

            start = pd.Timestamp(
                row["scheduled_start"]
            )

            end = pd.Timestamp(
                row["scheduled_end"]
            )

            intervals.append(
                (
                    start,
                    end,
                    row["task_id"],
                )
            )

        intervals.sort(
            key=lambda item: item[0]
        )

        for i in range(len(intervals) - 1):

            current = intervals[i]
            next_interval = intervals[i + 1]

            if current[1] > next_interval[0]:

                result.maintenance_conflicts.append(
                    f"{current[2]} overlaps "
                    f"{next_interval[2]} "
                    f"in block {block_id}"
                )

    # ---------------------------------------------------------
    # 4. Train conflict check
    # ---------------------------------------------------------

    for _, scheduled in schedule.iterrows():

        task_id = scheduled["task_id"]
        block_id = scheduled["block_window_id"]

        if block_id not in blocks_by_id:
            continue

        block = blocks_by_id[block_id]

        scheduled_start = pd.Timestamp(
            scheduled["scheduled_start"]
        )

        scheduled_end = pd.Timestamp(
            scheduled["scheduled_end"]
        )

        block_section = block["section_id"]

        for _, train in trains.iterrows():

            if train["section_id"] != block_section:
                continue

            train_start = pd.Timestamp(
                train["entry_time"]
            )

            train_end = pd.Timestamp(
                train["exit_time"]
            )

            # Intervals overlap when:
            #
            # start_A < end_B
            # AND
            # end_A > start_B

            if (
                scheduled_start < train_end
                and scheduled_end > train_start
            ):

                result.train_conflicts.append(
                    f"{task_id} overlaps train "
                    f"{train['movement_id']} "
                    f"in section {block_section}"
                )

    return result


# =========================================================
# Reporting
# =========================================================

def print_validation_report(
    schedule: pd.DataFrame,
    result: ValidationResult,
    schedule_name: str,
):
    """Print a human-readable validation report."""

    print("\n" + "=" * 60)
    print(f"RAILBLOCK {schedule_name.upper()} SCHEDULE VALIDATION")
    print("=" * 60)

    print(
        "Status:",
        "VALID" if result.is_valid else "INVALID",
    )

    print(
        "Tasks checked:",
        len(schedule),
    )

    print(
        "Timing violations:",
        len(result.timing_violations),
    )

    print(
        "Block violations:",
        len(result.block_violations),
    )

    print(
        "Duration violations:",
        len(result.duration_violations),
    )

    print(
        "Section violations:",
        len(result.section_violations),
    )

    print(
        "Duplicate tasks:",
        len(result.duplicate_tasks),
    )

    print(
        "Maintenance conflicts:",
        len(result.maintenance_conflicts),
    )

    print(
        "Train conflicts:",
        len(result.train_conflicts),
    )

    print(
        "Total violations:",
        result.total_violations,
    )

    print("=" * 60)


# =========================================================
# Command-line entry point
# =========================================================

def validate_file(schedule_file, schedule_name):
    """Validate one schedule file."""

    tasks, blocks, trains, schedule = (
        load_validation_data(schedule_file)
    )

    result = validate_schedule(
        tasks,
        blocks,
        trains,
        schedule,
    )

    print_validation_report(
        schedule,
        result,
        schedule_name,
    )

    return result


def main():
    """Validate baseline and risk-aware schedules."""

    baseline_result = validate_file(
        BASELINE_SCHEDULE_FILE,
        "baseline",
    )

    risk_aware_result = validate_file(
        RISK_AWARE_SCHEDULE_FILE,
        "risk-aware",
    )

    print("\n" + "=" * 60)
    print("FINAL VALIDATION COMPARISON")
    print("=" * 60)

    print(
        "Baseline valid:",
        baseline_result.is_valid,
    )

    print(
        "Risk-aware valid:",
        risk_aware_result.is_valid,
    )

    print(
        "Baseline violations:",
        baseline_result.total_violations,
    )

    print(
        "Risk-aware violations:",
        risk_aware_result.total_violations,
    )

    print("=" * 60)


if __name__ == "__main__":
    main()