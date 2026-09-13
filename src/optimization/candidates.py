import pandas as pd


def generate_candidates(tasks_df: pd.DataFrame, blocks_df: pd.DataFrame):
    """
    Generate feasible task -> block assignments.

    A task can be assigned to a block when:
    1. The block is available.
    2. The task and block belong to the same section.
    3. The block is long enough for the task.
    4. The block ends after the task's earliest start.
    5. The block starts before the task's deadline.

    Returns:
        list[dict]: Candidate task-block assignments.
    """

    candidates = []

    for _, task in tasks_df.iterrows():

        task_earliest = pd.Timestamp(task["earliest_start"])
        task_deadline = pd.Timestamp(task["deadline"])
        task_duration = int(task["estimated_duration_min"])

        for _, block in blocks_df.iterrows():

            # 1. Block must be available
            if block["availability_status"] != "available":
                continue

            # 2. Task and block must be on the same section
            if task["section_id"] != block["section_id"]:
                continue

            block_start = pd.Timestamp(
                f"{block['date']} {block['start_time']}"
            )
            block_end = pd.Timestamp(
                f"{block['date']} {block['end_time']}"
            )

            block_duration = int(
                (block_end - block_start).total_seconds() / 60
            )

            # 3. Block must be long enough
            if block_duration < task_duration:
                continue

            # 4. Block must end after task's earliest start
            if block_end < task_earliest:
                continue

            # 5. Task must be able to finish before the deadline
            possible_start = max(block_start, task_earliest)
            possible_end = possible_start + pd.Timedelta(minutes=task_duration)

            if possible_end > block_end:
             continue

            if possible_end > task_deadline:
             continue

            candidates.append(
                {
                    "task_id": task["task_id"],
                    "block_window_id": block["block_window_id"],
                    "section_id": task["section_id"],
                    "task_duration_min": task_duration,
                    "block_start": block_start,
                    "block_end": block_end,
                    "block_duration_min": block_duration,
                }
            )

    return candidates