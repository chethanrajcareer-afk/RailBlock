import pandas as pd

from src.optimization.candidates import generate_candidates


def test_candidate_requires_same_section():
    tasks = pd.DataFrame(
        [
            {
                "task_id": "T1",
                "section_id": "SEC001",
                "estimated_duration_min": 60,
                "earliest_start": "2026-01-05T08:00:00",
                "deadline": "2026-01-05T18:00:00",
            }
        ]
    )

    blocks = pd.DataFrame(
        [
            {
                "block_window_id": "B1",
                "section_id": "SEC002",
                "date": "2026-01-05",
                "start_time": "10:00",
                "end_time": "12:00",
                "availability_status": "available",
            }
        ]
    )

    candidates = generate_candidates(tasks, blocks)

    assert len(candidates) == 0


def test_candidate_must_finish_before_deadline():
    tasks = pd.DataFrame(
        [
            {
                "task_id": "T1",
                "section_id": "SEC001",
                "estimated_duration_min": 120,
                "earliest_start": "2026-01-05T08:00:00",
                "deadline": "2026-01-05T10:00:00",
            }
        ]
    )

    blocks = pd.DataFrame(
        [
            {
                "block_window_id": "B1",
                "section_id": "SEC001",
                "date": "2026-01-05",
                "start_time": "09:00",
                "end_time": "11:00",
                "availability_status": "available",
            }
        ]
    )

    candidates = generate_candidates(tasks, blocks)

    assert len(candidates) == 0