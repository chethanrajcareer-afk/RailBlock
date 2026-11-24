import json
from pathlib import Path

import pandas as pd

from src.optimization.experiment_report import write_experiment_report
from src.optimization.validator import ValidationResult


def _write_artifact(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def test_write_experiment_report_records_auditable_run_metadata(
    tmp_path: Path,
) -> None:
    inputs = {
        "maintenance_tasks": _write_artifact(
            tmp_path / "maintenance_tasks.csv", "task_id\nT1\nT2\n"
        ),
        "block_availability": _write_artifact(
            tmp_path / "block_availability.csv", "block_window_id\nB1\n"
        ),
        "train_movements": _write_artifact(
            tmp_path / "train_movements.csv", "train_id\nTR1\n"
        ),
        "asset_risk_scores": _write_artifact(
            tmp_path / "asset_risk_scores.csv", "asset_id\nA1\nA2\n"
        ),
    }
    schedule_path = _write_artifact(
        tmp_path / "risk_aware_schedule.csv", "task_id\nT1\n"
    )
    report_path = tmp_path / "risk_aware_experiment_report.json"

    report = write_experiment_report(
        project_root=tmp_path,
        report_path=report_path,
        input_paths=inputs,
        source_paths={"report": inputs["maintenance_tasks"]},
        tasks=pd.DataFrame({"task_id": ["T1", "T2"]}),
        blocks=pd.DataFrame({"block_window_id": ["B1"]}),
        trains=pd.DataFrame({"train_id": ["TR1"]}),
        risk_scores=pd.DataFrame({"asset_id": ["A1", "A2"]}),
        schedule=pd.DataFrame({"task_id": ["T1"]}),
        schedule_path=schedule_path,
        risk_weight=100,
        max_time_seconds=30,
        num_workers=8,
        solver_status="OPTIMAL",
        objective_value=123.456789,
        wall_time_seconds=0.25,
        num_conflicts=4,
        num_branches=12,
        candidate_assignment_count=3,
        tasks_with_candidates=1,
        blocks_used=1,
        validation=ValidationResult(),
        generated_at_utc="2026-11-24T00:00:00+00:00",
        revision="test-revision",
    )

    assert report == report_path

    payload = json.loads(report_path.read_text(encoding="utf-8"))

    assert payload["git_revision"] == "test-revision"
    assert payload["git_worktree_clean"] is None
    assert payload["configuration"]["risk_weight"] == 100
    assert payload["solver"]["status"] == "OPTIMAL"
    assert payload["solver"]["objective_value"] == 123.456789
    assert payload["candidates"] == {
        "candidate_assignments": 3,
        "tasks_with_model_feasible_candidates": 1,
        "tasks_without_model_feasible_candidates": 1,
        "candidate_task_coverage_pct": 50.0,
    }
    assert payload["schedule"]["schedule_coverage_pct"] == 50.0
    assert payload["schedule"]["artifact"]["rows"] == 1
    assert len(payload["inputs"]["maintenance_tasks"]["sha256"]) == 64
    assert len(payload["source_artifacts"]["report"]["sha256"]) == 64
    assert payload["validation"]["status"] == "VALID"
