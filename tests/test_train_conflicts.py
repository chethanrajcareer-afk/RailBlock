import pandas as pd
import pytest
from ortools.sat.python import cp_model

from src.optimization.model import (
    OptimizationData as BaselineOptimizationData,
)
from src.optimization.model import build_model
from src.optimization.model_risk_aware import (
    OptimizationData as RiskAwareOptimizationData,
)
from src.optimization.model_risk_aware import build_risk_aware_model
from src.optimization.validator import validate_schedule


def _tasks() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "task_id": "TASK-1",
                "asset_id": "ASSET-1",
                "department": "Engineering",
                "section_id": "SEC-1",
                "estimated_duration_min": 60,
                "earliest_start": "2026-01-10 10:00:00",
                "deadline": "2026-01-10 11:00:00",
                "criticality": "high",
                "urgency": "high",
            }
        ]
    )


def _blocks() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "block_window_id": "BLOCK-1",
                "section_id": "SEC-1",
                "date": "2026-01-10",
                "start_time": "10:00:00",
                "end_time": "11:00:00",
                "availability_status": "available",
            }
        ]
    )


def _trains(service_date: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "movement_id": "TRAIN-1",
                "section_id": "SEC-1",
                "service_date": service_date,
                "entry_time": "10:15:00",
                "exit_time": "10:45:00",
            }
        ]
    )


def _schedule() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "task_id": "TASK-1",
                "block_window_id": "BLOCK-1",
                "scheduled_start": "2026-01-10 10:00:00",
                "scheduled_end": "2026-01-10 11:00:00",
            }
        ]
    )


@pytest.mark.parametrize("risk_aware", [False, True])
def test_models_reject_task_forced_to_overlap_train_on_same_service_date(
    risk_aware: bool,
) -> None:
    """A train on the block date makes the forced assignment infeasible."""
    tasks = _tasks()
    blocks = _blocks()
    trains = _trains("2026-01-10")

    if risk_aware:
        data = RiskAwareOptimizationData(tasks, blocks, trains)
        risk_scores = pd.DataFrame(
            {
                "asset_id": ["ASSET-1"],
                "predicted_failure_probability": [0.2],
            }
        )
        model, assignment_vars, *_ = build_risk_aware_model(
            data,
            risk_scores,
        )
    else:
        data = BaselineOptimizationData(tasks, blocks, trains)
        model, assignment_vars, *_ = build_model(data)

    assignment_var = next(iter(assignment_vars.values()))
    model.Add(assignment_var == 1)

    status = cp_model.CpSolver().Solve(model)

    assert status == cp_model.INFEASIBLE


@pytest.mark.parametrize("risk_aware", [False, True])
def test_models_allow_same_clock_time_on_a_different_service_date(
    risk_aware: bool,
) -> None:
    """A train on another date must not block the maintenance window."""
    tasks = _tasks()
    blocks = _blocks()
    trains = _trains("2026-01-11")

    if risk_aware:
        data = RiskAwareOptimizationData(tasks, blocks, trains)
        risk_scores = pd.DataFrame(
            {
                "asset_id": ["ASSET-1"],
                "predicted_failure_probability": [0.2],
            }
        )
        model, assignment_vars, *_ = build_risk_aware_model(
            data,
            risk_scores,
        )
    else:
        data = BaselineOptimizationData(tasks, blocks, trains)
        model, assignment_vars, *_ = build_model(data)

    assignment_var = next(iter(assignment_vars.values()))
    model.Add(assignment_var == 1)

    status = cp_model.CpSolver().Solve(model)

    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)


def test_validator_reports_train_conflict_on_same_service_date() -> None:
    result = validate_schedule(
        _tasks(),
        _blocks(),
        _trains("2026-01-10"),
        _schedule(),
    )

    assert result.train_conflicts == [
        "TASK-1 overlaps train TRAIN-1 in section SEC-1"
    ]


def test_validator_allows_same_clock_time_on_different_service_date() -> None:
    result = validate_schedule(
        _tasks(),
        _blocks(),
        _trains("2026-01-11"),
        _schedule(),
    )

    assert result.train_conflicts == []
