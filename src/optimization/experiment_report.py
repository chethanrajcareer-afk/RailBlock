"""Create an auditable report for one RailBlock risk-aware solver run.

The report is deliberately a JSON artifact rather than console-only output.
It records which synthetic inputs, configuration values, and solver results
produced a schedule, making an experiment easier to reproduce and compare.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


REPORT_SCHEMA_VERSION = 1


def sha256_file(path: Path) -> str:
    """Return a SHA-256 fingerprint for one experiment artifact."""

    digest = hashlib.sha256()

    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(65_536), b""):
            digest.update(chunk)

    return digest.hexdigest()


def git_revision(project_root: Path) -> str:
    """Return the current Git revision, or a clear fallback outside Git."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"

    return result.stdout.strip()


def git_worktree_clean(project_root: Path) -> bool | None:
    """Return whether the source tree is clean, when Git is available."""

    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None

    return not result.stdout.strip()


def artifact_metadata(
    path: Path,
    project_root: Path,
    row_count: int | None = None,
) -> dict[str, Any]:
    """Describe a file used or produced by a scheduling experiment."""

    try:
        relative_path = path.relative_to(project_root).as_posix()
    except ValueError:
        relative_path = str(path)

    metadata: dict[str, Any] = {
        "path": relative_path,
        "sha256": sha256_file(path),
    }

    if row_count is not None:
        metadata["rows"] = int(row_count)

    return metadata


def validation_summary(validation: Any | None) -> dict[str, Any]:
    """Convert the independent validator result into JSON-safe metrics."""

    if validation is None:
        return {
            "status": "NOT_RUN",
            "total_violations": None,
        }

    return {
        "status": "VALID" if validation.is_valid else "INVALID",
        "total_violations": int(validation.total_violations),
        "timing_violations": len(validation.timing_violations),
        "block_violations": len(validation.block_violations),
        "duration_violations": len(validation.duration_violations),
        "section_violations": len(validation.section_violations),
        "duplicate_tasks": len(validation.duplicate_tasks),
        "maintenance_conflicts": len(validation.maintenance_conflicts),
        "train_conflicts": len(validation.train_conflicts),
    }


def write_experiment_report(
    *,
    project_root: Path,
    report_path: Path,
    input_paths: dict[str, Path],
    source_paths: dict[str, Path] | None,
    tasks: pd.DataFrame,
    blocks: pd.DataFrame,
    trains: pd.DataFrame,
    risk_scores: pd.DataFrame,
    schedule: pd.DataFrame,
    schedule_path: Path,
    risk_weight: int,
    max_time_seconds: float,
    num_workers: int,
    solver_status: str,
    objective_value: float | None,
    wall_time_seconds: float,
    num_conflicts: int,
    num_branches: int,
    candidate_assignment_count: int,
    tasks_with_candidates: int,
    blocks_used: int,
    validation: Any | None,
    generated_at_utc: str | None = None,
    revision: str | None = None,
) -> Path:
    """Write one self-describing report for a risk-aware solver run."""

    total_tasks = len(tasks)
    scheduled_tasks = len(schedule)
    tasks_without_candidates = total_tasks - tasks_with_candidates

    input_rows = {
        "maintenance_tasks": len(tasks),
        "block_availability": len(blocks),
        "train_movements": len(trains),
        "asset_risk_scores": len(risk_scores),
    }

    inputs = {
        name: artifact_metadata(path, project_root, input_rows[name])
        for name, path in input_paths.items()
    }

    coverage_pct = (
        round(scheduled_tasks / total_tasks * 100, 2)
        if total_tasks
        else 0.0
    )
    candidate_task_coverage_pct = (
        round(tasks_with_candidates / total_tasks * 100, 2)
        if total_tasks
        else 0.0
    )

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at_utc": generated_at_utc
        or datetime.now(timezone.utc).isoformat(),
        "git_revision": revision or git_revision(project_root),
        "git_worktree_clean": git_worktree_clean(project_root),
        "scope": {
            "project": "RailBlock",
            "model": "risk_aware_cp_sat",
            "data": "fictional_synthetic",
        },
        "inputs": inputs,
        "configuration": {
            "risk_weight": int(risk_weight),
            "max_time_seconds": float(max_time_seconds),
            "num_workers": int(num_workers),
        },
        "solver": {
            "status": solver_status,
            "objective_value": (
                round(float(objective_value), 6)
                if objective_value is not None
                else None
            ),
            "wall_time_seconds": round(float(wall_time_seconds), 6),
            "num_conflicts": int(num_conflicts),
            "num_branches": int(num_branches),
        },
        "candidates": {
            "candidate_assignments": int(candidate_assignment_count),
            "tasks_with_model_feasible_candidates": int(
                tasks_with_candidates
            ),
            "tasks_without_model_feasible_candidates": int(
                tasks_without_candidates
            ),
            "candidate_task_coverage_pct": candidate_task_coverage_pct,
        },
        "schedule": {
            "tasks_total": int(total_tasks),
            "tasks_scheduled": int(scheduled_tasks),
            "tasks_unscheduled": int(total_tasks - scheduled_tasks),
            "schedule_coverage_pct": coverage_pct,
            "blocks_used": int(blocks_used),
            "artifact": artifact_metadata(
                schedule_path,
                project_root,
                scheduled_tasks,
            ),
        },
        "validation": validation_summary(validation),
    }

    if source_paths:
        report["source_artifacts"] = {
            name: artifact_metadata(path, project_root)
            for name, path in source_paths.items()
        }

    report_path.parent.mkdir(parents=True, exist_ok=True)

    with report_path.open("w", encoding="utf-8") as file_handle:
        json.dump(report, file_handle, indent=2, sort_keys=True)
        file_handle.write("\n")

    return report_path
