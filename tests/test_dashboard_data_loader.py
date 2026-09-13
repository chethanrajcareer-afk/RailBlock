"""Tests for the dashboard's read-only, validated data-loading boundary."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from dashboard.utils.data_loader import DashboardDataError, load_dashboard_data


def test_dashboard_loader_uses_existing_validated_synthetic_data() -> None:
    data = load_dashboard_data()

    assert len(data["stations"]) == 18
    assert len(data["sections"]) == 17
    assert len(data["maintenance_tasks"]) == 108
    assert len(data["train_movements"]) == 560
    assert len(data["block_availability"]) == 357
    assert pd.api.types.is_datetime64_any_dtype(data["maintenance_tasks"]["earliest_start"])
    assert pd.api.types.is_datetime64_any_dtype(data["train_movements"]["service_date"])


def test_dashboard_loader_reports_missing_expected_files(tmp_path: Path) -> None:
    with pytest.raises(DashboardDataError, match="Expected Phase 0 synthetic data is missing"):
        load_dashboard_data(tmp_path)
