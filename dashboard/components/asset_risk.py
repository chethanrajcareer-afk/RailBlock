"""Asset-risk dashboard component for RailBlock."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


RISK_LEVEL_ORDER = [
    "Critical",
    "High",
    "Medium",
    "Low",
]


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def _project_root() -> Path:
    """Return the RailBlock repository root."""

    return Path(__file__).resolve().parents[2]


def _risk_file() -> Path:
    """Return the generated asset-risk score file."""

    return _project_root() / "outputs" / "asset_risk_scores.csv"


def _maintenance_file() -> Path:
    """Return the synthetic maintenance-task dataset."""

    return _project_root() / "data" / "synthetic" / "maintenance_tasks.csv"


def _schedule_file() -> Path:
    """Return the optimized maintenance schedule."""

    return _project_root() / "outputs" / "optimized_schedule.csv"


# ---------------------------------------------------------------------------
# Asset risk data
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading asset risk scores...")
def load_asset_risk_scores() -> pd.DataFrame:
    """Load the latest generated asset-risk scores."""

    path = _risk_file()

    if not path.is_file():
        raise FileNotFoundError(
            f"Asset risk output was not found: {path}. "
            "Run the Phase 2.3 risk-scoring pipeline first."
        )

    risk = pd.read_csv(path)

    required_columns = {
        "asset_id",
        "department",
        "asset_type",
        "section_id",
        "criticality",
        "snapshot_date",
        "predicted_failure_probability",
        "risk_level",
    }

    missing = required_columns.difference(risk.columns)

    if missing:
        raise ValueError(
            "Asset risk output is missing required columns: "
            + ", ".join(sorted(missing))
        )

    risk["predicted_failure_probability"] = pd.to_numeric(
        risk["predicted_failure_probability"],
        errors="coerce",
    )

    risk["snapshot_date"] = risk["snapshot_date"].astype(str)
    risk["risk_level"] = risk["risk_level"].astype(str)

    if risk["predicted_failure_probability"].isna().any():
        raise ValueError(
            "Asset risk output contains invalid predicted failure probabilities."
        )

    return (
        risk.sort_values(
            "predicted_failure_probability",
            ascending=False,
        )
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Maintenance / optimization integration
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading maintenance planning results...")
def load_maintenance_decisions() -> pd.DataFrame:
    """
    Connect maintenance tasks with the optimized schedule.

    Every maintenance task is retained. Scheduled tasks are enriched with
    their assigned block and timing. Tasks absent from the optimized schedule
    are marked as not scheduled.
    """

    maintenance_path = _maintenance_file()
    schedule_path = _schedule_file()

    if not maintenance_path.is_file():
        raise FileNotFoundError(
            f"Maintenance dataset was not found: {maintenance_path}."
        )

    if not schedule_path.is_file():
        raise FileNotFoundError(
            f"Optimized schedule was not found: {schedule_path}. "
            "Run the optimization engine first."
        )

    tasks = pd.read_csv(maintenance_path)
    schedule = pd.read_csv(schedule_path)

    required_task_columns = {
        "task_id",
        "asset_id",
        "department",
        "section_id",
        "task_type",
        "criticality",
        "estimated_duration_min",
    }

    missing_task_columns = required_task_columns.difference(
        tasks.columns
    )

    if missing_task_columns:
        raise ValueError(
            "Maintenance dataset is missing required columns: "
            + ", ".join(sorted(missing_task_columns))
        )

    if "task_id" not in schedule.columns:
        raise ValueError(
            "Optimized schedule is missing the required 'task_id' column."
        )

    # Keep only one schedule record per task.
    schedule = schedule.drop_duplicates(
        subset=["task_id"],
        keep="first",
    ).copy()

    # Rename schedule columns when they exist so the final dataframe has
    # predictable dashboard-friendly names.
    rename_map = {}

    if "block_window_id" in schedule.columns:
        rename_map["block_window_id"] = "assigned_block"

    if "block_date" in schedule.columns:
        rename_map["block_date"] = "assigned_date"

    if "scheduled_start" in schedule.columns:
        rename_map["scheduled_start"] = "assigned_start"

    if "scheduled_end" in schedule.columns:
        rename_map["scheduled_end"] = "assigned_end"

    schedule = schedule.rename(columns=rename_map)

    schedule_columns = ["task_id"]

    for column in (
        "assigned_block",
        "assigned_date",
        "assigned_start",
        "assigned_end",
    ):
        if column in schedule.columns:
            schedule_columns.append(column)

    schedule = schedule[schedule_columns]

    decisions = tasks.merge(
        schedule,
        on="task_id",
        how="left",
    )

    decisions["planning_status"] = decisions[
        "assigned_block"
    ].notna().map(
        {
            True: "Scheduled",
            False: "Not scheduled",
        }
    )

    decisions["planning_status_icon"] = decisions[
        "planning_status"
    ].map(
        {
            "Scheduled": "Scheduled",
            "Not scheduled": "Not scheduled",
        }
    )

    return decisions


def _merge_risk_with_planning(
    risk: pd.DataFrame,
    decisions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach maintenance planning information to asset-risk predictions.

    Assets can have multiple maintenance tasks. In that case the asset is
    considered scheduled if at least one associated task is scheduled.
    """

    task_decisions = decisions.copy()

    scheduled_by_asset = (
        task_decisions.groupby("asset_id")
        .agg(
            maintenance_task_count=("task_id", "count"),
            scheduled_task_count=(
                "planning_status",
                lambda values: int(
                    (values == "Scheduled").sum()
                ),
            ),
        )
        .reset_index()
    )

    task_decisions["planning_status_rank"] = task_decisions[
        "planning_status"
    ].map(
        {
            "Scheduled": 1,
            "Not scheduled": 0,
        }
    )

    # Prefer a scheduled task when an asset has both scheduled and
    # unscheduled maintenance tasks.
    representative = (
        task_decisions.sort_values(
            [
                "asset_id",
                "planning_status_rank",
            ],
            ascending=[True, False],
        )
        .drop_duplicates(
            subset=["asset_id"],
            keep="first",
        )
        .copy()
    )

    representative_columns = [
        "asset_id",
        "task_id",
        "task_type",
        "estimated_duration_min",
        "planning_status",
    ]

    for column in (
        "assigned_block",
        "assigned_date",
        "assigned_start",
        "assigned_end",
    ):
        if column in representative.columns:
            representative_columns.append(column)

    representative = representative[
        representative_columns
    ]

    representative = representative.rename(
        columns={
            "task_id": "maintenance_task_id",
            "task_type": "maintenance_work",
            "estimated_duration_min": "maintenance_duration_min",
        }
    )

    merged = risk.merge(
        scheduled_by_asset,
        on="asset_id",
        how="left",
    )

    merged = merged.merge(
        representative,
        on="asset_id",
        how="left",
    )

    merged["maintenance_task_count"] = (
        merged["maintenance_task_count"]
        .fillna(0)
        .astype(int)
    )

    merged["scheduled_task_count"] = (
        merged["scheduled_task_count"]
        .fillna(0)
        .astype(int)
    )

    merged["asset_planning_status"] = merged[
        "scheduled_task_count"
    ].map(
        lambda count: (
            "Scheduled"
            if count > 0
            else (
                "Not scheduled"
                if count == 0
                else "Not scheduled"
            )
        )
    )

    merged.loc[
        merged["maintenance_task_count"] == 0,
        "asset_planning_status",
    ] = "No maintenance task"

    return merged


# ---------------------------------------------------------------------------
# KPI section
# ---------------------------------------------------------------------------

def _render_kpis(risk: pd.DataFrame) -> None:
    """Render risk-level counts."""

    counts = (
        risk["risk_level"]
        .value_counts()
        .reindex(
            RISK_LEVEL_ORDER,
            fill_value=0,
        )
    )

    columns = st.columns(4)

    columns[0].metric(
        "Critical",
        int(counts["Critical"]),
    )

    columns[1].metric(
        "High",
        int(counts["High"]),
    )

    columns[2].metric(
        "Medium",
        int(counts["Medium"]),
    )

    columns[3].metric(
        "Low",
        int(counts["Low"]),
    )


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def _render_distribution(
    risk: pd.DataFrame,
) -> None:
    """Render the overall risk-level distribution."""

    counts = (
        risk["risk_level"]
        .value_counts()
        .reindex(
            RISK_LEVEL_ORDER,
            fill_value=0,
        )
        .reset_index()
    )

    counts.columns = [
        "risk_level",
        "assets",
    ]

    figure = px.bar(
        counts,
        x="risk_level",
        y="assets",
        category_orders={
            "risk_level": RISK_LEVEL_ORDER,
        },
        text="assets",
        title="Asset risk distribution",
    )

    figure.update_layout(
        xaxis_title="Risk level",
        yaxis_title="Assets",
        showlegend=False,
    )

    st.plotly_chart(
        figure,
        width="stretch",
    )


def _render_department_distribution(
    risk: pd.DataFrame,
) -> None:
    """Render risk-level counts by department."""

    grouped = (
        risk.groupby(
            [
                "department",
                "risk_level",
            ],
            as_index=False,
        )
        .size()
        .rename(
            columns={
                "size": "assets",
            }
        )
    )

    figure = px.bar(
        grouped,
        x="department",
        y="assets",
        color="risk_level",
        category_orders={
            "risk_level": RISK_LEVEL_ORDER,
        },
        barmode="stack",
        title="Risk distribution by department",
    )

    figure.update_layout(
        xaxis_title="Department",
        yaxis_title="Assets",
        legend_title="Risk level",
    )

    st.plotly_chart(
        figure,
        width="stretch",
    )


# ---------------------------------------------------------------------------
# Highest-risk assets
# ---------------------------------------------------------------------------

def _render_top_risk_assets(
    risk: pd.DataFrame,
) -> None:
    """Render the highest-risk assets with planning status."""

    st.subheader("Highest-risk assets")

    top = risk.head(10).copy()

    top["predicted_failure_probability"] = (
        top["predicted_failure_probability"] * 100
    ).round(2)

    display_columns = [
        "asset_id",
        "department",
        "asset_type",
        "section_id",
        "criticality",
        "predicted_failure_probability",
        "risk_level",
        "asset_planning_status",
    ]

    display = top[
        [
            column
            for column in display_columns
            if column in top.columns
        ]
    ].rename(
        columns={
            "asset_id": "Asset",
            "department": "Department",
            "asset_type": "Asset type",
            "section_id": "Section",
            "criticality": "Criticality",
            "predicted_failure_probability": (
                "Predicted failure risk (%)"
            ),
            "risk_level": "Risk level",
            "asset_planning_status": "Planning status",
        }
    )

    st.dataframe(
        display,
        width="stretch",
        hide_index=True,
    )


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

def _render_filters(
    risk: pd.DataFrame,
) -> pd.DataFrame:
    """Apply interactive risk filters."""

    st.subheader("Risk filters")

    columns = st.columns(4)

    departments = sorted(
        risk["department"]
        .dropna()
        .unique()
    )

    sections = sorted(
        risk["section_id"]
        .dropna()
        .unique()
    )

    selected_departments = columns[0].multiselect(
        "Department",
        departments,
    )

    selected_sections = columns[1].multiselect(
        "Section",
        sections,
    )

    selected_levels = columns[2].multiselect(
        "Risk level",
        RISK_LEVEL_ORDER,
    )

    selected_status = columns[3].multiselect(
        "Planning status",
        [
            "Scheduled",
            "Not scheduled",
            "No maintenance task",
        ],
    )

    filtered = risk.copy()

    if selected_departments:
        filtered = filtered[
            filtered["department"].isin(
                selected_departments
            )
        ]

    if selected_sections:
        filtered = filtered[
            filtered["section_id"].isin(
                selected_sections
            )
        ]

    if selected_levels:
        filtered = filtered[
            filtered["risk_level"].isin(
                selected_levels
            )
        ]

    if selected_status:
        filtered = filtered[
            filtered["asset_planning_status"].isin(
                selected_status
            )
        ]

    return filtered


# ---------------------------------------------------------------------------
# Asset risk register
# ---------------------------------------------------------------------------

def _render_asset_table(
    risk: pd.DataFrame,
) -> None:
    """Render the filtered asset-risk table."""

    st.subheader("Asset risk register")

    if risk.empty:
        st.warning(
            "No assets match the selected filters."
        )
        return

    display = risk.copy()

    display["predicted_failure_probability"] = (
        display["predicted_failure_probability"] * 100
    ).round(2)

    display_columns = [
        "asset_id",
        "department",
        "asset_type",
        "section_id",
        "criticality",
        "snapshot_date",
        "predicted_failure_probability",
        "risk_level",
        "asset_planning_status",
        "maintenance_task_id",
        "assigned_block",
    ]

    display = display[
        [
            column
            for column in display_columns
            if column in display.columns
        ]
    ].rename(
        columns={
            "asset_id": "Asset",
            "department": "Department",
            "asset_type": "Asset type",
            "section_id": "Section",
            "criticality": "Criticality",
            "snapshot_date": "Snapshot",
            "predicted_failure_probability": (
                "Predicted failure risk (%)"
            ),
            "risk_level": "Risk level",
            "asset_planning_status": "Planning status",
            "maintenance_task_id": "Maintenance task",
            "assigned_block": "Assigned block",
        }
    )

    st.caption(
        f"Showing {len(display)} asset(s) in the selected view."
    )

    st.dataframe(
        display,
        width="stretch",
        hide_index=True,
    )


# ---------------------------------------------------------------------------
# Risk → maintenance decision
# ---------------------------------------------------------------------------

def _render_planning_decision(
    row: pd.Series,
) -> None:
    """Render the connection between risk prediction and planning."""

    st.markdown(
        "#### Maintenance planning decision"
    )

    status = row.get(
        "asset_planning_status",
        "No maintenance task",
    )

    if status == "Scheduled":

        st.success(
            "This asset has maintenance work included "
            "in the optimized maintenance plan."
        )

        cards = st.columns(4)

        cards[0].metric(
            "Task",
            str(
                row.get(
                    "maintenance_task_id",
                    "N/A",
                )
            ),
        )

        cards[1].metric(
            "Block",
            str(
                row.get(
                    "assigned_block",
                    "N/A",
                )
            ),
        )

        cards[2].metric(
            "Duration",
            (
                f"{float(row['maintenance_duration_min']):.0f} min"
                if pd.notna(
                    row.get(
                        "maintenance_duration_min"
                    )
                )
                else "N/A"
            ),
        )

        cards[3].metric(
            "Status",
            "Scheduled",
        )

        timing_rows = []

        if pd.notna(
            row.get("assigned_date")
        ):
            timing_rows.append(
                (
                    "Planning date",
                    str(row["assigned_date"]),
                )
            )

        if pd.notna(
            row.get("assigned_start")
        ):
            timing_rows.append(
                (
                    "Start",
                    str(row["assigned_start"]),
                )
            )

        if pd.notna(
            row.get("assigned_end")
        ):
            timing_rows.append(
                (
                    "End",
                    str(row["assigned_end"]),
                )
            )

        if timing_rows:
            timing = pd.DataFrame(
                timing_rows,
                columns=[
                    "Planning detail",
                    "Value",
                ],
            )

            st.dataframe(
                timing,
                width="stretch",
                hide_index=True,
            )

    elif status == "Not scheduled":

        st.warning(
            "This asset has maintenance work in the planning "
            "dataset, but none of its associated maintenance "
            "tasks are included in the optimized schedule."
        )

        task_id = row.get(
            "maintenance_task_id",
            None,
        )

        if pd.notna(task_id):
            st.markdown(
                f"**Representative maintenance task:** `{task_id}`"
            )

        st.caption(
            "Risk prediction does not override the optimizer's "
            "operational constraints. A task may remain unscheduled "
            "because no feasible block exists or because another "
            "feasible task is selected by the optimization model."
        )

    else:

        st.info(
            "No maintenance task is currently associated with "
            "this asset in the planning dataset."
        )


# ---------------------------------------------------------------------------
# Asset inspection
# ---------------------------------------------------------------------------

def _render_selected_asset(
    risk: pd.DataFrame,
) -> None:
    """Render details for a selected asset."""

    if risk.empty:
        return

    st.subheader("Inspect an asset")

    asset_ids = risk["asset_id"].tolist()

    selected_asset = st.selectbox(
        "Select asset",
        asset_ids,
        key="asset_risk_selector",
    )

    row = risk.loc[
        risk["asset_id"] == selected_asset
    ].iloc[0]

    probability = float(
        row["predicted_failure_probability"]
    )

    st.markdown(
        f"### {selected_asset}"
    )

    cards = st.columns(4)

    cards[0].metric(
        "Department",
        str(row["department"]),
    )

    cards[1].metric(
        "Section",
        str(row["section_id"]),
    )

    cards[2].metric(
        "Predicted risk",
        f"{probability * 100:.2f}%",
    )

    cards[3].metric(
        "Risk level",
        str(row["risk_level"]),
    )

    st.divider()

    _render_planning_decision(row)

    st.divider()

    st.markdown("#### Asset profile")

    profile_rows = [
        (
            "Asset type",
            row["asset_type"],
        ),
        (
            "Criticality",
            row["criticality"],
        ),
        (
            "Risk snapshot",
            row["snapshot_date"],
        ),
    ]

    if "maintenance_task_count" in row.index:
        profile_rows.append(
            (
                "Maintenance tasks",
                row["maintenance_task_count"],
            )
        )

    if "scheduled_task_count" in row.index:
        profile_rows.append(
            (
                "Scheduled tasks",
                row["scheduled_task_count"],
            )
        )

    profile = pd.DataFrame(
        profile_rows,
        columns=[
            "Attribute",
            "Value",
        ],
    )

    st.dataframe(
        profile,
        width="stretch",
        hide_index=True,
    )


# ---------------------------------------------------------------------------
# Main dashboard
# ---------------------------------------------------------------------------

def render_asset_risk_dashboard() -> None:
    """Render the complete asset-risk dashboard."""

    st.title("Asset Risk Command Center")

    st.caption(
        "Predicted asset failure risk connected to maintenance "
        "planning decisions."
    )

    try:
        risk = load_asset_risk_scores()
        decisions = load_maintenance_decisions()

        risk = _merge_risk_with_planning(
            risk,
            decisions,
        )

    except (
        FileNotFoundError,
        ValueError,
    ) as error:

        st.error(
            "Asset-risk or maintenance-planning results "
            "could not be loaded."
        )

        st.code(str(error))

        return

    if risk.empty:
        st.warning(
            "The asset-risk dataset contains no assets."
        )

        return

    snapshot = risk[
        "snapshot_date"
    ].iloc[0]

    st.info(
        f"Risk snapshot: **{snapshot}**. "
        "Risk probabilities are model outputs from the "
        "synthetic-data experiment and should not be interpreted "
        "as real railway failure probabilities."
    )

    # -----------------------------------------------------------------------
    # Risk KPIs
    # -----------------------------------------------------------------------

    _render_kpis(risk)

    st.divider()

    # -----------------------------------------------------------------------
    # Risk charts
    # -----------------------------------------------------------------------

    chart_left, chart_right = st.columns(2)

    with chart_left:
        _render_distribution(risk)

    with chart_right:
        _render_department_distribution(risk)

    st.divider()

    # -----------------------------------------------------------------------
    # Highest-risk assets
    # -----------------------------------------------------------------------

    _render_top_risk_assets(risk)

    st.divider()

    # -----------------------------------------------------------------------
    # Filters
    # -----------------------------------------------------------------------

    filtered = _render_filters(risk)

    _render_asset_table(filtered)

    st.divider()

    # -----------------------------------------------------------------------
    # Selected asset
    # -----------------------------------------------------------------------

    _render_selected_asset(filtered)