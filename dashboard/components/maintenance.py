"""Maintenance task filters and visualisations."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st


DEPARTMENT_COLORS = {
    "Engineering": "#1F77B4",
    "S&T": "#7A5195",
    "Traction": "#F2B134",
}


def _filter_tasks(tasks: pd.DataFrame) -> pd.DataFrame:
    """Render task filters and return the selected subset."""

    st.markdown("#### Filters")
    columns = st.columns(5)
    department = columns[0].multiselect(
        "Department", sorted(tasks["department"].unique()), key="maintenance_department"
    )
    section = columns[1].multiselect(
        "Section", sorted(tasks["section_id"].unique()), key="maintenance_section"
    )
    criticality = columns[2].multiselect(
        "Criticality", ["low", "medium", "high", "critical"], key="maintenance_criticality"
    )
    urgency = columns[3].multiselect(
        "Urgency", ["low", "medium", "high"], key="maintenance_urgency"
    )
    status = columns[4].multiselect(
        "Status", sorted(tasks["status"].unique()), key="maintenance_status"
    )

    filtered = tasks.copy()
    for column, selected in (
        ("department", department),
        ("section_id", section),
        ("criticality", criticality),
        ("urgency", urgency),
        ("status", status),
    ):
        if selected:
            filtered = filtered[filtered[column].isin(selected)]
    return filtered


def render_maintenance_explorer(tasks: pd.DataFrame) -> None:
    """Render the filterable maintenance-task view and department chart."""

    st.subheader("Maintenance task explorer")
    filtered = _filter_tasks(tasks)
    chart_data = (
        filtered.groupby("department", as_index=False)
        .size()
        .rename(columns={"size": "task_count"})
    )
    chart = px.bar(
        chart_data,
        x="department",
        y="task_count",
        color="department",
        color_discrete_map=DEPARTMENT_COLORS,
        title="Maintenance tasks by department",
        labels={"department": "Department", "task_count": "Tasks"},
    )
    chart.update_layout(showlegend=False, margin={"l": 0, "r": 0, "t": 45, "b": 0})

    chart_column, table_column = st.columns((1, 2))
    with chart_column:
        st.plotly_chart(chart, use_container_width=True)
    with table_column:
        st.caption(f"{len(filtered):,} task(s) match the current filters.")
        display_columns = [
            "task_id",
            "department",
            "asset_type",
            "section_id",
            "task_type",
            "criticality",
            "urgency",
            "estimated_duration_min",
            "earliest_start",
            "deadline",
            "status",
        ]
        display = filtered.loc[:, display_columns].sort_values(["deadline", "task_id"]).copy()
        display["earliest_start"] = display["earliest_start"].dt.strftime("%Y-%m-%d %H:%M")
        display["deadline"] = display["deadline"].dt.strftime("%Y-%m-%d %H:%M")
        st.dataframe(display, hide_index=True, use_container_width=True, height=360)
