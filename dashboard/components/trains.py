"""Train movement filters and visualisations."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st


def _filter_movements(movements: pd.DataFrame) -> pd.DataFrame:
    """Render movement filters and return the selected subset."""

    st.markdown("#### Filters")
    columns = st.columns(5)
    service_dates = columns[0].multiselect(
        "Service date",
        sorted(movements["service_date"].dt.date.unique()),
        key="train_service_date",
    )
    section = columns[1].multiselect(
        "Section", sorted(movements["section_id"].unique()), key="train_section"
    )
    train_type = columns[2].multiselect(
        "Train type", sorted(movements["train_type"].unique()), key="train_type"
    )
    direction = columns[3].multiselect(
        "Direction", ["UP", "DOWN"], key="train_direction"
    )
    priority = columns[4].multiselect(
        "Priority", ["low", "medium", "high"], key="train_priority"
    )

    filtered = movements.copy()
    if service_dates:
        filtered = filtered[filtered["service_date"].dt.date.isin(service_dates)]
    for column, selected in (
        ("section_id", section),
        ("train_type", train_type),
        ("direction", direction),
        ("priority", priority),
    ):
        if selected:
            filtered = filtered[filtered[column].isin(selected)]
    return filtered


def render_train_explorer(movements: pd.DataFrame) -> None:
    """Render a filterable train-movement table and section-volume chart."""

    st.subheader("Train movement explorer")
    filtered = _filter_movements(movements)
    chart_data = (
        filtered.groupby("section_id", as_index=False)
        .size()
        .rename(columns={"size": "movement_count"})
        .sort_values("section_id")
    )
    chart = px.bar(
        chart_data,
        x="section_id",
        y="movement_count",
        color="movement_count",
        color_continuous_scale="Blues",
        title="Train movements by section",
        labels={"section_id": "Section", "movement_count": "Movements"},
    )
    chart.update_layout(coloraxis_showscale=False, margin={"l": 0, "r": 0, "t": 45, "b": 0})

    chart_column, table_column = st.columns((1, 2))
    with chart_column:
        st.plotly_chart(chart, use_container_width=True)
    with table_column:
        st.caption(f"{len(filtered):,} movement record(s) match the current filters.")
        display_columns = [
            "movement_id",
            "train_id",
            "train_type",
            "service_date",
            "section_id",
            "entry_time",
            "exit_time",
            "direction",
            "priority",
            "expected_delay_min",
        ]
        display = filtered.loc[:, display_columns].sort_values(
            ["service_date", "entry_time", "movement_id"]
        ).copy()
        display["service_date"] = display["service_date"].dt.strftime("%Y-%m-%d")
        display["entry_time"] = display["entry_time"].astype(str)
        display["exit_time"] = display["exit_time"].astype(str)
        st.dataframe(display, hide_index=True, use_container_width=True, height=360)
