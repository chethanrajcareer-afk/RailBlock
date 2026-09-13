"""Maintenance block availability filters and visualisations."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st


def with_block_duration(blocks: pd.DataFrame) -> pd.DataFrame:
    """Add a duration column without changing the validated source frame."""

    result = blocks.copy()
    dates = pd.to_datetime(result["date"]).dt.strftime("%Y-%m-%d")
    starts = pd.to_datetime(dates + " " + result["start_time"].astype(str))
    ends = pd.to_datetime(dates + " " + result["end_time"].astype(str))
    result["duration_min"] = ((ends - starts).dt.total_seconds() / 60).round().astype(int)
    return result


def _filter_blocks(blocks: pd.DataFrame) -> pd.DataFrame:
    """Render availability filters and return the selected subset."""

    st.markdown("#### Filters")
    columns = st.columns(3)
    dates = columns[0].multiselect(
        "Date", sorted(blocks["date"].dt.date.unique()), key="block_date"
    )
    sections = columns[1].multiselect(
        "Section", sorted(blocks["section_id"].unique()), key="block_section"
    )
    status = columns[2].multiselect(
        "Availability status",
        ["available", "unavailable"],
        key="block_status",
    )
    filtered = blocks.copy()
    if dates:
        filtered = filtered[filtered["date"].dt.date.isin(dates)]
    if sections:
        filtered = filtered[filtered["section_id"].isin(sections)]
    if status:
        filtered = filtered[filtered["availability_status"].isin(status)]
    return filtered


def render_block_explorer(blocks: pd.DataFrame) -> None:
    """Render available capacity by section and the detailed block-window table."""

    st.subheader("Block availability explorer")
    blocks_with_duration = with_block_duration(blocks)
    filtered = _filter_blocks(blocks_with_duration)
    available_minutes = (
        filtered.loc[filtered["availability_status"] == "available"]
        .groupby("section_id", as_index=False)["duration_min"]
        .sum()
        .sort_values("section_id")
    )
    chart = px.bar(
        available_minutes,
        x="section_id",
        y="duration_min",
        color_discrete_sequence=["#14B8A6"],
        title="Available block-window minutes by section",
        labels={"section_id": "Section", "duration_min": "Available minutes"},
    )
    chart.update_layout(showlegend=False, margin={"l": 0, "r": 0, "t": 45, "b": 0})

    chart_column, table_column = st.columns((1, 2))
    with chart_column:
        st.plotly_chart(chart, use_container_width=True)
    with table_column:
        st.caption(f"{len(filtered):,} block window(s) match the current filters.")
        display_columns = [
            "block_window_id",
            "section_id",
            "date",
            "start_time",
            "end_time",
            "duration_min",
            "availability_status",
            "reason",
        ]
        display = filtered.loc[:, display_columns].sort_values(
            ["date", "section_id", "start_time"]
        ).copy()
        display["date"] = display["date"].dt.strftime("%Y-%m-%d")
        display["start_time"] = display["start_time"].astype(str)
        display["end_time"] = display["end_time"].astype(str)
        st.dataframe(display, hide_index=True, use_container_width=True, height=360)
