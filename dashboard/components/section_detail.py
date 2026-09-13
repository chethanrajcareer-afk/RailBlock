"""Compact operational context for one fictional corridor section."""

from __future__ import annotations

import pandas as pd
import streamlit as st


def render_section_detail(
    stations: pd.DataFrame,
    sections: pd.DataFrame,
    tasks: pd.DataFrame,
    movements: pd.DataFrame,
    blocks: pd.DataFrame,
) -> str:
    """Render a selected section's synthetic operational picture and return its ID."""

    st.subheader("Section detail")
    selected_section = st.selectbox(
        "Select a section",
        sections["section_id"].tolist(),
        key="section_detail_selection",
    )
    section = sections.loc[sections["section_id"] == selected_section].iloc[0]
    station_names = stations.set_index("station_id")["station_name"].to_dict()
    matching_tasks = tasks.loc[tasks["section_id"] == selected_section]
    matching_movements = movements.loc[movements["section_id"] == selected_section]
    available_blocks = blocks.loc[
        (blocks["section_id"] == selected_section)
        & (blocks["availability_status"] == "available")
    ]

    st.markdown(
        f"**{section['from_station']} — {station_names[section['from_station']]}**  "
        f"to **{section['to_station']} — {station_names[section['to_station']]}**"
    )
    first_row = st.columns(4)
    first_row[0].metric("Distance", f"{section['distance_km']:.1f} km")
    first_row[1].metric("Normal running time", f"{section['normal_running_time_min']} min")
    first_row[2].metric("Minimum block duration", f"{section['minimum_block_duration_min']} min")
    first_row[3].metric("Maintenance tasks", len(matching_tasks))
    second_row = st.columns(3)
    second_row[0].metric("Train movements", len(matching_movements))
    second_row[1].metric("Available block windows", len(available_blocks))
    second_row[2].metric("Critical tasks", int((matching_tasks["criticality"] == "critical").sum()))
    return selected_section
