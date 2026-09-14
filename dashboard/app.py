"""Streamlit entry point for RailBlock's Phase 0 Data Explorer."""

from __future__ import annotations


import sys
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.components.blocks import render_block_explorer, with_block_duration
from dashboard.components.corridor import render_corridor
from dashboard.components.maintenance import render_maintenance_explorer
from dashboard.components.planning import render_planning_dashboard
from dashboard.components.section_detail import render_section_detail
from dashboard.components.timeline import render_timeline
from dashboard.components.trains import render_train_explorer
from dashboard.utils.data_loader import DashboardDataError, default_data_dir, load_dashboard_data


st.set_page_config(
    page_title="RailBlock — Intelligent Maintenance Block Planning",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_data(show_spinner="Loading validated synthetic data...")
def _load_cached_data(data_directory: str):
    """Cache validated CSVs between Streamlit reruns."""

    return load_dashboard_data(data_directory)


def _render_kpis(data: dict) -> None:
    """Render the data-overview cards without introducing planning logic."""

    stations = data["stations"]
    sections = data["sections"]
    tasks = data["maintenance_tasks"]
    movements = data["train_movements"]
    blocks = with_block_duration(data["block_availability"])
    available_blocks = blocks.loc[blocks["availability_status"] == "available"]

    st.subheader("Planning overview")
    primary_cards = st.columns(5)
    primary_cards[0].metric("Stations", len(stations))
    primary_cards[1].metric("Sections", len(sections))
    primary_cards[2].metric("Maintenance tasks", len(tasks))
    primary_cards[3].metric("Train movements", len(movements))
    primary_cards[4].metric("Available block windows", len(available_blocks))

    department_cards = st.columns(3)
    for position, department in enumerate(("Engineering", "S&T", "Traction")):
        department_cards[position].metric(
            f"{department} tasks", int((tasks["department"] == department).sum())
        )
    operational_cards = st.columns(3)
    operational_cards[0].metric("Critical tasks", int((tasks["criticality"] == "critical").sum()))
    operational_cards[1].metric(
        "Average maintenance duration", f"{tasks['estimated_duration_min'].mean():.0f} min"
    )
    operational_cards[2].metric(
        "Available block minutes", f"{available_blocks['duration_min'].sum():,.0f} min"
    )


def _apply_control_room_style() -> None:
    """Add restrained visual cues suitable for a planning-control interface."""

    st.markdown(
        """
        <style>
        [data-testid="stMetric"] {
            background: #162433;
            border-left: 4px solid #1F77B4;
            border-radius: 4px;
            padding: 0.7rem;
        }
        [data-testid="stMetricLabel"] {
            color: #B7C6D6;
        }
        [data-testid="stMetricValue"] {
            color: #F4F7FA;
        }
        .stMainBlockContainer {
            padding-top: 2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    """Render the Phase 0 exploration dashboard."""

    _apply_control_room_style()
    st.title("RailBlock — Railway Maintenance Planning Explorer")
    st.caption(
    "Optimization-driven maintenance planning prototype using fictional "
    "synthetic railway data. This does not represent operational Indian "
    "Railways data or rules."
    )

    try:
        data = _load_cached_data(str(default_data_dir()))
    except DashboardDataError as error:
        st.error("The Phase 0 datasets could not be loaded.")
        st.code(str(error))
        st.stop()

    _render_kpis(data)

    st.divider()
    render_planning_dashboard(
    data["maintenance_tasks"],
    data["train_movements"],
)

    st.divider()
    render_corridor(data["stations"])
    st.divider()
    selected_section = render_section_detail(
        data["stations"],
        data["sections"],
        data["maintenance_tasks"],
        data["train_movements"],
        data["block_availability"],
    )
    st.divider()
    render_timeline(
        selected_section,
        data["maintenance_tasks"],
        data["train_movements"],
        data["block_availability"],
    )
    st.divider()
    render_maintenance_explorer(data["maintenance_tasks"])
    st.divider()
    render_train_explorer(data["train_movements"])
    st.divider()
    render_block_explorer(data["block_availability"])


if __name__ == "__main__":
    main()
