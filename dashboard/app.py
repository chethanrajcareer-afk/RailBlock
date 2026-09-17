"""RailBlock Streamlit decision-support dashboard."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st


# ---------------------------------------------------------------------------
# Project path
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Dashboard components
# ---------------------------------------------------------------------------

from dashboard.components.asset_risk import render_asset_risk_dashboard
from dashboard.components.blocks import (
    render_block_explorer,
    with_block_duration,
)
from dashboard.components.corridor import render_corridor
from dashboard.components.maintenance import render_maintenance_explorer
from dashboard.components.planning import render_planning_dashboard
from dashboard.components.section_detail import render_section_detail
from dashboard.components.timeline import render_timeline
from dashboard.components.trains import render_train_explorer
from dashboard.utils.data_loader import (
    DashboardDataError,
    default_data_dir,
    load_dashboard_data,
)


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="RailBlock — Maintenance Planning",
    page_icon="🚆",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

def _apply_control_room_style() -> None:
    """Apply restrained styling suitable for a railway planning interface."""

    st.markdown(
        """
        <style>
        .stMainBlockContainer {
            padding-top: 1.5rem;
            padding-bottom: 3rem;
        }

        [data-testid="stMetric"] {
            background: #162433;
            border-left: 4px solid #1F77B4;
            border-radius: 6px;
            padding: 0.75rem;
        }

        [data-testid="stMetricLabel"] {
            color: #B7C6D6;
        }

        [data-testid="stMetricValue"] {
            color: #F4F7FA;
        }

        section[data-testid="stSidebar"] {
            border-right: 1px solid rgba(128, 128, 128, 0.2);
        }

        .railblock-header {
            padding: 0.25rem 0 1rem 0;
        }

        .railblock-subtitle {
            color: #8FA3B8;
            font-size: 0.95rem;
        }

        .railblock-disclaimer {
            color: #8FA3B8;
            font-size: 0.8rem;
            line-height: 1.4;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading validated synthetic data...")
def _load_cached_data(data_directory: str):
    """Cache validated dashboard datasets between Streamlit reruns."""

    return load_dashboard_data(data_directory)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _render_sidebar() -> str:
    """Render dashboard navigation and return the selected page."""

    with st.sidebar:
        st.markdown(
            """
            <div class="railblock-header">
                <h1>🚆 RailBlock</h1>
                <div class="railblock-subtitle">
                    Railway Maintenance Planning
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        page = st.radio(
            "Navigation",
            [
                "🏠 Overview",
                "🗓️ Optimized Plan",
                "⚠️ Asset Risk",
                "🚆 Timeline",
                "🔍 Decision Explorer",
                "📋 Unscheduled Work",
                "🧪 What-If Analysis",
                "🗂️ Data Explorer",
            ],
            label_visibility="collapsed",
        )

        st.divider()

        st.markdown(
            """
            <div class="railblock-disclaimer">
                <strong>Prototype</strong><br>
                Uses fictional synthetic railway data.
                This dashboard does not represent operational
                Indian Railways data or rules.
            </div>
            """,
            unsafe_allow_html=True,
        )

    return page


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

def _render_header(title: str, subtitle: str) -> None:
    """Render a consistent page header."""

    st.title(title)
    st.caption(subtitle)


# ---------------------------------------------------------------------------
# Overview KPIs
# ---------------------------------------------------------------------------

def _render_overview_kpis(data: dict) -> None:
    """Render high-level synthetic planning metrics."""

    stations = data["stations"]
    sections = data["sections"]
    tasks = data["maintenance_tasks"]
    movements = data["train_movements"]

    blocks = with_block_duration(data["block_availability"])

    available_blocks = blocks.loc[
        blocks["availability_status"] == "available"
    ]

    st.subheader("Planning overview")

    primary_cards = st.columns(5)

    primary_cards[0].metric(
        "Stations",
        len(stations),
    )

    primary_cards[1].metric(
        "Sections",
        len(sections),
    )

    primary_cards[2].metric(
        "Maintenance tasks",
        len(tasks),
    )

    primary_cards[3].metric(
        "Train movements",
        len(movements),
    )

    primary_cards[4].metric(
        "Available block windows",
        len(available_blocks),
    )

    department_cards = st.columns(3)

    for position, department in enumerate(
        ("Engineering", "S&T", "Traction")
    ):
        department_cards[position].metric(
            f"{department} tasks",
            int((tasks["department"] == department).sum()),
        )

    operational_cards = st.columns(3)

    operational_cards[0].metric(
        "Critical tasks",
        int((tasks["criticality"] == "critical").sum()),
    )

    operational_cards[1].metric(
        "Average maintenance duration",
        f"{tasks['estimated_duration_min'].mean():.0f} min",
    )

    operational_cards[2].metric(
        "Available block minutes",
        f"{available_blocks['duration_min'].sum():,.0f} min",
    )


# ---------------------------------------------------------------------------
# Page: Overview
# ---------------------------------------------------------------------------

def _render_overview(data: dict) -> None:
    """Render the dashboard overview."""

    _render_header(
        "RailBlock",
        "AI-assisted railway maintenance block planning and asset risk "
        "decision support.",
    )

    st.info(
        "This prototype combines constraint optimization, asset-risk "
        "prediction, schedule validation, and explainable decision analysis "
        "using fictional synthetic railway data."
    )

    _render_overview_kpis(data)

    st.divider()

    st.subheader("Railway corridor")

    render_corridor(data["stations"])

    st.divider()

    st.subheader("Section detail")

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


# ---------------------------------------------------------------------------
# Page: Optimized Plan
# ---------------------------------------------------------------------------

def _render_optimized_plan(data: dict) -> None:
    """Render the existing optimization dashboard."""

    _render_header(
        "Optimized Maintenance Plan",
        "Explore the maintenance schedule generated by the RailBlock "
        "optimization engine.",
    )

    render_planning_dashboard(
        data["maintenance_tasks"],
        data["train_movements"],
    )


# ---------------------------------------------------------------------------
# Page: Asset Risk
# ---------------------------------------------------------------------------

def _render_asset_risk() -> None:
    """Render the Phase 2.5 asset-risk decision-support dashboard."""

    _render_header(
        "Asset Risk",
        "Predicted asset failure risk and maintenance-priority analysis.",
    )

    render_asset_risk_dashboard()


# ---------------------------------------------------------------------------
# Page: Timeline
# ---------------------------------------------------------------------------
def _render_timeline_page(data: dict) -> None:
    """Render the corridor timeline as a dedicated page."""

    _render_header(
        "Train & Block Timeline",
        "Visualize synthetic train movements, available block windows, "
        "and maintenance request windows.",
    )

    selected_section = st.selectbox(
        "Select section",
        data["sections"]["section_id"].tolist(),
        key="timeline_page_section",
    )

    render_timeline(
        selected_section,
        data["maintenance_tasks"],
        data["train_movements"],
        data["block_availability"],
    )


# ---------------------------------------------------------------------------
# Page: Decision Explorer
# ---------------------------------------------------------------------------

def _render_decision_explorer(data: dict) -> None:
    """Expose the existing planning/explainability interface."""

    _render_header(
        "Decision Explorer",
        "Inspect why maintenance tasks were selected, scheduled, "
        "or left unscheduled.",
    )

    render_planning_dashboard(
        data["maintenance_tasks"],
        data["train_movements"],
    )


# ---------------------------------------------------------------------------
# Page: Unscheduled Work
# ---------------------------------------------------------------------------

def _render_unscheduled_work(data: dict) -> None:
    """Expose the planning dashboard for unscheduled-task analysis."""

    _render_header(
        "Unscheduled Maintenance",
        "Explore maintenance work that was not included in the "
        "optimized schedule.",
    )

    render_planning_dashboard(
        data["maintenance_tasks"],
        data["train_movements"],
    )


# ---------------------------------------------------------------------------
# Page: What-if Analysis
# ---------------------------------------------------------------------------

def _render_what_if() -> None:
    """Render the current What-if Analysis placeholder."""

    _render_header(
        "What-If Analysis",
        "Evaluate how planning decisions change when optimization "
        "assumptions change.",
    )

    st.info(
        "The risk-weight sensitivity engine is already implemented. "
        "The interactive What-If interface will connect to it in a "
        "later Phase 2.5 milestone."
    )


# ---------------------------------------------------------------------------
# Page: Data Explorer
# ---------------------------------------------------------------------------

def _render_data_explorer(data: dict) -> None:
    """Render the original exploratory data views."""

    _render_header(
        "Data Explorer",
        "Inspect the validated synthetic railway planning datasets.",
    )

    st.subheader("Maintenance requests")

    render_maintenance_explorer(
        data["maintenance_tasks"]
    )

    st.divider()

    st.subheader("Train movements")

    render_train_explorer(
        data["train_movements"]
    )

    st.divider()

    st.subheader("Block availability")

    render_block_explorer(
        data["block_availability"]
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the RailBlock Streamlit dashboard."""

    _apply_control_room_style()

    page = _render_sidebar()

    try:
        data = _load_cached_data(
            str(default_data_dir())
        )

    except DashboardDataError as error:
        st.error("The RailBlock datasets could not be loaded.")
        st.code(str(error))
        st.stop()

    if page == "🏠 Overview":

        _render_overview(data)

    elif page == "🗓️ Optimized Plan":

        _render_optimized_plan(data)

    elif page == "⚠️ Asset Risk":

        _render_asset_risk()

    elif page == "🚆 Timeline":

        _render_timeline_page(data)

    elif page == "🔍 Decision Explorer":

        _render_decision_explorer(data)

    elif page == "📋 Unscheduled Work":

        _render_unscheduled_work(data)

    elif page == "🧪 What-If Analysis":

        _render_what_if()

    elif page == "🗂️ Data Explorer":

        _render_data_explorer(data)


if __name__ == "__main__":
    main()