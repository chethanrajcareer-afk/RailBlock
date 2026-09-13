"""Fictional railway corridor visualisation."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


RAIL_BLUE = "#1F77B4"
RAIL_AMBER = "#F2B134"
GRID_COLOR = "#D9E2EC"


def build_corridor_figure(stations: pd.DataFrame) -> go.Figure:
    """Return a horizontal, distance-scaled view of the fictional corridor."""

    ordered = stations.sort_values("sequence")
    labels = [f"{row.station_id}<br>{row.station_name}" for row in ordered.itertuples()]
    hover_text = [
        f"<b>{row.station_name}</b><br>{row.station_id}<br>Sequence: {row.sequence}<br>"
        f"Km from origin: {row.km_from_origin:.1f}"
        for row in ordered.itertuples()
    ]
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=ordered["km_from_origin"],
            y=[0] * len(ordered),
            mode="lines",
            line={"color": RAIL_BLUE, "width": 7},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=ordered["km_from_origin"],
            y=[0] * len(ordered),
            mode="markers+text",
            marker={"size": 15, "color": RAIL_AMBER, "line": {"color": RAIL_BLUE, "width": 2}},
            text=labels,
            textposition="top center",
            textfont={"size": 10, "color": "#16324F"},
            hovertext=hover_text,
            hoverinfo="text",
            showlegend=False,
        )
    )
    figure.update_layout(
        height=310,
        margin={"l": 15, "r": 15, "t": 35, "b": 40},
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis={
            "title": "Fictional corridor distance from origin (km)",
            "showgrid": True,
            "gridcolor": GRID_COLOR,
            "zeroline": False,
        },
        yaxis={"visible": False, "range": [-0.45, 0.45]},
    )
    return figure


def render_corridor(stations: pd.DataFrame) -> None:
    """Render the corridor overview and the explicit station sequence."""

    st.subheader("Fictional railway corridor")
    st.caption("A linear representation of the synthetic corridor. Station positions are scaled by fictional kilometre distance.")
    st.plotly_chart(build_corridor_figure(stations), use_container_width=True)
    station_path = " → ".join(stations.sort_values("sequence")["station_id"].tolist())
    st.markdown(f"**Station sequence:** {station_path}")
