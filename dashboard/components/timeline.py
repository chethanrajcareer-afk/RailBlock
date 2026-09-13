"""Non-optimising timeline that exposes competing synthetic corridor use."""

from __future__ import annotations

from datetime import date

import pandas as pd
import plotly.express as px
import streamlit as st


TIMELINE_COLORS = {
    "Train movement": "#1F77B4",
    "Available block window": "#14B8A6",
    "Maintenance request window": "#F2B134",
}


def _timestamp_for_day(day: pd.Timestamp, value: object) -> pd.Timestamp:
    """Combine a selected day and a time-only CSV field."""

    return pd.to_datetime(f"{day:%Y-%m-%d} {value}")


def build_timeline_events(
    selected_date: date,
    selected_section: str,
    tasks: pd.DataFrame,
    movements: pd.DataFrame,
    blocks: pd.DataFrame,
) -> pd.DataFrame:
    """Build display events only; no task is assigned to any block window."""

    day_start = pd.Timestamp(selected_date)
    day_end = day_start + pd.Timedelta(days=1)
    event_frames: list[pd.DataFrame] = []

    train_rows = movements.loc[
        (movements["service_date"].dt.date == selected_date)
        & (movements["section_id"] == selected_section)
    ].copy()
    if not train_rows.empty:
        train_rows["start"] = train_rows["entry_time"].map(
            lambda value: _timestamp_for_day(day_start, value)
        )
        train_rows["end"] = train_rows["exit_time"].map(
            lambda value: _timestamp_for_day(day_start, value)
        )
        event_frames.append(
            pd.DataFrame(
                {
                    "start": train_rows["start"],
                    "end": train_rows["end"],
                    "lane": "Train movements",
                    "category": "Train movement",
                    "label": train_rows["train_id"],
                    "detail": (
                        train_rows["train_type"]
                        + " / "
                        + train_rows["direction"]
                        + " / priority "
                        + train_rows["priority"]
                    ),
                }
            )
        )

    block_rows = blocks.loc[
        (blocks["date"].dt.date == selected_date)
        & (blocks["section_id"] == selected_section)
        & (blocks["availability_status"] == "available")
    ].copy()
    if not block_rows.empty:
        block_rows["start"] = block_rows["start_time"].map(
            lambda value: _timestamp_for_day(day_start, value)
        )
        block_rows["end"] = block_rows["end_time"].map(
            lambda value: _timestamp_for_day(day_start, value)
        )
        event_frames.append(
            pd.DataFrame(
                {
                    "start": block_rows["start"],
                    "end": block_rows["end"],
                    "lane": "Available block windows",
                    "category": "Available block window",
                    "label": block_rows["block_window_id"],
                    "detail": block_rows["reason"],
                }
            )
        )

    task_rows = tasks.loc[
        (tasks["section_id"] == selected_section)
        & (tasks["earliest_start"] < day_end)
        & (tasks["deadline"] > day_start)
    ].copy()
    if not task_rows.empty:
        task_rows["start"] = task_rows["earliest_start"].clip(lower=day_start)
        task_rows["end"] = task_rows["deadline"].clip(upper=day_end)
        event_frames.append(
            pd.DataFrame(
                {
                    "start": task_rows["start"],
                    "end": task_rows["end"],
                    "lane": "Maintenance requests",
                    "category": "Maintenance request window",
                    "label": task_rows["task_id"],
                    "detail": (
                        task_rows["task_type"]
                        + " / "
                        + task_rows["criticality"]
                        + " / "
                        + task_rows["estimated_duration_min"].astype(str)
                        + " min required"
                    ),
                }
            )
        )

    if not event_frames:
        return pd.DataFrame(columns=["start", "end", "lane", "category", "label", "detail"])
    return pd.concat(event_frames, ignore_index=True).sort_values(["lane", "start"])


def render_timeline(
    selected_section: str,
    tasks: pd.DataFrame,
    movements: pd.DataFrame,
    blocks: pd.DataFrame,
) -> None:
    """Render a timeline that makes the maintenance-versus-traffic conflict visible."""

    st.subheader("Section access timeline")
    available_dates = sorted(
        set(movements["service_date"].dt.date).union(blocks["date"].dt.date)
    )
    selected_date = st.selectbox("Timeline date", available_dates, key="timeline_date")
    events = build_timeline_events(selected_date, selected_section, tasks, movements, blocks)
    st.caption(
        "Maintenance bars show request windows from earliest start through deadline. "
        "They are not scheduled tasks, assigned blocks, or optimizer output."
    )
    if events.empty:
        st.info("No synthetic train movements, available windows, or maintenance requests match this date and section.")
        return

    figure = px.timeline(
        events,
        x_start="start",
        x_end="end",
        y="lane",
        color="category",
        hover_name="label",
        hover_data={"detail": True, "start": "|%H:%M", "end": "|%H:%M", "lane": False},
        color_discrete_map=TIMELINE_COLORS,
    )
    day_start = pd.Timestamp(selected_date)
    figure.update_yaxes(autorange="reversed", title=None)
    figure.update_xaxes(
        title="Time",
        range=[day_start, day_start + pd.Timedelta(days=1)],
        tickformat="%H:%M",
    )
    figure.update_layout(
        height=430,
        legend_title_text="",
        margin={"l": 0, "r": 0, "t": 25, "b": 0},
        bargap=0.35,
    )
    st.plotly_chart(figure, use_container_width=True)
