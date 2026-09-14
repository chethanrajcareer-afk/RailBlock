"""RailBlock optimized maintenance planning dashboard components."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs"

SCHEDULE_FILE = OUTPUT_DIR / "optimized_schedule.csv"
DIAGNOSTICS_FILE = OUTPUT_DIR / "schedule_diagnostics.csv"
UTILIZATION_FILE = OUTPUT_DIR / "block_utilization.csv"
INFEASIBILITY_FILE = OUTPUT_DIR / "infeasibility_analysis.csv"


# =========================================================
# Data loading
# =========================================================


def load_planning_outputs() -> dict[str, pd.DataFrame]:
    """Load optimizer output files without modifying them."""

    if not SCHEDULE_FILE.is_file():
        raise FileNotFoundError(
            f"Optimized schedule not found: {SCHEDULE_FILE}"
        )

    schedule = pd.read_csv(SCHEDULE_FILE)

    outputs = {
    "schedule": schedule,
}

    if DIAGNOSTICS_FILE.is_file():
        outputs["diagnostics"] = pd.read_csv(DIAGNOSTICS_FILE)

    if UTILIZATION_FILE.is_file():
        outputs["utilization"] = pd.read_csv(UTILIZATION_FILE)

    if INFEASIBILITY_FILE.is_file():
        outputs["infeasibility"] = pd.read_csv(INFEASIBILITY_FILE)

    return outputs


def _prepare_schedule(schedule: pd.DataFrame) -> pd.DataFrame:
    """Prepare optimizer output for dashboard display."""

    result = schedule.copy()

    for column in ("block_date",):
        if column in result.columns:
            result[column] = pd.to_datetime(result[column])

    for column in ("scheduled_start", "scheduled_end"):
        if column in result.columns:
            result[column] = pd.to_datetime(result[column])

    return result


# =========================================================
# KPI calculations
# =========================================================


def calculate_planning_kpis(
    tasks: pd.DataFrame,
    schedule: pd.DataFrame,
) -> dict[str, float]:
    """Calculate high-level planning metrics."""

    total_tasks = len(tasks)
    scheduled_tasks = len(schedule)

    used_blocks = (
        schedule["block_window_id"].nunique()
        if not schedule.empty
        else 0
    )

    total_scheduled_minutes = (
        schedule["estimated_duration_min"].sum()
        if not schedule.empty
        else 0
    )

    if "utilization" in schedule.columns:
        average_utilization = schedule["utilization"].mean()
    else:
        average_utilization = None

    return {
        "total_tasks": total_tasks,
        "scheduled_tasks": scheduled_tasks,
        "unscheduled_tasks": total_tasks - scheduled_tasks,
        "used_blocks": used_blocks,
        "scheduled_minutes": total_scheduled_minutes,
        "average_utilization": average_utilization,
    }


def render_planning_kpis(
    tasks: pd.DataFrame,
    schedule: pd.DataFrame,
) -> None:
    """Render the primary RailBlock planning KPIs."""

    total_tasks = len(tasks)
    scheduled_tasks = len(schedule)
    unscheduled_tasks = total_tasks - scheduled_tasks

    used_blocks = (
        schedule["block_window_id"].nunique()
        if not schedule.empty
        else 0
    )

    scheduled_minutes = (
        schedule["estimated_duration_min"].sum()
        if not schedule.empty
        else 0
    )

    coverage = (
        scheduled_tasks / total_tasks * 100
        if total_tasks
        else 0
    )

    st.subheader("RailBlock optimized plan")

    cards = st.columns(4)

    cards[0].metric(
        "Maintenance tasks",
        f"{total_tasks:,}",
    )

    cards[1].metric(
        "Tasks planned",
        f"{scheduled_tasks:,}",
        delta=f"{coverage:.1f}% coverage",
    )

    cards[2].metric(
        "Tasks not scheduled",
        f"{unscheduled_tasks:,}",
    )

    cards[3].metric(
        "Blocks used",
        f"{used_blocks:,}",
    )

    st.caption(
        f"{scheduled_minutes:,.0f} minutes of maintenance work "
        "scheduled across the optimized plan."
    )


# =========================================================
# Schedule table
# =========================================================


def render_schedule_table(schedule: pd.DataFrame) -> None:
    """Render the optimized weekly maintenance schedule."""

    st.subheader("Weekly maintenance plan")

    if schedule.empty:
        st.info("No maintenance tasks were scheduled.")
        return

    display = schedule.copy()

    display["block_date"] = display["block_date"].dt.strftime(
        "%Y-%m-%d"
    )

    display["scheduled_start"] = display["scheduled_start"].dt.strftime(
        "%Y-%m-%d %H:%M"
    )

    display["scheduled_end"] = display["scheduled_end"].dt.strftime(
        "%Y-%m-%d %H:%M"
    )

    display_columns = [
        "task_id",
        "department",
        "section_id",
        "task_type",
        "criticality",
        "urgency",
        "estimated_duration_min",
        "block_window_id",
        "block_date",
        "scheduled_start",
        "scheduled_end",
    ]

    available_columns = [
        column
        for column in display_columns
        if column in display.columns
    ]

    display = display[available_columns].sort_values(
        ["block_date", "scheduled_start", "task_id"]
    )

    st.caption(
        f"{len(display):,} maintenance task(s) scheduled."
    )

    st.dataframe(
        display,
        hide_index=True,
        width="stretch",
        height=430,
    )


# =========================================================
# Filters
# =========================================================


def filter_schedule(schedule: pd.DataFrame) -> pd.DataFrame:
    """Apply planning filters to the optimized schedule."""

    st.markdown("#### Plan filters")

    columns = st.columns(4)

    departments = columns[0].multiselect(
        "Department",
        sorted(schedule["department"].dropna().unique()),
        key="planning_department",
    )

    sections = columns[1].multiselect(
        "Section",
        sorted(schedule["section_id"].dropna().unique()),
        key="planning_section",
    )

    criticality = columns[2].multiselect(
        "Criticality",
        ["low", "medium", "high", "critical"],
        key="planning_criticality",
    )

    dates = columns[3].multiselect(
        "Block date",
        sorted(schedule["block_date"].dt.date.unique()),
        key="planning_date",
    )

    filtered = schedule.copy()

    if departments:
        filtered = filtered[
            filtered["department"].isin(departments)
        ]

    if sections:
        filtered = filtered[
            filtered["section_id"].isin(sections)
        ]

    if criticality:
        filtered = filtered[
            filtered["criticality"].isin(criticality)
        ]

    if dates:
        filtered = filtered[
            filtered["block_date"].dt.date.isin(dates)
        ]

    return filtered


# =========================================================
# Weekly block utilization
# =========================================================


def render_weekly_block_utilization(
    schedule: pd.DataFrame,
    utilization: pd.DataFrame | None = None,
) -> None:
    """Render daily maintenance usage across the optimized week."""

    st.subheader("Weekly block utilization")

    if utilization is None or utilization.empty:
        st.info("No utilization diagnostics are available.")
        return

    data = utilization.copy()
    if "block_date" in data.columns:
        data["block_date"] = pd.to_datetime(data["block_date"])
    elif "date" in data.columns:
        data["block_date"] = pd.to_datetime(data["date"])
    else:
        st.info("Block-date information is unavailable.")
        return

    used = data.loc[data["tasks_scheduled"] > 0].copy()
    if used.empty:
        st.info("No maintenance blocks were used.")
        return

    daily = (
        used.groupby(used["block_date"].dt.date, as_index=False)
        .agg(
            used_blocks=("block_window_id", "nunique"),
            maintenance_minutes=("scheduled_minutes", "sum"),
            block_minutes=("block_minutes", "sum"),
        )
    )
    daily["unused_minutes"] = daily["block_minutes"] - daily["maintenance_minutes"]
    daily["utilization_pct"] = (
        daily["maintenance_minutes"] / daily["block_minutes"] * 100
    ).where(daily["block_minutes"] > 0, 0)

    if schedule is not None and not schedule.empty and "block_date" in schedule.columns:
        scheduled_dates = set(pd.to_datetime(schedule["block_date"]).dt.date)
        daily = daily.loc[daily["block_date"].isin(scheduled_dates)].copy()

    if daily.empty:
        st.info("No weekly utilization data matches the current plan.")
        return

    daily["date_label"] = pd.to_datetime(daily["block_date"]).dt.strftime("%a %d %b")

    figure = px.bar(
        daily,
        x="date_label",
        y="utilization_pct",
        text="utilization_pct",
        hover_data={
            "used_blocks": True,
            "maintenance_minutes": ":,.0f",
            "block_minutes": ":,.0f",
            "unused_minutes": ":,.0f",
            "utilization_pct": ":.1f",
            "date_label": False,
        },
        labels={
            "date_label": "Date",
            "utilization_pct": "Maintenance utilization (%)",
            "used_blocks": "Used blocks",
            "maintenance_minutes": "Maintenance minutes",
            "block_minutes": "Available block minutes",
            "unused_minutes": "Unused minutes",
        },
        title="Daily utilization of maintenance blocks",
    )
    figure.update_traces(texttemplate="%{text:.1f}%", textposition="outside", cliponaxis=False)
    figure.update_yaxes(range=[0, 100], ticksuffix="%")
    figure.update_layout(showlegend=False, margin={"l": 0, "r": 0, "t": 55, "b": 0})
    st.plotly_chart(figure, width="stretch")

    summary = daily[[
        "block_date", "used_blocks", "maintenance_minutes",
        "block_minutes", "unused_minutes", "utilization_pct",
    ]].copy()
    summary["block_date"] = pd.to_datetime(summary["block_date"]).dt.strftime("%Y-%m-%d")
    for column in ("maintenance_minutes", "block_minutes", "unused_minutes"):
        summary[column] = summary[column].round().astype(int)
    summary["utilization_pct"] = summary["utilization_pct"].round(1)
    summary = summary.rename(columns={
        "block_date": "Date",
        "used_blocks": "Used blocks",
        "maintenance_minutes": "Maintenance (min)",
        "block_minutes": "Available (min)",
        "unused_minutes": "Unused (min)",
        "utilization_pct": "Utilization (%)",
    })
    st.dataframe(summary, hide_index=True, width="stretch")

# =========================================================
# Block utilization
# =========================================================


def render_block_utilization(
    schedule: pd.DataFrame,
    utilization: pd.DataFrame | None = None,
) -> None:
    """Render a visual summary of maintenance block utilization."""

    st.subheader("Block utilization")

    if utilization is None or utilization.empty:
        st.info("No utilization diagnostics are available.")
        return

    data = utilization.copy()

    # Only blocks actually used by the optimizer.
    used = data.loc[
        data["tasks_scheduled"] > 0
    ].copy()

    if used.empty:
        st.info("No maintenance blocks were used.")
        return

    average_utilization = used["utilization_pct"].mean()
    total_scheduled = used["scheduled_minutes"].sum()
    total_unused = used["unused_minutes"].sum()

    cards = st.columns(3)

    cards[0].metric(
        "Used blocks",
        f"{len(used):,}",
    )

    cards[1].metric(
        "Average utilization",
        f"{average_utilization:.1f}%",
    )

    cards[2].metric(
        "Unused capacity",
        f"{total_unused:,.0f} min",
    )

    st.markdown("#### Lowest-utilized maintenance blocks")
    st.caption("Showing the 15 used blocks with the most remaining capacity.")

    chart_data = (
        used[
            [
                "block_window_id",
                "utilization_pct",
            ]
        ]
        .sort_values(
            "utilization_pct",
            ascending=True,
        )
        .head(15)
    )

    figure = px.bar(
        chart_data,
        x="block_window_id",
        y="utilization_pct",
        title="Maintenance block utilization",
        labels={
            "block_window_id": "Block",
            "utilization_pct": "Utilization (%)",
        },
    )

    figure.update_yaxes(
        range=[0, 100],
        ticksuffix="%",
    )

    figure.update_layout(
        showlegend=False,
        margin={"l": 0, "r": 0, "t": 45, "b": 0},
    )

    st.plotly_chart(
        figure,
        width="stretch",
    )

# =========================================================
# Unscheduled work
# =========================================================


def render_unscheduled_work(
    tasks: pd.DataFrame,
    schedule: pd.DataFrame,
    infeasibility: pd.DataFrame | None = None,
) -> None:
    """Render unscheduled maintenance work with actionable diagnostics."""

    st.subheader("Work not scheduled")

    scheduled_ids = set(schedule["task_id"])
    unscheduled = tasks.loc[
        ~tasks["task_id"].isin(scheduled_ids)
    ].copy()

    if unscheduled.empty:
        st.success("All maintenance tasks were scheduled.")
        return

    # -----------------------------------------------------
    # Attach diagnostic reasons when available.
    # -----------------------------------------------------

    if infeasibility is not None and not infeasibility.empty and "task_id" in infeasibility.columns:
        reason_columns = [
            column
            for column in (
                "task_id",
                "reason",
                "infeasibility_reason",
            )
            if column in infeasibility.columns
        ]

        diagnostic = infeasibility[reason_columns].drop_duplicates("task_id")
        unscheduled = unscheduled.merge(
            diagnostic,
            on="task_id",
            how="left",
        )
    else:
        unscheduled["reason"] = pd.NA
        unscheduled["infeasibility_reason"] = pd.NA

    def _raw_reason(row: pd.Series) -> str:
        for column in ("infeasibility_reason", "reason"):
            value = row.get(column)
            if pd.notna(value) and str(value).strip():
                return str(value).strip()
        return ""

    def _reason_category(row: pd.Series) -> str:
        raw = _raw_reason(row).upper()
        if "BLOCK_TOO_SHORT" in raw or "TOO SHORT" in raw:
            return "Block too short"
        if "TIMING_CONFLICT" in raw or "TIMING CONFLICT" in raw:
            return "Timing conflict"
        if "NO_FEASIBLE_BLOCK" in raw or "NO AVAILABLE BLOCK" in raw:
            return "No feasible block"
        if "OPTIMIZER_SELECTION" in raw or "OPTIMIZER" in raw:
            return "Not selected by optimizer"
        return "No specific reason recorded"

    unscheduled["reason_category"] = unscheduled.apply(
        _reason_category,
        axis=1,
    )

    criticality_order = {
        "critical": 0,
        "high": 1,
        "medium": 2,
        "low": 3,
    }
    urgency_order = {
        "high": 0,
        "medium": 1,
        "low": 2,
    }

    unscheduled["criticality_rank"] = (
        unscheduled["criticality"]
        .astype(str)
        .str.lower()
        .map(criticality_order)
        .fillna(99)
    )
    unscheduled["urgency_rank"] = (
        unscheduled["urgency"]
        .astype(str)
        .str.lower()
        .map(urgency_order)
        .fillna(99)
    )

    if "deadline" in unscheduled.columns:
        unscheduled["deadline"] = pd.to_datetime(
            unscheduled["deadline"],
            errors="coerce",
        )

    unscheduled = unscheduled.sort_values(
        ["criticality_rank", "urgency_rank", "deadline", "task_id"]
    )

    # -----------------------------------------------------
    # Attention summary
    # -----------------------------------------------------

    critical_count = int(
        (unscheduled["criticality"].astype(str).str.lower() == "critical").sum()
    )
    high_count = int(
        (unscheduled["criticality"].astype(str).str.lower() == "high").sum()
    )
    no_feasible_count = int(
        unscheduled["reason_category"].isin(
            ["Block too short", "Timing conflict", "No feasible block"]
        ).sum()
    )
    optimizer_count = int(
        (unscheduled["reason_category"] == "Not selected by optimizer").sum()
    )

    st.warning(
        f"{len(unscheduled):,} maintenance task(s) were not included in the optimized plan."
    )

    summary_cols = st.columns(4)
    summary_cols[0].metric("Unscheduled", f"{len(unscheduled):,}")
    summary_cols[1].metric("Critical", f"{critical_count:,}")
    summary_cols[2].metric("High priority", f"{high_count:,}")
    summary_cols[3].metric("Feasibility issues", f"{no_feasible_count:,}")

    # -----------------------------------------------------
    # Reason breakdown
    # -----------------------------------------------------

    st.markdown("#### Why work was not scheduled")

    reason_counts = (
        unscheduled["reason_category"]
        .value_counts()
        .rename_axis("Reason")
        .reset_index(name="Tasks")
    )

    reason_col, chart_col = st.columns([1, 2])

    with reason_col:
        st.dataframe(
            reason_counts,
            hide_index=True,
            width="stretch",
        )

    with chart_col:
        reason_chart = px.bar(
            reason_counts,
            x="Tasks",
            y="Reason",
            orientation="h",
            text="Tasks",
            title="Unscheduled tasks by reason",
        )
        reason_chart.update_layout(
            showlegend=False,
            margin={"l": 0, "r": 0, "t": 45, "b": 0},
        )
        reason_chart.update_traces(
            textposition="outside",
            cliponaxis=False,
        )
        st.plotly_chart(reason_chart, width="stretch")

    if optimizer_count:
        st.caption(
            f"{optimizer_count} task(s) had feasible candidates but were not selected in the optimized plan."
        )

    # -----------------------------------------------------
    # Priority-first worklist
    # -----------------------------------------------------

    st.markdown("#### Priority worklist")
    st.caption(
        "Unscheduled work is ordered by criticality, urgency, and required completion date."
    )

    display = unscheduled.copy()
    display["criticality"] = display["criticality"].astype(str).str.title()
    display["urgency"] = display["urgency"].astype(str).str.title()

    if "deadline" in display.columns:
        display["deadline"] = display["deadline"].dt.strftime("%Y-%m-%d")

    display_columns = [
        "task_id",
        "department",
        "section_id",
        "task_type",
        "criticality",
        "urgency",
        "estimated_duration_min",
        "deadline",
        "reason_category",
    ]
    display_columns = [
        column for column in display_columns if column in display.columns
    ]

    st.dataframe(
        display[display_columns],
        hide_index=True,
        width="stretch",
        height=340,
    )

    # -----------------------------------------------------
    # Selected-task explanation
    # -----------------------------------------------------

    selected_task_id = st.selectbox(
        "Inspect an unscheduled task",
        unscheduled["task_id"].tolist(),
        key="unscheduled_task_explorer",
    )

    selected = unscheduled.loc[
        unscheduled["task_id"] == selected_task_id
    ].iloc[0]

    detail_cols = st.columns(4)
    detail_cols[0].metric("Department", str(selected["department"]))
    detail_cols[1].metric("Section", str(selected["section_id"]))
    detail_cols[2].metric(
        "Criticality",
        str(selected["criticality"]).title(),
    )
    detail_cols[3].metric(
        "Duration",
        f"{int(selected['estimated_duration_min'])} min",
    )

    category = selected["reason_category"]
    raw_reason = _raw_reason(selected)

    explanation = {
        "Block too short": (
            "The available maintenance windows for this task are shorter "
            "than the work duration, so the task cannot fit into those windows."
        ),
        "Timing conflict": (
            "The task's allowed timing does not line up with a suitable "
            "maintenance window."
        ),
        "No feasible block": (
            "No available block satisfied the basic timing and duration "
            "requirements for this task."
        ),
        "Not selected by optimizer": (
            "The task had at least one feasible scheduling option, but the "
            "optimizer chose other work for the available plan."
        ),
        "No specific reason recorded": (
            "The available diagnostics do not contain a specific reason for "
            "this task's exclusion."
        ),
    }[category]

    st.markdown(f"**Why:** {explanation}")

    if raw_reason:
        st.caption(f"Diagnostic classification: {raw_reason}")

    if "deadline" in selected.index and pd.notna(selected["deadline"]):
        st.caption(
            "Required completion date: "
            f"{pd.Timestamp(selected['deadline']).strftime('%Y-%m-%d')}"
        )

    # =========================================================
# Why this block?
# =========================================================


def render_decision_explanation(
    tasks: pd.DataFrame,
    schedule: pd.DataFrame,
    diagnostics: pd.DataFrame | None,
    utilization: pd.DataFrame | None,
    trains: pd.DataFrame | None = None,
) -> None:
    """Explain why a scheduled task was assigned to its block."""

    st.subheader("Why this block?")

    if schedule.empty:
        st.info("No scheduled tasks are available.")
        return

    selected_task_id = st.selectbox(
        "Select a maintenance task",
        schedule["task_id"].tolist(),
        key="why_this_block_task",
    )

    selected = schedule.loc[
        schedule["task_id"] == selected_task_id
    ].iloc[0]

    task_matches = tasks.loc[
        tasks["task_id"] == selected_task_id
    ]

    task = (
        task_matches.iloc[0]
        if not task_matches.empty
        else selected
    )

    block_id = selected["block_window_id"]

    st.markdown(
        f"### {selected_task_id}"
    )

    st.caption(
        "Decision: Scheduled by the RailBlock optimization engine"
    )

    detail_cols = st.columns(4)

    detail_cols[0].metric(
        "Department",
        str(selected["department"]),
    )

    detail_cols[1].metric(
        "Section",
        str(selected["section_id"]),
    )

    detail_cols[2].metric(
        "Work",
        str(selected["task_type"]),
    )

    detail_cols[3].metric(
        "Duration",
        f"{int(selected['estimated_duration_min'])} min",
    )

    st.markdown("---")

    schedule_cols = st.columns(3)

    schedule_cols[0].metric(
        "Assigned block",
        block_id,
    )

    schedule_cols[1].metric(
        "Start",
        pd.Timestamp(
            selected["scheduled_start"]
        ).strftime("%d %b %H:%M"),
    )

    schedule_cols[2].metric(
        "End",
        pd.Timestamp(
            selected["scheduled_end"]
        ).strftime("%d %b %H:%M"),
    )

    # -----------------------------------------------------
    # Block information
    # -----------------------------------------------------

    block_row = None

    if utilization is not None and not utilization.empty:
        matches = utilization.loc[
            utilization["block_window_id"] == block_id
        ]

        if not matches.empty:
            block_row = matches.iloc[0]

    st.markdown("#### Why RailBlock chose this block")

    st.success(
    "RailBlock scheduled this task in the selected block "
    "because the work fits the available window while "
    "meeting its timing and operational requirements."
)
    reasons: list[str] = []

    criticality = str(
        task.get("criticality", "")
    ).lower()

    urgency = str(
        task.get("urgency", "")
    ).lower()

    if criticality == "critical":
        reasons.append(
            "This work has critical importance and was given high priority."
        )
    elif criticality == "high":
        reasons.append(
            "This work has high importance and was given higher priority."
        )
    elif criticality == "medium":
        reasons.append(
            "This work has moderate importance."
        )
    else:
        reasons.append(
            "This work has lower relative priority."
        )

    if urgency == "high":
        reasons.append(
            "The work is highly urgent, increasing the need to complete it promptly."
        )
    elif urgency == "medium":
        reasons.append(
            "The work has moderate urgency."
        )
    else:
        reasons.append(
            "There is less immediate urgency for this work."
        )

    # Timing
    scheduled_start = pd.Timestamp(
        selected["scheduled_start"]
    )

    scheduled_end = pd.Timestamp(
        selected["scheduled_end"]
    )

    earliest_start = task.get("earliest_start")

    deadline = task.get("deadline")

    if pd.notna(earliest_start):
        earliest_start = pd.Timestamp(earliest_start)

        if scheduled_start >= earliest_start:
            reasons.append(
                "The scheduled start satisfies the task's earliest-start requirement."
            )

    if pd.notna(deadline):
        deadline = pd.Timestamp(deadline)

        if scheduled_end <= deadline:
            reasons.append(
                "The scheduled completion is within the required deadline."
            )

    # Block fit
    if block_row is not None:
        block_minutes = float(
            block_row["block_minutes"]
        )

        scheduled_minutes = float(
            block_row["scheduled_minutes"]
        )

        unused_minutes = float(
            block_row["unused_minutes"]
        )

        utilization_pct = float(
            block_row["utilization_pct"]
        )

        if scheduled_minutes <= block_minutes:
            reasons.append(
                f"The maintenance work fits inside the available "
                f"{block_minutes:.0f}-minute block."
            )

        if unused_minutes > 0:
            reasons.append(
                f"{unused_minutes:.0f} minutes remain unused in this block."
            )

        st.markdown("---")

        st.markdown("#### Block utilization")

        utilization_cols = st.columns(3)

        utilization_cols[0].metric(
            "Block capacity",
            f"{block_minutes:.0f} min",
        )

        utilization_cols[1].metric(
            "Maintenance",
            f"{scheduled_minutes:.0f} min",
        )

        utilization_cols[2].metric(
            "Utilization",
            f"{utilization_pct:.1f}%",
        )

        st.progress(
            min(max(utilization_pct / 100, 0.0), 1.0)
        )

    # -----------------------------------------------------
    # Multi-department coordination
    # -----------------------------------------------------

    block_tasks = schedule.loc[
        schedule["block_window_id"] == block_id
    ]

    departments = sorted(
        block_tasks["department"]
        .dropna()
        .unique()
    )

    if len(departments) >= 2:
        reasons.append(
            "This block coordinates maintenance work from "
            f"{', '.join(departments)}."
        )

        st.success(
            "Multi-department coordination: "
            f"{' + '.join(departments)}"
        )

    # -----------------------------------------------------
    # Train protection
    # -----------------------------------------------------

    train_conflict = False

    overlapping_trains = 0

    if trains is not None and not trains.empty:
        block_section = selected["section_id"]
        service_date = pd.Timestamp(
            selected["block_date"]
        ).date()

        section_trains = trains.loc[
            trains["section_id"] == block_section
        ].copy()

        if not section_trains.empty:
            section_trains["service_date"] = pd.to_datetime(
                section_trains["service_date"]
            )

            section_trains = section_trains.loc[
                section_trains["service_date"].dt.date
                == service_date
            ]

            for _, train in section_trains.iterrows():
                train_start = pd.Timestamp(
                    f"{service_date} {train['entry_time']}"
                )

                train_end = pd.Timestamp(
                    f"{service_date} {train['exit_time']}"
                )

                if (
                    scheduled_start < train_end
                    and scheduled_end > train_start
                ):
                    overlapping_trains += 1

            train_conflict = overlapping_trains > 0

    if train_conflict:
        st.error(
            f"{overlapping_trains} scheduled train movement(s) "
            "overlap this maintenance period."
        )
        reasons.append(
            "Train movement overlap was detected and should be reviewed."
        )
    else:
        reasons.append(
            "No scheduled train movement overlaps the maintenance period."
        )

    # -----------------------------------------------------
    # Display reasons
    # -----------------------------------------------------

    for reason in reasons:
        st.markdown(f"✓ {reason}")

    # -----------------------------------------------------
    # Diagnostics engine explanation
    # -----------------------------------------------------

    if diagnostics is not None and not diagnostics.empty:
        diagnostic_match = diagnostics.loc[
            diagnostics["task_id"] == selected_task_id
        ]

        if not diagnostic_match.empty:
            diagnostic = diagnostic_match.iloc[0]

            explanation = diagnostic.get(
                "explanation",
                "",
            )

            if pd.notna(explanation) and str(explanation).strip():
                st.info(
                    f"RailBlock decision summary: {explanation}"
                )
                


# =========================================================
# Main planning view
# =========================================================


def render_planning_dashboard(
    tasks: pd.DataFrame,
    trains: pd.DataFrame | None = None,
) -> None:
    """Render the complete optimized planning section."""

    try:
        outputs = load_planning_outputs()
    except FileNotFoundError as error:
        st.error("The optimized RailBlock plan is not available.")
        st.code(str(error))
        return

    schedule = _prepare_schedule(outputs["schedule"])

    diagnostics = outputs.get("diagnostics")
    utilization = outputs.get("utilization")
    infeasibility = outputs.get("infeasibility")

    render_planning_kpis(tasks, schedule)

    st.divider()

    filtered_schedule = filter_schedule(schedule)

    render_schedule_table(filtered_schedule)

    st.divider()

    render_decision_explanation(
        tasks,
        schedule,
        diagnostics,
        utilization,
        trains,
    )

    st.divider()

    render_weekly_block_utilization(
        filtered_schedule,
        utilization,
    )

    st.divider()

    render_block_utilization(
        filtered_schedule,
        utilization,
    )

    st.divider()

    render_unscheduled_work(
        tasks,
        schedule,
        infeasibility,
    )