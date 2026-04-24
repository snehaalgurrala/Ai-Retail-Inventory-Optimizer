from contextlib import contextmanager

import pandas as pd
import streamlit as st


def render_section_header(title: str, subtitle: str = "", icon: str = "") -> None:
    """Render a readable section header using native Streamlit text."""
    heading = f"{icon} {title}" if icon else str(title)
    st.subheader(heading)
    if subtitle:
        st.caption(str(subtitle))


def render_info_panel(
    title: str,
    body: str = "",
    status: str = "info",
) -> None:
    """Render a small native message panel."""
    message = f"**{title}**"
    if body:
        message = f"{message}\n\n{body}"

    if status == "success":
        st.success(message)
    elif status == "warning":
        st.warning(message)
    elif status == "error":
        st.error(message)
    else:
        st.info(message)


def render_empty_state(
    title: str,
    body: str = "",
    action_label: str | None = None,
    key: str | None = None,
) -> bool:
    """Render a consistent empty state and optionally return an action click."""
    with st.container(border=True):
        st.markdown(f"**{title}**")
        if body:
            st.caption(str(body))
        if action_label:
            return st.button(action_label, key=key, use_container_width=True)
    return False


def render_recommendation_summary(
    title: str,
    summary: str,
    insights: list[str] | None = None,
    icon: str = "",
    button_label: str = "Review",
    button_key: str | None = None,
) -> bool:
    """Render a compact recommendation summary with native components."""
    with st.container(border=True):
        heading = f"{icon} {title}" if icon else str(title)
        st.markdown(f"**{heading}**")
        st.metric("Summary", str(summary))
        for insight in (insights or [])[:3]:
            if insight:
                st.caption(str(insight))
        return st.button(button_label, key=button_key, use_container_width=True)


def render_orchestrator_summary_card(
    title: str,
    database_health: str,
    total_recommendations: int,
    high_priority_alerts: int,
    last_run_time: str,
    summary: str,
) -> None:
    """Render the main orchestrator summary card with native Streamlit blocks."""
    with st.container(border=True):
        st.subheader(str(title))
        metric_columns = st.columns(3, gap="medium")
        metric_columns[0].metric("Database Health", str(database_health))
        metric_columns[1].metric(
            "Total Recommendations",
            f"{int(total_recommendations):,}",
        )
        metric_columns[2].metric(
            "High Priority Alerts",
            f"{int(high_priority_alerts):,}",
        )
        st.caption(f"Last agent run: {last_run_time}")
        st.markdown(f"**Overall Analysis**  \n{summary}")


def render_agent_status_card(
    agent_name: str,
    description: str,
    latest_finding_count: int | str,
    priority_level: str,
    latest_insight: str,
) -> None:
    """Render a compact, theme-safe agent KPI card."""
    with st.container(border=True):
        st.markdown(f"**{agent_name}**")
        st.caption(str(description))
        metric_columns = st.columns(2, gap="small")
        metric_columns[0].metric("Latest Findings", str(latest_finding_count))
        metric_columns[1].metric("Priority", str(priority_level))
        st.caption(str(latest_insight))


def render_page_header(title: str, subtitle: str = "") -> None:
    """Compatibility helper for page-level headings."""
    st.title(str(title))
    if subtitle:
        st.caption(str(subtitle))


def render_kpi_card(
    title: str,
    value: str,
    subtext: str,
    color: str,
) -> None:
    """Compatibility helper for KPI cards with native metric text."""
    with st.container(border=True):
        st.metric(str(title), str(value))
        if subtext:
            st.caption(str(subtext))


@contextmanager
def render_content_container(
    title: str = "",
    subtitle: str = "",
    compact: bool = False,
):
    """Compatibility context manager around a native bordered container."""
    with st.container(border=True):
        if title:
            st.markdown(f"**{title}**")
        if subtitle:
            st.caption(str(subtitle))
        yield


def render_content_card_start(
    title: str = "",
    subtitle: str = "",
    compact: bool = False,
) -> None:
    """Compatibility helper for older call sites."""
    if title:
        st.markdown(f"**{title}**")
    if subtitle:
        st.caption(str(subtitle))


def render_content_card_end() -> None:
    """No-op kept for backward compatibility."""
    return None


def render_recommendation_card(
    recommendation: pd.Series,
    approve_callback=None,
    reject_callback=None,
    disabled_statuses: set[str] | None = None,
) -> None:
    """Render one detailed recommendation review card."""
    disabled_statuses = disabled_statuses or {"approved", "rejected"}
    type_icon_map = {
        "reorder": "\U0001F4E6",
        "stock_transfer": "\u2194",
        "discount": "\U0001F4B8",
        "clearance": "\u23F3",
        "supplier_risk_alert": "\u26A0",
        "overstock_alert": "\U0001F4CA",
        "stockout_prevention_alert": "\U0001F6A8",
    }

    recommendation_type = str(recommendation.get("recommendation_type", ""))
    icon = type_icon_map.get(recommendation_type, "\u2022")
    title = f"{icon} {recommendation_type} - {recommendation.get('product_name', '')}"

    store_id = recommendation.get("store_id", "")
    if pd.notna(store_id) and str(store_id):
        title += f" ({store_id})"

    with st.container(border=True):
        top_left, top_right = st.columns([3, 1])
        top_left.markdown(f"**{title.replace('_', ' ').title()}**")
        top_right.caption(f"Priority: {recommendation.get('priority', '')}")
        top_right.caption(f"Status: {recommendation.get('status', '')}")

        st.markdown(f"**Action:** {recommendation.get('action', '')}")
        st.caption(f"Reason: {recommendation.get('reason', '')}")
        st.caption(f"Evidence: {recommendation.get('evidence', '')}")

        quantity = recommendation.get("suggested_quantity", "")
        if pd.notna(quantity):
            quantity = str(quantity).strip()
            if quantity:
                st.caption(f"Suggested quantity: {quantity}")

        button_left, button_right, _ = st.columns([1, 1, 5])
        recommendation_id = recommendation.get("recommendation_id", "")
        current_status = str(recommendation.get("status", "pending"))
        disabled = current_status in disabled_statuses

        if button_left.button(
            "Approve",
            key=f"approve_{recommendation_id}",
            disabled=disabled,
        ) and approve_callback:
            approve_callback(recommendation)

        if button_right.button(
            "Reject",
            key=f"reject_{recommendation_id}",
            disabled=disabled,
        ) and reject_callback:
            reject_callback(recommendation)
