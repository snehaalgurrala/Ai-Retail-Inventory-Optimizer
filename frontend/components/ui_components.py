import pandas as pd
import streamlit as st


def render_kpi_card(
    title: str,
    value: str,
    subtext: str,
    color: str,
) -> None:
    """Render a custom SaaS-style KPI card."""
    st.markdown(
        f"""
        <div class="kpi-card kpi-{color}">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-subtext">{subtext}</div>
            <div class="kpi-indicator"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(icon: str, title: str, subtitle: str = "") -> None:
    """Render a consistent icon-led section header."""
    subtitle_html = (
        f'<div class="section-header-subtitle">{subtitle}</div>' if subtitle else ""
    )
    st.markdown(
        f"""
        <div class="section-header">
            <div class="section-header-icon">{icon}</div>
            <div>
                <div class="section-header-title">{title}</div>
                {subtitle_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
        top_left.subheader(title.replace("_", " ").title())
        top_right.markdown(f"**Priority:** {recommendation.get('priority', '')}")
        top_right.markdown(f"**Status:** {recommendation.get('status', '')}")

        st.markdown(f"**Action:** {recommendation.get('action', '')}")
        st.markdown(f"**Reason:** {recommendation.get('reason', '')}")
        st.markdown(f"**Evidence:** {recommendation.get('evidence', '')}")

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
