from contextlib import contextmanager
from html import escape

import pandas as pd
import streamlit as st


def apply_command_center_styles() -> None:
    """Inject theme-safe styles for the home dashboard command center."""
    st.markdown(
        """
        <style>
        .home-command-header {
            background: var(--airio-card, #FFFFFF);
            border: 1px solid var(--airio-border, #D8E2EC);
            border-top: 6px solid var(--airio-primary-navy, #183F5F);
            border-radius: 14px;
            padding: 1rem 1.15rem;
            box-shadow: 0 8px 20px rgba(10, 31, 51, 0.06);
            margin-bottom: 0.8rem;
        }
        .home-command-header::after {
            content: "";
            display: block;
            width: 72px;
            height: 3px;
            background: var(--airio-green, #6CB33F);
            border-radius: 999px;
            margin-top: 0.75rem;
        }
        .command-header-title {
            font-size: 1.85rem;
            font-weight: 800;
            line-height: 1.15;
            color: var(--airio-primary-navy, #183F5F);
            margin: 0;
        }
        .command-header-subtitle {
            color: rgba(10, 31, 51, 0.72);
            margin-top: 0.35rem;
            font-size: 0.95rem;
        }
        .command-meta {
            text-align: right;
            color: rgba(10, 31, 51, 0.66);
            font-size: 0.84rem;
            margin-bottom: 0.4rem;
        }
        .command-card,
        .agent-mini-card {
            border-radius: 14px;
            border: 1px solid var(--airio-border, #D8E2EC);
            border-top: 4px solid var(--airio-primary-navy, #183F5F);
            background: var(--airio-card, #FFFFFF);
            box-shadow: 0 8px 18px rgba(10, 31, 51, 0.055);
            overflow: hidden;
        }
        .command-card {
            padding: 1rem 1.1rem 1rem 1.1rem;
            margin-bottom: 0.6rem;
        }
        .agent-mini-card {
            padding: 0.85rem 0.9rem 0.9rem 0.9rem;
            min-height: 235px;
        }
        .card-accent {
            height: 3px;
            border-radius: 999px;
            margin: -0.05rem 0 0.85rem 0;
        }
        .accent-blue { background: #183F5F; }
        .accent-purple { background: #183F5F; }
        .accent-teal { background: #6CB33F; }
        .accent-orange { background: #C76A12; }
        .accent-green { background: #6CB33F; }
        .card-kicker {
            font-size: 0.76rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            font-weight: 700;
            color: rgba(10, 31, 51, 0.62);
            margin-bottom: 0.3rem;
        }
        .card-title {
            color: var(--airio-deep-navy, #0A1F33);
            font-size: 1.16rem;
            font-weight: 700;
            line-height: 1.2;
            margin: 0 0 0.2rem 0;
        }
        .card-copy {
            color: rgba(10, 31, 51, 0.80);
            font-size: 0.93rem;
            line-height: 1.45;
            margin: 0;
        }
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.75rem;
            margin: 1rem 0 0.9rem 0;
        }
        .mini-metric {
            padding: 0.7rem 0.8rem;
            border-radius: 12px;
            background: #F5F8FB;
            border: 1px solid var(--airio-border, #D8E2EC);
            border-left: 3px solid var(--airio-green, #6CB33F);
        }
        .mini-label {
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: rgba(10, 31, 51, 0.58);
            margin-bottom: 0.2rem;
        }
        .mini-value {
            font-size: 1.08rem;
            font-weight: 700;
            color: var(--airio-primary-navy, #183F5F);
            line-height: 1.2;
        }
        .agent-top {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 0.7rem;
            margin-bottom: 0.65rem;
        }
        .priority-badge {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 0.22rem 0.56rem;
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 700;
            border: 1px solid transparent;
            white-space: nowrap;
        }
        .priority-high {
            color: #7f1d1d;
            background: var(--airio-soft-red, #FFF1F2);
            border-color: rgba(180, 35, 24, 0.24);
        }
        .priority-medium {
            color: #854d0e;
            background: var(--airio-soft-amber, #FFF7E8);
            border-color: rgba(199, 106, 18, 0.24);
        }
        .priority-low, .priority-info {
            color: #285F12;
            background: rgba(108, 179, 63, 0.14);
            border-color: rgba(108, 179, 63, 0.25);
        }
        .agent-statline {
            display: flex;
            align-items: baseline;
            gap: 0.45rem;
            margin: 0.5rem 0 0.55rem 0;
        }
        .agent-stat {
            font-size: 1.45rem;
            font-weight: 800;
            color: var(--airio-primary-navy, #183F5F);
            line-height: 1.1;
        }
        .agent-statlabel {
            font-size: 0.82rem;
            color: rgba(10, 31, 51, 0.60);
        }
        .agent-action {
            margin-top: 0.7rem;
            padding-top: 0.65rem;
            border-top: 1px solid var(--airio-border, #D8E2EC);
            color: rgba(10, 31, 51, 0.84);
            font-size: 0.86rem;
            line-height: 1.4;
        }
        .agent-action strong {
            color: var(--airio-deep-navy, #0A1F33);
        }
        @media (max-width: 1200px) {
            .metric-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            .agent-mini-card { min-height: 250px; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


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


def render_command_center_orchestrator_card(
    database_health: str,
    total_recommendations: int,
    high_priority_alerts: int,
    last_run_time: str,
    low_stock_alert: str,
    top_risk: str,
    top_opportunity: str,
    executive_summary: str,
    executive_recommendation: str,
    summary_source: str = "",
    surplus_alternative_alert: str = "",
) -> None:
    """Render the premium orchestrator command card."""
    summary_note = ""
    if str(summary_source).strip().lower() == "fallback":
        summary_note = '<div class="card-copy" style="margin-top:0.65rem;">Using the latest data-based fallback summary.</div>'

    html = f"""
    <div class="command-card">
      <div class="card-accent accent-blue"></div>
      <div class="card-kicker">Orchestrator</div>
      <div class="card-title">Executive Agent Summary</div>
      <p class="card-copy">{escape(executive_summary)}</p>
      <div class="metric-grid">
        <div class="mini-metric">
          <div class="mini-label">Database Health</div>
          <div class="mini-value">{escape(str(database_health))}</div>
        </div>
        <div class="mini-metric">
          <div class="mini-label">Recommendations</div>
          <div class="mini-value">{int(total_recommendations):,}</div>
        </div>
        <div class="mini-metric">
          <div class="mini-label">High Priority</div>
          <div class="mini-value">{int(high_priority_alerts):,}</div>
        </div>
        <div class="mini-metric">
          <div class="mini-label">Last Run</div>
          <div class="mini-value">{escape(str(last_run_time))}</div>
        </div>
      </div>
      <div class="agent-action"><strong>Low stock alert:</strong> {escape(str(low_stock_alert))}</div>
      <div class="agent-action"><strong>Surplus/Alternative Alert:</strong> {escape(str(surplus_alternative_alert or 'No major surplus or transfer alternative detected right now.'))}</div>
      <div class="agent-action"><strong>Top risk:</strong> {escape(str(top_risk))}</div>
      <div class="agent-action" style="margin-top:0.35rem;"><strong>Top opportunity:</strong> {escape(str(top_opportunity))}</div>
      <div class="agent-action" style="margin-top:0.35rem;"><strong>Executive recommendation:</strong> {escape(str(executive_recommendation))}</div>
      {summary_note}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


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


def render_agent_command_card(
    agent_name: str,
    role_label: str,
    finding_count: int | str,
    priority_level: str,
    summary: str,
    recommended_action: str,
    accent: str = "blue",
) -> None:
    """Render a compact premium card for one specialist agent."""
    priority_text = _normalize_badge_text(priority_level)
    priority_class = f"priority-{str(priority_level or 'info').strip().lower()}"
    accent_class = f"accent-{escape(str(accent or 'blue'))}"

    html = f"""
    <div class="agent-mini-card">
      <div class="card-accent {accent_class}"></div>
      <div class="agent-top">
        <div>
          <div class="card-kicker">{escape(str(role_label))}</div>
          <div class="card-title">{escape(str(agent_name))}</div>
        </div>
        <span class="priority-badge {priority_class}">{escape(priority_text)}</span>
      </div>
      <div class="agent-statline">
        <div class="agent-stat">{escape(str(finding_count))}</div>
        <div class="agent-statlabel">latest findings</div>
      </div>
      <p class="card-copy">{escape(str(summary))}</p>
      <div class="agent-action"><strong>Action:</strong> {escape(str(recommended_action))}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_low_stock_alert_card(summary_text: str) -> None:
    """Render the summary banner for low-stock alerts."""
    html = f"""
    <div class="command-card" style="margin-top:0.2rem;">
      <div class="card-accent accent-orange"></div>
      <div class="card-kicker">Low Stock Alerts</div>
      <div class="card-title">🚨 Low Stock Alerts</div>
      <p class="card-copy">{escape(str(summary_text))}</p>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_surplus_alternative_alert_card(summary_text: str) -> None:
    """Render the summary banner for surplus stock and alternative availability."""
    html = f"""
    <div class="command-card" style="margin-top:0.2rem;">
      <div class="card-accent accent-teal"></div>
      <div class="card-kicker">Stock Surplus & Alternative Availability Alerts</div>
      <div class="card-title">📦 Stock Surplus & Alternatives</div>
      <p class="card-copy">{escape(str(summary_text))}</p>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def _normalize_badge_text(priority_level: str) -> str:
    text = str(priority_level or "").strip().title()
    return text or "Info"


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
        "transfer": "\u2194",
        "exclusive_availability": "E",
        "alternative_option": "A",
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
