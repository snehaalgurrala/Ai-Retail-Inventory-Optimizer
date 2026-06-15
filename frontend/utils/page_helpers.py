import sys
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from backend.utils.data_loader import load_all_data  # noqa: E402
from frontend.components.theme import THEME_COLORS, apply_enterprise_theme  # noqa: E402


CHART_COLORS = {
    "blue": THEME_COLORS["primary_navy"],
    "purple": "#476C8B",
    "orange": THEME_COLORS["warning_orange"],
    "green": THEME_COLORS["fresh_green"],
    "red": THEME_COLORS["risk_red"],
}

CHART_PALETTE = [
    CHART_COLORS["blue"],
    CHART_COLORS["green"],
    CHART_COLORS["purple"],
    THEME_COLORS["soft_green"],
    "#7EA2BE",
    CHART_COLORS["orange"],
    "#8FB8D2",
    CHART_COLORS["red"],
]

CHART_GRID_COLOR = "rgba(127, 127, 127, 0.18)"
CHART_AXIS_LINE_COLOR = "rgba(127, 127, 127, 0.24)"
CHART_TRANSPARENT = "rgba(0,0,0,0)"
CHART_SEPARATOR = "rgba(100, 116, 139, 0.32)"


DASHBOARD_CSS = """
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 2.5rem;
}
/* Custom KPI cards (render_kpi_card): wrap instead of clipping so the full
   label and value are always visible. Matches the Customer Intelligence cards. */
.airio-kpi-card {
    position: relative;
    border: 1px solid var(--airio-border, #D8E2EC);
    border-top: 4px solid var(--airio-primary-navy, #183F5F);
    border-radius: 16px;
    background: linear-gradient(180deg, #FFFFFF 0%, #FCFDFE 100%);
    box-shadow: 0 10px 22px rgba(10, 31, 51, 0.06);
    padding: 0.95rem 1.05rem;
    min-height: 128px;
    height: 100%;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    transition: transform 0.18s ease, box-shadow 0.18s ease;
}
.airio-kpi-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 20px 40px rgba(10, 31, 51, 0.13);
}
/* Faint accent wash that intensifies on hover for a premium feel. */
.airio-kpi-card::after {
    content: "";
    position: absolute;
    top: -40%;
    right: -20%;
    width: 150px;
    height: 150px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(24,63,95,0.06), transparent 70%);
    opacity: 0.8;
    transition: opacity 0.18s ease;
    pointer-events: none;
}
.airio-kpi-card:hover::after { opacity: 1; }
.airio-kpi-card.blue { border-top-color: #183F5F; }
.airio-kpi-card.purple { border-top-color: #6D4FB0; }
.airio-kpi-card.orange { border-top-color: #C76A12; }
.airio-kpi-card.green { border-top-color: #6CB33F; }
.airio-kpi-card.red { border-top-color: #B42318; }
.airio-kpi-head {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.3rem;
}
.airio-kpi-icon {
    flex: 0 0 auto;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 30px;
    height: 30px;
    border-radius: 9px;
    background: var(--airio-soft-blue, #EAF1F7);
    font-size: 1rem;
    line-height: 1;
}
.airio-kpi-card.green .airio-kpi-icon { background: rgba(108, 179, 63, 0.16); }
.airio-kpi-card.orange .airio-kpi-icon { background: var(--airio-soft-amber, #FFF7E8); }
.airio-kpi-card.red .airio-kpi-icon { background: var(--airio-soft-red, #FFF1F2); }
.airio-kpi-card.purple .airio-kpi-icon { background: rgba(109, 79, 176, 0.12); }
.airio-kpi-kicker {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    font-weight: 700;
    color: rgba(10, 31, 51, 0.6);
    overflow-wrap: anywhere;
    word-break: break-word;
}
.airio-kpi-value {
    font-size: 1.62rem;
    font-weight: 800;
    color: var(--airio-deep-navy, #0A1F33);
    line-height: 1.16;
    overflow-wrap: anywhere;
    word-break: break-word;
}
.airio-kpi-note {
    margin-top: auto;
    padding-top: 0.45rem;
    font-size: 0.76rem;
    color: rgba(10, 31, 51, 0.6);
    overflow-wrap: anywhere;
    word-break: break-word;
}
.airio-kpi-support {
    margin-top: 0.35rem;
    font-size: 0.78rem;
    font-weight: 700;
    color: var(--airio-primary-navy, #183F5F);
    overflow-wrap: anywhere;
    word-break: break-word;
}
/* Native st.metric fallback: stop nowrap/ellipsis clipping for any remaining
   metric cards (e.g. count tiles) so values stay fully visible. */
div[data-testid="stMetric"] { overflow: visible !important; }
div[data-testid="stMetricValue"],
div[data-testid="stMetricValue"] > div,
div[data-testid="stMetricLabel"],
div[data-testid="stMetricLabel"] > div,
div[data-testid="stMetricLabel"] p {
    white-space: normal !important;
    overflow: visible !important;
    overflow-wrap: anywhere !important;
    word-break: break-word !important;
    text-overflow: clip !important;
}
</style>
"""


def apply_page_style() -> None:
    """Apply light-touch component styling without changing Streamlit chrome."""
    apply_enterprise_theme()
    st.markdown(DASHBOARD_CSS, unsafe_allow_html=True)


def render_section_header(icon: str, title: str, subtitle: str = "") -> None:
    """Compatibility wrapper for the shared UI component."""
    from frontend.components.ui_components import render_section_header as component

    component(title=title, subtitle=subtitle, icon=icon)


def render_page_header(title: str, subtitle: str = "") -> None:
    """Compatibility wrapper for the shared page header component."""
    st.markdown(
        f"""
        <div class="airio-page-header">
          <div class="airio-page-header-title">{str(title)}</div>
          <div class="airio-page-header-subtitle">{str(subtitle or "")}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_card(
    title: str,
    value: str,
    subtext: str,
    color: str,
    icon: str = "",
    support: str = "",
) -> None:
    """Compatibility wrapper for the shared UI component."""
    from frontend.components.ui_components import render_kpi_card as component

    component(title, value, subtext, color, icon=icon, support=support)


def render_ai_insight_panel(
    insights: list[str],
    title: str = "AI Insights",
    icon: str = "🧠",
) -> None:
    """Compatibility wrapper for the shared AI insight panel component."""
    from frontend.components.ui_components import render_ai_insight_panel as component

    component(insights, title=title, icon=icon)


def clean_display_df(df: pd.DataFrame, placeholder: str = "-") -> pd.DataFrame:
    """Compatibility wrapper for the shared display-cleanup helper."""
    from frontend.components.ui_components import clean_display_df as component

    return component(df, placeholder=placeholder)


def render_table(
    df: pd.DataFrame,
    *,
    max_height: int = 460,
    empty_message: str = "No data to display.",
    formatters: dict | None = None,
) -> None:
    """Compatibility wrapper for the shared branded HTML table component."""
    from frontend.components.ui_components import render_table as component

    component(
        df,
        max_height=max_height,
        empty_message=empty_message,
        formatters=formatters,
    )


def apply_chart_theme(chart, height: int | None = 360):
    """Apply a light-touch Plotly style that still respects Streamlit themes."""
    if chart is None:
        return None

    chart_height = height if height is not None else chart.layout.height or 360

    chart.update_layout(
        template="plotly",
        paper_bgcolor=CHART_TRANSPARENT,
        plot_bgcolor=CHART_TRANSPARENT,
        title=None,
        colorway=CHART_PALETTE,
        margin=dict(l=28, r=24, t=12, b=30),
        height=chart_height,
        legend=dict(
            bgcolor=CHART_TRANSPARENT,
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )
    chart.update_xaxes(
        automargin=True,
        gridcolor=CHART_GRID_COLOR,
        zerolinecolor=CHART_AXIS_LINE_COLOR,
        linecolor=CHART_AXIS_LINE_COLOR,
        showline=True,
    )
    chart.update_yaxes(
        automargin=True,
        gridcolor=CHART_GRID_COLOR,
        zerolinecolor=CHART_AXIS_LINE_COLOR,
        linecolor=CHART_AXIS_LINE_COLOR,
        showline=True,
    )
    return chart


def style_bar_chart(chart, color: str = "blue"):
    """Apply theme and bar styling."""
    if chart is None:
        return None

    chart = apply_chart_theme(chart)
    chart.update_traces(
        marker_color=CHART_COLORS.get(color, CHART_COLORS["blue"]),
        marker_line_width=0,
        opacity=0.92,
    )
    return chart


def style_sales_trend_chart(chart):
    """Apply smooth line, markers, and fill styling for sales trend charts."""
    if chart is None:
        return None

    chart = apply_chart_theme(chart, height=390)
    chart.update_traces(
        mode="lines+markers",
        line=dict(
            color=CHART_COLORS["blue"],
            width=3,
            shape="spline",
            smoothing=1.2,
        ),
        marker=dict(
            size=7,
            color=CHART_COLORS["blue"],
            line=dict(width=2, color=CHART_SEPARATOR),
        ),
        fill="tozeroy",
        fillcolor="rgba(56, 189, 248, 0.18)",
    )
    chart.update_layout(hovermode="x unified")
    return chart


def style_donut_chart(chart):
    """Apply theme and donut-specific styling."""
    if chart is None:
        return None

    chart = apply_chart_theme(chart, height=390)
    chart.update_traces(
        hole=0.58,
        marker=dict(
            colors=CHART_PALETTE,
            line=dict(color=CHART_SEPARATOR, width=2),
        ),
        textposition="outside",
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>Stock units: %{value:,}<extra></extra>",
    )
    chart.update_layout(
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.12,
            xanchor="center",
            x=0.5,
        ),
    )
    return chart


def render_chart_card(title: str, subtitle: str, chart, empty_message: str) -> None:
    """Render a chart inside a native Streamlit bordered container."""
    with st.container(border=True):
        st.markdown(f"**{title}**")
        if subtitle:
            st.caption(subtitle)
        show_chart(chart, empty_message)


@st.cache_data
def load_app_data() -> dict[str, pd.DataFrame]:
    return load_all_data()


def load_data_or_stop() -> dict[str, pd.DataFrame]:
    try:
        return load_app_data()
    except Exception as error:
        st.error("Could not load the raw CSV data.")
        st.exception(error)
        st.stop()


def safe_sum(df: pd.DataFrame, column: str) -> int:
    if df.empty or column not in df.columns:
        return 0

    return int(pd.to_numeric(df[column], errors="coerce").fillna(0).sum())


def show_chart(chart, empty_message: str) -> None:
    if chart is None:
        st.info(empty_message)
        return

    apply_chart_theme(chart, height=None)
    st.plotly_chart(chart, use_container_width=True, theme="streamlit")
