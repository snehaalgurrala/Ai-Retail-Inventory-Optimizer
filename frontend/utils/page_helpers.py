import sys
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from backend.utils.data_loader import load_all_data  # noqa: E402


CHART_COLORS = {
    "blue": "#38bdf8",
    "purple": "#a78bfa",
    "orange": "#f59e0b",
    "green": "#22c55e",
    "red": "#ef4444",
}

CHART_PALETTE = [
    CHART_COLORS["blue"],
    CHART_COLORS["purple"],
    CHART_COLORS["orange"],
    CHART_COLORS["green"],
    "#14b8a6",
    "#f472b6",
    "#60a5fa",
    "#c084fc",
]


DASHBOARD_CSS = """
<style>
:root {
    --app-bg: var(--background-color, #0b1120);
    --card-bg: var(--secondary-background-color, #111827);
    --card-bg-soft: color-mix(in srgb, var(--secondary-background-color, #111827) 82%, var(--background-color, #0b1120));
    --card-border: color-mix(in srgb, var(--text-color, #e5eefb) 16%, transparent);
    --text-main: var(--text-color, #e5eefb);
    --text-muted: color-mix(in srgb, var(--text-color, #e5eefb) 62%, transparent);
    --primary-blue: var(--primary-color, #38bdf8);
    --primary-blue-soft: color-mix(in srgb, var(--primary-color, #38bdf8) 16%, transparent);
    --success: #22c55e;
    --warning: #f59e0b;
    --danger: #ef4444;
    --shadow-soft: 0 18px 45px rgba(0, 0, 0, 0.28);
    --radius-lg: 16px;
    --radius-md: 12px;
}

.stApp {
    background:
        radial-gradient(circle at top left, var(--primary-blue-soft), transparent 28rem),
        linear-gradient(180deg, var(--app-bg) 0%, color-mix(in srgb, var(--app-bg) 88%, var(--card-bg)) 100%);
    color: var(--text-main);
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2.5rem;
    max-width: 1280px;
}

h1 {
    color: var(--text-main);
    font-size: 2.15rem !important;
    font-weight: 750 !important;
    letter-spacing: 0;
    margin-bottom: 0.45rem !important;
}

h2, h3 {
    color: var(--text-main);
    font-weight: 680 !important;
    letter-spacing: 0;
    margin-top: 1rem !important;
    margin-bottom: 0.65rem !important;
}

p, label, span {
    color: inherit;
}

div[data-testid="stMetric"] {
    background: linear-gradient(145deg, var(--card-bg), var(--card-bg-soft));
    border: 1px solid var(--card-border);
    border-radius: var(--radius-lg);
    padding: 1rem 1.1rem;
    box-shadow: var(--shadow-soft);
}

div[data-testid="stMetric"] label,
div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
    color: var(--text-muted) !important;
    font-size: 0.82rem !important;
}

div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: var(--text-main) !important;
    font-size: 1.65rem !important;
    font-weight: 750 !important;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: color-mix(in srgb, var(--card-bg) 82%, transparent);
    border: 1px solid var(--card-border);
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow-soft);
    padding: 0.25rem;
}

div[data-testid="stDataFrame"] {
    border: 1px solid var(--card-border);
    border-radius: var(--radius-lg);
    overflow: hidden;
    box-shadow: 0 12px 28px rgba(0, 0, 0, 0.20);
}

div[data-testid="stPlotlyChart"] {
    background: color-mix(in srgb, var(--card-bg) 72%, transparent);
    border: 1px solid var(--card-border);
    border-radius: var(--radius-lg);
    padding: 0.75rem;
    box-shadow: var(--shadow-soft);
}

.stButton > button {
    background: rgba(14, 165, 233, 0.10);
    color: #e0f2fe;
    border: 1px solid rgba(56, 189, 248, 0.55);
    border-radius: 999px;
    padding: 0.6rem 1rem;
    font-weight: 700;
    box-shadow: 0 10px 24px rgba(14, 165, 233, 0.18);
    transition: transform 140ms ease, border-color 140ms ease, background 140ms ease;
}

.stButton > button:hover {
    background: linear-gradient(135deg, rgba(14, 165, 233, 0.90), rgba(56, 189, 248, 0.92));
    color: #04111f;
    border: 1px solid rgba(125, 211, 252, 0.95);
    transform: translateY(-1px);
}

.stButton > button:disabled {
    background: #334155;
    color: #94a3b8;
    box-shadow: none;
}

section[data-testid="stSidebar"] {
    background: var(--card-bg);
    border-right: 1px solid var(--card-border);
}

div[data-baseweb="select"] > div,
div[data-testid="stTextInput"] input,
textarea {
    background: var(--card-bg) !important;
    border-color: var(--card-border) !important;
    border-radius: var(--radius-md) !important;
}

.stAlert {
    border-radius: var(--radius-md);
}

hr {
    border-color: rgba(148, 163, 184, 0.18);
    margin: 1.6rem 0;
}

.section-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin: 1.8rem 0 0.9rem;
}

.section-header-icon {
    display: grid;
    place-items: center;
    width: 42px;
    height: 42px;
    border-radius: 14px;
    background: rgba(56, 189, 248, 0.14);
    border: 1px solid rgba(56, 189, 248, 0.22);
    font-size: 1.25rem;
}

.section-header-title {
    color: var(--text-main);
    font-size: 1.35rem;
    font-weight: 800;
    line-height: 1.15;
}

.section-header-subtitle {
    color: var(--text-muted);
    font-size: 0.92rem;
    margin-top: 0.18rem;
}

.dashboard-section {
    background: color-mix(in srgb, var(--card-bg) 78%, transparent);
    border: 1px solid var(--card-border);
    border-radius: var(--radius-lg);
    padding: 1.1rem;
    margin: 1rem 0;
    box-shadow: var(--shadow-soft);
}

.chart-heading {
    color: var(--text-main);
    font-size: 1.05rem;
    font-weight: 750;
    margin-bottom: 0.2rem;
}

.chart-subtitle {
    color: var(--text-muted);
    font-size: 0.88rem;
    margin-bottom: 0.9rem;
}

.kpi-card {
    position: relative;
    min-height: 142px;
    background: linear-gradient(145deg, var(--card-bg), var(--card-bg-soft));
    border: 1px solid var(--card-border);
    border-radius: 18px;
    padding: 1.15rem 1.2rem;
    overflow: hidden;
    box-shadow: var(--shadow-soft);
    transition: transform 160ms ease, border-color 160ms ease, box-shadow 160ms ease;
}

.kpi-card:hover {
    transform: translateY(-3px);
    border-color: rgba(226, 232, 240, 0.28);
    box-shadow: 0 22px 54px rgba(0, 0, 0, 0.34);
}

.kpi-card::before {
    content: "";
    position: absolute;
    inset: 0 auto 0 0;
    width: 4px;
    background: var(--kpi-color);
}

.kpi-card::after {
    content: "";
    position: absolute;
    top: -34px;
    right: -30px;
    width: 110px;
    height: 110px;
    border-radius: 999px;
    background: var(--kpi-glow);
}

.kpi-title {
    color: var(--text-muted);
    font-size: 0.82rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 0.65rem;
}

.kpi-value {
    color: var(--text-main);
    font-size: 2rem;
    line-height: 1.05;
    font-weight: 800;
    letter-spacing: 0;
    margin-bottom: 0.7rem;
}

.kpi-subtext {
    color: var(--text-muted);
    font-size: 0.9rem;
    line-height: 1.35;
    max-width: 15rem;
}

.kpi-indicator {
    position: absolute;
    right: 1rem;
    bottom: 1rem;
    width: 12px;
    height: 12px;
    border-radius: 999px;
    background: var(--kpi-color);
    box-shadow: 0 0 0 6px var(--kpi-glow);
}

.kpi-blue {
    --kpi-color: #38bdf8;
    --kpi-glow: rgba(56, 189, 248, 0.18);
}

.kpi-purple {
    --kpi-color: #a78bfa;
    --kpi-glow: rgba(167, 139, 250, 0.18);
}

.kpi-orange {
    --kpi-color: #f59e0b;
    --kpi-glow: rgba(245, 158, 11, 0.18);
}

.kpi-red {
    --kpi-color: #ef4444;
    --kpi-glow: rgba(239, 68, 68, 0.18);
}

.recommendation-card {
    background: linear-gradient(145deg, var(--card-bg), var(--card-bg-soft));
    border: 1px solid var(--card-border);
    border-left: 4px solid var(--primary-blue);
    border-radius: var(--radius-lg);
    padding: 1rem;
    margin: 0.85rem 0;
    box-shadow: var(--shadow-soft);
}

.summary-card {
    position: relative;
    min-height: 230px;
    background: linear-gradient(145deg, var(--card-bg), var(--card-bg-soft));
    border: 1px solid var(--card-border);
    border-radius: 18px;
    padding: 1.15rem;
    margin-bottom: 0.65rem;
    overflow: hidden;
    box-shadow: var(--shadow-soft);
    transition: transform 160ms ease, border-color 160ms ease, box-shadow 160ms ease;
}

.summary-card:hover {
    transform: translateY(-3px);
    border-color: rgba(226, 232, 240, 0.28);
    box-shadow: 0 24px 58px rgba(0, 0, 0, 0.36);
}

.summary-card::before {
    content: "";
    position: absolute;
    inset: 0 auto 0 0;
    width: 5px;
    background: var(--summary-accent);
}

.summary-card::after {
    content: "";
    position: absolute;
    top: -42px;
    right: -36px;
    width: 132px;
    height: 132px;
    border-radius: 999px;
    background: var(--summary-glow);
}

.summary-card-top {
    position: relative;
    z-index: 1;
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 1rem;
}

.summary-card-title {
    color: var(--text-main);
    font-size: 1.05rem;
    font-weight: 780;
    margin-bottom: 0.5rem;
}

.summary-card-summary {
    color: var(--text-main);
    font-size: 1.55rem;
    font-weight: 800;
    letter-spacing: 0;
}

.summary-card-icon {
    display: grid;
    place-items: center;
    width: 44px;
    height: 44px;
    border-radius: 14px;
    background: var(--summary-glow);
    font-size: 1.35rem;
    color: var(--summary-accent);
    border: 1px solid var(--summary-glow);
}

.summary-card-insights {
    position: relative;
    z-index: 1;
    color: var(--text-muted);
    font-size: 0.9rem;
    line-height: 1.45;
    margin: 1rem 0 0;
    padding-left: 1.1rem;
}

.summary-card-insights li {
    margin-bottom: 0.45rem;
}

.summary-card-blue {
    --summary-accent: #38bdf8;
    --summary-glow: rgba(56, 189, 248, 0.16);
}

.summary-card-purple {
    --summary-accent: #a78bfa;
    --summary-glow: rgba(167, 139, 250, 0.16);
}

.summary-card-orange {
    --summary-accent: #f59e0b;
    --summary-glow: rgba(245, 158, 11, 0.16);
}

.summary-card-red {
    --summary-accent: #ef4444;
    --summary-glow: rgba(239, 68, 68, 0.16);
}

.soft-caption {
    color: var(--text-muted);
    font-size: 0.9rem;
}
</style>
"""

THEME_AWARE_CSS = """
<style>
:root {
    --ui-bg: var(--background-color, #0b1120);
    --ui-surface: var(--secondary-background-color, #111827);
    --ui-text: var(--text-color, #e5eefb);
    --ui-primary: var(--primary-color, #38bdf8);
    --ui-muted: color-mix(in srgb, var(--text-color, #e5eefb) 62%, transparent);
    --ui-border: color-mix(in srgb, var(--text-color, #e5eefb) 16%, transparent);
    --ui-card: color-mix(in srgb, var(--secondary-background-color, #111827) 92%, var(--background-color, #0b1120));
    --ui-card-soft: color-mix(in srgb, var(--secondary-background-color, #111827) 76%, var(--background-color, #0b1120));
    --ui-primary-soft: color-mix(in srgb, var(--primary-color, #38bdf8) 16%, transparent);
    --shadow-soft: 0 18px 45px color-mix(in srgb, #000000 22%, transparent);
}

.stApp {
    background:
        radial-gradient(circle at top left, color-mix(in srgb, var(--ui-primary) 12%, transparent), transparent 28rem),
        linear-gradient(180deg, var(--ui-bg) 0%, color-mix(in srgb, var(--ui-bg) 88%, var(--ui-surface)) 100%);
    color: var(--ui-text);
}

h1, h2, h3,
.section-header-title,
.chart-heading,
.summary-card-title,
.summary-card-summary,
.kpi-value,
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: var(--ui-text) !important;
}

.section-header-subtitle,
.chart-subtitle,
.kpi-title,
.kpi-subtext,
.summary-card-insights,
.soft-caption,
div[data-testid="stMetric"] label,
div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
    color: var(--ui-muted) !important;
}

.dashboard-section,
.summary-card,
.kpi-card,
.recommendation-card,
div[data-testid="stMetric"] {
    background: linear-gradient(145deg, var(--ui-card), var(--ui-card-soft));
    border-color: var(--ui-border);
    box-shadow: var(--shadow-soft);
}

div[data-testid="stPlotlyChart"],
div[data-testid="stDataFrame"],
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: color-mix(in srgb, var(--ui-surface) 76%, transparent);
    border-color: var(--ui-border);
    box-shadow: var(--shadow-soft);
}

.section-header-icon {
    background: var(--ui-primary-soft);
    border-color: color-mix(in srgb, var(--ui-primary) 28%, transparent);
}

section[data-testid="stSidebar"] {
    background: var(--ui-surface);
    border-right: 1px solid var(--ui-border);
}

div[data-baseweb="select"] > div,
div[data-testid="stTextInput"] input,
textarea {
    background: var(--ui-surface) !important;
    border-color: var(--ui-border) !important;
    color: var(--ui-text) !important;
}

hr {
    border-color: var(--ui-border);
}

.stButton > button {
    background: color-mix(in srgb, var(--ui-primary) 11%, transparent);
    color: var(--ui-text);
    border-color: color-mix(in srgb, var(--ui-primary) 54%, transparent);
}

.stButton > button:hover {
    background: color-mix(in srgb, var(--ui-primary) 86%, var(--ui-surface));
    color: var(--ui-text);
    border-color: var(--ui-primary);
}

.stButton > button:disabled {
    background: color-mix(in srgb, var(--ui-surface) 84%, transparent);
    color: var(--ui-muted);
    border-color: var(--ui-border);
}

.summary-card-icon {
    background: var(--summary-glow);
    color: var(--summary-accent);
}
</style>
"""


def apply_page_style() -> None:
    """Apply the shared modern SaaS dashboard styling."""
    st.markdown(DASHBOARD_CSS, unsafe_allow_html=True)
    st.markdown(THEME_AWARE_CSS, unsafe_allow_html=True)


def render_section_header(icon: str, title: str, subtitle: str = "") -> None:
    """Compatibility wrapper for the shared UI component."""
    from frontend.components.ui_components import render_section_header as component

    component(icon, title, subtitle)


def render_kpi_card(
    title: str,
    value: str,
    subtext: str,
    color: str,
) -> None:
    """Compatibility wrapper for the shared UI component."""
    from frontend.components.ui_components import render_kpi_card as component

    component(title, value, subtext, color)


def apply_chart_theme(chart, height: int = 360):
    """Apply a consistent dark Plotly chart theme."""
    if chart is None:
        return None

    chart.update_layout(
        template="plotly",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#64748b", size=12),
        title=None,
        colorway=CHART_PALETTE,
        margin=dict(l=24, r=24, t=16, b=24),
        height=height,
        hoverlabel=dict(
            bgcolor="rgba(15, 23, 42, 0.94)",
            bordercolor="rgba(148, 163, 184, 0.42)",
            font_color="#e2e8f0",
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#64748b"),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )
    chart.update_xaxes(
        gridcolor="rgba(148, 163, 184, 0.16)",
        zerolinecolor="rgba(148, 163, 184, 0.20)",
        linecolor="rgba(148, 163, 184, 0.20)",
    )
    chart.update_yaxes(
        gridcolor="rgba(148, 163, 184, 0.16)",
        zerolinecolor="rgba(148, 163, 184, 0.20)",
        linecolor="rgba(148, 163, 184, 0.20)",
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
            line=dict(width=2, color="rgba(15, 23, 42, 0.86)"),
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
            line=dict(color="rgba(15, 23, 42, 0.86)", width=2),
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
    """Render a chart inside a styled dashboard section."""
    with st.container():
        st.markdown('<div class="dashboard-section">', unsafe_allow_html=True)
        st.markdown(f'<div class="chart-heading">{title}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="chart-subtitle">{subtitle}</div>',
            unsafe_allow_html=True,
        )
        show_chart(chart, empty_message)
        st.markdown("</div>", unsafe_allow_html=True)


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

    apply_chart_theme(chart)
    st.plotly_chart(chart, use_container_width=True)
