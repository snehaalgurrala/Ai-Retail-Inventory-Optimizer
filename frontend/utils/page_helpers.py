import sys
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from backend.utils.data_loader import load_all_data  # noqa: E402


DASHBOARD_CSS = """
<style>
:root {
    --app-bg: #0b1120;
    --card-bg: #111827;
    --card-bg-soft: #172033;
    --card-border: rgba(148, 163, 184, 0.18);
    --text-main: #e5eefb;
    --text-muted: #9ca3af;
    --primary-blue: #38bdf8;
    --primary-blue-soft: rgba(56, 189, 248, 0.16);
    --success: #22c55e;
    --warning: #f59e0b;
    --danger: #ef4444;
    --shadow-soft: 0 18px 45px rgba(0, 0, 0, 0.28);
    --radius-lg: 16px;
    --radius-md: 12px;
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(56, 189, 248, 0.10), transparent 28rem),
        linear-gradient(180deg, #0b1120 0%, #0f172a 100%);
    color: var(--text-main);
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2.5rem;
    max-width: 1280px;
}

h1 {
    color: #f8fafc;
    font-size: 2rem !important;
    font-weight: 750 !important;
    letter-spacing: 0;
    margin-bottom: 0.25rem !important;
}

h2, h3 {
    color: #f8fafc;
    font-weight: 680 !important;
    letter-spacing: 0;
}

p, label, span {
    color: inherit;
}

div[data-testid="stMetric"] {
    background: linear-gradient(145deg, rgba(17, 24, 39, 0.98), rgba(23, 32, 51, 0.92));
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
    color: #f8fafc !important;
    font-size: 1.65rem !important;
    font-weight: 750 !important;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(17, 24, 39, 0.82);
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
    background: rgba(17, 24, 39, 0.72);
    border: 1px solid var(--card-border);
    border-radius: var(--radius-lg);
    padding: 0.75rem;
    box-shadow: var(--shadow-soft);
}

.stButton > button {
    background: linear-gradient(135deg, #0ea5e9, #38bdf8);
    color: #04111f;
    border: 0;
    border-radius: var(--radius-md);
    padding: 0.55rem 1rem;
    font-weight: 700;
    box-shadow: 0 10px 24px rgba(14, 165, 233, 0.28);
}

.stButton > button:hover {
    background: linear-gradient(135deg, #38bdf8, #7dd3fc);
    color: #04111f;
    border: 0;
}

.stButton > button:disabled {
    background: #334155;
    color: #94a3b8;
    box-shadow: none;
}

section[data-testid="stSidebar"] {
    background: #0f172a;
    border-right: 1px solid var(--card-border);
}

div[data-baseweb="select"] > div,
div[data-testid="stTextInput"] input,
textarea {
    background: #111827 !important;
    border-color: rgba(148, 163, 184, 0.26) !important;
    border-radius: var(--radius-md) !important;
}

.stAlert {
    border-radius: var(--radius-md);
}

hr {
    border-color: rgba(148, 163, 184, 0.18);
    margin: 1.25rem 0;
}

.dashboard-section {
    background: rgba(17, 24, 39, 0.78);
    border: 1px solid var(--card-border);
    border-radius: var(--radius-lg);
    padding: 1.1rem;
    margin: 1rem 0;
    box-shadow: var(--shadow-soft);
}

.recommendation-card {
    background: linear-gradient(145deg, rgba(17, 24, 39, 0.98), rgba(15, 23, 42, 0.96));
    border: 1px solid var(--card-border);
    border-left: 4px solid var(--primary-blue);
    border-radius: var(--radius-lg);
    padding: 1rem;
    margin: 0.85rem 0;
    box-shadow: var(--shadow-soft);
}

.soft-caption {
    color: var(--text-muted);
    font-size: 0.9rem;
}
</style>
"""


def apply_page_style() -> None:
    """Apply the shared modern SaaS dashboard styling."""
    st.markdown(DASHBOARD_CSS, unsafe_allow_html=True)


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

    chart.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e5eefb"),
        title_font=dict(color="#f8fafc"),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    st.plotly_chart(chart, use_container_width=True)
