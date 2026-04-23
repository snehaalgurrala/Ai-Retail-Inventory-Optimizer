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
</style>
"""


def apply_page_style() -> None:
    """Apply light-touch component styling without changing Streamlit chrome."""
    st.markdown(DASHBOARD_CSS, unsafe_allow_html=True)


def render_section_header(icon: str, title: str, subtitle: str = "") -> None:
    """Compatibility wrapper for the shared UI component."""
    from frontend.components.ui_components import render_section_header as component

    component(title=title, subtitle=subtitle, icon=icon)


def render_page_header(title: str, subtitle: str = "") -> None:
    """Compatibility wrapper for the shared page header component."""
    from frontend.components.ui_components import render_page_header as component

    component(title, subtitle)


def render_kpi_card(
    title: str,
    value: str,
    subtext: str,
    color: str,
) -> None:
    """Compatibility wrapper for the shared UI component."""
    from frontend.components.ui_components import render_kpi_card as component

    component(title, value, subtext, color)


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
