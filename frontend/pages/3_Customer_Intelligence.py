from html import escape
from pathlib import Path
import math
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from backend.db import repository  # noqa: E402
from backend.services import customer_intelligence_service as cis  # noqa: E402
from frontend.utils.page_helpers import (  # noqa: E402
    CHART_COLORS,
    apply_page_style,
    render_ai_insight_panel,
    render_chart_card,
    render_page_header,
    render_table,
    style_bar_chart,
)


st.set_page_config(
    page_title="Customer Intelligence",
    page_icon="👥",
    layout="wide",
)


# Oracle is the single source of truth. "customer" here is the real end-customer
# dimension (BZ_MOCK_CUSTOMER -> ORDER_HEADER -> ORDER_LINE), not the branch.
@st.cache_data(show_spinner="Loading customer & order data from Oracle...")
def load_customer_intelligence_data() -> dict[str, pd.DataFrame]:
    return {
        "customers": repository.load_customers(safe=True),
        "orders": repository.load_orders(safe=True),
        "order_lines": repository.load_order_lines(safe=True),
        "products": repository.load_products(safe=True),
        "inventory": repository.load_inventory(safe=True),
    }


def money(value: float) -> str:
    return f"${float(value):,.0f}"


def compact(value: str, max_len: int = 26) -> str:
    value = str(value or "")
    return value if len(value) <= max_len else f"{value[: max_len - 1]}…"


SPOTLIGHT_CSS = """
<style>
.ci-spotlight-card {
    border-radius: 14px;
    border: 1px solid var(--airio-border, #D8E2EC);
    border-top: 4px solid var(--airio-primary-navy, #183F5F);
    background: var(--airio-card, #FFFFFF);
    box-shadow: 0 8px 18px rgba(10, 31, 51, 0.06);
    padding: 0.95rem 1rem 1rem 1rem;
    min-height: 168px;
}
.ci-spotlight-rank {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 700;
    color: rgba(10, 31, 51, 0.55);
}
.ci-spotlight-name {
    font-size: 1.05rem;
    font-weight: 800;
    color: var(--airio-primary-navy, #183F5F);
    line-height: 1.2;
    margin: 0.15rem 0 0.55rem 0;
    min-height: 2.5em;
}
.ci-spotlight-revenue {
    font-size: 1.5rem;
    font-weight: 800;
    color: var(--airio-deep-navy, #0A1F33);
    line-height: 1.1;
}
.ci-spotlight-sub { font-size: 0.8rem; color: rgba(10, 31, 51, 0.6); margin-bottom: 0.55rem; }
.ci-spotlight-badges { display: flex; flex-wrap: wrap; gap: 0.35rem; }
.ci-badge {
    display: inline-flex;
    padding: 0.16rem 0.5rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 700;
    background: rgba(108, 179, 63, 0.14);
    color: #285F12;
    border: 1px solid rgba(108, 179, 63, 0.25);
}
.ci-badge.tier { background: #F0F4F8; color: #183F5F; border-color: #D8E2EC; }
div[data-testid="stMetric"] { min-height: 92px; }
div[data-testid="stMetric"] label {
    color: color-mix(in srgb, var(--text-color) 62%, transparent);
    font-weight: 700;
}
</style>
"""


def render_spotlight(spotlight: list[dict]) -> None:
    if not spotlight:
        st.info("No customers with orders are available to spotlight.")
        return
    columns = st.columns(len(spotlight), gap="medium")
    for index, (column, card) in enumerate(zip(columns, spotlight), start=1):
        with column:
            st.markdown(
                f"""
                <div class="ci-spotlight-card">
                  <div class="ci-spotlight-rank">#{index} Customer</div>
                  <div class="ci-spotlight-name">{escape(card['customer_name'])}</div>
                  <div class="ci-spotlight-revenue">{money(card['revenue'])}</div>
                  <div class="ci-spotlight-sub">{int(card['orders'])} orders</div>
                  <div class="ci-spotlight-badges">
                    <span class="ci-badge">{escape(card['segment'])}</span>
                    <span class="ci-badge tier">{escape(card['tier'])}</span>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def make_top_products_chart(product_df: pd.DataFrame, value_col: str, label: str):
    if product_df.empty:
        return None
    plot_df = product_df.sort_values(value_col, ascending=True)
    chart = px.bar(
        plot_df,
        x=value_col,
        y="product_name",
        orientation="h",
        text=value_col,
        labels={value_col: label, "product_name": "Product"},
    )
    template = "%{text:,.0f}" if value_col == "quantity" else "$%{text:,.0f}"
    chart.update_traces(texttemplate=template, textposition="outside")
    color = "green" if value_col == "quantity" else "blue"
    return style_bar_chart(chart, color)


def make_impact_chart(impact_df: pd.DataFrame):
    if impact_df.empty:
        return None
    plot_df = impact_df.sort_values("risk_score", ascending=True)
    chart = px.bar(
        plot_df,
        x="risk_score",
        y="customer_name",
        orientation="h",
        text="risk_score",
        labels={"risk_score": "Inventory Pressure Score", "customer_name": "Customer"},
    )
    chart.update_traces(texttemplate="%{text}", textposition="outside")
    return style_bar_chart(chart, "orange")


TREND_BADGE = {
    "Growing": ("#166534", "#dcfce7"),
    "Stable": ("#1e3a5f", "#e0ecf7"),
    "Declining": ("#991b1b", "#fee2e2"),
}
RISK_BADGE = {"High": ("#991b1b", "#fee2e2"), "Medium": ("#92400e", "#fef3c7")}


KPI_CSS = """
<style>
.ci-kpi-grid {
    display: grid;
    grid-template-columns: repeat(5, minmax(0, 1fr));
    gap: 0.7rem;
    margin: 0.15rem 0 0.5rem 0;
}
.ci-kpi-card { perspective: 1200px; height: 152px; outline: none; }
.ci-kpi-inner {
    position: relative; width: 100%; height: 100%;
    transition: transform .55s cubic-bezier(.2, .7, .2, 1);
    transform-style: preserve-3d;
}
.ci-kpi-card:hover .ci-kpi-inner,
.ci-kpi-card:focus-within .ci-kpi-inner { transform: rotateY(180deg); }
.ci-kpi-face {
    position: absolute; inset: 0;
    -webkit-backface-visibility: hidden; backface-visibility: hidden;
    border-radius: 14px;
    border: 1px solid var(--airio-border, #D8E2EC);
    background: var(--airio-card, #FFFFFF);
    box-shadow: 0 8px 18px rgba(10, 31, 51, 0.06);
    padding: 0.8rem 0.85rem;
    display: flex; flex-direction: column;
    overflow: hidden;
}
.ci-kpi-front { border-top: 4px solid var(--airio-primary-navy, #183F5F); }
.ci-kpi-card:hover .ci-kpi-front { box-shadow: 0 14px 30px rgba(10, 31, 51, 0.14); }
.ci-kpi-back {
    transform: rotateY(180deg);
    border-top: 4px solid var(--airio-green, #6CB33F);
    background: linear-gradient(180deg, #FFFFFF 0%, #F5F8FB 100%);
}
.ci-kpi-kicker {
    font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05em;
    font-weight: 700; color: rgba(10, 31, 51, 0.58); margin-bottom: 0.28rem;
}
.ci-kpi-value {
    color: var(--airio-deep-navy, #0A1F33); font-weight: 800;
    line-height: 1.12; overflow-wrap: anywhere; word-break: break-word;
}
.ci-kpi-value.num { font-size: 2rem; }
.ci-kpi-value.name { font-size: 1.02rem; }
.ci-kpi-note { margin-top: auto; font-size: 0.76rem; color: rgba(10, 31, 51, 0.6); }
.ci-kpi-hint {
    position: absolute; top: 0.5rem; right: 0.7rem;
    font-size: 0.66rem; font-weight: 700; letter-spacing: 0.03em;
    color: rgba(10, 31, 51, 0.3);
}
.ci-kpi-back-title {
    font-size: 0.72rem; font-weight: 800; text-transform: uppercase;
    letter-spacing: 0.04em; color: var(--airio-primary-navy, #183F5F);
    margin-bottom: 0.4rem;
}
.ci-kpi-line {
    font-size: 0.77rem; color: rgba(10, 31, 51, 0.82); line-height: 1.34;
    margin-bottom: 0.1rem; overflow-wrap: anywhere;
}
.ci-kpi-line b { color: var(--airio-deep-navy, #0A1F33); font-weight: 700; }
@media (max-width: 1200px) {
    .ci-kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .ci-kpi-card { height: 160px; }
}
@media (max-width: 560px) {
    .ci-kpi-grid { grid-template-columns: 1fr; }
}
</style>
"""


def _kpi_lines(pairs: list) -> str:
    """Render labelled back-face lines from (label, value) pairs (label optional)."""
    rows = []
    for label, value in pairs:
        if label:
            rows.append(f'<div class="ci-kpi-line"><b>{escape(str(label))}:</b> {escape(str(value))}</div>')
        else:
            rows.append(f'<div class="ci-kpi-line">{escape(str(value))}</div>')
    return "".join(rows)


def _kpi_card(kicker: str, value: str, value_class: str, note: str,
              back_title: str, lines: list) -> str:
    """One flip card: front (headline) -> back (3-5 supporting lines on hover).

    Built as a single whitespace-free string: leading indentation or blank lines
    would make Streamlit's Markdown parser treat the HTML as a code block and
    render the tags as literal text.
    """
    front = (
        '<div class="ci-kpi-face ci-kpi-front">'
        f'<div class="ci-kpi-kicker">{escape(kicker)}</div>'
        '<div class="ci-kpi-hint">hover ↻</div>'
        f'<div class="ci-kpi-value {value_class}">{escape(value)}</div>'
        f'<div class="ci-kpi-note">{escape(note)}</div>'
        '</div>'
    )
    back = (
        '<div class="ci-kpi-face ci-kpi-back">'
        f'<div class="ci-kpi-back-title">{escape(back_title)}</div>'
        f'{_kpi_lines(lines)}'
        '</div>'
    )
    return (
        '<div class="ci-kpi-card" tabindex="0">'
        f'<div class="ci-kpi-inner">{front}{back}</div>'
        '</div>'
    )


def render_executive_kpis(
    kpis: dict,
    top_customers_df: pd.DataFrame,
    trends_df: pd.DataFrame,
    abnormal_df: pd.DataFrame,
    impact_df: pd.DataFrame,
    dormant_df: pd.DataFrame,
) -> None:
    """Render the 5 KPI flip cards. Reuses already-computed values only."""
    # 1) Top Customer
    if not top_customers_df.empty:
        top = top_customers_df.iloc[0]
        top_lines = [
            ("Revenue", money(top["revenue"])),
            ("Contribution", f"{top['contribution_pct']:.1f}% of total"),
            ("Orders", f"{int(top['orders'])}  ·  Units {int(top['units'])}"),
            ("Profile", f"{top['customer_segment'] or '-'} · {top['contract_tier'] or '-'}"),
        ]
    else:
        top_lines = [("", "No orders recorded yet.")]

    # 2) Highest Growth
    growth_name = kpis["growth_customer_name"]
    growth_match = trends_df[trends_df["customer_name"] == growth_name] if not trends_df.empty else trends_df
    if not growth_match.empty:
        g = growth_match.iloc[0]
        growth_lines = [
            ("Revenue change", f"{g['change_pct']:+.0f}% vs first half"),
            ("First half", money(g["first_half_revenue"])),
            ("Second half", money(g["second_half_revenue"])),
            ("Trend", str(g["trend"])),
            ("", "Short order window — directional signal."),
        ]
        growth_note = f"+{kpis['growth_customer_change']:.0f}% vs first half" if kpis["growth_customer_change"] else "Short history"
    else:
        growth_lines = [("", "Not enough dated order history.")]
        growth_note = "Short history"

    # 3) Abnormal Orders
    if not abnormal_df.empty:
        high = int((abnormal_df["risk_level"] == "High").sum())
        medium = int((abnormal_df["risk_level"] == "Medium").sum())
        worst = abnormal_df.iloc[0]
        abnormal_lines = [
            ("Flagged lines", f"{len(abnormal_df)}"),
            ("Severity", f"{high} High · {medium} Medium"),
            ("Largest", f"{worst['customer_name']} (+{worst['deviation_pct']:.0f}%)"),
            ("Product", f"{worst['product_name']} — {int(worst['current_quantity'])} vs {worst['historical_avg']:.0f} avg"),
            ("", "Per-product baseline method."),
        ]
    else:
        abnormal_lines = [
            ("", "No lines exceed the per-product baseline."),
            ("", "Method: per-product average quantity."),
        ]

    # 4) Stockout-Risk Customers
    if not impact_df.empty:
        lead = impact_df.iloc[0]
        impact_lines = [
            ("Customers at risk", f"{len(impact_df)}"),
            ("Top pressure", f"{lead['customer_name']} (score {int(lead['risk_score'])})"),
            ("At-risk volume", f"{int(lead['at_risk_units'])} units · {int(lead['affected_products'])} product(s)"),
            ("", "Trigger: stock at/below reorder point."),
        ]
    else:
        impact_lines = [("", "No customers ordering at-risk products.")]

    # 5) Dormant Accounts
    if not dormant_df.empty:
        names = ", ".join(dormant_df["customer_name"].head(3).tolist())
        more = len(dormant_df) - 3
        if more > 0:
            names += f" +{more} more"
        segs = ", ".join(sorted(set(dormant_df.get("customer_segment", pd.Series(dtype=str)).dropna().astype(str)))[:3])
        dormant_lines = [
            ("Active, zero orders", f"{len(dormant_df)}"),
            ("Accounts", names),
            ("Segments", segs or "-"),
            ("", "Re-engagement opportunity, not an error."),
        ]
    else:
        dormant_lines = [("", "Every active customer has ordered.")]

    cards = [
        _kpi_card("Top Customer", kpis["top_customer_name"], "name",
                  money(kpis["top_customer_revenue"]) + " revenue",
                  "Top Customer — detail", top_lines),
        _kpi_card("Highest Growth", kpis["growth_customer_name"], "name",
                  growth_note, "Growth — detail", growth_lines),
        _kpi_card("Abnormal Orders", f"{kpis['abnormal_orders']:,}", "num",
                  "Lines above product baseline", "Abnormal Orders — detail", abnormal_lines),
        _kpi_card("Stockout-Risk Customers", f"{kpis['stockout_risk_customers']:,}", "num",
                  "Ordering at-risk products", "Inventory Pressure — detail", impact_lines),
        _kpi_card("Dormant Accounts", f"{kpis['dormant_accounts']:,}", "num",
                  "Active, zero orders", "Dormant Accounts — detail", dormant_lines),
    ]
    st.markdown(f'<div class="ci-kpi-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


# ===========================================================================
# AI Order Intelligence Center — hero (abnormal order detection)
# ===========================================================================
HERO_CSS = """
<style>
/* --- Section 1: AI Executive Summary briefing panel --- */
.ci-hero {
    position: relative;
    overflow: hidden;
    border-radius: 18px;
    padding: 1.35rem 1.6rem 1.5rem 1.6rem;
    margin: 0.2rem 0 1.1rem 0;
    color: #FFFFFF;
    background: linear-gradient(135deg, #0A1F33 0%, #183F5F 58%, #1E4E76 100%);
    box-shadow: 0 20px 44px rgba(10, 31, 51, 0.26);
}
.ci-hero::after {
    content: "";
    position: absolute;
    top: -45%;
    right: -8%;
    width: 360px;
    height: 360px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(108, 179, 63, 0.22), transparent 68%);
    pointer-events: none;
}
.ci-hero-kicker {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 800;
    color: rgba(255, 255, 255, 0.72);
}
.ci-hero-title {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    font-size: 1.32rem;
    font-weight: 800;
    line-height: 1.18;
    margin-top: 0.15rem;
}
.ci-hero-sub { font-size: 0.86rem; opacity: 0.82; margin-top: 0.25rem; }
.ci-hero-grid {
    position: relative;
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(155px, 1fr));
    gap: 0.7rem;
    margin-top: 1.05rem;
}
.ci-hero-stat {
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.16);
    border-radius: 12px;
    padding: 0.7rem 0.85rem;
    transition: transform 0.16s ease, background 0.16s ease;
}
.ci-hero-stat:hover { transform: translateY(-3px); background: rgba(255, 255, 255, 0.13); }
.ci-hero-stat.alert { border-color: rgba(248, 113, 113, 0.55); background: rgba(248, 113, 113, 0.16); }
.ci-hero-stat .k {
    font-size: 0.66rem; text-transform: uppercase; letter-spacing: 0.05em;
    opacity: 0.74; font-weight: 700;
}
.ci-hero-stat .v { font-size: 1.42rem; font-weight: 800; margin-top: 0.16rem; line-height: 1.12; }
.ci-hero-stat .v.name { font-size: 1.02rem; line-height: 1.22; }

/* --- Section 2: AI Customer Intelligence cards --- */
.ci-ai-card {
    position: relative;
    overflow: hidden;
    border-radius: 16px;
    border: 1px solid var(--airio-border, #D8E2EC);
    border-left: 5px solid var(--risk-color, #183F5F);
    background: linear-gradient(180deg, #FFFFFF 0%, #FBFDFE 100%);
    box-shadow: 0 10px 24px rgba(10, 31, 51, 0.07);
    padding: 1rem 1.1rem 1.05rem 1.1rem;
    transition: transform 0.18s ease, box-shadow 0.18s ease;
}
.ci-ai-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 24px 46px rgba(10, 31, 51, 0.16);
}
.ci-ai-head { display: flex; justify-content: space-between; align-items: flex-start; gap: 0.6rem; }
.ci-ai-rank {
    font-size: 0.7rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.04em; color: rgba(10, 31, 51, 0.52);
}
.ci-ai-name {
    font-size: 1.12rem; font-weight: 800; color: var(--airio-deep-navy, #0A1F33);
    line-height: 1.18; margin-top: 0.15rem;
}
.ci-risk-badge {
    flex: 0 0 auto;
    padding: 0.24rem 0.62rem; border-radius: 999px;
    font-size: 0.72rem; font-weight: 800; white-space: nowrap;
}
.ci-ai-metrics { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.5rem; margin: 0.8rem 0 0.55rem 0; }
.ci-ai-metric { background: var(--airio-soft-blue, #EAF1F7); border-radius: 10px; padding: 0.5rem 0.6rem; }
.ci-ai-metric .k {
    font-size: 0.64rem; text-transform: uppercase; letter-spacing: 0.04em;
    color: rgba(10, 31, 51, 0.6); font-weight: 700;
}
.ci-ai-metric .v {
    font-size: 1.02rem; font-weight: 800; color: var(--airio-deep-navy, #0A1F33);
    margin-top: 0.1rem; overflow-wrap: anywhere;
}
.ci-prod-line {
    font-size: 0.8rem; color: rgba(10, 31, 51, 0.72);
    line-height: 1.4; margin-bottom: 0.55rem; overflow-wrap: anywhere;
}
.ci-prod-line b { color: var(--airio-deep-navy, #0A1F33); }
.ci-ai-explain {
    background: linear-gradient(180deg, #FFFFFF 0%, #F7FBF3 100%);
    border-left: 3px solid var(--airio-green, #6CB33F);
    border-radius: 10px;
    padding: 0.6rem 0.72rem;
    font-size: 0.86rem; line-height: 1.46; color: rgba(10, 31, 51, 0.88);
}
.ci-ai-explain .tag {
    display: block; margin-bottom: 0.22rem;
    font-size: 0.66rem; font-weight: 800; text-transform: uppercase;
    letter-spacing: 0.05em; color: var(--airio-green, #6CB33F);
}

/* --- Section 3: expandable deep analysis --- */
.ci-deep-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.55rem; margin: 0.1rem 0 0.2rem 0; }
.ci-deep-stat {
    border: 1px solid var(--airio-border, #D8E2EC);
    border-radius: 10px; padding: 0.55rem 0.65rem; background: var(--airio-card, #FFFFFF);
}
.ci-deep-stat .k {
    font-size: 0.63rem; text-transform: uppercase; letter-spacing: 0.04em;
    color: rgba(10, 31, 51, 0.58); font-weight: 700;
}
.ci-deep-stat .v { font-size: 1.04rem; font-weight: 800; color: var(--airio-deep-navy, #0A1F33); margin-top: 0.12rem; }
.ci-deep-block {
    background: linear-gradient(180deg, #FFFFFF 0%, #F7FBF3 100%);
    border: 1px solid var(--airio-border, #D8E2EC);
    border-radius: 10px; padding: 0.6rem 0.78rem; margin-top: 0.55rem;
}
.ci-deep-block.actions { background: linear-gradient(180deg, #FFFFFF 0%, #FFF7E8 100%); }
.ci-deep-block-title {
    font-size: 0.67rem; font-weight: 800; text-transform: uppercase;
    letter-spacing: 0.05em; margin-bottom: 0.35rem;
}
.ci-deep-bullet {
    position: relative; padding-left: 1rem;
    font-size: 0.84rem; line-height: 1.5; color: rgba(10, 31, 51, 0.86);
}
.ci-deep-bullet::before { content: "•"; position: absolute; left: 0.2rem; color: var(--airio-green, #6CB33F); font-weight: 800; }
.ci-deep-text {
    font-size: 0.84rem; line-height: 1.55; color: rgba(10, 31, 51, 0.86);
    margin: 0.12rem 0 0.35rem 0; overflow-wrap: anywhere;
}
.ci-deep-text.closing {
    margin-top: 0.55rem; font-style: italic; color: rgba(10, 31, 51, 0.78);
    border-top: 1px dashed var(--airio-border, #D8E2EC); padding-top: 0.45rem;
}
.ci-deep-subtitle {
    font-size: 0.74rem; font-weight: 700; color: rgba(10, 31, 51, 0.7);
    margin: 0.45rem 0 0.28rem 0;
}
.ci-deep-label {
    font-size: 0.64rem; text-transform: uppercase; letter-spacing: 0.04em;
    color: rgba(10, 31, 51, 0.55); font-weight: 700; margin-top: 0.45rem;
}
.ci-deep-pattern {
    font-size: 0.96rem; font-weight: 800; color: var(--airio-deep-navy, #0A1F33);
    letter-spacing: 0.01em; margin-top: 0.1rem; overflow-wrap: anywhere;
}
.ci-deep-latest { font-size: 0.92rem; font-weight: 800; color: #B42318; margin-top: 0.1rem; }

/* --- Section 4: AI alert feed --- */
.ci-alert {
    display: flex; align-items: center; gap: 0.75rem;
    background: var(--airio-card, #FFFFFF);
    border: 1px solid var(--airio-border, #D8E2EC);
    border-left: 4px solid var(--alert-color, #C76A12);
    border-radius: 12px; padding: 0.65rem 0.9rem; margin-bottom: 0.5rem;
    box-shadow: 0 6px 14px rgba(10, 31, 51, 0.05);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.ci-alert:hover { transform: translateX(4px); box-shadow: 0 12px 24px rgba(10, 31, 51, 0.11); }
.ci-alert-icon { font-size: 1.18rem; line-height: 1; }
.ci-alert-body { flex: 1; min-width: 0; }
.ci-alert-title { font-size: 0.9rem; font-weight: 700; color: var(--airio-deep-navy, #0A1F33); line-height: 1.3; }
.ci-alert-meta { font-size: 0.74rem; color: rgba(10, 31, 51, 0.6); margin-top: 0.12rem; overflow-wrap: anywhere; }
.ci-alert-score {
    flex: 0 0 auto;
    font-size: 0.76rem; font-weight: 800; padding: 0.22rem 0.6rem; border-radius: 999px;
    background: var(--airio-soft-blue, #EAF1F7); color: var(--airio-primary-navy, #183F5F);
}
</style>
"""


# Risk presentation (shared by cards + alert feed). Aligned to the theme palette.
RISK_STYLE = {
    "High": {"color": "#B42318", "badge_bg": "#FEE2E2", "badge_fg": "#991B1B", "icon": "🚨"},
    "Medium": {"color": "#C76A12", "badge_bg": "#FEF3C7", "badge_fg": "#92400E", "icon": "⚠️"},
}


def build_abnormal_cards(
    abnormal_df: pd.DataFrame,
    facts: pd.DataFrame,
    at_risk_ids: set[str],
    inventory: pd.DataFrame | None = None,
) -> list[dict]:
    """Build one AI card per abnormal order line, sorted by severity (deviation).

    Every flagged line becomes its own card — not aggregated to one-per-customer —
    so all abnormal orders are surfaced. Pure view-layer enrichment: per-line
    revenue comes from ``facts`` (the authoritative line_total), the on-hand stock
    and the product's prior order sequence are pulled in so the AI narrative can
    cite real numbers, and risk_score is scaled relative to the largest deviation
    in the set. No service logic changes.
    """
    if abnormal_df is None or abnormal_df.empty:
        return []

    # Recover per-line revenue + product_id by matching the flagged lines back to
    # the fact frame on (order_nbr, product_name).
    flook = facts.copy()
    if "order_nbr" not in flook.columns:
        flook["order_nbr"] = flook.get("order_id", "")
    flook["order_nbr"] = flook["order_nbr"].astype(str)
    lookup = (
        flook.groupby(["order_nbr", "product_name"], as_index=False)
        .agg(product_id=("product_id", "first"), line_revenue=("revenue", "sum"))
    )

    ab = abnormal_df.copy()
    ab["order_nbr"] = ab["order_nbr"].astype(str)
    ab = ab.merge(lookup, on=["order_nbr", "product_name"], how="left")
    ab["line_revenue"] = pd.to_numeric(ab["line_revenue"], errors="coerce").fillna(0.0)
    ab["product_id"] = ab["product_id"].astype(str)

    # On-hand stock per product (summed across branches), if inventory is available.
    stock_by_product: dict[str, int] = {}
    if inventory is not None and not inventory.empty and "product_id" in inventory.columns:
        inv = inventory.copy()
        inv["product_id"] = inv["product_id"].astype(str)
        inv["_stock"] = pd.to_numeric(inv.get("stock_level"), errors="coerce").fillna(0)
        stock_by_product = inv.groupby("product_id")["_stock"].sum().round().astype(int).to_dict()

    # Each product's full order-quantity sequence in date order (for the history
    # narrative + "typical range").
    series_src = facts.copy()
    if "order_date" in series_src.columns:
        series_src = series_src.sort_values("order_date")
    series_src["product_id"] = series_src["product_id"].astype(str)
    series_src["_qty"] = pd.to_numeric(series_src["quantity"], errors="coerce").fillna(0).round().astype(int)
    qty_series_by_product = series_src.groupby("product_id")["_qty"].apply(list).to_dict()

    # How many abnormal lines each customer has (drives "recurring behaviour" copy).
    cust_counts = ab.groupby("customer_name").size().to_dict()

    global_max_dev = float(ab["deviation_pct"].max()) or 1.0

    cards: list[dict] = []
    for position, (_, row) in enumerate(ab.iterrows()):
        deviation = float(row["deviation_pct"])
        name = str(row["customer_name"])
        product_id = str(row["product_id"])
        current_quantity = int(row["current_quantity"])

        # Historical (non-anomalous) order sequence for this product: drop one
        # occurrence of the flagged quantity so the baseline reflects normal demand.
        full_series = list(qty_series_by_product.get(product_id, []))
        history = full_series.copy()
        if current_quantity in history:
            history.remove(current_quantity)
        if history:
            hist_low, hist_high = int(min(history)), int(max(history))
        else:
            baseline = max(1, math.ceil(float(row["historical_avg"])))
            hist_low = hist_high = baseline

        cards.append({
            "uid": f"abn{position}",
            "customer_name": name,
            "product_name": str(row["product_name"]),
            "product_id": product_id,
            "order_nbr": str(row["order_nbr"]),
            "risk_level": str(row["risk_level"]),
            "risk_score": int(round(min(100.0, deviation / global_max_dev * 100.0))),
            "deviation_pct": deviation,
            "current_quantity": current_quantity,
            "historical_avg": float(row["historical_avg"]),
            "revenue_impact": float(row["line_revenue"]),
            "at_risk": product_id in at_risk_ids,
            "customer_abnormal_lines": int(cust_counts.get(name, 1)),
            "current_inventory": stock_by_product.get(product_id),  # None if unknown
            "hist_series": history[-6:],
            "hist_low": hist_low,
            "hist_high": hist_high,
        })
    cards.sort(key=lambda c: (c["risk_score"], c["deviation_pct"]), reverse=True)
    return cards


def render_executive_summary(cards: list[dict], abnormal_df: pd.DataFrame) -> None:
    """Section 1 — premium AI briefing panel summarising the abnormal-order signal."""
    total_lines = int(len(abnormal_df))
    customers_impacted = int(abnormal_df["customer_name"].nunique()) if not abnormal_df.empty else 0
    products_impacted = int(abnormal_df["product_name"].nunique()) if not abnormal_df.empty else 0
    revenue_exposure = sum(c["revenue_impact"] for c in cards)
    top_customer = cards[0]["customer_name"] if cards else "—"
    if not abnormal_df.empty:
        worst_line = abnormal_df.sort_values("deviation_pct", ascending=False).iloc[0]
        top_product = str(worst_line["product_name"])
    else:
        top_product = "—"

    stats = [
        ("alert", "Abnormal orders detected", f"{total_lines:,}", ""),
        ("", "Customers impacted", f"{customers_impacted:,}", ""),
        ("", "Products impacted", f"{products_impacted:,}", ""),
        ("alert", "Revenue exposure", money(revenue_exposure), ""),
        ("", "Highest-risk customer", compact(top_customer, 22), "name"),
        ("", "Highest-risk product", compact(top_product, 22), "name"),
    ]
    chips = "".join(
        f'<div class="ci-hero-stat {variant}"><div class="k">{escape(label)}</div>'
        f'<div class="v {value_cls}">{escape(value)}</div></div>'
        for variant, label, value, value_cls in stats
    )
    st.markdown(
        '<div class="ci-hero">'
        '<div class="ci-hero-kicker">AI Order Intelligence Center · Live from Oracle</div>'
        '<div class="ci-hero-title">🚨 AI Executive Summary</div>'
        '<div class="ci-hero-sub">Autonomous monitoring of customer ordering behaviour against '
        'per-product demand baselines.</div>'
        f'<div class="ci-hero-grid">{chips}</div>'
        '</div>',
        unsafe_allow_html=True,
    )


def _card_html(card: dict, rank: int) -> str:
    rs = RISK_STYLE[card["risk_level"]]
    explanation = (
        f"{escape(card['customer_name'])} ordered {card['deviation_pct']:.0f}% above historical "
        f"demand for {escape(card['product_name'])}."
    )
    sub = ("This order may create inventory pressure." if card["at_risk"]
           else "Current order quantity significantly exceeds baseline demand.")
    inv_txt = "Pressure detected" if card["at_risk"] else "Stable"
    return (
        f'<div class="ci-ai-card" style="--risk-color:{rs["color"]}">'
        '<div class="ci-ai-head"><div>'
        f'<div class="ci-ai-rank">#{rank} · Risk Score {card["risk_score"]}/100</div>'
        f'<div class="ci-ai-name">{escape(card["customer_name"])}</div></div>'
        f'<div class="ci-risk-badge" style="background:{rs["badge_bg"]};color:{rs["badge_fg"]}">'
        f'{rs["icon"]} {card["risk_level"]} Risk</div></div>'
        f'<div class="ci-prod-line">Product: <b>{escape(card["product_name"])}</b></div>'
        '<div class="ci-ai-metrics">'
        f'<div class="ci-ai-metric"><div class="k">Deviation</div><div class="v">+{card["deviation_pct"]:.0f}%</div></div>'
        f'<div class="ci-ai-metric"><div class="k">Current Qty</div><div class="v">{card["current_quantity"]:,}</div></div>'
        f'<div class="ci-ai-metric"><div class="k">Hist. Avg</div><div class="v">{math.ceil(card["historical_avg"]):,}</div></div>'
        '</div>'
        '<div class="ci-ai-metrics">'
        f'<div class="ci-ai-metric"><div class="k">Revenue Impact</div><div class="v">{money(card["revenue_impact"])}</div></div>'
        f'<div class="ci-ai-metric"><div class="k">Inventory</div><div class="v">{escape(inv_txt)}</div></div>'
        '</div>'
        f'<div class="ci-ai-explain"><span class="tag">🧠 AI Analysis</span>{explanation} {escape(sub)}</div>'
        '</div>'
    )


def _analysis_paragraphs(card: dict) -> list[str]:
    """The AI-analyst narrative — grounded in this customer's real numbers."""
    name = card["customer_name"]
    prod = card["product_name"]
    cur = card["current_quantity"]
    dev = card["deviation_pct"]
    avg = math.ceil(card["historical_avg"])
    lo, hi = card["hist_low"], card["hist_high"]

    paras: list[str] = []
    if lo != hi:
        paras.append(f"{name} typically orders between {lo:,} and {hi:,} units of {prod} per order.")
    else:
        paras.append(f"{name} typically orders around {lo:,} units of {prod} per order.")
    paras.append(
        f"The latest order was placed for {cur:,} units — approximately {dev:.0f}% above the "
        f"historical average demand of {avg:,} units for this product."
    )
    inv = card.get("current_inventory")
    if inv is not None and inv > 0:
        share = cur / inv * 100.0
        paras.append(
            f"This single order would consume about {share:.0f}% of the {inv:,} units currently in "
            f"stock, and is materially higher than {name}'s previous purchasing behaviour."
        )
    else:
        paras.append(
            f"This order is materially higher than {name}'s previous purchasing behaviour for {prod}."
        )
    return paras


def _pattern_hypotheses(card: dict) -> list[str]:
    """Business hypotheses for the spike, presented as 'the pattern may indicate'."""
    return [
        "A new customer contract or project",
        "A planned bulk procurement cycle",
        "Inventory buffering on the customer's side",
        "A temporary demand surge",
    ]


def _closing_inventory_line(card: dict) -> str | None:
    """A grounded closing sentence about stockout pressure, when stock is known."""
    inv = card.get("current_inventory")
    cur = card["current_quantity"]
    if inv is not None and cur:
        if inv < cur:
            return (
                f"Based on current inventory of {inv:,} units, continued demand at this rate may "
                "increase stockout pressure if replenishment actions are not taken."
            )
        if card["at_risk"]:
            return (
                "With this product already at or below its reorder point, sustained demand at this "
                "level may increase stockout pressure if replenishment is not accelerated."
            )
        return (
            f"Current inventory of {inv:,} units can absorb this order, but repeat orders of this "
            "size would draw stock down quickly."
        )
    if card["at_risk"]:
        return (
            "This product is already at or below its reorder point, so sustained demand at this "
            "level may increase stockout pressure."
        )
    return None


def _inventory_impact(card: dict) -> tuple[list[tuple[str, str]] | None, str]:
    """Return (stat rows, risk-assessment sentence) for the inventory impact block."""
    inv = card.get("current_inventory")
    cur = card["current_quantity"]
    if inv is None:
        return None, (
            "Live inventory for this product is unavailable; assess replenishment manually "
            "against the order size."
        )
    coverage = (inv / cur) if cur else 0.0
    rows = [
        ("Current Inventory", f"{inv:,} units"),
        ("Expected Consumption", f"{cur:,} units"),
        ("Inventory Coverage", f"{coverage:.2f}x order size"),
    ]
    if coverage < 1:
        risk = (
            f"Current inventory ({inv:,} units) is below this order's size and may be insufficient "
            "if similar orders continue over the next replenishment cycle."
        )
    elif coverage < 2:
        risk = (
            f"Current inventory covers roughly {coverage:.1f}x this order; a repeat order from "
            f"{card['customer_name']} would draw stock down quickly."
        )
    else:
        risk = (
            f"Current inventory covers about {coverage:.1f}x this order size, so immediate stockout "
            "risk is low."
        )
    return rows, risk


def _order_history(card: dict) -> tuple[str | None, int, str]:
    """Return (historical-pattern string, latest qty, observation sentence)."""
    hist = card.get("hist_series") or []
    cur = card["current_quantity"]
    prod = card["product_name"]
    if not hist:
        return None, cur, f"No prior order history is available for {prod} to compare against this order."
    pattern = " → ".join(f"{q:,}" for q in hist)
    spread = max(hist) - min(hist)
    if spread <= max(3, 0.15 * max(hist)):
        obs = (
            f"Demand for {prod} held steady across recent orders before this sudden increase "
            f"to {cur:,} units."
        )
    else:
        obs = (
            f"Demand for {prod} varied modestly across recent orders before this order broke "
            f"sharply above the range to {cur:,} units."
        )
    return pattern, cur, obs


def _deep_actions(card: dict) -> list[str]:
    """Contextual, customer/product-specific recommended actions."""
    name = card["customer_name"]
    prod = card["product_name"]
    inv = card.get("current_inventory")

    actions = [
        f"Confirm whether this order is associated with a new project or contract for {name}",
        f"Monitor follow-up orders from {name} over the next 7–14 days",
    ]
    if inv is not None and inv < card["current_quantity"]:
        actions.append(f"Consider transferring {prod} inventory from lower-demand locations to cover the shortfall")
    elif card["at_risk"]:
        actions.append(f"Expedite replenishment for {prod}, which is at or below its reorder point")
    else:
        actions.append(f"Consider transferring {prod} inventory from lower-demand locations if demand persists")
    actions.append(f"Increase procurement planning for {prod} if demand continues at this level")
    if card["customer_abnormal_lines"] > 1:
        actions.append(
            f"Review {name}'s broader ordering pattern — {card['customer_abnormal_lines']} of their "
            "order lines are flagged abnormal"
        )
    return actions


def make_trend_chart(facts: pd.DataFrame, card: dict):
    """Mini demand-history sparkline for the card's most abnormal product."""
    series = facts[facts["product_name"] == card["product_name"]]
    qtys: list[int] = []
    if not series.empty:
        ordered = series.sort_values("order_date") if "order_date" in series.columns else series
        qtys = pd.to_numeric(ordered["quantity"], errors="coerce").fillna(0).round().astype(int).tolist()
    qtys = qtys[-11:]

    current = card["current_quantity"]
    colors: list[str] = []
    marked = False
    for q in qtys:
        if not marked and q == current:
            colors.append("#B42318")
            marked = True
        else:
            colors.append("#183F5F")
    if not marked:  # guarantee the anomaly is always visible
        qtys.append(current)
        colors.append("#B42318")
    if not qtys:
        return None

    fig = go.Figure(go.Bar(
        x=list(range(1, len(qtys) + 1)),
        y=qtys,
        marker_color=colors,
        text=qtys,
        textposition="outside",
    ))
    fig.update_traces(textfont_size=10, cliponaxis=False)
    fig.update_layout(
        height=175,
        margin=dict(l=8, r=8, t=20, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


def render_deep_analysis(card: dict, facts: pd.DataFrame) -> None:
    """Section 3 — expandable deep analysis shown inside each card's expander."""
    stats = [
        ("Historical Avg", f"{math.ceil(card['historical_avg']):,}"),
        ("Current Qty", f"{card['current_quantity']:,}"),
        ("Deviation", f"+{card['deviation_pct']:.0f}%"),
        ("Risk Level", card["risk_level"]),
        ("Revenue Impact", money(card["revenue_impact"])),
        ("Inventory", "Pressure detected" if card["at_risk"] else "Stable"),
    ]
    grid = '<div class="ci-deep-grid">' + "".join(
        f'<div class="ci-deep-stat"><div class="k">{escape(k)}</div><div class="v">{escape(v)}</div></div>'
        for k, v in stats
    ) + '</div>'
    st.markdown(grid, unsafe_allow_html=True)

    # -- Order History narrative -------------------------------------------
    pattern, latest, observation = _order_history(card)
    history_html = '<div class="ci-deep-block"><div class="ci-deep-block-title" style="color:#183F5F">📜 Order History</div>'
    if pattern:
        history_html += (
            '<div class="ci-deep-label">Historical Pattern</div>'
            f'<div class="ci-deep-pattern">{escape(pattern)} units</div>'
        )
    history_html += (
        '<div class="ci-deep-label">Latest Order</div>'
        f'<div class="ci-deep-latest">{latest:,} units</div>'
        '<div class="ci-deep-label">Observation</div>'
        f'<div class="ci-deep-text">{escape(observation)}</div>'
        '</div>'
    )
    st.markdown(history_html, unsafe_allow_html=True)

    st.caption(f"📈 Demand history — {compact(card['product_name'], 34)} (anomaly highlighted in red)")
    chart = make_trend_chart(facts, card)
    if chart is not None:
        st.plotly_chart(
            chart,
            use_container_width=True,
            theme="streamlit",
            key=f"trend_{card['uid']}",
        )

    # -- AI Analysis narrative ---------------------------------------------
    analysis_html = '<div class="ci-deep-block"><div class="ci-deep-block-title" style="color:#183F5F">🧠 AI Analysis</div>'
    analysis_html += "".join(f'<div class="ci-deep-text">{escape(p)}</div>' for p in _analysis_paragraphs(card))
    analysis_html += '<div class="ci-deep-subtitle">The pattern may indicate:</div>'
    analysis_html += "".join(f'<div class="ci-deep-bullet">{escape(b)}</div>' for b in _pattern_hypotheses(card))
    closing = _closing_inventory_line(card)
    if closing:
        analysis_html += f'<div class="ci-deep-text closing">{escape(closing)}</div>'
    analysis_html += '</div>'
    st.markdown(analysis_html, unsafe_allow_html=True)

    # -- Inventory Impact --------------------------------------------------
    inv_rows, inv_risk = _inventory_impact(card)
    inv_html = '<div class="ci-deep-block"><div class="ci-deep-block-title" style="color:#183F5F">📦 Inventory Impact</div>'
    if inv_rows:
        inv_html += '<div class="ci-deep-grid">' + "".join(
            f'<div class="ci-deep-stat"><div class="k">{escape(k)}</div><div class="v">{escape(v)}</div></div>'
            for k, v in inv_rows
        ) + '</div>'
    inv_html += (
        '<div class="ci-deep-label">Risk Assessment</div>'
        f'<div class="ci-deep-text">{escape(inv_risk)}</div>'
        '</div>'
    )
    st.markdown(inv_html, unsafe_allow_html=True)

    # -- Recommended Actions -----------------------------------------------
    actions_html = '<div class="ci-deep-block actions"><div class="ci-deep-block-title" style="color:#C76A12">✅ Recommended Actions</div>' + "".join(
        f'<div class="ci-deep-bullet">{escape(a)}</div>' for a in _deep_actions(card)
    ) + '</div>'
    st.markdown(actions_html, unsafe_allow_html=True)


def render_ai_cards(cards: list[dict], facts: pd.DataFrame) -> None:
    """Section 2 — large interactive AI cards, two per row, each with deep analysis."""
    for start in range(0, len(cards), 2):
        row = st.columns(2, gap="large")
        for column, (offset, card) in zip(row, enumerate(cards[start:start + 2])):
            with column:
                st.markdown(_card_html(card, start + offset + 1), unsafe_allow_html=True)
                with st.expander("🔬 Expand Analysis"):
                    render_deep_analysis(card, facts)


def _alert_html(card: dict) -> str:
    rs = RISK_STYLE[card["risk_level"]]
    if card["customer_abnormal_lines"] > 1:
        title = (
            f"{escape(card['customer_name'])} showing recurring abnormal ordering behaviour "
            f"({card['customer_abnormal_lines']} lines) — latest +{card['deviation_pct']:.0f}%"
        )
    else:
        title = f"{escape(card['customer_name'])} ordered {card['deviation_pct']:.0f}% above normal demand"
    meta = (
        f"{escape(card['product_name'])} · {card['current_quantity']:,} units vs "
        f"{math.ceil(card['historical_avg']):,} avg"
    )
    return (
        f'<div class="ci-alert" style="--alert-color:{rs["color"]}">'
        f'<div class="ci-alert-icon">{rs["icon"]}</div>'
        f'<div class="ci-alert-body"><div class="ci-alert-title">{title}</div>'
        f'<div class="ci-alert-meta">{meta}</div></div>'
        f'<div class="ci-alert-score">Score {card["risk_score"]}</div>'
        '</div>'
    )


def render_alert_feed(cards: list[dict]) -> None:
    """Section 4 — vertical alert feed sorted by risk score (descending)."""
    st.markdown("".join(_alert_html(card) for card in cards), unsafe_allow_html=True)


def render_detailed_abnormal_table(abnormal_df: pd.DataFrame) -> None:
    """Section 5 — every flagged order line as a native Streamlit dataframe."""
    if abnormal_df.empty:
        st.success("No abnormal order quantities detected against per-product baselines.")
        return
    display_abnormal = abnormal_df.rename(
        columns={
            "customer_name": "Customer",
            "product_name": "Product",
            "order_nbr": "Order",
            "order_date": "Date",
            "historical_avg": "Historical Avg",
            "current_quantity": "Current Qty",
            "deviation_pct": "Deviation %",
            "risk_level": "Risk Level",
        }
    )
    st.dataframe(
        display_abnormal,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Historical Avg": st.column_config.NumberColumn(format="%.1f"),
            "Current Qty": st.column_config.NumberColumn(format="%d"),
            "Deviation %": st.column_config.NumberColumn(format="+%.0f%%"),
        },
    )


def render_ai_order_intelligence_center(
    abnormal_cards: list[dict],
    abnormal_df: pd.DataFrame,
    facts: pd.DataFrame,
) -> None:
    """The full premium AI section: summary -> cards -> deep analysis -> feed -> table.

    Rendered as one self-contained block placed directly below Customer Spotlight.
    """
    st.subheader("🚨 AI Order Intelligence Center")
    st.caption(
        "An AI analyst continuously monitoring customer ordering behaviour against "
        "per-product demand baselines, explaining every abnormal order in business language."
    )

    # Section 1 — AI Executive Summary
    render_executive_summary(abnormal_cards, abnormal_df)

    if not abnormal_cards:
        st.success(
            "✅ All clear — no order lines exceed their per-product demand baseline. "
            "The AI monitor will surface anomalies here as soon as they appear."
        )
        return

    # Section 2 + 3 — AI Customer Intelligence cards & expandable deep analysis
    st.markdown("#### 🤖 AI Customer Intelligence")
    st.caption(
        "One AI-analysed card for every abnormal order, ranked by severity. "
        "Expand any card for the full demand-history breakdown and recommended actions."
    )
    render_ai_cards(abnormal_cards, facts)

    # Section 5 — Detailed abnormal order table (Streamlit dataframe)
    st.markdown("#### 📋 Abnormal Order Detail")
    st.caption("Every flagged order line behind the cards above.")
    render_detailed_abnormal_table(abnormal_df)


# ---------------------------------------------------------------------------
# Page body
# ---------------------------------------------------------------------------
apply_page_style()
st.markdown(SPOTLIGHT_CSS, unsafe_allow_html=True)
st.markdown(KPI_CSS, unsafe_allow_html=True)
st.markdown(HERO_CSS, unsafe_allow_html=True)

render_page_header(
    "👥 Customer Intelligence",
    "Real customer demand patterns, abnormal ordering, inventory pressure, and "
    "growth signals — sourced live from Oracle (BZ_MOCK_CUSTOMER → ORDER_HEADER → ORDER_LINE).",
)

try:
    data = load_customer_intelligence_data()
except Exception as error:
    st.error("Could not load customer/order data from Oracle.")
    st.exception(error)
    st.stop()

customers = data["customers"]
orders = data["orders"]
order_lines = data["order_lines"]
products = data["products"]
inventory = data["inventory"]

facts = cis.prepare_customer_orders(order_lines, orders, customers, products)

if facts.empty:
    st.warning(
        "No customer order activity is available from Oracle "
        "(BZ_MOCK_ORDER_HEADER / BZ_MOCK_ORDER_LINE returned no rows)."
    )
    st.stop()

order_count = int(facts["order_id"].nunique())
buyer_count = int(facts["customer_id"].nunique())
date_min = facts["order_date"].min()
date_max = facts["order_date"].max()
window_caption = ""
if pd.notna(date_min) and pd.notna(date_max):
    window_caption = (
        f"Order window: {date_min:%d %b %Y} → {date_max:%d %b %Y} · "
        f"{order_count:,} orders · {buyer_count} active buyers · {len(customers)} customers on file"
    )
st.caption(window_caption)

kpis = cis.executive_kpis(facts, customers, orders, inventory)

# Supporting datasets, computed once and reused by both the KPI hover detail and
# their dedicated sections below (no recalculation — identical service outputs).
top_customers_df = cis.top_customers(facts, limit=10)
abnormal_df = cis.detect_abnormal_orders(facts)
trends_df = cis.customer_demand_trends(facts)
impact_df = cis.inventory_impact(facts, inventory)
dormant_df = cis.dormant_accounts(customers, orders)

# Reuse the existing reorder-point logic to flag at-risk products for the cards.
at_risk_ids = cis.at_risk_products(inventory)
abnormal_cards = build_abnormal_cards(abnormal_df, facts, at_risk_ids, inventory)

# -- Executive KPIs --------------------------------------------------------
st.subheader("Executive KPIs")
st.caption("Hover (or tap) a card to flip it and reveal the supporting metrics.")
render_executive_kpis(kpis, top_customers_df, trends_df, abnormal_df, impact_df, dormant_df)

# -- Customer Spotlight -----------------------------------------------------
st.subheader("⭐ Customer Spotlight")
st.caption("Top 5 customers by order revenue.")
render_spotlight(cis.customer_spotlight(facts, limit=5))

st.divider()

# ==========================================================================
# AI Order Intelligence Center — new premium section (directly below Spotlight)
# ==========================================================================
render_ai_order_intelligence_center(abnormal_cards, abnormal_df, facts)

st.divider()

# -- Top Products ----------------------------------------------------------
st.subheader("Top Products")
products_by_revenue = cis.top_products(facts, metric="revenue", limit=10)
products_by_quantity = cis.top_products(facts, metric="quantity", limit=10)

product_left, product_right = st.columns(2, gap="large")
with product_left:
    render_chart_card(
        "Top 10 Products by Revenue",
        "Revenue uses the authoritative order-line total (LINE_TOTAL_AMT).",
        make_top_products_chart(products_by_revenue, "revenue", "Revenue"),
        "No product revenue data is available.",
    )
with product_right:
    render_chart_card(
        "Top 10 Products by Quantity",
        "Units ordered across all customers.",
        make_top_products_chart(products_by_quantity, "quantity", "Units"),
        "No product quantity data is available.",
    )

st.divider()

# -- Top Customers ---------------------------------------------------------
st.subheader("Top Customers")
render_table(
    top_customers_df.rename(
        columns={
            "customer_name": "Customer",
            "customer_segment": "Segment",
            "contract_tier": "Tier",
            "orders": "Orders",
            "units": "Units",
            "revenue": "Revenue",
            "contribution_pct": "Contribution %",
        }
    )[["Customer", "Segment", "Tier", "Orders", "Units", "Revenue", "Contribution %"]],
    formatters={
        "Revenue": lambda value: f"${float(value):,.2f}",
        "Contribution %": lambda value: f"{float(value):.1f}%",
    },
)

st.divider()

# -- Customer Demand Trends ------------------------------------------------
st.subheader("Customer Demand Trends")
st.caption("Based on available order history (first half vs second half of the order window).")
if trends_df.empty:
    st.info("Not enough dated order history to compute demand trends.")
else:
    growing = trends_df[trends_df["trend"] == "Growing"]
    stable = trends_df[trends_df["trend"] == "Stable"]
    declining = trends_df[trends_df["trend"] == "Declining"]
    trend_cols = st.columns(3, gap="medium")
    for column, (label, frame) in zip(
        trend_cols,
        [("📈 Growing", growing), ("➡️ Stable", stable), ("📉 Declining", declining)],
    ):
        with column:
            with st.container(border=True):
                st.metric(label, f"{len(frame)}")
                for _, row in frame.iterrows():
                    st.caption(f"{compact(row['customer_name'])} ({row['change_pct']:+.0f}%)")
                if frame.empty:
                    st.caption("—")

st.divider()

# -- Inventory Impact Analysis ---------------------------------------------
st.subheader("Inventory Impact Analysis")
st.caption("Customers driving inventory pressure by ordering products at/below their reorder point.")
if impact_df.empty:
    st.info("No customers are currently ordering products that are at stockout risk.")
else:
    impact_left, impact_right = st.columns([1.1, 1], gap="large")
    with impact_left:
        render_chart_card(
            "Inventory Pressure by Customer",
            "Relative score (0–100) based on units ordered on at-risk products.",
            make_impact_chart(impact_df),
            "No inventory pressure to display.",
        )
    with impact_right:
        render_table(
            impact_df.rename(
                columns={
                    "customer_name": "Customer",
                    "at_risk_units": "At-Risk Units",
                    "at_risk_revenue": "At-Risk Revenue",
                    "affected_products": "Products",
                    "risk_score": "Risk Score",
                }
            )[["Customer", "At-Risk Units", "At-Risk Revenue", "Products", "Risk Score"]],
            max_height=420,
            formatters={
                "At-Risk Revenue": lambda value: f"${float(value):,.0f}",
                "Risk Score": lambda value: f"{int(round(float(value)))}",
            },
        )

st.divider()

# -- Dormant Accounts ------------------------------------------------------
st.subheader("Dormant Accounts")
st.caption("Active customers with zero orders — a re-engagement opportunity, not an error.")
if dormant_df.empty:
    st.success("Every active customer has placed at least one order.")
else:
    render_table(
        dormant_df.rename(
            columns={
                "customer_name": "Customer",
                "customer_segment": "Segment",
                "contract_tier": "Tier",
                "industry": "Industry",
                "city": "City",
                "credit_limit": "Credit Limit",
                "signup_date": "Signed Up",
            }
        ),
        formatters={
            "Credit Limit": lambda value: f"${float(value):,.0f}",
        },
    )

st.divider()

# -- AI Insights -----------------------------------------------------------
st.subheader("AI Insights")
render_ai_insight_panel(
    list(cis.generate_customer_insights(facts, customers, orders, inventory)),
    title="Customer Intelligence Insights",
    icon="👥",
)

with st.expander("Customer order line records"):
    record_cols = [
        c for c in ["order_nbr", "order_date", "customer_name", "customer_segment",
                    "product_name", "category", "quantity", "revenue", "order_status"]
        if c in facts.columns
    ]
    render_table(facts[record_cols], max_height=420)
