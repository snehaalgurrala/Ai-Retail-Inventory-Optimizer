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
from backend.services import customer_order_limits as col  # noqa: E402
from backend.services import inventory_scope  # noqa: E402
from backend.services.abnormal_order_intelligence import (  # noqa: E402
    RISK_ASSESSMENT_STYLE,
    RISK_DISPLAY_LABEL,
    _RISK_PRIORITY,
    _customer_behaviour_assessment,
    _deep_actions,
    _demand_trend,
    _ensure_assessment,
    _executive_summary,
    _inventory_impact_narrative,
    _product_demand_context,
    _what_happened,
    build_abnormal_cards,
)
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
.ci-threshold-pill {
    display: inline-flex; align-items: center; gap: 0.4rem;
    margin-top: 1.85rem;
    padding: 0.5rem 0.95rem;
    border-radius: 999px;
    background: rgba(108, 179, 63, 0.14);
    border: 1px solid rgba(108, 179, 63, 0.35);
    color: var(--airio-primary-navy, #183F5F);
    font-size: 0.9rem; font-weight: 700;
}
.ci-threshold-pill b { color: var(--airio-deep-navy, #0A1F33); font-weight: 800; }
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


def render_order_quantity_limits(customers: pd.DataFrame) -> None:
    """Editable per-customer order quantity limits (persisted to JSON, not Oracle).

    The Order Simulator caps each product line in a customer's cart at this value;
    the default is ``DEFAULT_LIMIT`` units for any customer without a saved limit.
    Saving writes a small config file and mirrors the values into session_state so
    both pages read identical limits — the BZ_MOCK_* schema is never touched.
    """
    st.subheader("🛒 Order Quantity Limits")
    st.caption(
        f"Default is {col.DEFAULT_LIMIT} units per product per customer — adjust any "
        "customer below. Only customers who can log into the Order Simulator will "
        "actually hit this limit."
    )
    if customers.empty or "customer_id" not in customers.columns:
        st.info("No customers are available to configure.")
        return

    rows = (
        customers.sort_values("customer_name")
        if "customer_name" in customers.columns
        else customers
    )
    grid = pd.DataFrame(
        {
            "Customer ID": rows["customer_id"].astype(str).tolist(),
            "Customer Name": rows.get(
                "customer_name", pd.Series(dtype=str)
            ).astype(str).tolist(),
            "Segment": rows.get("customer_segment", pd.Series(dtype=str))
            .fillna("—")
            .astype(str)
            .tolist(),
            "Limit (units)": [col.get_limit(cid) for cid in rows["customer_id"]],
        }
    )

    edited = st.data_editor(
        grid,
        key="ci_order_limits_editor",
        hide_index=True,
        use_container_width=True,
        disabled=["Customer ID", "Customer Name", "Segment"],
        column_config={
            "Limit (units)": st.column_config.NumberColumn(
                "Limit (units)",
                min_value=1,
                step=1,
                format="%d",
                help="Maximum units of any single product this customer may add to their cart.",
            ),
        },
    )

    if st.button("💾 Save limits", key="ci_save_order_limits"):
        updated: dict[int, int] = {}
        for _, record in edited.iterrows():
            try:
                cid = int(record["Customer ID"])
                lim = int(record["Limit (units)"])
            except (TypeError, ValueError):
                continue
            updated[cid] = lim if lim >= 1 else 1
        col.save_limits(updated)
        # Mirror into session so the simulator (same Streamlit session) reads the
        # freshly saved values immediately.
        st.session_state[col.SESSION_KEY] = updated
        st.success("Order quantity limits saved.")
        st.toast("Order quantity limits saved", icon="✅")


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
    threshold_pct: float = 50.0,
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

    # 3) Demand Opportunities Identified
    if not abnormal_df.empty:
        high = int((abnormal_df["risk_level"] == "High").sum())
        medium = int((abnormal_df["risk_level"] == "Medium").sum())
        worst = abnormal_df.iloc[0]
        abnormal_lines = [
            ("Opportunities", f"{len(abnormal_df)}"),
            ("Threshold", f"≥ +{int(threshold_pct)}% over baseline"),
            ("Intensity", f"{high} High Demand · {medium} Moderate Demand"),
            ("Largest", f"{worst['customer_name']} (+{worst['deviation_pct']:.0f}%)"),
            ("Product", f"{worst['product_name']} — {int(worst['current_quantity'])} vs {worst['historical_avg']:.0f} avg"),
        ]
    else:
        abnormal_lines = [
            ("", f"No orders exceed the +{int(threshold_pct)}% demand threshold."),
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
        _kpi_card("Demand Opportunities Identified", f"{kpis['abnormal_orders']:,}", "num",
                  f"Orders ≥ +{int(threshold_pct)}% over baseline", "Demand Opportunities — detail", abnormal_lines),
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
/* Subtle pulse/glow on the latest abnormal bar — scoped to the demand-history
   charts only (keyed "trend_*") so other Plotly charts are unaffected. The final
   bar in the trace is always the latest order. */
@keyframes ciLatestPulse {
    0%, 100% { filter: drop-shadow(0 0 0px rgba(180, 35, 24, 0.0)); }
    50% { filter: drop-shadow(0 0 7px rgba(180, 35, 24, 0.92)); }
}
[class*="st-key-trend_"] g.points > g.point:last-of-type path {
    animation: ciLatestPulse 1.7s ease-in-out infinite;
}

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

/* --- Executive investigation panels (full-width abnormal-order cards) --- */
.ci-risk-summary {
    display: flex; flex-wrap: wrap; gap: 0.6rem; margin: 0.2rem 0 0.9rem 0;
}
.ci-risk-chip {
    flex: 1 1 0; min-width: 140px;
    display: flex; align-items: center; gap: 0.6rem;
    border: 1px solid var(--airio-border, #D8E2EC);
    border-left: 5px solid var(--chip-color, #183F5F);
    border-radius: 12px; padding: 0.55rem 0.85rem;
    background: var(--airio-card, #FFFFFF);
    box-shadow: 0 6px 14px rgba(10, 31, 51, 0.05);
}
.ci-risk-chip .dot { font-size: 1.05rem; line-height: 1; }
.ci-risk-chip .body { display: flex; flex-direction: column; line-height: 1.1; }
.ci-risk-chip .k {
    font-size: 0.64rem; text-transform: uppercase; letter-spacing: 0.04em;
    font-weight: 700; color: rgba(10, 31, 51, 0.6);
}
.ci-risk-chip .v { font-size: 1.32rem; font-weight: 800; color: var(--chip-color, #183F5F); }

.ci-exec-card {
    position: relative; overflow: hidden; width: 100%;
    border-radius: 18px;
    border: 1px solid var(--airio-border, #D8E2EC);
    border-left: 7px solid var(--risk-color, #183F5F);
    background: linear-gradient(180deg, #FFFFFF 0%, #FBFDFE 100%);
    box-shadow: 0 12px 30px rgba(10, 31, 51, 0.09);
    padding: 1.15rem 1.4rem 1.25rem 1.4rem;
    transition: transform 0.16s ease, box-shadow 0.16s ease;
}
.ci-exec-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 26px 50px rgba(10, 31, 51, 0.15);
}
.ci-exec-head {
    display: flex; justify-content: space-between; align-items: flex-start;
    gap: 1.2rem; flex-wrap: wrap;
}
.ci-exec-headl { min-width: 0; flex: 1 1 60%; }
.ci-exec-rank {
    font-size: 0.7rem; font-weight: 800; text-transform: uppercase;
    letter-spacing: 0.05em; color: rgba(10, 31, 51, 0.5);
}
.ci-exec-name {
    font-size: 1.5rem; font-weight: 800; color: var(--airio-deep-navy, #0A1F33);
    line-height: 1.15; margin-top: 0.15rem; overflow-wrap: anywhere;
}
.ci-exec-prod {
    font-size: 0.95rem; color: rgba(10, 31, 51, 0.75);
    margin-top: 0.28rem; overflow-wrap: anywhere;
}
.ci-exec-prod b { color: var(--risk-color, #183F5F); font-weight: 800; }
.ci-exec-headr {
    flex: 0 0 auto; display: flex; flex-direction: column;
    align-items: flex-end; gap: 0.35rem; text-align: right;
}
@media (max-width: 900px) {
    .ci-exec-name { font-size: 1.28rem; }
}

/* Executive Summary — the plain-English lead on every abnormal-order card. */
.ci-exec-summary {
    margin-top: 1.05rem;
    background: linear-gradient(180deg, #FFFFFF 0%, #F6FAFE 100%);
    border: 1px solid var(--airio-border, #D8E2EC);
    border-left: 4px solid var(--risk-color, #183F5F);
    border-radius: 12px;
    padding: 0.9rem 1.05rem;
}
.ci-exec-summary-tag {
    font-size: 0.66rem; font-weight: 800; text-transform: uppercase;
    letter-spacing: 0.06em; color: var(--risk-color, #183F5F); margin-bottom: 0.38rem;
}
.ci-exec-summary-text {
    font-size: 1.02rem; line-height: 1.6; color: var(--airio-deep-navy, #0A1F33);
}

/* AI Investigation Report — narrative briefing header inside the expander. */
.ci-report-head {
    font-size: 1.14rem; font-weight: 800; color: var(--airio-deep-navy, #0A1F33);
    letter-spacing: 0.01em; margin: 0.1rem 0 0.75rem 0;
    padding-bottom: 0.45rem; border-bottom: 2px solid var(--airio-border, #D8E2EC);
}
.ci-report-lead {
    font-size: 0.82rem; line-height: 1.5; color: rgba(10, 31, 51, 0.62);
    margin: -0.45rem 0 0.85rem 0;
}

/* Technical Details — collapsed numeric drill-down for advanced users. */
.ci-tech {
    border: 1px solid var(--airio-border, #D8E2EC);
    border-radius: 12px; background: var(--airio-card, #FFFFFF);
    margin-top: 0.5rem; padding: 0 0.95rem;
}
.ci-tech > summary {
    cursor: pointer; list-style: none; padding: 0.7rem 0;
    font-weight: 800; font-size: 0.8rem; letter-spacing: 0.04em;
    text-transform: uppercase; color: rgba(10, 31, 51, 0.7);
}
.ci-tech > summary::-webkit-details-marker { display: none; }
.ci-tech > summary::before { content: "▸ "; color: rgba(10, 31, 51, 0.45); }
.ci-tech[open] > summary::before { content: "▾ "; }
.ci-tech-body { padding: 0.1rem 0 0.95rem 0; }
.ci-tech-note {
    margin-top: 0.55rem; font-size: 0.76rem; line-height: 1.5;
    color: rgba(10, 31, 51, 0.58);
}
</style>
"""


# Risk presentation (shared by cards + alert feed). Aligned to the theme palette.
RISK_STYLE = {
    "High": {"color": "#B42318", "badge_bg": "#FEE2E2", "badge_fg": "#991B1B", "icon": "🚨"},
    "Medium": {"color": "#C76A12", "badge_bg": "#FEF3C7", "badge_fg": "#92400E", "icon": "⚠️"},
}


# build_abnormal_cards now lives in backend.services.abnormal_order_intelligence
# (imported at the top) so the dashboard Abnormal Order Intelligence Report and
# this page build identical cards from a single source of truth.


def render_executive_summary(
    cards: list[dict], abnormal_df: pd.DataFrame, threshold_pct: float = 50.0
) -> None:
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
        ("alert", "Demand opportunities identified", f"{total_lines:,}", ""),
        ("", "Customers involved", f"{customers_impacted:,}", ""),
        ("", "Products involved", f"{products_impacted:,}", ""),
        ("alert", "Revenue opportunity", money(revenue_exposure), ""),
        ("", "Top demand customer", compact(top_customer, 22), "name"),
        ("", "Top demand product", compact(top_product, 22), "name"),
    ]
    chips = "".join(
        f'<div class="ci-hero-stat {variant}"><div class="k">{escape(label)}</div>'
        f'<div class="v {value_cls}">{escape(value)}</div></div>'
        for variant, label, value, value_cls in stats
    )
    st.markdown(
        '<div class="ci-hero">'
        '<div class="ci-hero-kicker">Customer Demand Intelligence · Live from Oracle</div>'
        '<div class="ci-hero-title">📈 AI Demand Summary</div>'
        '<div class="ci-hero-sub">AI-powered analysis of customer demand patterns against '
        f'per-product demand baselines · Demand Sensitivity Threshold: {int(threshold_pct)}%</div>'
        f'<div class="ci-hero-grid">{chips}</div>'
        '</div>',
        unsafe_allow_html=True,
    )


def _card_html(card: dict, rank: int) -> str:
    """Full-width executive investigation panel for a single abnormal order.

    Business-first: the card leads with a plain-English Executive Summary and the
    Risk Level badge. All numeric KPIs, scores and formulas live in the collapsed
    Technical Details section inside the expander, not on the card face.
    """
    ra = _ensure_assessment(card)
    band = ra["band"]
    style = RISK_ASSESSMENT_STYLE[band]
    summary = _executive_summary(card)
    return (
        f'<div class="ci-exec-card" style="--risk-color:{style["color"]}">'
        '<div class="ci-exec-head">'
        '<div class="ci-exec-headl">'
        f'<div class="ci-exec-rank">#{rank} · Demand Opportunity Review</div>'
        f'<div class="ci-exec-name">{escape(card["customer_name"])}</div>'
        f'<div class="ci-exec-prod">Product: <b>{escape(card["product_name"])}</b></div>'
        '</div>'
        '<div class="ci-exec-headr">'
        f'<div class="ci-risk-badge" style="background:{style["bg"]};color:{style["color"]}">'
        f'{style["icon"]} {RISK_DISPLAY_LABEL[band]}</div>'
        '</div>'
        '</div>'
        '<div class="ci-exec-summary">'
        '<div class="ci-exec-summary-tag">🧠 Executive Summary</div>'
        f'<div class="ci-exec-summary-text">{escape(summary)}</div>'
        '</div>'
        '</div>'
    )


def _technical_details_html(card: dict) -> str:
    """Collapsed numeric drill-down (scores, ratios, formula) for advanced users."""
    ra = _ensure_assessment(card)
    inv = card.get("current_inventory")
    cur = int(card["current_quantity"])
    avg = math.ceil(card["historical_avg"])
    lo, hi = int(card["hist_low"]), int(card["hist_high"])
    dev = float(card["deviation_pct"])
    impact = ra["impact_pct"]
    impact_str = f"{impact:.0f}% of on-hand stock" if impact is not None else "N/A — stock unavailable"
    coverage = f"{inv / cur:.2f}× order size" if (inv and cur) else "N/A"
    inv_str = f"{inv:,} units" if inv is not None else "Unavailable"
    hist_range = f"{lo:,} – {hi:,} units" if lo != hi else f"{lo:,} units"

    rows = [
        ("Demand Score", f"{ra['score']} / 100"),
        ("Demand Level", RISK_DISPLAY_LABEL[ra["band"]]),
        ("Deviation from Avg", f"+{dev:.0f}%"),
        ("Historical Avg", f"{avg:,} units"),
        ("Historical Max", f"{hi:,} units"),
        ("Historical Range", hist_range),
        ("Latest Order Qty", f"{cur:,} units"),
        ("Current Inventory", inv_str),
        ("Inventory Impact %", impact_str),
        ("Coverage Ratio", coverage),
        ("Revenue Impact", money(card["revenue_impact"])),
        ("Stockout Risk", ra["stockout_risk"]),
    ]
    grid = '<div class="ci-deep-grid">' + "".join(
        f'<div class="ci-deep-stat"><div class="k">{escape(k)}</div><div class="v">{escape(v)}</div></div>'
        for k, v in rows
    ) + '</div>'
    note = (
        '<div class="ci-tech-note">Demand score blends deviation from average (30%), '
        'increase above the historical maximum (25%), inventory impact (20%), '
        'reorder-point pressure (15%) and recent demand trend (10%), normalised to 0–100. '
        'Levels: 0–30 Normal Demand Activity · 31–60 Moderate Demand Activity · '
        '61–85 High Demand Activity · 86–100 Significant Opportunity.</div>'
    )
    return (
        '<details class="ci-tech">'
        '<summary>⚙️ Technical Details</summary>'
        f'<div class="ci-tech-body">{grid}{note}</div>'
        '</details>'
    )


# The abnormal-order narrative generators (_demand_trend, _what_happened,
# _inventory_impact_narrative, _product_demand_context,
# _customer_behaviour_assessment, _deep_actions) now live in
# backend.services.abnormal_order_intelligence and are imported at the top, so the
# page and the dashboard email report tell an identical story.


CHART_BLUE = "#183F5F"   # historical orders
CHART_GREEN = "#6CB33F"  # latest order, normal
CHART_RED = "#B42318"    # latest order, abnormal


def make_trend_chart(card: dict):
    """Complete order-history chart: every order for the product, in real sequence.

    Plots the product's full chronological order sequence (``order_series``) — one
    bar per actual order, never an average or aggregate — and highlights the
    evaluated/abnormal order in red at its true position (``current_index``). All
    other orders use Bunzl blue, so the spike that triggered the anomaly is
    immediately visible against genuine ordering behaviour. Falls back to
    ``hist_series + [current_quantity]`` only if the full sequence is unavailable.
    """
    series = [int(q) for q in (card.get("order_series") or [])]
    cur = int(card["current_quantity"])
    if series:
        idx = int(card.get("current_index", len(series) - 1))
        idx = max(0, min(idx, len(series) - 1))
    else:
        # Degraded mode (older service frame): prior orders then the evaluated one.
        series = [int(q) for q in (card.get("hist_series") or [])] + [cur]
        idx = len(series) - 1
    if not series:
        return None

    n = len(series)
    eval_color = CHART_RED if bool(card.get("risk_level")) else CHART_GREEN
    colors = [eval_color if i == idx else CHART_BLUE for i in range(n)]
    line_widths = [2 if i == idx else 0 for i in range(n)]
    line_colors = [eval_color if i == idx else "rgba(0,0,0,0)" for i in range(n)]

    fig = go.Figure(go.Bar(
        x=list(range(1, n + 1)),
        y=series,
        marker_color=colors,
        marker_line_color=line_colors,
        marker_line_width=line_widths,
        text=[f"{q:,}" for q in series],
        textposition="outside",
        width=0.64,
        hovertemplate="%{y:,} units<extra></extra>",
    ))
    fig.update_traces(textfont_size=10, cliponaxis=False)
    # Callout floating above the highlighted (evaluated) bar, wherever it falls.
    fig.add_annotation(
        x=idx + 1,
        y=series[idx],
        text="High Demand",
        showarrow=False,
        yshift=26,
        font=dict(size=10, color=eval_color),
        xanchor="center",
    )
    headroom = max(series) * 1.3 if max(series) else 1
    # Order-position labels (Order 1 … Order n); the evaluated one reads "Latest"
    # since it is always the most recent order, pinned at the far right.
    # Shown when the sequence is short enough to stay legible; hidden otherwise.
    if n <= 14:
        order_labels = [("Latest" if i == idx else f"Order {i + 1}") for i in range(n)]
        xaxis = dict(
            tickmode="array",
            tickvals=list(range(1, n + 1)),
            ticktext=order_labels,
            tickfont=dict(size=9, color="rgba(10,31,51,0.55)"),
            showgrid=False, zeroline=False, showline=False,
        )
        bottom_margin = 26
    else:
        xaxis = dict(visible=False)
        bottom_margin = 8
    fig.update_layout(
        height=190 + (bottom_margin - 8),
        margin=dict(l=8, r=8, t=36, b=bottom_margin),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        bargap=0.28,
        xaxis=xaxis,
        yaxis=dict(visible=False, range=[0, headroom]),
    )
    return fig


# -- Risk Assessment (enhancement layer) -----------------------------------
# The composite 0-100 risk score and 4-band verdict (RISK_ASSESSMENT_STYLE,
# _RISK_PRIORITY, _ensure_assessment, _risk_assessment) now live in
# backend.services.abnormal_order_intelligence and are imported at the top. That
# module is the single source of truth shared with the dashboard Abnormal Order
# Intelligence Report, so risk bands and scores match exactly across surfaces.


def _narrative_block(
    title: str,
    color: str,
    paras: list[str],
    subtitle: str | None = None,
    bullets: list[str] | None = None,
    actions: bool = False,
) -> str:
    """Render one section of the AI Investigation Report as a styled narrative block."""
    cls = "ci-deep-block actions" if actions else "ci-deep-block"
    html = f'<div class="{cls}"><div class="ci-deep-block-title" style="color:{color}">{title}</div>'
    html += "".join(f'<div class="ci-deep-text">{escape(p)}</div>' for p in paras)
    if subtitle:
        html += f'<div class="ci-deep-subtitle">{escape(subtitle)}</div>'
    if bullets:
        html += "".join(f'<div class="ci-deep-bullet">{escape(b)}</div>' for b in bullets)
    html += '</div>'
    return html


def render_deep_analysis(card: dict, facts: pd.DataFrame) -> None:
    """AI Investigation Report — a narrative supply-chain briefing for one order.

    The AI interprets the real calculations and tells the story behind the anomaly:
    what happened, the inventory impact, the product's demand context, a hedged read
    on customer behaviour, and a business recommendation. Every number, score and
    formula is tucked into the collapsed Technical Details block at the end for
    advanced users — the briefing face carries no KPI cards.
    """
    NAVY, AMBER = "#183F5F", "#C76A12"

    # -- Report header -----------------------------------------------------
    st.markdown('<div class="ci-report-head">🧠 Customer Demand Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="ci-report-lead">AI-generated briefing interpreting this order against '
        'real demand baselines and live inventory — prepared as a supply-chain analyst would.</div>',
        unsafe_allow_html=True,
    )

    # -- Historical order graph (kept exactly as before) -------------------
    st.caption(f"📈 Complete order history — {compact(card['product_name'], 34)} (latest high-demand order in red)")
    chart = make_trend_chart(card)
    if chart is not None:
        st.plotly_chart(
            chart,
            use_container_width=True,
            theme="streamlit",
            key=f"trend_{card['uid']}",
        )

    # -- What Happened? ----------------------------------------------------
    st.markdown(
        _narrative_block("📌 What Happened?", NAVY, _what_happened(card)),
        unsafe_allow_html=True,
    )

    # -- Inventory Impact --------------------------------------------------
    st.markdown(
        _narrative_block("📦 Inventory Impact", NAVY, _inventory_impact_narrative(card)),
        unsafe_allow_html=True,
    )

    # -- Product Demand Context --------------------------------------------
    st.markdown(
        _narrative_block("📈 Product Demand Context", NAVY, _product_demand_context(card)),
        unsafe_allow_html=True,
    )

    # -- Customer Behaviour Assessment (hedged hypotheses) -----------------
    behaviour_paras, hypotheses = _customer_behaviour_assessment(card)
    st.markdown(
        _narrative_block(
            "🏢 Customer Behaviour Assessment", NAVY, behaviour_paras,
            subtitle="This behaviour may indicate:", bullets=hypotheses,
        ),
        unsafe_allow_html=True,
    )

    # -- Business Recommendation -------------------------------------------
    st.markdown(
        _narrative_block(
            "🎯 Business Recommendation", AMBER, [], bullets=_deep_actions(card), actions=True,
        ),
        unsafe_allow_html=True,
    )

    # -- Technical Details (collapsed numbers/scores/formula) --------------
    st.markdown(_technical_details_html(card), unsafe_allow_html=True)


def render_ai_cards(cards: list[dict], facts: pd.DataFrame) -> None:
    """Section 2 — full-width executive panels, one per row, each with deep analysis.

    Cards arrive pre-sorted by risk priority (Critical → High → Medium → Low, then
    by composite risk score) so the most important abnormal orders read first.
    """
    for rank, card in enumerate(cards, start=1):
        st.markdown(_card_html(card, rank), unsafe_allow_html=True)
        with st.expander("🧠 Customer Demand Analysis — What Happened, Inventory Impact, Demand Context, Customer Behaviour & Recommendation"):
            render_deep_analysis(card, facts)
        st.markdown('<div style="height:0.7rem"></div>', unsafe_allow_html=True)


def _alert_html(card: dict) -> str:
    rs = RISK_STYLE[card["risk_level"]]
    if card["customer_abnormal_lines"] > 1:
        title = (
            f"{escape(card['customer_name'])} showing sustained high-demand ordering activity "
            f"({card['customer_abnormal_lines']} lines) — latest +{card['deviation_pct']:.0f}%"
        )
    else:
        title = f"{escape(card['customer_name'])} ordered {card['deviation_pct']:.0f}% above typical demand"
    meta = (
        f"{escape(card['product_name'])} · {card['current_quantity']:,} units vs "
        f"{math.ceil(card['historical_avg']):,} avg"
    )
    return (
        f'<div class="ci-alert" style="--alert-color:{rs["color"]}">'
        f'<div class="ci-alert-icon">{rs["icon"]}</div>'
        f'<div class="ci-alert-body"><div class="ci-alert-title">{title}</div>'
        f'<div class="ci-alert-meta">{meta}</div></div>'
        f'<div class="ci-alert-score">Demand {card["risk_score"]}</div>'
        '</div>'
    )


def render_alert_feed(cards: list[dict]) -> None:
    """Section 4 — vertical alert feed sorted by risk score (descending)."""
    st.markdown("".join(_alert_html(card) for card in cards), unsafe_allow_html=True)


def render_detailed_abnormal_table(abnormal_df: pd.DataFrame) -> None:
    """Section 5 — every flagged order line as a native Streamlit dataframe."""
    if abnormal_df.empty:
        st.success("No elevated demand detected against per-product baselines.")
        return
    rename = {
        "customer_name": "Customer",
        "product_name": "Product",
        "order_nbr": "Order",
        "order_date": "Date",
        "historical_avg": "Historical Avg",
        "historical_max": "Historical Max",
        "current_quantity": "Current Qty",
        "deviation_pct": "Deviation %",
        "risk_level": "Demand Level",
    }
    # Only the business-facing columns; hist_series / hist_count / historical_min
    # are internal inputs to the narrative, not for the flat table.
    display_abnormal = abnormal_df[[c for c in rename if c in abnormal_df.columns]].rename(columns=rename)
    # Present the raw detection bands as positive, demand-focused labels. The
    # detector emits "High"/"Medium" here; map both to their display wording.
    if "Demand Level" in display_abnormal.columns:
        _level_label = {"High": "High Demand Activity", "Medium": "Moderate Demand Activity"}
        display_abnormal["Demand Level"] = display_abnormal["Demand Level"].map(
            lambda v: _level_label.get(str(v), str(v))
        )
    st.dataframe(
        display_abnormal,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Historical Avg": st.column_config.NumberColumn(format="%.1f"),
            "Historical Max": st.column_config.NumberColumn(format="%d"),
            "Current Qty": st.column_config.NumberColumn(format="%d"),
            "Deviation %": st.column_config.NumberColumn(format="+%.0f%%"),
        },
    )


# Persisted key for the anomaly-detection sensitivity control (number input).
ABNORMAL_THRESHOLD_KEY = "ci_abnormal_threshold"


def render_threshold_control() -> float:
    """Anomaly-detection sensitivity control, shown directly above the section.

    Reads/writes ``st.session_state[ABNORMAL_THRESHOLD_KEY]`` so the value persists
    across reruns; the page reads the same key at the top to derive its analytics.
    Returns the current threshold percentage.
    """
    st.number_input(
        "Deviation Threshold (%)",
        min_value=10,
        max_value=200,
        step=5,
        key=ABNORMAL_THRESHOLD_KEY,
        help=(
            "Affects demand-opportunity detection only — it does not change any other metric "
            "on this page."
        ),
    )
    st.caption(
        "Orders exceeding this deviation percentage above historical product demand "
        "will be surfaced as demand opportunities."
    )
    return float(st.session_state[ABNORMAL_THRESHOLD_KEY])


def render_risk_summary(cards: list[dict]) -> None:
    """Section summary — count of abnormal orders in each composite risk band."""
    counts = {"Critical": 0, "High": 0, "Medium": 0, "Low": 0}
    for card in cards:
        counts[_ensure_assessment(card)["band"]] += 1
    chips = "".join(
        f'<div class="ci-risk-chip" style="--chip-color:{RISK_ASSESSMENT_STYLE[band]["color"]}">'
        f'<span class="dot">{RISK_ASSESSMENT_STYLE[band]["icon"]}</span>'
        f'<span class="body"><span class="k">{RISK_DISPLAY_LABEL[band]}</span>'
        f'<span class="v">{counts[band]}</span></span></div>'
        for band in ("Critical", "High", "Medium", "Low")
    )
    st.markdown(f'<div class="ci-risk-summary">{chips}</div>', unsafe_allow_html=True)


def render_ai_order_intelligence_center(
    abnormal_cards: list[dict],
    abnormal_df: pd.DataFrame,
    facts: pd.DataFrame,
    threshold_pct: float = 50.0,
) -> None:
    """The full premium AI section: summary -> cards -> deep analysis -> feed -> table.

    Rendered as one self-contained block placed directly below Customer Spotlight.
    """
    st.subheader("📈 Customer Demand Insights")
    st.caption(
        "AI-powered analysis of customer demand patterns, purchasing behaviour, inventory "
        "impact, and emerging demand opportunities — strongest demand signals first."
    )

    # Anomaly-detection sensitivity control — placed directly above the section.
    threshold_pct = render_threshold_control()

    # Section 1 — AI Executive Summary
    render_executive_summary(abnormal_cards, abnormal_df, threshold_pct)

    if not abnormal_cards:
        st.success(
            f"✅ No elevated demand right now — no order lines exceed their per-product demand "
            f"baseline at the {int(threshold_pct)}% threshold. The AI monitor will surface new "
            "demand opportunities here as soon as they appear."
        )
        return

    # Order the panels by executive risk priority: Critical → High → Medium → Low,
    # then by composite risk score (descending) within each band.
    ranked_cards = sorted(
        abnormal_cards,
        key=lambda c: (_RISK_PRIORITY[_ensure_assessment(c)["band"]], -_ensure_assessment(c)["score"]),
    )

    # Section summary — abnormal-order counts per risk band.
    render_risk_summary(ranked_cards)

    # Section 2 + 3 — full-width review panels & expandable deep analysis
    st.markdown("#### 🤖 Customer Demand Analysis")
    st.caption(
        "One full-width Demand Opportunity Review for every high-demand order, ordered by "
        "demand intensity. Expand any panel for the full demand-history breakdown, AI Analysis "
        "and recommended actions."
    )
    render_ai_cards(ranked_cards, facts)

    # Section 5 — Detailed demand-opportunity table (Streamlit dataframe)
    st.markdown("#### 📋 Demand Opportunity Detail")
    st.caption("Every demand opportunity behind the cards above.")
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
    "Real customer demand patterns, demand insights, inventory pressure, and "
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

# -- Abnormal-order sensitivity --------------------------------------------
# The deviation-threshold control itself now lives directly above the Abnormal
# Order Detection section (rendered by ``render_threshold_control``). It persists
# its value in ``st.session_state[ABNORMAL_THRESHOLD_KEY]``; we read that value
# here so the whole page re-derives its anomaly analytics from the chosen
# sensitivity on every rerun.
if ABNORMAL_THRESHOLD_KEY not in st.session_state:
    st.session_state[ABNORMAL_THRESHOLD_KEY] = int(cis.DEFAULT_ABNORMAL_DEVIATION_PCT)
deviation_threshold = float(st.session_state[ABNORMAL_THRESHOLD_KEY])

kpis = cis.executive_kpis(facts, customers, orders, inventory, min_deviation_pct=deviation_threshold)

# Supporting datasets, computed once and reused by both the KPI hover detail and
# their dedicated sections below (no recalculation — identical service outputs).
# Everything anomaly-related is re-derived from ``deviation_threshold`` so the KPI
# cards, AI cards, charts, tables and insights all refresh when the slider moves.
top_customers_df = cis.top_customers(facts, limit=10)
abnormal_df = cis.detect_abnormal_orders(facts, min_deviation_pct=deviation_threshold)
trends_df = cis.customer_demand_trends(facts)
impact_df = cis.inventory_impact(facts, inventory)
dormant_df = cis.dormant_accounts(customers, orders)

# Reuse the existing reorder-point logic to flag at-risk products for the cards.
at_risk_ids = cis.at_risk_products(inventory)
abnormal_cards = build_abnormal_cards(
    abnormal_df, facts, at_risk_ids, inventory, threshold_pct=deviation_threshold
)

# -- Executive KPIs --------------------------------------------------------
st.subheader("Executive KPIs")
st.caption("Hover (or tap) a card to flip it and reveal the supporting metrics.")
render_executive_kpis(kpis, top_customers_df, trends_df, abnormal_df, impact_df, dormant_df, deviation_threshold)

# -- Customer Spotlight -----------------------------------------------------
st.subheader("⭐ Customer Spotlight")
st.caption("Top 5 customers by order revenue.")
render_spotlight(cis.customer_spotlight(facts, limit=5))

st.divider()

# -- Order Quantity Limits --------------------------------------------------
# Writes a JSON config (never Oracle). Mirror the saved limits into session so
# the Order Simulator and this page share one in-session view.
st.session_state.setdefault(col.SESSION_KEY, col.load_limits())
render_order_quantity_limits(customers)

st.divider()

# ==========================================================================
# AI Order Intelligence Center — new premium section (directly below Spotlight)
# ==========================================================================
render_ai_order_intelligence_center(abnormal_cards, abnormal_df, facts, deviation_threshold)

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
    list(cis.generate_customer_insights(
        facts, customers, orders, inventory, min_deviation_pct=deviation_threshold
    )),
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
