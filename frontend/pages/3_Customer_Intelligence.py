from html import escape
from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from backend.db import repository  # noqa: E402
from backend.services import customer_intelligence_service as cis  # noqa: E402
from frontend.utils.page_helpers import (  # noqa: E402
    CHART_COLORS,
    apply_page_style,
    clean_display_df,
    render_ai_insight_panel,
    render_chart_card,
    render_page_header,
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


# ---------------------------------------------------------------------------
# Page body
# ---------------------------------------------------------------------------
apply_page_style()
st.markdown(SPOTLIGHT_CSS, unsafe_allow_html=True)
st.markdown(KPI_CSS, unsafe_allow_html=True)

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

# -- Section 1: Executive KPIs ---------------------------------------------
st.subheader("Executive KPIs")
st.caption("Hover (or tap) a card to flip it and reveal the supporting metrics.")
render_executive_kpis(kpis, top_customers_df, trends_df, abnormal_df, impact_df, dormant_df)

# -- Customer Spotlight -----------------------------------------------------
st.subheader("⭐ Customer Spotlight")
st.caption("Top 5 customers by order revenue.")
render_spotlight(cis.customer_spotlight(facts, limit=5))

st.divider()

# -- Section 2: Top Products ------------------------------------------------
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

# -- Section 3: Top Customers ----------------------------------------------
st.subheader("Top Customers")
st.dataframe(
    clean_display_df(
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
        )[["Customer", "Segment", "Tier", "Orders", "Units", "Revenue", "Contribution %"]]
    ),
    use_container_width=True,
    hide_index=True,
    column_config={
        "Revenue": st.column_config.NumberColumn(format="$%.2f"),
        "Contribution %": st.column_config.NumberColumn(format="%.1f%%"),
    },
)

st.divider()

# -- Section 4: Abnormal Order Detection -----------------------------------
st.subheader("Abnormal Order Detection")
st.caption(
    "Order lines whose quantity is well above the product's historical average. "
    "Per-product baseline (deviation-driven) — not per-customer z-scores, given the short history."
)
if abnormal_df.empty:
    st.success("No abnormal order quantities detected against per-product baselines.")
else:
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
        clean_display_df(display_abnormal),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Deviation %": st.column_config.NumberColumn(format="+%.0f%%"),
        },
    )

st.divider()

# -- Section 5: Customer Demand Trends -------------------------------------
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

# -- Section 6: Inventory Impact Analysis ----------------------------------
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
        st.dataframe(
            clean_display_df(
                impact_df.rename(
                    columns={
                        "customer_name": "Customer",
                        "at_risk_units": "At-Risk Units",
                        "at_risk_revenue": "At-Risk Revenue",
                        "affected_products": "Products",
                        "risk_score": "Risk Score",
                    }
                )[["Customer", "At-Risk Units", "At-Risk Revenue", "Products", "Risk Score"]]
            ),
            use_container_width=True,
            hide_index=True,
            column_config={
                "At-Risk Revenue": st.column_config.NumberColumn(format="$%.0f"),
                "Risk Score": st.column_config.ProgressColumn(
                    format="%d", min_value=0, max_value=100
                ),
            },
        )

st.divider()

# -- Section 7: Dormant Accounts -------------------------------------------
st.subheader("Dormant Accounts")
st.caption("Active customers with zero orders — a re-engagement opportunity, not an error.")
if dormant_df.empty:
    st.success("Every active customer has placed at least one order.")
else:
    st.dataframe(
        clean_display_df(
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
            )
        ),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Credit Limit": st.column_config.NumberColumn(format="$%.0f"),
        },
    )

st.divider()

# -- Section 8: AI Insights -------------------------------------------------
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
    st.dataframe(clean_display_df(facts[record_cols]), use_container_width=True, hide_index=True)
