from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from backend.services.store_inventory_service import (  # noqa: E402
    build_category_summary,
    build_store_comparison,
    build_store_inventory_summary,
    build_store_inventory_view,
    build_store_kpis,
    filter_inventory_by_store,
    filter_recommendations_for_store,
    get_overstock_items,
    get_understock_items,
    load_store_inventory_inputs,
)
from frontend.utils.page_helpers import (  # noqa: E402
    apply_chart_theme,
    apply_page_style,
    render_chart_card,
    render_kpi_card,
    render_page_header,
    style_bar_chart,
    style_donut_chart,
)


st.set_page_config(
    page_title="Inventory",
    page_icon="I",
    layout="wide",
)

apply_page_style()


@st.cache_data
def load_inventory_dashboard_data() -> dict[str, pd.DataFrame]:
    return load_store_inventory_inputs()


def money(value: float) -> str:
    return f"{float(value or 0):,.2f}"


def format_table(df: pd.DataFrame, columns: list[str], rename_map: dict[str, str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[rename_map.get(column, column) for column in columns])
    available = [column for column in columns if column in df.columns]
    return df[available].rename(columns=rename_map)


def status_donut(view: pd.DataFrame):
    if view.empty or "stock_status" not in view.columns:
        return None
    summary = (
        view.groupby("stock_status", as_index=False)["product_id"]
        .count()
        .rename(columns={"product_id": "row_count"})
    )
    chart = px.pie(summary, names="stock_status", values="row_count")
    return style_donut_chart(chart)


def category_quantity_bar(view: pd.DataFrame):
    summary = build_category_summary(view)
    if summary.empty:
        return None
    chart = px.bar(
        summary,
        x="category",
        y="inventory_quantity",
        labels={"category": "Category", "inventory_quantity": "Inventory Quantity"},
    )
    return style_bar_chart(chart, "blue")


def category_value_bar(view: pd.DataFrame):
    summary = build_category_summary(view)
    if summary.empty or "inventory_value" not in summary.columns:
        return None
    chart = px.bar(
        summary,
        x="category",
        y="inventory_value",
        labels={"category": "Category", "inventory_value": "Inventory Value"},
    )
    return style_bar_chart(chart, "green")


def top_stock_bar(view: pd.DataFrame, ascending: bool):
    if view.empty:
        return None
    ranked = view.sort_values(["current_quantity", "product_name"], ascending=[ascending, True]).head(10)
    chart = px.bar(
        ranked,
        x="current_quantity",
        y="product_name",
        orientation="h",
        color="stock_status",
        labels={"current_quantity": "Quantity", "product_name": "Product", "stock_status": "Status"},
    )
    chart = apply_chart_theme(chart)
    chart.update_layout(yaxis=dict(categoryorder="total ascending"))
    return chart


def render_recommendation_cards(recommendations: pd.DataFrame) -> None:
    if recommendations.empty:
        st.info("No active recommendations for this store right now.")
        return

    for _, row in recommendations.head(8).iterrows():
        with st.container(border=True):
            top = st.columns([2.2, 1, 1])
            top[0].markdown(f"**{row.get('recommendation_type', '')}**")
            top[1].caption(f"Priority: {row.get('priority', '')}")
            top[2].caption(f"Source: {row.get('source_agent', '')}")
            st.write(f"**Product:** {row.get('product_name', '')}")
            st.write(f"**Action:** {row.get('action', '')}")
            st.caption(f"Reason: {row.get('reason', '')}")


def render_store_comparison(view: pd.DataFrame) -> None:
    comparison = build_store_comparison(view)
    if comparison.empty:
        st.info("No store comparison data is available.")
        return

    st.subheader("Store Comparison")
    st.dataframe(
        comparison.rename(
            columns={
                "store_name": "Store",
                "city": "City",
                "inventory_quantity": "Inventory Quantity",
                "product_count": "Products",
                "low_stock_count": "Low Stock",
                "overstock_count": "Overstock",
                "inventory_value": "Inventory Value",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    row1_left, row1_right = st.columns(2, gap="large")
    with row1_left:
        chart = px.bar(
            comparison,
            x="store_name",
            y="inventory_quantity",
            labels={"store_name": "Store", "inventory_quantity": "Inventory Quantity"},
        )
        render_chart_card(
            "Inventory Quantity By Store",
            "Total units currently available by branch.",
            style_bar_chart(chart, "blue"),
            "No store quantity data is available.",
        )
    with row1_right:
        low_chart = px.bar(
            comparison,
            x="store_name",
            y="low_stock_count",
            labels={"store_name": "Store", "low_stock_count": "Low-stock Count"},
        )
        render_chart_card(
            "Low-stock Count By Store",
            "Number of product rows at or below reorder threshold.",
            style_bar_chart(low_chart, "orange"),
            "No low-stock comparison data is available.",
        )

    row2_left, row2_right = st.columns(2, gap="large")
    with row2_left:
        value_chart = px.bar(
            comparison,
            x="store_name",
            y="inventory_value",
            labels={"store_name": "Store", "inventory_value": "Inventory Value"},
        )
        render_chart_card(
            "Inventory Value By Store",
            "Estimated value from quantity multiplied by selling price.",
            style_bar_chart(value_chart, "green"),
            "No store value data is available.",
        )
    with row2_right:
        category_store = build_category_summary(view, by_store=True)
        chart = None
        if not category_store.empty:
            chart = px.bar(
                category_store,
                x="store_name",
                y="inventory_quantity",
                color="category",
                labels={"store_name": "Store", "inventory_quantity": "Inventory Quantity", "category": "Category"},
            )
            chart = apply_chart_theme(chart)
        render_chart_card(
            "Category Distribution Across Stores",
            "Inventory quantity split by category and branch.",
            chart,
            "No category-store distribution is available.",
        )


render_page_header(
    "📦 Inventory Intelligence",
    "Store-wise inventory health, stock status, and manager-ready actions.",
)

try:
    frames = load_inventory_dashboard_data()
except Exception as error:
    st.error("Could not load inventory dashboard data.")
    st.exception(error)
    st.stop()

inventory_view = build_store_inventory_view(
    frames["inventory"],
    frames["products"],
    frames["stores"],
    frames.get("suppliers", pd.DataFrame()),
    frames.get("sales", pd.DataFrame()),
)

if inventory_view.empty:
    st.info("No inventory data is available.")
    st.stop()

stores = frames["stores"].copy()
store_options = ["All Stores"]
store_label_to_id = {"All Stores": "All Stores"}
if not stores.empty and {"store_id", "store_name"}.issubset(stores.columns):
    for _, row in stores.sort_values("store_name").iterrows():
        label = f"{row.get('store_name', row.get('store_id'))} ({row.get('city', '')})"
        store_options.append(label)
        store_label_to_id[label] = str(row.get("store_id", ""))

selected_label = st.selectbox("Store", store_options, index=0)
selected_store_id = store_label_to_id.get(selected_label, "All Stores")
selected_view = filter_inventory_by_store(inventory_view, selected_store_id)

if selected_store_id != "All Stores":
    store_row = stores[stores["store_id"].astype(str).eq(str(selected_store_id))]
    if not store_row.empty:
        row = store_row.iloc[0]
        st.caption(
            f"{row.get('store_name', selected_store_id)} • {row.get('city', 'Unknown city')} • "
            f"Capacity {row.get('capacity', 'N/A')}"
        )
else:
    st.caption("All branches are included. Store comparison mode is active.")

kpis = build_store_kpis(selected_view)
kpi_cols = st.columns(6, gap="medium")
with kpi_cols[0]:
    render_kpi_card("Inventory Qty", f"{kpis['total_quantity']:,}", "Total units", "blue")
with kpi_cols[1]:
    render_kpi_card("Products", f"{kpis['product_count']:,}", "Unique products", "purple")
with kpi_cols[2]:
    render_kpi_card("Low Stock", f"{kpis['low_stock_count']:,}", "At or below threshold", "orange")
with kpi_cols[3]:
    render_kpi_card("Overstock", f"{kpis['overstock_count']:,}", "High stock rows", "green")
with kpi_cols[4]:
    render_kpi_card("Slow / Dead", f"{kpis['slow_dead_count']:,}", "No recent movement", "red")
with kpi_cols[5]:
    render_kpi_card("Inventory Value", money(kpis["inventory_value"]), "Qty × selling price", "green")

st.divider()

if selected_store_id == "All Stores":
    render_store_comparison(selected_view)
    st.divider()

st.subheader("AI Inventory Insight")
store_summary_label = selected_label if selected_store_id != "All Stores" else "All stores"
store_recommendations = filter_recommendations_for_store(frames["recommendations"], selected_store_id)
st.info(build_store_inventory_summary(selected_view, store_summary_label, store_recommendations))

st.subheader("Store-wise Inventory Table")
inventory_table = format_table(
    selected_view.sort_values(["stock_status", "category", "product_name"]),
    [
        "product_name",
        "category",
        "current_quantity",
        "reorder_threshold",
        "stock_status",
        "selling_price",
        "supplier_name",
    ],
    {
        "product_name": "Product",
        "category": "Category",
        "current_quantity": "Current Quantity",
        "reorder_threshold": "Reorder Threshold",
        "stock_status": "Stock Status",
        "selling_price": "Selling Price",
        "supplier_name": "Supplier",
    },
)
st.dataframe(inventory_table, use_container_width=True, hide_index=True)

st.divider()

understock = get_understock_items(selected_view)
overstock = get_overstock_items(selected_view)

left_section, right_section = st.columns(2, gap="large")
with left_section:
    st.subheader("Understock")
    if understock.empty:
        st.success("No understock items for this store selection.")
    else:
        st.dataframe(
            format_table(
                understock,
                [
                    "product_name",
                    "current_quantity",
                    "reorder_threshold",
                    "shortage_quantity",
                    "suggested_reorder_quantity",
                    "priority",
                    "ai_recommendation",
                ],
                {
                    "product_name": "Product",
                    "current_quantity": "Current Qty",
                    "reorder_threshold": "Threshold",
                    "shortage_quantity": "Shortage Qty",
                    "suggested_reorder_quantity": "Suggested Reorder Qty",
                    "priority": "Priority",
                    "ai_recommendation": "AI Recommendation",
                },
            ),
            use_container_width=True,
            hide_index=True,
        )

with right_section:
    st.subheader("Overstock")
    if overstock.empty:
        st.success("No overstock items for this store selection.")
    else:
        overstock_view = overstock.copy()
        overstock_view["suggested_action"] = overstock_view["recent_daily_sales_velocity"].map(
            lambda velocity: "Discount / clearance review" if float(velocity or 0) <= 0.2 else "Transfer or promotion"
        )
        st.dataframe(
            format_table(
                overstock_view,
                [
                    "product_name",
                    "current_quantity",
                    "reorder_threshold",
                    "surplus_quantity",
                    "suggested_action",
                    "ai_recommendation",
                ],
                {
                    "product_name": "Product",
                    "current_quantity": "Current Qty",
                    "reorder_threshold": "Threshold",
                    "surplus_quantity": "Surplus Qty",
                    "suggested_action": "Suggested Action",
                    "ai_recommendation": "AI Recommendation",
                },
            ),
            use_container_width=True,
            hide_index=True,
        )

st.divider()

st.subheader("Store-specific AI Recommendations")
render_recommendation_cards(store_recommendations)

st.divider()

st.subheader("Category-wise Graphs")
chart_a, chart_b = st.columns(2, gap="large")
with chart_a:
    render_chart_card(
        "Category-wise Inventory Quantity",
        "Quantity available by category.",
        category_quantity_bar(selected_view),
        "No category quantity data is available.",
    )
with chart_b:
    render_chart_card(
        "Category-wise Inventory Value",
        "Estimated inventory value by category.",
        category_value_bar(selected_view),
        "No category value data is available.",
    )

chart_c, chart_d = st.columns(2, gap="large")
with chart_c:
    render_chart_card(
        "Stock Status Distribution",
        "Low Stock, Healthy, and Overstock mix.",
        status_donut(selected_view),
        "No stock status data is available.",
    )
with chart_d:
    render_chart_card(
        "Top 10 Highest Stock Products",
        "Products with the highest on-hand quantity.",
        top_stock_bar(selected_view, ascending=False),
        "No highest-stock data is available.",
    )

render_chart_card(
    "Top 10 Lowest Stock Products",
    "Products with the lowest on-hand quantity.",
    top_stock_bar(selected_view, ascending=True),
    "No lowest-stock data is available.",
)
