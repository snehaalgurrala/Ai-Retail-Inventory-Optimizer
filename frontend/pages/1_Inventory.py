from pathlib import Path
import sys

import plotly.express as px
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from frontend.utils.page_helpers import (
    apply_page_style,
    load_data_or_stop,
    render_page_header,
    render_kpi_card,
    render_chart_card,
    safe_sum,
    style_bar_chart,
    style_donut_chart,
)


st.set_page_config(
    page_title="Inventory",
    page_icon="I",
    layout="wide",
)

apply_page_style()

render_page_header(
    "📦 Inventory",
    "Current stock snapshot from inventory.csv.",
)

data = load_data_or_stop()
inventory = data["inventory"]
products = data["products"]
stores = data["stores"]

inventory_view = inventory.copy()
if not inventory_view.empty:
    if {"product_id", "product_name", "category"}.issubset(products.columns):
        inventory_view = inventory_view.merge(
            products[["product_id", "product_name", "category"]],
            on="product_id",
            how="left",
        )
    if {"store_id", "store_name", "city"}.issubset(stores.columns):
        inventory_view = inventory_view.merge(
            stores[["store_id", "store_name", "city"]],
            on="store_id",
            how="left",
        )

total_stock = safe_sum(inventory, "stock_level")
low_stock_count = 0
if {"stock_level", "reorder_threshold"}.issubset(inventory.columns):
    low_stock_count = int(
        (
            inventory["stock_level"].fillna(0)
            <= inventory["reorder_threshold"].fillna(0)
        ).sum()
    )

kpi_columns = st.columns(4, gap="medium")
with kpi_columns[0]:
    render_kpi_card("Stock Units", f"{total_stock:,}", "Current inventory units", "purple")
with kpi_columns[1]:
    render_kpi_card("Product-store Rows", f"{len(inventory):,}", "Inventory records", "blue")
with kpi_columns[2]:
    render_kpi_card("Low Stock Rows", f"{low_stock_count:,}", "At or below threshold", "orange")
with kpi_columns[3]:
    render_kpi_card(
        "Stores",
        f"{inventory['store_id'].nunique() if 'store_id' in inventory else 0:,}",
        "Store locations in inventory",
        "red",
    )

st.divider()

left_chart, right_chart = st.columns(2, gap="large")

with left_chart:
    stock_by_store = None
    if not inventory_view.empty and {"store_id", "stock_level"}.issubset(inventory_view.columns):
        stock_by_store = (
            inventory_view.groupby(
                "store_name" if "store_name" in inventory_view.columns else "store_id",
                as_index=False,
            )["stock_level"]
            .sum()
            .sort_values("stock_level", ascending=False)
        )
        stock_by_store.columns = ["store", "stock_level"]
        stock_by_store = px.bar(
            stock_by_store,
            x="store",
            y="stock_level",
            labels={"store": "Store", "stock_level": "Stock Units"},
        )
        stock_by_store = style_bar_chart(stock_by_store, "blue")
    render_chart_card(
        "Current Stock by Store",
        "Total stock units available in each store.",
        stock_by_store,
        "No stock-by-store data is available.",
    )

with right_chart:
    stock_by_category = None
    if not inventory_view.empty and {"category", "stock_level"}.issubset(inventory_view.columns):
        stock_by_category = (
            inventory_view.dropna(subset=["category"])
            .groupby("category", as_index=False)["stock_level"]
            .sum()
            .sort_values("stock_level", ascending=False)
        )
        stock_by_category = px.pie(
            stock_by_category,
            names="category",
            values="stock_level",
        )
        stock_by_category = style_donut_chart(stock_by_category)
    render_chart_card(
        "Inventory Distribution",
        "Donut view of stock units across product categories.",
        stock_by_category,
        "No category stock data is available.",
    )

st.subheader("📋 Current Stock Table")
if inventory_view.empty:
    st.info("inventory.csv is empty.")
else:
    st.dataframe(inventory_view, use_container_width=True, hide_index=True)

st.subheader("⚠ Low Stock Rows")
if inventory_view.empty or not {"stock_level", "reorder_threshold"}.issubset(inventory_view.columns):
    st.info("Low stock status cannot be calculated from the available columns.")
else:
    low_stock = inventory_view[
        inventory_view["stock_level"].fillna(0)
        <= inventory_view["reorder_threshold"].fillna(0)
    ]
    if low_stock.empty:
        st.success("No current stock rows are at or below reorder threshold.")
    else:
        st.dataframe(low_stock, use_container_width=True, hide_index=True)
