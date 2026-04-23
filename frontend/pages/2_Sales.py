from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from frontend.utils.page_helpers import (
    apply_page_style,
    load_data_or_stop,
    safe_sum,
    show_chart,
)


st.set_page_config(
    page_title="Sales",
    page_icon="S",
    layout="wide",
)

apply_page_style()

st.title("Sales")
st.caption("Sales history from sales.csv.")

data = load_data_or_stop()
sales = data["sales"]
products = data["products"]
stores = data["stores"]

sales_view = sales.copy()
if not sales_view.empty:
    if {"product_id", "product_name", "category"}.issubset(products.columns):
        sales_view = sales_view.merge(
            products[["product_id", "product_name", "category"]],
            on="product_id",
            how="left",
        )
    if {"store_id", "store_name", "city"}.issubset(stores.columns):
        sales_view = sales_view.merge(
            stores[["store_id", "store_name", "city"]],
            on="store_id",
            how="left",
        )
    if {"quantity_sold", "selling_price"}.issubset(sales_view.columns):
        sales_view["sales_value"] = (
            pd.to_numeric(sales_view["quantity_sold"], errors="coerce").fillna(0)
            * pd.to_numeric(sales_view["selling_price"], errors="coerce").fillna(0)
        )

total_sales_quantity = safe_sum(sales, "quantity_sold")
total_sales_value = safe_sum(sales_view, "sales_value")
daily_count = sales["date"].nunique() if "date" in sales.columns else 0
average_daily_quantity = total_sales_quantity / daily_count if daily_count else 0

kpi_columns = st.columns(4)
kpi_columns[0].metric("Sales Rows", f"{len(sales):,}")
kpi_columns[1].metric("Quantity Sold", f"{total_sales_quantity:,}")
kpi_columns[2].metric("Sales Value", f"{total_sales_value:,}")
kpi_columns[3].metric("Avg Daily Qty", f"{average_daily_quantity:,.1f}")

st.divider()

trend_chart = None
if not sales.empty and {"date", "quantity_sold"}.issubset(sales.columns):
    sales_trend = (
        sales.assign(
            quantity_sold=pd.to_numeric(
                sales["quantity_sold"],
                errors="coerce",
            ).fillna(0)
        )
        .groupby("date", as_index=False)["quantity_sold"]
        .sum()
        .sort_values("date")
    )
    trend_chart = px.line(
        sales_trend,
        x="date",
        y="quantity_sold",
        title="Sales Quantity Trend",
        markers=True,
        labels={"date": "Date", "quantity_sold": "Quantity Sold"},
    )
show_chart(trend_chart, "No date-based sales trend data is available.")

left_chart, right_chart = st.columns(2)

with left_chart:
    product_chart = None
    if not sales_view.empty and {"product_id", "quantity_sold"}.issubset(sales_view.columns):
        product_label = "product_name" if "product_name" in sales_view.columns else "product_id"
        top_products = (
            sales_view.assign(
                quantity_sold=pd.to_numeric(
                    sales_view["quantity_sold"],
                    errors="coerce",
                ).fillna(0)
            )
            .groupby(product_label, as_index=False)["quantity_sold"]
            .sum()
            .sort_values("quantity_sold", ascending=False)
            .head(10)
        )
        product_chart = px.bar(
            top_products,
            x="quantity_sold",
            y=product_label,
            orientation="h",
            title="Top 10 Selling Products",
            labels={"quantity_sold": "Quantity Sold", product_label: "Product"},
        )
    show_chart(product_chart, "No product-level sales data is available.")

with right_chart:
    store_chart = None
    if not sales_view.empty and {"store_id", "quantity_sold"}.issubset(sales_view.columns):
        store_label = "store_name" if "store_name" in sales_view.columns else "store_id"
        sales_by_store = (
            sales_view.assign(
                quantity_sold=pd.to_numeric(
                    sales_view["quantity_sold"],
                    errors="coerce",
                ).fillna(0)
            )
            .groupby(store_label, as_index=False)["quantity_sold"]
            .sum()
            .sort_values("quantity_sold", ascending=False)
        )
        store_chart = px.bar(
            sales_by_store,
            x=store_label,
            y="quantity_sold",
            title="Sales by Store",
            labels={store_label: "Store", "quantity_sold": "Quantity Sold"},
        )
    show_chart(store_chart, "No store-level sales data is available.")

st.subheader("Sales Table")
if sales_view.empty:
    st.info("sales.csv is empty.")
else:
    st.dataframe(sales_view, use_container_width=True, hide_index=True)
