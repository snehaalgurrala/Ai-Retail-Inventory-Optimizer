import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from backend.utils.data_loader import load_all_data  # noqa: E402
from frontend.utils.page_helpers import apply_page_style  # noqa: E402


st.set_page_config(
    page_title="AI Retail Inventory Optimizer",
    page_icon="A",
    layout="wide",
)


@st.cache_data
def load_dashboard_data() -> dict[str, pd.DataFrame]:
    return load_all_data()


def safe_sum(df: pd.DataFrame, column: str) -> int:
    if df.empty or column not in df.columns:
        return 0

    return int(pd.to_numeric(df[column], errors="coerce").fillna(0).sum())


def get_current_inventory_quantity(
    inventory: pd.DataFrame,
    transactions: pd.DataFrame,
) -> tuple[int, str]:
    if not inventory.empty and "stock_level" in inventory.columns:
        return safe_sum(inventory, "stock_level"), "inventory.csv stock snapshot"

    if transactions.empty or "transaction_type" not in transactions.columns:
        return 0, "No usable inventory source"

    snapshot_rows = transactions[
        transactions["transaction_type"].eq("inventory_snapshot")
    ]
    if not snapshot_rows.empty and "quantity" in snapshot_rows.columns:
        return safe_sum(snapshot_rows, "quantity"), "transactions.csv inventory snapshots"

    quantity = pd.to_numeric(transactions.get("quantity"), errors="coerce").fillna(0)
    transaction_type = transactions["transaction_type"].fillna("")

    signed_quantity = quantity.copy()
    signed_quantity[transaction_type.isin(["sale", "transfer_out"])] *= -1
    signed_quantity[transaction_type.eq("transfer_in")] *= 1

    return int(signed_quantity.sum()), "transactions.csv movement history"


def build_category_chart(products: pd.DataFrame):
    if products.empty or "category" not in products.columns:
        return None

    category_counts = (
        products.dropna(subset=["category"])
        .groupby("category", as_index=False)
        .size()
        .rename(columns={"size": "product_count"})
        .sort_values("product_count", ascending=False)
    )
    if category_counts.empty:
        return None

    return px.bar(
        category_counts,
        x="category",
        y="product_count",
        title="Category-wise Product Count",
        labels={"category": "Category", "product_count": "Products"},
    )


def build_top_products_chart(sales: pd.DataFrame, products: pd.DataFrame):
    if sales.empty or not {"product_id", "quantity_sold"}.issubset(sales.columns):
        return None

    top_products = (
        sales.assign(
            quantity_sold=pd.to_numeric(
                sales["quantity_sold"],
                errors="coerce",
            ).fillna(0)
        )
        .groupby("product_id", as_index=False)["quantity_sold"]
        .sum()
        .sort_values("quantity_sold", ascending=False)
        .head(10)
    )

    if "product_name" in products.columns:
        top_products = top_products.merge(
            products[["product_id", "product_name"]],
            on="product_id",
            how="left",
        )
        top_products["product_label"] = top_products["product_name"].fillna(
            top_products["product_id"]
        )
    else:
        top_products["product_label"] = top_products["product_id"]

    return px.bar(
        top_products,
        x="quantity_sold",
        y="product_label",
        orientation="h",
        title="Top 10 Selling Products",
        labels={"quantity_sold": "Quantity Sold", "product_label": "Product"},
    )


def build_sales_by_store_chart(sales: pd.DataFrame, stores: pd.DataFrame):
    if sales.empty or not {"store_id", "quantity_sold"}.issubset(sales.columns):
        return None

    sales_by_store = (
        sales.assign(
            quantity_sold=pd.to_numeric(
                sales["quantity_sold"],
                errors="coerce",
            ).fillna(0)
        )
        .groupby("store_id", as_index=False)["quantity_sold"]
        .sum()
        .sort_values("quantity_sold", ascending=False)
    )

    if "store_name" in stores.columns:
        sales_by_store = sales_by_store.merge(
            stores[["store_id", "store_name"]],
            on="store_id",
            how="left",
        )
        sales_by_store["store_label"] = sales_by_store["store_name"].fillna(
            sales_by_store["store_id"]
        )
    else:
        sales_by_store["store_label"] = sales_by_store["store_id"]

    return px.bar(
        sales_by_store,
        x="store_label",
        y="quantity_sold",
        title="Sales by Store",
        labels={"store_label": "Store", "quantity_sold": "Quantity Sold"},
    )


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
        hovermode="x unified",
    )
    st.plotly_chart(chart, use_container_width=True)


apply_page_style()

st.title("AI Retail Inventory Optimizer")
st.caption("First dashboard built from the CSV files in data/raw.")

try:
    data = load_dashboard_data()
except Exception as error:
    st.error("Could not load the dashboard data.")
    st.exception(error)
    st.stop()

products = data["products"]
sales = data["sales"]
stores = data["stores"]
suppliers = data["suppliers"]
inventory = data["inventory"]
transactions = data["transactions"]

current_inventory_quantity, inventory_source = get_current_inventory_quantity(
    inventory,
    transactions,
)

kpi_columns = st.columns(5)
kpi_columns[0].metric("Products", f"{len(products):,}")
kpi_columns[1].metric("Stores", f"{len(stores):,}")
kpi_columns[2].metric("Suppliers", f"{len(suppliers):,}")
kpi_columns[3].metric("Sales Quantity", f"{safe_sum(sales, 'quantity_sold'):,}")
kpi_columns[4].metric("Current Inventory", f"{current_inventory_quantity:,}")

st.caption(f"Current inventory source: {inventory_source}")

st.divider()

left_chart, right_chart = st.columns(2)
with left_chart:
    show_chart(
        build_category_chart(products),
        "No category data is available for the product count chart.",
    )

with right_chart:
    show_chart(
        build_sales_by_store_chart(sales, stores),
        "No sales-by-store data is available.",
    )

show_chart(
    build_top_products_chart(sales, products),
    "No product sales data is available.",
)
