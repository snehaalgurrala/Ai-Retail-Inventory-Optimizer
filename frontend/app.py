import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from backend.utils.data_loader import load_all_data  # noqa: E402
from frontend.components.ui_components import (  # noqa: E402
    render_kpi_card,
    render_recommendation_summary,
)
from frontend.utils.page_helpers import (  # noqa: E402
    apply_page_style,
    render_chart_card,
    style_bar_chart,
    style_donut_chart,
    style_sales_trend_chart,
)


st.set_page_config(
    page_title="AI Retail Inventory Optimizer",
    page_icon="📊",
    layout="wide",
)


@st.cache_data
def load_dashboard_data() -> dict[str, pd.DataFrame]:
    return load_all_data()


def safe_sum(df: pd.DataFrame, column: str) -> int:
    if df.empty or column not in df.columns:
        return 0

    return int(pd.to_numeric(df[column], errors="coerce").fillna(0).sum())


def processed_row_count(filename: str) -> int:
    """Return row count for a processed output when it exists."""
    file_path = PROJECT_ROOT / "data" / "processed" / filename
    if not file_path.exists():
        return 0

    try:
        return len(pd.read_csv(file_path))
    except Exception:
        return 0


def load_processed_output(filename: str) -> pd.DataFrame:
    """Load one processed output when available."""
    file_path = PROJECT_ROOT / "data" / "processed" / filename
    if not file_path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(file_path)
    except Exception:
        return pd.DataFrame()


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


def build_sales_trend_chart(sales: pd.DataFrame):
    if sales.empty or not {"date", "quantity_sold"}.issubset(sales.columns):
        return None

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

    chart = px.line(
        sales_trend,
        x="date",
        y="quantity_sold",
        markers=True,
        labels={"date": "Date", "quantity_sold": "Quantity Sold"},
    )
    return style_sales_trend_chart(chart)


def build_inventory_distribution_chart(
    inventory: pd.DataFrame,
    products: pd.DataFrame,
):
    if inventory.empty or not {"product_id", "stock_level"}.issubset(inventory.columns):
        return None

    inventory_view = inventory.copy()
    if {"product_id", "category"}.issubset(products.columns):
        inventory_view = inventory_view.merge(
            products[["product_id", "category"]],
            on="product_id",
            how="left",
        )

    if "category" not in inventory_view.columns:
        return None

    stock_by_category = (
        inventory_view.dropna(subset=["category"])
        .assign(
            stock_level=pd.to_numeric(
                inventory_view["stock_level"],
                errors="coerce",
            ).fillna(0)
        )
        .groupby("category", as_index=False)["stock_level"]
        .sum()
        .sort_values("stock_level", ascending=False)
    )
    if stock_by_category.empty:
        return None

    chart = px.pie(
        stock_by_category,
        names="category",
        values="stock_level",
    )
    return style_donut_chart(chart)


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

    chart = px.bar(
        category_counts,
        x="category",
        y="product_count",
        labels={"category": "Category", "product_count": "Products"},
    )
    return style_bar_chart(chart, "purple")


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

    chart = px.bar(
        top_products,
        x="quantity_sold",
        y="product_label",
        orientation="h",
        labels={"quantity_sold": "Quantity Sold", "product_label": "Product"},
    )
    return style_bar_chart(chart, "blue")


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

    chart = px.bar(
        sales_by_store,
        x="store_label",
        y="quantity_sold",
        labels={"store_label": "Store", "quantity_sold": "Quantity Sold"},
    )
    return style_bar_chart(chart, "green")


def recommendation_type_rows(
    recommendations: pd.DataFrame,
    recommendation_types: list[str],
) -> pd.DataFrame:
    if recommendations.empty or "recommendation_type" not in recommendations.columns:
        return pd.DataFrame()

    return recommendations[
        recommendations["recommendation_type"].astype(str).isin(recommendation_types)
    ].copy()


def unique_count(df: pd.DataFrame, column: str) -> int:
    if df.empty or column not in df.columns:
        return 0
    return df[column].fillna("").astype(str).replace("", pd.NA).dropna().nunique()


def high_priority_count(df: pd.DataFrame) -> int:
    if df.empty or "priority" not in df.columns:
        return 0
    return int(df["priority"].astype(str).str.lower().eq("high").sum())


def pending_count(df: pd.DataFrame) -> int:
    if df.empty or "status" not in df.columns:
        return 0
    return int(df["status"].astype(str).str.lower().eq("pending").sum())


def first_action(df: pd.DataFrame) -> str:
    if df.empty or "action" not in df.columns:
        return "No current action available."

    actions = df["action"].fillna("").astype(str)
    actions = actions[actions != ""]
    if actions.empty:
        return "No current action available."

    return actions.iloc[0]


def build_recommendation_summary_cards(recommendations: pd.DataFrame) -> list[dict]:
    pricing_rows = recommendation_type_rows(recommendations, ["discount", "clearance"])
    transfer_rows = recommendation_type_rows(recommendations, ["stock_transfer"])
    dead_stock_rows = recommendation_type_rows(recommendations, ["clearance"])
    risk_rows = recommendation_type_rows(
        recommendations,
        ["supplier_risk_alert", "overstock_alert", "stockout_prevention_alert"],
    )

    return [
        {
            "title": "Pricing & Discounts",
            "icon": "💸",
            "summary": f"{unique_count(pricing_rows, 'product_id'):,} products affected",
            "accent": "blue",
            "button_key": "dashboard_review_pricing",
            "insights": [
                f"{len(pricing_rows):,} pricing recommendations available.",
                f"{pending_count(pricing_rows):,} pending review.",
                first_action(pricing_rows),
            ],
        },
        {
            "title": "Stock Transfer",
            "icon": "↔",
            "summary": f"{len(transfer_rows):,} transfer opportunities",
            "accent": "purple",
            "button_key": "dashboard_review_transfer",
            "insights": [
                f"{unique_count(transfer_rows, 'store_id'):,} destination stores involved.",
                f"{unique_count(transfer_rows, 'product_id'):,} products can be balanced.",
                first_action(transfer_rows),
            ],
        },
        {
            "title": "Dead Stock",
            "icon": "⏳",
            "summary": f"{unique_count(dead_stock_rows, 'product_id'):,} products affected",
            "accent": "orange",
            "button_key": "dashboard_review_dead_stock",
            "insights": [
                f"{len(dead_stock_rows):,} clearance recommendations found.",
                f"{high_priority_count(dead_stock_rows):,} high priority rows.",
                first_action(dead_stock_rows),
            ],
        },
        {
            "title": "Risk Alerts",
            "icon": "⚠",
            "summary": f"{len(risk_rows):,} active alerts",
            "accent": "red",
            "button_key": "dashboard_review_risk",
            "insights": [
                f"{high_priority_count(risk_rows):,} high priority alerts.",
                f"{unique_count(risk_rows, 'product_id'):,} products need review.",
                first_action(risk_rows),
            ],
        },
    ]


apply_page_style()

with st.container():
    st.title("AI Retail Inventory Optimizer")
    st.caption(
        "A data-grounded inventory command center for sales movement, current "
        "stock health, and explainable recommendation workflows."
    )

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

total_sales_quantity = safe_sum(sales, "quantity_sold")
low_stock_count = 0
if {"stock_level", "reorder_threshold"}.issubset(inventory.columns):
    low_stock_count = int(
        (
            pd.to_numeric(inventory["stock_level"], errors="coerce").fillna(0)
            <= pd.to_numeric(
                inventory["reorder_threshold"],
                errors="coerce",
            ).fillna(0)
        ).sum()
    )
dead_stock_count = processed_row_count("dead_stock_candidates.csv")
recommendations = load_processed_output("recommendations.csv")

st.subheader("Business Overview")
st.caption("Core performance indicators from inventory, sales, and analyzer outputs.")

kpi_columns = st.columns(4, gap="medium")
with kpi_columns[0]:
    render_kpi_card(
        "Total Sales",
        f"{total_sales_quantity:,}",
        f"{len(sales):,} sales rows in the current dataset",
        "blue",
    )
with kpi_columns[1]:
    render_kpi_card(
        "Total Inventory",
        f"{current_inventory_quantity:,}",
        inventory_source,
        "purple",
    )
with kpi_columns[2]:
    render_kpi_card(
        "Low Stock",
        f"{low_stock_count:,}",
        "Rows at or below reorder threshold",
        "orange",
    )
with kpi_columns[3]:
    render_kpi_card(
        "Dead Stock",
        f"{dead_stock_count:,}",
        "Candidates from processed analyzer output",
        "red",
    )

st.caption(f"Current inventory source: {inventory_source}")
st.divider()

st.subheader("Performance Trends")
st.caption("Sales movement and inventory composition in a balanced two-column view.")

middle_left, middle_right = st.columns(2, gap="large")
with middle_left:
    render_chart_card(
        "Sales Trend",
        "Daily sales movement from the available sales history.",
        build_sales_trend_chart(sales),
        "No date-based sales trend data is available.",
    )

with middle_right:
    render_chart_card(
        "Inventory Distribution",
        "Current stock units distributed across product categories.",
        build_inventory_distribution_chart(inventory, products),
        "No category inventory data is available.",
    )

st.divider()

st.subheader("AI Recommendations")
st.caption("Grouped decision areas from the latest generated recommendations.")

if recommendations.empty:
    st.info("No recommendations are available yet. Run the agents to generate them.")
else:
    recommendation_cards = build_recommendation_summary_cards(recommendations)
    for row_start in range(0, len(recommendation_cards), 2):
        card_columns = st.columns(2, gap="large")
        for index, card_data in enumerate(recommendation_cards[row_start:row_start + 2]):
            with card_columns[index]:
                if render_recommendation_summary(
                    title=card_data["title"],
                    summary=card_data["summary"],
                    insights=card_data["insights"],
                    icon=card_data["icon"],
                    button_key=card_data["button_key"],
                ):
                    st.switch_page("pages/3_Recommendations.py")
