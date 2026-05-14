from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from backend.services.sales_analytics_service import (  # noqa: E402
    SalesFilters,
    apply_sales_filters,
    branch_comparison,
    branch_options,
    branch_sales_summary,
    category_sales,
    generate_sales_insights,
    inventory_sales_comparison,
    overview_metrics,
    prepare_sales_dataset,
    product_performance,
    trend_data,
)
from frontend.utils.page_helpers import (  # noqa: E402
    CHART_PALETTE,
    apply_chart_theme,
    apply_page_style,
    load_data_or_stop,
    render_chart_card,
    render_page_header,
    style_bar_chart,
    style_donut_chart,
    style_sales_trend_chart,
)


st.set_page_config(
    page_title="Sales",
    page_icon="S",
    layout="wide",
)


def apply_sales_styles() -> None:
    st.markdown(
        """
        <style>
        div[data-testid="stMetric"] {
            min-height: 96px;
        }
        div[data-testid="stMetric"] label {
            color: color-mix(in srgb, var(--text-color) 62%, transparent);
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def build_sales_view(
    sales: pd.DataFrame,
    stores: pd.DataFrame,
    products: pd.DataFrame,
) -> pd.DataFrame:
    return prepare_sales_dataset(sales, stores, products)


def money(value: float) -> str:
    return f"{float(value):,.0f}"


def compact(value: str, max_len: int = 28) -> str:
    value = str(value or "")
    return value if len(value) <= max_len else f"{value[: max_len - 1]}..."


def render_sales_kpis(metrics: dict) -> None:
    cards = [
        ("Total Sales", money(metrics["total_sales"]), "Revenue from filtered sales"),
        ("Total Orders", f"{int(metrics['total_orders']):,}", "Distinct sales records"),
        ("Top Selling Product", compact(metrics["top_product"]), "By units sold"),
        ("Least Selling Product", compact(metrics["least_product"]), "Lowest unit movement"),
        ("Revenue Trend", metrics["revenue_trend"], "Recent vs early period"),
        ("Fastest Moving Product", compact(metrics["fastest_moving_product"]), "Recent velocity leader"),
    ]
    columns = st.columns(6, gap="small")
    for column, (label, value, note) in zip(columns, cards):
        with column:
            with st.container(border=True):
                st.metric(label, value)
                st.caption(note)


def make_branch_bar(branch_df: pd.DataFrame):
    if branch_df.empty:
        return None
    chart = px.bar(
        branch_df.sort_values("sales_value", ascending=True),
        x="sales_value",
        y="branch_label",
        orientation="h",
        text="sales_value",
        labels={"branch_label": "Branch", "sales_value": "Total Sales"},
    )
    chart.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    return style_bar_chart(chart, "blue")


def make_trend_chart(trend_df: pd.DataFrame, metric: str):
    if trend_df.empty:
        return None
    chart = px.line(
        trend_df,
        x="date",
        y=metric,
        markers=True,
        labels={"date": "Date", metric: "Sales" if metric == "sales_value" else "Units Sold"},
    )
    return style_sales_trend_chart(chart)


def make_product_chart(product_df: pd.DataFrame):
    if product_df.empty:
        return None
    chart = px.bar(
        product_df.sort_values("quantity_sold", ascending=True),
        x="quantity_sold",
        y="product_name",
        orientation="h",
        color="category",
        labels={"quantity_sold": "Units Sold", "product_name": "Product", "category": "Category"},
        color_discrete_sequence=CHART_PALETTE,
    )
    return apply_chart_theme(chart, height=410)


def make_category_chart(category_df: pd.DataFrame):
    if category_df.empty:
        return None
    chart = px.pie(
        category_df,
        names="category",
        values="sales_value",
        color_discrete_sequence=CHART_PALETTE,
    )
    chart = style_donut_chart(chart)
    chart.update_traces(hovertemplate="<b>%{label}</b><br>Sales: %{value:,.0f}<extra></extra>")
    return chart


def make_inventory_sales_chart(comparison_df: pd.DataFrame):
    if comparison_df.empty:
        return None
    plot_df = comparison_df.head(10).copy()
    chart = px.bar(
        plot_df,
        x="product_name",
        y=["stock_level", "quantity_sold"],
        barmode="group",
        labels={"product_name": "Product", "value": "Units", "variable": "Metric"},
        color_discrete_sequence=["#0ea5a4", "#f59e0b"],
    )
    chart.update_layout(xaxis_tickangle=-25)
    return apply_chart_theme(chart, height=410)


apply_page_style()
apply_sales_styles()

render_page_header(
    "Sales Analytics",
    "Branch-wise sales performance, product movement, category mix, and inventory pressure from sales.csv.",
)

data = load_data_or_stop()
sales = data["sales"]
stores = data["stores"]
products = data["products"]
inventory = data.get("inventory", pd.DataFrame())

sales_view = build_sales_view(sales, stores, products)
if sales_view.empty:
    st.info("sales.csv is empty or unavailable.")
    st.stop()

options = branch_options(stores, sales_view)
branch_labels = ["All Branches"] + [item["label"] for item in options]
label_to_store = {item["label"]: item["store_id"] for item in options}

control_cols = st.columns([1.2, 1, 1], gap="medium")
with control_cols[0]:
    branch_scope = st.selectbox("Branch / store / city", branch_labels, index=0)
with control_cols[1]:
    trend_frequency = st.segmented_control(
        "Trend view",
        options=["Daily", "Weekly"],
        default="Daily",
    )
with control_cols[2]:
    trend_metric_label = st.segmented_control(
        "Trend metric",
        options=["Revenue", "Units"],
        default="Revenue",
    )

min_date = sales_view["date"].min().date()
max_date = sales_view["date"].max().date()

with st.expander("Filters", expanded=True):
    filter_cols = st.columns(4, gap="medium")
    with filter_cols[0]:
        selected_range = st.date_input(
            "Date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )
    with filter_cols[1]:
        available_categories = sorted(sales_view["category"].dropna().astype(str).unique().tolist())
        selected_categories = st.multiselect("Category", available_categories)
    with filter_cols[2]:
        product_lookup = (
            sales_view[["product_id", "product_name"]]
            .drop_duplicates()
            .sort_values("product_name")
        )
        product_labels = product_lookup["product_name"].tolist()
        selected_product_labels = st.multiselect("Product", product_labels)
    with filter_cols[3]:
        selected_filter_branches = st.multiselect(
            "Additional branch filter",
            [item["label"] for item in options],
            default=[],
        )

selected_store_ids: list[str] = []
if branch_scope != "All Branches":
    selected_store_ids.append(label_to_store[branch_scope])
if selected_filter_branches:
    selected_store_ids.extend(label_to_store[label] for label in selected_filter_branches)
selected_store_ids = sorted(set(selected_store_ids))

product_name_to_id = dict(zip(product_lookup["product_name"], product_lookup["product_id"]))
selected_product_ids = tuple(str(product_name_to_id[label]) for label in selected_product_labels)

if isinstance(selected_range, tuple) and len(selected_range) == 2:
    start_date, end_date = selected_range
else:
    start_date, end_date = min_date, max_date

filters = SalesFilters(
    store_ids=tuple(selected_store_ids),
    categories=tuple(selected_categories),
    product_ids=selected_product_ids,
    start_date=pd.Timestamp(start_date),
    end_date=pd.Timestamp(end_date),
)
filtered_sales = apply_sales_filters(sales_view, filters)

st.subheader("Executive Sales Overview")
render_sales_kpis(overview_metrics(filtered_sales, sales_view))

export_cols = st.columns([1, 1, 4], gap="medium")
with export_cols[0]:
    st.download_button(
        "Download CSV",
        data=filtered_sales.to_csv(index=False).encode("utf-8"),
        file_name="filtered_sales_export.csv",
        mime="text/csv",
        use_container_width=True,
    )
with export_cols[1]:
    st.caption(f"{len(filtered_sales):,} filtered rows")

st.divider()

st.subheader("Branch Analytics")
branch_df = branch_sales_summary(filtered_sales)
trend_metric = "sales_value" if trend_metric_label == "Revenue" else "quantity_sold"
trend_frequency_code = "W" if trend_frequency == "Weekly" else "D"
trend_df = trend_data(filtered_sales, frequency=trend_frequency_code)

branch_left, branch_right = st.columns(2, gap="large")
with branch_left:
    render_chart_card(
        "Branch-wise Total Sales",
        "Total revenue by branch for the selected filters.",
        make_branch_bar(branch_df),
        "No branch sales data is available for the selected filters.",
    )
with branch_right:
    render_chart_card(
        f"{trend_frequency} Sales Trend",
        "Revenue or unit movement over time for the selected branch scope.",
        make_trend_chart(trend_df, trend_metric),
        "No trend data is available for the selected filters.",
    )

st.subheader("Branch Comparison")
comparison_options = [item["label"] for item in options]
default_compare = comparison_options[:2] if len(comparison_options) >= 2 else comparison_options
selected_compare_labels = st.multiselect(
    "Compare branches",
    comparison_options,
    default=default_compare,
)
selected_compare_ids = tuple(label_to_store[label] for label in selected_compare_labels)
comparison_df = branch_comparison(sales_view, inventory, selected_compare_ids)
if comparison_df.empty:
    st.info("No branch comparison data is available.")
else:
    st.dataframe(
        comparison_df.rename(
            columns={
                "branch_label": "Branch",
                "total_sales": "Total Sales",
                "order_count": "Order Count",
                "top_product": "Top Product",
                "top_product_units": "Top Product Units",
                "low_stock_risk": "Low Stock Risk",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

st.divider()

st.subheader("Product Performance")
product_df = product_performance(filtered_sales, limit=10)
category_df = category_sales(filtered_sales)
inventory_sales_df = inventory_sales_comparison(filtered_sales, inventory, products)

product_left, product_right = st.columns(2, gap="large")
with product_left:
    render_chart_card(
        "Product-wise Sales",
        "Top products ranked by units sold in the selected branch scope.",
        make_product_chart(product_df),
        "No product sales data is available for the selected filters.",
    )
with product_right:
    render_chart_card(
        "Category-wise Sales Mix",
        "Revenue share by product category.",
        make_category_chart(category_df),
        "No category sales data is available for the selected filters.",
    )

render_chart_card(
    "Inventory vs Sales Comparison",
    "Current stock compared with recent sales movement for the selected branch scope.",
    make_inventory_sales_chart(inventory_sales_df),
    "No inventory comparison data is available for the selected filters.",
)

st.divider()

st.subheader("AI Insights")
insights = generate_sales_insights(filtered_sales, sales_view, comparison_df)
for insight in insights:
    st.info(insight)

with st.expander("Filtered sales records"):
    st.dataframe(filtered_sales, use_container_width=True, hide_index=True)
