import sys
import importlib
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from backend.agents.orchestrator_agent import run_agent_graph  # noqa: E402
from backend.services.email_service import send_low_stock_alert_email  # noqa: E402
from backend.services.low_stock_service import get_low_stock_items  # noqa: E402
from backend.services import agent_summary_service  # noqa: E402
from backend.utils.data_loader import load_all_data  # noqa: E402
from frontend.components.ui_components import (  # noqa: E402
    apply_command_center_styles,
    render_agent_command_card,
    render_command_center_orchestrator_card,
    render_kpi_card,
    render_low_stock_alert_card,
)
from frontend.utils.page_helpers import (  # noqa: E402
    apply_page_style,
    render_chart_card,
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


def _get_agent_summary_service():
    """Reload the summary service safely during Streamlit hot reloads."""
    return importlib.reload(agent_summary_service)


def processed_file_path(filename: str) -> Path:
    return PROJECT_ROOT / "data" / "processed" / filename


def load_processed_output(filename: str) -> pd.DataFrame:
    file_path = processed_file_path(filename)
    if not file_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(file_path)
    except Exception:
        return pd.DataFrame()


def processed_files_exist(filenames: list[str]) -> bool:
    return all(processed_file_path(filename).exists() for filename in filenames)


@st.cache_data
def load_agent_dashboard_outputs() -> dict[str, pd.DataFrame]:
    if processed_files_exist(["agent_outputs.csv", "orchestrator_summary.csv"]):
        service = _get_agent_summary_service()
        ensure_fn = getattr(service, "ensure_agent_card_summaries", None)
        if callable(ensure_fn):
            ensure_fn()
    return {
        "agent_outputs": load_processed_output("agent_outputs.csv"),
        "agent_card_summaries": load_processed_output("agent_card_summaries.csv"),
        "orchestrator_summary": load_processed_output("orchestrator_summary.csv"),
        "recommendations": load_processed_output("recommendations.csv"),
    }


def safe_sum(df: pd.DataFrame, column: str) -> int:
    if df.empty or column not in df.columns:
        return 0
    return int(pd.to_numeric(df[column], errors="coerce").fillna(0).sum())


def processed_row_count(filename: str) -> int:
    file_path = processed_file_path(filename)
    if not file_path.exists():
        return 0
    try:
        return len(pd.read_csv(file_path))
    except Exception:
        return 0


def latest_timestamp_from_files(filenames: list[str]) -> str:
    latest_time = None
    for filename in filenames:
        file_path = processed_file_path(filename)
        if file_path.exists():
            modified_time = datetime.fromtimestamp(file_path.stat().st_mtime)
            if latest_time is None or modified_time > latest_time:
                latest_time = modified_time
    if latest_time is None:
        return "No agent run recorded yet"
    return latest_time.strftime("%d %b %Y, %I:%M %p")


def latest_timestamp_iso_from_files(filenames: list[str]) -> str:
    latest_time = None
    for filename in filenames:
        file_path = processed_file_path(filename)
        if file_path.exists():
            modified_time = datetime.fromtimestamp(file_path.stat().st_mtime)
            if latest_time is None or modified_time > latest_time:
                latest_time = modified_time
    if latest_time is None:
        return ""
    return latest_time.isoformat(timespec="seconds")


def database_health_summary() -> str:
    required_files = [
        PROJECT_ROOT / "data" / "raw" / "inventory.csv",
        PROJECT_ROOT / "data" / "raw" / "products.csv",
        PROJECT_ROOT / "data" / "raw" / "sales.csv",
        PROJECT_ROOT / "data" / "raw" / "stores.csv",
        PROJECT_ROOT / "data" / "raw" / "suppliers.csv",
        PROJECT_ROOT / "data" / "raw" / "transactions.csv",
        processed_file_path("recommendations.csv"),
        processed_file_path("agent_outputs.csv"),
        processed_file_path("orchestrator_summary.csv"),
    ]
    available_count = sum(path.exists() for path in required_files)
    total_count = len(required_files)
    if available_count == total_count:
        return f"Healthy ({available_count}/{total_count})"
    if available_count >= total_count - 2:
        return f"Watch ({available_count}/{total_count})"
    return f"Needs Attention ({available_count}/{total_count})"


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


def latest_recommendations_table(recommendations: pd.DataFrame) -> pd.DataFrame:
    if recommendations.empty:
        return pd.DataFrame()

    latest = recommendations.copy()
    if "priority" in latest.columns:
        latest["_priority_rank"] = (
            latest["priority"]
            .fillna("")
            .astype(str)
            .str.lower()
            .map({"high": 0, "medium": 1, "low": 2})
            .fillna(3)
        )
        latest = latest.sort_values(["_priority_rank", "recommendation_id"])

    columns = [
        column
        for column in [
            "recommendation_id",
            "recommendation_type",
            "product_name",
            "store_id",
            "priority",
            "action",
            "reason",
            "source_agent",
            "status",
        ]
        if column in latest.columns
    ]
    return latest.head(10)[columns]


apply_page_style()
apply_command_center_styles()

refresh_message = st.session_state.pop("dashboard_refresh_message", "")
refresh_email_message = st.session_state.pop("dashboard_email_message", "")
refresh_email_warning = st.session_state.pop("dashboard_email_warning", "")
agent_output_files = [
    "agent_outputs.csv",
    "agent_card_summaries.csv",
    "orchestrator_summary.csv",
    "recommendations.csv",
]

if refresh_message:
    st.success(refresh_message)
if refresh_email_message:
    st.info(refresh_email_message)
if refresh_email_warning:
    st.warning(refresh_email_warning)

try:
    data = load_dashboard_data()
except Exception as error:
    st.error("Could not load the dashboard data.")
    st.exception(error)
    st.stop()

products = data["products"]
sales = data["sales"]
inventory = data["inventory"]
transactions = data["transactions"]

agent_output_state = load_agent_dashboard_outputs()
agent_outputs = agent_output_state["agent_outputs"]
agent_card_summaries = agent_output_state["agent_card_summaries"]
orchestrator_summary_df = agent_output_state["orchestrator_summary"]
recommendations = agent_output_state["recommendations"]
low_stock_alerts = get_low_stock_items(save_output=True)
summary_service = _get_agent_summary_service()
build_low_stock_alert_text = getattr(
    summary_service,
    "build_low_stock_alert_text",
    lambda df: "No critical low-stock alerts right now.",
)

has_agent_run = processed_files_exist(
    ["agent_outputs.csv", "orchestrator_summary.csv", "recommendations.csv"]
)

if has_agent_run and agent_card_summaries.empty:
    generate_fn = getattr(summary_service, "generate_agent_card_summaries", None)
    if callable(generate_fn):
        agent_card_summaries, orchestrator_summary_df = generate_fn(
            agent_outputs_df=agent_outputs,
            recommendations_df=recommendations,
            orchestrator_summary_df=orchestrator_summary_df,
            save_output=True,
        )

current_inventory_quantity, inventory_source = get_current_inventory_quantity(
    inventory,
    transactions,
)
total_sales_quantity = safe_sum(sales, "quantity_sold")
dead_stock_count = processed_row_count("dead_stock_candidates.csv")
last_agent_run_time = latest_timestamp_from_files(agent_output_files)
last_updated_timestamp = latest_timestamp_iso_from_files(agent_output_files)

low_stock_count = int(len(low_stock_alerts))

header_left, header_right = st.columns([4.8, 1.2], gap="large")
with header_left:
    st.markdown(
        '<div class="command-header-title">Agent Command Center</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="command-header-subtitle">A premium view of orchestrator health, specialist agent updates, and the latest actions worth taking.</div>',
        unsafe_allow_html=True,
    )
with header_right:
    st.markdown(
        f'<div class="command-meta">Last run: {last_agent_run_time}</div>',
        unsafe_allow_html=True,
    )
    if st.button("Run / Refresh Agents", use_container_width=True):
        try:
            with st.spinner("Refreshing all agents on the latest data..."):
                final_state = run_agent_graph(save_output=True)
                generate_fn = getattr(summary_service, "generate_agent_card_summaries", None)
                if callable(generate_fn):
                    generate_fn(save_output=True)
                refreshed_low_stock_df = get_low_stock_items(save_output=True)
                email_result = send_low_stock_alert_email(refreshed_low_stock_df)
                load_dashboard_data.clear()
                load_agent_dashboard_outputs.clear()
                st.cache_data.clear()
            refreshed_count = len(final_state.get("unified_recommendations", []))
            refreshed_time = final_state.get("combined_output", {}).get(
                "run_time",
                "just now",
            )
            st.session_state["dashboard_refresh_message"] = (
                f"Agents refreshed successfully. {refreshed_count:,} recommendations generated at {refreshed_time}."
            )
            if email_result.get("warning"):
                st.session_state["dashboard_email_warning"] = str(
                    email_result.get("warning", "")
                )
            elif email_result.get("message"):
                st.session_state["dashboard_email_message"] = str(
                    email_result.get("message", "")
                )
            st.rerun()
        except Exception as error:
            st.error("Could not refresh agents.")
            st.exception(error)

if has_agent_run and not orchestrator_summary_df.empty:
    orchestrator_row = orchestrator_summary_df.iloc[0]
    render_command_center_orchestrator_card(
        database_health=str(
            orchestrator_row.get("database_health", database_health_summary())
        ),
        total_recommendations=int(
            pd.to_numeric(
                orchestrator_row.get("total_recommendations", 0),
                errors="coerce",
            )
            or 0
        ),
        high_priority_alerts=int(
            pd.to_numeric(
                orchestrator_row.get("high_priority_alerts", 0),
                errors="coerce",
            )
            or 0
        ),
        last_run_time=str(
            orchestrator_row.get("last_agent_run_time", last_agent_run_time)
        ),
        low_stock_alert=str(
            orchestrator_row.get(
                "low_stock_alert",
                build_low_stock_alert_text(low_stock_alerts),
            )
        ),
        top_risk=str(
            orchestrator_row.get(
                "top_risk",
                "No major risk stands out in the latest run.",
            )
        ),
        top_opportunity=str(
            orchestrator_row.get(
                "top_opportunity",
                "No standout commercial opportunity is available yet.",
            )
        ),
        executive_summary=str(
            orchestrator_row.get(
                "executive_summary",
                orchestrator_row.get(
                    "summary",
                    "Run agents to generate latest analysis.",
                ),
            )
        ),
        executive_recommendation=str(
            orchestrator_row.get(
                "executive_recommendation",
                "Review the highest-priority risks first, then move on the strongest commercial opportunity.",
            )
        ),
        summary_source=str(orchestrator_row.get("summary_source", "")),
    )
else:
    st.info('No agent run found. Click Run / Refresh Agents.')

if last_updated_timestamp:
    st.caption(f"Last updated: {last_updated_timestamp}")

st.caption(
    "Each specialist card shows the latest summarized insight, urgency, and next action from the newest agent run."
)

agent_order = [
    "inventory_agent",
    "pricing_agent",
    "transfer_agent",
    "risk_agent",
    "procurement_agent",
]
agent_lookup: dict[str, pd.Series] = {}
if not agent_card_summaries.empty and "agent_name" in agent_card_summaries.columns:
    for _, row in agent_card_summaries.iterrows():
        agent_lookup[str(row.get("agent_name", ""))] = row

agent_columns = st.columns(5, gap="medium")
for index, agent_key in enumerate(agent_order):
    with agent_columns[index]:
        row = agent_lookup.get(agent_key)
        if row is None:
            render_agent_command_card(
                agent_name=agent_key.replace("_", " ").title(),
                role_label="Agent summary",
                finding_count=0,
                priority_level="Info",
                summary="Run agents to generate the latest analysis.",
                recommended_action="Refresh the agents to populate this card.",
                accent="blue",
            )
        else:
            render_agent_command_card(
                agent_name=str(row.get("display_name", agent_key.replace("_", " ").title())),
                role_label=str(row.get("role_label", "Agent summary")),
                finding_count=int(
                    pd.to_numeric(row.get("finding_count", 0), errors="coerce") or 0
                ),
                priority_level=str(row.get("priority_level", "Info")),
                summary=str(
                    row.get("summary", "Run agents to generate the latest analysis.")
                ),
                recommended_action=str(
                    row.get("recommended_action", "Review the latest rows first.")
                ),
                accent=str(row.get("accent", "blue")),
            )

st.divider()

st.subheader("🚨 Low Stock Alerts")
if low_stock_alerts.empty:
    st.success("No critical low-stock alerts right now.")
else:
    render_low_stock_alert_card(build_low_stock_alert_text(low_stock_alerts))
    with st.container(border=True):
        preview_df = low_stock_alerts[
            [
                column
                for column in [
                    "product_name",
                    "store_name",
                    "city",
                    "current_quantity",
                    "reorder_threshold",
                    "suggested_reorder_quantity",
                    "priority",
                ]
                if column in low_stock_alerts.columns
            ]
        ].head(5)
        st.dataframe(
            preview_df,
            use_container_width=True,
            hide_index=True,
        )

st.divider()

st.subheader("Latest Recommendations")
st.caption("The newest recommendation queue from the latest orchestrator run.")

if not has_agent_run or recommendations.empty:
    st.info('No agent run found. Click Run / Refresh Agents.')
else:
    with st.container(border=True):
        st.dataframe(
            latest_recommendations_table(recommendations),
            use_container_width=True,
            hide_index=True,
        )

st.divider()

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
