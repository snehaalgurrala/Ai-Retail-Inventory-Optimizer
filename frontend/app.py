import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from backend.utils.data_loader import load_all_data  # noqa: E402
from backend.agents.orchestrator_agent import run_agent_graph  # noqa: E402
from frontend.components.ui_components import (  # noqa: E402
    render_agent_status_card,
    render_kpi_card,
    render_orchestrator_summary_card,
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


@st.cache_data
def load_agent_dashboard_outputs() -> dict[str, pd.DataFrame]:
    """Load the saved agent and orchestrator outputs used on the home dashboard."""
    return {
        "agent_outputs": load_processed_output("agent_outputs.csv"),
        "orchestrator_summary": load_processed_output("orchestrator_summary.csv"),
        "recommendations": load_processed_output("recommendations.csv"),
    }


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


def processed_file_path(filename: str) -> Path:
    return PROJECT_ROOT / "data" / "processed" / filename


def processed_files_exist(filenames: list[str]) -> bool:
    return all(processed_file_path(filename).exists() for filename in filenames)


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


def highest_priority_label(df: pd.DataFrame) -> str:
    if df.empty or "priority" not in df.columns:
        return "Info"

    priorities = df["priority"].fillna("").astype(str).str.lower()
    if priorities.eq("high").any():
        return "High"
    if priorities.eq("medium").any():
        return "Medium"
    if priorities.eq("low").any():
        return "Low"
    return "Info"


def top_reason(df: pd.DataFrame) -> str:
    if df.empty or "reason" not in df.columns:
        return ""

    reasons = df["reason"].fillna("").astype(str)
    reasons = reasons[reasons != ""]
    if reasons.empty:
        return ""
    return reasons.iloc[0]


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
        processed_file_path("low_stock_items.csv"),
        processed_file_path("stockout_risk_items.csv"),
        processed_file_path("overstock_items.csv"),
        processed_file_path("dead_stock_candidates.csv"),
    ]
    available_count = sum(path.exists() for path in required_files)
    total_count = len(required_files)
    if available_count == total_count:
        return f"Healthy ({available_count}/{total_count})"
    if available_count >= total_count - 2:
        return f"Watch ({available_count}/{total_count})"
    return f"Needs Attention ({available_count}/{total_count})"


def build_orchestrator_summary(
    recommendations: pd.DataFrame,
    low_stock_df: pd.DataFrame,
    stockout_risk_df: pd.DataFrame,
    overstock_df: pd.DataFrame,
    dead_stock_df: pd.DataFrame,
) -> str:
    if (
        recommendations.empty
        and low_stock_df.empty
        and stockout_risk_df.empty
        and overstock_df.empty
        and dead_stock_df.empty
    ):
        return "Run agents to generate latest analysis."

    summary_parts = []
    if not low_stock_df.empty:
        summary_parts.append(f"{len(low_stock_df):,} low stock items")
    if not stockout_risk_df.empty:
        summary_parts.append(f"{len(stockout_risk_df):,} stockout risks")
    if not overstock_df.empty:
        summary_parts.append(f"{len(overstock_df):,} overstock rows")
    if not dead_stock_df.empty:
        summary_parts.append(f"{len(dead_stock_df):,} dead stock candidates")
    if not recommendations.empty:
        summary_parts.append(f"{len(recommendations):,} active recommendations")

    return ". ".join(summary_parts) + "."


def agent_rows(recommendations: pd.DataFrame, source_agent: str) -> pd.DataFrame:
    if recommendations.empty or "source_agent" not in recommendations.columns:
        return pd.DataFrame()

    return recommendations[
        recommendations["source_agent"].fillna("").astype(str).eq(source_agent)
    ].copy()


def build_inventory_agent_card(
    low_stock_df: pd.DataFrame,
    high_demand_df: pd.DataFrame,
    slow_moving_df: pd.DataFrame,
    stockout_risk_df: pd.DataFrame,
) -> dict[str, str | int]:
    finding_count = (
        len(low_stock_df)
        + len(high_demand_df)
        + len(slow_moving_df)
        + len(stockout_risk_df)
    )

    if finding_count == 0:
        return {
            "agent_name": "Inventory Analysis Agent",
            "description": (
                "Analyzes demand trends, slow and fast movers, stock movement, "
                "and root causes."
            ),
            "latest_finding_count": 0,
            "priority_level": "Info",
            "latest_insight": "Run agents to generate latest analysis.",
        }

    insight_parts = []
    if not high_demand_df.empty:
        insight_parts.append(f"{len(high_demand_df):,} high demand items")
    if not slow_moving_df.empty:
        insight_parts.append(f"{len(slow_moving_df):,} slow moving items")
    if not low_stock_df.empty:
        insight_parts.append(f"{len(low_stock_df):,} low stock items")

    priority = "High" if not stockout_risk_df.empty or not low_stock_df.empty else "Medium"
    return {
        "agent_name": "Inventory Analysis Agent",
        "description": (
            "Analyzes demand trends, slow and fast movers, stock movement, "
            "and root causes."
        ),
        "latest_finding_count": finding_count,
        "priority_level": priority,
        "latest_insight": ". ".join(insight_parts[:3]) + ".",
    }


def build_recommendation_agent_card(
    agent_name: str,
    description: str,
    recommendations: pd.DataFrame,
) -> dict[str, str | int]:
    if recommendations.empty:
        return {
            "agent_name": agent_name,
            "description": description,
            "latest_finding_count": 0,
            "priority_level": "Info",
            "latest_insight": "Run agents to generate latest analysis.",
        }

    insight = top_reason(recommendations) or first_action(recommendations)
    return {
        "agent_name": agent_name,
        "description": description,
        "latest_finding_count": len(recommendations),
        "priority_level": highest_priority_label(recommendations),
        "latest_insight": insight or "Latest recommendations are ready for review.",
    }


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


def get_agent_output_row(agent_outputs: pd.DataFrame, agent_name: str) -> pd.Series | None:
    if agent_outputs.empty or "agent_name" not in agent_outputs.columns:
        return None

    matching_rows = agent_outputs[
        agent_outputs["agent_name"].fillna("").astype(str).eq(agent_name)
    ]
    if matching_rows.empty:
        return None
    return matching_rows.iloc[0]


def build_agent_card_from_output(
    agent_outputs: pd.DataFrame,
    agent_key: str,
    fallback_name: str,
    description: str,
) -> dict[str, str | int]:
    agent_row = get_agent_output_row(agent_outputs, agent_key)
    if agent_row is None:
        return {
            "agent_name": fallback_name,
            "description": description,
            "latest_finding_count": 0,
            "priority_level": "Info",
            "latest_insight": "Run agents to generate latest analysis.",
        }

    return {
        "agent_name": fallback_name,
        "description": description,
        "latest_finding_count": int(
            pd.to_numeric(agent_row.get("finding_count", 0), errors="coerce") or 0
        ),
        "priority_level": str(agent_row.get("priority_level", "Info")).title(),
        "latest_insight": str(
            agent_row.get("latest_insight", "Run agents to generate latest analysis.")
        ),
    }


apply_page_style()

refresh_message = st.session_state.pop("dashboard_refresh_message", "")
agent_output_files = [
    "agent_outputs.csv",
    "orchestrator_summary.csv",
    "recommendations.csv",
]

with st.container():
    st.title("AI Retail Inventory Optimizer")
    st.caption(
        "A data-grounded inventory command center for sales movement, current "
        "stock health, and explainable recommendation workflows."
    )

if refresh_message:
    st.success(refresh_message)

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
orchestrator_summary_df = agent_output_state["orchestrator_summary"]
recommendations = agent_output_state["recommendations"]
has_agent_run = processed_files_exist(agent_output_files)

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
low_stock_items = load_processed_output("low_stock_items.csv")
stockout_risk_items = load_processed_output("stockout_risk_items.csv")
overstock_items = load_processed_output("overstock_items.csv")
dead_stock_items = load_processed_output("dead_stock_candidates.csv")
high_demand_items = load_processed_output("high_demand_items.csv")
slow_moving_items = load_processed_output("slow_moving_items.csv")
last_agent_run_time = latest_timestamp_from_files(agent_output_files)
last_updated_timestamp = latest_timestamp_iso_from_files(agent_output_files)

st.subheader("Agent Command Center")
st.caption(
    "A shared view of orchestrator health and the most recent findings from each "
    "specialist agent."
)

if has_agent_run and not orchestrator_summary_df.empty:
    orchestrator_row = orchestrator_summary_df.iloc[0]
    render_orchestrator_summary_card(
        title="Orchestrator Agent",
        database_health=str(orchestrator_row.get("database_health", database_health_summary())),
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
        last_run_time=str(orchestrator_row.get("last_agent_run_time", last_agent_run_time)),
        summary=str(
            orchestrator_row.get(
                "summary",
                "Run agents to generate latest analysis.",
            )
        ),
    )
else:
    st.info('No agent run found. Click Run / Refresh Agents.')

refresh_left, refresh_right = st.columns([1, 5], gap="large")
with refresh_left:
    if st.button("Run / Refresh Agents", use_container_width=True):
        try:
            with st.spinner("Running orchestrator and refreshing all agent outputs..."):
                final_state = run_agent_graph(save_output=True)
                load_dashboard_data.clear()
                load_agent_dashboard_outputs.clear()
                st.cache_data.clear()
            refreshed_count = len(final_state.get("unified_recommendations", []))
            refreshed_time = final_state.get("combined_output", {}).get(
                "run_time",
                "just now",
            )
            st.session_state["dashboard_refresh_message"] = (
                f"Agents refreshed successfully. {refreshed_count:,} recommendations "
                f"generated at {refreshed_time}."
            )
            st.rerun()
        except Exception as error:
            st.error("Could not refresh agents.")
            st.exception(error)
with refresh_right:
    st.caption(
        "This runs the orchestrator across inventory, pricing, transfer, risk, "
        "and procurement agents using the latest CSV data, then rewrites the "
        "dashboard output files."
    )

if last_updated_timestamp:
    st.caption(f"Last updated: {last_updated_timestamp}")

inventory_agent_card = build_agent_card_from_output(
    agent_outputs,
    "inventory_agent",
    "Inventory Analysis Agent",
    "Analyzes demand trends, slow/fast movers, stock movement, and root causes.",
)
pricing_agent_card = build_agent_card_from_output(
    agent_outputs,
    "pricing_agent",
    "Pricing Agent",
    "Analyzes discount opportunities, markdowns, bundling, and pricing strategy.",
)
transfer_agent_card = build_agent_card_from_output(
    agent_outputs,
    "transfer_agent",
    "Transfer / Supply Agent",
    "Analyzes stock imbalance, transfer vs reorder, and source/target store decisions.",
)
risk_agent_card = build_agent_card_from_output(
    agent_outputs,
    "risk_agent",
    "Risk Agent",
    "Analyzes stockout risk, overstock risk, and supplier delay risk.",
)
procurement_agent_card = build_agent_card_from_output(
    agent_outputs,
    "procurement_agent",
    "Procurement Agent",
    "Analyzes purchase quantity, reorder timing, and vendor suggestions.",
)

agent_cards = [
    inventory_agent_card,
    pricing_agent_card,
    transfer_agent_card,
    risk_agent_card,
    procurement_agent_card,
]

for row_start in range(0, len(agent_cards), 3):
    card_columns = st.columns(3, gap="large")
    for index, agent_card in enumerate(agent_cards[row_start:row_start + 3]):
        with card_columns[index]:
            render_agent_status_card(
                agent_name=str(agent_card["agent_name"]),
                description=str(agent_card["description"]),
                latest_finding_count=agent_card["latest_finding_count"],
                priority_level=str(agent_card["priority_level"]),
                latest_insight=str(agent_card["latest_insight"]),
            )

st.divider()

st.subheader("Latest Recommendations")
st.caption("Latest recommendation output from the most recent orchestrator run.")

if not has_agent_run or recommendations.empty:
    st.info('No agent run found. Click Run / Refresh Agents.')
else:
    latest_recommendations = recommendations.sort_values(
        ["priority", "recommendation_id"],
        ascending=[True, True],
    ).head(12)
    st.dataframe(
        latest_recommendations[
            [
                column
                for column in [
                    "recommendation_id",
                    "recommendation_type",
                    "product_name",
                    "store_id",
                    "priority",
                    "action",
                    "reason",
                    "evidence",
                    "source_agent",
                    "status",
                ]
                if column in latest_recommendations.columns
            ]
        ],
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

st.divider()
