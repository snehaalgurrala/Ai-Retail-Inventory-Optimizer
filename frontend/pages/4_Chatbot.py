from pathlib import Path
import sys

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from frontend.utils.page_helpers import apply_page_style  # noqa: E402

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


st.set_page_config(
    page_title="Chatbot",
    page_icon="C",
    layout="wide",
)

apply_page_style()


@st.cache_data
def load_csv(filename: str) -> pd.DataFrame:
    """Load one processed CSV file if it exists."""
    file_path = PROCESSED_DATA_DIR / filename
    if not file_path.exists():
        return pd.DataFrame()
    return pd.read_csv(file_path)


@st.cache_data
def load_chatbot_data() -> dict[str, pd.DataFrame]:
    """Load all datasets used by the rule-based assistant."""
    return {
        "current_inventory": load_csv("current_inventory.csv"),
        "sales_summary": load_csv("sales_summary.csv"),
        "store_inventory_summary": load_csv("store_inventory_summary.csv"),
        "low_stock_items": load_csv("low_stock_items.csv"),
        "dead_stock_candidates": load_csv("dead_stock_candidates.csv"),
        "overstock_items": load_csv("overstock_items.csv"),
        "high_demand_items": load_csv("high_demand_items.csv"),
        "slow_moving_items": load_csv("slow_moving_items.csv"),
        "stockout_risk_items": load_csv("stockout_risk_items.csv"),
        "recommendations": load_csv("recommendations.csv"),
    }


def top_rows(
    df: pd.DataFrame,
    sort_column: str | None = None,
    ascending: bool = False,
    limit: int = 10,
) -> pd.DataFrame:
    """Return a compact dataframe for display."""
    if df.empty:
        return df

    display_df = df.copy()
    if sort_column and sort_column in display_df.columns:
        display_df[sort_column] = pd.to_numeric(
            display_df[sort_column],
            errors="coerce",
        ).fillna(0)
        display_df = display_df.sort_values(sort_column, ascending=ascending)

    return display_df.head(limit)


def choose_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Keep only columns that exist in the dataframe."""
    if df.empty:
        return df
    available_columns = [column for column in columns if column in df.columns]
    return df[available_columns]


def recommendation_rows(
    recommendations: pd.DataFrame,
    recommendation_type: str,
) -> pd.DataFrame:
    if recommendations.empty or "recommendation_type" not in recommendations.columns:
        return pd.DataFrame()

    return recommendations[
        recommendations["recommendation_type"].astype(str).eq(recommendation_type)
    ].copy()


def answer_with_dataset(
    title: str,
    df: pd.DataFrame,
    empty_message: str,
    sort_column: str | None = None,
    ascending: bool = False,
    limit: int = 10,
) -> tuple[str, pd.DataFrame]:
    """Build a count-based answer with a supporting table."""
    if df.empty:
        return empty_message, pd.DataFrame()

    result = top_rows(df, sort_column, ascending, limit)
    answer = f"{title}: I found {len(df):,} matching rows in the project data."
    return answer, result


def best_sales_suggestions(data: dict[str, pd.DataFrame]) -> tuple[str, pd.DataFrame]:
    """Suggest sales actions from recommendation and trend outputs only."""
    recommendations = data["recommendations"]
    high_demand = data["high_demand_items"]
    low_stock = data["low_stock_items"]
    discount_rows = recommendation_rows(recommendations, "discount")
    clearance_rows = recommendation_rows(recommendations, "clearance")
    stockout_rows = recommendation_rows(recommendations, "stockout_prevention_alert")

    points = []
    if not high_demand.empty:
        points.append(
            f"Keep high-demand products available; {len(high_demand):,} product-store rows have strong recent sales velocity."
        )
    if not low_stock.empty:
        points.append(
            f"Replenish low-stock products before promotion; {len(low_stock):,} rows are at or below reorder threshold."
        )
    if not discount_rows.empty:
        points.append(
            f"Use targeted discounts for slow or overstocked items; {len(discount_rows):,} discount recommendations exist."
        )
    if not clearance_rows.empty:
        points.append(
            f"Prioritize clearance where shelf-life or weak movement creates pressure; {len(clearance_rows):,} clearance recommendations exist."
        )
    if not stockout_rows.empty:
        points.append(
            f"Protect sales by acting on stockout risks; {len(stockout_rows):,} stockout prevention alerts exist."
        )

    if not points:
        return (
            "I do not have enough processed recommendation data to suggest sales actions.",
            pd.DataFrame(),
        )

    table = pd.concat(
        [
            high_demand.assign(signal="high demand"),
            low_stock.assign(signal="low stock"),
        ],
        ignore_index=True,
    )
    table = choose_columns(
        top_rows(table, "recent_daily_sales_velocity", ascending=False, limit=10),
        [
            "signal",
            "product_id",
            "product_name",
            "store_id",
            "store_name",
            "stock_level",
            "recent_daily_sales_velocity",
            "days_of_stock_remaining",
            "evidence",
        ],
    )

    return " ".join(points), table


def least_sold_marketing_suggestions(
    data: dict[str, pd.DataFrame],
) -> tuple[str, pd.DataFrame]:
    """Suggest marketing actions for least-sold products using sales data."""
    sales_summary = data["sales_summary"]
    recommendations = data["recommendations"]
    discount_rows = recommendation_rows(recommendations, "discount")
    clearance_rows = recommendation_rows(recommendations, "clearance")

    if sales_summary.empty or "product_id" not in sales_summary.columns:
        return (
            "I cannot identify least-sold items because sales summary data is missing.",
            pd.DataFrame(),
        )

    product_sales = sales_summary.copy()
    product_sales["total_quantity_sold"] = pd.to_numeric(
        product_sales.get("total_quantity_sold", 0),
        errors="coerce",
    ).fillna(0)
    product_sales = (
        product_sales.groupby("product_id", as_index=False)
        .agg(total_quantity_sold=("total_quantity_sold", "sum"))
        .sort_values("total_quantity_sold", ascending=True)
    )

    inventory = data["current_inventory"]
    if not inventory.empty:
        product_names = inventory[["product_id", "product_name"]].drop_duplicates()
        product_sales = product_sales.merge(product_names, on="product_id", how="left")

    least_sold = product_sales.head(10)
    points = [
        "Focus marketing on the least-sold products shown in the table.",
        "Use small discounts or bundles only where discount or clearance recommendations already support that action.",
        "Avoid promoting items that are low in stock unless replenishment is planned first.",
    ]

    supporting_recommendations = pd.concat(
        [discount_rows, clearance_rows],
        ignore_index=True,
    )
    if not supporting_recommendations.empty:
        least_product_ids = set(least_sold["product_id"].astype(str))
        supporting_recommendations = supporting_recommendations[
            supporting_recommendations["product_id"].astype(str).isin(
                least_product_ids
            )
        ]
        if not supporting_recommendations.empty:
            points.append(
                f"{len(supporting_recommendations):,} existing discount or clearance recommendations match these least-sold products."
            )

    return " ".join(points), choose_columns(
        least_sold,
        ["product_id", "product_name", "total_quantity_sold"],
    )


def answer_question(question: str, data: dict[str, pd.DataFrame]) -> tuple[str, pd.DataFrame]:
    """Answer supported questions using project data only."""
    text = question.lower().strip()

    if not text:
        return "Ask a question about inventory, sales movement, suppliers, or recommendations.", pd.DataFrame()

    if "low stock" in text or "reorder" in text:
        answer, table = answer_with_dataset(
            "Low stock items",
            data["low_stock_items"],
            "No low stock items were found in the processed analyzer output.",
            sort_column="days_of_stock_remaining",
            ascending=True,
        )
        return answer, choose_columns(
            table,
            [
                "product_id",
                "product_name",
                "store_id",
                "store_name",
                "stock_level",
                "effective_reorder_threshold",
                "days_of_stock_remaining",
                "evidence",
            ],
        )

    if "dead stock" in text or "deadstock" in text:
        answer, table = answer_with_dataset(
            "Dead stock candidates",
            data["dead_stock_candidates"],
            "No dead stock candidates were found in the processed analyzer output.",
            sort_column="days_of_stock_remaining",
            ascending=False,
        )
        return answer, choose_columns(
            table,
            [
                "product_id",
                "product_name",
                "store_id",
                "store_name",
                "stock_level",
                "recent_30_day_quantity_sold",
                "shelf_life_days",
                "days_of_stock_remaining",
                "evidence",
            ],
        )

    if "which store" in text and ("excess" in text or "capacity" in text):
        answer, table = answer_with_dataset(
            "Store inventory capacity summary",
            data["store_inventory_summary"],
            "Store inventory summary data is not available.",
            sort_column="capacity_used_percent",
            ascending=False,
        )
        return answer, choose_columns(
            table,
            [
                "store_id",
                "store_name",
                "city",
                "capacity",
                "total_inventory",
                "remaining_capacity",
                "capacity_used_percent",
            ],
        )

    if "excess stock" in text or "overstock" in text or "over stock" in text:
        answer, table = answer_with_dataset(
            "Overstock or excess stock items",
            data["overstock_items"],
            "No overstock items were found in the processed analyzer output.",
            sort_column="days_of_stock_remaining",
            ascending=False,
        )
        return answer, choose_columns(
            table,
            [
                "product_id",
                "product_name",
                "store_id",
                "store_name",
                "stock_level",
                "effective_reorder_threshold",
                "days_of_stock_remaining",
                "evidence",
            ],
        )

    if "transfer" in text:
        transfer_rows = recommendation_rows(data["recommendations"], "stock_transfer")
        answer, table = answer_with_dataset(
            "Stock transfer recommendations",
            transfer_rows,
            "No stock transfer recommendations were found.",
        )
        return answer, choose_columns(
            table,
            [
                "recommendation_id",
                "product_id",
                "product_name",
                "store_id",
                "priority",
                "action",
                "reason",
                "evidence",
                "suggested_quantity",
                "status",
            ],
        )

    if "high demand" in text or "fast moving" in text or "best selling" in text:
        answer, table = answer_with_dataset(
            "High demand products",
            data["high_demand_items"],
            "No high demand items were found in the processed analyzer output.",
            sort_column="recent_daily_sales_velocity",
            ascending=False,
        )
        return answer, choose_columns(
            table,
            [
                "product_id",
                "product_name",
                "store_id",
                "store_name",
                "recent_7_day_quantity_sold",
                "recent_30_day_quantity_sold",
                "recent_daily_sales_velocity",
                "evidence",
            ],
        )

    if "slow moving" in text or "least sold" in text or "least selling" in text:
        return least_sold_marketing_suggestions(data)

    if "supplier" in text and ("risk" in text or "risky" in text):
        supplier_rows = recommendation_rows(
            data["recommendations"],
            "supplier_risk_alert",
        )
        answer, table = answer_with_dataset(
            "Supplier risk alerts",
            supplier_rows,
            "No supplier risk alerts were found in recommendations.",
        )
        return answer, choose_columns(
            table,
            [
                "recommendation_id",
                "product_id",
                "product_name",
                "priority",
                "action",
                "reason",
                "evidence",
                "status",
            ],
        )

    if (
        "increase" in text
        and "sales" in text
        or "improve" in text
        and "sales" in text
        or "suggest" in text
        or "suggestion" in text
    ):
        return best_sales_suggestions(data)

    if "stockout" in text or "stock out" in text:
        rows = data["stockout_risk_items"]
        answer, table = answer_with_dataset(
            "Stockout risk items",
            rows,
            "No stockout risk items were found in the processed analyzer output.",
            sort_column="days_of_stock_remaining",
            ascending=True,
        )
        return answer, choose_columns(
            table,
            [
                "product_id",
                "product_name",
                "store_id",
                "store_name",
                "stock_level",
                "recent_daily_sales_velocity",
                "days_of_stock_remaining",
                "evidence",
            ],
        )

    return (
        "I can only answer from the project data available in data/processed. Try asking about low stock, dead stock, overstock, transfers, high demand products, supplier risk, stockout risk, or sales suggestions.",
        pd.DataFrame(),
    )


SAMPLE_QUESTIONS = [
    "Which items are low in stock?",
    "What are the dead stock items?",
    "Which store has excess stock?",
    "Which store needs transfer?",
    "Which products are high demand?",
    "Which suppliers are risky?",
    "What is the best way to increase sales based on our trends?",
    "What marketing can help sell our least sold items?",
]


st.title("💬 Chatbot")
st.caption("Rule-based project assistant grounded only in processed retail data.")

data = load_chatbot_data()

missing_outputs = [
    name for name, df in data.items() if df.empty and name != "recommendations"
]
if missing_outputs:
    st.warning(
        "Some processed outputs are empty or missing. Run the processing, analyzer, and agent services for fuller answers."
    )

with st.sidebar:
    st.header("Sample Questions")
    for sample_question in SAMPLE_QUESTIONS:
        if st.button(sample_question, use_container_width=True):
            st.session_state["chatbot_question"] = sample_question

question = st.chat_input("Ask about inventory, recommendations, suppliers, or sales trends")

if "chatbot_question" not in st.session_state:
    st.session_state["chatbot_question"] = ""

if question:
    st.session_state["chatbot_question"] = question

if not st.session_state["chatbot_question"]:
    st.subheader("💡 Ask a Data Question")
    st.write(
        "Use the sample questions in the sidebar or ask your own project question. Answers are generated from the processed CSV files only."
    )
else:
    user_question = st.session_state["chatbot_question"]
    with st.chat_message("user"):
        st.write(user_question)

    answer, table = answer_question(user_question, data)
    with st.chat_message("assistant"):
        st.write(answer)
        if not table.empty:
            st.dataframe(table, use_container_width=True, hide_index=True)
        else:
            st.caption("No supporting table is available for this answer.")
