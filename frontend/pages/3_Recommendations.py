from datetime import datetime
from pathlib import Path
import sys

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from frontend.utils.page_helpers import apply_page_style  # noqa: E402

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
RECOMMENDATIONS_FILE = PROCESSED_DATA_DIR / "recommendations.csv"
DECISIONS_FILE = PROCESSED_DATA_DIR / "recommendation_decisions.csv"

DECISION_COLUMNS = [
    "recommendation_id",
    "decision",
    "decided_at",
    "recommendation_type",
    "product_id",
    "product_name",
    "store_id",
]


st.set_page_config(
    page_title="Recommendations",
    page_icon="R",
    layout="wide",
)

apply_page_style()


@st.cache_data
def load_recommendations() -> pd.DataFrame:
    """Load recommendations, generating them once if the CSV is missing."""
    if not RECOMMENDATIONS_FILE.exists():
        from backend.agents.orchestrator_agent import run_all_agents

        run_all_agents()

    if not RECOMMENDATIONS_FILE.exists():
        return pd.DataFrame()

    return pd.read_csv(RECOMMENDATIONS_FILE)


def load_decisions() -> pd.DataFrame:
    """Load approve/reject decisions stored by this page."""
    if not DECISIONS_FILE.exists():
        return pd.DataFrame(columns=DECISION_COLUMNS)

    decisions = pd.read_csv(DECISIONS_FILE)
    for column in DECISION_COLUMNS:
        if column not in decisions.columns:
            decisions[column] = ""

    return decisions[DECISION_COLUMNS]


def save_decision(recommendation: pd.Series, decision: str) -> None:
    """Append an approval or rejection decision."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    decisions = load_decisions()
    new_decision = {
        "recommendation_id": recommendation.get("recommendation_id", ""),
        "decision": decision,
        "decided_at": datetime.now().isoformat(timespec="seconds"),
        "recommendation_type": recommendation.get("recommendation_type", ""),
        "product_id": recommendation.get("product_id", ""),
        "product_name": recommendation.get("product_name", ""),
        "store_id": recommendation.get("store_id", ""),
    }

    decisions = pd.concat(
        [decisions, pd.DataFrame([new_decision])],
        ignore_index=True,
    )
    decisions.to_csv(DECISIONS_FILE, index=False)


def apply_latest_decisions(
    recommendations: pd.DataFrame,
    decisions: pd.DataFrame,
) -> pd.DataFrame:
    """Show approved/rejected status from the latest stored decision."""
    recommendations = recommendations.copy()
    if "status" not in recommendations.columns:
        recommendations["status"] = "pending"

    if decisions.empty or "recommendation_id" not in decisions.columns:
        return recommendations

    latest_decisions = (
        decisions.dropna(subset=["recommendation_id"])
        .sort_values("decided_at")
        .drop_duplicates("recommendation_id", keep="last")
    )
    latest_decisions = latest_decisions[["recommendation_id", "decision"]]

    recommendations = recommendations.merge(
        latest_decisions,
        on="recommendation_id",
        how="left",
    )
    recommendations["status"] = recommendations["decision"].fillna(
        recommendations["status"]
    )
    recommendations = recommendations.drop(columns=["decision"])

    return recommendations


def unique_options(df: pd.DataFrame, column: str) -> list[str]:
    if df.empty or column not in df.columns:
        return []

    values = df[column].fillna("").astype(str)
    return sorted(value for value in values.unique() if value)


def filter_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    """Apply sidebar filters to recommendation rows."""
    with st.sidebar:
        st.header("Filters")

        selected_types = st.multiselect(
            "Type",
            unique_options(df, "recommendation_type"),
        )
        selected_products = st.multiselect(
            "Product",
            unique_options(df, "product_name"),
        )
        selected_stores = st.multiselect(
            "Store",
            unique_options(df, "store_id"),
        )
        selected_priorities = st.multiselect(
            "Priority",
            unique_options(df, "priority"),
        )
        selected_statuses = st.multiselect(
            "Status",
            unique_options(df, "status"),
        )

    filtered = df.copy()

    if selected_types:
        filtered = filtered[
            filtered["recommendation_type"].astype(str).isin(selected_types)
        ]
    if selected_products:
        filtered = filtered[
            filtered["product_name"].astype(str).isin(selected_products)
        ]
    if selected_stores:
        filtered = filtered[filtered["store_id"].astype(str).isin(selected_stores)]
    if selected_priorities:
        filtered = filtered[filtered["priority"].astype(str).isin(selected_priorities)]
    if selected_statuses:
        filtered = filtered[filtered["status"].astype(str).isin(selected_statuses)]

    return filtered


def show_recommendation_card(recommendation: pd.Series) -> None:
    """Render one practical recommendation card."""
    title = (
        f"{recommendation.get('recommendation_type', '')}"
        f" - {recommendation.get('product_name', '')}"
    )
    store_id = recommendation.get("store_id", "")
    if pd.notna(store_id) and str(store_id):
        title += f" ({store_id})"

    with st.container(border=True):
        top_left, top_right = st.columns([3, 1])
        top_left.subheader(title.replace("_", " ").title())
        top_right.markdown(f"**Priority:** {recommendation.get('priority', '')}")
        top_right.markdown(f"**Status:** {recommendation.get('status', '')}")

        st.markdown(f"**Action:** {recommendation.get('action', '')}")
        st.markdown(f"**Reason:** {recommendation.get('reason', '')}")
        st.markdown(f"**Evidence:** {recommendation.get('evidence', '')}")

        if pd.notna(recommendation.get("suggested_quantity", "")):
            quantity = str(recommendation.get("suggested_quantity", "")).strip()
            if quantity:
                st.caption(f"Suggested quantity: {quantity}")

        button_left, button_right, _ = st.columns([1, 1, 5])
        recommendation_id = recommendation.get("recommendation_id", "")
        current_status = str(recommendation.get("status", "pending"))
        disabled = current_status in {"approved", "rejected"}

        if button_left.button(
            "Approve",
            key=f"approve_{recommendation_id}",
            disabled=disabled,
        ):
            save_decision(recommendation, "approved")
            st.rerun()

        if button_right.button(
            "Reject",
            key=f"reject_{recommendation_id}",
            disabled=disabled,
        ):
            save_decision(recommendation, "rejected")
            st.rerun()


st.title("Recommendations")
st.caption("Data-driven actions from processed inventory, sales, and supplier signals.")

try:
    recommendations = load_recommendations()
except Exception as error:
    st.error("Could not load recommendations.")
    st.exception(error)
    st.stop()

decisions = load_decisions()
recommendations = apply_latest_decisions(recommendations, decisions)

if recommendations.empty:
    st.info("No recommendations are available yet.")
    st.stop()

total_recommendations = len(recommendations)
pending_recommendations = int(recommendations["status"].eq("pending").sum())
high_priority_recommendations = int(
    recommendations["priority"].astype(str).str.lower().eq("high").sum()
)

kpi_columns = st.columns(3)
kpi_columns[0].metric("Total Recommendations", f"{total_recommendations:,}")
kpi_columns[1].metric("Pending", f"{pending_recommendations:,}")
kpi_columns[2].metric("High Priority", f"{high_priority_recommendations:,}")

st.divider()

filtered_recommendations = filter_recommendations(recommendations)
st.subheader("Recommendation List")
st.caption(f"Showing {len(filtered_recommendations):,} of {total_recommendations:,}.")

table_columns = [
    "recommendation_id",
    "recommendation_type",
    "product_name",
    "store_id",
    "priority",
    "status",
    "action",
    "reason",
    "evidence",
    "suggested_quantity",
]
visible_columns = [
    column for column in table_columns if column in filtered_recommendations.columns
]

if filtered_recommendations.empty:
    st.info("No recommendations match the selected filters.")
else:
    st.dataframe(
        filtered_recommendations[visible_columns],
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Review Actions")
    for _, recommendation in filtered_recommendations.iterrows():
        show_recommendation_card(recommendation)
