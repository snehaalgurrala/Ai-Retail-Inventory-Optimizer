from datetime import datetime
from pathlib import Path
import sys

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from frontend.components.cards import render_summary_card  # noqa: E402
from frontend.components.ui_components import render_recommendation_card  # noqa: E402
from frontend.utils.page_helpers import (  # noqa: E402
    apply_page_style,
    render_page_header,
    render_kpi_card,
    render_section_header,
)
from backend.memory.memory_store import save_decision_record  # noqa: E402

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
    save_decision_record(recommendation, decision, new_decision["decided_at"])


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
        st.caption("Narrow the recommendation list without changing the data.")
        st.divider()

        type_options = unique_options(df, "recommendation_type")
        default_types = [
            recommendation_type
            for recommendation_type in st.session_state.get(
                "recommendation_type_filter",
                [],
            )
            if recommendation_type in type_options
        ]
        selected_types = st.multiselect(
            "Type",
            type_options,
            default=default_types,
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


def recommendation_type_rows(
    recommendations: pd.DataFrame,
    recommendation_types: list[str],
) -> pd.DataFrame:
    """Return rows for one recommendation summary group."""
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


def most_common_action(df: pd.DataFrame) -> str:
    if df.empty or "action" not in df.columns:
        return "No current action available."

    actions = df["action"].fillna("").astype(str)
    actions = actions[actions != ""]
    if actions.empty:
        return "No current action available."

    return actions.iloc[0]


def build_summary_card_data(recommendations: pd.DataFrame) -> list[dict]:
    """Build grouped summary data for recommendation cards."""
    pricing_rows = recommendation_type_rows(
        recommendations,
        ["discount", "clearance"],
    )
    transfer_rows = recommendation_type_rows(
        recommendations,
        ["stock_transfer"],
    )
    dead_stock_rows = recommendation_type_rows(
        recommendations,
        ["clearance"],
    )
    risk_rows = recommendation_type_rows(
        recommendations,
        [
            "supplier_risk_alert",
            "overstock_alert",
            "stockout_prevention_alert",
        ],
    )

    return [
        {
            "title": "Pricing & Discounts",
            "icon": "💸",
            "summary": f"{unique_count(pricing_rows, 'product_id'):,} products affected",
            "accent": "blue",
            "button_key": "review_pricing",
            "types": ["discount", "clearance"],
            "insights": [
                f"{len(pricing_rows):,} pricing recommendations available.",
                f"{pending_count(pricing_rows):,} are still pending review.",
                most_common_action(pricing_rows),
            ],
        },
        {
            "title": "Stock Transfer",
            "icon": "↔",
            "summary": f"{len(transfer_rows):,} transfer opportunities",
            "accent": "purple",
            "button_key": "review_transfer",
            "types": ["stock_transfer"],
            "insights": [
                f"{unique_count(transfer_rows, 'store_id'):,} destination stores involved.",
                f"{unique_count(transfer_rows, 'product_id'):,} products can be balanced across stores.",
                most_common_action(transfer_rows),
            ],
        },
        {
            "title": "Dead Stock",
            "icon": "⏳",
            "summary": f"{unique_count(dead_stock_rows, 'product_id'):,} products affected",
            "accent": "orange",
            "button_key": "review_dead_stock",
            "types": ["clearance"],
            "insights": [
                f"{len(dead_stock_rows):,} clearance recommendations found.",
                f"{high_priority_count(dead_stock_rows):,} high priority clearance rows.",
                most_common_action(dead_stock_rows),
            ],
        },
        {
            "title": "Risk Alerts",
            "icon": "⚠",
            "summary": f"{len(risk_rows):,} active alerts",
            "accent": "red",
            "button_key": "review_risk",
            "types": [
                "supplier_risk_alert",
                "overstock_alert",
                "stockout_prevention_alert",
            ],
            "insights": [
                f"{high_priority_count(risk_rows):,} high priority risk alerts.",
                f"{unique_count(risk_rows, 'product_id'):,} products need risk review.",
                most_common_action(risk_rows),
            ],
        },
    ]


def approve_recommendation(recommendation: pd.Series) -> None:
    """Store an approval decision and refresh the page."""
    save_decision(recommendation, "approved")
    st.rerun()


def reject_recommendation(recommendation: pd.Series) -> None:
    """Store a rejection decision and refresh the page."""
    save_decision(recommendation, "rejected")
    st.rerun()


render_page_header(
    "🤖 Recommendations",
    "Data-driven actions from processed inventory, sales, and supplier signals.",
)

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

kpi_columns = st.columns(3, gap="medium")
with kpi_columns[0]:
    render_kpi_card(
        "Total Recommendations",
        f"{total_recommendations:,}",
        "Generated decision records",
        "blue",
    )
with kpi_columns[1]:
    render_kpi_card(
        "Pending",
        f"{pending_recommendations:,}",
        "Awaiting review decision",
        "purple",
    )
with kpi_columns[2]:
    render_kpi_card(
        "High Priority",
        f"{high_priority_recommendations:,}",
        "Needs faster attention",
        "red",
    )

st.divider()

render_section_header(
    "🧭",
    "Recommendation Overview",
    "Grouped decision areas from the generated recommendations.",
)

summary_cards = build_summary_card_data(recommendations)
for row_start in range(0, len(summary_cards), 2):
    card_columns = st.columns(2, gap="large")
    for index, card_data in enumerate(summary_cards[row_start:row_start + 2]):
        with card_columns[index]:
            if render_summary_card(
                title=card_data["title"],
                icon=card_data["icon"],
                summary=card_data["summary"],
                insights=card_data["insights"],
                accent=card_data["accent"],
                button_key=card_data["button_key"],
            ):
                st.session_state["recommendation_type_filter"] = card_data["types"]
                st.rerun()

st.divider()

filtered_recommendations = filter_recommendations(recommendations)
render_section_header(
    "📋",
    "Recommendation List",
    f"Showing {len(filtered_recommendations):,} of {total_recommendations:,}.",
)

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

    st.subheader("✅ Review Actions")
    for _, recommendation in filtered_recommendations.iterrows():
        render_recommendation_card(
            recommendation,
            approve_callback=approve_recommendation,
            reject_callback=reject_recommendation,
        )
