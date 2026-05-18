from pathlib import Path
import sys

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from frontend.components.cards import render_summary_card  # noqa: E402
from frontend.utils.page_helpers import (  # noqa: E402
    apply_page_style,
    render_page_header,
    render_kpi_card,
    render_section_header,
)
from backend.services.llm_reasoner import llm_is_configured  # noqa: E402
from backend.services.recommendation_execution_service import (  # noqa: E402
    DECISION_COLUMNS,
    DECISIONS_FILE,
    RECOMMENDATIONS_FILE,
    approve_recommendation,
    build_recommendation_context,
    get_recommendation_explanation,
    reject_recommendation,
)


st.set_page_config(
    page_title="Recommendations",
    page_icon="R",
    layout="wide",
)

apply_page_style()


TYPE_META = {
    "discount": {"icon": "💸", "label": "Discount", "accent": "#2563eb"},
    "clearance": {"icon": "🧹", "label": "Clearance", "accent": "#f97316"},
    "stock_transfer": {"icon": "↔", "label": "Stock Transfer", "accent": "#7c3aed"},
    "transfer": {"icon": "T", "label": "Transfer", "accent": "#0f766e"},
    "exclusive_availability": {"icon": "E", "label": "Exclusive Availability", "accent": "#0f766e"},
    "alternative_option": {"icon": "A", "label": "Alternative Option", "accent": "#0891b2"},
    "reorder": {"icon": "📦", "label": "Reorder", "accent": "#16a34a"},
    "supplier_risk_alert": {"icon": "⚠", "label": "Supplier Risk", "accent": "#dc2626"},
    "overstock_alert": {"icon": "📊", "label": "Overstock Alert", "accent": "#ea580c"},
    "stockout_prevention_alert": {"icon": "🚨", "label": "Stockout Risk", "accent": "#b91c1c"},
}

PRIORITY_OPTIONS = ["high", "medium", "low"]


def inject_recommendation_styles() -> None:
    st.markdown(
        """
        <style>
        .recommendation-shell {
            border: 1px solid color-mix(in srgb, var(--text-color) 9%, transparent);
            border-radius: 20px;
            padding: 1rem 1rem 0.7rem 1rem;
            margin-bottom: 1rem;
            background:
                radial-gradient(circle at top right, color-mix(in srgb, var(--secondary-background-color) 82%, transparent), transparent 32%),
                linear-gradient(180deg, color-mix(in srgb, var(--background-color) 65%, var(--secondary-background-color) 35%), var(--secondary-background-color));
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.08);
        }
        .recommendation-head {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            align-items: flex-start;
            margin-bottom: 0.6rem;
        }
        .recommendation-title {
            font-size: 1.04rem;
            font-weight: 700;
            color: var(--text-color);
            margin-bottom: 0.18rem;
        }
        .recommendation-subtitle {
            color: color-mix(in srgb, var(--text-color) 68%, transparent);
            font-size: 0.87rem;
        }
        .recommendation-summary {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.65rem;
            margin: 0.75rem 0 0.55rem 0;
        }
        .recommendation-chip {
            border-radius: 14px;
            padding: 0.7rem 0.8rem;
            border: 1px solid color-mix(in srgb, var(--text-color) 8%, transparent);
            background: color-mix(in srgb, var(--secondary-background-color) 86%, transparent);
        }
        .recommendation-chip-label {
            font-size: 0.69rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: color-mix(in srgb, var(--text-color) 58%, transparent);
            margin-bottom: 0.18rem;
        }
        .recommendation-chip-value {
            font-size: 0.97rem;
            color: var(--text-color);
            font-weight: 700;
            line-height: 1.28;
        }
        .priority-pill, .status-pill {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            padding: 0.24rem 0.62rem;
            font-size: 0.74rem;
            font-weight: 700;
            margin-left: 0.35rem;
            border: 1px solid transparent;
        }
        .priority-high { color: #7f1d1d; background: rgba(248, 113, 113, 0.18); border-color: rgba(248, 113, 113, 0.28); }
        .priority-medium { color: #854d0e; background: rgba(250, 204, 21, 0.18); border-color: rgba(250, 204, 21, 0.28); }
        .priority-low { color: #1d4ed8; background: rgba(96, 165, 250, 0.18); border-color: rgba(96, 165, 250, 0.28); }
        .status-pending { color: #334155; background: rgba(148, 163, 184, 0.16); border-color: rgba(148, 163, 184, 0.2); }
        .status-approved { color: #166534; background: rgba(74, 222, 128, 0.18); border-color: rgba(74, 222, 128, 0.3); }
        .status-rejected { color: #991b1b; background: rgba(248, 113, 113, 0.16); border-color: rgba(248, 113, 113, 0.26); }
        .reasoning-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.7rem;
            margin-top: 0.4rem;
        }
        .reasoning-box {
            border-radius: 14px;
            padding: 0.8rem 0.9rem;
            border: 1px solid color-mix(in srgb, var(--text-color) 8%, transparent);
            background: color-mix(in srgb, var(--secondary-background-color) 86%, transparent);
        }
        .reasoning-title {
            font-size: 0.73rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: color-mix(in srgb, var(--text-color) 56%, transparent);
            margin-bottom: 0.28rem;
        }
        .reasoning-body {
            color: var(--text-color);
            font-size: 0.9rem;
            line-height: 1.42;
        }
        @media (max-width: 1100px) {
            .recommendation-summary, .reasoning-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_recommendation_styles()


@st.cache_data
def load_recommendations() -> pd.DataFrame:
    if not RECOMMENDATIONS_FILE.exists():
        from backend.agents.orchestrator_agent import run_all_agents

        run_all_agents()

    if not RECOMMENDATIONS_FILE.exists():
        return pd.DataFrame()

    recommendations = pd.read_csv(RECOMMENDATIONS_FILE)
    for column in ["manager_note", "decision_reason", "status_updated_at"]:
        if column not in recommendations.columns:
            recommendations[column] = ""
    return recommendations


@st.cache_data
def load_decisions() -> pd.DataFrame:
    if not DECISIONS_FILE.exists():
        return pd.DataFrame(columns=DECISION_COLUMNS)

    decisions = pd.read_csv(DECISIONS_FILE)
    for column in DECISION_COLUMNS:
        if column not in decisions.columns:
            decisions[column] = ""
    return decisions


def clear_recommendation_caches() -> None:
    load_recommendations.clear()
    load_decisions.clear()


def apply_latest_decisions(
    recommendations: pd.DataFrame,
    decisions: pd.DataFrame,
) -> pd.DataFrame:
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
    merge_columns = [
        column
        for column in [
            "recommendation_id",
            "decision",
            "manager_note",
            "rejection_reason",
            "action_summary",
            "execution_status",
        ]
        if column in latest_decisions.columns
    ]
    latest_decisions = latest_decisions[merge_columns]

    merged = recommendations.merge(
        latest_decisions,
        on="recommendation_id",
        how="left",
    )
    if "decision" in merged.columns:
        merged["status"] = merged["decision"].fillna(merged["status"])
        merged = merged.drop(columns=["decision"])
    return merged


def unique_options(df: pd.DataFrame, column: str) -> list[str]:
    if df.empty or column not in df.columns:
        return []
    values = df[column].fillna("").astype(str)
    return sorted(value for value in values.unique() if value)


def filter_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    with st.sidebar:
        st.header("Filters")
        st.caption("Focus the review queue without changing source data.")
        st.divider()

        selected_types = st.multiselect(
            "Type",
            unique_options(df, "recommendation_type"),
            default=[
                recommendation_type
                for recommendation_type in st.session_state.get("recommendation_type_filter", [])
                if recommendation_type in unique_options(df, "recommendation_type")
            ],
        )
        selected_products = st.multiselect("Product", unique_options(df, "product_name"))
        selected_stores = st.multiselect("Store", unique_options(df, "store_id"))
        selected_priorities = st.multiselect("Priority", unique_options(df, "priority"))
        selected_statuses = st.multiselect("Status", unique_options(df, "status"))

    filtered = df.copy()
    if selected_types:
        filtered = filtered[filtered["recommendation_type"].astype(str).isin(selected_types)]
    if selected_products:
        filtered = filtered[filtered["product_name"].astype(str).isin(selected_products)]
    if selected_stores:
        filtered = filtered[filtered["store_id"].astype(str).isin(selected_stores)]
    if selected_priorities:
        filtered = filtered[filtered["priority"].astype(str).isin(selected_priorities)]
    if selected_statuses:
        filtered = filtered[filtered["status"].astype(str).isin(selected_statuses)]
    return filtered


def recommendation_type_rows(recommendations: pd.DataFrame, recommendation_types: list[str]) -> pd.DataFrame:
    if recommendations.empty or "recommendation_type" not in recommendations.columns:
        return pd.DataFrame()
    return recommendations[recommendations["recommendation_type"].astype(str).isin(recommendation_types)].copy()


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
    pricing_rows = recommendation_type_rows(recommendations, ["discount", "clearance"])
    transfer_rows = recommendation_type_rows(
        recommendations,
        ["stock_transfer", "transfer", "exclusive_availability", "alternative_option"],
    )
    procurement_rows = recommendation_type_rows(recommendations, ["reorder"])
    risk_rows = recommendation_type_rows(
        recommendations,
        ["supplier_risk_alert", "overstock_alert", "stockout_prevention_alert"],
    )
    return [
        {
            "title": "Pricing & Clearance",
            "icon": "💸",
            "summary": f"{len(pricing_rows):,} actions",
            "accent": "blue",
            "button_key": "review_pricing",
            "types": ["discount", "clearance"],
            "insights": [
                f"{unique_count(pricing_rows, 'product_id'):,} products affected.",
                f"{pending_count(pricing_rows):,} still need approval.",
                most_common_action(pricing_rows),
            ],
        },
        {
            "title": "Stock Transfer",
            "icon": "↔",
            "summary": f"{len(transfer_rows):,} opportunities",
            "accent": "purple",
            "button_key": "review_transfer",
            "types": ["stock_transfer", "transfer", "exclusive_availability", "alternative_option"],
            "insights": [
                f"{unique_count(transfer_rows, 'product_id'):,} products can be rebalanced.",
                f"{pending_count(transfer_rows):,} transfers are pending.",
                most_common_action(transfer_rows),
            ],
        },
        {
            "title": "Procurement",
            "icon": "📦",
            "summary": f"{len(procurement_rows):,} replenishments",
            "accent": "green",
            "button_key": "review_procurement",
            "types": ["reorder"],
            "insights": [
                f"{high_priority_count(procurement_rows):,} high-priority reorders.",
                f"{pending_count(procurement_rows):,} await manager review.",
                most_common_action(procurement_rows),
            ],
        },
        {
            "title": "Risk Alerts",
            "icon": "⚠",
            "summary": f"{len(risk_rows):,} active alerts",
            "accent": "orange",
            "button_key": "review_risk",
            "types": ["supplier_risk_alert", "overstock_alert", "stockout_prevention_alert"],
            "insights": [
                f"{high_priority_count(risk_rows):,} high-priority alerts.",
                f"{pending_count(risk_rows):,} still open.",
                most_common_action(risk_rows),
            ],
        },
    ]


def priority_badge(priority: str) -> str:
    normalized = str(priority or "low").strip().lower()
    if normalized not in {"high", "medium", "low"}:
        normalized = "low"
    return f"<span class='priority-pill priority-{normalized}'>{normalized.title()}</span>"


def status_badge(status: str) -> str:
    normalized = str(status or "pending").strip().lower()
    if normalized not in {"pending", "approved", "rejected"}:
        normalized = "pending"
    return f"<span class='status-pill status-{normalized}'>{normalized.title()}</span>"


def compact_value_summary(recommendation: pd.Series, context: dict) -> str:
    recommendation_type = str(recommendation.get("recommendation_type", ""))
    if recommendation_type == "discount":
        return (
            f"{context['recommended_discount_percent']:.1f}% off • "
            f"{context['current_price']:.2f} to {context['discounted_price']:.2f}"
        )
    if recommendation_type == "clearance":
        return (
            f"Clearance {context['recommended_discount_percent']:.1f}% • "
            f"{context['current_price']:.2f} to {context['discounted_price']:.2f}"
        )
    if recommendation_type in {"stock_transfer", "transfer"}:
        return (
            f"{context['source_store_id'] or 'source'} to {context['target_store_id'] or recommendation.get('store_id', '')} • "
            f"{context['suggested_quantity']} units"
        )
    if recommendation_type == "exclusive_availability":
        return (
            f"{recommendation.get('store_name') or recommendation.get('store_id', '')} - "
            f"{context.get('available_quantity', recommendation.get('available_quantity', 0))} units"
        )
    if recommendation_type == "alternative_option":
        return (
            f"{recommendation.get('alternative_product_name', '')} at "
            f"{recommendation.get('alternative_store_name') or recommendation.get('alternative_store_id', '')} - "
            f"{context.get('available_quantity', recommendation.get('available_quantity', 0))} units"
        )
    if recommendation_type == "reorder":
        return f"Reorder {context['suggested_quantity']} units"
    return f"{TYPE_META.get(recommendation_type, {}).get('label', recommendation_type)} • {recommendation.get('priority', '')}"


def render_reasoning_box(title: str, body: str) -> None:
    st.markdown(
        (
            "<div class='reasoning-box'>"
            f"<div class='reasoning-title'>{title}</div>"
            f"<div class='reasoning-body'>{body}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def get_card_explanation(recommendation: pd.Series, context: dict) -> dict:
    cache = st.session_state.setdefault("recommendation_explanations", {})
    recommendation_id = str(recommendation.get("recommendation_id", ""))
    explanation = cache.get(recommendation_id)
    if explanation:
        return explanation

    explanation = get_recommendation_explanation(recommendation, context)
    cache[recommendation_id] = explanation
    return explanation


def explanation_block(recommendation: pd.Series, context: dict) -> None:
    recommendation_id = str(recommendation.get("recommendation_id", ""))
    generate_key = f"generate_llm_{recommendation_id}"
    if llm_is_configured() and st.button(
        "Generate/refine LLM explanation",
        key=generate_key,
        use_container_width=False,
    ):
        st.session_state.setdefault("recommendation_explanations", {}).pop(recommendation_id, None)

    explanation = get_card_explanation(recommendation, context)
    st.caption(f"Explanation source: {explanation.get('explanation_source', 'fallback')}")

    reasoning_pairs = [
        ("Why This Was Generated", explanation.get("why_generated", "")),
        ("Data Factors Analyzed", explanation.get("factors_analyzed", "")),
        ("Expected Business Impact", explanation.get("expected_business_impact", "")),
        ("How Inventory Or Sales May Improve", explanation.get("sales_inventory_improvement", "")),
        ("Risk If Ignored", explanation.get("risk_if_ignored", "")),
        ("Confidence Level", explanation.get("confidence_level", "")),
        ("Source Agent", explanation.get("source_agent", "")),
        ("Supporting Evidence", explanation.get("supporting_evidence", "")),
    ]
    for index in range(0, len(reasoning_pairs), 2):
        columns = st.columns(2, gap="medium")
        for column_index, (title, body) in enumerate(reasoning_pairs[index:index + 2]):
            with columns[column_index]:
                render_reasoning_box(title, body)


def render_type_specific_details(recommendation: pd.Series, context: dict) -> None:
    recommendation_type = str(recommendation.get("recommendation_type", ""))
    if recommendation_type == "discount":
        st.write(f"Current price: {context['current_price']:.2f}")
        st.write(f"Recommended discount: {context['recommended_discount_percent']:.1f}%")
        st.write(f"Discounted price: {context['discounted_price']:.2f}")
        st.write(f"Reason for discount: {recommendation.get('reason', '')}")
        st.write("Expected sales lift explanation: Lower pricing is intended to improve sell-through on slow or overstocked inventory.")
        st.write(
            "Factors analyzed: "
            f"slow movement, overstock, recent sales velocity {context['recent_daily_velocity']:.2f}, "
            f"stock level {context['current_stock']}, shelf life {context['shelf_life_days']}."
        )
    elif recommendation_type in {"stock_transfer", "transfer"}:
        st.write(f"Source store: {context['source_store_name'] or context['source_store_id'] or 'Missing'}")
        st.write(f"Target store: {context['target_store_name'] or context['target_store_id'] or 'Missing'}")
        st.write(f"Transfer quantity: {context['suggested_quantity']}")
        st.write(f"Reason: {recommendation.get('reason', '')}")
        st.write("Expected benefit: Moves supply to a store with tighter coverage faster than new procurement.")
        st.write("Risk if not transferred: The target store may stock out while surplus remains elsewhere.")
    elif recommendation_type == "exclusive_availability":
        st.write(f"Product: {recommendation.get('product_name', '')}")
        st.write(f"Exclusive store: {recommendation.get('store_name') or recommendation.get('store_id', '')}")
        st.write(f"Available quantity: {context.get('available_quantity', recommendation.get('available_quantity', 0))}")
        st.write(f"Category: {context.get('category', '')}")
        st.write(f"Alternative suggestion: {recommendation.get('action', '')}")
        st.write(f"Why it matters: {recommendation.get('reason', '')}")
    elif recommendation_type == "alternative_option":
        st.write(f"Low-stock product: {recommendation.get('product_name', '')}")
        st.write(f"Low-stock store: {recommendation.get('store_name') or recommendation.get('store_id', '')}")
        st.write(f"Alternative product: {recommendation.get('alternative_product_name', '')}")
        st.write(f"Alternative store: {recommendation.get('alternative_store_name') or recommendation.get('alternative_store_id', '')}")
        st.write(f"Available quantity: {context.get('available_quantity', recommendation.get('available_quantity', 0))}")
        st.write(f"Category: {context.get('category', '')}")
        st.write(f"Action suggestion: {recommendation.get('action', '')}")
    elif recommendation_type == "reorder":
        st.write(f"Current stock: {context['current_stock']}")
        st.write(f"Threshold: {context['threshold']}")
        st.write(f"Suggested reorder quantity: {context['suggested_quantity']}")
        st.write(f"Priority: {recommendation.get('priority', '')}")
        st.write(f"Reason: {recommendation.get('reason', '')}")
        if context.get("supplier_name"):
            st.write(
                f"Supplier info: {context['supplier_name']} "
                f"(delivery {context['avg_delivery_days']:.0f} days, reliability {context['reliability_score']:.2f})"
            )
    elif recommendation_type == "clearance":
        st.write(f"Current stock: {context['current_stock']}")
        st.write(f"Sales movement: {context['recent_30_day_sales']} units in the last 30 days")
        st.write(f"Suggested clearance markdown: {context['recommended_discount_percent']:.1f}%")
        st.write("Expected benefit: Faster sell-through, lower expiry exposure, and less dead-stock buildup.")
    else:
        st.write(f"Risk type: {TYPE_META.get(recommendation_type, {}).get('label', recommendation_type)}")
        st.write(f"Risk reason: {recommendation.get('reason', '')}")
        st.write(f"Mitigation action: {recommendation.get('action', '')}")


def default_edit_values(recommendation: pd.Series, context: dict) -> dict:
    return {
        "discount_percent": float(context.get("recommended_discount_percent", 10.0)),
        "suggested_quantity": int(context.get("suggested_quantity", 0)),
        "source_store": str(context.get("source_store_id", "")),
        "target_store": str(context.get("target_store_id") or recommendation.get("store_id", "")),
        "action_text": str(recommendation.get("action", "")),
        "priority": str(recommendation.get("priority", "medium")).lower() or "medium",
        "manager_note": str(recommendation.get("manager_note", "")),
    }


def render_edit_form(recommendation: pd.Series, context: dict, disabled: bool) -> None:
    recommendation_type = str(recommendation.get("recommendation_type", ""))
    recommendation_id = str(recommendation.get("recommendation_id", ""))
    defaults = default_edit_values(recommendation, context)

    with st.form(key=f"edit_form_{recommendation_id}", clear_on_submit=False):
        left, right = st.columns(2, gap="large")
        edited_values = {}

        with left:
            if recommendation_type in {"discount", "clearance"}:
                edited_values["discount_percent"] = st.number_input(
                    "Discount percentage",
                    min_value=0.1,
                    max_value=99.0,
                    value=float(defaults["discount_percent"]),
                    step=0.5,
                    disabled=disabled,
                )
            else:
                edited_values["discount_percent"] = defaults["discount_percent"]

            if recommendation_type in {"stock_transfer", "transfer", "reorder"}:
                edited_values["suggested_quantity"] = st.number_input(
                    "Suggested quantity",
                    min_value=0,
                    value=max(0, int(defaults["suggested_quantity"])),
                    step=1,
                    disabled=disabled,
                )
            else:
                edited_values["suggested_quantity"] = defaults["suggested_quantity"]

            if recommendation_type in {"stock_transfer", "transfer"}:
                edited_values["source_store"] = st.text_input(
                    "Source store",
                    value=defaults["source_store"],
                    disabled=disabled,
                )
                edited_values["target_store"] = st.text_input(
                    "Target store",
                    value=defaults["target_store"],
                    disabled=disabled,
                )
            else:
                edited_values["source_store"] = defaults["source_store"]
                edited_values["target_store"] = st.text_input(
                    "Store",
                    value=defaults["target_store"],
                    disabled=disabled,
                )

        with right:
            edited_values["priority"] = st.selectbox(
                "Priority",
                PRIORITY_OPTIONS,
                index=PRIORITY_OPTIONS.index(defaults["priority"]) if defaults["priority"] in PRIORITY_OPTIONS else 1,
                disabled=disabled,
            )
            edited_values["action_text"] = st.text_area(
                "Action text",
                value=defaults["action_text"],
                height=120,
                disabled=disabled,
            )
            edited_values["manager_note"] = st.text_area(
                "Manager note",
                value=defaults["manager_note"],
                height=100,
                disabled=disabled,
                placeholder="Optional execution note or escalation context.",
            )

        rejection_reason = st.text_area(
            "Rejection reason",
            value=str(recommendation.get("rejection_reason", "")),
            height=80,
            disabled=disabled,
            placeholder="Required only when rejecting.",
        )
        action_columns = st.columns([1.2, 1.2, 3.6], gap="medium")
        approve_click = action_columns[0].form_submit_button(
            "Approve & Execute",
            type="primary",
            disabled=disabled,
            use_container_width=True,
        )
        reject_click = action_columns[1].form_submit_button(
            "Reject",
            disabled=disabled,
            use_container_width=True,
        )

        if approve_click:
            try:
                result = approve_recommendation(recommendation_id, edited_values)
                clear_recommendation_caches()
                st.session_state.setdefault("recommendation_messages", {})[recommendation_id] = {
                    "kind": "success",
                    "message": result["message"],
                }
                st.rerun()
            except Exception as error:
                st.error(str(error))

        if reject_click:
            try:
                result = reject_recommendation(recommendation_id, rejection_reason)
                clear_recommendation_caches()
                st.session_state.setdefault("recommendation_messages", {})[recommendation_id] = {
                    "kind": "warning",
                    "message": result["message"],
                }
                st.rerun()
            except Exception as error:
                st.error(str(error))


def render_card_feedback(recommendation_id: str) -> None:
    messages = st.session_state.get("recommendation_messages", {})
    payload = messages.get(recommendation_id)
    if not payload:
        return
    if payload["kind"] == "success":
        st.success(payload["message"])
    else:
        st.warning(payload["message"])


def render_recommendation_execution_card(recommendation: pd.Series) -> None:
    recommendation_type = str(recommendation.get("recommendation_type", ""))
    recommendation_id = str(recommendation.get("recommendation_id", ""))
    meta = TYPE_META.get(recommendation_type, {"icon": "•", "label": recommendation_type, "accent": "#475569"})
    context = build_recommendation_context(recommendation)
    current_status = str(recommendation.get("status", "pending")).lower()
    disabled = current_status in {"approved", "rejected"}
    store_name = context.get("store_name") or str(recommendation.get("store_name") or recommendation.get("store_id", ""))
    compact_summary = compact_value_summary(recommendation, context)
    is_availability_card = recommendation_type in {"exclusive_availability", "alternative_option"}

    with st.container(border=True):
        st.markdown(
            (
                "<div class='recommendation-head'>"
                "<div>"
                f"<div class='recommendation-title'>{meta['icon']} {meta['label']} • {recommendation.get('product_name', '')}</div>"
                f"<div class='recommendation-subtitle'>{store_name or 'No store scope'} • {recommendation_id}</div>"
                "</div>"
                "<div>"
                f"{priority_badge(recommendation.get('priority', ''))}{status_badge(recommendation.get('status', 'pending'))}"
                "</div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            (
                "<div class='recommendation-summary'>"
                f"<div class='recommendation-chip'><div class='recommendation-chip-label'>Suggested Action</div><div class='recommendation-chip-value'>{recommendation.get('action', '')}</div></div>"
                f"<div class='recommendation-chip'><div class='recommendation-chip-label'>Key Metric</div><div class='recommendation-chip-value'>{compact_summary}</div></div>"
                f"<div class='recommendation-chip'><div class='recommendation-chip-label'>Current Stock / Price</div><div class='recommendation-chip-value'>{context.get('current_stock', 0)} units • {context.get('current_price', 0):.2f}</div></div>"
                f"<div class='recommendation-chip'><div class='recommendation-chip-label'>Risk / Reason</div><div class='recommendation-chip-value'>{recommendation.get('reason', '')}</div></div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )

        render_card_feedback(recommendation_id)

        if is_availability_card:
            with st.expander("View availability reasoning", expanded=False):
                render_type_specific_details(recommendation, context)
                st.divider()
                render_reasoning_box(
                    "Where Available",
                    str(
                        recommendation.get("alternative_store_name")
                        or recommendation.get("store_name")
                        or recommendation.get("alternative_store_id")
                        or recommendation.get("store_id")
                        or "No store available."
                    ),
                )
                render_reasoning_box(
                    "Where Unavailable",
                    str(
                        context.get("evidence_map", {}).get("unavailable_stores")
                        or "Other stores have zero quantity or no inventory row for the exclusive product."
                    ),
                )
                render_reasoning_box(
                    "Why Exclusive",
                    "The product has quantity above zero in only one store in the current inventory data.",
                )
                render_reasoning_box(
                    "Business Help",
                    "This gives the team a substitute to promote when the original product is low or unavailable.",
                )
                st.divider()
                explanation_block(recommendation, context)
                st.divider()
                render_edit_form(recommendation, context, disabled=disabled)
            return

        with st.expander("View AI reasoning and impact", expanded=False):
            render_type_specific_details(recommendation, context)
            st.divider()
            explanation_block(recommendation, context)
            st.divider()
            render_edit_form(recommendation, context, disabled=disabled)


render_page_header(
    "🤖 Recommendations",
    "Advanced human-in-the-loop execution for pricing, transfer, reorder, clearance, and risk actions.",
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
pending_recommendations = int(recommendations["status"].astype(str).str.lower().eq("pending").sum())
high_priority_recommendations = int(
    recommendations["priority"].astype(str).str.lower().eq("high").sum()
)

kpi_columns = st.columns(3, gap="medium")
with kpi_columns[0]:
    render_kpi_card("Total Recommendations", f"{total_recommendations:,}", "Generated by the agent stack", "blue")
with kpi_columns[1]:
    render_kpi_card("Pending Execution", f"{pending_recommendations:,}", "Awaiting a human decision", "purple")
with kpi_columns[2]:
    render_kpi_card("High Priority", f"{high_priority_recommendations:,}", "Should be reviewed first", "red")

st.divider()

render_section_header(
    "🧭",
    "Recommendation Overview",
    "Grouped decision areas from the generated recommendations.",
)

summary_cards = build_summary_card_data(recommendations)
for row_start in range(0, len(summary_cards), 2):
    columns = st.columns(2, gap="large")
    for index, card_data in enumerate(summary_cards[row_start:row_start + 2]):
        with columns[index]:
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
    "Execution Queue",
    f"Showing {len(filtered_recommendations):,} of {total_recommendations:,} recommendations.",
)
st.caption(
    "Approving executes the underlying action, updates the source CSV files, records the decision, and refreshes page data safely."
)

table_columns = [
    "recommendation_id",
    "recommendation_type",
    "product_name",
    "store_id",
    "priority",
    "status",
    "action",
]
visible_columns = [column for column in table_columns if column in filtered_recommendations.columns]

if filtered_recommendations.empty:
    st.info("No recommendations match the selected filters.")
else:
    st.dataframe(
        filtered_recommendations[visible_columns],
        use_container_width=True,
        hide_index=True,
    )

    for _, recommendation in filtered_recommendations.iterrows():
        render_recommendation_execution_card(recommendation)
