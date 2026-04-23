import pandas as pd

from backend.services.llm_reasoner import reason_over_recommendations
from backend.services.recommendation_engine import (
    generate_stock_transfer_recommendations,
)


SOURCE_AGENT = "transfer_agent"


def _tag_source_agent(recommendations: list[dict]) -> list[dict]:
    """Mark recommendations with this agent as the source."""
    for recommendation in recommendations:
        recommendation["source_agent"] = SOURCE_AGENT
    return recommendations


def _row_lookup(df: pd.DataFrame) -> dict[tuple[str, str], dict]:
    """Build a quick lookup for product-store rows."""
    if df.empty:
        return {}

    lookup = {}
    for _, row in df.iterrows():
        lookup[(str(row.get("product_id", "")), str(row.get("store_id", "")))] = (
            row.to_dict()
        )
    return lookup


def _build_llm_candidates(inputs: dict, recommendations: list[dict]) -> list[dict]:
    """Prepare compact transfer context for the LLM."""
    low_stock_lookup = _row_lookup(inputs.get("low_stock_items", pd.DataFrame()))

    candidates = []
    for index, recommendation in enumerate(recommendations, start=1):
        product_id = str(recommendation.get("product_id", ""))
        store_id = str(recommendation.get("store_id", ""))
        low_stock_row = low_stock_lookup.get((product_id, store_id), {})

        candidates.append(
            {
                "candidate_id": f"transfer_{index}",
                "product_id": product_id,
                "product_name": recommendation.get("product_name", ""),
                "destination_store_id": store_id,
                "current_priority": recommendation.get("priority", ""),
                "rule_action": recommendation.get("action", ""),
                "rule_reason": recommendation.get("reason", ""),
                "rule_evidence": recommendation.get("evidence", ""),
                "stock_level": low_stock_row.get("stock_level", ""),
                "effective_reorder_threshold": low_stock_row.get(
                    "effective_reorder_threshold",
                    "",
                ),
                "recent_daily_sales_velocity": low_stock_row.get(
                    "recent_daily_sales_velocity",
                    "",
                ),
                "days_of_stock_remaining": low_stock_row.get(
                    "days_of_stock_remaining",
                    "",
                ),
            }
        )

    return candidates


def _apply_llm_decisions(
    recommendations: list[dict],
    decisions: dict[str, dict],
) -> list[dict]:
    """Use LLM output to refine transfer priorities and explanations."""
    if not decisions:
        return recommendations

    updated_recommendations = []
    for index, recommendation in enumerate(recommendations, start=1):
        decision = decisions.get(f"transfer_{index}")
        if not decision:
            updated_recommendations.append(recommendation)
            continue

        if not decision.get("keep_recommendation", True):
            continue

        if decision.get("selected_strategy") == "hold":
            continue

        recommendation["priority"] = decision.get(
            "priority",
            recommendation.get("priority", "medium"),
        )
        recommendation["action"] = decision.get("action", recommendation.get("action", ""))
        recommendation["reason"] = decision.get("reason", recommendation.get("reason", ""))
        recommendation["evidence"] = decision.get(
            "evidence",
            recommendation.get("evidence", ""),
        )
        updated_recommendations.append(recommendation)

    return updated_recommendations


def analyze_transfer_opportunities(
    inputs: dict,
    config: dict | None = None,
) -> list[dict]:
    """Identify stock transfer opportunities across stores."""
    recommendations = generate_stock_transfer_recommendations(
        inputs.get("current_inventory"),
        inputs.get("low_stock_items"),
        config,
    )
    llm_decisions = reason_over_recommendations(
        agent_name=SOURCE_AGENT,
        agent_goal=(
            "Decide whether a transfer candidate should move stock now or be held, "
            "and explain the decision using only the provided metrics."
        ),
        candidates=_build_llm_candidates(inputs, recommendations),
        allowed_strategies=["stock_transfer", "hold"],
        shared_context={
            "inventory_rows": len(inputs.get("current_inventory", pd.DataFrame())),
            "low_stock_count": len(inputs.get("low_stock_items", pd.DataFrame())),
        },
    )
    recommendations = _apply_llm_decisions(recommendations, llm_decisions)
    return _tag_source_agent(recommendations)


def run(
    inputs: dict,
    config: dict | None = None,
) -> list[dict]:
    """Entry point used by the orchestrator."""
    return analyze_transfer_opportunities(inputs, config)
