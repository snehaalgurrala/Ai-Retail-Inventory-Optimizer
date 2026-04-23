import pandas as pd

from backend.services.llm_reasoner import reason_over_recommendations
from backend.services.recommendation_engine import (
    generate_clearance_recommendations,
    generate_discount_recommendations,
)


SOURCE_AGENT = "pricing_agent"


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
        product_id = str(row.get("product_id", ""))
        store_id = str(row.get("store_id", ""))
        lookup[(product_id, store_id)] = row.to_dict()
    return lookup


def _build_llm_candidates(inputs: dict, recommendations: list[dict]) -> list[dict]:
    """Prepare compact pricing context for the LLM."""
    slow_lookup = _row_lookup(inputs.get("slow_moving_items", pd.DataFrame()))
    overstock_lookup = _row_lookup(inputs.get("overstock_items", pd.DataFrame()))
    dead_stock_lookup = _row_lookup(inputs.get("dead_stock_candidates", pd.DataFrame()))

    candidates = []
    for index, recommendation in enumerate(recommendations, start=1):
        product_id = str(recommendation.get("product_id", ""))
        store_id = str(recommendation.get("store_id", ""))
        key = (product_id, store_id)

        slow_row = slow_lookup.get(key, {})
        overstock_row = overstock_lookup.get(key, {})
        dead_stock_row = dead_stock_lookup.get(key, {})

        candidates.append(
            {
                "candidate_id": f"pricing_{index}",
                "product_id": product_id,
                "product_name": recommendation.get("product_name", ""),
                "store_id": store_id,
                "current_recommendation_type": recommendation.get(
                    "recommendation_type",
                    "",
                ),
                "current_priority": recommendation.get("priority", ""),
                "rule_action": recommendation.get("action", ""),
                "rule_reason": recommendation.get("reason", ""),
                "rule_evidence": recommendation.get("evidence", ""),
                "days_of_stock_remaining": slow_row.get(
                    "days_of_stock_remaining",
                    overstock_row.get(
                        "days_of_stock_remaining",
                        dead_stock_row.get("days_of_stock_remaining", ""),
                    ),
                ),
                "recent_daily_sales_velocity": slow_row.get(
                    "recent_daily_sales_velocity",
                    overstock_row.get("recent_daily_sales_velocity", ""),
                ),
                "recent_30_day_quantity_sold": dead_stock_row.get(
                    "recent_30_day_quantity_sold",
                    slow_row.get("recent_30_day_quantity_sold", ""),
                ),
                "stock_level": slow_row.get(
                    "stock_level",
                    overstock_row.get("stock_level", dead_stock_row.get("stock_level", "")),
                ),
                "shelf_life_days": dead_stock_row.get("shelf_life_days", ""),
            }
        )

    return candidates


def _apply_llm_decisions(
    recommendations: list[dict],
    decisions: dict[str, dict],
) -> list[dict]:
    """Use LLM output to refine pricing strategy, explanation, and priority."""
    if not decisions:
        return recommendations

    updated_recommendations = []
    for index, recommendation in enumerate(recommendations, start=1):
        decision = decisions.get(f"pricing_{index}")
        if not decision:
            updated_recommendations.append(recommendation)
            continue

        if not decision.get("keep_recommendation", True):
            continue

        selected_strategy = decision.get("selected_strategy", "")
        if selected_strategy == "hold":
            continue

        recommendation["recommendation_type"] = selected_strategy or recommendation.get(
            "recommendation_type",
            "",
        )
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


def analyze_pricing_opportunities(
    inputs: dict,
    config: dict | None = None,
) -> list[dict]:
    """Identify discount and clearance opportunities from analyzer outputs."""
    recommendations = []
    recommendations.extend(
        generate_discount_recommendations(
            inputs.get("slow_moving_items"),
            inputs.get("overstock_items"),
            config,
        )
    )
    recommendations.extend(
        generate_clearance_recommendations(
            inputs.get("dead_stock_candidates"),
            config,
        )
    )
    llm_decisions = reason_over_recommendations(
        agent_name=SOURCE_AGENT,
        agent_goal=(
            "Choose whether pricing candidates should be discounted, cleared, or held, "
            "and provide grounded explanations."
        ),
        candidates=_build_llm_candidates(inputs, recommendations),
        allowed_strategies=["discount", "clearance", "hold"],
        shared_context={
            "slow_moving_count": len(inputs.get("slow_moving_items", pd.DataFrame())),
            "overstock_count": len(inputs.get("overstock_items", pd.DataFrame())),
            "dead_stock_count": len(
                inputs.get("dead_stock_candidates", pd.DataFrame())
            ),
        },
    )
    recommendations = _apply_llm_decisions(recommendations, llm_decisions)
    return _tag_source_agent(recommendations)


def run(
    inputs: dict,
    config: dict | None = None,
) -> list[dict]:
    """Entry point used by the orchestrator."""
    return analyze_pricing_opportunities(inputs, config)
