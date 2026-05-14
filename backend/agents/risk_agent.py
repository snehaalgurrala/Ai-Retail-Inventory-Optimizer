import pandas as pd

from backend.memory.learning_loop import get_learning_context
from backend.services.recommendation_engine import (
    generate_overstock_alerts,
    generate_stockout_prevention_alerts,
    generate_supplier_risk_alerts,
)


SOURCE_AGENT = "risk_agent"


def _tag_source_agent(recommendations: list[dict]) -> list[dict]:
    """Mark recommendations with this agent as the source."""
    for recommendation in recommendations:
        recommendation["source_agent"] = SOURCE_AGENT
    return recommendations


def _product_lookup(df: pd.DataFrame) -> dict[str, dict]:
    """Build a product-level lookup for supplier performance rows."""
    if df.empty or "product_id" not in df.columns:
        return {}

    lookup = {}
    for _, row in df.iterrows():
        lookup[str(row.get("product_id", ""))] = row.to_dict()
    return lookup


def _product_store_lookup(df: pd.DataFrame) -> dict[tuple[str, str], dict]:
    """Build a product-store lookup for risk rows."""
    if df.empty:
        return {}

    lookup = {}
    for _, row in df.iterrows():
        lookup[(str(row.get("product_id", "")), str(row.get("store_id", "")))] = (
            row.to_dict()
        )
    return lookup


def _build_llm_candidates(inputs: dict, recommendations: list[dict]) -> list[dict]:
    """Prepare compact risk context for the LLM."""
    stockout_lookup = _product_store_lookup(
        inputs.get("stockout_risk_items", pd.DataFrame())
    )
    overstock_lookup = _product_store_lookup(inputs.get("overstock_items", pd.DataFrame()))
    product_lookup = _product_lookup(inputs.get("product_performance", pd.DataFrame()))

    candidates = []
    for index, recommendation in enumerate(recommendations, start=1):
        product_id = str(recommendation.get("product_id", ""))
        store_id = str(recommendation.get("store_id", ""))

        stockout_row = stockout_lookup.get((product_id, store_id), {})
        overstock_row = overstock_lookup.get((product_id, store_id), {})
        product_row = product_lookup.get(product_id, {})
        memory_context = get_learning_context(
            product_id=product_id,
            store_id=store_id,
            recommendation_type=str(
                recommendation.get("recommendation_type", "")
            ),
        )

        candidates.append(
            {
                "candidate_id": f"risk_{index}",
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
                "days_of_stock_remaining": stockout_row.get(
                    "days_of_stock_remaining",
                    overstock_row.get("days_of_stock_remaining", ""),
                ),
                "recent_daily_sales_velocity": stockout_row.get(
                    "recent_daily_sales_velocity",
                    overstock_row.get("recent_daily_sales_velocity", ""),
                ),
                "stock_level": stockout_row.get(
                    "stock_level",
                    overstock_row.get("stock_level", ""),
                ),
                "supplier_id": product_row.get("supplier_id", ""),
                "supplier_name": product_row.get("supplier_name", ""),
                "reliability_score": product_row.get("reliability_score", ""),
                "avg_delivery_days": product_row.get("avg_delivery_days", ""),
                "memory_hint": memory_context.get("memory_hint", ""),
                "learning_hint": memory_context.get("learning_hint", ""),
                "recent_decisions": memory_context.get("recent_decisions", []),
                "recent_outcomes": memory_context.get("recent_outcomes", []),
                "recent_learning_insights": memory_context.get(
                    "recent_learning_insights",
                    [],
                ),
            }
        )

    return candidates


def _apply_llm_decisions(
    recommendations: list[dict],
    decisions: dict[str, dict],
) -> list[dict]:
    """Use LLM output to refine risk type, priority, and explanation."""
    if not decisions:
        return recommendations

    updated_recommendations = []
    for index, recommendation in enumerate(recommendations, start=1):
        decision = decisions.get(f"risk_{index}")
        if not decision:
            updated_recommendations.append(recommendation)
            continue

        if not decision.get("keep_recommendation", True):
            continue

        selected_strategy = decision.get("selected_strategy", "")
        if selected_strategy == "monitor":
            continue

        if selected_strategy:
            recommendation["recommendation_type"] = selected_strategy
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


def analyze_risks(
    inputs: dict,
    config: dict | None = None,
) -> list[dict]:
    """Identify stockout, overstock, and supplier risks from pandas outputs."""
    recommendations = []
    recommendations.extend(
        generate_supplier_risk_alerts(
            inputs.get("product_performance", pd.DataFrame()),
            inputs.get("suppliers", pd.DataFrame()),
            config,
        )
    )
    recommendations.extend(generate_overstock_alerts(inputs.get("overstock_items", pd.DataFrame())))
    recommendations.extend(
        generate_stockout_prevention_alerts(
            inputs.get("stockout_risk_items", pd.DataFrame()),
            config,
        )
    )

    return _tag_source_agent(recommendations)


def run(
    inputs: dict,
    config: dict | None = None,
) -> list[dict]:
    """Entry point used by the orchestrator."""
    return analyze_risks(inputs, config)
