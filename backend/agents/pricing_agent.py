import pandas as pd

from backend.agents.tools import (
    describe_agent_tools,
    invoke_agent_tool,
)
from backend.memory.learning_loop import get_learning_context
from backend.services.llm_reasoner import reason_over_recommendations
from backend.services.llm_reasoner import select_tools_for_agent


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
        memory_context = get_learning_context(
            product_id=product_id,
            store_id=store_id,
            recommendation_type=str(
                recommendation.get("recommendation_type", "")
            ),
        )

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
    """Identify pricing opportunities through LangChain tool calls."""
    inventory_summary_tool = invoke_agent_tool(
        "get_current_inventory_summary",
        {"limit": 5},
    )
    sales_summary_tool = invoke_agent_tool(
        "get_sales_summary",
        {"limit": 5},
    )
    dead_stock_tool = invoke_agent_tool(
        "get_dead_stock_candidates",
        {"limit": 5},
    )

    selected_tools = select_tools_for_agent(
        agent_name=SOURCE_AGENT,
        agent_goal=(
            "Use pricing-related tools to review inventory movement and create "
            "discount or clearance candidates."
        ),
        available_tools=describe_agent_tools(SOURCE_AGENT),
        context={
            "slow_moving_count": len(inputs.get("slow_moving_items", pd.DataFrame())),
            "overstock_count": len(inputs.get("overstock_items", pd.DataFrame())),
            "dead_stock_count": len(
                inputs.get("dead_stock_candidates", pd.DataFrame())
            ),
            "inventory_summary": inventory_summary_tool.get("summary", {}),
            "sales_summary": sales_summary_tool.get("summary", {}),
            "dead_stock_summary": dead_stock_tool.get("summary", {}),
        },
        default_tools=["recommend_discount"],
    )
    if "recommend_discount" not in selected_tools:
        selected_tools.append("recommend_discount")

    for tool_name in selected_tools:
        if tool_name == "recommend_discount":
            tool_output = invoke_agent_tool(
                tool_name,
                {"config": config or {}, "limit": 0},
            )
            recommendations = tool_output.get("records", [])
            break
    else:
        recommendations = []

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
            "inventory_tool_context": inventory_summary_tool,
            "sales_tool_context": sales_summary_tool,
            "dead_stock_tool_context": dead_stock_tool,
            "system_memory": inputs.get("memory_context", {}),
            "system_learning": inputs.get("learning_context", {}),
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
