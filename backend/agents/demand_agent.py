from backend.services.recommendation_engine import (
    generate_reorder_recommendations,
)


SOURCE_AGENT = "demand_agent"


def _tag_source_agent(recommendations: list[dict]) -> list[dict]:
    """Mark recommendations with this agent as the source."""
    for recommendation in recommendations:
        recommendation["source_agent"] = SOURCE_AGENT
    return recommendations


def analyze_demand(
    inputs: dict,
    config: dict | None = None,
) -> list[dict]:
    """Analyze demand movement and recommend replenishment where needed."""
    recommendations = generate_reorder_recommendations(
        inputs.get("low_stock_items"),
        config,
    )
    return _tag_source_agent(recommendations)


def run(
    inputs: dict,
    config: dict | None = None,
) -> list[dict]:
    """Entry point used by the orchestrator."""
    return analyze_demand(inputs, config)
