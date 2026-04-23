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

    return _tag_source_agent(recommendations)


def run(
    inputs: dict,
    config: dict | None = None,
) -> list[dict]:
    """Entry point used by the orchestrator."""
    return analyze_pricing_opportunities(inputs, config)
