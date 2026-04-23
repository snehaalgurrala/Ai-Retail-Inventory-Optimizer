from backend.services.recommendation_engine import (
    generate_stock_transfer_recommendations,
)


SOURCE_AGENT = "transfer_agent"


def _tag_source_agent(recommendations: list[dict]) -> list[dict]:
    """Mark recommendations with this agent as the source."""
    for recommendation in recommendations:
        recommendation["source_agent"] = SOURCE_AGENT
    return recommendations


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
    return _tag_source_agent(recommendations)


def run(
    inputs: dict,
    config: dict | None = None,
) -> list[dict]:
    """Entry point used by the orchestrator."""
    return analyze_transfer_opportunities(inputs, config)
