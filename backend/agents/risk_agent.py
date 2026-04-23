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


def analyze_risks(
    inputs: dict,
    config: dict | None = None,
) -> list[dict]:
    """Identify stockout, overstock, and supplier risks."""
    recommendations = []
    recommendations.extend(
        generate_supplier_risk_alerts(
            inputs.get("product_performance"),
            inputs.get("suppliers"),
            config,
        )
    )
    recommendations.extend(generate_overstock_alerts(inputs.get("overstock_items")))
    recommendations.extend(
        generate_stockout_prevention_alerts(
            inputs.get("stockout_risk_items"),
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
