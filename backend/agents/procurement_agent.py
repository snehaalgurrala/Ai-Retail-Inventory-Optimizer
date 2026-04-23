import pandas as pd

from backend.services.recommendation_engine import (
    generate_reorder_recommendations,
    get_config,
)


SOURCE_AGENT = "procurement_agent"


def _number_column(df: pd.DataFrame, column: str) -> pd.Series:
    """Return a numeric column, or zeroes if it is missing."""
    if column not in df.columns:
        return pd.Series(0, index=df.index)
    return pd.to_numeric(df[column], errors="coerce").fillna(0)


def _tag_source_agent(recommendations: list[dict]) -> list[dict]:
    """Mark recommendations with this agent as the source."""
    for recommendation in recommendations:
        recommendation["source_agent"] = SOURCE_AGENT
    return recommendations


def _build_supplier_context(
    product_performance: pd.DataFrame,
    suppliers: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """Join product rows with supplier delivery and reliability signals."""
    if product_performance.empty or suppliers.empty:
        return pd.DataFrame()

    supplier_columns = ["supplier_id", "supplier_name"]
    if "reliability_score" in suppliers.columns:
        supplier_columns.append("reliability_score")
    if "avg_delivery_days" in suppliers.columns:
        supplier_columns.append("avg_delivery_days")

    supplier_view = suppliers[supplier_columns].drop_duplicates("supplier_id").copy()
    supplier_view["reliability_score"] = _number_column(
        supplier_view,
        "reliability_score",
    )
    supplier_view["avg_delivery_days"] = _number_column(
        supplier_view,
        "avg_delivery_days",
    )

    product_columns = ["product_id"]
    if "supplier_id" in product_performance.columns:
        product_columns.append("supplier_id")

    if len(product_columns) == 1:
        return pd.DataFrame()

    product_view = product_performance[product_columns].drop_duplicates("product_id")
    supplier_context = product_view.merge(
        supplier_view,
        on="supplier_id",
        how="left",
    )

    supplier_context["supplier_is_risky"] = (
        supplier_context["reliability_score"]
        < config["supplier_reliability_threshold"]
    ) | (
        supplier_context["avg_delivery_days"]
        >= config["supplier_delivery_days_threshold"]
    )

    return supplier_context


def analyze_procurement(
    inputs: dict,
    config: dict | None = None,
) -> list[dict]:
    """Create procurement-focused reorder recommendations with supplier context."""
    config = get_config(config)
    low_stock_items = inputs.get("low_stock_items", pd.DataFrame())
    recommendations = generate_reorder_recommendations(low_stock_items, config)

    if not recommendations:
        return recommendations

    supplier_context = _build_supplier_context(
        inputs.get("product_performance", pd.DataFrame()),
        inputs.get("suppliers", pd.DataFrame()),
        config,
    )
    if supplier_context.empty:
        return _tag_source_agent(recommendations)

    supplier_map = supplier_context.set_index("product_id").to_dict(orient="index")

    for recommendation in recommendations:
        product_id = recommendation.get("product_id", "")
        supplier_row = supplier_map.get(product_id)
        if not supplier_row:
            continue

        supplier_name = str(supplier_row.get("supplier_name", "") or "")
        reliability_score = supplier_row.get("reliability_score", 0)
        avg_delivery_days = supplier_row.get("avg_delivery_days", 0)
        supplier_is_risky = bool(supplier_row.get("supplier_is_risky", False))

        supplier_evidence = (
            f"supplier_name={supplier_name or 'unknown'}, "
            f"reliability_score={round(float(reliability_score), 2)}, "
            f"avg_delivery_days={round(float(avg_delivery_days), 2)}"
        )
        existing_evidence = str(recommendation.get("evidence", "")).strip()
        recommendation["evidence"] = (
            f"{existing_evidence}, {supplier_evidence}"
            if existing_evidence
            else supplier_evidence
        )

        if supplier_is_risky:
            recommendation["priority"] = "high"
            recommendation["reason"] = (
                "Stock is below the reorder point and supplier risk may delay replenishment."
            )
            recommendation["action"] = (
                f"{recommendation['action']} Review supplier timing before placing the order."
            )

    return _tag_source_agent(recommendations)


def run(
    inputs: dict,
    config: dict | None = None,
) -> list[dict]:
    """Entry point used by the LangGraph orchestrator."""
    return analyze_procurement(inputs, config)
