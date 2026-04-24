from typing import Any
from threading import Lock

import pandas as pd
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from backend.services.data_processor import (
    build_processed_datasets,
    current_inventory_df,
    product_performance_df,
    sales_summary_df,
)
from backend.services.inventory_analyzer import build_inventory_analysis
from backend.services.recommendation_engine import (
    generate_clearance_recommendations,
    generate_discount_recommendations,
    generate_overstock_alerts,
    generate_reorder_recommendations,
    generate_stock_transfer_recommendations,
    generate_stockout_prevention_alerts,
    generate_supplier_risk_alerts,
    get_config,
    load_recommendation_inputs,
)
from backend.tools.inventory_tools import (
    get_current_inventory_summary,
    get_dead_stock_candidates,
    get_low_stock_items,
    get_store_stock_imbalance,
)
from backend.tools.recommendation_tools import (
    get_procurement_candidates,
    get_supplier_risk_summary,
)
from backend.tools.sales_tools import get_sales_summary


DEFAULT_LIMIT = 100
DATA_PREP_LOCK = Lock()


class DataFilterInput(BaseModel):
    product_id: str = ""
    store_id: str = ""
    supplier_id: str = ""
    limit: int = DEFAULT_LIMIT


class RecommendationToolInput(BaseModel):
    config: dict[str, Any] = Field(default_factory=dict)
    product_id: str = ""
    store_id: str = ""
    limit: int = DEFAULT_LIMIT


def _prepare_inputs(config: dict | None = None) -> dict[str, pd.DataFrame]:
    """Build processed data and analysis outputs before tool execution."""
    config = get_config(config)
    with DATA_PREP_LOCK:
        build_processed_datasets()
        build_inventory_analysis(config)
        return load_recommendation_inputs()


def _clean_records(df: pd.DataFrame, limit: int = DEFAULT_LIMIT) -> list[dict]:
    """Convert a dataframe into JSON-friendly records."""
    if df.empty:
        return []

    records_df = df.copy()
    if limit > 0:
        records_df = records_df.head(limit)

    records_df = records_df.replace([float("inf"), float("-inf")], None)
    records_df = records_df.astype(object).where(pd.notnull(records_df), None)

    for column in records_df.columns:
        if pd.api.types.is_datetime64_any_dtype(records_df[column]):
            records_df[column] = records_df[column].astype(str)

    return records_df.to_dict(orient="records")


def _filter_dataframe(
    df: pd.DataFrame,
    product_id: str = "",
    store_id: str = "",
    supplier_id: str = "",
) -> pd.DataFrame:
    """Apply simple optional filters to a dataframe."""
    filtered = df.copy()

    if product_id and "product_id" in filtered.columns:
        filtered = filtered[filtered["product_id"].astype(str) == str(product_id)]
    if store_id and "store_id" in filtered.columns:
        filtered = filtered[filtered["store_id"].astype(str) == str(store_id)]
    if supplier_id and "supplier_id" in filtered.columns:
        filtered = filtered[filtered["supplier_id"].astype(str) == str(supplier_id)]

    return filtered


def _build_tool_output(
    tool_name: str,
    df: pd.DataFrame,
    limit: int,
    summary: dict | None = None,
) -> dict:
    """Return a structured tool response."""
    return {
        "tool_name": tool_name,
        "count": int(len(df)),
        "records": _clean_records(df, limit=limit),
        "summary": summary or {},
    }


def _get_inventory_data(
    product_id: str = "",
    store_id: str = "",
    supplier_id: str = "",
    limit: int = DEFAULT_LIMIT,
) -> dict:
    df = current_inventory_df()
    df = _filter_dataframe(df, product_id, store_id, supplier_id)
    summary = {
        "total_stock_level": float(
            pd.to_numeric(df.get("stock_level", 0), errors="coerce").fillna(0).sum()
        ),
        "store_count": int(df["store_id"].nunique()) if "store_id" in df.columns else 0,
        "product_count": (
            int(df["product_id"].nunique()) if "product_id" in df.columns else 0
        ),
    }
    return _build_tool_output("get_inventory_data", df, limit, summary)


def _get_sales_trend(
    product_id: str = "",
    store_id: str = "",
    supplier_id: str = "",
    limit: int = DEFAULT_LIMIT,
) -> dict:
    df = sales_summary_df()
    df = _filter_dataframe(df, product_id, store_id, supplier_id)
    summary = {
        "total_quantity_sold": float(
            pd.to_numeric(
                df.get("total_quantity_sold", 0),
                errors="coerce",
            ).fillna(0).sum()
        ),
        "total_revenue": float(
            pd.to_numeric(df.get("total_revenue", 0), errors="coerce").fillna(0).sum()
        ),
        "store_count": int(df["store_id"].nunique()) if "store_id" in df.columns else 0,
    }
    return _build_tool_output("get_sales_trend", df, limit, summary)


def _get_product_performance(
    product_id: str = "",
    store_id: str = "",
    supplier_id: str = "",
    limit: int = DEFAULT_LIMIT,
) -> dict:
    df = product_performance_df()
    df = _filter_dataframe(df, product_id, store_id, supplier_id)
    summary = {
        "product_count": (
            int(df["product_id"].nunique()) if "product_id" in df.columns else 0
        ),
        "total_quantity_sold": float(
            pd.to_numeric(
                df.get("total_quantity_sold", 0),
                errors="coerce",
            ).fillna(0).sum()
        ),
        "current_stock_level": float(
            pd.to_numeric(
                df.get("current_stock_level", 0),
                errors="coerce",
            ).fillna(0).sum()
        ),
    }
    return _build_tool_output("get_product_performance", df, limit, summary)


def _analyze_risk(
    config: dict[str, Any] | None = None,
    product_id: str = "",
    store_id: str = "",
    limit: int = DEFAULT_LIMIT,
) -> dict:
    inputs = _prepare_inputs(config)
    recommendations = []
    recommendations.extend(
        generate_supplier_risk_alerts(
            inputs.get("product_performance", pd.DataFrame()),
            inputs.get("suppliers", pd.DataFrame()),
            config,
        )
    )
    recommendations.extend(
        generate_overstock_alerts(inputs.get("overstock_items", pd.DataFrame()))
    )
    recommendations.extend(
        generate_stockout_prevention_alerts(
            inputs.get("stockout_risk_items", pd.DataFrame()),
            config,
        )
    )

    df = pd.DataFrame(recommendations)
    if df.empty:
        return _build_tool_output("analyze_risk", df, limit, {"by_type": {}})

    df = _filter_dataframe(df, product_id, store_id)
    summary = {
        "by_type": df["recommendation_type"].value_counts().to_dict()
        if "recommendation_type" in df.columns
        else {},
    }
    return _build_tool_output("analyze_risk", df, limit, summary)


def _recommend_discount(
    config: dict[str, Any] | None = None,
    product_id: str = "",
    store_id: str = "",
    limit: int = DEFAULT_LIMIT,
) -> dict:
    inputs = _prepare_inputs(config)
    recommendations = []
    recommendations.extend(
        generate_discount_recommendations(
            inputs.get("slow_moving_items", pd.DataFrame()),
            inputs.get("overstock_items", pd.DataFrame()),
            config,
        )
    )
    recommendations.extend(
        generate_clearance_recommendations(
            inputs.get("dead_stock_candidates", pd.DataFrame()),
            config,
        )
    )

    df = pd.DataFrame(recommendations)
    if df.empty:
        return _build_tool_output("recommend_discount", df, limit, {"by_type": {}})

    df = _filter_dataframe(df, product_id, store_id)
    summary = {
        "by_type": df["recommendation_type"].value_counts().to_dict()
        if "recommendation_type" in df.columns
        else {},
    }
    return _build_tool_output("recommend_discount", df, limit, summary)


def _recommend_transfer(
    config: dict[str, Any] | None = None,
    product_id: str = "",
    store_id: str = "",
    limit: int = DEFAULT_LIMIT,
) -> dict:
    inputs = _prepare_inputs(config)
    df = pd.DataFrame(
        generate_stock_transfer_recommendations(
            inputs.get("current_inventory", pd.DataFrame()),
            inputs.get("low_stock_items", pd.DataFrame()),
            config,
        )
    )
    if df.empty:
        return _build_tool_output("recommend_transfer", df, limit, {})

    df = _filter_dataframe(df, product_id, store_id)
    summary = {
        "suggested_quantity_total": float(
            pd.to_numeric(
                df.get("suggested_quantity", 0),
                errors="coerce",
            ).fillna(0).sum()
        ),
    }
    return _build_tool_output("recommend_transfer", df, limit, summary)


def _recommend_procurement(
    config: dict[str, Any] | None = None,
    product_id: str = "",
    store_id: str = "",
    limit: int = DEFAULT_LIMIT,
) -> dict:
    inputs = _prepare_inputs(config)
    df = pd.DataFrame(
        generate_reorder_recommendations(
            inputs.get("low_stock_items", pd.DataFrame()),
            config,
        )
    )
    if df.empty:
        return _build_tool_output("recommend_procurement", df, limit, {})

    df = _filter_dataframe(df, product_id, store_id)
    summary = {
        "suggested_quantity_total": float(
            pd.to_numeric(
                df.get("suggested_quantity", 0),
                errors="coerce",
            ).fillna(0).sum()
        ),
    }
    return _build_tool_output("recommend_procurement", df, limit, summary)


get_inventory_data = StructuredTool.from_function(
    func=_get_inventory_data,
    name="get_inventory_data",
    description=(
        "Get current inventory data with optional product_id or store_id filters."
    ),
    args_schema=DataFilterInput,
)

get_sales_trend = StructuredTool.from_function(
    func=_get_sales_trend,
    name="get_sales_trend",
    description=(
        "Get processed product-store sales trend data with revenue and recent sales metrics."
    ),
    args_schema=DataFilterInput,
)

get_product_performance = StructuredTool.from_function(
    func=_get_product_performance,
    name="get_product_performance",
    description=(
        "Get product-level performance data with inventory, sales, and supplier fields."
    ),
    args_schema=DataFilterInput,
)

analyze_risk = StructuredTool.from_function(
    func=_analyze_risk,
    name="analyze_risk",
    description=(
        "Create supplier risk, overstock, and stockout prevention recommendation candidates."
    ),
    args_schema=RecommendationToolInput,
)

recommend_discount = StructuredTool.from_function(
    func=_recommend_discount,
    name="recommend_discount",
    description=(
        "Create discount and clearance recommendation candidates from pricing-related analysis."
    ),
    args_schema=RecommendationToolInput,
)

recommend_transfer = StructuredTool.from_function(
    func=_recommend_transfer,
    name="recommend_transfer",
    description="Create stock transfer recommendation candidates across stores.",
    args_schema=RecommendationToolInput,
)

recommend_procurement = StructuredTool.from_function(
    func=_recommend_procurement,
    name="recommend_procurement",
    description="Create reorder and procurement recommendation candidates.",
    args_schema=RecommendationToolInput,
)


TOOL_REGISTRY = {
    "get_inventory_data": get_inventory_data,
    "get_sales_trend": get_sales_trend,
    "get_product_performance": get_product_performance,
    "get_current_inventory_summary": get_current_inventory_summary,
    "get_sales_summary": get_sales_summary,
    "get_low_stock_items": get_low_stock_items,
    "get_dead_stock_candidates": get_dead_stock_candidates,
    "get_store_stock_imbalance": get_store_stock_imbalance,
    "get_supplier_risk_summary": get_supplier_risk_summary,
    "get_procurement_candidates": get_procurement_candidates,
    "analyze_risk": analyze_risk,
    "recommend_discount": recommend_discount,
    "recommend_transfer": recommend_transfer,
    "recommend_procurement": recommend_procurement,
}


AGENT_TOOLBOX = {
    "pricing_agent": [
        "get_sales_summary",
        "get_dead_stock_candidates",
        "get_inventory_data",
        "get_sales_trend",
        "recommend_discount",
    ],
    "transfer_agent": [
        "get_current_inventory_summary",
        "get_store_stock_imbalance",
        "get_low_stock_items",
        "get_inventory_data",
        "get_sales_trend",
        "recommend_transfer",
    ],
    "risk_agent": [
        "get_supplier_risk_summary",
        "get_low_stock_items",
        "get_dead_stock_candidates",
        "get_product_performance",
        "get_inventory_data",
        "analyze_risk",
    ],
    "procurement_agent": [
        "get_procurement_candidates",
        "get_supplier_risk_summary",
        "get_low_stock_items",
        "get_inventory_data",
        "get_product_performance",
        "recommend_procurement",
    ],
}


def get_agent_tools(agent_name: str) -> list[StructuredTool]:
    """Return the LangChain tool objects available to one agent."""
    return [
        TOOL_REGISTRY[tool_name]
        for tool_name in AGENT_TOOLBOX.get(agent_name, [])
        if tool_name in TOOL_REGISTRY
    ]


def describe_agent_tools(agent_name: str) -> list[dict[str, str]]:
    """Return tool metadata for LLM tool selection."""
    return [
        {
            "name": tool.name,
            "description": tool.description,
        }
        for tool in get_agent_tools(agent_name)
    ]


def invoke_agent_tool(tool_name: str, payload: dict | None = None) -> dict:
    """Invoke one tool by name and return its structured output."""
    if tool_name not in TOOL_REGISTRY:
        raise KeyError(f"Unknown tool: {tool_name}")
    return TOOL_REGISTRY[tool_name].invoke(payload or {})


def tool_records_to_dataframe(tool_output: dict) -> pd.DataFrame:
    """Convert a tool response back into a dataframe for internal agent use."""
    return pd.DataFrame(tool_output.get("records", []))
