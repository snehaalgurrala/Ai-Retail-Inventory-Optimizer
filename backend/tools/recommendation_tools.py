from threading import Lock

import pandas as pd
from langchain_core.tools import tool

from backend.services.data_processor import build_processed_datasets
from backend.services.inventory_analyzer import build_inventory_analysis
from backend.services.recommendation_engine import (
    generate_reorder_recommendations,
    generate_supplier_risk_alerts,
    get_config,
    load_recommendation_inputs,
)


DATA_PREP_LOCK = Lock()


def _prepare_inputs(config: dict | None = None) -> dict[str, pd.DataFrame]:
    """Reload processed and analyzer data so tool outputs stay current."""
    config = get_config(config)
    with DATA_PREP_LOCK:
        build_processed_datasets()
        build_inventory_analysis(config)
        return load_recommendation_inputs()


def _safe_records(df: pd.DataFrame, columns: list[str], limit: int = 10) -> list[dict]:
    """Return compact JSON-friendly records."""
    if df.empty:
        return []

    available_columns = [column for column in columns if column in df.columns]
    records_df = df[available_columns].copy()
    if limit > 0:
        records_df = records_df.head(limit)

    records_df = records_df.replace([float("inf"), float("-inf")], None)
    records_df = records_df.astype(object).where(pd.notnull(records_df), None)
    return records_df.to_dict(orient="records")


@tool
def get_supplier_risk_summary(
    config: dict | None = None,
    product_id: str = "",
    limit: int = 10,
) -> dict:
    """Return compact supplier risk alerts grounded in product and supplier data."""
    config = get_config(config)
    inputs = _prepare_inputs(config)
    risk_rows = pd.DataFrame(
        generate_supplier_risk_alerts(
            inputs.get("product_performance", pd.DataFrame()),
            inputs.get("suppliers", pd.DataFrame()),
            config,
        )
    )

    if product_id and "product_id" in risk_rows.columns:
        risk_rows = risk_rows[risk_rows["product_id"].astype(str) == str(product_id)]

    summary = {
        "row_count": int(len(risk_rows)),
        "high_priority_count": (
            int(risk_rows["priority"].fillna("").astype(str).str.lower().eq("high").sum())
            if "priority" in risk_rows.columns
            else 0
        ),
    }
    records = _safe_records(
        risk_rows,
        [
            "product_id",
            "product_name",
            "priority",
            "action",
            "reason",
            "evidence",
        ],
        limit=limit,
    )
    return {
        "tool_name": "get_supplier_risk_summary",
        "summary": summary,
        "records": records,
    }


@tool
def get_procurement_candidates(
    config: dict | None = None,
    store_id: str = "",
    product_id: str = "",
    limit: int = 10,
) -> dict:
    """Return compact reorder candidates for procurement reasoning."""
    config = get_config(config)
    inputs = _prepare_inputs(config)
    procurement_rows = pd.DataFrame(
        generate_reorder_recommendations(
            inputs.get("low_stock_items", pd.DataFrame()),
            config,
        )
    )

    if store_id and "store_id" in procurement_rows.columns:
        procurement_rows = procurement_rows[
            procurement_rows["store_id"].astype(str) == str(store_id)
        ]
    if product_id and "product_id" in procurement_rows.columns:
        procurement_rows = procurement_rows[
            procurement_rows["product_id"].astype(str) == str(product_id)
        ]

    suggested_quantity = pd.to_numeric(
        procurement_rows.get("suggested_quantity", 0),
        errors="coerce",
    ).fillna(0)
    summary = {
        "row_count": int(len(procurement_rows)),
        "product_count": (
            int(procurement_rows["product_id"].nunique())
            if "product_id" in procurement_rows.columns
            else 0
        ),
        "store_count": (
            int(procurement_rows["store_id"].nunique())
            if "store_id" in procurement_rows.columns
            else 0
        ),
        "suggested_quantity_total": float(suggested_quantity.sum()),
    }
    records = _safe_records(
        procurement_rows,
        [
            "product_id",
            "product_name",
            "store_id",
            "priority",
            "action",
            "reason",
            "evidence",
            "suggested_quantity",
        ],
        limit=limit,
    )
    return {
        "tool_name": "get_procurement_candidates",
        "summary": summary,
        "records": records,
    }
