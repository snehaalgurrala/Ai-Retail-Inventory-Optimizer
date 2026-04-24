from threading import Lock

import pandas as pd
from langchain_core.tools import tool

from backend.services.data_processor import build_processed_datasets
from backend.services.inventory_analyzer import build_inventory_analysis
from backend.services.recommendation_engine import get_config, load_recommendation_inputs


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
def get_sales_summary(
    store_id: str = "",
    product_id: str = "",
    limit: int = 10,
) -> dict:
    """Return a compact sales summary with totals and recent movement highlights."""
    inputs = _prepare_inputs()
    sales_summary = inputs.get("current_inventory", pd.DataFrame()).merge(
        inputs.get("product_performance", pd.DataFrame())[
            [
                column
                for column in [
                    "product_id",
                    "total_quantity_sold",
                    "total_revenue",
                    "recent_7_day_quantity_sold",
                    "recent_30_day_quantity_sold",
                    "last_sale_date",
                ]
                if column in inputs.get("product_performance", pd.DataFrame()).columns
            ]
        ],
        on="product_id",
        how="left",
    )

    if store_id and "store_id" in sales_summary.columns:
        sales_summary = sales_summary[sales_summary["store_id"].astype(str) == str(store_id)]
    if product_id and "product_id" in sales_summary.columns:
        sales_summary = sales_summary[
            sales_summary["product_id"].astype(str) == str(product_id)
        ]

    total_quantity_sold = pd.to_numeric(
        sales_summary.get("total_quantity_sold", 0),
        errors="coerce",
    ).fillna(0)
    total_revenue = pd.to_numeric(
        sales_summary.get("total_revenue", 0),
        errors="coerce",
    ).fillna(0)
    summary = {
        "row_count": int(len(sales_summary)),
        "product_count": int(sales_summary["product_id"].nunique()) if "product_id" in sales_summary.columns else 0,
        "store_count": int(sales_summary["store_id"].nunique()) if "store_id" in sales_summary.columns else 0,
        "total_quantity_sold": float(total_quantity_sold.sum()),
        "total_revenue": float(total_revenue.sum()),
    }
    records = _safe_records(
        sales_summary.sort_values(
            ["recent_7_day_quantity_sold", "total_quantity_sold"],
            ascending=[False, False],
        ),
        [
            "product_id",
            "product_name",
            "store_id",
            "store_name",
            "category",
            "recent_7_day_quantity_sold",
            "recent_30_day_quantity_sold",
            "total_quantity_sold",
            "total_revenue",
            "last_sale_date",
        ],
        limit=limit,
    )
    return {
        "tool_name": "get_sales_summary",
        "summary": summary,
        "records": records,
    }

