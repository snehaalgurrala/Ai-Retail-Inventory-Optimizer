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
def get_current_inventory_summary(
    store_id: str = "",
    product_id: str = "",
    limit: int = 10,
) -> dict:
    """Return a compact summary of current inventory with a few representative rows."""
    inputs = _prepare_inputs()
    inventory = inputs.get("current_inventory", pd.DataFrame()).copy()

    if store_id and "store_id" in inventory.columns:
        inventory = inventory[inventory["store_id"].astype(str) == str(store_id)]
    if product_id and "product_id" in inventory.columns:
        inventory = inventory[inventory["product_id"].astype(str) == str(product_id)]

    stock_level = pd.to_numeric(inventory.get("stock_level", 0), errors="coerce").fillna(0)
    summary = {
        "row_count": int(len(inventory)),
        "product_count": int(inventory["product_id"].nunique()) if "product_id" in inventory.columns else 0,
        "store_count": int(inventory["store_id"].nunique()) if "store_id" in inventory.columns else 0,
        "total_stock_level": float(stock_level.sum()),
        "top_categories": (
            inventory["category"].fillna("").astype(str).value_counts().head(5).to_dict()
            if "category" in inventory.columns
            else {}
        ),
    }
    records = _safe_records(
        inventory.sort_values("stock_level", ascending=False),
        [
            "product_id",
            "product_name",
            "category",
            "store_id",
            "store_name",
            "stock_level",
            "inventory_reorder_threshold",
            "selling_price",
        ],
        limit=limit,
    )
    return {
        "tool_name": "get_current_inventory_summary",
        "summary": summary,
        "records": records,
    }


@tool
def get_low_stock_items(
    store_id: str = "",
    product_id: str = "",
    limit: int = 10,
) -> dict:
    """Return a compact low-stock view for grounded replenishment reasoning."""
    inputs = _prepare_inputs()
    low_stock_items = inputs.get("low_stock_items", pd.DataFrame()).copy()

    if store_id and "store_id" in low_stock_items.columns:
        low_stock_items = low_stock_items[
            low_stock_items["store_id"].astype(str) == str(store_id)
        ]
    if product_id and "product_id" in low_stock_items.columns:
        low_stock_items = low_stock_items[
            low_stock_items["product_id"].astype(str) == str(product_id)
        ]

    summary = {
        "row_count": int(len(low_stock_items)),
        "product_count": int(low_stock_items["product_id"].nunique()) if "product_id" in low_stock_items.columns else 0,
        "store_count": int(low_stock_items["store_id"].nunique()) if "store_id" in low_stock_items.columns else 0,
    }
    records = _safe_records(
        low_stock_items,
        [
            "product_id",
            "product_name",
            "store_id",
            "store_name",
            "stock_level",
            "effective_reorder_threshold",
            "days_of_stock_remaining",
            "reason",
            "evidence",
        ],
        limit=limit,
    )
    return {
        "tool_name": "get_low_stock_items",
        "summary": summary,
        "records": records,
    }


@tool
def get_dead_stock_candidates(
    store_id: str = "",
    product_id: str = "",
    limit: int = 10,
) -> dict:
    """Return a compact dead-stock candidate view with recent movement evidence."""
    inputs = _prepare_inputs()
    dead_stock_candidates = inputs.get("dead_stock_candidates", pd.DataFrame()).copy()

    if store_id and "store_id" in dead_stock_candidates.columns:
        dead_stock_candidates = dead_stock_candidates[
            dead_stock_candidates["store_id"].astype(str) == str(store_id)
        ]
    if product_id and "product_id" in dead_stock_candidates.columns:
        dead_stock_candidates = dead_stock_candidates[
            dead_stock_candidates["product_id"].astype(str) == str(product_id)
        ]

    summary = {
        "row_count": int(len(dead_stock_candidates)),
        "product_count": (
            int(dead_stock_candidates["product_id"].nunique())
            if "product_id" in dead_stock_candidates.columns
            else 0
        ),
        "store_count": (
            int(dead_stock_candidates["store_id"].nunique())
            if "store_id" in dead_stock_candidates.columns
            else 0
        ),
    }
    records = _safe_records(
        dead_stock_candidates,
        [
            "product_id",
            "product_name",
            "store_id",
            "store_name",
            "stock_level",
            "recent_30_day_quantity_sold",
            "shelf_life_days",
            "days_of_stock_remaining",
            "reason",
            "evidence",
        ],
        limit=limit,
    )
    return {
        "tool_name": "get_dead_stock_candidates",
        "summary": summary,
        "records": records,
    }


@tool
def get_store_stock_imbalance(limit: int = 10) -> dict:
    """Return compact store-level stock imbalance signals for transfer reasoning."""
    inputs = _prepare_inputs()
    inventory = inputs.get("current_inventory", pd.DataFrame()).copy()
    low_stock_items = inputs.get("low_stock_items", pd.DataFrame()).copy()
    overstock_items = inputs.get("overstock_items", pd.DataFrame()).copy()

    if inventory.empty:
        return {
            "tool_name": "get_store_stock_imbalance",
            "summary": {"store_count": 0, "imbalanced_store_count": 0},
            "records": [],
        }

    inventory["stock_level"] = pd.to_numeric(
        inventory.get("stock_level", 0),
        errors="coerce",
    ).fillna(0)
    store_totals = (
        inventory.groupby(["store_id", "store_name"], as_index=False)
        .agg(total_stock_level=("stock_level", "sum"))
    )
    low_counts = (
        low_stock_items.groupby("store_id", as_index=False)
        .size()
        .rename(columns={"size": "low_stock_count"})
        if not low_stock_items.empty and "store_id" in low_stock_items.columns
        else pd.DataFrame(columns=["store_id", "low_stock_count"])
    )
    overstock_counts = (
        overstock_items.groupby("store_id", as_index=False)
        .size()
        .rename(columns={"size": "overstock_count"})
        if not overstock_items.empty and "store_id" in overstock_items.columns
        else pd.DataFrame(columns=["store_id", "overstock_count"])
    )

    imbalance = store_totals.merge(low_counts, on="store_id", how="left")
    imbalance = imbalance.merge(overstock_counts, on="store_id", how="left")
    imbalance["low_stock_count"] = pd.to_numeric(
        imbalance.get("low_stock_count", 0),
        errors="coerce",
    ).fillna(0)
    imbalance["overstock_count"] = pd.to_numeric(
        imbalance.get("overstock_count", 0),
        errors="coerce",
    ).fillna(0)
    imbalance["imbalance_score"] = (
        imbalance["overstock_count"] - imbalance["low_stock_count"]
    ).abs()

    summary = {
        "store_count": int(len(imbalance)),
        "imbalanced_store_count": int((imbalance["imbalance_score"] > 0).sum()),
        "stores_with_low_stock": int((imbalance["low_stock_count"] > 0).sum()),
        "stores_with_overstock": int((imbalance["overstock_count"] > 0).sum()),
    }
    records = _safe_records(
        imbalance.sort_values(
            ["imbalance_score", "total_stock_level"],
            ascending=[False, False],
        ),
        [
            "store_id",
            "store_name",
            "total_stock_level",
            "low_stock_count",
            "overstock_count",
            "imbalance_score",
        ],
        limit=limit,
    )
    return {
        "tool_name": "get_store_stock_imbalance",
        "summary": summary,
        "records": records,
    }

