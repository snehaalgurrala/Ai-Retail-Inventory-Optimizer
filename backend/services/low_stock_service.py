from pathlib import Path

import pandas as pd

from backend.db import repository
from backend.services.inventory_prediction_service import (
    DEPLETION_ALERT_DAYS,
    DEFAULT_SUPPLIER_LEAD_DAYS,
    DEMAND_SPIKE_PERCENT,
    SAFETY_FACTOR,
    get_predictive_inventory_alerts,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

INVENTORY_PATH = RAW_DATA_DIR / "inventory.csv"
PRODUCTS_PATH = RAW_DATA_DIR / "products.csv"
STORES_PATH = RAW_DATA_DIR / "stores.csv"
SALES_PATH = RAW_DATA_DIR / "sales.csv"
SUPPLIERS_PATH = RAW_DATA_DIR / "suppliers.csv"
LOW_STOCK_OUTPUT_PATH = PROCESSED_DATA_DIR / "low_stock_alerts.csv"


def calculate_priority(current_quantity: float, reorder_threshold: float) -> str:
    """Classify low-stock urgency from the current gap against threshold."""
    current_quantity = float(current_quantity or 0)
    reorder_threshold = float(reorder_threshold or 0)
    if current_quantity <= 0:
        return "High"
    if reorder_threshold <= 0:
        return "Medium"

    ratio = current_quantity / reorder_threshold if reorder_threshold else 0
    if ratio <= 0.5:
        return "High"
    # At or below the reorder point the item needs replenishment — only stock that
    # is strictly above the reorder point is "Low" (healthy) priority.
    if current_quantity <= reorder_threshold:
        return "Medium"
    return "Low"


def suggest_reorder_quantity(
    current_quantity: float,
    reorder_threshold: float,
    recent_daily_sales_velocity: float | None = None,
) -> int:
    """Suggest a reorder quantity using real sales movement when available."""
    current_quantity = float(current_quantity or 0)
    reorder_threshold = float(reorder_threshold or 0)
    fallback_quantity = max(int(ceil((reorder_threshold * 2) - current_quantity)), 0)

    velocity = float(recent_daily_sales_velocity or 0)
    if velocity <= 0:
        return fallback_quantity

    fourteen_day_cover = ceil(velocity * 14)
    data_grounded_quantity = max(int(reorder_threshold + fourteen_day_cover - current_quantity), 0)
    return max(fallback_quantity, data_grounded_quantity)


def get_low_stock_items(save_output: bool = True) -> pd.DataFrame:
    """Return predictive product-store inventory alerts with threshold fallback."""
    inventory_df = repository.load_inventory(safe=True)
    products_df = repository.load_products(safe=True)
    stores_df = repository.load_stores(safe=True)
    sales_df = repository.load_sales(safe=True)
    suppliers_df = repository.load_suppliers(safe=True)

    required_inventory_columns = {"product_id", "store_id", "stock_level"}
    if inventory_df.empty or not required_inventory_columns.issubset(inventory_df.columns):
        return pd.DataFrame()

    low_stock_df = get_predictive_inventory_alerts(
        inventory_df,
        products_df,
        stores_df,
        sales_df,
        suppliers_df,
        config={
            "depletion_alert_days": DEPLETION_ALERT_DAYS,
            "demand_spike_percent": DEMAND_SPIKE_PERCENT,
            "safety_factor": SAFETY_FACTOR,
            "default_supplier_lead_days": DEFAULT_SUPPLIER_LEAD_DAYS,
        },
    )

    if low_stock_df.empty:
        if save_output:
            PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                columns=[
                    "product_id",
                    "product_name",
                    "category",
                    "store_id",
                    "store_name",
                    "city",
                    "supplier_id",
                    "supplier_name",
                    "current_quantity",
                    "reorder_threshold",
                    "shortage_quantity",
                    "suggested_reorder_quantity",
                    "recent_daily_sales_velocity",
                    "priority",
                    "avg_daily_sales",
                    "predicted_days_remaining",
                    "demand_trend",
                    "risk_score",
                    "risk_category",
                    "confidence_level",
                    "suggested_reorder_qty",
                    "suggested_transfer_branch",
                    "alert_reason",
                    "ai_alert_message",
                    "depletion_window",
                    "urgency_label",
                    "depletion_tooltip",
                ]
            ).to_csv(LOW_STOCK_OUTPUT_PATH, index=False)
        return low_stock_df

    low_stock_df["shortage_quantity"] = (
        pd.to_numeric(low_stock_df["reorder_threshold"], errors="coerce").fillna(0)
        - pd.to_numeric(low_stock_df["current_quantity"], errors="coerce").fillna(0)
    ).clip(lower=0)
    low_stock_df["recent_daily_sales_velocity"] = pd.to_numeric(
        low_stock_df.get("recent_daily_sales_velocity", 0), errors="coerce"
    ).fillna(0)
    low_stock_df["priority"] = low_stock_df["risk_category"].map(
        {"Critical": "High", "High": "High", "Medium": "Medium", "Healthy": "Low"}
    ).fillna("Medium")

    priority_rank = {"High": 0, "Medium": 1, "Low": 2}
    low_stock_df["_priority_rank"] = (
        low_stock_df["priority"].map(priority_rank).fillna(3)
    )
    low_stock_df = low_stock_df.sort_values(
        ["_priority_rank", "predicted_days_remaining", "risk_score"],
        ascending=[True, True, False],
    )

    preferred_columns = [
        "product_id",
        "product_name",
        "category",
        "store_id",
        "store_name",
        "city",
        "supplier_id",
        "supplier_name",
        "current_quantity",
        "reorder_threshold",
        "shortage_quantity",
        "suggested_reorder_quantity",
        "recent_daily_sales_velocity",
        "priority",
        "avg_daily_sales",
        "moving_avg_daily_sales",
        "predicted_days_remaining",
        "demand_trend",
        "demand_spike",
        "risk_score",
        "risk_category",
        "confidence_level",
        "suggested_reorder_qty",
        "supplier_lead_time_days",
        "suggested_transfer_branch",
        "suggested_transfer_branch_id",
        "suggested_transfer_qty",
        "transfer_recommendation",
        "alert_reason",
        "ai_alert_message",
        "depletion_window",
        "urgency_label",
        "depletion_tooltip",
        "fallback_threshold_alert",
    ]
    available_columns = [column for column in preferred_columns if column in low_stock_df.columns]
    low_stock_df = low_stock_df[available_columns].reset_index(drop=True)

    if save_output:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        low_stock_df.to_csv(LOW_STOCK_OUTPUT_PATH, index=False)

    return low_stock_df
