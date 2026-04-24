from math import ceil
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

INVENTORY_PATH = RAW_DATA_DIR / "inventory.csv"
PRODUCTS_PATH = RAW_DATA_DIR / "products.csv"
STORES_PATH = RAW_DATA_DIR / "stores.csv"
SALES_PATH = RAW_DATA_DIR / "sales.csv"
SUPPLIERS_PATH = RAW_DATA_DIR / "suppliers.csv"
LOW_STOCK_OUTPUT_PATH = PROCESSED_DATA_DIR / "low_stock_alerts.csv"


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _sales_velocity_lookup(sales_df: pd.DataFrame) -> pd.DataFrame:
    """Build recent per-product-per-store sales velocity for reorder suggestions."""
    if sales_df.empty or not {"date", "product_id", "store_id", "quantity_sold"}.issubset(
        sales_df.columns
    ):
        return pd.DataFrame(columns=["product_id", "store_id", "recent_daily_sales_velocity"])

    sales_view = sales_df.copy()
    sales_view["date"] = pd.to_datetime(sales_view["date"], errors="coerce")
    sales_view["quantity_sold"] = pd.to_numeric(
        sales_view["quantity_sold"], errors="coerce"
    ).fillna(0)
    sales_view = sales_view.dropna(subset=["date"])
    if sales_view.empty:
        return pd.DataFrame(columns=["product_id", "store_id", "recent_daily_sales_velocity"])

    latest_date = sales_view["date"].max()
    recent_sales = sales_view[
        sales_view["date"] >= latest_date - pd.Timedelta(days=30)
    ].copy()
    if recent_sales.empty:
        return pd.DataFrame(columns=["product_id", "store_id", "recent_daily_sales_velocity"])

    velocity = (
        recent_sales.groupby(["product_id", "store_id"], as_index=False)["quantity_sold"]
        .sum()
        .rename(columns={"quantity_sold": "recent_30_day_quantity_sold"})
    )
    velocity["recent_daily_sales_velocity"] = (
        pd.to_numeric(velocity["recent_30_day_quantity_sold"], errors="coerce").fillna(0) / 30
    )
    return velocity[["product_id", "store_id", "recent_daily_sales_velocity"]]


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
    if current_quantity < reorder_threshold:
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
    """Return product-store rows where current stock is at or below threshold."""
    inventory_df = _safe_read_csv(INVENTORY_PATH)
    products_df = _safe_read_csv(PRODUCTS_PATH)
    stores_df = _safe_read_csv(STORES_PATH)
    sales_df = _safe_read_csv(SALES_PATH)
    suppliers_df = _safe_read_csv(SUPPLIERS_PATH)

    required_inventory_columns = {"product_id", "store_id", "stock_level"}
    if inventory_df.empty or not required_inventory_columns.issubset(inventory_df.columns):
        return pd.DataFrame()

    inventory_view = inventory_df.copy()
    inventory_view["stock_level"] = pd.to_numeric(
        inventory_view["stock_level"], errors="coerce"
    ).fillna(0)

    if "reorder_threshold" in inventory_view.columns:
        inventory_view["inventory_reorder_threshold"] = pd.to_numeric(
            inventory_view["reorder_threshold"], errors="coerce"
        ).fillna(0)
        inventory_view = inventory_view.drop(columns=["reorder_threshold"])
    else:
        inventory_view["inventory_reorder_threshold"] = 0

    if not products_df.empty:
        products_view = products_df.copy()
        if "reorder_threshold" in products_view.columns:
            products_view["product_reorder_threshold"] = pd.to_numeric(
                products_view["reorder_threshold"], errors="coerce"
            ).fillna(0)
            products_view = products_view.drop(columns=["reorder_threshold"])
        inventory_view = inventory_view.merge(products_view, on="product_id", how="left")

    if not stores_df.empty:
        inventory_view = inventory_view.merge(stores_df, on="store_id", how="left")

    if not suppliers_df.empty and "supplier_id" in inventory_view.columns:
        inventory_view = inventory_view.merge(suppliers_df, on="supplier_id", how="left")

    velocity_lookup = _sales_velocity_lookup(sales_df)
    if not velocity_lookup.empty:
        inventory_view = inventory_view.merge(
            velocity_lookup,
            on=["product_id", "store_id"],
            how="left",
        )

    inventory_view["current_quantity"] = pd.to_numeric(
        inventory_view["stock_level"], errors="coerce"
    ).fillna(0)
    inventory_view["reorder_threshold"] = pd.to_numeric(
        inventory_view.get("inventory_reorder_threshold", 0), errors="coerce"
    ).fillna(0)
    if "product_reorder_threshold" in inventory_view.columns:
        inventory_view["reorder_threshold"] = inventory_view["reorder_threshold"].where(
            inventory_view["reorder_threshold"] > 0,
            pd.to_numeric(
                inventory_view["product_reorder_threshold"], errors="coerce"
            ).fillna(0),
        )

    low_stock_df = inventory_view[
        inventory_view["current_quantity"] <= inventory_view["reorder_threshold"]
    ].copy()

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
    low_stock_df["suggested_reorder_quantity"] = low_stock_df.apply(
        lambda row: suggest_reorder_quantity(
            row.get("current_quantity", 0),
            row.get("reorder_threshold", 0),
            row.get("recent_daily_sales_velocity", 0),
        ),
        axis=1,
    )
    low_stock_df["priority"] = low_stock_df.apply(
        lambda row: calculate_priority(
            row.get("current_quantity", 0),
            row.get("reorder_threshold", 0),
        ),
        axis=1,
    )

    priority_rank = {"High": 0, "Medium": 1, "Low": 2}
    low_stock_df["_priority_rank"] = (
        low_stock_df["priority"].map(priority_rank).fillna(3)
    )
    low_stock_df = low_stock_df.sort_values(
        ["_priority_rank", "shortage_quantity", "current_quantity"],
        ascending=[True, False, True],
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
    ]
    available_columns = [column for column in preferred_columns if column in low_stock_df.columns]
    low_stock_df = low_stock_df[available_columns].reset_index(drop=True)

    if save_output:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        low_stock_df.to_csv(LOW_STOCK_OUTPUT_PATH, index=False)

    return low_stock_df
