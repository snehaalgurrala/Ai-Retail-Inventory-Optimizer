from pathlib import Path

import pandas as pd

from backend.services.low_stock_service import get_low_stock_items


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

INVENTORY_PATH = RAW_DATA_DIR / "inventory.csv"
PRODUCTS_PATH = RAW_DATA_DIR / "products.csv"
STORES_PATH = RAW_DATA_DIR / "stores.csv"


SURPLUS_MULTIPLIER = 2


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _number_column(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(0, index=df.index)
    return pd.to_numeric(df[column], errors="coerce").fillna(0)


def _text(value, fallback: str = "") -> str:
    if pd.isna(value):
        return fallback
    text = str(value or "").strip()
    return text or fallback


def _load_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        _safe_read_csv(INVENTORY_PATH),
        _safe_read_csv(PRODUCTS_PATH),
        _safe_read_csv(STORES_PATH),
    )


def _prepare_inventory(
    inventory: pd.DataFrame,
    products: pd.DataFrame,
    stores: pd.DataFrame,
) -> pd.DataFrame:
    if inventory.empty or not {"product_id", "store_id", "stock_level"}.issubset(inventory.columns):
        return pd.DataFrame()

    inv = inventory.copy()
    inv["product_id"] = inv["product_id"].astype(str)
    inv["store_id"] = inv["store_id"].astype(str)
    inv["current_quantity"] = _number_column(inv, "stock_level")
    inv["inventory_reorder_threshold"] = _number_column(inv, "reorder_threshold")

    if not products.empty and "product_id" in products.columns:
        product_columns = [
            column
            for column in ["product_id", "product_name", "category", "reorder_threshold"]
            if column in products.columns
        ]
        product_view = products[product_columns].copy()
        product_view["product_id"] = product_view["product_id"].astype(str)
        if "reorder_threshold" in product_view.columns:
            product_view = product_view.rename(columns={"reorder_threshold": "product_reorder_threshold"})
        inv = inv.merge(product_view, on="product_id", how="left")

    if not stores.empty and "store_id" in stores.columns:
        store_columns = [
            column for column in ["store_id", "store_name", "city"] if column in stores.columns
        ]
        store_view = stores[store_columns].copy()
        store_view["store_id"] = store_view["store_id"].astype(str)
        inv = inv.merge(store_view, on="store_id", how="left")

    inv["reorder_threshold"] = inv["inventory_reorder_threshold"]
    if "product_reorder_threshold" in inv.columns:
        inv["reorder_threshold"] = inv["reorder_threshold"].where(
            inv["reorder_threshold"] > 0,
            _number_column(inv, "product_reorder_threshold"),
        )
    inv["product_name"] = inv.get("product_name", inv["product_id"]).fillna(inv["product_id"]).astype(str)
    inv["category"] = inv.get("category", "").fillna("").astype(str)
    inv["store_name"] = inv.get("store_name", inv["store_id"]).fillna(inv["store_id"]).astype(str)
    inv["city"] = inv.get("city", "").fillna("").astype(str)
    return inv


def get_surplus_stock_items(
    inventory: pd.DataFrame | None = None,
    products: pd.DataFrame | None = None,
    stores: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Detect product-store rows where stock is at least twice the reorder threshold."""
    if inventory is None or products is None or stores is None:
        inventory, products, stores = _load_frames()

    inv = _prepare_inventory(inventory, products, stores)
    columns = [
        "product_id",
        "product_name",
        "category",
        "store_id",
        "store_name",
        "city",
        "current_quantity",
        "reorder_threshold",
        "surplus_quantity",
        "opportunity_level",
        "priority",
        "reason",
    ]
    if inv.empty:
        return pd.DataFrame(columns=columns)

    inv["surplus_quantity"] = (inv["current_quantity"] - inv["reorder_threshold"]).clip(lower=0)
    surplus = inv[
        (inv["reorder_threshold"] > 0)
        & (inv["current_quantity"] >= inv["reorder_threshold"] * SURPLUS_MULTIPLIER)
    ].copy()
    if surplus.empty:
        return pd.DataFrame(columns=columns)

    ratio = surplus["current_quantity"] / surplus["reorder_threshold"].replace(0, pd.NA)
    surplus["opportunity_level"] = ratio.map(
        lambda value: "High" if pd.notna(value) and value >= 3 else "Medium"
    )
    surplus["priority"] = surplus["opportunity_level"]
    surplus["current_quantity"] = surplus["current_quantity"].round().astype(int)
    surplus["reorder_threshold"] = surplus["reorder_threshold"].round().astype(int)
    surplus["surplus_quantity"] = surplus["surplus_quantity"].round().astype(int)
    surplus["reason"] = surplus.apply(
        lambda row: (
            f"{row['product_name']} has {int(row['current_quantity'])} units at "
            f"{row['store_name']}, above the {int(row['reorder_threshold'])} unit threshold."
        ),
        axis=1,
    )
    return surplus[columns].sort_values(
        ["opportunity_level", "surplus_quantity", "product_name"],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def get_alternative_availability_for_low_stock(
    inventory: pd.DataFrame | None = None,
    products: pd.DataFrame | None = None,
    stores: pd.DataFrame | None = None,
    low_stock_items: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Find same-product transfer sources and same-category alternatives for low stock."""
    if inventory is None or products is None or stores is None:
        inventory, products, stores = _load_frames()
    if low_stock_items is None:
        low_stock_items = get_low_stock_items(save_output=False)

    inv = _prepare_inventory(inventory, products, stores)
    surplus = get_surplus_stock_items(inventory, products, stores)
    columns = [
        "low_stock_product",
        "low_stock_store",
        "alternative_product",
        "alternative_store",
        "available_quantity",
        "alternative_type",
        "suggested_action",
        "reason",
    ]
    if inv.empty or low_stock_items.empty:
        return pd.DataFrame(columns=columns)

    low_stock = low_stock_items.copy()
    low_stock["product_id"] = low_stock.get("product_id", "").astype(str)
    low_stock["store_id"] = low_stock.get("store_id", "").astype(str)
    if "category" not in low_stock.columns and not products.empty:
        low_stock = low_stock.merge(
            products[["product_id", "product_name", "category"]],
            on="product_id",
            how="left",
        )
    if "store_name" not in low_stock.columns and not stores.empty:
        low_stock = low_stock.merge(
            stores[["store_id", "store_name", "city"]],
            on="store_id",
            how="left",
        )

    rows: list[dict] = []
    healthy_stock = inv[
        (inv["current_quantity"] > 0)
        & (
            (inv["reorder_threshold"] <= 0)
            | (inv["current_quantity"] >= inv["reorder_threshold"] * SURPLUS_MULTIPLIER)
        )
    ].copy()

    for _, low_row in low_stock.iterrows():
        low_product_id = _text(low_row.get("product_id"))
        low_store_id = _text(low_row.get("store_id"))
        low_product = _text(low_row.get("product_name"), low_product_id)
        low_store = _text(low_row.get("store_name"), low_store_id)
        category = _text(low_row.get("category"))
        if not low_product_id or not low_store_id:
            continue

        same_product = surplus[
            (surplus["product_id"].astype(str) == low_product_id)
            & (surplus["store_id"].astype(str) != low_store_id)
        ].copy()
        for _, source in same_product.iterrows():
            qty = int(source.get("current_quantity", 0))
            source_store = _text(source.get("store_name"), _text(source.get("store_id")))
            rows.append(
                {
                    "low_stock_product": low_product,
                    "low_stock_store": low_store,
                    "alternative_product": _text(source.get("product_name"), low_product),
                    "alternative_store": source_store,
                    "available_quantity": qty,
                    "alternative_type": "same_product_transfer",
                    "suggested_action": f"Consider transferring {low_product} from {source_store} to {low_store}.",
                    "reason": f"{source_store} has surplus {low_product} with {qty} units available.",
                }
            )

        if category:
            category_options = healthy_stock[
                (healthy_stock["category"].astype(str) == category)
                & (healthy_stock["product_id"].astype(str) != low_product_id)
            ].copy()
            category_options = category_options.sort_values(
                ["current_quantity", "product_name"],
                ascending=[False, True],
            ).head(3)
            for _, option in category_options.iterrows():
                qty = int(round(float(option.get("current_quantity", 0))))
                option_product = _text(option.get("product_name"), _text(option.get("product_id")))
                option_store = _text(option.get("store_name"), _text(option.get("store_id")))
                rows.append(
                    {
                        "low_stock_product": low_product,
                        "low_stock_store": low_store,
                        "alternative_product": option_product,
                        "alternative_store": option_store,
                        "available_quantity": qty,
                        "alternative_type": "category_alternative",
                        "suggested_action": f"Offer {option_product} from {option_store} as an alternative to {low_product}.",
                        "reason": f"{option_product} has strong stock in the same {category} category.",
                    }
                )

    alternatives = pd.DataFrame(rows)
    if alternatives.empty:
        return pd.DataFrame(columns=columns)

    alternatives["_type_rank"] = alternatives["alternative_type"].map(
        {"same_product_transfer": 0, "category_alternative": 1}
    ).fillna(2)
    alternatives = alternatives.sort_values(
        ["_type_rank", "available_quantity", "alternative_product"],
        ascending=[True, False, True],
    ).drop_duplicates(
        subset=["low_stock_product", "low_stock_store", "alternative_product", "alternative_store", "alternative_type"],
        keep="first",
    )
    return alternatives.drop(columns=["_type_rank"]).reset_index(drop=True)


def build_surplus_alternative_alert_text(
    surplus_df: pd.DataFrame,
    alternatives_df: pd.DataFrame,
) -> str:
    """Create a compact manager-facing summary for the Home page."""
    surplus_count = len(surplus_df)
    store_count = (
        surplus_df["store_id"].astype(str).nunique()
        if not surplus_df.empty and "store_id" in surplus_df.columns
        else 0
    )
    alternative_count = len(alternatives_df)

    if surplus_count == 0 and alternative_count == 0:
        return "No major surplus or alternative availability detected right now."

    parts = []
    if surplus_count:
        parts.append(
            f"{surplus_count} items have surplus stock across {store_count} stores."
        )
    if alternative_count:
        top = alternatives_df.iloc[0]
        parts.append(
            f"{top['low_stock_product']} is low in {top['low_stock_store']}, "
            f"but {int(top['available_quantity'])} units of {top['alternative_product']} are available at {top['alternative_store']}."
        )
    elif surplus_count:
        top = surplus_df.iloc[0]
        parts.append(
            f"{top['product_name']} has surplus stock and can be considered for transfer or promotion."
        )
    return " ".join(parts)
