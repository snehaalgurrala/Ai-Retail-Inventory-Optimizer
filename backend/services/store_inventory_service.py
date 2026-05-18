from pathlib import Path
from typing import Any

import pandas as pd

from backend.services.llm_reasoner import humanize_analytics_payload, llm_is_configured


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


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


def _text(value: Any, fallback: str = "") -> str:
    if pd.isna(value):
        return fallback
    text = str(value or "").strip()
    return text or fallback


def load_store_inventory_inputs() -> dict[str, pd.DataFrame]:
    """Load raw inventory context and processed recommendation context."""
    return {
        "inventory": _safe_read_csv(RAW_DATA_DIR / "inventory.csv"),
        "products": _safe_read_csv(RAW_DATA_DIR / "products.csv"),
        "stores": _safe_read_csv(RAW_DATA_DIR / "stores.csv"),
        "sales": _safe_read_csv(RAW_DATA_DIR / "sales.csv"),
        "suppliers": _safe_read_csv(RAW_DATA_DIR / "suppliers.csv"),
        "recommendations": _safe_read_csv(PROCESSED_DATA_DIR / "recommendations.csv"),
        "agent_outputs": _safe_read_csv(PROCESSED_DATA_DIR / "agent_outputs.csv"),
    }


def build_store_inventory_view(
    inventory: pd.DataFrame,
    products: pd.DataFrame,
    stores: pd.DataFrame,
    suppliers: pd.DataFrame | None = None,
    sales: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a product-store inventory view with status, value, and sales velocity."""
    if inventory.empty or not {"product_id", "store_id", "stock_level"}.issubset(inventory.columns):
        return pd.DataFrame()

    view = inventory.copy()
    view["product_id"] = view["product_id"].astype(str)
    view["store_id"] = view["store_id"].astype(str)
    view["current_quantity"] = _number_column(view, "stock_level")
    view["inventory_reorder_threshold"] = _number_column(view, "reorder_threshold")

    if not products.empty and "product_id" in products.columns:
        product_columns = [
            column
            for column in [
                "product_id",
                "product_name",
                "category",
                "cost_price",
                "selling_price",
                "reorder_threshold",
                "supplier_id",
            ]
            if column in products.columns
        ]
        product_view = products[product_columns].copy()
        product_view["product_id"] = product_view["product_id"].astype(str)
        if "reorder_threshold" in product_view.columns:
            product_view = product_view.rename(columns={"reorder_threshold": "product_reorder_threshold"})
        view = view.merge(product_view, on="product_id", how="left")

    if suppliers is not None and not suppliers.empty and "supplier_id" in view.columns and "supplier_id" in suppliers.columns:
        supplier_columns = [
            column for column in ["supplier_id", "supplier_name"] if column in suppliers.columns
        ]
        supplier_view = suppliers[supplier_columns].copy()
        supplier_view["supplier_id"] = supplier_view["supplier_id"].astype(str)
        view["supplier_id"] = view["supplier_id"].astype(str)
        view = view.merge(supplier_view, on="supplier_id", how="left")

    if not stores.empty and "store_id" in stores.columns:
        store_columns = [
            column for column in ["store_id", "store_name", "city", "capacity"] if column in stores.columns
        ]
        store_view = stores[store_columns].copy()
        store_view["store_id"] = store_view["store_id"].astype(str)
        view = view.merge(store_view, on="store_id", how="left")

    view["reorder_threshold"] = view["inventory_reorder_threshold"]
    if "product_reorder_threshold" in view.columns:
        view["reorder_threshold"] = view["reorder_threshold"].where(
            view["reorder_threshold"] > 0,
            _number_column(view, "product_reorder_threshold"),
        )

    if sales is not None and not sales.empty and {"date", "product_id", "store_id", "quantity_sold"}.issubset(sales.columns):
        sales_view = sales.copy()
        sales_view["date"] = pd.to_datetime(sales_view["date"], errors="coerce")
        sales_view["product_id"] = sales_view["product_id"].astype(str)
        sales_view["store_id"] = sales_view["store_id"].astype(str)
        sales_view["quantity_sold"] = _number_column(sales_view, "quantity_sold")
        sales_view = sales_view[sales_view["date"].notna()].copy()
        if not sales_view.empty:
            latest_date = sales_view["date"].max()
            recent = sales_view[sales_view["date"] >= latest_date - pd.Timedelta(days=30)].copy()
            velocity = (
                recent.groupby(["product_id", "store_id"], as_index=False)["quantity_sold"]
                .sum()
                .rename(columns={"quantity_sold": "recent_30_day_quantity_sold"})
            )
            velocity["recent_daily_sales_velocity"] = velocity["recent_30_day_quantity_sold"] / 30
            view = view.merge(velocity, on=["product_id", "store_id"], how="left")

    for column in ["recent_30_day_quantity_sold", "recent_daily_sales_velocity"]:
        view[column] = _number_column(view, column)

    view["selling_price"] = _number_column(view, "selling_price")
    view["cost_price"] = _number_column(view, "cost_price")
    view["estimated_inventory_value"] = view["current_quantity"] * view["selling_price"]
    view["cost_inventory_value"] = view["current_quantity"] * view["cost_price"]
    view["shortage_quantity"] = (view["reorder_threshold"] - view["current_quantity"]).clip(lower=0)
    view["suggested_reorder_quantity"] = ((view["reorder_threshold"] * 2) - view["current_quantity"]).clip(lower=0).round().astype(int)
    view["surplus_quantity"] = (view["current_quantity"] - view["reorder_threshold"]).clip(lower=0)

    view["stock_status"] = "Healthy"
    view.loc[view["current_quantity"] <= view["reorder_threshold"], "stock_status"] = "Low Stock"
    view.loc[
        (view["reorder_threshold"] > 0)
        & (view["current_quantity"] >= view["reorder_threshold"] * 2),
        "stock_status",
    ] = "Overstock"
    view.loc[
        (view["stock_status"].eq("Healthy"))
        & (view["recent_daily_sales_velocity"] <= 0.2)
        & (view["current_quantity"] >= view["reorder_threshold"] * 1.5)
        & (view["reorder_threshold"] > 0),
        "stock_status",
    ] = "Overstock"

    view["priority"] = view.apply(
        lambda row: "High"
        if row["current_quantity"] <= max(row["reorder_threshold"] * 0.5, 0)
        else "Medium"
        if row["stock_status"] == "Low Stock"
        else "Medium"
        if row["stock_status"] == "Overstock"
        else "Low",
        axis=1,
    )
    view["ai_recommendation"] = view.apply(_row_recommendation, axis=1)

    defaults = {
        "product_name": view["product_id"],
        "category": "",
        "store_name": view["store_id"],
        "city": "",
        "capacity": 0,
        "supplier_name": "",
    }
    for column, fallback in defaults.items():
        if column not in view.columns:
            view[column] = fallback
        view[column] = view[column].fillna(fallback if isinstance(fallback, str) else 0)
    return view


def _row_recommendation(row: pd.Series) -> str:
    product = _text(row.get("product_name"), _text(row.get("product_id"), "This product"))
    status = _text(row.get("stock_status"))
    if status == "Low Stock":
        shortage = int(round(float(row.get("shortage_quantity", 0))))
        reorder = int(round(float(row.get("suggested_reorder_quantity", 0))))
        return f"Replenish {product}; shortage is {shortage} units and suggested reorder is {reorder} units."
    if status == "Overstock":
        surplus = int(round(float(row.get("surplus_quantity", 0))))
        if float(row.get("recent_daily_sales_velocity", 0)) <= 0.2:
            return f"Review {product} for discount, clearance, or transfer; surplus is {surplus} units with slow movement."
        return f"Consider transfer or promotion for {product}; surplus is {surplus} units."
    return f"{product} is within a healthy stock range."


def filter_inventory_by_store(inventory_view: pd.DataFrame, store_id: str) -> pd.DataFrame:
    if inventory_view.empty or not store_id or store_id == "All Stores":
        return inventory_view.copy()
    return inventory_view[inventory_view["store_id"].astype(str) == str(store_id)].copy()


def build_store_kpis(inventory_view: pd.DataFrame) -> dict[str, Any]:
    if inventory_view.empty:
        return {
            "total_quantity": 0,
            "product_count": 0,
            "low_stock_count": 0,
            "overstock_count": 0,
            "slow_dead_count": 0,
            "inventory_value": 0.0,
        }
    slow_dead = inventory_view[
        (inventory_view["current_quantity"] > 0)
        & (inventory_view["recent_30_day_quantity_sold"] <= 0)
    ]
    return {
        "total_quantity": int(round(float(inventory_view["current_quantity"].sum()))),
        "product_count": int(inventory_view["product_id"].astype(str).nunique()),
        "low_stock_count": int(inventory_view["stock_status"].eq("Low Stock").sum()),
        "overstock_count": int(inventory_view["stock_status"].eq("Overstock").sum()),
        "slow_dead_count": int(len(slow_dead)),
        "inventory_value": float(inventory_view["estimated_inventory_value"].sum()),
    }


def get_understock_items(inventory_view: pd.DataFrame) -> pd.DataFrame:
    if inventory_view.empty:
        return pd.DataFrame()
    return inventory_view[inventory_view["stock_status"].eq("Low Stock")].copy().sort_values(
        ["priority", "shortage_quantity"],
        ascending=[True, False],
    )


def get_overstock_items(inventory_view: pd.DataFrame) -> pd.DataFrame:
    if inventory_view.empty:
        return pd.DataFrame()
    return inventory_view[inventory_view["stock_status"].eq("Overstock")].copy().sort_values(
        ["surplus_quantity", "current_quantity"],
        ascending=[False, False],
    )


def filter_recommendations_for_store(recommendations: pd.DataFrame, store_id: str) -> pd.DataFrame:
    if recommendations.empty:
        return pd.DataFrame()
    recs = recommendations.copy()
    for column in ["store_id", "alternative_store_id"]:
        if column not in recs.columns:
            recs[column] = ""
    if store_id and store_id != "All Stores":
        recs = recs[
            recs["store_id"].astype(str).eq(str(store_id))
            | recs["alternative_store_id"].astype(str).eq(str(store_id))
        ].copy()
    if "priority" in recs.columns:
        recs["_priority_rank"] = recs["priority"].fillna("").astype(str).str.lower().map(
            {"high": 0, "medium": 1, "low": 2}
        ).fillna(3)
        recs = recs.sort_values(["_priority_rank", "recommendation_id"])
        recs = recs.drop(columns=["_priority_rank"])
    return recs


def build_store_comparison(inventory_view: pd.DataFrame) -> pd.DataFrame:
    if inventory_view.empty:
        return pd.DataFrame()
    comparison = (
        inventory_view.groupby(["store_id", "store_name", "city"], as_index=False)
        .agg(
            inventory_quantity=("current_quantity", "sum"),
            product_count=("product_id", "nunique"),
            low_stock_count=("stock_status", lambda values: int((values == "Low Stock").sum())),
            overstock_count=("stock_status", lambda values: int((values == "Overstock").sum())),
            inventory_value=("estimated_inventory_value", "sum"),
        )
    )
    return comparison.sort_values("inventory_quantity", ascending=False).reset_index(drop=True)


def build_category_summary(inventory_view: pd.DataFrame, by_store: bool = False) -> pd.DataFrame:
    if inventory_view.empty:
        return pd.DataFrame()
    group_columns = ["category"]
    if by_store:
        group_columns = ["store_name", "category"]
    return (
        inventory_view.groupby(group_columns, as_index=False)
        .agg(
            inventory_quantity=("current_quantity", "sum"),
            inventory_value=("estimated_inventory_value", "sum"),
            product_count=("product_id", "nunique"),
        )
        .sort_values("inventory_quantity", ascending=False)
    )


def build_store_inventory_summary(
    inventory_view: pd.DataFrame,
    selected_store_label: str,
    recommendations: pd.DataFrame | None = None,
) -> str:
    if inventory_view.empty:
        return "No inventory rows are available for this store selection."

    kpis = build_store_kpis(inventory_view)
    low_stock = get_understock_items(inventory_view)
    overstock = get_overstock_items(inventory_view)
    subject = "All stores have" if str(selected_store_label).strip().lower() == "all stores" else f"{selected_store_label} has"
    leading_parts = [
        f"{subject} {kpis['low_stock_count']} low-stock items",
        f"{kpis['overstock_count']} overstocked products",
        f"and {kpis['product_count']} active products.",
    ]
    detail_parts = []
    if not low_stock.empty:
        row = low_stock.iloc[0]
        detail_parts.append(f"{row['product_name']} needs replenishment.")
    if not overstock.empty:
        row = overstock.iloc[0]
        detail_parts.append(f"{row['product_name']} may be considered for transfer, discount, or clearance.")
    if recommendations is not None and not recommendations.empty:
        detail_parts.append(f"{len(recommendations)} active recommendations are linked to this scope.")

    fallback = " ".join(leading_parts + detail_parts)
    if not llm_is_configured():
        return fallback

    payload = {
        "answer": fallback,
        "explanation": "",
        "suggestions": [],
        "follow_up_question": "",
        "confidence": "high",
        "supporting_points": [
            f"low_stock_count={kpis['low_stock_count']}",
            f"overstock_count={kpis['overstock_count']}",
            f"inventory_value={round(kpis['inventory_value'], 2)}",
        ],
    }
    try:
        humanized = humanize_analytics_payload(
            question=f"Summarize inventory for {selected_store_label}",
            payload=payload,
            supporting_points=payload["supporting_points"],
        )
        return str(humanized.get("answer") or fallback)
    except Exception:
        return fallback
