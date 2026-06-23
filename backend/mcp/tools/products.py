"""Product tools: product master, top/bottom products, and product performance.

Rankings are computed live from the Oracle sales history joined to products.
The product-master tools join the product catalogue (BZ_MOCK_PRODUCT) to the
network-wide inventory view (BZ_MOCK_INVENTORY) so the chatbot can answer product
master, reorder-point, and stock-planning questions at product grain.
"""

from __future__ import annotations

import pandas as pd

from backend.mcp import config, context as ctx
from backend.services import inventory_scope


def _sales_with_products() -> pd.DataFrame:
    """Sales joined to product/store names, with numeric quantity and revenue."""
    context = ctx.get_context()
    sales = context.raw("sales")
    products = context.raw("products")
    stores = context.raw("stores")
    if sales.empty:
        return pd.DataFrame()

    sales["product_id"] = sales["product_id"].astype(str)
    sales["store_id"] = sales["store_id"].astype(str)
    sales["quantity_sold"] = ctx.num(sales, "quantity_sold")
    sales["selling_price"] = ctx.num(sales, "selling_price")
    sales["revenue"] = sales["quantity_sold"] * sales["selling_price"]

    if not products.empty:
        products["product_id"] = products["product_id"].astype(str)
        sales = sales.merge(
            products[[c for c in ["product_id", "product_name", "category"] if c in products.columns]],
            on="product_id", how="left",
        )
    if not stores.empty:
        stores["store_id"] = stores["store_id"].astype(str)
        sales = sales.merge(
            stores[[c for c in ["store_id", "store_name", "city"] if c in stores.columns]],
            on="store_id", how="left",
        )
    return sales


def _rank_products(direction: str, store_id: str, category: str, metric: str, limit: int) -> pd.DataFrame:
    sales = _sales_with_products()
    if sales.empty:
        return pd.DataFrame()
    if store_id:
        sales = sales[sales["store_id"] == str(store_id)]
    if category and "category" in sales.columns:
        sales = sales[sales["category"].astype(str).str.strip().str.lower() == category.strip().lower()]
    if sales.empty:
        return pd.DataFrame()

    grouped = (
        sales.groupby(["product_id", "product_name"], as_index=False)
        .agg(units_sold=("quantity_sold", "sum"), revenue=("revenue", "sum"))
    )
    sort_col = "revenue" if metric == "revenue" else "units_sold"
    ascending = direction == "ascending"
    grouped = grouped.sort_values([sort_col, "product_name"], ascending=[ascending, True])
    grouped["units_sold"] = grouped["units_sold"].round().astype(int)
    grouped["revenue"] = grouped["revenue"].round(2)
    return grouped.head(limit).reset_index(drop=True)


def get_top_products(store_id: str = "", category: str = "", metric: str = "units", limit: int = 10) -> dict:
    """Best-selling products. metric is 'units' (default) or 'revenue'."""
    limit = ctx.clamp_limit(limit)
    metric = "revenue" if str(metric).lower().startswith("rev") else "units"
    ranked = _rank_products("descending", store_id, category, metric, limit)
    return {
        "tool": "get_top_products",
        "summary": {
            "scope": store_id or "all_stores",
            "category": category or "all",
            "metric": metric,
            "count": int(len(ranked)),
        },
        "records": ctx.records(ranked, ["product_id", "product_name", "units_sold", "revenue"], limit),
        "sources": ctx.sources("sales", "products", "stores"),
    }


def get_bottom_products(store_id: str = "", category: str = "", metric: str = "units", limit: int = 10) -> dict:
    """Worst-selling products (weakest demand). metric is 'units' or 'revenue'."""
    limit = ctx.clamp_limit(limit)
    metric = "revenue" if str(metric).lower().startswith("rev") else "units"
    ranked = _rank_products("ascending", store_id, category, metric, limit)
    return {
        "tool": "get_bottom_products",
        "summary": {
            "scope": store_id or "all_stores",
            "category": category or "all",
            "metric": metric,
            "count": int(len(ranked)),
        },
        "records": ctx.records(ranked, ["product_id", "product_name", "units_sold", "revenue"], limit),
        "sources": ctx.sources("sales", "products", "stores"),
    }


# ---------------------------------------------------------------------------
# Product master (catalogue joined to network-wide inventory & reorder point)
# ---------------------------------------------------------------------------
_MASTER_COLUMNS = [
    "product_id", "product_name", "category", "unit_price",
    "current_inventory", "reorder_point", "units_to_reorder",
    "below_reorder", "inventory_status", "supplier_name", "lead_time_days",
]

_MASTER_NOTE = (
    "Current inventory and reorder point are network-wide totals summed across all "
    "branches (BZ_MOCK_INVENTORY). Available inventory equals on-hand here; lead time "
    "is the supplier's average delivery days. Safety stock is not tracked in Oracle."
)


def _list_limit(limit: int) -> int:
    """Effective row limit for whole-catalogue product-master listings.

    A product-master listing covers the complete (small) product dimension, so an
    unspecified limit must mean "return them all" — not the paginated default of
    10. An explicit positive limit is honoured (and capped by the hard max).
    """
    if limit and int(limit) > 0:
        return ctx.clamp_limit(limit)
    return config.max_record_limit()


def _product_master_frame() -> pd.DataFrame:
    """Product catalogue joined to network-wide stock, reorder point, and lead time.

    Stock and reorder point are resolved through ``inventory_scope`` so the figures
    are identical to every other surface (pages, email, simulator, recommendations).
    """
    context = ctx.get_context()
    products = context.raw("products")
    inventory = context.raw("inventory")
    suppliers = context.raw("suppliers")
    if products.empty or "product_id" not in products.columns:
        return pd.DataFrame()

    frame = products.copy()
    frame["product_id"] = frame["product_id"].astype(str)

    # Network-wide on-hand and reorder point per product (single source of truth).
    stock = inventory_scope.stock_by_product(inventory)
    reorder = inventory_scope.reorder_by_product(inventory)
    frame["current_inventory"] = frame["product_id"].map(stock).fillna(0).astype(int)
    frame["reorder_point"] = frame["product_id"].map(reorder).fillna(0).astype(int)

    frame["unit_price"] = ctx.num(frame, "selling_price").round(2)
    frame["units_to_reorder"] = (
        (frame["reorder_point"] - frame["current_inventory"]).clip(lower=0).astype(int)
    )
    frame["below_reorder"] = (frame["reorder_point"] > 0) & (
        frame["current_inventory"] <= frame["reorder_point"]
    )
    frame["inventory_status"] = "Healthy"
    frame.loc[frame["below_reorder"], "inventory_status"] = "Below Reorder Point"
    frame.loc[frame["reorder_point"] <= 0, "inventory_status"] = "No Reorder Point Set"

    # Supplier name + lead time via the product->supplier link.
    if not suppliers.empty and {"supplier_id"}.issubset(suppliers.columns) and "supplier_id" in frame.columns:
        sup = suppliers.copy()
        sup["supplier_id"] = sup["supplier_id"].astype(str)
        keep = [c for c in ["supplier_id", "supplier_name", "avg_delivery_days"] if c in sup.columns]
        frame["supplier_id"] = frame["supplier_id"].astype(str)
        frame = frame.merge(sup[keep], on="supplier_id", how="left")
    if "avg_delivery_days" in frame.columns:
        frame = frame.rename(columns={"avg_delivery_days": "lead_time_days"})
        frame["lead_time_days"] = ctx.num(frame, "lead_time_days").round().astype(int)
    return frame


def _filter_master(frame: pd.DataFrame, product_id: str, product_name: str, category: str) -> pd.DataFrame:
    if product_id:
        frame = frame[frame["product_id"] == str(product_id)]
    if product_name and "product_name" in frame.columns:
        frame = frame[
            frame["product_name"].astype(str).str.contains(product_name.strip(), case=False, na=False)
        ]
    if category and "category" in frame.columns:
        frame = frame[
            frame["category"].astype(str).str.strip().str.lower() == category.strip().lower()
        ]
    return frame


def get_product_master(
    product_id: str = "", product_name: str = "", category: str = "", limit: int = 0
) -> dict:
    """Product master table with reorder point, current inventory, available inventory, category, unit price, supplier and lead time for every product.
    Use this for any product list/catalogue question or named-product lookup: 'show all
    products', 'product list with reorder point', 'product list with current inventory',
    'reorder point of Product X', 'is Product X below its reorder point', 'which supplier
    has Product X'. Filter by product_id, product_name, or category; omit all to list every product."""
    frame = _product_master_frame()
    if frame.empty:
        return {
            "tool": "get_product_master",
            "summary": {"product_count": 0},
            "records": [],
            "sources": ctx.sources("products", "inventory", "suppliers"),
            "notes": "No product records are available from Oracle.",
        }
    frame = _filter_master(frame, product_id, product_name, category)
    if "product_name" in frame.columns:
        frame = frame.sort_values("product_name")
    return {
        "tool": "get_product_master",
        "summary": {
            "scope": product_id or product_name or category or "all_products",
            "product_count": int(len(frame)),
            "below_reorder_count": int(frame["below_reorder"].sum()) if "below_reorder" in frame.columns else 0,
        },
        "records": ctx.records(frame, _MASTER_COLUMNS, _list_limit(limit)),
        "sources": ctx.sources("products", "inventory", "suppliers"),
        "notes": _MASTER_NOTE,
    }


def get_products_below_reorder(category: str = "", limit: int = 0) -> dict:
    """Products at or below their reorder point network-wide, with reorder point, current inventory and shortfall (replenishment / reorder planning).
    Use this for 'which products are below reorder point', 'products at stockout risk by
    reorder point', 'what needs reordering'. Sorted by largest shortfall first. Optional category filter."""
    frame = _product_master_frame()
    if frame.empty:
        return {
            "tool": "get_products_below_reorder",
            "summary": {"below_reorder_count": 0},
            "records": [],
            "sources": ctx.sources("products", "inventory", "suppliers"),
            "notes": "No product records are available from Oracle.",
        }
    frame = _filter_master(frame, "", "", category)
    frame = frame[frame["below_reorder"]].sort_values("units_to_reorder", ascending=False)
    return {
        "tool": "get_products_below_reorder",
        "summary": {
            "scope": category or "all_products",
            "below_reorder_count": int(len(frame)),
        },
        "records": ctx.records(frame, _MASTER_COLUMNS, _list_limit(limit)),
        "sources": ctx.sources("products", "inventory", "suppliers"),
        "notes": _MASTER_NOTE,
    }


def get_product_performance(product_id: str = "", category: str = "", limit: int = 10) -> dict:
    """Per-product sales performance (units, revenue, store reach, avg price)."""
    limit = ctx.clamp_limit(limit)
    sales = _sales_with_products()
    if sales.empty:
        return {
            "tool": "get_product_performance",
            "summary": {"count": 0},
            "records": [],
            "sources": ctx.sources("sales", "products"),
            "notes": "No sales records available from Oracle.",
        }
    if product_id:
        sales = sales[sales["product_id"] == str(product_id)]
    if category and "category" in sales.columns:
        sales = sales[sales["category"].astype(str).str.strip().str.lower() == category.strip().lower()]

    grouped = (
        sales.groupby(["product_id", "product_name"], as_index=False)
        .agg(
            units_sold=("quantity_sold", "sum"),
            revenue=("revenue", "sum"),
            store_reach=("store_id", "nunique"),
            avg_selling_price=("selling_price", "mean"),
        )
        .sort_values("revenue", ascending=False)
    )
    grouped["units_sold"] = grouped["units_sold"].round().astype(int)
    grouped["revenue"] = grouped["revenue"].round(2)
    grouped["avg_selling_price"] = grouped["avg_selling_price"].round(2)
    return {
        "tool": "get_product_performance",
        "summary": {"count": int(len(grouped)), "scope": product_id or category or "all"},
        "records": ctx.records(grouped, ["product_id", "product_name", "units_sold", "revenue", "store_reach", "avg_selling_price"], limit),
        "sources": ctx.sources("sales", "products"),
    }
