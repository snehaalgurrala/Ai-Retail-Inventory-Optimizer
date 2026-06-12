"""Product tools: top products, bottom products, and product performance.

Rankings are computed live from the Oracle sales history joined to products.
"""

from __future__ import annotations

import pandas as pd

from backend.mcp import context as ctx


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
