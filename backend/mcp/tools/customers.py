"""Customer tools — branch-as-customer model.

IMPORTANT DATA NOTE: Oracle has no customer/end-user dimension (no customer_id in
sales or transactions, and no customer table). For Bunzl's B2B distribution model
the ordering unit is the branch/store, so "customer" here means the ordering
branch. Everything is computed live from Oracle sales + transactions. Each tool
states this assumption in its ``notes`` so the chatbot can be transparent.
"""

from __future__ import annotations

import pandas as pd

from backend.mcp import context as ctx


_CUSTOMER_NOTE = (
    "Oracle has no end-customer dimension; 'customer' = ordering branch/store "
    "(B2B model), computed from sales history."
)


def _branch_sales() -> pd.DataFrame:
    """Sales joined to branch names, with numeric quantity and revenue."""
    context = ctx.get_context()
    sales = context.raw("sales")
    stores = context.raw("stores")
    if sales.empty:
        return pd.DataFrame()
    sales["store_id"] = sales["store_id"].astype(str)
    sales["product_id"] = sales["product_id"].astype(str)
    sales["quantity_sold"] = ctx.num(sales, "quantity_sold")
    sales["revenue"] = sales["quantity_sold"] * ctx.num(sales, "selling_price")
    if "date" in sales.columns:
        sales["date"] = pd.to_datetime(sales["date"], errors="coerce")
    if not stores.empty:
        stores["store_id"] = stores["store_id"].astype(str)
        sales = sales.merge(
            stores[[c for c in ["store_id", "store_name", "city"] if c in stores.columns]],
            on="store_id", how="left",
        )
    return sales


def get_top_customers(metric: str = "revenue", limit: int = 10) -> dict:
    """Top ordering branches ('customers') by revenue (default) or units."""
    limit = ctx.clamp_limit(limit)
    metric = "units" if str(metric).lower().startswith("unit") else "revenue"
    sales = _branch_sales()
    if sales.empty:
        return {
            "tool": "get_top_customers",
            "summary": {"count": 0},
            "records": [],
            "sources": ctx.sources("sales", "stores"),
            "notes": _CUSTOMER_NOTE,
        }
    grouped = (
        sales.groupby([c for c in ["store_id", "store_name", "city"] if c in sales.columns], as_index=False)
        .agg(units_ordered=("quantity_sold", "sum"),
             revenue=("revenue", "sum"),
             distinct_products=("product_id", "nunique"),
             order_lines=("quantity_sold", "size"))
    )
    sort_col = "units_ordered" if metric == "units" else "revenue"
    grouped = grouped.sort_values(sort_col, ascending=False)
    grouped["units_ordered"] = grouped["units_ordered"].round().astype(int)
    grouped["revenue"] = grouped["revenue"].round(2)
    return {
        "tool": "get_top_customers",
        "summary": {"metric": metric, "branch_count": int(len(grouped))},
        "records": ctx.records(
            grouped,
            ["store_id", "store_name", "city", "units_ordered", "revenue", "distinct_products", "order_lines"],
            limit,
        ),
        "sources": ctx.sources("sales", "stores"),
        "notes": _CUSTOMER_NOTE,
    }


def get_customer_order_analysis(store_id: str = "", limit: int = 10) -> dict:
    """Ordering profile per branch: volume, order lines, average line size,
    product variety, and most recent activity date."""
    limit = ctx.clamp_limit(limit)
    sales = _branch_sales()
    if sales.empty:
        return {
            "tool": "get_customer_order_analysis",
            "summary": {"count": 0},
            "records": [],
            "sources": ctx.sources("sales", "stores"),
            "notes": _CUSTOMER_NOTE,
        }
    if store_id:
        sales = sales[sales["store_id"] == str(store_id)]
    group_cols = [c for c in ["store_id", "store_name", "city"] if c in sales.columns]
    agg = sales.groupby(group_cols, as_index=False).agg(
        order_lines=("quantity_sold", "size"),
        units_ordered=("quantity_sold", "sum"),
        revenue=("revenue", "sum"),
        avg_line_size=("quantity_sold", "mean"),
        distinct_products=("product_id", "nunique"),
        last_order_date=("date", "max") if "date" in sales.columns else ("quantity_sold", "size"),
    )
    agg["units_ordered"] = agg["units_ordered"].round().astype(int)
    agg["revenue"] = agg["revenue"].round(2)
    agg["avg_line_size"] = agg["avg_line_size"].round(2)
    if "last_order_date" in agg.columns:
        agg["last_order_date"] = agg["last_order_date"].astype(str)
    agg = agg.sort_values("units_ordered", ascending=False)
    return {
        "tool": "get_customer_order_analysis",
        "summary": {"scope": store_id or "all_branches", "branch_count": int(len(agg))},
        "records": ctx.records(
            agg,
            ["store_id", "store_name", "city", "order_lines", "units_ordered",
             "revenue", "avg_line_size", "distinct_products", "last_order_date"],
            limit,
        ),
        "sources": ctx.sources("sales", "stores"),
        "notes": _CUSTOMER_NOTE,
    }


def detect_abnormal_ordering(limit: int = 10, z_threshold: float = 3.0) -> dict:
    """Detect abnormally large order lines using a per-product z-score.

    For each product, computes the mean and standard deviation of order
    quantities across branches; an order line is abnormal when its quantity is at
    least ``z_threshold`` standard deviations above the product's mean.
    """
    limit = ctx.clamp_limit(limit)
    z_threshold = max(2.0, float(z_threshold or 3.0))
    sales = _branch_sales()
    if sales.empty or "product_id" not in sales.columns:
        return {
            "tool": "detect_abnormal_ordering",
            "summary": {"abnormal_count": 0},
            "records": [],
            "sources": ctx.sources("sales", "stores"),
            "notes": _CUSTOMER_NOTE,
        }

    stats = sales.groupby("product_id")["quantity_sold"].agg(["mean", "std"]).reset_index()
    stats = stats.rename(columns={"mean": "product_mean", "std": "product_std"})
    flagged = sales.merge(stats, on="product_id", how="left")
    flagged["product_std"] = flagged["product_std"].fillna(0)
    # z-score; products with zero variance cannot produce an outlier.
    flagged["z_score"] = 0.0
    mask = flagged["product_std"] > 0
    flagged.loc[mask, "z_score"] = (
        (flagged.loc[mask, "quantity_sold"] - flagged.loc[mask, "product_mean"])
        / flagged.loc[mask, "product_std"]
    )
    abnormal = flagged[flagged["z_score"] >= z_threshold].copy()
    abnormal["z_score"] = abnormal["z_score"].round(2)
    abnormal["product_mean"] = abnormal["product_mean"].round(2)
    abnormal["expected_max"] = (abnormal["product_mean"] + z_threshold * abnormal["product_std"]).round(1)
    abnormal["quantity_sold"] = abnormal["quantity_sold"].round().astype(int)
    abnormal = abnormal.sort_values("z_score", ascending=False)

    # Attach product names if available.
    products = ctx.get_context().raw("products")
    if not products.empty and "product_name" in products.columns:
        products["product_id"] = products["product_id"].astype(str)
        abnormal = abnormal.merge(products[["product_id", "product_name"]], on="product_id", how="left")

    return {
        "tool": "detect_abnormal_ordering",
        "summary": {
            "abnormal_count": int(len(abnormal)),
            "z_threshold": z_threshold,
        },
        "records": ctx.records(
            abnormal,
            ["sale_id", "date", "store_id", "store_name", "product_id", "product_name",
             "quantity_sold", "product_mean", "expected_max", "z_score"],
            limit,
        ),
        "sources": ctx.sources("sales", "stores"),
        "notes": _CUSTOMER_NOTE + " Abnormal = quantity ≥ mean + z·std per product.",
    }
