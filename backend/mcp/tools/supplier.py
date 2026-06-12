"""Supplier and procurement tools.

- get_supplier_analysis: supplier reliability, lead time, and the demand they
  carry (products + units + revenue), computed live from Oracle.
- get_procurement_risk: products at stockout risk weighted by how slow/unreliable
  their supplier is, i.e. where a reorder is both needed and hard to fulfil.
"""

from __future__ import annotations

import pandas as pd

from backend.mcp import context as ctx


def _supplier_demand_frame() -> pd.DataFrame:
    """Supplier-level rollup: products carried, units/revenue sold, reliability."""
    context = ctx.get_context()
    suppliers = context.raw("suppliers")
    products = context.raw("products")
    sales = context.raw("sales")
    if suppliers.empty:
        return pd.DataFrame()

    suppliers["supplier_id"] = suppliers["supplier_id"].astype(str)
    rollup = suppliers.copy()
    rollup["reliability_score"] = ctx.num(rollup, "reliability_score")
    rollup["avg_delivery_days"] = ctx.num(rollup, "avg_delivery_days")

    if not products.empty and "supplier_id" in products.columns:
        products["supplier_id"] = products["supplier_id"].astype(str)
        products["product_id"] = products["product_id"].astype(str)
        product_counts = products.groupby("supplier_id", as_index=False).agg(
            product_count=("product_id", "nunique")
        )
        rollup = rollup.merge(product_counts, on="supplier_id", how="left")

        if not sales.empty:
            sales["product_id"] = sales["product_id"].astype(str)
            sales["quantity_sold"] = ctx.num(sales, "quantity_sold")
            sales["revenue"] = sales["quantity_sold"] * ctx.num(sales, "selling_price")
            sales_by_supplier = sales.merge(
                products[["product_id", "supplier_id"]], on="product_id", how="left"
            )
            demand = sales_by_supplier.groupby("supplier_id", as_index=False).agg(
                units_sold=("quantity_sold", "sum"), revenue=("revenue", "sum")
            )
            rollup = rollup.merge(demand, on="supplier_id", how="left")

    for column in ["product_count", "units_sold", "revenue"]:
        if column not in rollup.columns:
            rollup[column] = 0
        rollup[column] = ctx.num(rollup, column)
    rollup["units_sold"] = rollup["units_sold"].round().astype(int)
    rollup["revenue"] = rollup["revenue"].round(2)
    rollup["reliability_pct"] = (rollup["reliability_score"] * 100).round(1)
    return rollup


def get_supplier_analysis(limit: int = 10) -> dict:
    """Supplier reliability, delivery lead time, and the demand they carry."""
    limit = ctx.clamp_limit(limit)
    rollup = _supplier_demand_frame()
    if rollup.empty:
        return {
            "tool": "get_supplier_analysis",
            "summary": {"supplier_count": 0},
            "records": [],
            "sources": ctx.sources("suppliers", "products", "sales"),
            "notes": "No supplier rows available from Oracle.",
        }
    ranked = rollup.sort_values(["reliability_score", "revenue"], ascending=[True, False])
    low_reliability = int((rollup["reliability_score"] < 0.90).sum())
    return {
        "tool": "get_supplier_analysis",
        "summary": {
            "supplier_count": int(len(rollup)),
            "low_reliability_count": low_reliability,
            "avg_reliability_pct": round(float(rollup["reliability_pct"].mean()), 1),
        },
        "records": ctx.records(
            ranked,
            ["supplier_id", "supplier_name", "reliability_pct", "avg_delivery_days",
             "product_count", "units_sold", "revenue"],
            limit,
        ),
        "sources": ctx.sources("suppliers", "products", "sales"),
        "notes": "Suppliers are listed least-reliable first to surface risk.",
    }


def get_procurement_risk(limit: int = 10) -> dict:
    """Reorder-critical products weighted by supplier reliability and lead time.

    procurement_risk = stockout risk_score, increased when the product's supplier
    is unreliable (low reliability) or slow (long delivery lead time). Highest
    means 'needs reordering soon AND hard to fulfil'.
    """
    limit = ctx.clamp_limit(limit)
    view = ctx.get_context().predictive_view()
    if view.empty:
        return {
            "tool": "get_procurement_risk",
            "summary": {"count": 0},
            "records": [],
            "sources": ctx.sources("inventory", "products", "stores", "sales", "suppliers"),
            "notes": "No predictive inventory rows available from Oracle.",
        }

    view = view.copy()
    reliability = ctx.num(view, "reliability_score").clip(lower=0, upper=1)
    lead_days = ctx.num(view, "avg_delivery_days")
    lead_factor = (lead_days / 14.0).clip(upper=1)  # normalise: 14+ days = max pressure
    supplier_pressure = ((1 - reliability) * 25) + (lead_factor * 15)
    view["procurement_risk_score"] = (
        ctx.num(view, "risk_score") + supplier_pressure
    ).clip(upper=100).round().astype(int)

    # Focus on items that actually need reordering.
    needs_reorder = view[
        (view.get("predictive_alert", False) == True)  # noqa: E712
        | (ctx.num(view, "suggested_reorder_qty") > 0)
    ].copy()
    if needs_reorder.empty:
        needs_reorder = view.copy()
    ranked = needs_reorder.sort_values("procurement_risk_score", ascending=False)
    return {
        "tool": "get_procurement_risk",
        "summary": {
            "flagged_count": int(len(needs_reorder)),
            "high_risk_count": int((ranked["procurement_risk_score"] >= 70).sum()),
        },
        "records": ctx.records(
            ranked,
            ["product_id", "product_name", "store_id", "store_name",
             "current_quantity", "predicted_days_remaining", "suggested_reorder_qty",
             "reliability_score", "avg_delivery_days", "risk_score",
             "procurement_risk_score", "alert_reason"],
            limit,
        ),
        "sources": ctx.sources("inventory", "products", "stores", "sales", "suppliers"),
    }
