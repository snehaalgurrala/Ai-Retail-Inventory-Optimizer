"""Inventory tools: health, low stock, and overstock.

Reuses backend.services.store_inventory_service via the shared store-inventory
view (built once, live from Oracle).
"""

from __future__ import annotations

from backend.mcp import context as ctx
from backend.services.store_inventory_service import (
    build_store_comparison,
    build_store_kpis,
    filter_inventory_by_store,
    get_overstock_items as _svc_overstock_items,
    get_understock_items as _svc_understock_items,
)


_INVENTORY_COLUMNS = [
    "product_id", "product_name", "category", "store_id", "store_name", "city",
    "current_quantity", "reorder_threshold", "stock_status",
    "shortage_quantity", "surplus_quantity", "selling_price", "supplier_name",
]


def get_inventory_health(store_id: str = "") -> dict:
    """Overall inventory health KPIs and stock-status mix, optionally per store.

    Returns total quantity, product count, low-stock / overstock / slow-dead
    counts, inventory value, and a per-store comparison.
    """
    view = ctx.get_context().store_inventory_view()
    if view.empty:
        return {
            "tool": "get_inventory_health",
            "summary": {"row_count": 0},
            "records": [],
            "sources": ctx.sources("inventory", "products", "stores", "sales"),
            "notes": "No inventory rows are available from Oracle.",
        }

    scoped = filter_inventory_by_store(view, store_id) if store_id else view
    kpis = build_store_kpis(scoped)
    status_mix = (
        scoped["stock_status"].value_counts().to_dict()
        if "stock_status" in scoped.columns
        else {}
    )
    comparison = build_store_comparison(view) if not store_id else build_store_comparison(scoped)
    return {
        "tool": "get_inventory_health",
        "summary": {
            "scope": store_id or "all_stores",
            **kpis,
            "status_mix": {str(k): int(v) for k, v in status_mix.items()},
        },
        "records": ctx.records(
            comparison,
            ["store_id", "store_name", "city", "inventory_quantity", "product_count",
             "low_stock_count", "overstock_count", "inventory_value"],
            ctx.clamp_limit(0),
        ),
        "sources": ctx.sources("inventory", "products", "stores", "sales"),
    }


def get_low_stock_items(store_id: str = "", limit: int = 10) -> dict:
    """Items at or below reorder threshold (replenishment candidates)."""
    limit = ctx.clamp_limit(limit)
    view = ctx.get_context().store_inventory_view()
    scoped = filter_inventory_by_store(view, store_id) if store_id else view
    low = _svc_understock_items(scoped)
    return {
        "tool": "get_low_stock_items",
        "summary": {
            "scope": store_id or "all_stores",
            "low_stock_count": int(len(low)),
        },
        "records": ctx.records(low, _INVENTORY_COLUMNS + ["urgency_label", "depletion_window", "ai_recommendation"], limit),
        "sources": ctx.sources("inventory", "products", "stores", "sales"),
    }


def get_overstock_items(store_id: str = "", limit: int = 10) -> dict:
    """Items with stock well above reorder needs (transfer / clearance candidates)."""
    limit = ctx.clamp_limit(limit)
    view = ctx.get_context().store_inventory_view()
    scoped = filter_inventory_by_store(view, store_id) if store_id else view
    over = _svc_overstock_items(scoped)
    return {
        "tool": "get_overstock_items",
        "summary": {
            "scope": store_id or "all_stores",
            "overstock_count": int(len(over)),
        },
        "records": ctx.records(over, _INVENTORY_COLUMNS + ["recent_daily_sales_velocity", "ai_recommendation"], limit),
        "sources": ctx.sources("inventory", "products", "stores", "sales"),
    }
