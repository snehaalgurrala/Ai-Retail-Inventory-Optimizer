"""Reference tools: stores, products, suppliers, and location validation.

All read raw Oracle tables directly through the context bundle.
"""

from __future__ import annotations

from backend.mcp import config, context as ctx
from backend.services.location_validation import validate_requested_location


def _list_limit(limit: int) -> int:
    """Effective row limit for whole-dimension list tools.

    These list a complete small reference dimension (products, stores, suppliers),
    so an unspecified limit must mean "return them all" — not the paginated default
    of 10, which would make "give me all products" report 16 but show only 10. An
    explicit positive limit is still honoured (and capped by the hard max).
    """
    if limit and int(limit) > 0:
        return ctx.clamp_limit(limit)
    return config.max_record_limit()


def list_stores(limit: int = 0) -> dict:
    """List the branches/stores (the B2B ordering units), with city and capacity."""
    limit = _list_limit(limit)
    stores = ctx.get_context().raw("stores")
    return {
        "tool": "list_stores",
        "summary": {"store_count": int(len(stores))},
        "records": ctx.records(
            stores.sort_values("store_id") if "store_id" in stores.columns else stores,
            ["store_id", "store_name", "city", "capacity"],
            limit,
        ),
        "sources": ctx.sources("stores"),
    }


def list_products(category: str = "", limit: int = 0) -> dict:
    """List products in the catalog, optionally filtered to one category."""
    limit = _list_limit(limit)
    products = ctx.get_context().raw("products")
    if category and "category" in products.columns:
        products = products[
            products["category"].astype(str).str.strip().str.lower()
            == category.strip().lower()
        ]
    return {
        "tool": "list_products",
        "summary": {
            "product_count": int(len(products)),
            "categories": (
                sorted(products["category"].dropna().astype(str).unique().tolist())[:20]
                if "category" in products.columns
                else []
            ),
        },
        "records": ctx.records(
            products,
            ["product_id", "product_name", "category", "cost_price", "selling_price", "supplier_id"],
            limit,
        ),
        "sources": ctx.sources("products"),
    }


def list_suppliers(limit: int = 0) -> dict:
    """List suppliers with delivery lead time and reliability score."""
    limit = _list_limit(limit)
    suppliers = ctx.get_context().raw("suppliers")
    return {
        "tool": "list_suppliers",
        "summary": {"supplier_count": int(len(suppliers))},
        "records": ctx.records(
            suppliers,
            ["supplier_id", "supplier_name", "avg_delivery_days", "reliability_score"],
            limit,
        ),
        "sources": ctx.sources("suppliers"),
    }


def validate_location(location: str) -> dict:
    """Check whether a city/branch is in scope before answering location questions."""
    result = validate_requested_location(location)
    return {
        "tool": "validate_location",
        "summary": {
            "location": location,
            "is_available": bool(result.is_available),
        },
        "records": [],
        "sources": ctx.sources("stores"),
        "notes": (
            ""
            if result.is_available
            else str((result.payload or {}).get("answer", "Location is not in scope."))
        ),
    }
