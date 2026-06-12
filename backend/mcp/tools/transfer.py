"""Transfer tool: cross-branch stock rebalancing opportunities.

Reuses backend.services.transfer_analysis_service, fed live from Oracle.
"""

from __future__ import annotations

import pandas as pd

from backend.mcp import context as ctx
from backend.services.transfer_analysis_service import analyze_transfer_opportunities


def get_transfer_opportunities(limit: int = 10) -> dict:
    """Source→target transfer suggestions: a branch with surplus covers another
    branch's shortage of the same product before new procurement is needed."""
    limit = ctx.clamp_limit(limit)
    context = ctx.get_context()
    result = analyze_transfer_opportunities(
        inventory=context.raw("inventory"),
        sales=context.raw("sales"),
        stores=context.raw("stores"),
        products=context.raw("products"),
        limit=limit,
    )
    opportunities = result.get("opportunities", None)
    rows = 0 if opportunities is None or opportunities.empty else len(opportunities)
    return {
        "tool": "get_transfer_opportunities",
        "summary": {"opportunity_count": int(rows)},
        "records": ctx.records(
            opportunities if opportunities is not None else pd.DataFrame(),
            ["product_name", "source_store_name", "target_store_name",
             "suggested_quantity", "priority", "source_stock", "target_stock",
             "target_days_remaining"],
            limit,
        ),
        "sources": ctx.sources("inventory", "sales", "stores", "products"),
    }
