"""Forecasting tools: stockout risk, demand forecast, high-demand items.

All derive from the predictive inventory view (built live from Oracle by
backend.services.inventory_prediction_service).
"""

from __future__ import annotations

from backend.mcp import context as ctx


_FORECAST_COLUMNS = [
    "product_id", "product_name", "store_id", "store_name", "category",
    "current_quantity", "avg_daily_sales", "moving_avg_daily_sales",
    "predicted_days_remaining", "depletion_window", "demand_trend",
    "demand_spike", "confidence_level", "risk_score", "risk_category",
    "suggested_reorder_qty", "alert_reason",
]


def _scoped_predictive(store_id: str = "", product_id: str = ""):
    view = ctx.get_context().predictive_view()
    if view.empty:
        return view
    if store_id and "store_id" in view.columns:
        view = view[view["store_id"].astype(str) == str(store_id)]
    if product_id and "product_id" in view.columns:
        view = view[view["product_id"].astype(str) == str(product_id)]
    return view


def get_stockout_risk(store_id: str = "", limit: int = 10) -> dict:
    """Products most at risk of stocking out, ranked by predictive risk score.

    Flags items predicted to deplete within the alert window (or below reorder
    threshold with no recent sales). Includes a suggested reorder quantity.
    """
    limit = ctx.clamp_limit(limit)
    view = _scoped_predictive(store_id=store_id)
    if view.empty:
        return {
            "tool": "get_stockout_risk",
            "summary": {"at_risk_count": 0, "scope": store_id or "all_stores"},
            "records": [],
            "sources": ctx.sources("inventory", "products", "stores", "sales", "suppliers"),
            "notes": "No predictive inventory rows available from Oracle.",
        }
    at_risk = view[view.get("predictive_alert", False) == True].copy()  # noqa: E712
    if at_risk.empty:
        at_risk = view.copy()
    at_risk = at_risk.sort_values("risk_score", ascending=False)
    return {
        "tool": "get_stockout_risk",
        "summary": {
            "scope": store_id or "all_stores",
            "at_risk_count": int((view.get("predictive_alert", False) == True).sum()),  # noqa: E712
            "critical_count": int((at_risk.get("risk_category", "") == "Critical").sum()),
        },
        "records": ctx.records(at_risk, _FORECAST_COLUMNS, limit),
        "sources": ctx.sources("inventory", "products", "stores", "sales", "suppliers"),
    }


def get_demand_forecast(product_id: str = "", store_id: str = "", limit: int = 10) -> dict:
    """Per product-store demand signal: average daily sales, predicted days of
    cover, demand trend (Rising/Falling/Stable/Spike), and confidence."""
    limit = ctx.clamp_limit(limit)
    view = _scoped_predictive(store_id=store_id, product_id=product_id)
    if view.empty:
        return {
            "tool": "get_demand_forecast",
            "summary": {"count": 0},
            "records": [],
            "sources": ctx.sources("inventory", "products", "stores", "sales"),
            "notes": "No predictive inventory rows available from Oracle.",
        }
    view = view.sort_values("avg_daily_sales", ascending=False)
    return {
        "tool": "get_demand_forecast",
        "summary": {
            "scope": " / ".join(filter(None, [product_id, store_id])) or "all",
            "rows": int(len(view)),
            "rising_count": int((view.get("demand_trend", "") == "Rising").sum()),
            "spike_count": int(view.get("demand_spike", False).astype(bool).sum()) if "demand_spike" in view.columns else 0,
        },
        "records": ctx.records(view, _FORECAST_COLUMNS, limit),
        "sources": ctx.sources("inventory", "products", "stores", "sales"),
    }


def get_high_demand_items(store_id: str = "", limit: int = 10) -> dict:
    """Products with rising demand or a demand spike (protect their stock cover)."""
    limit = ctx.clamp_limit(limit)
    view = _scoped_predictive(store_id=store_id)
    if view.empty:
        return {
            "tool": "get_high_demand_items",
            "summary": {"count": 0, "scope": store_id or "all_stores"},
            "records": [],
            "sources": ctx.sources("inventory", "products", "stores", "sales"),
            "notes": "No predictive inventory rows available from Oracle.",
        }
    trend = view.get("demand_trend", "")
    spike = view.get("demand_spike", False)
    high = view[(trend.isin(["Rising", "Demand Spike"])) | (spike.astype(bool))].copy()
    high = high.sort_values("avg_daily_sales", ascending=False)
    return {
        "tool": "get_high_demand_items",
        "summary": {"scope": store_id or "all_stores", "high_demand_count": int(len(high))},
        "records": ctx.records(high, _FORECAST_COLUMNS, limit),
        "sources": ctx.sources("inventory", "products", "stores", "sales"),
    }
