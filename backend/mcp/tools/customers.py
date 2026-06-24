"""Customer & order tools — real customer/order model.

Oracle DOES have a real end-customer dimension: BZ_MOCK_CUSTOMER ->
BZ_MOCK_ORDER_HEADER -> BZ_MOCK_ORDER_LINE. These tools answer from that model
(loaded live via the shared MCP context), so they reflect freshly placed orders —
including those created in the Customer Order Simulator — the moment the context
is refreshed. All abnormal-order logic is shared with the Customer Intelligence
page via ``customer_intelligence_service`` so the chatbot and the page agree.
"""

from __future__ import annotations

import pandas as pd

from backend.mcp import context as ctx
from backend.services import customer_intelligence_service as cis


_CUSTOMER_NOTE = (
    "'Customer' is the real end-customer (BZ_MOCK_CUSTOMER); orders come from "
    "BZ_MOCK_ORDER_HEADER/ORDER_LINE. Computed live from Oracle."
)


def _facts() -> pd.DataFrame:
    """Line-level customer/order fact frame from the shared context."""
    return ctx.get_context().customer_order_facts()


def get_top_customers(metric: str = "revenue", limit: int = 10) -> dict:
    """Top end-customers by order revenue (default) or units ordered."""
    limit = ctx.clamp_limit(limit)
    metric = "units" if str(metric).lower().startswith("unit") else "revenue"
    facts = _facts()
    if facts is None or facts.empty:
        return {
            "tool": "get_top_customers",
            "summary": {"count": 0},
            "records": [],
            "sources": ctx.sources("orders", "order_lines", "customers"),
            "notes": _CUSTOMER_NOTE,
        }
    leaders = cis.top_customers(facts, limit=None)
    sort_col = "units" if metric == "units" else "revenue"
    leaders = leaders.sort_values(sort_col, ascending=False)
    return {
        "tool": "get_top_customers",
        "summary": {"metric": metric, "customer_count": int(len(leaders))},
        "records": ctx.records(
            leaders,
            ["customer_name", "customer_segment", "contract_tier",
             "orders", "units", "revenue", "contribution_pct"],
            limit,
        ),
        "sources": ctx.sources("orders", "order_lines", "customers"),
        "notes": _CUSTOMER_NOTE,
    }


def get_customer_order_analysis(customer: str = "", limit: int = 10) -> dict:
    """Per-customer ordering profile: orders, units, revenue, average line size,
    product variety, and most recent order date. Optionally filter by customer
    name (partial match)."""
    limit = ctx.clamp_limit(limit)
    facts = _facts()
    if facts is None or facts.empty:
        return {
            "tool": "get_customer_order_analysis",
            "summary": {"count": 0},
            "records": [],
            "sources": ctx.sources("orders", "order_lines", "customers"),
            "notes": _CUSTOMER_NOTE,
        }
    work = facts.copy()
    if customer:
        work = work[work["customer_name"].astype(str).str.contains(str(customer), case=False, na=False)]
    if work.empty:
        return {
            "tool": "get_customer_order_analysis",
            "summary": {"scope": customer or "all_customers", "count": 0},
            "records": [],
            "sources": ctx.sources("orders", "order_lines", "customers"),
            "notes": _CUSTOMER_NOTE,
        }
    agg = work.groupby(["customer_id", "customer_name"], as_index=False).agg(
        orders=("order_id", "nunique"),
        order_lines=("quantity", "size"),
        units_ordered=("quantity", "sum"),
        revenue=("revenue", "sum"),
        avg_line_size=("quantity", "mean"),
        distinct_products=("product_id", "nunique"),
        last_order_date=("order_date", "max"),
    )
    agg["units_ordered"] = agg["units_ordered"].round().astype(int)
    agg["revenue"] = agg["revenue"].round(2)
    agg["avg_line_size"] = agg["avg_line_size"].round(2)
    agg["last_order_date"] = pd.to_datetime(agg["last_order_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    agg = agg.sort_values("revenue", ascending=False)
    return {
        "tool": "get_customer_order_analysis",
        "summary": {"scope": customer or "all_customers", "customer_count": int(len(agg))},
        "records": ctx.records(
            agg,
            ["customer_name", "orders", "order_lines", "units_ordered", "revenue",
             "avg_line_size", "distinct_products", "last_order_date"],
            limit,
        ),
        "sources": ctx.sources("orders", "order_lines", "customers"),
        "notes": _CUSTOMER_NOTE,
    }


def get_customer_products(customer: str = "", metric: str = "quantity", limit: int = 25) -> dict:
    """Products a specific customer has actually ordered — the product-level breakdown.

    THE tool for any product-level customer-order question, e.g. "what products did
    <customer> order?", "show all products ordered by <customer>", "which product
    does <customer> buy most frequently?", "which product generated the most revenue
    for <customer>?", "what was the latest product <customer> ordered?", or "quantity
    purchased per product for <customer>". Unlike ``get_customer_order_analysis``
    (which returns only a distinct-product COUNT), this resolves the full chain
    BZ_MOCK_CUSTOMER -> ORDER_HEADER -> ORDER_LINE -> BZ_MOCK_PRODUCT and returns one
    row per product with its name, category, total quantity, revenue, the number of
    orders containing it, and the most recent order date.

    ``customer`` is matched case-insensitively as a partial name and is required to
    scope the answer to one customer. ``metric`` sets the sort order: "quantity"
    (default), "revenue", "frequency" (most orders first), or "recent" (latest first).
    """
    limit = ctx.clamp_limit(limit)
    facts = _facts()
    if facts is None or facts.empty:
        return {
            "tool": "get_customer_products",
            "summary": {"scope": customer or "all_customers", "count": 0},
            "records": [],
            "sources": ctx.sources("orders", "order_lines", "customers", "products"),
            "notes": _CUSTOMER_NOTE,
        }
    work = facts.copy()
    if customer:
        work = work[work["customer_name"].astype(str).str.contains(str(customer), case=False, na=False)]
    if work.empty:
        return {
            "tool": "get_customer_products",
            "summary": {"scope": customer or "all_customers", "count": 0},
            "records": [],
            "sources": ctx.sources("orders", "order_lines", "customers", "products"),
            "notes": _CUSTOMER_NOTE + f" No orders found for a customer matching '{customer}'.",
        }

    group_cols = [c for c in ["customer_name", "product_id", "product_name", "category"]
                  if c in work.columns]
    by_product = work.groupby(group_cols, as_index=False).agg(
        total_quantity=("quantity", "sum"),
        revenue=("revenue", "sum"),
        orders=("order_id", "nunique"),
        last_order_date=("order_date", "max"),
    )
    by_product["total_quantity"] = by_product["total_quantity"].round().astype(int)
    by_product["revenue"] = by_product["revenue"].round(2)
    by_product["last_order_date"] = pd.to_datetime(
        by_product["last_order_date"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")

    sort_by = {
        "revenue": "revenue",
        "frequency": "orders",
        "freq": "orders",
        "orders": "orders",
        "recent": "last_order_date",
        "latest": "last_order_date",
    }.get(str(metric).lower(), "total_quantity")
    by_product = by_product.sort_values(sort_by, ascending=False)

    matched_customers = sorted(work["customer_name"].dropna().astype(str).unique())
    return {
        "tool": "get_customer_products",
        "summary": {
            "scope": customer or "all_customers",
            "matched_customers": matched_customers,
            "product_count": int(len(by_product)),
            "total_units": int(by_product["total_quantity"].sum()),
            "sorted_by": sort_by,
        },
        "records": ctx.records(
            by_product,
            ["customer_name", "product_name", "category", "total_quantity",
             "revenue", "orders", "last_order_date"],
            limit,
        ),
        "sources": ctx.sources("orders", "order_lines", "customers", "products"),
        "notes": _CUSTOMER_NOTE + " One row per product the customer has ordered.",
    }


def get_recent_orders(limit: int = 10) -> dict:
    """Most recent customer orders (latest first) from BZ_MOCK_ORDER_HEADER.

    Use for "latest customer order", "what was just ordered", or "recent orders
    today" — returns each order's number, date, customer, branch, status and
    total, newest first, so a freshly placed order appears at the top.
    """
    limit = ctx.clamp_limit(limit)
    context = ctx.get_context()
    orders = context.raw("orders")
    customers = context.raw("customers")
    if orders.empty:
        return {
            "tool": "get_recent_orders",
            "summary": {"count": 0},
            "records": [],
            "sources": ctx.sources("orders", "customers"),
            "notes": _CUSTOMER_NOTE,
        }
    o = orders.copy()
    o["order_date"] = pd.to_datetime(o.get("order_date"), errors="coerce")
    o["order_total"] = ctx.num(o, "order_total").round(2)
    if not customers.empty:
        cust_cols = [c for c in ["customer_id", "customer_name", "customer_segment", "contract_tier"]
                     if c in customers.columns]
        c = customers[cust_cols].copy()
        c["customer_id"] = c["customer_id"].astype(str)
        o["customer_id"] = o["customer_id"].astype(str)
        o = o.merge(c, on="customer_id", how="left")
    sort_cols = [col for col in ["order_date", "order_id"] if col in o.columns]
    o = o.sort_values(sort_cols, ascending=False)
    o["order_date"] = o["order_date"].dt.strftime("%Y-%m-%d")
    return {
        "tool": "get_recent_orders",
        "summary": {"order_count": int(len(o))},
        "records": ctx.records(
            o,
            ["order_nbr", "order_date", "customer_name", "customer_segment",
             "contract_tier", "store_id", "order_status", "order_total"],
            limit,
        ),
        "sources": ctx.sources("orders", "customers"),
        "notes": _CUSTOMER_NOTE,
    }


def detect_abnormal_ordering(
    limit: int = 10, z_threshold: float = 3.0, min_deviation_pct: float = 0.0
) -> dict:
    """Detect abnormal customer orders against each product's prior demand baseline.

    Evaluates the latest order of every product against the average of its prior
    orders (the order being judged is excluded from its own baseline). An order is
    abnormal when its quantity is at least ``min_deviation_pct`` above that prior
    average. Use this for "any abnormal orders today?", "unusual orders", or
    "abnormal orders above 40%" (pass min_deviation_pct=40). When no percentage is
    given the Customer Intelligence default threshold is used; ``z_threshold`` is
    accepted for backward compatibility but ignored. Results match the Customer
    Intelligence page and include the customer, product, quantity, historical
    average / maximum, deviation %, and risk level.
    """
    limit = ctx.clamp_limit(limit)
    threshold = float(min_deviation_pct or 0.0)
    if threshold <= 0.0:
        threshold = float(cis.DEFAULT_ABNORMAL_DEVIATION_PCT)

    facts = _facts()
    abnormal = cis.detect_abnormal_orders(facts, min_deviation_pct=threshold)
    if abnormal is None or abnormal.empty:
        return {
            "tool": "detect_abnormal_ordering",
            "summary": {"abnormal_count": 0, "min_deviation_pct": round(threshold, 1)},
            "records": [],
            "sources": ctx.sources("orders", "order_lines", "customers"),
            "notes": _CUSTOMER_NOTE
            + f" Abnormal = latest order >= prior-order average + {threshold:.0f}% per product.",
        }
    return {
        "tool": "detect_abnormal_ordering",
        "summary": {
            "abnormal_count": int(len(abnormal)),
            "method": "deviation_pct",
            "min_deviation_pct": round(threshold, 1),
        },
        "records": ctx.records(
            abnormal,
            ["customer_name", "product_name", "order_nbr", "order_date",
             "current_quantity", "historical_avg", "historical_max",
             "deviation_pct", "risk_level"],
            limit,
        ),
        "sources": ctx.sources("orders", "order_lines", "customers"),
        "notes": _CUSTOMER_NOTE
        + f" Abnormal = latest order >= prior-order average + {threshold:.0f}% per product.",
    }


def get_customer_demand_trends(trend: str = "", limit: int = 10) -> dict:
    """Customers whose demand is Growing, Stable, or Declining over the order window.

    THE tool for "which customers are growing?", "who is declining/churning?",
    "show demand trends", or "which account grew the most?". Splits the order
    window at its midpoint and compares each customer's second-half revenue to the
    first half: > +10% = Growing, < -10% = Declining, else Stable (brand-new buyers
    count as Growing). Mirrors the Customer Intelligence page exactly. Pass
    ``trend`` = "growing", "declining", or "stable" to return only that bucket;
    leave it blank for all customers, ranked by revenue change (largest growth first).
    A directional signal over a short window, not a forecast.
    """
    limit = ctx.clamp_limit(limit)
    facts = _facts()
    trends = cis.customer_demand_trends(facts)
    if trends is None or trends.empty:
        return {
            "tool": "get_customer_demand_trends",
            "summary": {"count": 0},
            "records": [],
            "sources": ctx.sources("orders", "order_lines", "customers"),
            "notes": _CUSTOMER_NOTE
            + " Trend = second-half vs first-half order revenue (+/-10% bands).",
        }
    wanted = str(trend).strip().lower()
    if wanted:
        canonical = {"grow": "Growing", "growing": "Growing", "up": "Growing",
                     "declin": "Declining", "declining": "Declining", "down": "Declining",
                     "churn": "Declining", "stable": "Stable", "flat": "Stable"}
        target = next((v for k, v in canonical.items() if wanted.startswith(k)), None)
        if target:
            trends = trends[trends["trend"] == target]
    counts = (
        cis.customer_demand_trends(facts)["trend"].value_counts().to_dict()
        if facts is not None and not facts.empty else {}
    )
    return {
        "tool": "get_customer_demand_trends",
        "summary": {
            "scope": wanted or "all_customers",
            "customer_count": int(len(trends)),
            "growing": int(counts.get("Growing", 0)),
            "stable": int(counts.get("Stable", 0)),
            "declining": int(counts.get("Declining", 0)),
        },
        "records": ctx.records(
            trends,
            ["customer_name", "trend", "change_pct",
             "first_half_revenue", "second_half_revenue"],
            limit,
        ),
        "sources": ctx.sources("orders", "order_lines", "customers"),
        "notes": _CUSTOMER_NOTE
        + " Trend = second-half vs first-half order revenue (+/-10% bands).",
    }


def get_dormant_accounts(limit: int = 10) -> dict:
    """Active customers who have never placed an order — a re-engagement list.

    THE tool for "which customers are dormant?", "who hasn't ordered?", "inactive
    accounts", or "re-engagement opportunities". Returns active (active_flg='Y')
    BZ_MOCK_CUSTOMER rows with zero orders in BZ_MOCK_ORDER_HEADER, with each
    account's segment, contract tier, industry, city, credit limit, and signup
    date. Matches the Dormant Accounts KPI on the Customer Intelligence page.
    """
    limit = ctx.clamp_limit(limit)
    context = ctx.get_context()
    customers = context.raw("customers")
    orders = context.raw("orders")
    dormant = cis.dormant_accounts(customers, orders)
    if dormant is None or dormant.empty:
        return {
            "tool": "get_dormant_accounts",
            "summary": {"dormant_count": 0},
            "records": [],
            "sources": ctx.sources("customers", "orders"),
            "notes": _CUSTOMER_NOTE + " Dormant = active customer with zero orders.",
        }
    if "signup_date" in dormant.columns:
        dormant["signup_date"] = pd.to_datetime(
            dormant["signup_date"], errors="coerce"
        ).dt.strftime("%Y-%m-%d")
    return {
        "tool": "get_dormant_accounts",
        "summary": {"dormant_count": int(len(dormant))},
        "records": ctx.records(
            dormant,
            ["customer_name", "customer_segment", "contract_tier", "industry",
             "city", "credit_limit", "signup_date"],
            limit,
        ),
        "sources": ctx.sources("customers", "orders"),
        "notes": _CUSTOMER_NOTE + " Dormant = active customer with zero orders.",
    }


def get_order_inventory_impact(limit: int = 10) -> dict:
    """Inventory pressure created by customer orders on at-risk products.

    Use for "inventory impact of recent orders" or "which customers strain
    inventory": lists customers ordering products that are at/below their reorder
    point, with the at-risk units, revenue, affected products, and a 0-100 risk
    score — the same view as the Customer Intelligence page.
    """
    limit = ctx.clamp_limit(limit)
    context = ctx.get_context()
    facts = context.customer_order_facts()
    inventory = context.raw("inventory")
    impact = cis.inventory_impact(facts, inventory)
    if impact is None or impact.empty:
        return {
            "tool": "get_order_inventory_impact",
            "summary": {"count": 0},
            "records": [],
            "sources": ctx.sources("orders", "order_lines", "inventory"),
            "notes": _CUSTOMER_NOTE + " At risk = stock at/below reorder point in any branch.",
        }
    return {
        "tool": "get_order_inventory_impact",
        "summary": {"customers_at_risk": int(len(impact))},
        "records": ctx.records(
            impact,
            ["customer_name", "at_risk_units", "at_risk_revenue",
             "affected_products", "products_detail", "risk_score"],
            limit,
        ),
        "sources": ctx.sources("orders", "order_lines", "inventory"),
        "notes": _CUSTOMER_NOTE + " At risk = stock at/below reorder point in any branch.",
    }
