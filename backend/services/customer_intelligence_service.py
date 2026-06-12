"""Customer Intelligence analytics.

Pure-function analytics over Oracle's real customer/order model
(``BZ_MOCK_CUSTOMER`` -> ``BZ_MOCK_ORDER_HEADER`` -> ``BZ_MOCK_ORDER_LINE``).
Every function takes DataFrames (loaded via ``backend.db.repository``) and returns
DataFrames / plain dicts, mirroring the ``sales_analytics_service`` pattern so the
page stays a thin view layer.

Revenue convention: ``LINE_TOTAL_AMT`` (the ``line_total`` column) is the
authoritative revenue figure. It already incorporates contract pricing and
discounts, so revenue is never recomputed from quantity x price.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# Order line is "abnormal" when its quantity is well above the product's
# historical average. Detection is deviation-driven (NOT z-score gated): with a
# short order history per-product variance is large, so z-scores understate real
# outliers. A small absolute floor avoids flagging noise on tiny-mean products.
_ABNORMAL_MIN_DEVIATION_PCT = 75.0   # at least +75% over the product baseline
_ABNORMAL_MIN_ABS_GAP = 5.0          # and at least 5 units above the mean
_HIGH_RISK_DEVIATION_PCT = 150.0     # >= +150% over baseline => High risk

# Demand trend: split the order window in half and compare revenue.
_TREND_GROWTH_BAND = 10.0            # +/-10% revenue change => "Stable"


def _num(df: pd.DataFrame, column: str) -> pd.Series:
    """Numeric view of a column, 0-filled, safe when the column is absent."""
    if column not in df.columns:
        return pd.Series(0.0, index=df.index, dtype="float64")
    return pd.to_numeric(df[column], errors="coerce").fillna(0.0)


# ---------------------------------------------------------------------------
# Core fact table
# ---------------------------------------------------------------------------
def prepare_customer_orders(
    order_lines: pd.DataFrame,
    orders: pd.DataFrame,
    customers: pd.DataFrame,
    products: pd.DataFrame,
) -> pd.DataFrame:
    """Build the line-level fact frame joining lines -> orders -> customers/products.

    One row per order line, enriched with customer, product, branch, and date.
    ``revenue`` is sourced from ``line_total`` (authoritative).
    """
    if order_lines is None or order_lines.empty or orders is None or orders.empty:
        return pd.DataFrame()

    lines = order_lines.copy()
    lines["product_id"] = lines["product_id"].astype(str)
    lines["order_id"] = lines["order_id"].astype(str)
    lines["quantity"] = _num(lines, "quantity")
    lines["revenue"] = _num(lines, "line_total")

    header_cols = [
        c for c in ["order_id", "order_nbr", "customer_id", "store_id",
                    "order_date", "order_channel", "order_status"]
        if c in orders.columns
    ]
    header = orders[header_cols].copy()
    header["order_id"] = header["order_id"].astype(str)
    if "customer_id" in header.columns:
        header["customer_id"] = header["customer_id"].astype(str)
    if "order_date" in header.columns:
        header["order_date"] = pd.to_datetime(header["order_date"], errors="coerce")

    facts = lines.merge(header, on="order_id", how="left")

    if customers is not None and not customers.empty:
        cust_cols = [
            c for c in ["customer_id", "customer_name", "customer_segment",
                        "contract_tier", "industry", "city", "state"]
            if c in customers.columns
        ]
        cust = customers[cust_cols].copy()
        cust["customer_id"] = cust["customer_id"].astype(str)
        facts = facts.merge(cust, on="customer_id", how="left")

    if products is not None and not products.empty:
        prod_cols = [c for c in ["product_id", "product_name", "category"]
                     if c in products.columns]
        prod = products[prod_cols].copy()
        prod["product_id"] = prod["product_id"].astype(str)
        facts = facts.merge(prod, on="product_id", how="left")

    if "customer_name" not in facts.columns:
        facts["customer_name"] = facts.get("customer_id", "")
    facts["customer_name"] = facts["customer_name"].fillna("Unknown Customer")
    if "product_name" not in facts.columns:
        facts["product_name"] = facts["product_id"]
    facts["product_name"] = facts["product_name"].fillna(facts["product_id"])
    for col in ("customer_segment", "contract_tier"):
        if col not in facts.columns:
            facts[col] = ""
        facts[col] = facts[col].fillna("")
    return facts


# ---------------------------------------------------------------------------
# Section 2 - Top products
# ---------------------------------------------------------------------------
def top_products(facts: pd.DataFrame, metric: str = "revenue", limit: int = 10) -> pd.DataFrame:
    """Top products by ``revenue`` (default) or ``quantity``."""
    if facts is None or facts.empty:
        return pd.DataFrame(columns=["product_id", "product_name", "category", "quantity", "revenue"])
    metric = "quantity" if str(metric).lower().startswith(("qty", "quan", "unit")) else "revenue"
    group_cols = [c for c in ["product_id", "product_name", "category"] if c in facts.columns]
    grouped = (
        facts.groupby(group_cols, as_index=False)
        .agg(quantity=("quantity", "sum"), revenue=("revenue", "sum"))
        .sort_values(metric, ascending=False)
    )
    grouped["quantity"] = grouped["quantity"].round().astype(int)
    grouped["revenue"] = grouped["revenue"].round(2)
    return grouped.head(limit).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Section 3 - Top customers
# ---------------------------------------------------------------------------
def top_customers(facts: pd.DataFrame, limit: int | None = 10) -> pd.DataFrame:
    """Customer leaderboard: orders, units, revenue, contribution %.

    ``contribution_pct`` is each customer's share of total order revenue.
    """
    columns = ["customer_id", "customer_name", "customer_segment", "contract_tier",
               "orders", "units", "revenue", "contribution_pct"]
    if facts is None or facts.empty:
        return pd.DataFrame(columns=columns)

    grouped = (
        facts.groupby(["customer_id", "customer_name"], as_index=False)
        .agg(
            customer_segment=("customer_segment", "first"),
            contract_tier=("contract_tier", "first"),
            orders=("order_id", "nunique"),
            units=("quantity", "sum"),
            revenue=("revenue", "sum"),
        )
    )
    total_revenue = grouped["revenue"].sum()
    grouped["contribution_pct"] = (
        (grouped["revenue"] / total_revenue * 100.0) if total_revenue else 0.0
    ).round(1)
    grouped["units"] = grouped["units"].round().astype(int)
    grouped["revenue"] = grouped["revenue"].round(2)
    grouped = grouped.sort_values("revenue", ascending=False).reset_index(drop=True)
    if limit:
        grouped = grouped.head(limit)
    return grouped[columns]


def customer_spotlight(facts: pd.DataFrame, limit: int = 5) -> list[dict]:
    """Top ``limit`` customers as card-ready dicts (name, revenue, orders, segment, tier)."""
    leaders = top_customers(facts, limit=limit)
    spotlight = []
    for _, row in leaders.iterrows():
        spotlight.append({
            "customer_name": str(row["customer_name"]),
            "revenue": float(row["revenue"]),
            "orders": int(row["orders"]),
            "segment": str(row["customer_segment"] or "-"),
            "tier": str(row["contract_tier"] or "-"),
        })
    return spotlight


# ---------------------------------------------------------------------------
# Section 4 - Abnormal order detection (per-product baseline)
# ---------------------------------------------------------------------------
def detect_abnormal_orders(facts: pd.DataFrame, limit: int | None = None) -> pd.DataFrame:
    """Flag order lines whose quantity deviates sharply from the product baseline.

    For each product the historical mean quantity is computed across all order
    lines; a line is abnormal when its quantity is at least
    ``_ABNORMAL_MIN_DEVIATION_PCT`` above that mean and at least
    ``_ABNORMAL_MIN_ABS_GAP`` units above it. The anomaly is attributed to the
    customer who placed the order.
    """
    columns = ["customer_name", "product_name", "order_nbr", "order_date",
               "historical_avg", "current_quantity", "deviation_pct", "risk_level"]
    if facts is None or facts.empty:
        return pd.DataFrame(columns=columns)

    work = facts.copy()
    work["product_mean"] = (
        work.groupby("product_id")["quantity"].transform("mean").fillna(0.0)
    )
    mean_safe = work["product_mean"].replace(0, np.nan)
    work["deviation_pct"] = ((work["quantity"] - work["product_mean"]) / mean_safe * 100.0).fillna(0.0)
    work["abs_gap"] = work["quantity"] - work["product_mean"]

    flagged = work[
        (work["deviation_pct"] >= _ABNORMAL_MIN_DEVIATION_PCT)
        & (work["abs_gap"] >= _ABNORMAL_MIN_ABS_GAP)
    ].copy()
    if flagged.empty:
        return pd.DataFrame(columns=columns)

    flagged["risk_level"] = np.where(
        flagged["deviation_pct"] >= _HIGH_RISK_DEVIATION_PCT, "High", "Medium"
    )
    flagged["historical_avg"] = flagged["product_mean"].round(1)
    flagged["current_quantity"] = flagged["quantity"].round().astype(int)
    flagged["deviation_pct"] = flagged["deviation_pct"].round(1)
    if "order_date" in flagged.columns:
        flagged["order_date"] = flagged["order_date"].dt.strftime("%Y-%m-%d")
    if "order_nbr" not in flagged.columns:
        flagged["order_nbr"] = flagged.get("order_id", "")

    flagged = flagged.sort_values("deviation_pct", ascending=False).reset_index(drop=True)
    if limit:
        flagged = flagged.head(limit)
    return flagged[[c for c in columns if c in flagged.columns]]


# ---------------------------------------------------------------------------
# Section 5 - Customer demand trends
# ---------------------------------------------------------------------------
def customer_demand_trends(facts: pd.DataFrame) -> pd.DataFrame:
    """Classify each customer as Growing / Stable / Declining over the order window.

    The window between the first and last order date is split at its midpoint;
    a customer's revenue in the second half is compared with the first half.
    Short histories make this a directional signal, not a forecast.
    """
    columns = ["customer_id", "customer_name", "first_half_revenue",
               "second_half_revenue", "change_pct", "trend"]
    if facts is None or facts.empty or "order_date" not in facts.columns:
        return pd.DataFrame(columns=columns)

    dated = facts.dropna(subset=["order_date"]).copy()
    if dated.empty:
        return pd.DataFrame(columns=columns)

    start, end = dated["order_date"].min(), dated["order_date"].max()
    midpoint = start + (end - start) / 2
    dated["half"] = np.where(dated["order_date"] <= midpoint, "first", "second")

    pivot = (
        dated.groupby(["customer_id", "customer_name", "half"], as_index=False)["revenue"].sum()
        .pivot_table(index=["customer_id", "customer_name"], columns="half",
                     values="revenue", fill_value=0.0)
        .reset_index()
    )
    pivot.columns.name = None
    if "first" not in pivot.columns:
        pivot["first"] = 0.0
    if "second" not in pivot.columns:
        pivot["second"] = 0.0
    pivot = pivot.rename(columns={"first": "first_half_revenue", "second": "second_half_revenue"})

    first = pivot["first_half_revenue"].replace(0, np.nan)
    pivot["change_pct"] = (
        (pivot["second_half_revenue"] - pivot["first_half_revenue"]) / first * 100.0
    )
    # New customers (no first-half revenue but second-half activity) => Growing.
    pivot.loc[pivot["first_half_revenue"].eq(0) & pivot["second_half_revenue"].gt(0), "change_pct"] = 100.0
    pivot["change_pct"] = pivot["change_pct"].fillna(0.0)

    def _label(change: float) -> str:
        if change > _TREND_GROWTH_BAND:
            return "Growing"
        if change < -_TREND_GROWTH_BAND:
            return "Declining"
        return "Stable"

    pivot["trend"] = pivot["change_pct"].map(_label)
    pivot["first_half_revenue"] = pivot["first_half_revenue"].round(2)
    pivot["second_half_revenue"] = pivot["second_half_revenue"].round(2)
    pivot["change_pct"] = pivot["change_pct"].round(1)
    return pivot.sort_values("change_pct", ascending=False).reset_index(drop=True)[columns]


def highest_growth_customer(facts: pd.DataFrame) -> dict:
    """Return the customer with the largest positive revenue change, or empty dict."""
    trends = customer_demand_trends(facts)
    if trends.empty:
        return {}
    growing = trends[trends["trend"] == "Growing"]
    pool = growing if not growing.empty else trends
    top = pool.iloc[0]
    return {
        "customer_name": str(top["customer_name"]),
        "change_pct": float(top["change_pct"]),
        "trend": str(top["trend"]),
    }


# ---------------------------------------------------------------------------
# Section 6 - Inventory impact analysis
# ---------------------------------------------------------------------------
def at_risk_products(inventory: pd.DataFrame) -> set[str]:
    """Product ids at or below their reorder point in any branch."""
    if inventory is None or inventory.empty:
        return set()
    inv = inventory.copy()
    if "product_id" not in inv.columns:
        return set()
    stock = pd.to_numeric(inv.get("stock_level"), errors="coerce").fillna(0)
    reorder = pd.to_numeric(inv.get("reorder_threshold"), errors="coerce").fillna(0)
    risky = inv.loc[stock <= reorder, "product_id"].astype(str)
    return set(risky.unique())


def inventory_impact(facts: pd.DataFrame, inventory: pd.DataFrame, limit: int | None = None) -> pd.DataFrame:
    """Customers driving inventory pressure by ordering at-risk products.

    A product is "at risk" when its stock is at/below the reorder point in any
    branch. For each customer ordering such products we sum the units/revenue on
    those products and derive a 0-100 ``risk_score`` (relative to the heaviest
    contributor).
    """
    columns = ["customer_id", "customer_name", "at_risk_units", "at_risk_revenue",
               "affected_products", "products_detail", "risk_score"]
    if facts is None or facts.empty:
        return pd.DataFrame(columns=columns)
    risky = at_risk_products(inventory)
    if not risky:
        return pd.DataFrame(columns=columns)

    pressure = facts[facts["product_id"].astype(str).isin(risky)].copy()
    if pressure.empty:
        return pd.DataFrame(columns=columns)

    grouped = (
        pressure.groupby(["customer_id", "customer_name"], as_index=False)
        .agg(
            at_risk_units=("quantity", "sum"),
            at_risk_revenue=("revenue", "sum"),
            affected_products=("product_id", "nunique"),
            products_detail=("product_name", lambda s: ", ".join(sorted(set(s.dropna().astype(str))))),
        )
    )
    max_units = grouped["at_risk_units"].max()
    grouped["risk_score"] = (
        (grouped["at_risk_units"] / max_units * 100.0) if max_units else 0.0
    ).round().astype(int)
    grouped["at_risk_units"] = grouped["at_risk_units"].round().astype(int)
    grouped["at_risk_revenue"] = grouped["at_risk_revenue"].round(2)
    grouped = grouped.sort_values("risk_score", ascending=False).reset_index(drop=True)
    if limit:
        grouped = grouped.head(limit)
    return grouped[columns]


# ---------------------------------------------------------------------------
# Section 7 - Dormant accounts
# ---------------------------------------------------------------------------
def dormant_accounts(customers: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    """Active customers (``active_flg='Y'``) that have never placed an order."""
    columns = ["customer_id", "customer_name", "customer_segment", "contract_tier",
               "industry", "city", "credit_limit", "signup_date"]
    if customers is None or customers.empty:
        return pd.DataFrame(columns=columns)
    cust = customers.copy()
    cust["customer_id"] = cust["customer_id"].astype(str)
    if "active_flg" in cust.columns:
        cust = cust[cust["active_flg"].astype(str).str.upper() == "Y"]

    ordered_ids: set[str] = set()
    if orders is not None and not orders.empty and "customer_id" in orders.columns:
        ordered_ids = set(orders["customer_id"].astype(str).unique())

    dormant = cust[~cust["customer_id"].isin(ordered_ids)].copy()
    if "credit_limit" in dormant.columns:
        dormant["credit_limit"] = pd.to_numeric(dormant["credit_limit"], errors="coerce").fillna(0).round(2)
    available = [c for c in columns if c in dormant.columns]
    return dormant[available].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Section 1 - Executive KPIs
# ---------------------------------------------------------------------------
def executive_kpis(
    facts: pd.DataFrame,
    customers: pd.DataFrame,
    orders: pd.DataFrame,
    inventory: pd.DataFrame,
) -> dict:
    """Assemble the five headline KPIs for the page."""
    leaders = top_customers(facts, limit=1)
    growth = highest_growth_customer(facts)
    abnormal = detect_abnormal_orders(facts)
    impact = inventory_impact(facts, inventory)
    dormant = dormant_accounts(customers, orders)

    top = leaders.iloc[0] if not leaders.empty else None
    return {
        "top_customer_name": str(top["customer_name"]) if top is not None else "No orders yet",
        "top_customer_revenue": float(top["revenue"]) if top is not None else 0.0,
        "growth_customer_name": growth.get("customer_name", "Insufficient history"),
        "growth_customer_change": growth.get("change_pct", 0.0),
        "abnormal_orders": int(len(abnormal)),
        "stockout_risk_customers": int(len(impact)),
        "dormant_accounts": int(len(dormant)),
    }


# ---------------------------------------------------------------------------
# Section 8 - AI insights
# ---------------------------------------------------------------------------
def _money(value: float) -> str:
    return f"${float(value):,.0f}"


def generate_customer_insights(
    facts: pd.DataFrame,
    customers: pd.DataFrame,
    orders: pd.DataFrame,
    inventory: pd.DataFrame,
) -> list[str]:
    """Concise, data-grounded business insights from the computed frames."""
    if facts is None or facts.empty:
        return ["No order activity is available from Oracle to generate customer insights."]

    insights: list[str] = []
    leaders = top_customers(facts, limit=None)
    total_revenue = float(leaders["revenue"].sum())

    if not leaders.empty:
        top = leaders.iloc[0]
        insights.append(
            f"{top['customer_name']} is the top customer at {_money(top['revenue'])} "
            f"({top['contribution_pct']:.0f}% of total order revenue) across {int(top['orders'])} orders."
        )
        top3_share = leaders.head(3)["contribution_pct"].sum()
        insights.append(
            f"Revenue is concentrated: the top 3 customers account for {top3_share:.0f}% of all order revenue "
            f"from {int(facts['customer_id'].nunique())} active buyers."
        )

    products = top_products(facts, metric="revenue", limit=3)
    if not products.empty:
        names = ", ".join(products["product_name"].head(3).tolist())
        insights.append(f"Top revenue products: {names}.")

    abnormal = detect_abnormal_orders(facts)
    if not abnormal.empty:
        worst = abnormal.iloc[0]
        insights.append(
            f"{int(len(abnormal))} abnormal order line(s) detected. Largest: {worst['customer_name']} ordered "
            f"{int(worst['current_quantity'])} of {worst['product_name']} vs a baseline of "
            f"{worst['historical_avg']:.0f} (+{worst['deviation_pct']:.0f}%)."
        )
    else:
        insights.append("No abnormal order quantities detected against per-product baselines.")

    dormant = dormant_accounts(customers, orders)
    if not dormant.empty:
        names = ", ".join(dormant["customer_name"].head(3).tolist())
        insights.append(
            f"{int(len(dormant))} active account(s) have zero orders - a re-engagement opportunity "
            f"(e.g. {names})."
        )

    impact = inventory_impact(facts, inventory)
    if not impact.empty:
        leader = impact.iloc[0]
        insights.append(
            f"{leader['customer_name']} drives the most inventory pressure, ordering "
            f"{int(leader['at_risk_units'])} units across {int(leader['affected_products'])} at-risk product(s)."
        )

    trends = customer_demand_trends(facts)
    if not trends.empty:
        growing = int((trends["trend"] == "Growing").sum())
        declining = int((trends["trend"] == "Declining").sum())
        insights.append(
            f"Demand momentum (short order window): {growing} growing and {declining} declining customer(s)."
        )
    return insights
