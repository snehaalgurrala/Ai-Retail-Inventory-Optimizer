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
#
# The deviation threshold is the primary sensitivity knob and is now caller
# supplied (the Customer Intelligence page exposes it as a slider, the chatbot
# can pass it per query). ``DEFAULT_ABNORMAL_DEVIATION_PCT`` is only the fallback
# when no value is provided.
DEFAULT_ABNORMAL_DEVIATION_PCT = 50.0   # at least +50% over the product baseline
_ABNORMAL_MIN_ABS_GAP = 5.0             # and at least 5 units above the mean
# High risk is graded relative to the active threshold (3x the configured
# deviation), so risk severity tracks the chosen sensitivity instead of a fixed
# cut-off. At the 50% default this reproduces the previous +150% High-risk line.
_HIGH_RISK_MULTIPLIER = 3.0

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
def prior_order_baseline(
    df: pd.DataFrame,
    value_col: str,
    group_col: str = "product_id",
    sort_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Attach per-row 'prior orders only' baseline statistics.

    For every row the baseline is computed from rows in the same ``group_col`` that
    occurred strictly BEFORE it in ``sort_cols`` order (chronological by default),
    so an order can never contaminate the baseline used to judge it. For the
    product order history ``6 -> 6 -> 160`` the row carrying 160 sees a prior
    baseline of mean 6, max 6, min 6 — the 160 itself is excluded.

    Adds the columns ``hist_count`` (number of prior orders), ``hist_mean``,
    ``hist_max``, ``hist_min`` and ``hist_std`` (sample std, NaN with <2 priors).
    The first order of each product has ``hist_count == 0`` and NaN statistics.
    Row order of the returned frame matches the input (a fresh RangeIndex).
    """
    work = df.reset_index(drop=True)
    if work.empty:
        for col in ("hist_count", "hist_mean", "hist_max", "hist_min", "hist_std"):
            work[col] = pd.Series(dtype="float64")
        return work

    sort_cols = [c for c in (sort_cols or []) if c in work.columns]
    work["_orig_pos"] = np.arange(len(work))
    # mergesort = stable, so equal sort keys keep their original relative order.
    work = work.sort_values([group_col, *sort_cols], kind="mergesort")

    grp = work.groupby(group_col, sort=False)[value_col]
    work["hist_count"] = grp.cumcount()                       # strictly-prior count
    prior_sum = grp.cumsum() - work[value_col]                # sum excluding current
    work["hist_mean"] = prior_sum / work["hist_count"].replace(0, np.nan)
    # Cumulative max/min then shift one row within the group → excludes current.
    work["hist_max"] = grp.cummax().groupby(work[group_col]).shift(1)
    work["hist_min"] = grp.cummin().groupby(work[group_col]).shift(1)
    # Expanding sample std then shift → std of the prior orders only.
    exp_std = grp.expanding().std().reset_index(level=0, drop=True)
    work["hist_std"] = exp_std.groupby(work[group_col]).shift(1)

    work = work.sort_values("_orig_pos").drop(columns="_orig_pos").reset_index(drop=True)
    return work


def detect_abnormal_orders(
    facts: pd.DataFrame,
    limit: int | None = None,
    min_deviation_pct: float = DEFAULT_ABNORMAL_DEVIATION_PCT,
) -> pd.DataFrame:
    """Flag the MOST RECENT order of each product when it deviates from baseline.

    Only the latest order per product is evaluated — the question this answers is
    "given everything we knew before today, is the newest order abnormal enough to
    require management attention?". Its baseline is computed from that product's
    orders that occurred strictly BEFORE it (see :func:`prior_order_baseline`), so
    the order being judged never contaminates its own baseline and earlier spikes
    stay part of the history rather than being flagged in their own right. The
    latest order is abnormal when its quantity is at least ``min_deviation_pct``
    above the prior-only mean and at least ``_ABNORMAL_MIN_ABS_GAP`` units above it.
    A product with only a single order has no prior history and is never flagged.
    The anomaly is attributed to the customer who placed the order.

    ``min_deviation_pct`` is the user-configurable sensitivity (the page slider /
    the chatbot threshold); High risk is graded at ``_HIGH_RISK_MULTIPLIER`` times
    that value so severity scales with the chosen sensitivity.
    """
    columns = ["customer_name", "product_name", "order_nbr", "order_date",
               "historical_avg", "historical_max", "historical_min", "hist_count",
               "hist_series", "order_series", "current_index",
               "current_quantity", "deviation_pct", "risk_level"]
    if facts is None or facts.empty:
        return pd.DataFrame(columns=columns)

    min_deviation_pct = float(min_deviation_pct)
    high_risk_threshold = min_deviation_pct * _HIGH_RISK_MULTIPLIER

    # Chronological tiebreak so "prior" is deterministic for same-day orders.
    sort_cols = [c for c in ["order_date", "order_nbr"] if c in facts.columns]
    work = prior_order_baseline(
        facts, value_col="quantity", group_col="product_id", sort_cols=sort_cols
    )

    # Baseline = prior orders only; the current line is excluded from its own mean.
    work["product_mean"] = work["hist_mean"]
    mean_safe = work["product_mean"].replace(0, np.nan)
    work["deviation_pct"] = (work["quantity"] - work["product_mean"]) / mean_safe * 100.0
    work["abs_gap"] = work["quantity"] - work["product_mean"]

    # Only the MOST RECENT order of each product is evaluated. The latest order is
    # judged against the COMPLETE history of prior orders; older spikes are part of
    # that baseline and are never flagged on their own. A flag therefore always
    # means "given everything we knew before, the newest order is abnormal" — and
    # because the latest order's ``hist_count`` equals its position at the end of
    # the full sequence, it always renders at the far right of the trend chart.
    latest_idx = (
        work.sort_values(["product_id", *sort_cols], kind="mergesort")
        .groupby("product_id", sort=False)
        .tail(1)
        .index
    )
    work["is_latest"] = work.index.isin(latest_idx)

    # A line can only be judged once it has at least one genuine prior order.
    flagged = work[
        work["is_latest"]
        & (work["hist_count"] >= 1)
        & (work["deviation_pct"] >= min_deviation_pct)
        & (work["abs_gap"] >= _ABNORMAL_MIN_ABS_GAP)
    ].copy()
    if flagged.empty:
        return pd.DataFrame(columns=columns)

    # Prior-only quantity history per product (same chronological order as the
    # baseline), so a flagged line at position ``hist_count`` takes the first
    # ``hist_count`` quantities as its genuine history (excludes the current line).
    ser = facts.copy()
    ser["product_id"] = ser["product_id"].astype(str)
    ser = ser.sort_values(["product_id", *sort_cols], kind="mergesort")
    ser["_q"] = pd.to_numeric(ser["quantity"], errors="coerce").fillna(0).round().astype(int)
    full_by_product = ser.groupby("product_id", sort=False)["_q"].apply(list).to_dict()

    flagged["risk_level"] = np.where(
        flagged["deviation_pct"] >= high_risk_threshold, "High", "Medium"
    )
    flagged["historical_avg"] = flagged["hist_mean"].round(1)
    flagged["historical_max"] = flagged["hist_max"].round().astype(int)
    flagged["historical_min"] = flagged["hist_min"].round().astype(int)
    flagged["hist_count"] = flagged["hist_count"].astype(int)
    flagged["current_quantity"] = flagged["quantity"].round().astype(int)
    flagged["deviation_pct"] = flagged["deviation_pct"].round(1)
    # ``hist_series`` = prior orders only (for narrative/trend). ``order_series`` =
    # the product's COMPLETE chronological order sequence (for the chart), with
    # ``current_index`` marking where the evaluated order sits inside it. Because
    # ``hist_count`` is the number of orders strictly before this line, it is also
    # the evaluated order's 0-based position in the full sequence.
    flagged["hist_series"] = [
        full_by_product.get(str(pid), [])[: int(hc)]
        for pid, hc in zip(flagged["product_id"], flagged["hist_count"])
    ]
    flagged["order_series"] = [
        full_by_product.get(str(pid), []) for pid in flagged["product_id"]
    ]
    flagged["current_index"] = flagged["hist_count"].astype(int)
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
    """Product ids at or below their reorder point under the active inventory scope.

    Delegates to the shared inventory-scope helper so "at risk" is judged on the
    same stock figure shown everywhere else — network totals (stock summed across
    branches vs the summed reorder point) by default, or a single branch when
    ``INVENTORY_SCOPE=branch``.
    """
    from backend.services import inventory_scope

    return inventory_scope.at_risk_product_ids(inventory)


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
    min_deviation_pct: float = DEFAULT_ABNORMAL_DEVIATION_PCT,
) -> dict:
    """Assemble the five headline KPIs for the page."""
    leaders = top_customers(facts, limit=1)
    growth = highest_growth_customer(facts)
    abnormal = detect_abnormal_orders(facts, min_deviation_pct=min_deviation_pct)
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
    min_deviation_pct: float = DEFAULT_ABNORMAL_DEVIATION_PCT,
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

    abnormal = detect_abnormal_orders(facts, min_deviation_pct=min_deviation_pct)
    if not abnormal.empty:
        worst = abnormal.iloc[0]
        insights.append(
            f"{int(len(abnormal))} abnormal order line(s) detected at the configured deviation threshold "
            f"of {min_deviation_pct:.0f}%. Largest: {worst['customer_name']} ordered "
            f"{int(worst['current_quantity'])} of {worst['product_name']} vs a baseline of "
            f"{worst['historical_avg']:.0f} (+{worst['deviation_pct']:.0f}%)."
        )
    else:
        insights.append(
            f"No abnormal order quantities detected against per-product baselines at the configured "
            f"deviation threshold of {min_deviation_pct:.0f}%."
        )

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
