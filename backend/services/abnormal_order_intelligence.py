"""Shared abnormal-order intelligence (card building, risk assessment, narrative).

This is the SINGLE SOURCE OF TRUTH for everything that sits on top of the raw
anomaly detection in :mod:`customer_intelligence_service`. The Customer
Intelligence page, the Abnormal Order Investigation panels, and the dashboard
Abnormal Order Intelligence Report all import from here so the risk bands, the
composite priority score, the inventory-impact maths and the AI investigation
narrative are identical everywhere.

Nothing in this module depends on Streamlit — it is pure functions over the
``detect_abnormal_orders`` output (plus the fact frame and live inventory), so it
can be used equally from a page, an email builder, or a background job.

Detection itself (which order lines are abnormal) is NOT here — it lives in
``customer_intelligence_service.detect_abnormal_orders``. This module only
*interprets* already-flagged lines.
"""

from __future__ import annotations

import math

import pandas as pd

from backend.services import inventory_scope


# Risk presentation (shared by page cards, alert feeds, and the email report).
RISK_ASSESSMENT_STYLE = {
    "Low":      {"color": "#1E8E3E", "bg": "#E6F4EA", "icon": "🟢"},
    "Medium":   {"color": "#A86E00", "bg": "#FEF9C3", "icon": "🟡"},
    "High":     {"color": "#C76A12", "bg": "#FFEDD5", "icon": "🟠"},
    "Critical": {"color": "#B42318", "bg": "#FEE2E2", "icon": "🔴"},
}

# Executive ordering — Critical first, then High, Medium, Low.
_RISK_PRIORITY = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}

# Business-facing display labels for the four internal severity bands. The band
# keys themselves (Low/Medium/High/Critical) are NEVER changed — all logic,
# sorting, styling and caching rely on them. Only the words shown to users change:
# positive, demand-focused language that frames each order as a customer demand
# signal rather than a problem to be flagged.
RISK_DISPLAY_LABEL = {
    "Low": "Normal Demand Activity",
    "Medium": "Moderate Demand Activity",
    "High": "High Demand Activity",
    "Critical": "Significant Opportunity",
}

# Possible business explanations surfaced alongside every abnormal order.
BUSINESS_REASONS = [
    "New customer contract",
    "Bulk procurement cycle",
    "Inventory buffering",
    "Project-driven demand",
]


def _money(value: float) -> str:
    return f"${float(value):,.0f}"


# ---------------------------------------------------------------------------
# Card building
# ---------------------------------------------------------------------------
def build_abnormal_cards(
    abnormal_df: pd.DataFrame,
    facts: pd.DataFrame,
    at_risk_ids: set[str],
    inventory: pd.DataFrame | None = None,
    threshold_pct: float = 50.0,
) -> list[dict]:
    """Build one AI card per abnormal order line, sorted by severity (deviation).

    Every flagged line becomes its own card — not aggregated to one-per-customer —
    so all abnormal orders are surfaced. Pure view-layer enrichment: per-line
    revenue comes from ``facts`` (the authoritative line_total), the on-hand stock
    and the product's prior order sequence are pulled in so the AI narrative can
    cite real numbers, and risk_score is scaled relative to the largest deviation
    in the set. No service logic changes.
    """
    if abnormal_df is None or abnormal_df.empty:
        return []

    # Recover per-line revenue + product_id (plus segment/category) by matching the
    # flagged lines back to the fact frame on (order_nbr, product_name).
    flook = facts.copy()
    if "order_nbr" not in flook.columns:
        flook["order_nbr"] = flook.get("order_id", "")
    flook["order_nbr"] = flook["order_nbr"].astype(str)
    agg_kwargs = dict(product_id=("product_id", "first"), line_revenue=("revenue", "sum"))
    if "customer_segment" in flook.columns:
        agg_kwargs["customer_segment"] = ("customer_segment", "first")
    if "category" in flook.columns:
        agg_kwargs["category"] = ("category", "first")
    lookup = flook.groupby(["order_nbr", "product_name"], as_index=False).agg(**agg_kwargs)

    ab = abnormal_df.copy()
    ab["order_nbr"] = ab["order_nbr"].astype(str)
    ab = ab.merge(lookup, on=["order_nbr", "product_name"], how="left")
    ab["line_revenue"] = pd.to_numeric(ab["line_revenue"], errors="coerce").fillna(0.0)
    ab["product_id"] = ab["product_id"].astype(str)

    # On-hand stock per product via the shared inventory-scope helper so the
    # card's "Current Inventory" matches the abnormal-order email, simulator, and
    # chatbot exactly (network-wide totals by default).
    stock_by_product = inventory_scope.stock_by_product(inventory)

    # How many abnormal lines each customer has (drives "recurring behaviour" copy).
    cust_counts = ab.groupby("customer_name").size().to_dict()

    global_max_dev = float(ab["deviation_pct"].max()) or 1.0

    cards: list[dict] = []
    for position, (_, row) in enumerate(ab.iterrows()):
        deviation = float(row["deviation_pct"])
        name = str(row["customer_name"])
        product_id = str(row["product_id"])
        current_quantity = int(row["current_quantity"])

        # Prior-only history, baseline mean/min/max all come straight from the
        # service (computed from orders BEFORE this one) — the current order is
        # never part of its own baseline. We never reconstruct it here. Reads are
        # defensive (``.get`` + NaN guards) so a stale/older service frame degrades
        # to the prior-only series rather than raising KeyError.
        history = [int(q) for q in (row.get("hist_series") or [])]
        avg_fallback = max(1, math.ceil(float(row["historical_avg"])))
        _min = row.get("historical_min")
        _max = row.get("historical_max")
        hist_low = int(_min) if pd.notna(_min) else (min(history) if history else avg_fallback)
        hist_high = int(_max) if pd.notna(_max) else (max(history) if history else avg_fallback)

        cards.append({
            "uid": f"abn{position}",
            "threshold_pct": float(threshold_pct),
            "customer_name": name,
            "customer_segment": str(row.get("customer_segment") or "") or "—",
            "product_name": str(row["product_name"]),
            "category": str(row.get("category") or "") or "—",
            "product_id": product_id,
            "order_nbr": str(row["order_nbr"]),
            "order_date": str(row.get("order_date") or ""),
            "risk_level": str(row["risk_level"]),
            "risk_score": int(round(min(100.0, deviation / global_max_dev * 100.0))),
            "deviation_pct": deviation,
            "current_quantity": current_quantity,
            "historical_avg": float(row["historical_avg"]),
            "revenue_impact": float(row["line_revenue"]),
            "at_risk": product_id in at_risk_ids,
            "customer_abnormal_lines": int(cust_counts.get(name, 1)),
            "current_inventory": stock_by_product.get(product_id),  # None if unknown
            # Prior-only sequence (current line excluded) drives the narrative/trend.
            "hist_series": history,
            # Complete chronological order sequence for the product + the position
            # of the evaluated order inside it — the chart plots every order as its
            # own bar and highlights the evaluated one in place (never an average).
            "order_series": [int(q) for q in (row.get("order_series") or [])],
            "current_index": int(row["current_index"]) if pd.notna(row.get("current_index")) else len(history),
            "hist_low": hist_low,
            "hist_high": hist_high,
        })
    cards.sort(key=lambda c: (c["risk_score"], c["deviation_pct"]), reverse=True)
    return cards


# ---------------------------------------------------------------------------
# Risk Assessment (composite, business-facing severity)
# ---------------------------------------------------------------------------
def _ensure_assessment(card: dict) -> dict:
    """Compute (once, then cache) the composite risk assessment on a card."""
    if "_ra" not in card:
        card["_ra"] = _risk_assessment(card)
    return card["_ra"]


def _risk_band(score: int) -> str:
    """Map a 0-100 composite score onto the four risk bands."""
    if score <= 30:
        return "Low"
    if score <= 60:
        return "Medium"
    if score <= 85:
        return "High"
    return "Critical"


def _stockout_risk_label(card: dict, inv: int | None, cur: int) -> str:
    """Plain-language stockout outlook for this single order."""
    if inv is None:
        return "Unknown — live stock unavailable"
    if inv < cur:
        return "High — order exceeds on-hand stock"
    if card.get("at_risk"):
        return "Elevated — product at/below reorder point"
    coverage = (inv / cur) if cur else 0.0
    if coverage < 2:
        return "Moderate — limited buffer remaining"
    return "Low — current stock can absorb the order"


def _risk_assessment(card: dict) -> dict:
    """Composite 0-100 risk score grounded in this line's real numbers.

    Five weighted components, each normalised to 0-1, then blended:
      • Deviation from average      (30) — how far above the baseline this order is
      • Increase above historical max (25) — how far it breaks the prior peak
      • Inventory impact            (20) — share of on-hand stock it consumes
      • Reorder-point pressure      (15) — product already at/below reorder point
      • Recent demand trend         (10) — is demand accelerating into this order
    """
    cur = int(card["current_quantity"])
    deviation = float(card["deviation_pct"])
    hist_high = max(1, int(card.get("hist_high") or 0))
    inv = card.get("current_inventory")
    hist = [int(q) for q in (card.get("hist_series") or [])]

    # 1) Deviation from average — 200%+ over baseline saturates the component.
    dev_sub = min(1.0, max(0.0, deviation / 200.0))

    # 2) Increase above the historical maximum — doubling the prior peak saturates.
    above_max_sub = min(1.0, (cur - hist_high) / hist_high) if cur > hist_high else 0.0

    # 3) Inventory impact — share of on-hand stock this single order consumes.
    if inv is not None and inv > 0:
        impact_share = cur / inv
        impact_sub = min(1.0, impact_share)
        impact_pct: float | None = impact_share * 100.0
    else:
        impact_sub = 0.3  # unknown stock → moderate; can't be ruled out
        impact_pct = None

    # 4) Reorder-point pressure — already flagged at/below reorder point upstream.
    reorder_sub = 1.0 if card.get("at_risk") else 0.0

    # 5) Recent demand trend — compare the back half of history to the front half.
    if len(hist) >= 2:
        mid = len(hist) // 2 or 1
        older = hist[:mid]
        recent = hist[mid:]
        older_avg = sum(older) / len(older)
        recent_avg = sum(recent) / len(recent)
        trend_sub = min(1.0, max(0.0, (recent_avg - older_avg) / older_avg)) if older_avg else 0.0
    else:
        trend_sub = 0.3  # too little history to read the trend confidently

    score = (
        dev_sub * 30.0
        + above_max_sub * 25.0
        + impact_sub * 20.0
        + reorder_sub * 15.0
        + trend_sub * 10.0
    )
    score = int(round(max(0.0, min(100.0, score))))
    return {
        "score": score,
        "band": _risk_band(score),
        "impact_pct": impact_pct,
        "stockout_risk": _stockout_risk_label(card, inv, cur),
    }


# ---------------------------------------------------------------------------
# AI investigation narrative (interprets the real numbers in business language)
# ---------------------------------------------------------------------------
def _executive_summary(card: dict) -> str:
    """Plain-English briefing: what happened, why it's unusual, how different it is,
    and whether action is required — no scores, formulas or percentages."""
    name = card["customer_name"]
    prod = card["product_name"]
    cur = int(card["current_quantity"])
    avg = math.ceil(card["historical_avg"])
    hi = int(card["hist_high"])
    dev = float(card["deviation_pct"])
    band = _ensure_assessment(card)["band"]
    inv = card.get("current_inventory")

    # How different — qualifier scales with the size of the jump.
    if dev >= 400:
        qualifier = "dramatically larger than"
    elif dev >= 150:
        qualifier = "significantly larger than"
    elif dev >= 80:
        qualifier = "noticeably larger than"
    else:
        qualifier = "larger than"

    s1 = f"{name} typically orders around {avg:,} units of {prod}."
    s2 = f"The latest order was for {cur:,} units, making it {qualifier} previous orders"
    s2 += " — the largest order recorded for this product." if cur > hi else "."

    # Whether action is required — phrased by risk band, with the driving cause.
    if card.get("at_risk"):
        cause = "Because the product is already near its reorder threshold, "
    elif inv is not None and inv < cur:
        cause = "Because this single order is larger than the stock currently on hand, "
    else:
        cause = ""
    verdict = {
        "Critical": "this order represents a significant increase in demand and is a strong "
                    "customer demand signal worth prioritising for inventory planning.",
        "High": "this order reflects elevated demand and is worth reviewing as an emerging "
                "demand opportunity.",
        "Medium": "this order shows higher-than-typical demand and is worth reviewing.",
        "Low": "this order is only slightly above typical demand, so routine monitoring is sufficient.",
    }[band]
    s3 = cause + verdict if cause else verdict[0].upper() + verdict[1:]
    return f"{s1} {s2} {s3}"


def _demand_trend(series: list[int]) -> str:
    """Read the product's recent demand direction from its order-quantity history.

    Compares the back half of the (anomaly-excluded) history to the front half and
    returns 'rising', 'declining' or 'stable'. Mirrors the trend component used by
    the composite risk score, so the narrative and the score stay consistent.
    """
    series = [int(q) for q in series if q is not None]
    if len(series) < 4:
        return "stable"
    mid = len(series) // 2
    older, recent = series[:mid], series[mid:]
    older_avg = sum(older) / len(older)
    if older_avg <= 0:
        return "stable"
    change = (sum(recent) / len(recent) - older_avg) / older_avg
    if change > 0.15:
        return "rising"
    if change < -0.15:
        return "declining"
    return "stable"


def _what_happened(card: dict) -> list[str]:
    """What Happened? — plain-English account of the anomaly, all real numbers."""
    name = card["customer_name"]
    prod = card["product_name"]
    cur = int(card["current_quantity"])
    avg = math.ceil(card["historical_avg"])
    hi = int(card["hist_high"])
    dev = float(card["deviation_pct"])

    paras = [f"{name} placed an order for {cur:,} units of {prod}."]
    paras.append(
        f"Historically, the largest order recorded for this product was {hi:,} units, while the "
        f"average order quantity has been approximately {avg:,} units."
    )
    diff_avg = cur - avg
    s = (
        f"This means the current order is {diff_avg:,} units higher than the historical average "
        f"and approximately {dev:.0f}% above normal demand patterns."
    )
    if cur > hi:
        s += f" It also sits {cur - hi:,} units above the previous maximum on record."
    elif cur < hi:
        s += f" It remains {hi - cur:,} units below the previous maximum of {hi:,} units."
    else:
        s += " It matches the previous maximum on record."
    paras.append(s)
    paras.append(
        "As a result, this order has been highlighted as a significant increase in demand "
        "compared to historical purchasing patterns and may indicate a new demand opportunity "
        "or procurement cycle."
    )
    return paras


def _inventory_impact_narrative(card: dict) -> list[str]:
    """Inventory Impact — what this order does to on-hand stock, in plain language."""
    cur = int(card["current_quantity"])
    inv = card.get("current_inventory")

    if inv is None:
        return [
            "Live inventory for this product is currently unavailable, so the exact stock impact "
            f"cannot be quantified. Given the order size of {cur:,} units, replenishment readiness "
            "should be reviewed before fulfilment."
        ]

    paras = [f"The warehouse currently has {inv:,} units available in inventory."]
    if inv > 0 and cur <= inv:
        share = cur / inv * 100.0
        remaining = inv - cur
        paras.append(
            f"This order would consume approximately {share:.0f}% of available stock, leaving only "
            f"{remaining:,} units remaining after fulfilment."
        )
    elif inv > 0:
        shortfall = cur - inv
        paras.append(
            f"This single order is larger than the entire stock on hand — it would consume all "
            f"{inv:,} available units and still fall short by {shortfall:,} units."
        )
    else:
        paras.append("There is effectively no stock on hand to fulfil this order.")

    if card.get("at_risk"):
        paras.append(
            "The product is already operating at or below its reorder threshold, increasing the "
            "likelihood of inventory pressure if additional orders arrive."
        )
    else:
        paras.append(
            "The product currently sits above its reorder threshold, but repeat orders of this "
            "size would draw stock down quickly."
        )
    return paras


def _product_demand_context(card: dict) -> list[str]:
    """Product Demand Context — the product's normal behaviour vs this order."""
    prod = card["product_name"]
    cur = int(card["current_quantity"])
    avg = math.ceil(card["historical_avg"])
    lo, hi = int(card["hist_low"]), int(card["hist_high"])

    trend = _demand_trend(card.get("hist_series") or [])
    trend_word = {
        "rising": "a gradually rising",
        "declining": "a softening",
        "stable": "a stable",
    }[trend]
    typical = (
        f"orders typically ranging between {lo:,} and {hi:,} units"
        if lo != hi else f"orders typically near {avg:,} units"
    )
    return [
        f"Across all customers, this product normally experiences {trend_word} demand pattern, "
        f"with {typical}.",
        f"The latest order of {cur:,} units is significantly larger than the quantities typically "
        f"observed and stands out from historical purchasing behaviour for {prod}.",
    ]


def _customer_behaviour_assessment(card: dict) -> tuple[list[str], list[str]]:
    """Customer Behaviour Assessment — hedged read on whether this is unusual.

    Returns (paragraphs, hypotheses). The hypotheses are deliberately tentative
    ("may indicate", "could suggest", "appears consistent with") so the briefing
    never claims certainty about the customer's intent.
    """
    name = card["customer_name"]
    cur = int(card["current_quantity"])
    hi = int(card["hist_high"])
    lines = int(card["customer_abnormal_lines"])

    if cur > hi:
        paras = [
            f"An order of this scale has not been recorded for this product before, so it is "
            f"unclear whether {name} routinely purchases at this level — it stands out as a "
            "notably stronger demand signal than everything seen to date."
        ]
    else:
        paras = [
            f"{name} has occasionally ordered at higher volumes, but the current order still sits "
            "well above its typical purchasing levels."
        ]
    if lines > 1:
        paras.append(
            f"{name} now has {lines} high-demand order lines, which could suggest a broader shift in "
            "their purchasing pattern rather than a one-off event."
        )

    hypotheses = [
        "A new customer project may be driving the increase",
        "A planned bulk procurement cycle could explain the volume",
        "The order appears consistent with inventory stockpiling",
        "It may indicate expansion activity on the customer's side",
        "It could suggest a temporary demand surge",
    ]
    return paras, hypotheses


def _deep_actions(card: dict) -> list[str]:
    """Contextual, customer/product-specific recommended actions."""
    name = card["customer_name"]
    prod = card["product_name"]
    inv = card.get("current_inventory")

    actions = [
        f"Confirm whether this order is associated with a new project or contract for {name}",
        f"Monitor follow-up orders from {name} over the next 7–14 days",
    ]
    if inv is not None and inv < card["current_quantity"]:
        actions.append(f"Consider transferring {prod} inventory from lower-demand locations to cover the shortfall")
    elif card["at_risk"]:
        actions.append(f"Expedite replenishment for {prod}, which is at or below its reorder point")
    else:
        actions.append(f"Consider transferring {prod} inventory from lower-demand locations if demand persists")
    actions.append(f"Increase procurement planning for {prod} if demand continues at this level")
    if card["customer_abnormal_lines"] > 1:
        actions.append(
            f"Review {name}'s broader ordering pattern — {card['customer_abnormal_lines']} of their "
            "order lines show elevated demand"
        )
    return actions


def business_impact(card: dict) -> list[str]:
    """Business-language impact bullets for a single abnormal order.

    Grounded in the same numbers as the rest of the briefing, but phrased for a
    management audience rather than an analyst.
    """
    cur = int(card["current_quantity"])
    inv = card.get("current_inventory")
    ra = _ensure_assessment(card)

    items: list[str] = []
    if inv is not None and inv > 0:
        share = cur / inv * 100.0
        items.append(
            f"Inventory pressure: this single order consumes about {share:.0f}% of current network "
            "stock, tightening availability for other customers."
        )
    else:
        items.append(
            "Inventory pressure: this order represents a significant draw on current network stock."
        )
    items.append(f"Potential stockout risk: {ra['stockout_risk']}.")
    items.append(
        "Procurement impact: replenishment volume and timing should be reviewed so the spike does "
        "not deplete buffer stock."
    )
    if int(card.get("customer_abnormal_lines", 1)) > 1:
        items.append(
            f"Customer concentration: {card['customer_name']} has multiple high-demand order "
            "lines, increasing the business's dependence on a single account."
        )
    else:
        items.append(
            "Customer concentration: a single customer is driving an outsized share of demand "
            "for this product."
        )
    items.append(
        f"Revenue opportunity: the order carries {_money(card['revenue_impact'])} in revenue that "
        "should be secured and serviced reliably."
    )
    return items
