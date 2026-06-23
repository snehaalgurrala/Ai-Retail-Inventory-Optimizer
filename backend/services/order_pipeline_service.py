"""Post-order pipeline: Customer Intelligence recalculation -> Recommendation agent.

Runs after a simulated order is committed (header + lines + inventory draw-down).
Stage 1 re-derives the Customer Intelligence abnormal-order analysis from the
fresh Oracle data, treating the just-inserted order as the latest order for its
products, and surfaces the per-line metrics (historical avg/max, deviation %,
inventory impact, composite risk score). Stage 2 re-runs the Recommendation
agent so reorder / transfer / supplier-risk recommendations reflect the new
stock levels. No other agents are triggered.

Everything here is pure orchestration over existing services — it owns no new
persistence. The abnormal-order "investigation entry" is the flagged line that
``customer_intelligence_service.detect_abnormal_orders`` now returns for the new
order; it shows on the Customer Intelligence page on its next render.
"""

from __future__ import annotations

import pandas as pd

from backend.db import repository
from backend.services import customer_intelligence_service as cis
from backend.services import inventory_scope


def _composite_risk_score(
    cur: int,
    deviation_pct: float,
    hist_high: int,
    inventory: int | None,
    hist_series: list[int],
    at_risk: bool,
) -> tuple[int, str, float | None]:
    """0-100 composite risk score + band + inventory-impact %.

    Mirrors the Customer Intelligence page's ``_risk_assessment`` so the simulator
    reports the same risk number the page shows: deviation (30) + increase above
    historical max (25) + inventory impact (20) + reorder pressure (15) + recent
    demand trend (10), normalised to 0-100.
    """
    cur = int(cur)
    deviation = float(deviation_pct)
    hist_high = max(1, int(hist_high or 0))
    hist = [int(q) for q in (hist_series or [])]

    dev_sub = min(1.0, max(0.0, deviation / 200.0))
    above_max_sub = min(1.0, (cur - hist_high) / hist_high) if cur > hist_high else 0.0
    if inventory is not None and inventory > 0:
        impact_share = cur / inventory
        impact_sub = min(1.0, impact_share)
        impact_pct: float | None = impact_share * 100.0
    else:
        impact_sub = 0.3
        impact_pct = None
    reorder_sub = 1.0 if at_risk else 0.0
    if len(hist) >= 2:
        mid = len(hist) // 2 or 1
        older, recent = hist[:mid], hist[mid:]
        older_avg = sum(older) / len(older)
        recent_avg = sum(recent) / len(recent)
        trend_sub = min(1.0, max(0.0, (recent_avg - older_avg) / older_avg)) if older_avg else 0.0
    else:
        trend_sub = 0.3

    score = (
        dev_sub * 30.0
        + above_max_sub * 25.0
        + impact_sub * 20.0
        + reorder_sub * 15.0
        + trend_sub * 10.0
    )
    score = int(round(max(0.0, min(100.0, score))))
    band = "Low" if score <= 30 else "Medium" if score <= 60 else "High" if score <= 85 else "Critical"
    return score, band, impact_pct


def recalculate_customer_intelligence(
    order_nbr: str, min_deviation_pct: float, branch_id=None
) -> dict:
    """Re-run abnormal-order detection over fresh Oracle data for the new order.

    Loads the current customer/order/inventory frames (which now include the
    just-placed order), rebuilds the fact table, and evaluates the latest order
    of each product against its prior-only baseline at the caller's deviation
    threshold. Returns the abnormal investigation entries attributable to
    ``order_nbr``, enriched with the customer / product / inventory context the
    Abnormal Order Investigation Alert email needs. Inventory figures follow the
    platform-wide inventory scope (network-wide totals by default); ``branch_id``
    is only consulted when ``INVENTORY_SCOPE=branch``.
    """
    orders = repository.load_orders(safe=True)
    order_lines = repository.load_order_lines(safe=True)
    customers = repository.load_customers(safe=True)
    products = repository.load_products(safe=True)
    inventory = repository.load_inventory(safe=True)

    facts = cis.prepare_customer_orders(order_lines, orders, customers, products)
    abnormal = cis.detect_abnormal_orders(facts, min_deviation_pct=float(min_deviation_pct))
    at_risk = cis.at_risk_products(inventory)
    # One inventory source for display AND scoring — same scope (network by
    # default) used by the Customer Intelligence page, simulator, and chatbot, so
    # the email's "Current Inventory" matches every other surface for this order.
    stock_map = inventory_scope.stock_by_product(inventory, branch_id=branch_id)
    reorder_map = inventory_scope.reorder_by_product(inventory, branch_id=branch_id)

    # detect_abnormal_orders does not carry product_id; recover it by matching the
    # flagged line back to the fact frame on (order_nbr, product_name) — the same
    # join the Customer Intelligence page uses to build its cards.
    name_to_pid: dict[tuple[str, str], str] = {}
    if facts is not None and not facts.empty:
        fl = facts.copy()
        nbr = fl["order_nbr"] if "order_nbr" in fl.columns else fl.get("order_id", "")
        fl["_nbr"] = nbr.astype(str)
        fl["product_id"] = fl["product_id"].astype(str)
        for _, r in fl[["_nbr", "product_name", "product_id"]].iterrows():
            name_to_pid.setdefault((r["_nbr"], str(r["product_name"])), r["product_id"])

    # Customer (segment/tier) + order date for this order; product category.
    seg_tier = ("", "")
    order_date = ""
    if facts is not None and not facts.empty:
        nbr_col = facts["order_nbr"] if "order_nbr" in facts.columns else facts.get("order_id", "")
        order_rows = facts[nbr_col.astype(str) == str(order_nbr)]
        if not order_rows.empty:
            first = order_rows.iloc[0]
            seg_tier = (str(first.get("customer_segment") or ""), str(first.get("contract_tier") or ""))
            if "order_date" in order_rows.columns and pd.notna(first.get("order_date")):
                order_date = pd.to_datetime(first["order_date"]).strftime("%B %d, %Y")
    category_by_pid: dict[str, str] = {}
    if products is not None and not products.empty and "category" in products.columns:
        pr = products.copy()
        pr["product_id"] = pr["product_id"].astype(str)
        category_by_pid = dict(zip(pr["product_id"], pr["category"].astype(str)))

    order_nbr = str(order_nbr)
    entries: list[dict] = []
    if abnormal is not None and not abnormal.empty and "order_nbr" in abnormal.columns:
        mine = abnormal[abnormal["order_nbr"].astype(str) == order_nbr]
        for _, row in mine.iterrows():
            product_name = str(row["product_name"])
            product_id = name_to_pid.get((order_nbr, product_name))
            current_inventory = stock_map.get(product_id) if product_id else None
            cur = int(row["current_quantity"])
            hist_max = int(row["historical_max"])
            score, band, impact_pct = _composite_risk_score(
                cur,
                float(row["deviation_pct"]),
                hist_max,
                current_inventory,
                row.get("hist_series") or [],
                bool(product_id and product_id in at_risk),
            )
            # Post-order on-hand (already reflects this order's decrement) and the
            # reorder point, both at the active inventory scope — consistent with
            # ``current_inventory`` above and the risk-score inventory impact.
            inventory_post = current_inventory
            reorder_point = reorder_map.get(product_id) if product_id else None
            entries.append({
                "customer_name": str(row["customer_name"]),
                "customer_segment": seg_tier[0],
                "contract_tier": seg_tier[1],
                "order_nbr": order_nbr,
                "order_date": order_date,
                "product_name": product_name,
                "product_id": product_id,
                "category": category_by_pid.get(product_id or "", ""),
                "current_quantity": cur,
                "historical_avg": float(row["historical_avg"]),
                "historical_max": hist_max,
                "deviation_pct": float(row["deviation_pct"]),
                "inventory_impact_pct": impact_pct,
                "current_inventory": current_inventory,
                # Inventory-scope figures for the email's math (pre = post + qty),
                # same scope as current_inventory so all email numbers agree.
                "inventory_post": inventory_post,
                "inventory_pre": (inventory_post + cur) if inventory_post is not None else None,
                "reorder_point": reorder_point,
                "historical_pattern": [int(q) for q in (row.get("order_series") or [])],
                "at_risk": bool(product_id and product_id in at_risk),
                "risk_score": score,
                "risk_band": band,
                "risk_level": str(row["risk_level"]),
            })

    return {
        "threshold": float(min_deviation_pct),
        "abnormal_total": int(0 if abnormal is None else len(abnormal)),
        "is_abnormal": bool(entries),
        "entries": entries,
    }


def dispatch_abnormal_order_alerts(ci_summary: dict) -> dict:
    """Send an Abnormal Order Investigation Alert for each qualifying entry.

    Fires the dedicated email whenever an entry's composite risk band is
    Critical / High / Medium (or the product is now at/below its reorder point —
    a critical inventory-risk signal). The Low Stock Alert email is untouched.
    """
    from backend.services import abnormal_order_email as aoe

    entries = (ci_summary or {}).get("entries") or []
    targets = [
        e for e in entries
        if aoe.should_alert(e.get("risk_band")) or e.get("at_risk")
    ]
    if not targets:
        return {"attempted": 0, "sent": 0, "message": "No Critical/High/Medium order to alert on."}

    sent = 0
    last_message = ""
    bands: list[str] = []
    for entry in targets:
        result = aoe.send_abnormal_order_alert_email(entry)
        last_message = result.get("message", "")
        if result.get("email_sent"):
            sent += 1
        bands.append(str(entry.get("risk_band")))

    return {
        "attempted": len(targets),
        "sent": sent,
        "bands": bands,
        "message": last_message,
    }


def _low_stock_reasoning(order_nbr: str, stock: int, reorder: int, suggested: int) -> str:
    """Business-actionable replenishment narrative for a low-stock email row.

    Frames the alert as an inventory-planning recommendation rather than a bare
    observation, and adapts to where the on-hand sits relative to the reorder
    point (out of stock / below threshold / exactly at threshold).
    """
    if stock <= 0:
        return (
            f"Inventory is fully depleted following order {order_nbr}. Place a replenishment "
            f"order of about {suggested} units immediately to restore cover and avoid backorders."
        )
    if stock < reorder:
        return (
            f"Inventory has fallen below the reorder point of {reorder} units after order "
            f"{order_nbr}, leaving {stock} units on hand. Initiate replenishment now "
            f"(suggested ~{suggested} units) to rebuild the safety buffer before further demand "
            "creates stockout risk."
        )
    return (
        f"Inventory has reached the reorder threshold of {reorder} units following order "
        f"{order_nbr}. Schedule replenishment (suggested ~{suggested} units) to restore a safety "
        "buffer before continued demand pushes the product into shortage."
    )


def dispatch_low_stock_alerts(
    order_nbr: str, product_ids, branch_id=None
) -> dict:
    """Send a Low Stock Alert email for order lines now at/below their reorder point.

    Built from the POST-order Oracle inventory (the order has already drawn stock
    down), this fires independently of the abnormal-order alert: any ordered
    product whose current on-hand is at or below its reorder point — under the
    active inventory scope (network-wide by default) — is listed. The standalone
    ``get_low_stock_items`` CSV pipeline is intentionally NOT used here because it
    reads ``data/raw`` rather than the freshly-decremented Oracle stock.
    """
    from backend.db import repository
    from backend.services import email_service, inventory_scope
    from backend.services.depletion_formatter import inventory_position_status
    from backend.services.low_stock_service import calculate_priority

    inventory = repository.load_inventory(safe=True)
    products = repository.load_products(safe=True)

    at_risk = inventory_scope.at_risk_product_ids(inventory, branch_id=branch_id)
    stock_map = inventory_scope.stock_by_product(inventory, branch_id=branch_id)
    reorder_map = inventory_scope.reorder_by_product(inventory, branch_id=branch_id)

    name_by_pid: dict[str, str] = {}
    category_by_pid: dict[str, str] = {}
    if products is not None and not products.empty and "product_id" in products.columns:
        pr = products.copy()
        pr["product_id"] = pr["product_id"].astype(str)
        name_by_pid = dict(zip(pr["product_id"], pr.get("product_name", pr["product_id"]).astype(str)))
        if "category" in pr.columns:
            category_by_pid = dict(zip(pr["product_id"], pr["category"].astype(str)))

    scope = inventory_scope.get_inventory_scope()
    if scope == "branch":
        store_id: object = branch_id if branch_id is not None else "branch"
        store_name = f"Branch {branch_id}" if branch_id is not None else "Branch"
    else:
        store_id = "network"
        store_name = "Network-wide inventory"

    order_nbr = str(order_nbr)
    rows: list[dict] = []
    for pid in dict.fromkeys(str(p) for p in (product_ids or [])):
        if pid not in at_risk:
            continue
        stock = int(stock_map.get(pid, 0))
        reorder = int(reorder_map.get(pid, 0))
        # Inventory-position classification (demand-agnostic): the order draw-down
        # is what triggered this alert, and there is no sales-velocity signal in
        # the Oracle path, so status/window come from stock vs. reorder point.
        position = inventory_position_status(stock, reorder)
        status, window = position if position else ("Reorder Required", "At reorder threshold")
        # Replenish back to a working target of 2x the reorder point — one reorder
        # point of cycle stock plus one of safety stock — so the suggested quantity
        # closes the gap from current on-hand up to that target.
        suggested_reorder = max(reorder * 2 - stock, 0)
        rows.append({
            "product_id": pid,
            "product_name": name_by_pid.get(pid, pid),
            "category": category_by_pid.get(pid, ""),
            "store_id": store_id,
            "store_name": store_name,
            "current_quantity": stock,
            "reorder_threshold": reorder,
            "shortage_quantity": max(reorder - stock, 0),
            "suggested_reorder_quantity": suggested_reorder,
            "recent_daily_sales_velocity": 0,
            "priority": calculate_priority(stock, reorder),
            "urgency_label": status,
            "depletion_window": window,
            "ai_alert_message": _low_stock_reasoning(order_nbr, stock, reorder, suggested_reorder),
        })

    if not rows:
        return {
            "attempted": 0,
            "sent": 0,
            "products": [],
            "message": "No ordered product fell to/below its reorder point.",
        }

    result = email_service.send_low_stock_alert_email(pd.DataFrame(rows))
    return {
        "attempted": len(rows),
        "sent": 1 if result.get("email_sent") else 0,
        "products": [r["product_name"] for r in rows],
        "message": result.get("message", ""),
    }


def refresh_chatbot_context() -> dict:
    """Drop the chatbot's warm Oracle snapshot so it sees the new order at once.

    The MCP chatbot reads Oracle through a short-lived context cache: an in-process
    copy (used by the in-process tool fallback) and, when the stdio engine is
    active, a separate copy inside the MCP server subprocess. We clear the
    in-process one and tear down the stdio session singleton so the next chatbot
    query spawns a fresh server that reloads Oracle — making the just-placed order
    answerable immediately rather than after the TTL elapses.
    """
    cleared = []
    try:
        from backend.mcp import context as mcp_ctx

        mcp_ctx.clear_context()
        cleared.append("in_process_context")
    except Exception:
        pass
    try:
        from backend.mcp import client as mcp_client

        mcp_client.reset_client()
        cleared.append("stdio_session")
    except Exception:
        pass
    return {"refreshed": cleared}


def recalculate_recommendations() -> dict:
    """Trigger the Recommendation agent and summarise the regenerated output.

    Re-runs the LangGraph orchestrator (``run_all_agents``) against the fresh
    inventory, which rewrites ``recommendations.csv`` (reorder / transfer /
    supplier-risk). Returns counts by recommendation type for confirmation.
    """
    from backend.agents.orchestrator_agent import run_all_agents

    df = run_all_agents(save_output=True)
    counts: dict[str, int] = {}
    if df is not None and not df.empty and "recommendation_type" in df.columns:
        counts = {str(k): int(v) for k, v in df["recommendation_type"].value_counts().items()}

    def _count(*types: str) -> int:
        return int(sum(counts.get(t, 0) for t in types))

    return {
        "total": int(0 if df is None else len(df)),
        "reorder": _count("reorder"),
        "transfer": _count("transfer", "stock_transfer"),
        "supplier_risk": _count("supplier_risk_alert"),
        "by_type": counts,
    }
