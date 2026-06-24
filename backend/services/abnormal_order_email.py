"""Abnormal Order Investigation Alert email.

A NEW, dedicated executive email template — separate from (and leaving entirely
unchanged) the Low Stock Alert email in ``email_service``. It is triggered when a
customer order is classified Critical / High / Medium risk by the Customer
Intelligence recalculation, and is intentionally more narrative and
business-focused than the low-stock report: it explains *why* the order is
unusual rather than dumping raw metrics.

Bunzl theme (blue + green + white) is applied with fully inline CSS so it renders
consistently across email clients, with a single-column, mobile-friendly layout.
"""

import smtplib
from concurrent.futures import Future
from datetime import datetime
from email.message import EmailMessage
from html import escape

from backend.services.abnormal_order_intelligence import RISK_DISPLAY_LABEL
from backend.services.email_service import EMAIL_EXECUTOR, THEME, _smtp_settings


SUBJECT = "📈 Customer Demand Intelligence Alert"

# Risk badge palette — Bunzl theme aligned (Critical red, High orange, Medium
# amber, Low green).
_BADGE = {
    "Critical": ("#B42318", "#FFFFFF"),
    "High": ("#C76A12", "#FFFFFF"),
    "Medium": ("#B8860B", "#FFFFFF"),
    "Low": ("#6CB33F", "#0A1F33"),
}

_TRIGGER_BANDS = {"Critical", "High", "Medium"}


# ---------------------------------------------------------------------------
# Derived figures + narrative (all "copy" lives with the template)
# ---------------------------------------------------------------------------
def _inventory_math(inv: dict) -> dict:
    """Derive the displayed inventory figures from pre/post branch stock."""
    pre = int(inv.get("inventory_pre") or 0)
    post = int(inv.get("inventory_post") or 0)
    qty = int(inv.get("current_quantity") or 0)
    reorder_point = inv.get("reorder_point")
    impact_pct = (qty / pre * 100.0) if pre > 0 else 100.0
    coverage = (pre / qty) if qty > 0 else 0.0
    if reorder_point is not None and post <= int(reorder_point):
        stockout = "Elevated"
    elif coverage and coverage < 1.5:
        stockout = "Elevated"
    elif coverage and coverage < 3.0:
        stockout = "Moderate"
    else:
        stockout = "Low"
    return {
        "pre": pre,
        "post": post,
        "qty": qty,
        "impact_pct": impact_pct,
        "coverage": coverage,
        "stockout": stockout,
    }


def _executive_summary(inv: dict) -> str:
    return (
        f"{inv['customer_name']} has placed an order for "
        f"{int(inv['current_quantity']):,} units of {inv['product_name']}. "
        f"This order is {inv['deviation_pct']:+.0f}% above the historical average demand "
        "and reflects a strong increase in purchasing activity. It may indicate an "
        "emerging demand opportunity and is worth reviewing for inventory planning."
    )


def _explanation(inv: dict) -> str:
    avg = max(1, round(float(inv["historical_avg"])))
    cur = int(inv["current_quantity"])
    hi = int(inv["historical_max"])
    text = (
        f"This customer normally orders around {avg:,} units of this product. "
        f"The latest order for {cur:,} units is substantially higher than all "
        "previous orders"
    )
    text += " and is the largest order recorded for this product." if cur > hi else "."
    return text


def _risk_reasoning(inv: dict, math: dict) -> list[str]:
    cur = int(inv["current_quantity"])
    hi = int(inv["historical_max"])
    reasons = [f"Above historical average (+{inv['deviation_pct']:.0f}%)"]
    if cur > hi:
        reasons.append(f"Above historical maximum of {hi:,} units")
    if math["impact_pct"] >= 50:
        reasons.append(f"High inventory consumption ({math['impact_pct']:.0f}% of available stock)")
    if inv.get("reorder_point") is not None and math["post"] <= int(inv["reorder_point"]):
        reasons.append("Product near reorder threshold after fulfilment")
    if float(inv["deviation_pct"]) >= 100:
        reasons.append("Customer behaviour change detected")
    return reasons


_BUSINESS_REASONS = [
    "New customer contract",
    "Bulk procurement cycle",
    "Inventory buffering",
    "Seasonal demand increase",
    "One-time project demand",
]


def _recommended_actions(inv: dict, math: dict) -> list[str]:
    actions = [
        "Verify order with account manager",
        "Contact customer for confirmation",
    ]
    if math["stockout"] in ("Elevated", "Moderate"):
        actions.append("Expedite replenishment")
    actions.append("Monitor follow-up orders")
    actions.append("Increase procurement planning")
    return actions


# ---------------------------------------------------------------------------
# HTML building blocks
# ---------------------------------------------------------------------------
def _badge_html(band: str) -> str:
    bg, fg = _BADGE.get(band, _BADGE["Medium"])
    return (
        f'<span style="display:inline-block;padding:6px 16px;border-radius:999px;'
        f'background:{bg};color:{fg};font-size:13px;font-weight:800;'
        f'text-transform:uppercase;letter-spacing:0.6px;">'
        f'{escape(RISK_DISPLAY_LABEL.get(band, band))}</span>'
    )


def _section(title: str, body_html: str) -> str:
    return (
        '<div style="padding:22px 30px;border-top:1px solid ' + THEME["soft_border"] + ';">'
        f'<h2 style="margin:0 0 14px 0;font-size:15px;font-weight:800;color:'
        f'{THEME["deep_navy"]};text-transform:uppercase;letter-spacing:0.5px;'
        f'border-left:4px solid {THEME["fresh_green"]};padding-left:10px;">{escape(title)}</h2>'
        f'{body_html}</div>'
    )


def _info_grid(pairs: list[tuple[str, str]]) -> str:
    """Two-column label/value grid (stacks on mobile via 100% width cells)."""
    cells = "".join(
        '<td style="padding:8px 10px;vertical-align:top;width:50%;">'
        f'<div style="font-size:11px;font-weight:700;color:{THEME["muted_text"]};'
        'text-transform:uppercase;letter-spacing:0.4px;">' + escape(label) + '</div>'
        f'<div style="font-size:15px;font-weight:700;color:{THEME["deep_navy"]};'
        'margin-top:3px;">' + value + '</div></td>'
        + ('</tr><tr>' if (i % 2 == 1) else '')
        for i, (label, value) in enumerate(pairs)
    )
    return (
        '<table role="presentation" width="100%" cellpadding="0" cellspacing="0" '
        f'style="border-collapse:collapse;background:{THEME["soft_blue"]};'
        f'border:1px solid {THEME["soft_border"]};border-radius:10px;">'
        f'<tr>{cells}</tr></table>'
    )


def _metric_tile(label: str, value: str, accent: str, highlight: bool = False) -> str:
    value_color = accent if highlight else THEME["deep_navy"]
    return (
        '<td style="padding:6px;width:33%;vertical-align:top;">'
        '<div style="background:#FFFFFF;border:1px solid ' + THEME["soft_border"] + ';'
        'border-top:3px solid ' + accent + ';border-radius:10px;padding:14px 12px;'
        'text-align:center;">'
        f'<div style="font-size:11px;font-weight:700;color:{THEME["muted_text"]};'
        'text-transform:uppercase;letter-spacing:0.4px;">' + escape(label) + '</div>'
        f'<div style="font-size:22px;font-weight:800;color:{value_color};margin-top:6px;">'
        + value + '</div></div></td>'
    )


def _bullets(items: list[str], accent: str) -> str:
    rows = "".join(
        '<tr><td style="padding:6px 0;vertical-align:top;width:22px;color:'
        + accent + ';font-weight:800;">•</td>'
        f'<td style="padding:6px 0;font-size:14px;color:{THEME["deep_navy"]};'
        'line-height:1.5;">' + escape(item) + '</td></tr>'
        for item in items
    )
    return f'<table role="presentation" width="100%" cellpadding="0" cellspacing="0">{rows}</table>'


def _pattern_html(pattern: list[int]) -> str:
    if not pattern:
        return "&mdash;"
    chips = []
    for i, qty in enumerate(pattern):
        is_last = i == len(pattern) - 1
        bg = THEME["soft_red"] if is_last else THEME["soft_blue"]
        color = THEME["danger"] if is_last else THEME["primary_navy"]
        border = THEME["danger"] if is_last else THEME["soft_border"]
        chips.append(
            f'<span style="display:inline-block;padding:4px 12px;border-radius:8px;'
            f'background:{bg};color:{color};border:1px solid {border};font-weight:800;'
            f'font-size:14px;">{int(qty):,}</span>'
        )
    arrow = (
        f'<span style="color:{THEME["muted_text"]};font-weight:700;padding:0 6px;">&rarr;</span>'
    )
    return arrow.join(chips)


# ---------------------------------------------------------------------------
# Full email
# ---------------------------------------------------------------------------
def build_abnormal_order_email_html(inv: dict) -> str:
    """Render the complete Abnormal Order Investigation Alert email (10 sections)."""
    band = str(inv.get("risk_band", "Medium"))
    score = int(inv.get("risk_score", 0))
    accent_navy = THEME["primary_navy"]
    green = THEME["fresh_green"]
    math = _inventory_math(inv)
    timestamp = datetime.now().strftime("%B %d, %Y at %I:%M %p")

    # 1) Header --------------------------------------------------------------
    header = (
        f'<div style="background:linear-gradient(135deg,{THEME["primary_navy"]},'
        f'{THEME["deep_navy"]});color:#FFFFFF;padding:34px 30px;text-align:center;">'
        '<div style="font-size:12px;font-weight:700;letter-spacing:1px;opacity:0.85;'
        'text-transform:uppercase;">Bunzl AI Supply Chain Platform</div>'
        '<h1 style="margin:8px 0 6px 0;font-size:26px;font-weight:800;">'
        '📈 Customer Demand Intelligence Alert</h1>'
        f'<div style="font-size:13px;opacity:0.85;margin-bottom:14px;">{escape(timestamp)}</div>'
        f'{_badge_html(band)}</div>'
    )

    # 2) Executive Summary ---------------------------------------------------
    summary = _section(
        "Executive Summary",
        '<div style="background:' + THEME["soft_amber"] + ';border:1px solid '
        + THEME["soft_border"] + ';border-left:4px solid ' + THEME["warning"] + ';'
        'border-radius:10px;padding:16px 18px;font-size:15px;line-height:1.6;color:'
        + THEME["deep_navy"] + ';">' + escape(_executive_summary(inv)) + '</div>',
    )

    # 3) Customer Information -------------------------------------------------
    customer = _section(
        "Customer Information",
        _info_grid([
            ("Customer Name", escape(str(inv["customer_name"]))),
            ("Customer Segment", escape(str(inv.get("customer_segment") or "—"))),
            ("Contract Tier", escape(str(inv.get("contract_tier") or "—"))),
            ("Order Number", escape(str(inv.get("order_nbr") or "—"))),
            ("Order Date", escape(str(inv.get("order_date") or "—"))),
        ]),
    )

    # 4) Product Information --------------------------------------------------
    reorder_point = inv.get("reorder_point")
    product = _section(
        "Product Information",
        _info_grid([
            ("Product Name", escape(str(inv["product_name"]))),
            ("Category", escape(str(inv.get("category") or "—"))),
            ("Current Inventory", f"{math['pre']:,} units"),
            ("Reorder Point", f"{int(reorder_point):,} units" if reorder_point is not None else "—"),
        ]),
    )

    # 5) AI Investigation Report --------------------------------------------
    metric_row1 = (
        '<table role="presentation" width="100%" cellpadding="0" cellspacing="0"><tr>'
        + _metric_tile("Historical Average", f"{round(float(inv['historical_avg'])):,}", accent_navy)
        + _metric_tile("Historical Maximum", f"{int(inv['historical_max']):,}", accent_navy)
        + _metric_tile("Latest Order", f"{int(inv['current_quantity']):,}", THEME["danger"], highlight=True)
        + '</tr></table>'
    )
    metric_row2 = (
        '<table role="presentation" width="100%" cellpadding="0" cellspacing="0"><tr>'
        + _metric_tile("Deviation", f"{inv['deviation_pct']:+.0f}%", THEME["danger"], highlight=True)
        + _metric_tile("Inventory Impact", f"{math['impact_pct']:.0f}%", THEME["warning"], highlight=True)
        + _metric_tile("Stockout Risk", escape(math["stockout"]), green)
        + '</tr></table>'
    )
    pattern_block = (
        '<div style="margin-top:14px;padding:14px 16px;background:#FFFFFF;border:1px solid '
        + THEME["soft_border"] + ';border-radius:10px;">'
        f'<div style="font-size:11px;font-weight:700;color:{THEME["muted_text"]};'
        'text-transform:uppercase;letter-spacing:0.4px;margin-bottom:8px;">Historical Pattern</div>'
        + _pattern_html(inv.get("historical_pattern") or []) + '</div>'
    )
    explanation = (
        '<div style="margin-top:14px;background:' + THEME["soft_blue"] + ';border-left:4px solid '
        + green + ';border-radius:10px;padding:14px 16px;font-size:14px;line-height:1.6;color:'
        + THEME["deep_navy"] + ';">' + escape(_explanation(inv)) + '</div>'
    )
    investigation = _section("Customer Demand Analysis", metric_row1 + metric_row2 + pattern_block + explanation)

    # 6) Demand Assessment ---------------------------------------------------
    bg, _ = _BADGE.get(band, _BADGE["Medium"])
    risk_head = (
        '<table role="presentation" width="100%" cellpadding="0" cellspacing="0"><tr>'
        + _metric_tile("Demand Level", RISK_DISPLAY_LABEL.get(band, band), bg, highlight=True)
        + _metric_tile("Demand Score", f"{score} / 100", bg, highlight=True)
        + '<td style="width:34%;"></td></tr></table>'
    )
    risk = _section(
        "Demand Assessment",
        risk_head
        + '<div style="margin-top:8px;font-size:12px;font-weight:700;color:'
        + THEME["muted_text"] + ';text-transform:uppercase;letter-spacing:0.4px;">Reasoning</div>'
        + _bullets(_risk_reasoning(inv, math), THEME["danger"]),
    )

    # 7) Possible Business Reasons ------------------------------------------
    reasons = _section("Possible Business Reasons", _bullets(_BUSINESS_REASONS, accent_navy))

    # 8) Recommended Actions -------------------------------------------------
    actions = _section(
        "Recommended Actions",
        '<div style="background:' + THEME["soft_blue"] + ';border:1px solid '
        + THEME["soft_border"] + ';border-left:4px solid ' + green + ';border-radius:10px;'
        'padding:10px 16px;">' + _bullets(_recommended_actions(inv, math), green) + '</div>',
    )

    # 9) Inventory Impact Summary -------------------------------------------
    impact = _section(
        "Inventory Impact Summary",
        _info_grid([
            ("Current Inventory", f"{math['pre']:,} units"),
            ("Order Quantity", f"{math['qty']:,} units"),
            ("Remaining Inventory", f"{math['post']:,} units"),
            ("Coverage", f"{math['coverage']:.2f}x order size"),
            ("Stockout Risk", escape(math["stockout"])),
        ]),
    )

    # 10) Footer -------------------------------------------------------------
    footer = (
        f'<div style="background:{THEME["deep_navy"]};color:#FFFFFF;padding:26px 30px;'
        'text-align:center;font-size:12px;line-height:1.7;">'
        '<div style="font-weight:700;opacity:0.95;">Generated by Bunzl AI Supply Chain '
        'Intelligence Platform</div>'
        '<div style="opacity:0.7;margin-top:6px;">This update was automatically generated by '
        'the Customer Intelligence Agent.</div></div>'
    )

    return (
        '<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1.0">'
        f'<title>{escape(SUBJECT)}</title></head>'
        f'<body style="margin:0;padding:20px;background:{THEME["light_bg"]};'
        'font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;">'
        '<div style="max-width:680px;margin:0 auto;background:#FFFFFF;border-radius:14px;'
        'overflow:hidden;box-shadow:0 10px 30px rgba(10,31,51,0.14);">'
        + header + summary + customer + product + investigation + risk
        + reasons + actions + impact + footer
        + '</div></body></html>'
    )


def _build_text_body(inv: dict) -> str:
    math = _inventory_math(inv)
    return "\n".join([
        "Customer Demand Intelligence Alert",
        f"Demand Level: {RISK_DISPLAY_LABEL.get(str(inv.get('risk_band')), inv.get('risk_band'))} "
        f"({int(inv.get('risk_score', 0))}/100)",
        "",
        _executive_summary(inv),
        "",
        f"Customer: {inv['customer_name']} | Segment: {inv.get('customer_segment') or '-'} | "
        f"Tier: {inv.get('contract_tier') or '-'}",
        f"Order: {inv.get('order_nbr') or '-'} | Date: {inv.get('order_date') or '-'}",
        f"Product: {inv['product_name']} | Category: {inv.get('category') or '-'}",
        "",
        f"Historical Average: {round(float(inv['historical_avg'])):,} units",
        f"Historical Maximum: {int(inv['historical_max']):,} units",
        f"Latest Order: {int(inv['current_quantity']):,} units",
        f"Deviation: {inv['deviation_pct']:+.0f}%",
        f"Inventory Impact: {math['impact_pct']:.0f}% of available stock",
        "",
        _explanation(inv),
        "",
        "— Bunzl AI Supply Chain Intelligence Platform",
    ])


# ---------------------------------------------------------------------------
# Sending
# ---------------------------------------------------------------------------
def should_alert(band: str) -> bool:
    """True when the order's risk band warrants an investigation alert."""
    return str(band) in _TRIGGER_BANDS


def send_abnormal_order_alert_email(inv: dict) -> dict:
    """Send a single Abnormal Order Investigation Alert to the configured manager."""
    settings = _smtp_settings()
    if not settings["smtp_email"] or not settings["smtp_password"] or not settings["manager_email"]:
        return {
            "success": False,
            "email_sent": False,
            "message": "SMTP credentials are missing (SMTP_EMAIL / SMTP_APP_PASSWORD / MANAGER_EMAIL).",
        }

    message = EmailMessage()
    message["Subject"] = f"{SUBJECT} — {inv.get('customer_name', '')} · {inv.get('product_name', '')}"
    message["From"] = settings["smtp_email"]
    message["To"] = settings["manager_email"]
    message.set_content(_build_text_body(inv))
    message.add_alternative(build_abnormal_order_email_html(inv), subtype="html")

    try:
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as server:
            server.starttls()
            server.login(settings["smtp_email"], settings["smtp_password"])
            server.send_message(message)
    except Exception as error:
        return {
            "success": False,
            "email_sent": False,
            "message": f"Customer Demand Intelligence Alert could not be sent: {error}",
        }

    return {
        "success": True,
        "email_sent": True,
        "message": f"Customer Demand Intelligence Alert sent to {settings['manager_email']}.",
    }


def queue_abnormal_order_alert_email(inv: dict) -> Future:
    """Send the alert on the shared background executor (non-blocking)."""
    return EMAIL_EXECUTOR.submit(send_abnormal_order_alert_email, inv)
