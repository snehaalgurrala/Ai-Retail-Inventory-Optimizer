from __future__ import annotations

import pandas as pd


def _to_float(value, default: float = 999.0) -> float:
    parsed = pd.to_numeric(value, errors="coerce")
    if pd.isna(parsed):
        return default
    return float(parsed)


def format_depletion_window(days_remaining) -> str:
    """Convert numeric predicted days into operational display language."""
    days = _to_float(days_remaining)
    if days < 1:
        return "Less than 1 day remaining"
    if days < 2:
        return "About 1 day remaining"
    if days < 5:
        return "2–5 days remaining"
    if days < 10:
        return "Less than 10 days remaining"
    if days < 30:
        return "2–4 weeks remaining"
    return "Inventory stable"


def depletion_urgency_label(days_remaining) -> str:
    """Return the display urgency label for a depletion window."""
    days = _to_float(days_remaining)
    if days < 2:
        return "Critical"
    if days < 5:
        return "High"
    if days < 10:
        return "Medium"
    return "Healthy"


def depletion_urgency_color(days_remaining) -> str:
    """Return a simple semantic color for display badges."""
    label = depletion_urgency_label(days_remaining)
    return {
        "Critical": "red",
        "High": "orange",
        "Medium": "amber",
        "Healthy": "green",
    }.get(label, "green")


def exact_depletion_tooltip(days_remaining) -> str:
    days = _to_float(days_remaining)
    if days >= 999:
        return "Estimated: unavailable"
    return f"Estimated: {days:.2f} days"


# Ranking used to reconcile a demand forecast with the inventory position: a
# lower number is more urgent. Both the position labels (Critical / Reorder
# Required / Monitor / Healthy) and the forecast labels (Critical / High /
# Medium / Healthy) are mapped onto the same scale so the more urgent of the two
# can always win.
URGENCY_RANK = {
    "critical": 0,
    "reorder required": 1,
    "high": 1,
    "monitor": 2,
    "medium": 2,
    "healthy": 3,
    "low": 3,
}


def urgency_rank(label) -> int:
    return URGENCY_RANK.get(str(label or "").strip().lower(), 3)


def inventory_position_status(current_quantity, reorder_threshold):
    """Classify an item purely on its stock position vs. its reorder point.

    Returns an ``(urgency, window)`` tuple, or ``None`` when there is no reorder
    point to compare against. This is demand-agnostic on purpose: it stays
    correct even when there is no recent sales signal (the common case for an
    order-triggered low-stock alert), so it never reports "Inventory stable" for
    an item that has actually reached its reorder threshold.

    Bands:
        Stock <= 0              -> Critical / "Out of stock"
        Stock < reorder point   -> Critical / "Below reorder threshold"
        Stock == reorder point  -> Reorder Required / "At reorder threshold"
        RP < Stock <= 1.5 x RP  -> Monitor / "Approaching reorder threshold"
        Stock > 1.5 x RP        -> Healthy / "Inventory stable"
    """
    reorder = _to_float(reorder_threshold, 0.0)
    if reorder <= 0:
        return None
    stock = _to_float(current_quantity, 0.0)
    if stock <= 0:
        return "Critical", "Out of stock"
    if stock < reorder:
        return "Critical", "Below reorder threshold"
    if stock <= reorder:  # equal to threshold (the earlier check ruled out <)
        return "Reorder Required", "At reorder threshold"
    if stock <= reorder * 1.5:
        return "Monitor", "Approaching reorder threshold"
    return "Healthy", "Inventory stable"


def depletion_sentence(product_name: str, days_remaining) -> str:
    """Short chatbot/report sentence using the humanized depletion window."""
    window = format_depletion_window(days_remaining)
    urgency = depletion_urgency_label(days_remaining).lower()
    product = str(product_name or "This item").strip() or "This item"
    if urgency == "critical":
        return (
            f"{product} inventory is critically low and may run out within the next day "
            "based on recent demand patterns."
        )
    if urgency == "high":
        return f"{product} inventory is at high risk and may deplete within 2–5 days based on recent demand patterns."
    if urgency == "medium":
        return f"{product} inventory needs monitoring with {window.lower()}."
    return f"{product} inventory is stable based on recent demand patterns."
