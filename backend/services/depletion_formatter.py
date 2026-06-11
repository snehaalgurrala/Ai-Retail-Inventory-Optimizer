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
