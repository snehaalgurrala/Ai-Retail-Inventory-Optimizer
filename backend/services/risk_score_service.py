from __future__ import annotations


def risk_category(score: float) -> str:
    """Map a 0-100 inventory risk score to a dashboard category."""
    score = max(0, min(100, float(score or 0)))
    if score <= 30:
        return "Healthy"
    if score <= 60:
        return "Medium"
    if score <= 80:
        return "High"
    return "Critical"


def calculate_inventory_risk_score(
    current_stock: float,
    avg_daily_sales: float,
    days_remaining: float,
    supplier_risk: float = 0,
    transfer_available: bool = False,
    branch_dependency: float = 0,
    demand_spike: bool = False,
    depletion_alert_days: float = 5,
) -> int:
    """Blend lightweight inventory signals into one bounded risk score."""
    current_stock = max(0, float(current_stock or 0))
    avg_daily_sales = max(0, float(avg_daily_sales or 0))
    days_remaining = float(days_remaining if days_remaining is not None else 999)
    supplier_risk = max(0, min(1, float(supplier_risk or 0)))
    branch_dependency = max(0, min(1, float(branch_dependency or 0)))
    depletion_alert_days = max(1, float(depletion_alert_days or 5))

    if avg_daily_sales <= 0:
        velocity_risk = 20 if current_stock <= 0 else 5
    elif days_remaining <= 0:
        velocity_risk = 60
    elif days_remaining >= depletion_alert_days * 2:
        velocity_risk = 0
    else:
        velocity_risk = max(0, (1 - (days_remaining / (depletion_alert_days * 2))) * 60)

    stock_pressure = 25 if current_stock <= 0 else max(0, min(25, 25 / (current_stock + 1)))
    spike_pressure = 15 if demand_spike else 0
    supplier_pressure = supplier_risk * 15
    dependency_pressure = branch_dependency * 10
    transfer_relief = 8 if transfer_available else 0

    score = velocity_risk + stock_pressure + spike_pressure + supplier_pressure + dependency_pressure - transfer_relief
    return int(round(max(0, min(100, score))))
