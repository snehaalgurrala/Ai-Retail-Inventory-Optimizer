from __future__ import annotations

from math import ceil, isfinite
from pathlib import Path

import pandas as pd

from backend.services.depletion_formatter import (
    depletion_sentence,
    depletion_urgency_label,
    exact_depletion_tooltip,
    format_depletion_window,
)
from backend.services.risk_score_service import calculate_inventory_risk_score, risk_category


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

DEPLETION_ALERT_DAYS = 5
SALES_RECENT_WINDOW_DAYS = 7
SALES_HISTORY_WINDOW_DAYS = 30
DEMAND_SPIKE_PERCENT = 75
SAFETY_FACTOR = 1.3
DEFAULT_SUPPLIER_LEAD_DAYS = 4


def _number_column(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(0, index=df.index)
    return pd.to_numeric(df[column], errors="coerce").fillna(0)


def _text(value, fallback: str = "") -> str:
    if pd.isna(value):
        return fallback
    text = str(value or "").strip()
    return text or fallback


def _safe_divide(numerator: float, denominator: float, fallback: float = 999.0) -> float:
    denominator = float(denominator or 0)
    if denominator <= 0:
        return fallback
    return float(numerator or 0) / denominator


def _velocity_frame(
    sales: pd.DataFrame,
    recent_window_days: int,
    history_window_days: int,
) -> pd.DataFrame:
    columns = [
        "product_id",
        "store_id",
        "recent_quantity_sold",
        "historical_quantity_sold",
        "recent_sales_days",
        "recent_avg_daily_sales",
        "historical_avg_daily_sales",
        "sales_consistency",
    ]
    if sales.empty or not {"date", "product_id", "store_id", "quantity_sold"}.issubset(sales.columns):
        return pd.DataFrame(columns=columns)

    sales_view = sales.copy()
    sales_view["date"] = pd.to_datetime(sales_view["date"], errors="coerce")
    sales_view["product_id"] = sales_view["product_id"].astype(str)
    sales_view["store_id"] = sales_view["store_id"].astype(str)
    sales_view["quantity_sold"] = _number_column(sales_view, "quantity_sold")
    sales_view = sales_view.dropna(subset=["date"])
    if sales_view.empty:
        return pd.DataFrame(columns=columns)

    latest_date = sales_view["date"].max().normalize()
    recent_start = latest_date - pd.Timedelta(days=max(1, recent_window_days) - 1)
    history_start = latest_date - pd.Timedelta(days=max(recent_window_days + 1, history_window_days) - 1)

    recent = sales_view[sales_view["date"].dt.normalize().between(recent_start, latest_date)].copy()
    history = sales_view[
        sales_view["date"].dt.normalize().between(history_start, recent_start - pd.Timedelta(days=1))
    ].copy()

    recent_daily = (
        recent.groupby(["product_id", "store_id", recent["date"].dt.normalize()], as_index=False)["quantity_sold"].sum()
        if not recent.empty
        else pd.DataFrame(columns=["product_id", "store_id", "date", "quantity_sold"])
    )
    if not recent_daily.empty:
        recent_daily = recent_daily.rename(columns={"date": "sale_day"})

    recent_summary = (
        recent.groupby(["product_id", "store_id"], as_index=False)
        .agg(recent_quantity_sold=("quantity_sold", "sum"), recent_sales_days=("date", "nunique"))
        if not recent.empty
        else pd.DataFrame(columns=["product_id", "store_id", "recent_quantity_sold", "recent_sales_days"])
    )
    history_summary = (
        history.groupby(["product_id", "store_id"], as_index=False)
        .agg(historical_quantity_sold=("quantity_sold", "sum"))
        if not history.empty
        else pd.DataFrame(columns=["product_id", "store_id", "historical_quantity_sold"])
    )

    keys = sales_view[["product_id", "store_id"]].drop_duplicates()
    velocity = keys.merge(recent_summary, on=["product_id", "store_id"], how="left")
    velocity = velocity.merge(history_summary, on=["product_id", "store_id"], how="left")
    velocity["recent_quantity_sold"] = _number_column(velocity, "recent_quantity_sold")
    velocity["historical_quantity_sold"] = _number_column(velocity, "historical_quantity_sold")
    velocity["recent_sales_days"] = _number_column(velocity, "recent_sales_days")

    history_days = max(1, history_window_days - recent_window_days)
    velocity["recent_avg_daily_sales"] = velocity["recent_quantity_sold"] / max(1, recent_window_days)
    velocity["historical_avg_daily_sales"] = velocity["historical_quantity_sold"] / history_days

    consistency = []
    if not recent_daily.empty:
        daily_lookup = recent_daily.groupby(["product_id", "store_id"])["quantity_sold"]
        for _, row in velocity.iterrows():
            values = daily_lookup.get_group((row["product_id"], row["store_id"])) if (row["product_id"], row["store_id"]) in daily_lookup.groups else pd.Series(dtype=float)
            mean = float(values.mean() or 0)
            std = float(values.std(ddof=0) or 0)
            consistency.append(1 / (1 + (std / mean))) if mean > 0 else consistency.append(0)
    else:
        consistency = [0] * len(velocity)
    velocity["sales_consistency"] = consistency

    return velocity[columns]


def _merge_metadata(
    inventory: pd.DataFrame,
    products: pd.DataFrame,
    stores: pd.DataFrame,
    suppliers: pd.DataFrame,
) -> pd.DataFrame:
    view = inventory.copy()
    view["product_id"] = view["product_id"].astype(str)
    view["store_id"] = view["store_id"].astype(str)
    view["current_quantity"] = _number_column(view, "stock_level")
    view["inventory_reorder_threshold"] = _number_column(view, "reorder_threshold")
    view = view.drop(columns=["reorder_threshold"], errors="ignore")

    if not products.empty and "product_id" in products.columns:
        product_view = products.copy()
        product_view["product_id"] = product_view["product_id"].astype(str)
        if "reorder_threshold" in product_view.columns:
            product_view = product_view.rename(columns={"reorder_threshold": "product_reorder_threshold"})
        view = view.merge(product_view, on="product_id", how="left")

    if not stores.empty and "store_id" in stores.columns:
        store_view = stores.copy()
        store_view["store_id"] = store_view["store_id"].astype(str)
        view = view.merge(store_view, on="store_id", how="left")

    if not suppliers.empty and "supplier_id" in view.columns and "supplier_id" in suppliers.columns:
        supplier_view = suppliers.copy()
        supplier_view["supplier_id"] = supplier_view["supplier_id"].astype(str)
        view["supplier_id"] = view["supplier_id"].astype(str)
        view = view.merge(supplier_view, on="supplier_id", how="left")

    view["reorder_threshold"] = view["inventory_reorder_threshold"]
    if "product_reorder_threshold" in view.columns:
        view["reorder_threshold"] = view["reorder_threshold"].where(
            view["reorder_threshold"] > 0,
            _number_column(view, "product_reorder_threshold"),
        )
    return view


def _add_transfer_recommendations(view: pd.DataFrame, config: dict) -> pd.DataFrame:
    view = view.copy()
    view["suggested_transfer_branch"] = ""
    view["suggested_transfer_branch_id"] = ""
    view["suggested_transfer_qty"] = 0
    view["transfer_recommendation"] = ""

    if view.empty:
        return view

    source_buffer_days = float(config.get("depletion_alert_days", DEPLETION_ALERT_DAYS))
    source_view = view.copy()
    source_view["source_buffer"] = source_view["avg_daily_sales"] * source_buffer_days
    source_view["transfer_surplus"] = (
        source_view["current_quantity"] - source_view[["reorder_threshold", "source_buffer"]].max(axis=1)
    ).clip(lower=0)

    for index, row in view.iterrows():
        product_id = str(row.get("product_id", ""))
        store_id = str(row.get("store_id", ""))
        matching = source_view[
            (source_view["product_id"].astype(str) == product_id)
            & (source_view["store_id"].astype(str) != store_id)
            & (source_view["transfer_surplus"] > 0)
        ].sort_values("transfer_surplus", ascending=False)
        if matching.empty:
            continue
        source = matching.iloc[0]
        needed = max(
            float(row.get("reorder_threshold", 0)) - float(row.get("current_quantity", 0)),
            float(row.get("avg_daily_sales", 0)) * source_buffer_days - float(row.get("current_quantity", 0)),
            0,
        )
        quantity = int(max(0, min(round(source.get("transfer_surplus", 0)), ceil(needed))))
        if quantity <= 0:
            quantity = int(max(0, round(min(source.get("transfer_surplus", 0), 25))))
        if quantity <= 0:
            continue
        source_name = _text(source.get("store_name"), _text(source.get("store_id")))
        view.at[index, "suggested_transfer_branch"] = source_name
        view.at[index, "suggested_transfer_branch_id"] = _text(source.get("store_id"))
        view.at[index, "suggested_transfer_qty"] = quantity
        view.at[index, "transfer_recommendation"] = (
            f"{source_name} can support with {quantity} surplus units before procurement."
        )
    return view


def build_predictive_inventory_view(
    inventory: pd.DataFrame,
    products: pd.DataFrame,
    stores: pd.DataFrame,
    sales: pd.DataFrame,
    suppliers: pd.DataFrame | None = None,
    config: dict | None = None,
) -> pd.DataFrame:
    """Create lightweight predictive inventory intelligence per product-store row."""
    config = {
        "depletion_alert_days": DEPLETION_ALERT_DAYS,
        "sales_recent_window_days": SALES_RECENT_WINDOW_DAYS,
        "sales_history_window_days": SALES_HISTORY_WINDOW_DAYS,
        "demand_spike_percent": DEMAND_SPIKE_PERCENT,
        "safety_factor": SAFETY_FACTOR,
        "default_supplier_lead_days": DEFAULT_SUPPLIER_LEAD_DAYS,
        **(config or {}),
    }
    suppliers = suppliers if suppliers is not None else pd.DataFrame()
    if inventory.empty or not {"product_id", "store_id", "stock_level"}.issubset(inventory.columns):
        return pd.DataFrame()

    view = _merge_metadata(inventory, products, stores, suppliers)
    velocity = _velocity_frame(
        sales,
        int(config["sales_recent_window_days"]),
        int(config["sales_history_window_days"]),
    )
    if not velocity.empty:
        view = view.merge(velocity, on=["product_id", "store_id"], how="left")

    for column in [
        "recent_quantity_sold",
        "historical_quantity_sold",
        "recent_sales_days",
        "recent_avg_daily_sales",
        "historical_avg_daily_sales",
        "sales_consistency",
        "reliability_score",
        "avg_delivery_days",
    ]:
        view[column] = _number_column(view, column)

    view["avg_daily_sales"] = view["recent_avg_daily_sales"].where(
        view["recent_avg_daily_sales"] > 0,
        view["historical_avg_daily_sales"],
    )
    view["moving_avg_daily_sales"] = (
        (view["recent_avg_daily_sales"] * 0.7) + (view["historical_avg_daily_sales"] * 0.3)
    ).round(3)
    view["predicted_days_remaining"] = view.apply(
        lambda row: round(_safe_divide(row.get("current_quantity", 0), row.get("avg_daily_sales", 0)), 2),
        axis=1,
    )
    spike_multiplier = 1 + (float(config["demand_spike_percent"]) / 100)
    view["demand_spike"] = (
        (view["historical_avg_daily_sales"] > 0)
        & (view["recent_avg_daily_sales"] >= view["historical_avg_daily_sales"] * spike_multiplier)
    )
    view["demand_trend"] = "Stable"
    view.loc[(view["avg_daily_sales"] <= 0), "demand_trend"] = "No sales data"
    view.loc[
        (view["recent_avg_daily_sales"] > view["historical_avg_daily_sales"] * 1.15)
        & (view["historical_avg_daily_sales"] > 0),
        "demand_trend",
    ] = "Rising"
    view.loc[
        (view["recent_avg_daily_sales"] < view["historical_avg_daily_sales"] * 0.85)
        & (view["historical_avg_daily_sales"] > 0),
        "demand_trend",
    ] = "Falling"
    view.loc[view["demand_spike"], "demand_trend"] = "Demand Spike"

    view = _add_transfer_recommendations(view, config)

    lead_days = view["avg_delivery_days"].where(
        view["avg_delivery_days"] > 0,
        float(config["default_supplier_lead_days"]),
    )
    view["supplier_lead_time_days"] = lead_days
    view["suggested_reorder_qty"] = (
        view["avg_daily_sales"] * lead_days * float(config["safety_factor"])
    ).map(ceil).clip(lower=0)
    fallback_reorder = ((view["reorder_threshold"] * 2) - view["current_quantity"]).clip(lower=0).map(ceil)
    view["suggested_reorder_qty"] = view["suggested_reorder_qty"].where(
        view["suggested_reorder_qty"] > 0,
        fallback_reorder,
    ).astype(int)

    has_recent_data = view["recent_quantity_sold"] > 0
    view["confidence_level"] = "Low"
    view.loc[has_recent_data, "confidence_level"] = "Medium"
    view.loc[
        has_recent_data
        & (view["recent_sales_days"] >= 3)
        & (view["sales_consistency"] >= 0.45),
        "confidence_level",
    ] = "High"
    view.loc[view["demand_spike"] & (view["recent_sales_days"] < 3), "confidence_level"] = "Medium"

    supplier_risk = (1 - view["reliability_score"]).clip(lower=0, upper=1)
    branch_dependency = view.groupby("product_id")["store_id"].transform("count")
    branch_dependency = (1 / branch_dependency.replace(0, 1)).clip(lower=0, upper=1)
    view["risk_score"] = [
        calculate_inventory_risk_score(
            current_stock=row.current_quantity,
            avg_daily_sales=row.avg_daily_sales,
            days_remaining=row.predicted_days_remaining,
            supplier_risk=supplier_risk.iloc[pos],
            transfer_available=bool(row.suggested_transfer_branch),
            branch_dependency=branch_dependency.iloc[pos],
            demand_spike=bool(row.demand_spike),
            depletion_alert_days=float(config["depletion_alert_days"]),
        )
        for pos, row in enumerate(view.itertuples(index=False))
    ]
    view["risk_category"] = view["risk_score"].map(risk_category)
    view["depletion_window"] = view["predicted_days_remaining"].map(format_depletion_window)
    view["urgency_label"] = view["predicted_days_remaining"].map(depletion_urgency_label)
    view["depletion_tooltip"] = view["predicted_days_remaining"].map(exact_depletion_tooltip)

    predictive_trigger = (
        (view["avg_daily_sales"] > 0)
        & (view["predicted_days_remaining"] <= float(config["depletion_alert_days"]))
    )
    threshold_trigger = (
        (view["avg_daily_sales"] <= 0)
        & (view["reorder_threshold"] > 0)
        & (view["current_quantity"] <= view["reorder_threshold"])
    )
    view["predictive_alert"] = predictive_trigger | threshold_trigger | view["demand_spike"]
    view["fallback_threshold_alert"] = threshold_trigger

    view["alert_reason"] = view.apply(_build_alert_reason, axis=1)
    view["ai_alert_message"] = view.apply(_build_ai_alert_message, axis=1)
    view["shortage_quantity"] = (view["reorder_threshold"] - view["current_quantity"]).clip(lower=0)
    view["suggested_reorder_quantity"] = view["suggested_reorder_qty"]
    view["recent_daily_sales_velocity"] = view["avg_daily_sales"]
    return view


def _build_alert_reason(row: pd.Series) -> str:
    product = _text(row.get("product_name"), _text(row.get("product_id"), "This product"))
    store = _text(row.get("store_name"), _text(row.get("store_id"), "this branch"))
    days = float(row.get("predicted_days_remaining", 999) or 999)
    window = format_depletion_window(days)
    if bool(row.get("fallback_threshold_alert", False)):
        return f"No recent sales history is available, so {product} at {store} is flagged using the reorder threshold fallback."
    if bool(row.get("demand_spike", False)):
        return f"Demand spike detected for {product} at {store}; recent velocity is above historical demand."
    if float(row.get("avg_daily_sales", 0) or 0) > 0:
        return f"Recent demand velocity predicts {product} at {store} has {window.lower()}."
    return f"{product} at {store} is currently monitored with limited sales signal."


def _build_ai_alert_message(row: pd.Series) -> str:
    product = _text(row.get("product_name"), _text(row.get("product_id"), "This product"))
    store = _text(row.get("store_name"), _text(row.get("store_id"), "this branch"))
    days = float(row.get("predicted_days_remaining", 999) or 999)
    trend = _text(row.get("demand_trend"), "stable demand").lower()
    transfer = _text(row.get("transfer_recommendation"))
    if isfinite(days) and days < 999 and float(row.get("avg_daily_sales", 0) or 0) > 0:
        message = depletion_sentence(f"{product} in {store}", days)
    elif bool(row.get("fallback_threshold_alert", False)):
        message = f"{product} in {store} needs review because sales history is limited and stock is at or below its reorder point."
    else:
        message = f"{product} in {store} is {str(row.get('risk_category', 'Healthy')).lower()} with {trend}."
    if bool(row.get("demand_spike", False)):
        message += " Demand spike detected."
    if transfer:
        message += f" {transfer}"
    return message


def get_predictive_inventory_alerts(
    inventory: pd.DataFrame,
    products: pd.DataFrame,
    stores: pd.DataFrame,
    sales: pd.DataFrame,
    suppliers: pd.DataFrame | None = None,
    config: dict | None = None,
) -> pd.DataFrame:
    view = build_predictive_inventory_view(inventory, products, stores, sales, suppliers, config)
    if view.empty:
        return view
    alerts = view[view["predictive_alert"]].copy()
    priority_rank = {"Critical": 0, "High": 1, "Medium": 2, "Healthy": 3}
    alerts["_risk_rank"] = alerts["risk_category"].map(priority_rank).fillna(4)
    return alerts.sort_values(
        ["_risk_rank", "predicted_days_remaining", "risk_score"],
        ascending=[True, True, False],
    ).drop(columns=["_risk_rank"], errors="ignore")
