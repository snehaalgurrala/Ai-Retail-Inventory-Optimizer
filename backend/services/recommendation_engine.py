from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


DEFAULT_CONFIG = {
    "reorder_cover_days": 7,
    "stockout_high_priority_days": 3,
    "stockout_medium_priority_days": 7,
    "transfer_source_buffer_multiplier": 1.5,
    "discount_days_of_stock": 45,
    "clearance_shelf_life_multiplier": 1.0,
    "supplier_reliability_threshold": 0.90,
    "supplier_delivery_days_threshold": 4,
}


RECOMMENDATION_COLUMNS = [
    "recommendation_id",
    "recommendation_type",
    "product_id",
    "product_name",
    "store_id",
    "priority",
    "action",
    "reason",
    "evidence",
    "suggested_quantity",
    "source_agent",
    "status",
]


def get_config(config: dict | None = None) -> dict:
    """Combine default recommendation settings with optional overrides."""
    final_config = DEFAULT_CONFIG.copy()
    if config:
        final_config.update(config)
    return final_config


def _read_processed_csv(filename: str) -> pd.DataFrame:
    """Read one processed CSV, returning an empty dataframe if it is missing."""
    file_path = PROCESSED_DATA_DIR / filename
    if not file_path.exists():
        return pd.DataFrame()
    return pd.read_csv(file_path)


def _read_raw_csv(filename: str) -> pd.DataFrame:
    """Read one raw CSV, returning an empty dataframe if it is missing."""
    file_path = RAW_DATA_DIR / filename
    if not file_path.exists():
        return pd.DataFrame()
    return pd.read_csv(file_path)


def load_recommendation_inputs() -> dict[str, pd.DataFrame]:
    """Load processed datasets, analyzer outputs, and supplier data."""
    return {
        "current_inventory": _read_processed_csv("current_inventory.csv"),
        "product_performance": _read_processed_csv("product_performance.csv"),
        "low_stock_items": _read_processed_csv("low_stock_items.csv"),
        "stockout_risk_items": _read_processed_csv("stockout_risk_items.csv"),
        "overstock_items": _read_processed_csv("overstock_items.csv"),
        "dead_stock_candidates": _read_processed_csv(
            "dead_stock_candidates.csv"
        ),
        "high_demand_items": _read_processed_csv("high_demand_items.csv"),
        "slow_moving_items": _read_processed_csv("slow_moving_items.csv"),
        "suppliers": _read_raw_csv("suppliers.csv"),
    }


def _number_value(row: pd.Series, column: str, default: float = 0) -> float:
    """Read a numeric value from a dataframe row."""
    value = row.get(column, default)
    value = pd.to_numeric(value, errors="coerce")
    if pd.isna(value):
        return default
    return float(value)


def _number_column(df: pd.DataFrame, column: str) -> pd.Series:
    """Return a numeric column, or zeroes if the column is missing."""
    if column not in df.columns:
        return pd.Series(0, index=df.index)
    return pd.to_numeric(df[column], errors="coerce").fillna(0)


def _text_value(row: pd.Series, column: str) -> str:
    """Read a text value from a dataframe row."""
    value = row.get(column, "")
    if pd.isna(value):
        return ""
    return str(value)


def _priority_from_days_of_stock(
    days_of_stock: float,
    config: dict,
) -> str:
    """Set alert priority from stock coverage."""
    if days_of_stock <= config["stockout_high_priority_days"]:
        return "high"
    if days_of_stock <= config["stockout_medium_priority_days"]:
        return "medium"
    return "low"


def _new_recommendation(
    recommendation_type: str,
    product_id: str,
    product_name: str,
    store_id: str,
    priority: str,
    action: str,
    reason: str,
    evidence: str,
    suggested_quantity: float | int | str = "",
) -> dict:
    """Create one recommendation record."""
    return {
        "recommendation_type": recommendation_type,
        "product_id": product_id,
        "product_name": product_name,
        "store_id": store_id,
        "priority": priority,
        "action": action,
        "reason": reason,
        "evidence": evidence,
        "suggested_quantity": suggested_quantity,
        "source_agent": "recommendation_engine",
        "status": "pending",
    }


def generate_reorder_recommendations(
    low_stock_items: pd.DataFrame,
    config: dict | None = None,
) -> list[dict]:
    """Recommend reorder quantities for low-stock product-store rows."""
    config = get_config(config)
    recommendations = []

    for _, row in low_stock_items.iterrows():
        stock = _number_value(row, "stock_level")
        threshold = _number_value(row, "effective_reorder_threshold")
        velocity = _number_value(row, "recent_daily_sales_velocity")
        target_stock = threshold + (velocity * config["reorder_cover_days"])
        suggested_quantity = max(0, round(target_stock - stock))

        if suggested_quantity <= 0:
            continue

        days_of_stock = _number_value(row, "days_of_stock_remaining", 999)
        priority = _priority_from_days_of_stock(days_of_stock, config)

        recommendations.append(
            _new_recommendation(
                recommendation_type="reorder",
                product_id=_text_value(row, "product_id"),
                product_name=_text_value(row, "product_name"),
                store_id=_text_value(row, "store_id"),
                priority=priority,
                action=f"Reorder {suggested_quantity} units.",
                reason="Stock is below the reorder point after recent demand is considered.",
                evidence=_text_value(row, "evidence"),
                suggested_quantity=suggested_quantity,
            )
        )

    return recommendations


def generate_stock_transfer_recommendations(
    current_inventory: pd.DataFrame,
    low_stock_items: pd.DataFrame,
    config: dict | None = None,
) -> list[dict]:
    """Recommend transfers from stores with surplus stock to low-stock stores."""
    config = get_config(config)
    recommendations = []

    if current_inventory.empty or low_stock_items.empty:
        return recommendations

    inventory = current_inventory.copy()
    inventory["stock_level"] = _number_column(inventory, "stock_level")
    inventory["effective_reorder_threshold"] = _number_column(
        inventory,
        "inventory_reorder_threshold",
    )
    missing_threshold = inventory["effective_reorder_threshold"] == 0
    inventory.loc[missing_threshold, "effective_reorder_threshold"] = (
        _number_column(inventory, "reorder_threshold")
    )

    source_limit = (
        inventory["effective_reorder_threshold"]
        * config["transfer_source_buffer_multiplier"]
    )
    inventory["transfer_surplus"] = inventory["stock_level"] - source_limit
    source_inventory = inventory[inventory["transfer_surplus"] > 0].copy()

    for _, destination in low_stock_items.iterrows():
        product_id = _text_value(destination, "product_id")
        destination_store = _text_value(destination, "store_id")
        stock = _number_value(destination, "stock_level")
        threshold = _number_value(destination, "effective_reorder_threshold")
        needed_quantity = max(0, round(threshold - stock))

        if needed_quantity <= 0:
            continue

        matching_sources = source_inventory[
            (source_inventory["product_id"].astype(str) == product_id)
            & (source_inventory["store_id"].astype(str) != destination_store)
        ].sort_values("transfer_surplus", ascending=False)

        if matching_sources.empty:
            continue

        source = matching_sources.iloc[0]
        suggested_quantity = min(
            needed_quantity,
            round(_number_value(source, "transfer_surplus")),
        )
        if suggested_quantity <= 0:
            continue

        source_store = _text_value(source, "store_id")
        source_store_name = _text_value(source, "store_name")
        destination_store_name = _text_value(destination, "store_name")

        recommendations.append(
            _new_recommendation(
                recommendation_type="stock_transfer",
                product_id=product_id,
                product_name=_text_value(destination, "product_name"),
                store_id=destination_store,
                priority="medium",
                action=(
                    f"Transfer {suggested_quantity} units from "
                    f"{source_store} to {destination_store}."
                ),
                reason="One store is below reorder point while another store has surplus stock.",
                evidence=(
                    f"source_store={source_store_name or source_store}, "
                    f"destination_store={destination_store_name or destination_store}, "
                    f"source_surplus={round(_number_value(source, 'transfer_surplus'), 2)}, "
                    f"destination_stock={round(stock, 2)}, "
                    f"destination_threshold={round(threshold, 2)}"
                ),
                suggested_quantity=suggested_quantity,
            )
        )

    return recommendations


def generate_discount_recommendations(
    slow_moving_items: pd.DataFrame,
    overstock_items: pd.DataFrame,
    config: dict | None = None,
) -> list[dict]:
    """Recommend discounts for slow-moving or overstocked items."""
    config = get_config(config)
    recommendations = []
    combined = pd.concat(
        [slow_moving_items, overstock_items],
        ignore_index=True,
    ).drop_duplicates(subset=["product_id", "store_id"])

    for _, row in combined.iterrows():
        days_of_stock = _number_value(row, "days_of_stock_remaining")
        if days_of_stock < config["discount_days_of_stock"]:
            continue

        recommendations.append(
            _new_recommendation(
                recommendation_type="discount",
                product_id=_text_value(row, "product_id"),
                product_name=_text_value(row, "product_name"),
                store_id=_text_value(row, "store_id"),
                priority="medium",
                action="Consider a discount to improve sell-through.",
                reason="Inventory coverage is high compared with recent sales velocity.",
                evidence=_text_value(row, "evidence"),
            )
        )

    return recommendations


def generate_clearance_recommendations(
    dead_stock_candidates: pd.DataFrame,
    config: dict | None = None,
) -> list[dict]:
    """Recommend clearance when stock is slow and shelf life creates pressure."""
    config = get_config(config)
    recommendations = []

    for _, row in dead_stock_candidates.iterrows():
        shelf_life_days = _number_value(row, "shelf_life_days")
        days_of_stock = _number_value(row, "days_of_stock_remaining")
        recent_30_day_sales = _number_value(row, "recent_30_day_quantity_sold")

        shelf_life_pressure = (
            shelf_life_days > 0
            and days_of_stock
            >= shelf_life_days * config["clearance_shelf_life_multiplier"]
        )
        no_recent_sales = recent_30_day_sales == 0
        if not (shelf_life_pressure or no_recent_sales):
            continue

        recommendations.append(
            _new_recommendation(
                recommendation_type="clearance",
                product_id=_text_value(row, "product_id"),
                product_name=_text_value(row, "product_name"),
                store_id=_text_value(row, "store_id"),
                priority="high" if shelf_life_pressure else "medium",
                action="Plan clearance for the affected stock.",
                reason="Stock has weak recent movement or may exceed shelf life.",
                evidence=_text_value(row, "evidence"),
            )
        )

    return recommendations


def generate_supplier_risk_alerts(
    product_performance: pd.DataFrame,
    suppliers: pd.DataFrame,
    config: dict | None = None,
) -> list[dict]:
    """Create supplier risk alerts from supplier reliability and delivery data."""
    config = get_config(config)
    recommendations = []

    if product_performance.empty or suppliers.empty:
        return recommendations

    supplier_data = suppliers.copy()
    supplier_data["reliability_score"] = _number_column(
        supplier_data,
        "reliability_score",
    )
    supplier_data["avg_delivery_days"] = _number_column(
        supplier_data,
        "avg_delivery_days",
    )

    product_data = product_performance.copy()
    if "supplier_name" in product_data.columns:
        product_data = product_data.drop(columns=["supplier_name"])
    if "avg_delivery_days" in product_data.columns:
        product_data = product_data.drop(columns=["avg_delivery_days"])
    if "reliability_score" in product_data.columns:
        product_data = product_data.drop(columns=["reliability_score"])

    if "supplier_id" not in product_data.columns:
        return recommendations

    product_data = product_data.merge(supplier_data, on="supplier_id", how="left")
    product_data["reliability_score"] = _number_column(
        product_data,
        "reliability_score",
    )
    product_data["avg_delivery_days"] = _number_column(
        product_data,
        "avg_delivery_days",
    )

    risky_products = product_data[
        (
            product_data["reliability_score"]
            < config["supplier_reliability_threshold"]
        )
        | (
            product_data["avg_delivery_days"]
            >= config["supplier_delivery_days_threshold"]
        )
    ].copy()

    for _, row in risky_products.iterrows():
        reliability_score = _number_value(row, "reliability_score")
        avg_delivery_days = _number_value(row, "avg_delivery_days")
        priority = (
            "high"
            if reliability_score < config["supplier_reliability_threshold"]
            else "medium"
        )

        recommendations.append(
            _new_recommendation(
                recommendation_type="supplier_risk_alert",
                product_id=_text_value(row, "product_id"),
                product_name=_text_value(row, "product_name"),
                store_id="",
                priority=priority,
                action="Review supplier performance before placing the next replenishment order.",
                reason="Supplier reliability or delivery time crosses the configured risk threshold.",
                evidence=(
                    f"supplier_id={_text_value(row, 'supplier_id')}, "
                    f"supplier_name={_text_value(row, 'supplier_name')}, "
                    f"reliability_score={round(reliability_score, 2)}, "
                    f"avg_delivery_days={round(avg_delivery_days, 2)}"
                ),
            )
        )

    return recommendations


def generate_overstock_alerts(
    overstock_items: pd.DataFrame,
) -> list[dict]:
    """Create overstock alerts from analyzer output."""
    recommendations = []

    for _, row in overstock_items.iterrows():
        recommendations.append(
            _new_recommendation(
                recommendation_type="overstock_alert",
                product_id=_text_value(row, "product_id"),
                product_name=_text_value(row, "product_name"),
                store_id=_text_value(row, "store_id"),
                priority="medium",
                action="Review replenishment and sell-through plan for this stock.",
                reason=_text_value(row, "reason"),
                evidence=_text_value(row, "evidence"),
            )
        )

    return recommendations


def generate_stockout_prevention_alerts(
    stockout_risk_items: pd.DataFrame,
    config: dict | None = None,
) -> list[dict]:
    """Create stockout prevention alerts from analyzer output."""
    config = get_config(config)
    recommendations = []

    for _, row in stockout_risk_items.iterrows():
        days_of_stock = _number_value(row, "days_of_stock_remaining", 999)
        priority = _priority_from_days_of_stock(days_of_stock, config)

        recommendations.append(
            _new_recommendation(
                recommendation_type="stockout_prevention_alert",
                product_id=_text_value(row, "product_id"),
                product_name=_text_value(row, "product_name"),
                store_id=_text_value(row, "store_id"),
                priority=priority,
                action="Take replenishment action before projected stockout.",
                reason=_text_value(row, "reason"),
                evidence=_text_value(row, "evidence"),
            )
        )

    return recommendations


def build_recommendations(config: dict | None = None) -> pd.DataFrame:
    """Generate all recommendations and write data/processed/recommendations.csv."""
    config = get_config(config)
    inputs = load_recommendation_inputs()

    recommendations = []
    recommendations.extend(
        generate_reorder_recommendations(inputs["low_stock_items"], config)
    )
    recommendations.extend(
        generate_stock_transfer_recommendations(
            inputs["current_inventory"],
            inputs["low_stock_items"],
            config,
        )
    )
    recommendations.extend(
        generate_discount_recommendations(
            inputs["slow_moving_items"],
            inputs["overstock_items"],
            config,
        )
    )
    recommendations.extend(
        generate_clearance_recommendations(
            inputs["dead_stock_candidates"],
            config,
        )
    )
    recommendations.extend(
        generate_supplier_risk_alerts(
            inputs["product_performance"],
            inputs["suppliers"],
            config,
        )
    )
    recommendations.extend(
        generate_overstock_alerts(inputs["overstock_items"])
    )
    recommendations.extend(
        generate_stockout_prevention_alerts(
            inputs["stockout_risk_items"],
            config,
        )
    )

    recommendations_df = pd.DataFrame(recommendations)
    if recommendations_df.empty:
        recommendations_df = pd.DataFrame(columns=RECOMMENDATION_COLUMNS)
    else:
        recommendations_df.insert(
            0,
            "recommendation_id",
            [
                f"REC{str(index + 1).zfill(5)}"
                for index in range(len(recommendations_df))
            ],
        )
        recommendations_df = recommendations_df[RECOMMENDATION_COLUMNS]

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    recommendations_df.to_csv(
        PROCESSED_DATA_DIR / "recommendations.csv",
        index=False,
    )

    return recommendations_df


if __name__ == "__main__":
    build_recommendations()
