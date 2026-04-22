from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


DEFAULT_CONFIG = {
    "low_stock_buffer_percent": 0,
    "stockout_risk_days": 7,
    "overstock_days": 45,
    "overstock_threshold_multiplier": 3,
    "dead_stock_recent_days": 30,
    "dead_stock_max_recent_quantity": 0,
    "high_demand_percentile": 0.75,
    "slow_moving_percentile": 0.25,
    "shelf_life_clearance_multiplier": 1.0,
}


def get_config(config: dict | None = None) -> dict:
    """Combine default thresholds with optional overrides."""
    final_config = DEFAULT_CONFIG.copy()
    if config:
        final_config.update(config)
    return final_config


def load_processed_data() -> dict[str, pd.DataFrame]:
    """Load processed datasets created by data_processor.py."""
    return {
        "current_inventory": pd.read_csv(
            PROCESSED_DATA_DIR / "current_inventory.csv"
        ),
        "sales_summary": pd.read_csv(PROCESSED_DATA_DIR / "sales_summary.csv"),
        "product_performance": pd.read_csv(
            PROCESSED_DATA_DIR / "product_performance.csv"
        ),
        "store_inventory_summary": pd.read_csv(
            PROCESSED_DATA_DIR / "store_inventory_summary.csv"
        ),
    }


def _fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values safely for analysis output."""
    df = df.copy()

    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            df[column] = df[column].fillna(0)
        else:
            df[column] = df[column].fillna("")

    return df


def _number_column(df: pd.DataFrame, column: str) -> pd.Series:
    """Return a numeric column, or zeroes if the column is missing."""
    if column not in df.columns:
        return pd.Series(0, index=df.index)
    return pd.to_numeric(df[column], errors="coerce").fillna(0)


def _select_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the most useful columns for dashboard and recommendation views."""
    preferred_columns = [
        "product_id",
        "product_name",
        "category",
        "store_id",
        "store_name",
        "city",
        "stock_level",
        "effective_reorder_threshold",
        "recent_7_day_quantity_sold",
        "recent_30_day_quantity_sold",
        "recent_daily_sales_velocity",
        "days_of_stock_remaining",
        "shelf_life_days",
        "supplier_id",
        "last_sale_date",
        "reason",
        "trigger",
        "evidence",
    ]
    available_columns = [
        column for column in preferred_columns if column in df.columns
    ]
    return df[available_columns].copy()


def create_analysis_base(
    current_inventory: pd.DataFrame | None = None,
    sales_summary: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge inventory with sales metrics and derive reusable analysis columns."""
    if current_inventory is None or sales_summary is None:
        processed_data = load_processed_data()
        current_inventory = processed_data["current_inventory"]
        sales_summary = processed_data["sales_summary"]

    df = current_inventory.copy()
    sales = sales_summary.copy()

    df = df.merge(
        sales,
        on=["product_id", "store_id"],
        how="left",
        suffixes=("", "_sales"),
    )

    df["stock_level"] = _number_column(df, "stock_level")
    df["inventory_reorder_threshold"] = _number_column(
        df,
        "inventory_reorder_threshold",
    )
    df["reorder_threshold"] = _number_column(df, "reorder_threshold")
    df["effective_reorder_threshold"] = df["inventory_reorder_threshold"]
    missing_inventory_threshold = df["effective_reorder_threshold"] == 0
    df.loc[missing_inventory_threshold, "effective_reorder_threshold"] = df.loc[
        missing_inventory_threshold,
        "reorder_threshold",
    ]

    df["recent_7_day_quantity_sold"] = _number_column(
        df,
        "recent_7_day_quantity_sold",
    )
    df["recent_30_day_quantity_sold"] = _number_column(
        df,
        "recent_30_day_quantity_sold",
    )
    df["total_quantity_sold"] = _number_column(df, "total_quantity_sold")
    df["shelf_life_days"] = _number_column(df, "shelf_life_days")

    df["recent_daily_sales_velocity"] = df["recent_7_day_quantity_sold"] / 7
    no_recent_7_day_sales = df["recent_7_day_quantity_sold"] == 0
    df.loc[no_recent_7_day_sales, "recent_daily_sales_velocity"] = (
        df.loc[no_recent_7_day_sales, "recent_30_day_quantity_sold"] / 30
    )

    df["days_of_stock_remaining"] = 0.0
    has_velocity = df["recent_daily_sales_velocity"] > 0
    df.loc[has_velocity, "days_of_stock_remaining"] = (
        df.loc[has_velocity, "stock_level"]
        / df.loc[has_velocity, "recent_daily_sales_velocity"]
    )

    return _fill_missing_values(df)


def identify_low_stock_items(config: dict | None = None) -> pd.DataFrame:
    """Find items where stock is at or below the configured reorder point."""
    config = get_config(config)
    df = create_analysis_base()

    threshold_with_buffer = df["effective_reorder_threshold"] * (
        1 + config["low_stock_buffer_percent"] / 100
    )
    result = df[
        (df["effective_reorder_threshold"] > 0)
        & (df["stock_level"] <= threshold_with_buffer)
    ].copy()

    result["reason"] = "stock is at or below the configured reorder point"
    result["trigger"] = (
        "stock_level <= reorder_threshold_with_buffer_"
        + str(config["low_stock_buffer_percent"])
        + "_percent"
    )
    result["evidence"] = (
        "stock="
        + result["stock_level"].round(2).astype(str)
        + ", reorder_threshold="
        + result["effective_reorder_threshold"].round(2).astype(str)
    )

    return _select_output_columns(result)


def identify_stockout_risk_items(config: dict | None = None) -> pd.DataFrame:
    """Find items that may run out soon based on recent sales velocity."""
    config = get_config(config)
    df = create_analysis_base()

    result = df[
        (df["stock_level"] > 0)
        & (df["recent_daily_sales_velocity"] > 0)
        & (df["days_of_stock_remaining"] <= config["stockout_risk_days"])
    ].copy()

    result["reason"] = "current stock may not cover recent sales velocity"
    result["trigger"] = (
        "days_of_stock_remaining <= "
        + str(config["stockout_risk_days"])
    )
    result["evidence"] = (
        "days_of_stock="
        + result["days_of_stock_remaining"].round(2).astype(str)
        + ", recent_daily_velocity="
        + result["recent_daily_sales_velocity"].round(2).astype(str)
    )

    return _select_output_columns(result)


def identify_overstock_items(config: dict | None = None) -> pd.DataFrame:
    """Find items with stock high relative to reorder point or sales velocity."""
    config = get_config(config)
    df = create_analysis_base()

    reorder_limit = (
        df["effective_reorder_threshold"]
        * config["overstock_threshold_multiplier"]
    )
    high_vs_reorder_point = (
        (df["effective_reorder_threshold"] > 0)
        & (df["stock_level"] >= reorder_limit)
    )
    high_vs_sales_velocity = (
        (df["recent_daily_sales_velocity"] > 0)
        & (df["days_of_stock_remaining"] >= config["overstock_days"])
    )

    result = df[high_vs_reorder_point | high_vs_sales_velocity].copy()

    result["reason"] = "stock is high compared with reorder point or sales velocity"
    result["trigger"] = (
        "stock >= reorder_threshold * "
        + str(config["overstock_threshold_multiplier"])
        + " or days_of_stock_remaining >= "
        + str(config["overstock_days"])
    )
    result["evidence"] = (
        "stock="
        + result["stock_level"].round(2).astype(str)
        + ", reorder_threshold="
        + result["effective_reorder_threshold"].round(2).astype(str)
        + ", days_of_stock="
        + result["days_of_stock_remaining"].round(2).astype(str)
    )

    return _select_output_columns(result)


def identify_dead_stock_candidates(config: dict | None = None) -> pd.DataFrame:
    """Find stocked items with little or no recent movement."""
    config = get_config(config)
    df = create_analysis_base()

    low_recent_sales = (
        df["recent_30_day_quantity_sold"]
        <= config["dead_stock_max_recent_quantity"]
    )
    stock_on_hand = df["stock_level"] > 0
    shelf_life_pressure = (
        (df["shelf_life_days"] > 0)
        & (df["days_of_stock_remaining"] > 0)
        & (
            df["days_of_stock_remaining"]
            >= df["shelf_life_days"] * config["shelf_life_clearance_multiplier"]
        )
    )

    result = df[stock_on_hand & (low_recent_sales | shelf_life_pressure)].copy()

    result["reason"] = "stock has low recent movement or may outlast shelf life"
    result["trigger"] = (
        "recent_"
        + str(config["dead_stock_recent_days"])
        + "_day_quantity <= "
        + str(config["dead_stock_max_recent_quantity"])
        + " or days_of_stock_remaining >= shelf_life_days * "
        + str(config["shelf_life_clearance_multiplier"])
    )
    result["evidence"] = (
        "stock="
        + result["stock_level"].round(2).astype(str)
        + ", recent_30_day_sales="
        + result["recent_30_day_quantity_sold"].round(2).astype(str)
        + ", shelf_life_days="
        + result["shelf_life_days"].round(2).astype(str)
        + ", days_of_stock="
        + result["days_of_stock_remaining"].round(2).astype(str)
    )

    return _select_output_columns(result)


def identify_high_demand_items(config: dict | None = None) -> pd.DataFrame:
    """Find items with recent sales velocity above a configurable percentile."""
    config = get_config(config)
    df = create_analysis_base()

    velocity = df["recent_daily_sales_velocity"]
    demand_cutoff = velocity[velocity > 0].quantile(
        config["high_demand_percentile"]
    )
    result = df[
        (df["recent_daily_sales_velocity"] > 0)
        & (df["recent_daily_sales_velocity"] >= demand_cutoff)
    ].copy()

    result["reason"] = "recent sales velocity is high relative to this dataset"
    result["trigger"] = (
        "recent_daily_sales_velocity >= percentile_"
        + str(config["high_demand_percentile"])
    )
    result["evidence"] = (
        "recent_daily_velocity="
        + result["recent_daily_sales_velocity"].round(2).astype(str)
        + ", cutoff="
        + round(demand_cutoff, 2).astype(str)
    )

    return _select_output_columns(result)


def identify_slow_moving_items(config: dict | None = None) -> pd.DataFrame:
    """Find stocked items with low sales velocity relative to this dataset."""
    config = get_config(config)
    df = create_analysis_base()

    velocity = df["recent_daily_sales_velocity"]
    movement_cutoff = velocity[velocity > 0].quantile(
        config["slow_moving_percentile"]
    )
    result = df[
        (df["stock_level"] > 0)
        & (
            (df["recent_daily_sales_velocity"] == 0)
            | (df["recent_daily_sales_velocity"] <= movement_cutoff)
        )
    ].copy()

    result["reason"] = "recent sales velocity is low relative to this dataset"
    result["trigger"] = (
        "recent_daily_sales_velocity <= percentile_"
        + str(config["slow_moving_percentile"])
    )
    result["evidence"] = (
        "recent_daily_velocity="
        + result["recent_daily_sales_velocity"].round(2).astype(str)
        + ", cutoff="
        + round(movement_cutoff, 2).astype(str)
    )

    return _select_output_columns(result)


def write_analysis_csv(df: pd.DataFrame, filename: str) -> Path:
    """Write one analysis dataframe to data/processed."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    file_path = PROCESSED_DATA_DIR / filename
    df.to_csv(file_path, index=False)
    return file_path


def build_inventory_analysis(config: dict | None = None) -> dict[str, pd.DataFrame]:
    """Build all inventory analysis outputs and save them as CSV files."""
    analysis_outputs = {
        "low_stock_items": identify_low_stock_items(config),
        "stockout_risk_items": identify_stockout_risk_items(config),
        "overstock_items": identify_overstock_items(config),
        "dead_stock_candidates": identify_dead_stock_candidates(config),
        "high_demand_items": identify_high_demand_items(config),
        "slow_moving_items": identify_slow_moving_items(config),
    }

    for name, df in analysis_outputs.items():
        write_analysis_csv(df, f"{name}.csv")

    return analysis_outputs


if __name__ == "__main__":
    build_inventory_analysis()
