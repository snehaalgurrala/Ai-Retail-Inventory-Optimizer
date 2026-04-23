from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException

from backend.agents.orchestrator_agent import run_all_agents
from backend.services.data_processor import (
    PROCESSED_DATA_DIR,
    build_processed_datasets,
)
from backend.services.inventory_analyzer import build_inventory_analysis


app = FastAPI(title="AI Retail Inventory Optimizer API")


def dataframe_to_records(df: pd.DataFrame) -> list[dict]:
    """Convert a dataframe to clean JSON records."""
    if df.empty:
        return []

    clean_df = df.copy()
    clean_df = clean_df.replace([float("inf"), float("-inf")], None)
    clean_df = clean_df.astype(object).where(pd.notnull(clean_df), None)

    for column in clean_df.columns:
        if pd.api.types.is_datetime64_any_dtype(clean_df[column]):
            clean_df[column] = clean_df[column].astype(str)

    return clean_df.to_dict(orient="records")


def read_processed_csv(filename: str) -> pd.DataFrame:
    """Read a processed CSV file or raise a clear API error."""
    file_path = PROCESSED_DATA_DIR / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Missing processed file: {file_path}")

    return pd.read_csv(file_path)


def ensure_processed_data() -> None:
    """Create processed datasets if they are not available yet."""
    required_files = [
        "current_inventory.csv",
        "sales_summary.csv",
        "product_performance.csv",
        "store_inventory_summary.csv",
    ]
    missing_files = [
        filename
        for filename in required_files
        if not (PROCESSED_DATA_DIR / filename).exists()
    ]

    if missing_files:
        build_processed_datasets()


def ensure_analysis_outputs() -> None:
    """Create analyzer outputs if they are not available yet."""
    ensure_processed_data()

    required_files = [
        "low_stock_items.csv",
        "stockout_risk_items.csv",
        "overstock_items.csv",
        "dead_stock_candidates.csv",
        "high_demand_items.csv",
        "slow_moving_items.csv",
    ]
    missing_files = [
        filename
        for filename in required_files
        if not (PROCESSED_DATA_DIR / filename).exists()
    ]

    if missing_files:
        build_inventory_analysis()


def ensure_recommendations() -> None:
    """Create recommendations if the final output is not available yet."""
    ensure_analysis_outputs()
    if not (PROCESSED_DATA_DIR / "recommendations.csv").exists():
        run_all_agents()


def get_processed_path(filename: str) -> Path:
    return PROCESSED_DATA_DIR / filename


@app.get("/")
def read_root():
    return {"message": "AI Retail Inventory Optimizer API is running"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/data/summary")
def data_summary():
    """Return simple row counts and key totals for processed datasets."""
    try:
        ensure_recommendations()

        current_inventory = read_processed_csv("current_inventory.csv")
        sales_summary = read_processed_csv("sales_summary.csv")
        product_performance = read_processed_csv("product_performance.csv")
        store_inventory_summary = read_processed_csv(
            "store_inventory_summary.csv"
        )
        recommendations = read_processed_csv("recommendations.csv")

        total_inventory = 0
        if "stock_level" in current_inventory.columns:
            total_inventory = int(
                pd.to_numeric(
                    current_inventory["stock_level"],
                    errors="coerce",
                ).fillna(0).sum()
            )

        total_revenue = 0
        if "total_revenue" in sales_summary.columns:
            total_revenue = float(
                pd.to_numeric(
                    sales_summary["total_revenue"],
                    errors="coerce",
                ).fillna(0).sum()
            )

        pending_recommendations = 0
        if "status" in recommendations.columns:
            pending_recommendations = int(
                recommendations["status"].fillna("").eq("pending").sum()
            )

        return {
            "processed_files": {
                "current_inventory": str(get_processed_path("current_inventory.csv")),
                "sales_summary": str(get_processed_path("sales_summary.csv")),
                "product_performance": str(
                    get_processed_path("product_performance.csv")
                ),
                "store_inventory_summary": str(
                    get_processed_path("store_inventory_summary.csv")
                ),
                "recommendations": str(get_processed_path("recommendations.csv")),
            },
            "row_counts": {
                "current_inventory": len(current_inventory),
                "sales_summary": len(sales_summary),
                "product_performance": len(product_performance),
                "store_inventory_summary": len(store_inventory_summary),
                "recommendations": len(recommendations),
            },
            "totals": {
                "inventory_units": total_inventory,
                "sales_revenue": total_revenue,
                "pending_recommendations": pending_recommendations,
            },
        }
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Could not build data summary: {error}",
        ) from error


@app.get("/inventory/current")
def current_inventory():
    """Return the current product-store inventory dataset."""
    try:
        ensure_processed_data()
        df = read_processed_csv("current_inventory.csv")
        return {"count": len(df), "data": dataframe_to_records(df)}
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Could not load current inventory: {error}",
        ) from error


@app.get("/sales/summary")
def sales_summary():
    """Return processed sales summary rows."""
    try:
        ensure_processed_data()
        df = read_processed_csv("sales_summary.csv")
        return {"count": len(df), "data": dataframe_to_records(df)}
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Could not load sales summary: {error}",
        ) from error


@app.get("/recommendations")
def recommendations():
    """Return the latest recommendation output."""
    try:
        ensure_recommendations()
        df = read_processed_csv("recommendations.csv")
        return {"count": len(df), "data": dataframe_to_records(df)}
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Could not load recommendations: {error}",
        ) from error


@app.post("/agents/run")
def run_agents():
    """Run all internal agents and return the combined recommendations."""
    try:
        recommendations_df = run_all_agents()
        return {
            "status": "completed",
            "count": len(recommendations_df),
            "data": dataframe_to_records(recommendations_df),
        }
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Could not run agents: {error}",
        ) from error
