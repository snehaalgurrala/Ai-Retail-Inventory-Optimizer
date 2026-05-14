from __future__ import annotations

from typing import Any

import pandas as pd


def _number_column(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(0, index=df.index)
    return pd.to_numeric(df[column], errors="coerce").fillna(0)


def _safe_text(value: Any, fallback: str = "") -> str:
    text = str(value or "").strip()
    return text if text else fallback


def _prepare_inventory(
    inventory: pd.DataFrame,
    stores: pd.DataFrame,
    products: pd.DataFrame,
) -> pd.DataFrame:
    if inventory.empty:
        return pd.DataFrame()

    df = inventory.copy()
    df["store_id"] = df.get("store_id", "").astype(str)
    df["product_id"] = df.get("product_id", "").astype(str)
    df["stock_level"] = _number_column(df, "stock_level")
    df["reorder_threshold"] = _number_column(df, "reorder_threshold")

    if not stores.empty and {"store_id", "store_name", "city"}.issubset(stores.columns):
        store_view = stores[["store_id", "store_name", "city"]].copy()
        store_view["store_id"] = store_view["store_id"].astype(str)
        df = df.merge(store_view, on="store_id", how="left")
    else:
        df["store_name"] = df["store_id"]
        df["city"] = ""

    if not products.empty and {"product_id", "product_name", "category"}.issubset(products.columns):
        product_view = products[["product_id", "product_name", "category"]].copy()
        product_view["product_id"] = product_view["product_id"].astype(str)
        df = df.merge(product_view, on="product_id", how="left")
    else:
        df["product_name"] = df["product_id"]
        df["category"] = ""

    df["store_name"] = df["store_name"].fillna(df["store_id"]).astype(str)
    df["city"] = df["city"].fillna("").astype(str)
    df["product_name"] = df["product_name"].fillna(df["product_id"]).astype(str)
    df["category"] = df["category"].fillna("").astype(str)
    return df


def _recent_sales_velocity(sales: pd.DataFrame) -> pd.DataFrame:
    if sales.empty or not {"date", "product_id", "store_id", "quantity_sold"}.issubset(sales.columns):
        return pd.DataFrame(columns=["product_id", "store_id", "recent_30_day_quantity_sold", "recent_daily_sales_velocity"])

    view = sales.copy()
    view["date"] = pd.to_datetime(view["date"], errors="coerce")
    view["product_id"] = view["product_id"].astype(str)
    view["store_id"] = view["store_id"].astype(str)
    view["quantity_sold"] = pd.to_numeric(view["quantity_sold"], errors="coerce").fillna(0)
    view = view[view["date"].notna()].copy()
    if view.empty:
        return pd.DataFrame(columns=["product_id", "store_id", "recent_30_day_quantity_sold", "recent_daily_sales_velocity"])

    latest_date = view["date"].max()
    recent = view[view["date"] >= latest_date - pd.Timedelta(days=30)].copy()
    if recent.empty:
        recent = view.copy()
    grouped = (
        recent.groupby(["product_id", "store_id"], as_index=False)["quantity_sold"]
        .sum()
        .rename(columns={"quantity_sold": "recent_30_day_quantity_sold"})
    )
    grouped["recent_daily_sales_velocity"] = grouped["recent_30_day_quantity_sold"] / 30
    return grouped


def build_transfer_analysis(
    inventory: pd.DataFrame,
    sales: pd.DataFrame,
    stores: pd.DataFrame,
    products: pd.DataFrame,
) -> pd.DataFrame:
    """Compute shortage and surplus signals for product-store transfer analysis."""
    inv = _prepare_inventory(inventory, stores, products)
    if inv.empty:
        return pd.DataFrame()

    velocity = _recent_sales_velocity(sales)
    analysis = inv.merge(velocity, on=["product_id", "store_id"], how="left")
    analysis["recent_30_day_quantity_sold"] = _number_column(analysis, "recent_30_day_quantity_sold")
    analysis["recent_daily_sales_velocity"] = _number_column(analysis, "recent_daily_sales_velocity")
    analysis["days_of_inventory_remaining"] = analysis.apply(
        lambda row: row["stock_level"] / row["recent_daily_sales_velocity"]
        if row["recent_daily_sales_velocity"] > 0
        else 999.0,
        axis=1,
    )
    analysis["shortage_quantity"] = (analysis["reorder_threshold"] - analysis["stock_level"]).clip(lower=0)
    analysis["target_stock_level"] = analysis["reorder_threshold"] + (analysis["recent_daily_sales_velocity"] * 7)
    analysis["target_shortage_quantity"] = (analysis["target_stock_level"] - analysis["stock_level"]).clip(lower=0)
    source_buffer = analysis["reorder_threshold"] + (analysis["recent_daily_sales_velocity"] * 14)
    analysis["surplus_quantity"] = (analysis["stock_level"] - source_buffer).clip(lower=0)
    analysis["shortage_score"] = (
        analysis["target_shortage_quantity"]
        + analysis["recent_daily_sales_velocity"] * 3
        + (analysis["days_of_inventory_remaining"] <= 7).astype(int) * 10
    )
    analysis["surplus_score"] = (
        analysis["surplus_quantity"]
        + (analysis["recent_daily_sales_velocity"] <= 1).astype(int) * 5
        + (analysis["days_of_inventory_remaining"] >= 30).astype(int) * 5
    )
    return analysis


def analyze_transfer_opportunities(
    inventory: pd.DataFrame,
    sales: pd.DataFrame,
    stores: pd.DataFrame,
    products: pd.DataFrame,
    limit: int = 5,
) -> dict[str, pd.DataFrame]:
    """Find source/target transfer opportunities from direct dataframe analytics."""
    analysis = build_transfer_analysis(inventory, sales, stores, products)
    if analysis.empty:
        empty = pd.DataFrame()
        return {"analysis": empty, "opportunities": empty, "shortages": empty, "surpluses": empty}

    shortages = analysis[
        (analysis["target_shortage_quantity"] > 0)
        | (analysis["stock_level"] <= analysis["reorder_threshold"])
        | ((analysis["days_of_inventory_remaining"] <= 7) & (analysis["recent_daily_sales_velocity"] > 0))
    ].copy()
    shortages = shortages.sort_values(["shortage_score", "recent_daily_sales_velocity"], ascending=[False, False])

    surpluses = analysis[
        (analysis["surplus_quantity"] > 0)
        & (analysis["stock_level"] > analysis["reorder_threshold"])
    ].copy()
    surpluses = surpluses.sort_values(["surplus_score", "surplus_quantity"], ascending=[False, False])

    rows: list[dict[str, Any]] = []
    for _, target in shortages.iterrows():
        product_id = _safe_text(target.get("product_id"))
        if not product_id:
            continue
        candidate_sources = surpluses[
            (surpluses["product_id"].astype(str) == product_id)
            & (surpluses["store_id"].astype(str) != str(target.get("store_id", "")))
        ].copy()
        if candidate_sources.empty:
            continue

        source = candidate_sources.iloc[0]
        needed = max(
            float(target.get("shortage_quantity", 0)),
            float(target.get("target_shortage_quantity", 0)),
        )
        suggested_quantity = int(max(0, min(round(needed), round(float(source.get("surplus_quantity", 0))))))
        if suggested_quantity <= 0:
            continue

        priority = "High" if (
            float(target.get("stock_level", 0)) <= float(target.get("reorder_threshold", 0))
            or float(target.get("days_of_inventory_remaining", 999)) <= 7
        ) else "Medium"
        rows.append(
            {
                "product_id": product_id,
                "product_name": _safe_text(target.get("product_name"), product_id),
                "category": _safe_text(target.get("category")),
                "source_store_id": _safe_text(source.get("store_id")),
                "source_store_name": _safe_text(source.get("store_name"), _safe_text(source.get("store_id"))),
                "source_city": _safe_text(source.get("city")),
                "target_store_id": _safe_text(target.get("store_id")),
                "target_store_name": _safe_text(target.get("store_name"), _safe_text(target.get("store_id"))),
                "target_city": _safe_text(target.get("city")),
                "suggested_quantity": suggested_quantity,
                "priority": priority,
                "source_stock": int(round(float(source.get("stock_level", 0)))),
                "source_surplus": int(round(float(source.get("surplus_quantity", 0)))),
                "source_velocity": round(float(source.get("recent_daily_sales_velocity", 0)), 2),
                "target_stock": int(round(float(target.get("stock_level", 0)))),
                "target_threshold": int(round(float(target.get("reorder_threshold", 0)))),
                "target_velocity": round(float(target.get("recent_daily_sales_velocity", 0)), 2),
                "target_days_remaining": round(float(target.get("days_of_inventory_remaining", 0)), 1),
                "opportunity_score": round(
                    float(target.get("shortage_score", 0)) + float(source.get("surplus_score", 0)),
                    2,
                ),
            }
        )

    opportunities = pd.DataFrame(rows)
    if not opportunities.empty:
        opportunities = opportunities.sort_values(
            ["priority", "opportunity_score", "suggested_quantity"],
            ascending=[True, False, False],
        ).head(limit)

    return {
        "analysis": analysis,
        "opportunities": opportunities,
        "shortages": shortages.head(limit),
        "surpluses": surpluses.head(limit),
    }
