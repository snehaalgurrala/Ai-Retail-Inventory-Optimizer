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


def find_exclusive_store_items(
    inventory: pd.DataFrame,
    products: pd.DataFrame,
    stores: pd.DataFrame,
) -> pd.DataFrame:
    """Return products with positive stock in exactly one store."""
    output_columns = [
        "product_id",
        "product_name",
        "category",
        "exclusive_store_id",
        "exclusive_store_name",
        "city",
        "available_quantity",
        "selling_price",
        "business_note",
        "alternative_recommendation",
        "unavailable_store_names",
        "inventory_store_rows",
    ]
    inv = _prepare_inventory(inventory, stores, products)
    if inv.empty:
        return pd.DataFrame(columns=output_columns)

    inv["stock_level"] = _number_column(inv, "stock_level")
    positive = inv[inv["stock_level"] > 0].copy()
    if positive.empty:
        return pd.DataFrame(columns=output_columns)

    store_count = (
        positive.groupby("product_id")["store_id"]
        .nunique()
        .reset_index(name="positive_store_count")
    )
    exclusive_products = store_count[store_count["positive_store_count"] == 1]["product_id"].astype(str)
    exclusive_rows = positive[positive["product_id"].astype(str).isin(exclusive_products)].copy()
    if exclusive_rows.empty:
        return pd.DataFrame(columns=output_columns)

    product_store_counts = (
        inv.groupby("product_id")["store_id"]
        .nunique()
        .reset_index(name="inventory_store_rows")
    )
    exclusive_rows = exclusive_rows.merge(product_store_counts, on="product_id", how="left")
    exclusive_rows["available_quantity"] = exclusive_rows["stock_level"].round().astype(int)
    exclusive_rows["exclusive_store_id"] = exclusive_rows["store_id"].astype(str)
    exclusive_rows["exclusive_store_name"] = exclusive_rows["store_name"].fillna(exclusive_rows["store_id"]).astype(str)
    exclusive_rows["business_note"] = exclusive_rows.apply(
        lambda row: (
            f"{row['product_name']} is available only at {row['exclusive_store_name']} "
            f"with {int(row['available_quantity'])} units."
        ),
        axis=1,
    )
    exclusive_rows["alternative_recommendation"] = exclusive_rows.apply(
        lambda row: (
            f"Promote {row['product_name']} as an alternative {row['category']} option "
            f"from {row['exclusive_store_name']} when similar products are low elsewhere."
        ),
        axis=1,
    )
    store_names_by_id = (
        stores.assign(store_id=stores["store_id"].astype(str))
        .set_index("store_id")["store_name"]
        .astype(str)
        .to_dict()
        if not stores.empty and {"store_id", "store_name"}.issubset(stores.columns)
        else {}
    )
    positive_store_sets = (
        positive.groupby("product_id")["store_id"]
        .apply(lambda values: {str(value) for value in values})
        .to_dict()
    )
    all_store_ids = set(stores["store_id"].astype(str).tolist()) if not stores.empty and "store_id" in stores.columns else set()
    exclusive_rows["unavailable_store_names"] = exclusive_rows["product_id"].astype(str).map(
        lambda product_id: " | ".join(
            store_names_by_id.get(store_id, store_id)
            for store_id in sorted(all_store_ids - positive_store_sets.get(product_id, set()))
        )
    )

    for column in output_columns:
        if column not in exclusive_rows.columns:
            exclusive_rows[column] = ""

    return exclusive_rows[output_columns].sort_values(
        ["available_quantity", "product_name"],
        ascending=[False, True],
    ).reset_index(drop=True)


def _build_low_stock_scope(
    inventory: pd.DataFrame,
    products: pd.DataFrame,
    stores: pd.DataFrame,
    low_stock_items: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if low_stock_items is not None and not low_stock_items.empty:
        low_stock = low_stock_items.copy()
        low_stock["product_id"] = low_stock.get("product_id", "").astype(str)
        low_stock["store_id"] = low_stock.get("store_id", "").astype(str)
        if "stock_level" not in low_stock.columns and "current_quantity" in low_stock.columns:
            low_stock["stock_level"] = low_stock["current_quantity"]
        if "effective_reorder_threshold" not in low_stock.columns and "reorder_threshold" in low_stock.columns:
            low_stock["effective_reorder_threshold"] = low_stock["reorder_threshold"]
        return low_stock

    inv = _prepare_inventory(inventory, stores, products)
    if inv.empty:
        return pd.DataFrame()
    inv["stock_level"] = _number_column(inv, "stock_level")
    inv["effective_reorder_threshold"] = _number_column(inv, "reorder_threshold")
    return inv[inv["stock_level"] <= inv["effective_reorder_threshold"]].copy()


def find_alternative_products_for_low_stock(
    inventory: pd.DataFrame,
    products: pd.DataFrame,
    stores: pd.DataFrame,
    low_stock_items: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Find same-category alternatives for low-stock or unavailable products."""
    inv = _prepare_inventory(inventory, stores, products)
    low_stock = _build_low_stock_scope(inventory, products, stores, low_stock_items)
    if inv.empty or low_stock.empty:
        return pd.DataFrame()

    inv["stock_level"] = _number_column(inv, "stock_level")
    low_stock["stock_level"] = _number_column(low_stock, "stock_level")
    if "category" not in low_stock.columns and not products.empty:
        product_view = products[["product_id", "product_name", "category"]].copy()
        product_view["product_id"] = product_view["product_id"].astype(str)
        low_stock = low_stock.merge(product_view, on="product_id", how="left")
    if "store_name" not in low_stock.columns and not stores.empty:
        store_view = stores[["store_id", "store_name", "city"]].copy()
        store_view["store_id"] = store_view["store_id"].astype(str)
        low_stock = low_stock.merge(store_view, on="store_id", how="left")

    exclusive = find_exclusive_store_items(inventory, products, stores)
    exclusive_keys = {
        (str(row.get("product_id", "")), str(row.get("exclusive_store_id", "")))
        for _, row in exclusive.iterrows()
    }

    available = inv[inv["stock_level"] > 0].copy()
    rows: list[dict[str, Any]] = []
    for _, low_row in low_stock.iterrows():
        low_product_id = _safe_text(low_row.get("product_id"))
        category = _safe_text(low_row.get("category"))
        if not low_product_id or not category:
            continue

        candidates = available[
            (available["category"].astype(str) == category)
            & (available["product_id"].astype(str) != low_product_id)
        ].copy()
        if candidates.empty:
            continue

        candidates["available_quantity"] = candidates["stock_level"].round().astype(int)
        candidates["is_exclusive"] = candidates.apply(
            lambda row: (str(row.get("product_id", "")), str(row.get("store_id", ""))) in exclusive_keys,
            axis=1,
        )
        candidates = candidates.sort_values(
            ["is_exclusive", "available_quantity", "product_name"],
            ascending=[False, False, True],
        )

        for _, alt in candidates.iterrows():
            is_exclusive = bool(alt.get("is_exclusive", False))
            low_stock_store = _safe_text(low_row.get("store_name"), _safe_text(low_row.get("store_id")))
            alt_store = _safe_text(alt.get("store_name"), _safe_text(alt.get("store_id")))
            rows.append(
                {
                    "low_stock_product_id": low_product_id,
                    "low_stock_product": _safe_text(low_row.get("product_name"), low_product_id),
                    "low_stock_store_id": _safe_text(low_row.get("store_id")),
                    "low_stock_store": low_stock_store,
                    "alternative_product_id": _safe_text(alt.get("product_id")),
                    "alternative_product": _safe_text(alt.get("product_name"), _safe_text(alt.get("product_id"))),
                    "alternative_store_id": _safe_text(alt.get("store_id")),
                    "alternative_store": alt_store,
                    "available_quantity": int(alt.get("available_quantity", 0)),
                    "category": category,
                    "is_exclusive": is_exclusive,
                    "reason": (
                        f"{alt.get('product_name')} is exclusively available at {alt_store} "
                        f"with {int(alt.get('available_quantity', 0))} units."
                        if is_exclusive
                        else f"{alt.get('product_name')} is available in the same {category} category at {alt_store}."
                    ),
                    "priority": "high" if is_exclusive else "medium",
                }
            )

    alternatives = pd.DataFrame(rows)
    if alternatives.empty:
        return alternatives

    alternatives["_exclusive_rank"] = alternatives["is_exclusive"].astype(int)
    alternatives = alternatives.sort_values(
        ["_exclusive_rank", "available_quantity", "alternative_product"],
        ascending=[False, False, True],
    ).drop(columns=["_exclusive_rank"])
    return alternatives.reset_index(drop=True)
