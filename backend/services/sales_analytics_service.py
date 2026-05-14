from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class SalesFilters:
    store_ids: tuple[str, ...] = ()
    categories: tuple[str, ...] = ()
    product_ids: tuple[str, ...] = ()
    start_date: pd.Timestamp | None = None
    end_date: pd.Timestamp | None = None


def _numeric(series: pd.Series | Any, index: pd.Index | None = None) -> pd.Series:
    if isinstance(series, pd.Series):
        return pd.to_numeric(series, errors="coerce").fillna(0)
    return pd.Series(0, index=index)


def prepare_sales_dataset(
    sales: pd.DataFrame,
    stores: pd.DataFrame,
    products: pd.DataFrame,
) -> pd.DataFrame:
    """Return sales.csv enriched with store, city, product, category, and revenue."""
    if sales.empty:
        return pd.DataFrame()

    df = sales.copy()
    df["store_id"] = df.get("store_id", "").astype(str)
    df["product_id"] = df.get("product_id", "").astype(str)
    df["date"] = pd.to_datetime(df.get("date"), errors="coerce")
    df["quantity_sold"] = _numeric(df.get("quantity_sold"), df.index)
    df["selling_price"] = _numeric(df.get("selling_price"), df.index)
    df["sales_value"] = df["quantity_sold"] * df["selling_price"]

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
    df["category"] = df["category"].fillna("Uncategorized").astype(str)
    df["branch_label"] = df.apply(
        lambda row: (
            f"{row['city']} - {row['store_name']}"
            if str(row.get("city", "")).strip()
            else str(row.get("store_name", row.get("store_id", "")))
        ),
        axis=1,
    )
    return df


def apply_sales_filters(df: pd.DataFrame, filters: SalesFilters) -> pd.DataFrame:
    filtered = df.copy()
    if filtered.empty:
        return filtered

    if filters.store_ids:
        filtered = filtered[filtered["store_id"].astype(str).isin(filters.store_ids)].copy()
    if filters.categories:
        filtered = filtered[filtered["category"].astype(str).isin(filters.categories)].copy()
    if filters.product_ids:
        filtered = filtered[filtered["product_id"].astype(str).isin(filters.product_ids)].copy()
    if filters.start_date is not None:
        filtered = filtered[filtered["date"] >= pd.Timestamp(filters.start_date)].copy()
    if filters.end_date is not None:
        end_date = pd.Timestamp(filters.end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        filtered = filtered[filtered["date"] <= end_date].copy()
    return filtered


def branch_options(stores: pd.DataFrame, sales_df: pd.DataFrame) -> list[dict[str, str]]:
    if stores.empty:
        store_ids = sorted(sales_df.get("store_id", pd.Series(dtype=str)).dropna().astype(str).unique())
        return [{"label": store_id, "store_id": store_id, "city": ""} for store_id in store_ids]

    view = stores.copy()
    view["store_id"] = view["store_id"].astype(str)
    view["store_name"] = view.get("store_name", view["store_id"]).fillna(view["store_id"]).astype(str)
    view["city"] = view.get("city", "").fillna("").astype(str)
    options = []
    for _, row in view.sort_values(["city", "store_name"]).iterrows():
        label = f"{row['city']} - {row['store_name']}" if row["city"] else row["store_name"]
        options.append({"label": label, "store_id": row["store_id"], "city": row["city"]})
    return options


def overview_metrics(filtered: pd.DataFrame, all_sales: pd.DataFrame) -> dict[str, Any]:
    if filtered.empty:
        return {
            "total_sales": 0.0,
            "total_orders": 0,
            "top_product": "No sales",
            "least_product": "No sales",
            "revenue_trend": "No trend",
            "fastest_moving_product": "No sales",
        }

    product_totals = product_performance(filtered)
    top_product = product_totals.iloc[0]["product_name"] if not product_totals.empty else "No sales"
    least_product = (
        product_totals.sort_values(["quantity_sold", "product_name"], ascending=[True, True]).iloc[0]["product_name"]
        if not product_totals.empty
        else "No sales"
    )

    daily = trend_data(filtered, frequency="D")
    revenue_trend = "Flat"
    if len(daily) >= 2:
        recent = float(daily.tail(min(7, len(daily)))["sales_value"].mean())
        baseline = float(daily.head(min(7, len(daily)))["sales_value"].mean())
        if recent > baseline * 1.05:
            revenue_trend = "Up"
        elif recent < baseline * 0.95:
            revenue_trend = "Down"

    velocity = sales_velocity(filtered, all_sales)
    fastest = velocity.iloc[0]["product_name"] if not velocity.empty else top_product
    return {
        "total_sales": float(filtered["sales_value"].sum()),
        "total_orders": int(filtered["sale_id"].nunique() if "sale_id" in filtered.columns else len(filtered)),
        "top_product": str(top_product),
        "least_product": str(least_product),
        "revenue_trend": revenue_trend,
        "fastest_moving_product": str(fastest),
    }


def branch_sales_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["branch_label", "city", "store_name", "quantity_sold", "sales_value", "order_count"])
    return (
        df.groupby(["store_id", "branch_label", "city", "store_name"], as_index=False)
        .agg(
            quantity_sold=("quantity_sold", "sum"),
            sales_value=("sales_value", "sum"),
            order_count=("sale_id", "nunique") if "sale_id" in df.columns else ("quantity_sold", "count"),
        )
        .sort_values("sales_value", ascending=False)
    )


def trend_data(df: pd.DataFrame, frequency: str = "D") -> pd.DataFrame:
    if df.empty or "date" not in df.columns:
        return pd.DataFrame(columns=["date", "quantity_sold", "sales_value", "order_count"])
    view = df.dropna(subset=["date"]).copy()
    if view.empty:
        return pd.DataFrame(columns=["date", "quantity_sold", "sales_value", "order_count"])
    period = view["date"].dt.to_period(frequency)
    grouped = (
        view.assign(period=period.dt.start_time)
        .groupby("period", as_index=False)
        .agg(
            quantity_sold=("quantity_sold", "sum"),
            sales_value=("sales_value", "sum"),
            order_count=("sale_id", "nunique") if "sale_id" in view.columns else ("quantity_sold", "count"),
        )
        .rename(columns={"period": "date"})
        .sort_values("date")
    )
    return grouped


def product_performance(df: pd.DataFrame, limit: int | None = 10) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["product_id", "product_name", "category", "quantity_sold", "sales_value", "order_count"])
    grouped = (
        df.groupby(["product_id", "product_name", "category"], as_index=False)
        .agg(
            quantity_sold=("quantity_sold", "sum"),
            sales_value=("sales_value", "sum"),
            order_count=("sale_id", "nunique") if "sale_id" in df.columns else ("quantity_sold", "count"),
        )
        .sort_values(["quantity_sold", "sales_value"], ascending=[False, False])
    )
    return grouped.head(limit) if limit else grouped


def category_sales(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["category", "quantity_sold", "sales_value"])
    return (
        df.groupby("category", as_index=False)
        .agg(quantity_sold=("quantity_sold", "sum"), sales_value=("sales_value", "sum"))
        .sort_values("sales_value", ascending=False)
    )


def sales_velocity(filtered: pd.DataFrame, all_sales: pd.DataFrame) -> pd.DataFrame:
    if filtered.empty or "date" not in filtered.columns:
        return pd.DataFrame()
    latest_date = all_sales["date"].max() if not all_sales.empty and "date" in all_sales.columns else filtered["date"].max()
    recent = filtered[filtered["date"] >= latest_date - pd.Timedelta(days=30)].copy()
    if recent.empty:
        recent = filtered.copy()
    grouped = product_performance(recent, limit=None)
    grouped["daily_sales_velocity"] = grouped["quantity_sold"] / 30
    return grouped.sort_values(["daily_sales_velocity", "quantity_sold"], ascending=[False, False])


def inventory_sales_comparison(
    filtered_sales: pd.DataFrame,
    inventory: pd.DataFrame,
    products: pd.DataFrame,
) -> pd.DataFrame:
    if filtered_sales.empty:
        return pd.DataFrame()
    sales_velocity_df = sales_velocity(filtered_sales, filtered_sales)
    if sales_velocity_df.empty:
        return pd.DataFrame()

    if inventory.empty:
        comparison = sales_velocity_df.copy()
        comparison["stock_level"] = 0
        comparison["reorder_threshold"] = 0
        comparison["days_of_stock_remaining"] = 0
        return comparison

    inv = inventory.copy()
    inv["product_id"] = inv["product_id"].astype(str)
    inv["store_id"] = inv["store_id"].astype(str)
    inv["stock_level"] = _numeric(inv.get("stock_level"), inv.index)
    inv["reorder_threshold"] = _numeric(inv.get("reorder_threshold"), inv.index)
    store_ids = filtered_sales["store_id"].dropna().astype(str).unique().tolist()
    if store_ids:
        inv = inv[inv["store_id"].isin(store_ids)].copy()
    inv_totals = (
        inv.groupby("product_id", as_index=False)
        .agg(stock_level=("stock_level", "sum"), reorder_threshold=("reorder_threshold", "sum"))
    )
    comparison = sales_velocity_df.merge(inv_totals, on="product_id", how="left")
    comparison["stock_level"] = comparison["stock_level"].fillna(0)
    comparison["reorder_threshold"] = comparison["reorder_threshold"].fillna(0)
    comparison["days_of_stock_remaining"] = comparison.apply(
        lambda row: row["stock_level"] / row["daily_sales_velocity"] if row["daily_sales_velocity"] > 0 else 0,
        axis=1,
    )
    return comparison.sort_values("daily_sales_velocity", ascending=False).head(12)


def branch_comparison(
    df: pd.DataFrame,
    inventory: pd.DataFrame,
    selected_store_ids: tuple[str, ...],
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    view = df[df["store_id"].astype(str).isin(selected_store_ids)].copy() if selected_store_ids else df.copy()
    if view.empty:
        return pd.DataFrame()

    summary = branch_sales_summary(view)
    top_products = (
        view.groupby(["store_id", "product_name"], as_index=False)["quantity_sold"]
        .sum()
        .sort_values(["store_id", "quantity_sold"], ascending=[True, False])
        .drop_duplicates("store_id")
        .rename(columns={"product_name": "top_product", "quantity_sold": "top_product_units"})
    )
    summary = summary.merge(top_products, on="store_id", how="left")

    if not inventory.empty:
        inv = inventory.copy()
        inv["store_id"] = inv["store_id"].astype(str)
        inv["stock_level"] = _numeric(inv.get("stock_level"), inv.index)
        inv["reorder_threshold"] = _numeric(inv.get("reorder_threshold"), inv.index)
        low_stock = (
            inv[inv["stock_level"] <= inv["reorder_threshold"]]
            .groupby("store_id", as_index=False)
            .size()
            .rename(columns={"size": "low_stock_risk"})
        )
        summary = summary.merge(low_stock, on="store_id", how="left")
    else:
        summary["low_stock_risk"] = 0
    summary["low_stock_risk"] = summary["low_stock_risk"].fillna(0).astype(int)
    return summary[
        [
            "branch_label",
            "total_sales" if "total_sales" in summary.columns else "sales_value",
            "order_count",
            "top_product",
            "top_product_units",
            "low_stock_risk",
        ]
    ].rename(columns={"sales_value": "total_sales"})


def generate_sales_insights(
    filtered: pd.DataFrame,
    all_filtered: pd.DataFrame,
    comparison: pd.DataFrame,
) -> list[str]:
    insights: list[str] = []
    branch_summary = branch_sales_summary(all_filtered)
    if not branch_summary.empty:
        top_branch = branch_summary.iloc[0]
        top_category = category_sales(all_filtered)
        category_text = ""
        if not top_category.empty:
            category_text = f", especially in {top_category.iloc[0]['category']} products"
        insights.append(
            f"{top_branch['store_name']} currently shows the strongest sales activity{category_text}."
        )

    products = product_performance(filtered, limit=None)
    if not products.empty:
        strongest = products.iloc[0]
        weakest = products.sort_values(["quantity_sold", "product_name"], ascending=[True, True]).iloc[0]
        insights.append(
            f"{strongest['product_name']} is the fastest seller in the selected scope with {int(strongest['quantity_sold']):,} units sold."
        )
        insights.append(
            f"{weakest['product_name']} has slower movement and may need pricing, placement, or reorder review."
        )

    if not comparison.empty and "low_stock_risk" in comparison.columns:
        risk_row = comparison.sort_values("low_stock_risk", ascending=False).iloc[0]
        if int(risk_row.get("low_stock_risk", 0)) > 0:
            insights.append(
                f"{risk_row['branch_label']} has {int(risk_row['low_stock_risk'])} low-stock risk items while serving active demand."
            )

    if not insights:
        insights.append("There is not enough sales activity in the selected filters to generate a confident branch insight.")
    return insights[:4]
