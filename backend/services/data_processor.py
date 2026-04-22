from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


RAW_FILES = {
    "inventory": "inventory.csv",
    "products": "products.csv",
    "sales": "sales.csv",
    "stores": "stores.csv",
    "suppliers": "suppliers.csv",
    "transactions": "transactions.csv",
}


def load_raw_data() -> dict[str, pd.DataFrame]:
    """Load all raw retail CSV files."""
    return {
        name: pd.read_csv(RAW_DATA_DIR / filename)
        for name, filename in RAW_FILES.items()
    }


def _to_datetime(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Convert a date column safely when it exists."""
    df = df.copy()
    if column in df.columns:
        df[column] = pd.to_datetime(df[column], errors="coerce")
    return df


def _fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values without changing the meaning of known data."""
    df = df.copy()

    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            df[column] = df[column].fillna(0)
        elif pd.api.types.is_datetime64_any_dtype(df[column]):
            continue
        else:
            df[column] = df[column].fillna("")

    return df


def _merge_product_and_store_metadata(
    inventory: pd.DataFrame,
    products: pd.DataFrame,
    stores: pd.DataFrame,
) -> pd.DataFrame:
    """Add product and store details to an inventory dataframe."""
    inventory = inventory.copy()

    if "reorder_threshold" in inventory.columns:
        inventory = inventory.rename(
            columns={"reorder_threshold": "inventory_reorder_threshold"}
        )

    df = inventory.merge(products, on="product_id", how="left")
    df = df.merge(stores, on="store_id", how="left")

    return _fill_missing_values(df)


def _inventory_is_current_snapshot(
    inventory: pd.DataFrame,
    transactions: pd.DataFrame,
) -> bool:
    """Check whether inventory.csv can be used as the current stock snapshot."""
    required_columns = {"product_id", "store_id", "stock_level", "last_updated"}
    if inventory.empty or not required_columns.issubset(inventory.columns):
        return False

    inventory_dates = pd.to_datetime(inventory["last_updated"], errors="coerce")
    if inventory_dates.notna().sum() == 0:
        return False

    if "date" not in transactions.columns or transactions.empty:
        return True

    transaction_dates = pd.to_datetime(transactions["date"], errors="coerce")
    if transaction_dates.notna().sum() == 0:
        return True

    return inventory_dates.max() >= transaction_dates.max()


def _derive_inventory_from_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    """Build current stock by applying transaction quantities."""
    transactions = _to_datetime(transactions, "date")
    transactions = transactions.copy()
    transactions["quantity"] = pd.to_numeric(
        transactions.get("quantity", 0),
        errors="coerce",
    ).fillna(0)

    required_columns = ["product_id", "store_id", "quantity", "date"]
    for column in required_columns:
        if column not in transactions.columns:
            transactions[column] = ""

    if "transaction_type" not in transactions.columns:
        transactions["transaction_type"] = ""

    transactions = transactions.sort_values("date")
    snapshot_rows = transactions[
        transactions["transaction_type"].eq("inventory_snapshot")
    ]

    if not snapshot_rows.empty:
        latest_snapshot_date = snapshot_rows["date"].max()
        latest_snapshot = snapshot_rows[
            snapshot_rows["date"].eq(latest_snapshot_date)
        ].copy()
        latest_snapshot = latest_snapshot.rename(columns={"quantity": "stock_level"})
        latest_snapshot = latest_snapshot[
            ["product_id", "store_id", "stock_level", "date"]
        ].rename(columns={"date": "last_updated"})

        later_transactions = transactions[transactions["date"] > latest_snapshot_date]
        if later_transactions.empty:
            return _fill_missing_values(latest_snapshot)

        starting_stock = latest_snapshot[["product_id", "store_id", "stock_level"]]
    else:
        later_transactions = transactions
        starting_stock = pd.DataFrame(
            columns=["product_id", "store_id", "stock_level"]
        )

    outbound_types = {"sale", "transfer_out", "damage", "waste", "return_to_supplier"}
    inbound_types = {"purchase", "restock", "transfer_in", "return", "inventory_in"}

    later_transactions = later_transactions.copy()
    later_transactions["signed_quantity"] = 0
    later_transactions.loc[
        later_transactions["transaction_type"].isin(inbound_types),
        "signed_quantity",
    ] = later_transactions["quantity"]
    later_transactions.loc[
        later_transactions["transaction_type"].isin(outbound_types),
        "signed_quantity",
    ] = -later_transactions["quantity"]

    movements = (
        later_transactions.groupby(["product_id", "store_id"], as_index=False)
        .agg(
            stock_change=("signed_quantity", "sum"),
            last_updated=("date", "max"),
        )
    )

    df = starting_stock.merge(movements, on=["product_id", "store_id"], how="outer")
    df["stock_level"] = pd.to_numeric(df["stock_level"], errors="coerce").fillna(0)
    df["stock_change"] = pd.to_numeric(df["stock_change"], errors="coerce").fillna(0)
    df["stock_level"] = df["stock_level"] + df["stock_change"]
    df = df.drop(columns=["stock_change"])

    return _fill_missing_values(df)


def current_inventory_df(
    inventory: pd.DataFrame | None = None,
    products: pd.DataFrame | None = None,
    stores: pd.DataFrame | None = None,
    transactions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create the current inventory dataset with product and store metadata."""
    raw_data = load_raw_data()
    inventory = raw_data["inventory"] if inventory is None else inventory.copy()
    products = raw_data["products"] if products is None else products.copy()
    stores = raw_data["stores"] if stores is None else stores.copy()
    transactions = (
        raw_data["transactions"] if transactions is None else transactions.copy()
    )

    inventory = _to_datetime(inventory, "last_updated")

    if _inventory_is_current_snapshot(inventory, transactions):
        current_inventory = inventory.copy()
    else:
        current_inventory = _derive_inventory_from_transactions(transactions)

    current_inventory["stock_level"] = pd.to_numeric(
        current_inventory.get("stock_level", 0),
        errors="coerce",
    ).fillna(0)

    return _merge_product_and_store_metadata(current_inventory, products, stores)


def sales_summary_df(sales: pd.DataFrame | None = None) -> pd.DataFrame:
    """Create product-store sales totals, revenue, and recent sales metrics."""
    raw_data = load_raw_data()
    sales = raw_data["sales"] if sales is None else sales.copy()
    sales = _to_datetime(sales, "date")

    sales["quantity_sold"] = pd.to_numeric(
        sales.get("quantity_sold", 0),
        errors="coerce",
    ).fillna(0)
    sales["selling_price"] = pd.to_numeric(
        sales.get("selling_price", 0),
        errors="coerce",
    ).fillna(0)
    sales["revenue"] = sales["quantity_sold"] * sales["selling_price"]

    summary = (
        sales.groupby(["product_id", "store_id"], as_index=False)
        .agg(
            total_quantity_sold=("quantity_sold", "sum"),
            total_revenue=("revenue", "sum"),
            last_sale_date=("date", "max"),
        )
    )

    summary["product_total_quantity_sold"] = summary.groupby("product_id")[
        "total_quantity_sold"
    ].transform("sum")
    summary["store_total_quantity_sold"] = summary.groupby("store_id")[
        "total_quantity_sold"
    ].transform("sum")

    if sales["date"].notna().any():
        latest_sale_date = sales["date"].max()
        recent_7_day_sales = sales[
            sales["date"] >= latest_sale_date - pd.Timedelta(days=7)
        ]
        recent_30_day_sales = sales[
            sales["date"] >= latest_sale_date - pd.Timedelta(days=30)
        ]

        recent_7_day_summary = (
            recent_7_day_sales.groupby(["product_id", "store_id"], as_index=False)
            .agg(recent_7_day_quantity_sold=("quantity_sold", "sum"))
        )
        recent_30_day_summary = (
            recent_30_day_sales.groupby(["product_id", "store_id"], as_index=False)
            .agg(recent_30_day_quantity_sold=("quantity_sold", "sum"))
        )

        summary = summary.merge(
            recent_7_day_summary,
            on=["product_id", "store_id"],
            how="left",
        )
        summary = summary.merge(
            recent_30_day_summary,
            on=["product_id", "store_id"],
            how="left",
        )
    else:
        summary["recent_7_day_quantity_sold"] = 0
        summary["recent_30_day_quantity_sold"] = 0

    return _fill_missing_values(summary)


def product_performance_df(
    products: pd.DataFrame | None = None,
    sales_summary: pd.DataFrame | None = None,
    current_inventory: pd.DataFrame | None = None,
    suppliers: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create product-level performance data with sales, inventory, and suppliers."""
    raw_data = load_raw_data()
    products = raw_data["products"] if products is None else products.copy()
    suppliers = raw_data["suppliers"] if suppliers is None else suppliers.copy()
    sales_summary = (
        sales_summary_df(raw_data["sales"])
        if sales_summary is None
        else sales_summary.copy()
    )
    current_inventory = (
        current_inventory_df(
            raw_data["inventory"],
            raw_data["products"],
            raw_data["stores"],
            raw_data["transactions"],
        )
        if current_inventory is None
        else current_inventory.copy()
    )

    product_sales = (
        sales_summary.groupby("product_id", as_index=False)
        .agg(
            total_quantity_sold=("total_quantity_sold", "sum"),
            total_revenue=("total_revenue", "sum"),
            recent_7_day_quantity_sold=("recent_7_day_quantity_sold", "sum"),
            recent_30_day_quantity_sold=("recent_30_day_quantity_sold", "sum"),
            last_sale_date=("last_sale_date", "max"),
        )
    )

    product_inventory = (
        current_inventory.groupby("product_id", as_index=False)
        .agg(current_stock_level=("stock_level", "sum"))
    )

    df = products.merge(product_sales, on="product_id", how="left")
    df = df.merge(product_inventory, on="product_id", how="left")

    if "supplier_id" in df.columns and "supplier_id" in suppliers.columns:
        df = df.merge(suppliers, on="supplier_id", how="left")

    return _fill_missing_values(df)


def store_inventory_summary_df(
    current_inventory: pd.DataFrame | None = None,
    stores: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create store-level inventory totals and capacity comparison."""
    raw_data = load_raw_data()
    stores = raw_data["stores"] if stores is None else stores.copy()
    current_inventory = (
        current_inventory_df(
            raw_data["inventory"],
            raw_data["products"],
            raw_data["stores"],
            raw_data["transactions"],
        )
        if current_inventory is None
        else current_inventory.copy()
    )

    summary = (
        current_inventory.groupby("store_id", as_index=False)
        .agg(total_inventory=("stock_level", "sum"))
    )

    df = stores.merge(summary, on="store_id", how="left")

    if "capacity" in df.columns:
        df["capacity"] = pd.to_numeric(df["capacity"], errors="coerce").fillna(0)
        df["remaining_capacity"] = df["capacity"] - df["total_inventory"].fillna(0)
        df["capacity_used_percent"] = 0.0
        has_capacity = df["capacity"] > 0
        df.loc[has_capacity, "capacity_used_percent"] = (
            df.loc[has_capacity, "total_inventory"] / df.loc[has_capacity, "capacity"]
        ) * 100

    return _fill_missing_values(df)


def write_processed_csv(df: pd.DataFrame, filename: str) -> Path:
    """Write one processed dataframe to data/processed."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    file_path = PROCESSED_DATA_DIR / filename
    df.to_csv(file_path, index=False)
    return file_path


def build_processed_datasets() -> dict[str, pd.DataFrame]:
    """Build all processed datasets and write them as CSV files."""
    raw_data = load_raw_data()

    current_inventory = current_inventory_df(
        raw_data["inventory"],
        raw_data["products"],
        raw_data["stores"],
        raw_data["transactions"],
    )
    sales_summary = sales_summary_df(raw_data["sales"])
    product_performance = product_performance_df(
        raw_data["products"],
        sales_summary,
        current_inventory,
        raw_data["suppliers"],
    )
    store_inventory_summary = store_inventory_summary_df(
        current_inventory,
        raw_data["stores"],
    )

    processed_data = {
        "current_inventory": current_inventory,
        "sales_summary": sales_summary,
        "product_performance": product_performance,
        "store_inventory_summary": store_inventory_summary,
    }

    for name, df in processed_data.items():
        write_processed_csv(df, f"{name}.csv")

    return processed_data


if __name__ == "__main__":
    build_processed_datasets()
