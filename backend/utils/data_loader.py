from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


DATA_FILES = {
    "products": {
        "filename": "products.csv",
        "required_columns": [
            "product_id",
            "product_name",
            "category",
            "cost_price",
            "selling_price",
            "shelf_life_days",
            "reorder_threshold",
            "supplier_id",
        ],
        "date_columns": [],
    },
    "sales": {
        "filename": "sales.csv",
        "required_columns": [
            "sale_id",
            "date",
            "product_id",
            "store_id",
            "quantity_sold",
            "selling_price",
        ],
        "date_columns": ["date"],
    },
    "stores": {
        "filename": "stores.csv",
        "required_columns": [
            "store_id",
            "store_name",
            "city",
            "capacity",
        ],
        "date_columns": [],
    },
    "suppliers": {
        "filename": "suppliers.csv",
        "required_columns": [
            "supplier_id",
            "supplier_name",
            "avg_delivery_days",
            "reliability_score",
        ],
        "date_columns": [],
    },
    "inventory": {
        "filename": "inventory.csv",
        "required_columns": [
            "product_id",
            "store_id",
            "stock_level",
            "reorder_threshold",
            "last_updated",
        ],
        "date_columns": ["last_updated"],
    },
    "transactions": {
        "filename": "transactions.csv",
        "required_columns": [
            "transaction_id",
            "date",
            "transaction_type",
            "product_id",
            "store_id",
            "quantity",
            "source",
            "remarks",
        ],
        "date_columns": ["date"],
    },
}


def _load_csv(dataset_name: str) -> pd.DataFrame:
    """Load one raw CSV file and run basic validation."""
    config = DATA_FILES[dataset_name]
    file_path = RAW_DATA_DIR / config["filename"]

    if not file_path.exists():
        raise FileNotFoundError(f"Missing data file: {file_path}")

    df = pd.read_csv(file_path)

    missing_columns = [
        column
        for column in config["required_columns"]
        if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(
            f"{config['filename']} is missing required columns: {missing_columns}"
        )

    required_data = df[config["required_columns"]]
    if required_data.isnull().any().any():
        missing_counts = required_data.isnull().sum()
        missing_counts = missing_counts[missing_counts > 0].to_dict()
        raise ValueError(
            f"{config['filename']} has missing values in required columns: "
            f"{missing_counts}"
        )

    for date_column in config["date_columns"]:
        df[date_column] = pd.to_datetime(df[date_column], errors="raise")

    return df


def load_products() -> pd.DataFrame:
    return _load_csv("products")


def load_sales() -> pd.DataFrame:
    return _load_csv("sales")


def load_stores() -> pd.DataFrame:
    return _load_csv("stores")


def load_suppliers() -> pd.DataFrame:
    return _load_csv("suppliers")


def load_inventory() -> pd.DataFrame:
    return _load_csv("inventory")


def load_transactions() -> pd.DataFrame:
    return _load_csv("transactions")


def load_all_data() -> dict[str, pd.DataFrame]:
    return {
        "products": load_products(),
        "sales": load_sales(),
        "stores": load_stores(),
        "suppliers": load_suppliers(),
        "inventory": load_inventory(),
        "transactions": load_transactions(),
    }
