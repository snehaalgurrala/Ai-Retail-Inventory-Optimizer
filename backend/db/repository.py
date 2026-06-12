"""Public data access API.

All dataset reads should go through this module. Named loaders return the same
DataFrame structure callers received from ``pd.read_csv`` before — no transforms.

Read semantics:
- ``safe=False`` (default): strict; raises if the source is missing.
- ``safe=True``: returns an empty DataFrame if the source is missing/unreadable.

Choose ``safe`` at each call site to match that site's original behavior.
"""

import pandas as pd

from backend.db.config import SUPPORTED_BACKENDS, get_data_backend
from backend.db.csv_backend import CsvBackend


# Identifier columns that are used as merge/join keys across datasets. They must
# carry a consistent string dtype so frames from different sources join cleanly.
# (CSV ids like "P101" are already strings; Oracle ids like "101" would otherwise
# round-trip through processed CSVs as int64/float64 and break merges.)
_ID_COLUMNS = ("product_id", "store_id", "supplier_id")


def _coerce_id(value):
    if pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, (int,)):
        return str(value)
    return str(value)


def _normalize_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Force known identifier columns to a stable string dtype."""
    if df.empty:
        return df
    for column in _ID_COLUMNS:
        if column in df.columns and df[column].dtype.kind in "iufO":
            if df[column].dtype.kind in "iuf":
                df[column] = df[column].map(_coerce_id)
            elif df[column].isna().any():
                df[column] = df[column].map(_coerce_id)
    return df


def _backend():
    """Return the active backend implementation based on ``DATA_BACKEND``."""
    name = get_data_backend()
    if name == "csv":
        return CsvBackend()
    if name == "oracle":
        # Imported lazily so csv mode never requires the oracle driver.
        from backend.db.oracle_backend import OracleBackend

        return OracleBackend()
    raise NotImplementedError(
        f"DATA_BACKEND='{name}' is not implemented. "
        f"Supported backends: {', '.join(SUPPORTED_BACKENDS)}."
    )


# ---------------------------------------------------------------------------
# Generic accessors (escape hatch for long-tail processed files)
# ---------------------------------------------------------------------------
def load_raw(name: str, *, safe: bool = False) -> pd.DataFrame:
    """Load a raw source table by logical name (e.g. ``"inventory"``)."""
    return _normalize_ids(_backend().read_raw(name, safe=safe))


def load_processed(name: str, *, safe: bool = False) -> pd.DataFrame:
    """Load a processed dataset by logical name (e.g. ``"recommendations"``)."""
    return _normalize_ids(_backend().read_processed(name, safe=safe))


# ---------------------------------------------------------------------------
# Raw source tables (system of record)
# ---------------------------------------------------------------------------
def load_products(*, safe: bool = False) -> pd.DataFrame:
    return load_raw("products", safe=safe)


def load_sales(*, safe: bool = False) -> pd.DataFrame:
    return load_raw("sales", safe=safe)


def load_stores(*, safe: bool = False) -> pd.DataFrame:
    return load_raw("stores", safe=safe)


def load_suppliers(*, safe: bool = False) -> pd.DataFrame:
    return load_raw("suppliers", safe=safe)


def load_inventory(*, safe: bool = False) -> pd.DataFrame:
    return load_raw("inventory", safe=safe)


def load_transactions(*, safe: bool = False) -> pd.DataFrame:
    return load_raw("transactions", safe=safe)


def load_customers(*, safe: bool = False) -> pd.DataFrame:
    """Load the end-customer dimension (``BZ_MOCK_CUSTOMER``)."""
    return load_raw("customers", safe=safe)


def load_orders(*, safe: bool = False) -> pd.DataFrame:
    """Load order headers (``BZ_MOCK_ORDER_HEADER``)."""
    return load_raw("orders", safe=safe)


def load_order_lines(*, safe: bool = False) -> pd.DataFrame:
    """Load order line items (``BZ_MOCK_ORDER_LINE``)."""
    return load_raw("order_lines", safe=safe)


def load_all_raw() -> dict[str, pd.DataFrame]:
    """Load all raw tables. Drop-in for ``data_processor.load_raw_data``.

    Key order matches the historical loader: inventory, products, sales, stores,
    suppliers, transactions.
    """
    return {
        "inventory": load_inventory(),
        "products": load_products(),
        "sales": load_sales(),
        "stores": load_stores(),
        "suppliers": load_suppliers(),
        "transactions": load_transactions(),
    }


# ---------------------------------------------------------------------------
# Processed: derived datasets (built by data_processor)
# ---------------------------------------------------------------------------
def load_current_inventory(*, safe: bool = False) -> pd.DataFrame:
    return load_processed("current_inventory", safe=safe)


def load_sales_summary(*, safe: bool = False) -> pd.DataFrame:
    return load_processed("sales_summary", safe=safe)


def load_product_performance(*, safe: bool = False) -> pd.DataFrame:
    return load_processed("product_performance", safe=safe)


def load_store_inventory_summary(*, safe: bool = False) -> pd.DataFrame:
    return load_processed("store_inventory_summary", safe=safe)


# ---------------------------------------------------------------------------
# Processed: inventory analysis outputs (built by inventory_analyzer)
# ---------------------------------------------------------------------------
def load_low_stock_items(*, safe: bool = False) -> pd.DataFrame:
    return load_processed("low_stock_items", safe=safe)


def load_stockout_risk_items(*, safe: bool = False) -> pd.DataFrame:
    return load_processed("stockout_risk_items", safe=safe)


def load_overstock_items(*, safe: bool = False) -> pd.DataFrame:
    return load_processed("overstock_items", safe=safe)


def load_dead_stock_candidates(*, safe: bool = False) -> pd.DataFrame:
    return load_processed("dead_stock_candidates", safe=safe)


def load_high_demand_items(*, safe: bool = False) -> pd.DataFrame:
    return load_processed("high_demand_items", safe=safe)


def load_slow_moving_items(*, safe: bool = False) -> pd.DataFrame:
    return load_processed("slow_moving_items", safe=safe)


# ---------------------------------------------------------------------------
# Processed: agent / transactional outputs
# ---------------------------------------------------------------------------
def load_recommendations(*, safe: bool = False) -> pd.DataFrame:
    return load_processed("recommendations", safe=safe)


def load_customer_orders(*, safe: bool = False) -> pd.DataFrame:
    """Customer orders (the project's only order dataset)."""
    return load_processed("customer_orders", safe=safe)
