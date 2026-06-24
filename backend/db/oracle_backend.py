"""Oracle implementation of the data access backend.

Selected when ``DATA_BACKEND=oracle``. It returns DataFrames whose columns,
order, and dtypes match exactly what ``CsvBackend`` returned from the raw CSVs,
so no downstream code (services, agents, pages, reports, chatbot) needs to know
the data now comes from Oracle.

Design split:
- ``read_raw``       -> query Oracle (``BZ_MOCK_*`` tables), then normalize.
- ``read_processed`` -> delegate to ``CsvBackend`` (processed datasets are
  app-generated artifacts written to ``data/processed`` by the pipeline; they do
  not live in Oracle and writes are out of scope for this phase).

Normalizations applied to reach CSV parity:
- IDs cast to str (Oracle NUMBER -> "101"); only internal join-consistency matters.
- Dates formatted as 'YYYY-MM-DD' strings.
- suppliers.reliability_score rescaled 0-100 -> 0-1 (matches the 0.90 threshold).
- transactions.transaction_type lower-cased (matches the app's lowercase sets).

Documented gaps filled with explicit defaults / synthesis (see _load_products):
- products.shelf_life_days: no Oracle source -> 0.
- products.reorder_threshold: product-grain absent in Oracle -> 0 (inventory's
  REORDER_POINT is the threshold the app actually uses).
- products.supplier_id: no PRODUCT->SUPPLIER FK in Oracle -> SYNTHESIZED
  deterministically so the supplier table is usable. REPLACE with a real column.
"""

import pandas as pd

from backend.db import paths
from backend.db.config import get_oracle_config
from backend.db.csv_backend import CsvBackend


# Number of suppliers to spread synthesized product->supplier links across.
# Matches BZ_MOCK_SUPPLIER ids 1..5. Replace this synthesis with a real
# PRODUCT.SUPPLIER_ID column when the schema provides one.
_SYNTH_SUPPLIER_COUNT = 5


# Exact output contract per dataset (column order matches the CSV files).
COLUMNS = {
    "products": [
        "product_id", "product_name", "category", "cost_price",
        "selling_price", "shelf_life_days", "reorder_threshold", "supplier_id",
    ],
    "sales": [
        "sale_id", "date", "product_id", "store_id",
        "quantity_sold", "selling_price",
    ],
    "stores": ["store_id", "store_name", "city", "capacity"],
    # Branch dimension with STATE + active flag (home-branch resolution needs
    # CITY+STATE; the "stores" contract intentionally omits state for parity).
    "branches": [
        "branch_id", "branch_name", "city", "state", "active_flg",
    ],
    "suppliers": [
        "supplier_id", "supplier_name", "avg_delivery_days", "reliability_score",
    ],
    "inventory": [
        "product_id", "store_id", "stock_level",
        "reorder_threshold", "last_updated",
    ],
    "transactions": [
        "transaction_id", "date", "transaction_type", "product_id",
        "store_id", "quantity", "source", "remarks",
    ],
    # Customer/order model (real end-customer dimension in Oracle).
    "customers": [
        "customer_id", "customer_nbr", "customer_name", "industry",
        "customer_segment", "city", "state", "contract_tier",
        "credit_limit", "signup_date", "active_flg",
    ],
    "orders": [
        "order_id", "order_nbr", "customer_id", "store_id", "order_date",
        "order_channel", "order_status", "payment_method", "order_total",
    ],
    "order_lines": [
        "order_line_id", "order_id", "product_id", "quantity",
        "unit_selling_price", "unit_cost_price", "discount_amt", "line_total",
    ],
}


# Module-level connection pool, created once on first query.
_POOL = None


def _get_pool():
    """Create (once) and return the oracledb connection pool."""
    global _POOL
    if _POOL is None:
        try:
            import oracledb  # lazy: only needed in oracle mode
        except ImportError as error:
            raise RuntimeError(
                "Oracle driver not installed. Run: pip install oracledb"
            ) from error

        cfg = get_oracle_config()
        _POOL = oracledb.create_pool(
            user=cfg["user"],
            password=cfg["password"],
            dsn=cfg["dsn"],
            min=cfg["pool_min"],
            max=cfg["pool_max"],
            increment=1,
        )
    return _POOL


def _coerce(
    df: pd.DataFrame,
    *,
    str_cols: tuple = (),
    int_cols: tuple = (),
    float_cols: tuple = (),
) -> pd.DataFrame:
    """Cast columns to the dtypes the CSV backend produced."""
    df = df.copy()
    for column in str_cols:
        if column in df.columns:
            df[column] = df[column].fillna("").astype(str)
    for column in int_cols:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0).astype(int)
    for column in float_cols:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").astype(float)
    return df


class OracleBackend:
    """Reads raw datasets from Oracle; delegates processed reads to CSV."""

    def __init__(self):
        self._csv = CsvBackend()

    # -- query helper -------------------------------------------------------
    def _query(self, sql: str) -> pd.DataFrame:
        pool = _get_pool()
        connection = pool.acquire()
        try:
            return pd.read_sql(sql, connection)
        finally:
            connection.close()

    # -- public interface (mirrors CsvBackend) ------------------------------
    def read_raw(self, name: str, *, safe: bool = False) -> pd.DataFrame:
        loader = self._RAW_LOADERS.get(name)
        if loader is None:
            # Unknown raw dataset: mirror CsvBackend's file-not-found behavior.
            if safe:
                return pd.DataFrame()
            raise FileNotFoundError(f"No Oracle mapping for raw dataset: {name}")
        if safe:
            try:
                return loader(self)
            except Exception:
                return pd.DataFrame()
        return loader(self)

    def read_processed(self, name: str, *, safe: bool = False) -> pd.DataFrame:
        # Processed datasets are produced by the pipeline as local CSVs.
        return self._csv.read_processed(name, safe=safe)

    # -- per-dataset loaders ------------------------------------------------
    def _load_products(self) -> pd.DataFrame:
        sql = """
            SELECT
                TO_CHAR(PRODUCT_ID)   AS "product_id",
                PRODUCT_NAME          AS "product_name",
                CATEGORY              AS "category",
                COST_PRICE            AS "cost_price",
                CURRENT_SELLING_PRICE AS "selling_price"
            FROM BZ_MOCK_PRODUCT
            WHERE ACTIVE_FLG = 'Y'
            ORDER BY PRODUCT_ID
        """
        df = self._query(sql)
        # Gaps with no Oracle source:
        df["shelf_life_days"] = 0           # no shelf-life column in Oracle
        df["reorder_threshold"] = 0         # product-grain threshold absent
        # SYNTHESIZED product->supplier link (no FK in Oracle). Deterministic so
        # supplier risk/procurement can function; replace with a real column.
        product_num = pd.to_numeric(df["product_id"], errors="coerce").fillna(0).astype(int)
        df["supplier_id"] = ((product_num % _SYNTH_SUPPLIER_COUNT) + 1).astype(str)
        df = _coerce(
            df,
            str_cols=("product_id", "product_name", "category", "supplier_id"),
            int_cols=("shelf_life_days", "reorder_threshold"),
            float_cols=("cost_price", "selling_price"),
        )
        return df[COLUMNS["products"]]

    def _load_sales(self) -> pd.DataFrame:
        sql = """
            SELECT
                TO_CHAR(SALES_HIST_ID)            AS "sale_id",
                TO_CHAR(SALES_DATE, 'YYYY-MM-DD') AS "date",
                TO_CHAR(PRODUCT_ID)               AS "product_id",
                TO_CHAR(BRANCH_ID)                AS "store_id",
                UNITS_SOLD                        AS "quantity_sold",
                AVG_SELLING_PRICE                 AS "selling_price"
            FROM BZ_MOCK_SALES_HISTORY
            ORDER BY SALES_DATE
        """
        df = self._query(sql)
        df = _coerce(
            df,
            str_cols=("sale_id", "date", "product_id", "store_id"),
            int_cols=("quantity_sold",),
            float_cols=("selling_price",),
        )
        return df[COLUMNS["sales"]]

    def _load_stores(self) -> pd.DataFrame:
        sql = """
            SELECT
                TO_CHAR(b.BRANCH_ID) AS "store_id",
                b.BRANCH_NAME        AS "store_name",
                b.CITY               AS "city",
                NVL(c.CAPACITY, 0)   AS "capacity"
            FROM BZ_MOCK_BRANCH b
            LEFT JOIN BZ_MOCK_BRANCH_CAPACITY c
                ON c.BRANCH_ID = b.BRANCH_ID
            WHERE b.ACTIVE_FLG = 'Y'
            ORDER BY b.BRANCH_ID
        """
        df = self._query(sql)
        df = _coerce(
            df,
            str_cols=("store_id", "store_name", "city"),
            int_cols=("capacity",),
        )
        return df[COLUMNS["stores"]]

    def _load_branches(self) -> pd.DataFrame:
        # Full branch dimension (incl. STATE + ACTIVE_FLG) for home-branch
        # resolution. Unlike "stores", keeps every branch so callers can apply
        # their own active filter, and exposes STATE for CITY+STATE matching.
        sql = """
            SELECT
                TO_CHAR(BRANCH_ID) AS "branch_id",
                BRANCH_NAME        AS "branch_name",
                CITY               AS "city",
                STATE              AS "state",
                ACTIVE_FLG         AS "active_flg"
            FROM BZ_MOCK_BRANCH
            ORDER BY BRANCH_ID
        """
        df = self._query(sql)
        df = _coerce(
            df,
            str_cols=("branch_id", "branch_name", "city", "state", "active_flg"),
        )
        return df[COLUMNS["branches"]]

    def _load_suppliers(self) -> pd.DataFrame:
        sql = """
            SELECT
                TO_CHAR(SUPPLIER_ID) AS "supplier_id",
                SUPPLIER_NAME        AS "supplier_name",
                AVG_DELIVERY_DAYS    AS "avg_delivery_days",
                RELIABILITY_SCORE    AS "reliability_score"
            FROM BZ_MOCK_SUPPLIER
            ORDER BY SUPPLIER_ID
        """
        df = self._query(sql)
        df = _coerce(
            df,
            str_cols=("supplier_id", "supplier_name"),
            int_cols=("avg_delivery_days",),
            float_cols=("reliability_score",),
        )
        # Oracle stores reliability on a 0-100 scale; the app expects 0-1
        # (threshold 0.90). Rescale when values are clearly percentages.
        score = df["reliability_score"]
        if not score.empty and score.max() > 1.5:
            df["reliability_score"] = score / 100.0
        return df[COLUMNS["suppliers"]]

    def _load_inventory(self) -> pd.DataFrame:
        sql = """
            SELECT
                TO_CHAR(PRODUCT_ID)                AS "product_id",
                TO_CHAR(BRANCH_ID)                 AS "store_id",
                ON_HAND_QTY                        AS "stock_level",
                REORDER_POINT                      AS "reorder_threshold",
                TO_CHAR(UPDATED_DTTM, 'YYYY-MM-DD') AS "last_updated"
            FROM BZ_MOCK_INVENTORY
            ORDER BY BRANCH_ID, PRODUCT_ID
        """
        df = self._query(sql)
        df = _coerce(
            df,
            str_cols=("product_id", "store_id", "last_updated"),
            int_cols=("stock_level", "reorder_threshold"),
        )
        return df[COLUMNS["inventory"]]

    def _load_transactions(self) -> pd.DataFrame:
        sql = """
            SELECT
                TO_CHAR(TRANSACTION_ID)                 AS "transaction_id",
                TO_CHAR(TRANSACTION_DATE, 'YYYY-MM-DD') AS "date",
                LOWER(TRANSACTION_TYPE)                 AS "transaction_type",
                TO_CHAR(PRODUCT_ID)                     AS "product_id",
                TO_CHAR(BRANCH_ID)                      AS "store_id",
                QUANTITY                                AS "quantity",
                SOURCE_DESC                             AS "source",
                REMARKS                                 AS "remarks"
            FROM BZ_MOCK_INVENTORY_TRANSACTION
            ORDER BY TRANSACTION_DATE
        """
        df = self._query(sql)
        df = _coerce(
            df,
            str_cols=(
                "transaction_id", "date", "transaction_type",
                "product_id", "store_id", "source", "remarks",
            ),
            int_cols=("quantity",),
        )
        return df[COLUMNS["transactions"]]

    # -- customer / order model ---------------------------------------------
    def _load_customers(self) -> pd.DataFrame:
        sql = """
            SELECT
                TO_CHAR(CUSTOMER_ID)               AS "customer_id",
                CUSTOMER_NBR                       AS "customer_nbr",
                CUSTOMER_NAME                      AS "customer_name",
                INDUSTRY                           AS "industry",
                CUSTOMER_SEGMENT                   AS "customer_segment",
                CITY                               AS "city",
                STATE                              AS "state",
                CONTRACT_TIER                      AS "contract_tier",
                CREDIT_LIMIT_AMT                   AS "credit_limit",
                TO_CHAR(SIGNUP_DATE, 'YYYY-MM-DD') AS "signup_date",
                ACTIVE_FLG                         AS "active_flg"
            FROM BZ_MOCK_CUSTOMER
            ORDER BY CUSTOMER_ID
        """
        df = self._query(sql)
        df = _coerce(
            df,
            str_cols=(
                "customer_id", "customer_nbr", "customer_name", "industry",
                "customer_segment", "city", "state", "contract_tier",
                "signup_date", "active_flg",
            ),
            float_cols=("credit_limit",),
        )
        return df[COLUMNS["customers"]]

    def _load_orders(self) -> pd.DataFrame:
        # BRANCH_ID is exposed as store_id for join-consistency with stores/sales.
        sql = """
            SELECT
                TO_CHAR(ORDER_ID)                 AS "order_id",
                ORDER_NBR                         AS "order_nbr",
                TO_CHAR(CUSTOMER_ID)              AS "customer_id",
                TO_CHAR(BRANCH_ID)                AS "store_id",
                TO_CHAR(ORDER_DATE, 'YYYY-MM-DD') AS "order_date",
                ORDER_CHANNEL                     AS "order_channel",
                ORDER_STATUS                      AS "order_status",
                PAYMENT_METHOD                    AS "payment_method",
                ORDER_TOTAL_AMT                   AS "order_total"
            FROM BZ_MOCK_ORDER_HEADER
            ORDER BY ORDER_DATE
        """
        df = self._query(sql)
        df = _coerce(
            df,
            str_cols=(
                "order_id", "order_nbr", "customer_id", "store_id", "order_date",
                "order_channel", "order_status", "payment_method",
            ),
            float_cols=("order_total",),
        )
        return df[COLUMNS["orders"]]

    def _load_order_lines(self) -> pd.DataFrame:
        # LINE_TOTAL_AMT is the authoritative revenue figure (it bakes in
        # contract pricing/discounts; do not recompute from qty*price).
        sql = """
            SELECT
                TO_CHAR(ORDER_LINE_ID) AS "order_line_id",
                TO_CHAR(ORDER_ID)      AS "order_id",
                TO_CHAR(PRODUCT_ID)    AS "product_id",
                QUANTITY               AS "quantity",
                UNIT_SELLING_PRICE     AS "unit_selling_price",
                UNIT_COST_PRICE        AS "unit_cost_price",
                DISCOUNT_AMT           AS "discount_amt",
                LINE_TOTAL_AMT         AS "line_total"
            FROM BZ_MOCK_ORDER_LINE
            ORDER BY ORDER_ID, ORDER_LINE_ID
        """
        df = self._query(sql)
        df = _coerce(
            df,
            str_cols=("order_line_id", "order_id", "product_id"),
            int_cols=("quantity",),
            float_cols=(
                "unit_selling_price", "unit_cost_price", "discount_amt", "line_total",
            ),
        )
        return df[COLUMNS["order_lines"]]

    _RAW_LOADERS = {
        "products": _load_products,
        "sales": _load_sales,
        "stores": _load_stores,
        "branches": _load_branches,
        "suppliers": _load_suppliers,
        "inventory": _load_inventory,
        "transactions": _load_transactions,
        "customers": _load_customers,
        "orders": _load_orders,
        "order_lines": _load_order_lines,
    }
