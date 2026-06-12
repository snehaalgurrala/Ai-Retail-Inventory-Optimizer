"""Transactional Oracle write layer (Phase 3).

Used only when ``DATA_BACKEND=oracle``. Every public operation runs inside a
single transaction: changes commit only on full success and roll back on any
error, so inventory state and the movement ledger never drift apart.

Authoritative writes supported (these have Oracle tables):
- inventory levels      -> BZ_MOCK_INVENTORY (ON_HAND_QTY / AVAILABLE_QTY)
- movement ledger       -> BZ_MOCK_INVENTORY_TRANSACTION  (Task 4)
- sales records         -> BZ_MOCK_SALES_HISTORY
- product price         -> BZ_MOCK_PRODUCT.CURRENT_SELLING_PRICE

IDs (product_id, store_id) arrive as strings ("101", "1") from the read layer
and are bound as Oracle NUMBER. New surrogate keys use MAX(id)+1 within the
transaction (single-writer mock; mirrors the old CSV id scheme).
"""

from contextlib import contextmanager
from datetime import datetime

from backend.db import oracle_backend


@contextmanager
def transaction():
    """Yield a cursor in a transaction; commit on success, rollback on error."""
    pool = oracle_backend._get_pool()
    connection = pool.acquire()
    cursor = connection.cursor()
    try:
        yield cursor
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        cursor.close()
        connection.close()


def _next_id(cursor, table: str, id_column: str) -> int:
    cursor.execute(f"SELECT NVL(MAX({id_column}), 0) + 1 FROM {table}")
    return int(cursor.fetchone()[0])


def _adjust_inventory(
    cursor,
    product_id,
    store_id,
    delta: int,
    when: datetime,
    *,
    allow_create: bool = False,
    threshold: int = 0,
) -> int:
    """Apply a signed delta to on-hand stock; returns the new on-hand level.

    Raises ValueError if the row is missing (and cannot be created) or the
    change would drive stock negative. Row is locked FOR UPDATE.
    """
    product = int(product_id)
    branch = int(store_id)
    delta = int(delta)

    cursor.execute(
        "SELECT INVENTORY_ID, ON_HAND_QTY, AVAILABLE_QTY "
        "FROM BZ_MOCK_INVENTORY "
        "WHERE PRODUCT_ID = :p AND BRANCH_ID = :b FOR UPDATE",
        {"p": product, "b": branch},
    )
    row = cursor.fetchone()

    if row is None:
        if not (allow_create and delta > 0):
            raise ValueError(
                f"No inventory row for product {product_id} at store {store_id}."
            )
        inventory_id = _next_id(cursor, "BZ_MOCK_INVENTORY", "INVENTORY_ID")
        cursor.execute(
            "INSERT INTO BZ_MOCK_INVENTORY "
            "(INVENTORY_ID, BRANCH_ID, PRODUCT_ID, ON_HAND_QTY, RESERVED_QTY, "
            " AVAILABLE_QTY, REORDER_POINT, SAFETY_STOCK, REORDER_QTY, "
            " LAST_RESTOCK_DATE, UPDATED_DTTM) "
            "VALUES (:i, :b, :p, :q, 0, :q, :rp, 0, 0, :w, :w)",
            {"i": inventory_id, "b": branch, "p": product, "q": delta,
             "rp": int(threshold), "w": when},
        )
        return delta

    inventory_id, on_hand, available = row
    on_hand = int(on_hand or 0)
    available = int(available or 0)
    new_on_hand = on_hand + delta
    if new_on_hand < 0:
        raise ValueError(
            f"Inventory would go negative for product {product_id} at store "
            f"{store_id} (have {on_hand}, requested change {delta})."
        )
    new_available = max(available + delta, 0)
    cursor.execute(
        "UPDATE BZ_MOCK_INVENTORY "
        "SET ON_HAND_QTY = :oh, AVAILABLE_QTY = :av, UPDATED_DTTM = :w, "
        "    LAST_SOLD_DATE = CASE WHEN :d < 0 THEN :w ELSE LAST_SOLD_DATE END "
        "WHERE INVENTORY_ID = :i",
        {"oh": new_on_hand, "av": new_available, "w": when, "d": delta, "i": inventory_id},
    )
    return new_on_hand


def _log_movement(
    cursor,
    product_id,
    store_id,
    txn_type: str,
    quantity: int,
    source: str,
    remarks: str,
    when: datetime,
) -> int:
    """Insert one row into BZ_MOCK_INVENTORY_TRANSACTION; returns its id."""
    transaction_id = _next_id(
        cursor, "BZ_MOCK_INVENTORY_TRANSACTION", "TRANSACTION_ID"
    )
    cursor.execute(
        "INSERT INTO BZ_MOCK_INVENTORY_TRANSACTION "
        "(TRANSACTION_ID, PRODUCT_ID, BRANCH_ID, TRANSACTION_DATE, "
        " TRANSACTION_TYPE, QUANTITY, SOURCE_DESC, REMARKS) "
        "VALUES (:t, :p, :b, :w, :ty, :q, :s, :r)",
        {
            "t": transaction_id,
            "p": int(product_id),
            "b": int(store_id),
            "w": when,
            "ty": str(txn_type).upper(),
            "q": int(quantity),
            "s": (source or "")[:100],
            "r": (remarks or "")[:300],
        },
    )
    return transaction_id


def _insert_sale(
    cursor,
    product_id,
    store_id,
    quantity: int,
    unit_price: float,
    when: datetime,
) -> int:
    """Insert one sale into BZ_MOCK_SALES_HISTORY; returns its id."""
    sales_id = _next_id(cursor, "BZ_MOCK_SALES_HISTORY", "SALES_HIST_ID")
    gross = round(int(quantity) * float(unit_price), 2)
    cursor.execute(
        "INSERT INTO BZ_MOCK_SALES_HISTORY "
        "(SALES_HIST_ID, SALES_DATE, BRANCH_ID, PRODUCT_ID, UNITS_SOLD, "
        " GROSS_SALES_AMT, DISCOUNT_AMT, NET_SALES_AMT, AVG_SELLING_PRICE) "
        "VALUES (:i, :w, :b, :p, :q, :g, 0, :g, :up)",
        {"i": sales_id, "w": when, "b": int(store_id), "p": int(product_id),
         "q": int(quantity), "g": gross, "up": round(float(unit_price), 2)},
    )
    return sales_id


# ---------------------------------------------------------------------------
# Public operations (each atomic)
# ---------------------------------------------------------------------------
def place_order(product_id, store_id, quantity: int, unit_price: float,
                when: datetime | None = None) -> dict:
    """Customer order: decrement stock + log SALE movement + record the sale.

    All three happen in one transaction. Returns new_stock, transaction_id,
    sale_id. Raises on insufficient stock (transaction rolled back).
    """
    when = when or datetime.now()
    with transaction() as cursor:
        new_stock = _adjust_inventory(cursor, product_id, store_id, -int(quantity), when)
        transaction_id = _log_movement(
            cursor, product_id, store_id, "SALE", quantity,
            "customer_order", "Customer order", when,
        )
        sale_id = _insert_sale(cursor, product_id, store_id, quantity, unit_price, when)
    return {"new_stock": new_stock, "transaction_id": transaction_id, "sale_id": sale_id}


def apply_movements(movements: list[dict]) -> list[dict]:
    """Apply a set of inventory movements atomically (all-or-nothing).

    Each movement: product_id, store_id, delta, txn_type, quantity (optional),
    source (optional), remarks (optional), allow_create (optional),
    threshold (optional). Returns [{new_stock, transaction_id}, ...].
    """
    when = datetime.now()
    results = []
    with transaction() as cursor:
        for move in movements:
            new_stock = _adjust_inventory(
                cursor,
                move["product_id"],
                move["store_id"],
                int(move["delta"]),
                when,
                allow_create=bool(move.get("allow_create", False)),
                threshold=int(move.get("threshold", 0)),
            )
            transaction_id = _log_movement(
                cursor,
                move["product_id"],
                move["store_id"],
                move["txn_type"],
                int(move.get("quantity", abs(int(move["delta"])))),
                move.get("source", "recommendation_execution"),
                move.get("remarks", ""),
                when,
            )
            results.append({"new_stock": new_stock, "transaction_id": transaction_id})
    return results


def set_product_price(product_id, new_price: float) -> None:
    """Update a product's current selling price (transactional)."""
    with transaction() as cursor:
        cursor.execute(
            "UPDATE BZ_MOCK_PRODUCT SET CURRENT_SELLING_PRICE = :p "
            "WHERE PRODUCT_ID = :id",
            {"p": round(float(new_price), 2), "id": int(product_id)},
        )
