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
from backend.db.config import get_inventory_scope


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


def _draw_down_network(cursor, product_id, quantity: int, when: datetime) -> int:
    """Decrement on-hand for a product across ALL branches (pooled fulfillment).

    Used when ``INVENTORY_SCOPE=network`` so an order draws from the network pool
    rather than a single branch — keeping order placement consistent with the
    network-wide stock shown in the catalogue. Locks the product's inventory rows
    (FOR UPDATE), draws from the most-stocked branch first, and rolls back via
    ValueError if the network pool cannot cover the quantity. Returns the
    product's remaining network on-hand.
    """
    product = int(product_id)
    qty = int(quantity)
    cursor.execute(
        "SELECT INVENTORY_ID, ON_HAND_QTY, AVAILABLE_QTY "
        "FROM BZ_MOCK_INVENTORY WHERE PRODUCT_ID = :p "
        "ORDER BY ON_HAND_QTY DESC FOR UPDATE",
        {"p": product},
    )
    rows = cursor.fetchall()
    total_on_hand = sum(int(r[1] or 0) for r in rows)
    if qty > total_on_hand:
        raise ValueError(
            f"Insufficient network stock for product {product_id} "
            f"(have {total_on_hand}, requested {qty})."
        )
    remaining = qty
    for inventory_id, on_hand, available in rows:
        if remaining <= 0:
            break
        on_hand = int(on_hand or 0)
        take = min(on_hand, remaining)
        if take <= 0:
            continue
        cursor.execute(
            "UPDATE BZ_MOCK_INVENTORY "
            "SET ON_HAND_QTY = :oh, AVAILABLE_QTY = :av, UPDATED_DTTM = :w, "
            "    LAST_SOLD_DATE = :w "
            "WHERE INVENTORY_ID = :i",
            {
                "oh": on_hand - take,
                "av": max(int(available or 0) - take, 0),
                "w": when,
                "i": inventory_id,
            },
        )
        remaining -= take
    return total_on_hand - qty


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


def _order_line_amounts(cursor, product_id, quantity: int) -> tuple[float, float, float]:
    """Resolve authoritative per-unit prices from the catalogue for one line.

    Reads CURRENT_SELLING_PRICE / COST_PRICE straight from BZ_MOCK_PRODUCT so the
    persisted line totals never depend on stale figures passed in from the UI.
    Returns (unit_selling_price, unit_cost_price, line_total).
    """
    cursor.execute(
        "SELECT NVL(CURRENT_SELLING_PRICE, 0), NVL(COST_PRICE, 0) "
        "FROM BZ_MOCK_PRODUCT WHERE PRODUCT_ID = :p",
        {"p": int(product_id)},
    )
    row = cursor.fetchone()
    if row is None:
        raise ValueError(f"Unknown product {product_id}.")
    selling = round(float(row[0]), 2)
    cost = round(float(row[1]), 2)
    line_total = round(int(quantity) * selling, 2)
    return selling, cost, line_total


def default_branch_id() -> int:
    """Lowest active branch — the simulator's single fulfillment branch."""
    with transaction() as cursor:
        cursor.execute(
            "SELECT MIN(BRANCH_ID) FROM BZ_MOCK_BRANCH WHERE ACTIVE_FLG = 'Y'"
        )
        row = cursor.fetchone()
        return int(row[0]) if row and row[0] is not None else 1


def place_customer_order(
    customer_id,
    items: list[dict],
    *,
    branch_id=None,
    channel: str = "ORDER_SIMULATOR",
    status: str = "PLACED",
    payment_method: str = "SIMULATION",
    when: datetime | None = None,
) -> dict:
    """Capture a customer order and draw down inventory, all in one transaction.

    Creates one ORDER_HEADER + its ORDER_LINE rows AND reduces on-hand stock in
    BZ_MOCK_INVENTORY (ON_HAND_QTY / AVAILABLE_QTY) at the order's branch for
    every line. Stock is validated as it is decremented — if any line would drive
    a branch below zero the whole transaction rolls back, so no partial order or
    inventory drift is ever persisted. No stock-movement ledger rows are written
    and no downstream agents are triggered.

    ``items`` is a list of ``{"product_id": ..., "quantity": ...}``. When
    ``branch_id`` is omitted the lowest active branch is used. Returns the new
    order's ids/totals plus the resulting per-product stock levels.
    """
    if not items:
        raise ValueError("Cannot place an order with no line items.")
    when = when or datetime.now()
    with transaction() as cursor:
        if branch_id is None:
            cursor.execute(
                "SELECT MIN(BRANCH_ID) FROM BZ_MOCK_BRANCH WHERE ACTIVE_FLG = 'Y'"
            )
            row = cursor.fetchone()
            branch_id = int(row[0]) if row and row[0] is not None else 1
        branch_id = int(branch_id)

        order_id = _next_id(cursor, "BZ_MOCK_ORDER_HEADER", "ORDER_ID")
        order_nbr = f"ORD-{when:%Y%m%d}-{order_id:04d}"

        # Price every line first (authoritative catalogue prices) so the header
        # total is exact before the header row is inserted.
        priced: list[tuple] = []
        order_total = 0.0
        for item in items:
            quantity = int(item["quantity"])
            if quantity <= 0:
                continue
            selling, cost, line_total = _order_line_amounts(
                cursor, item["product_id"], quantity
            )
            order_total += line_total
            priced.append((int(item["product_id"]), quantity, selling, cost, line_total))
        if not priced:
            raise ValueError("Cannot place an order with no positive-quantity lines.")
        order_total = round(order_total, 2)

        cursor.execute(
            "INSERT INTO BZ_MOCK_ORDER_HEADER "
            "(ORDER_ID, ORDER_NBR, CUSTOMER_ID, BRANCH_ID, ORDER_DATE, "
            " ORDER_CHANNEL, ORDER_STATUS, PAYMENT_METHOD, ORDER_TOTAL_AMT) "
            "VALUES (:oid, :nbr, :cid, :bid, :od, :ch, :sta, :pay, :tot)",
            {
                "oid": order_id, "nbr": order_nbr, "cid": int(customer_id),
                "bid": branch_id, "od": when, "ch": channel[:30],
                "sta": status[:30], "pay": payment_method[:30], "tot": order_total,
            },
        )

        # Network scope draws each line from the pooled stock across all branches
        # so placement matches the network-wide inventory shown in the catalogue;
        # branch scope decrements only the order's fulfilling branch. Either way
        # ``new_stock`` is the figure to display after the order.
        network = get_inventory_scope() == "network"

        line_id = _next_id(cursor, "BZ_MOCK_ORDER_LINE", "ORDER_LINE_ID")
        inventory: list[dict] = []
        for product_id, quantity, selling, cost, line_total in priced:
            cursor.execute(
                "INSERT INTO BZ_MOCK_ORDER_LINE "
                "(ORDER_LINE_ID, ORDER_ID, PRODUCT_ID, QUANTITY, "
                " UNIT_SELLING_PRICE, UNIT_COST_PRICE, DISCOUNT_AMT, LINE_TOTAL_AMT) "
                "VALUES (:lid, :oid, :pid, :q, :usp, :ucp, 0, :lt)",
                {
                    "lid": line_id, "oid": order_id, "pid": product_id, "q": quantity,
                    "usp": selling, "ucp": cost, "lt": line_total,
                },
            )
            line_id += 1
            # Validated, locking decrement. Raises (and rolls back the whole
            # order) if stock is insufficient.
            if network:
                new_stock = _draw_down_network(cursor, product_id, quantity, when)
            else:
                new_stock = _adjust_inventory(
                    cursor, product_id, branch_id, -quantity, when
                )
            inventory.append({
                "product_id": str(product_id),
                "quantity": quantity,
                "new_stock": new_stock,
            })

    return {
        "order_id": order_id,
        "order_nbr": order_nbr,
        "order_date": when,
        "branch_id": branch_id,
        "order_total": order_total,
        "line_count": len(priced),
        "inventory": inventory,
    }


def set_product_price(product_id, new_price: float) -> None:
    """Update a product's current selling price (transactional)."""
    with transaction() as cursor:
        cursor.execute(
            "UPDATE BZ_MOCK_PRODUCT SET CURRENT_SELLING_PRICE = :p "
            "WHERE PRODUCT_ID = :id",
            {"p": round(float(new_price), 2), "id": int(product_id)},
        )
