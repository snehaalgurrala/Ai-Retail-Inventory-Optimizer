from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
STORES_FILE = RAW_DATA_DIR / "stores.csv"
INVENTORY_FILE = RAW_DATA_DIR / "inventory.csv"
PRODUCTS_FILE = RAW_DATA_DIR / "products.csv"
SALES_FILE = RAW_DATA_DIR / "sales.csv"
TRANSACTIONS_FILE = RAW_DATA_DIR / "transactions.csv"
ORDERS_FILE = PROCESSED_DATA_DIR / "customer_orders.csv"

STORE_COLUMNS = ["store_id", "store_name", "city", "capacity"]
INVENTORY_COLUMNS = [
    "product_id",
    "store_id",
    "stock_level",
    "reorder_threshold",
    "last_updated",
]
PRODUCT_COLUMNS = [
    "product_id",
    "product_name",
    "category",
    "cost_price",
    "selling_price",
    "shelf_life_days",
    "reorder_threshold",
    "supplier_id",
]
SALES_COLUMNS = [
    "sale_id",
    "date",
    "product_id",
    "store_id",
    "quantity_sold",
    "selling_price",
]
TRANSACTION_COLUMNS = [
    "transaction_id",
    "date",
    "transaction_type",
    "product_id",
    "store_id",
    "quantity",
    "source",
    "remarks",
]
ORDER_COLUMNS = [
    "order_id",
    "order_date",
    "store_id",
    "store_name",
    "city",
    "product_id",
    "product_name",
    "quantity_ordered",
    "unit_price",
    "total_amount",
    "status",
]


def _empty_df(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def _read_csv(file_path: Path, columns: list[str]) -> pd.DataFrame:
    """Read a CSV safely, returning an empty dataframe with the expected schema."""
    if not file_path.exists():
        return _empty_df(columns)

    try:
        df = pd.read_csv(file_path)
    except Exception:
        return _empty_df(columns)

    for column in columns:
        if column not in df.columns:
            df[column] = ""

    return df[columns].copy()


def _write_csv(df: pd.DataFrame, file_path: Path, columns: list[str]) -> None:
    """Write a CSV through a temporary file to reduce corruption risk."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    writable_df = df.copy()
    for column in columns:
        if column not in writable_df.columns:
            writable_df[column] = ""

    temp_path = file_path.with_suffix(f"{file_path.suffix}.tmp")
    writable_df[columns].to_csv(temp_path, index=False)
    temp_path.replace(file_path)


def _next_prefixed_id(existing_df: pd.DataFrame, column: str, prefix: str) -> str:
    """Generate the next sequential ID using a prefix."""
    if existing_df.empty or column not in existing_df.columns:
        return f"{prefix}000001" if prefix != "TXN" else f"{prefix}0000001"

    values = existing_df[column].fillna("").astype(str)
    numeric_values = pd.to_numeric(
        values.str.replace(prefix, "", regex=False),
        errors="coerce",
    ).dropna()
    next_number = int(numeric_values.max()) + 1 if not numeric_values.empty else 1

    if prefix == "TXN":
        return f"{prefix}{str(next_number).zfill(7)}"
    return f"{prefix}{str(next_number).zfill(6)}"


def _load_required_data() -> dict[str, pd.DataFrame]:
    """Load all datasets needed by the order service."""
    return {
        "stores": _read_csv(STORES_FILE, STORE_COLUMNS),
        "inventory": _read_csv(INVENTORY_FILE, INVENTORY_COLUMNS),
        "products": _read_csv(PRODUCTS_FILE, PRODUCT_COLUMNS),
        "sales": _read_csv(SALES_FILE, SALES_COLUMNS),
        "transactions": _read_csv(TRANSACTIONS_FILE, TRANSACTION_COLUMNS),
        "orders": _read_csv(ORDERS_FILE, ORDER_COLUMNS),
    }


def get_available_products_by_store(store_id: str) -> pd.DataFrame:
    """Return inventory rows for one store merged with product information."""
    data = _load_required_data()
    inventory = data["inventory"]
    products = data["products"]

    if inventory.empty:
        return pd.DataFrame()

    store_inventory = inventory[
        inventory["store_id"].astype(str).eq(str(store_id))
    ].copy()
    if store_inventory.empty:
        return pd.DataFrame()

    store_inventory["stock_level"] = pd.to_numeric(
        store_inventory["stock_level"],
        errors="coerce",
    ).fillna(0)
    products["selling_price"] = pd.to_numeric(
        products["selling_price"],
        errors="coerce",
    ).fillna(0)

    view = store_inventory.merge(products, on="product_id", how="left")
    preferred_columns = [
        "product_id",
        "product_name",
        "category",
        "stock_level",
        "selling_price",
        "store_id",
        "reorder_threshold",
        "last_updated",
    ]
    available_columns = [column for column in preferred_columns if column in view.columns]
    return view[available_columns].sort_values(["product_name", "product_id"])


def validate_order(store_id: str, product_id: str, quantity: int) -> dict:
    """Validate whether an order can be placed for the given product and store."""
    if quantity <= 0:
        return {
            "success": False,
            "message": "Quantity must be greater than zero.",
            "order_data": {},
        }

    data = _load_required_data()
    stores = data["stores"]
    inventory = data["inventory"]
    products = data["products"]

    matching_store = stores[stores["store_id"].astype(str).eq(str(store_id))]
    if matching_store.empty:
        return {
            "success": False,
            "message": "Selected store was not found.",
            "order_data": {},
        }

    inventory_mask = (
        inventory["store_id"].astype(str).eq(str(store_id))
        & inventory["product_id"].astype(str).eq(str(product_id))
    )
    if not inventory_mask.any():
        return {
            "success": False,
            "message": "Selected product is not available in the chosen store.",
            "order_data": {},
        }

    matching_product = products[products["product_id"].astype(str).eq(str(product_id))]
    if matching_product.empty:
        return {
            "success": False,
            "message": "Product details could not be loaded.",
            "order_data": {},
        }

    available_quantity = int(
        pd.to_numeric(
            inventory.loc[inventory_mask, "stock_level"],
            errors="coerce",
        ).fillna(0).iloc[0]
    )
    if quantity > available_quantity:
        return {
            "success": False,
            "message": "Insufficient stock available",
            "order_data": {
                "store_id": str(store_id),
                "product_id": str(product_id),
                "available_quantity": available_quantity,
                "requested_quantity": int(quantity),
            },
        }

    store_row = matching_store.iloc[0]
    product_row = matching_product.iloc[0]
    unit_price = float(
        pd.to_numeric(product_row.get("selling_price", 0), errors="coerce")
    )

    return {
        "success": True,
        "message": "Order validation successful.",
        "order_data": {
            "store_id": str(store_id),
            "store_name": str(store_row.get("store_name", "")),
            "city": str(store_row.get("city", "")),
            "product_id": str(product_id),
            "product_name": str(product_row.get("product_name", "")),
            "category": str(product_row.get("category", "")),
            "quantity_ordered": int(quantity),
            "available_quantity": available_quantity,
            "unit_price": unit_price,
            "total_amount": float(quantity * unit_price),
        },
    }


def place_order(store_id: str, product_id: str, quantity: int) -> dict:
    """Place one order, update source CSVs, and return a structured response."""
    validation = validate_order(store_id, product_id, quantity)
    if not validation["success"]:
        return validation

    data = _load_required_data()
    stores = data["stores"]
    inventory = data["inventory"]
    sales = data["sales"]
    transactions = data["transactions"]
    orders = data["orders"]

    order_data = validation["order_data"].copy()
    order_date = datetime.now().isoformat(timespec="seconds")
    order_id = _next_prefixed_id(orders, "order_id", "ORD")
    sale_id = _next_prefixed_id(sales, "sale_id", "SALE")
    transaction_id = _next_prefixed_id(transactions, "transaction_id", "TXN")

    inventory_mask = (
        inventory["store_id"].astype(str).eq(str(store_id))
        & inventory["product_id"].astype(str).eq(str(product_id))
    )
    current_stock = int(
        pd.to_numeric(
            inventory.loc[inventory_mask, "stock_level"],
            errors="coerce",
        ).fillna(0).iloc[0]
    )
    new_stock_level = current_stock - int(quantity)
    inventory.loc[inventory_mask, "stock_level"] = new_stock_level
    inventory.loc[inventory_mask, "last_updated"] = order_date.split("T")[0]

    order_record = {
        "order_id": order_id,
        "order_date": order_date,
        "store_id": order_data["store_id"],
        "store_name": order_data["store_name"],
        "city": order_data["city"],
        "product_id": order_data["product_id"],
        "product_name": order_data["product_name"],
        "quantity_ordered": int(quantity),
        "unit_price": float(order_data["unit_price"]),
        "total_amount": float(order_data["total_amount"]),
        "status": "placed",
    }
    sales_record = {
        "sale_id": sale_id,
        "date": order_date.split("T")[0],
        "product_id": order_data["product_id"],
        "store_id": order_data["store_id"],
        "quantity_sold": int(quantity),
        "selling_price": float(order_data["unit_price"]),
    }
    transaction_record = {
        "transaction_id": transaction_id,
        "date": order_date.split("T")[0],
        "transaction_type": "sale",
        "product_id": order_data["product_id"],
        "store_id": order_data["store_id"],
        "quantity": int(quantity),
        "source": "customer_order",
        "remarks": f"Customer order {order_id}",
    }

    updated_orders = pd.concat([orders, pd.DataFrame([order_record])], ignore_index=True)
    updated_sales = pd.concat([sales, pd.DataFrame([sales_record])], ignore_index=True)
    updated_transactions = pd.concat(
        [transactions, pd.DataFrame([transaction_record])],
        ignore_index=True,
    )

    try:
        _write_csv(inventory, INVENTORY_FILE, INVENTORY_COLUMNS)
        _write_csv(updated_sales, SALES_FILE, SALES_COLUMNS)
        _write_csv(updated_transactions, TRANSACTIONS_FILE, TRANSACTION_COLUMNS)
        _write_csv(updated_orders, ORDERS_FILE, ORDER_COLUMNS)
    except Exception as error:
        return {
            "success": False,
            "message": f"Could not save the order safely: {error}",
            "order_data": {},
        }

    order_response = order_record.copy()
    order_response["sale_id"] = sale_id
    order_response["transaction_id"] = transaction_id
    order_response["remaining_stock"] = new_stock_level

    return {
        "success": True,
        "message": f"Order {order_id} placed successfully.",
        "order_data": order_response,
    }
