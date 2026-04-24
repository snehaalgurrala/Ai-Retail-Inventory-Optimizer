from datetime import datetime
from pathlib import Path
import sys

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from frontend.utils.page_helpers import apply_page_style, render_page_header  # noqa: E402


RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
STORES_FILE = RAW_DATA_DIR / "stores.csv"
INVENTORY_FILE = RAW_DATA_DIR / "inventory.csv"
PRODUCTS_FILE = RAW_DATA_DIR / "products.csv"
SALES_FILE = RAW_DATA_DIR / "sales.csv"
TRANSACTIONS_FILE = RAW_DATA_DIR / "transactions.csv"
ORDERS_FILE = PROCESSED_DATA_DIR / "customer_orders.csv"

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


st.set_page_config(
    page_title="Orders",
    page_icon="O",
    layout="wide",
)

apply_page_style()


@st.cache_data
def load_order_page_data() -> dict[str, pd.DataFrame]:
    """Load the raw datasets used by the Orders page."""
    return {
        "stores": pd.read_csv(STORES_FILE),
        "inventory": pd.read_csv(INVENTORY_FILE),
        "products": pd.read_csv(PRODUCTS_FILE),
        "sales": pd.read_csv(SALES_FILE),
        "transactions": pd.read_csv(TRANSACTIONS_FILE),
        "orders": load_orders(),
    }


def load_orders() -> pd.DataFrame:
    """Load existing customer orders when available."""
    if not ORDERS_FILE.exists():
        return pd.DataFrame(columns=ORDER_COLUMNS)

    try:
        orders = pd.read_csv(ORDERS_FILE)
    except Exception:
        return pd.DataFrame(columns=ORDER_COLUMNS)

    for column in ORDER_COLUMNS:
        if column not in orders.columns:
            orders[column] = ""

    return orders[ORDER_COLUMNS].copy()


def build_store_inventory_view(
    inventory: pd.DataFrame,
    products: pd.DataFrame,
    store_id: str,
) -> pd.DataFrame:
    """Return the selected store inventory with product details."""
    store_inventory = inventory[inventory["store_id"].astype(str) == str(store_id)].copy()
    if store_inventory.empty:
        return pd.DataFrame()

    store_inventory["stock_level"] = pd.to_numeric(
        store_inventory.get("stock_level", 0),
        errors="coerce",
    ).fillna(0)

    view = store_inventory.merge(products, on="product_id", how="left")
    view["selling_price"] = pd.to_numeric(
        view.get("selling_price", 0),
        errors="coerce",
    ).fillna(0)
    view["product_label"] = (
        view["product_name"].fillna(view["product_id"]).astype(str)
        + " ("
        + view["category"].fillna("Unknown").astype(str)
        + ")"
    )
    return view.sort_values(["product_name", "product_id"])


def _next_prefixed_id(existing_df: pd.DataFrame, column: str, prefix: str) -> str:
    """Generate the next sequential ID using an existing prefix pattern."""
    if existing_df.empty or column not in existing_df.columns:
        return f"{prefix}000001"

    series = existing_df[column].fillna("").astype(str)
    numeric_values = pd.to_numeric(
        series.str.replace(prefix, "", regex=False),
        errors="coerce",
    ).dropna()
    next_number = int(numeric_values.max()) + 1 if not numeric_values.empty else 1

    if prefix == "TXN":
        return f"{prefix}{str(next_number).zfill(7)}"
    return f"{prefix}{str(next_number).zfill(6)}"


def save_orders(orders: pd.DataFrame) -> None:
    """Write customer orders safely to processed data."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    writable_orders = orders.copy()
    for column in ORDER_COLUMNS:
        if column not in writable_orders.columns:
            writable_orders[column] = ""
    writable_orders[ORDER_COLUMNS].to_csv(ORDERS_FILE, index=False)


def _cart_key(store_id: str) -> str:
    """Return the session-state key for the selected store cart."""
    return f"orders_cart_{store_id}"


def get_cart(store_id: str) -> list[dict]:
    """Return the current in-memory cart for one store."""
    return list(st.session_state.get(_cart_key(store_id), []))


def set_cart(store_id: str, cart_items: list[dict]) -> None:
    """Persist cart items for the selected store."""
    st.session_state[_cart_key(store_id)] = cart_items


def add_to_cart(
    store_id: str,
    product_row: pd.Series,
    quantity_ordered: int,
) -> tuple[bool, str]:
    """Add one product line to the current cart."""
    if quantity_ordered <= 0:
        return False, "Enter a valid order quantity."

    product_id = str(product_row.get("product_id", ""))
    available_quantity = int(
        pd.to_numeric(product_row.get("stock_level", 0), errors="coerce")
    )
    if quantity_ordered > available_quantity:
        return False, "Insufficient stock available"

    cart_items = get_cart(store_id)
    unit_price = float(
        pd.to_numeric(product_row.get("selling_price", 0), errors="coerce")
    )

    for item in cart_items:
        if str(item.get("product_id", "")) == product_id:
            item["quantity_ordered"] = int(quantity_ordered)
            item["available_quantity"] = available_quantity
            item["unit_price"] = unit_price
            item["total_amount"] = int(quantity_ordered) * unit_price
            set_cart(store_id, cart_items)
            return True, f"{product_row.get('product_name', 'Product')} updated in the order."

    cart_items.append(
        {
            "product_id": product_id,
            "product_name": str(product_row.get("product_name", "")),
            "category": str(product_row.get("category", "")),
            "quantity_ordered": int(quantity_ordered),
            "available_quantity": available_quantity,
            "unit_price": unit_price,
            "total_amount": int(quantity_ordered) * unit_price,
        }
    )
    set_cart(store_id, cart_items)
    return True, f"{product_row.get('product_name', 'Product')} added to the order."


def remove_from_cart(store_id: str, product_id: str) -> None:
    """Remove one product from the current cart."""
    cart_items = [
        item
        for item in get_cart(store_id)
        if str(item.get("product_id", "")) != str(product_id)
    ]
    set_cart(store_id, cart_items)


def place_order(
    store_row: pd.Series,
    cart_items: list[dict],
) -> tuple[bool, str]:
    """Persist a multi-product order and update inventory, sales, and transactions."""
    if not cart_items:
        return False, "Add at least one product to place an order."

    inventory = pd.read_csv(INVENTORY_FILE)
    sales = pd.read_csv(SALES_FILE)
    transactions = pd.read_csv(TRANSACTIONS_FILE)
    orders = load_orders()

    store_id = str(store_row.get("store_id", ""))
    order_date = datetime.now().date().isoformat()
    order_id = _next_prefixed_id(orders, "order_id", "ORD")

    sale_start = _next_prefixed_id(sales, "sale_id", "SALE")
    sale_number = int(str(sale_start).replace("SALE", ""))
    transaction_start = _next_prefixed_id(transactions, "transaction_id", "TXN")
    transaction_number = int(str(transaction_start).replace("TXN", ""))

    order_records = []
    sales_records = []
    transaction_records = []

    for item in cart_items:
        product_id = str(item.get("product_id", ""))
        quantity_ordered = int(item.get("quantity_ordered", 0))
        inventory_mask = (
            inventory["store_id"].astype(str).eq(store_id)
            & inventory["product_id"].astype(str).eq(product_id)
        )
        if not inventory_mask.any():
            return False, f"Product {product_id} is not available in the chosen store."

        current_stock = pd.to_numeric(
            inventory.loc[inventory_mask, "stock_level"],
            errors="coerce",
        ).fillna(0)
        available_quantity = int(current_stock.iloc[0]) if not current_stock.empty else 0
        if quantity_ordered > available_quantity:
            return False, f"Insufficient stock available for {item.get('product_name', product_id)}."

        unit_price = float(pd.to_numeric(item.get("unit_price", 0), errors="coerce"))
        total_amount = float(quantity_ordered * unit_price)

        inventory.loc[inventory_mask, "stock_level"] = available_quantity - quantity_ordered
        inventory.loc[inventory_mask, "last_updated"] = order_date

        order_records.append(
            {
                "order_id": order_id,
                "order_date": order_date,
                "store_id": store_id,
                "store_name": str(store_row.get("store_name", "")),
                "city": str(store_row.get("city", "")),
                "product_id": product_id,
                "product_name": str(item.get("product_name", "")),
                "quantity_ordered": quantity_ordered,
                "unit_price": unit_price,
                "total_amount": total_amount,
                "status": "placed",
            }
        )
        sales_records.append(
            {
                "sale_id": f"SALE{str(sale_number).zfill(6)}",
                "date": order_date,
                "product_id": product_id,
                "store_id": store_id,
                "quantity_sold": quantity_ordered,
                "selling_price": unit_price,
            }
        )
        transaction_records.append(
            {
                "transaction_id": f"TXN{str(transaction_number).zfill(7)}",
                "date": order_date,
                "transaction_type": "sale",
                "product_id": product_id,
                "store_id": store_id,
                "quantity": quantity_ordered,
                "source": "customer_order",
                "remarks": f"Customer order {order_id}",
            }
        )
        sale_number += 1
        transaction_number += 1

    updated_orders = pd.concat([orders, pd.DataFrame(order_records)], ignore_index=True)
    updated_sales = pd.concat([sales, pd.DataFrame(sales_records)], ignore_index=True)
    updated_transactions = pd.concat(
        [transactions, pd.DataFrame(transaction_records)],
        ignore_index=True,
    )

    inventory.to_csv(INVENTORY_FILE, index=False)
    updated_sales.to_csv(SALES_FILE, index=False)
    updated_transactions.to_csv(TRANSACTIONS_FILE, index=False)
    save_orders(updated_orders)

    return True, f"Order {order_id} placed successfully with {len(cart_items)} products."


render_page_header(
    "🧾 Orders",
    "Create customer orders for one store at a time and update inventory, sales, and transactions safely.",
)

try:
    page_data = load_order_page_data()
except Exception as error:
    st.error("Could not load order page data.")
    st.exception(error)
    st.stop()

stores = page_data["stores"]
inventory = page_data["inventory"]
products = page_data["products"]
orders = page_data["orders"]

if stores.empty:
    st.info("No store data is available.")
    st.stop()

store_options = stores["store_id"].astype(str).tolist()
selected_store_id = st.selectbox("Select Store", store_options)
selected_store = stores[stores["store_id"].astype(str).eq(selected_store_id)].iloc[0]

store_left, store_right = st.columns([2, 3], gap="large")
with store_left:
    with st.container(border=True):
        st.markdown(f"**Store Name:** {selected_store.get('store_name', '')}")
        st.caption(f"City / Location: {selected_store.get('city', '')}")
        if "capacity" in selected_store.index:
            st.caption(f"Store Capacity: {selected_store.get('capacity', '')}")

store_inventory_view = build_store_inventory_view(inventory, products, selected_store_id)

with store_right:
    with st.container(border=True):
        st.markdown("**Store Inventory**")
        if store_inventory_view.empty:
            st.caption("No inventory rows are available for this store.")
        else:
            st.dataframe(
                store_inventory_view[
                    [
                        column
                        for column in [
                            "product_id",
                            "product_name",
                            "category",
                            "stock_level",
                            "selling_price",
                        ]
                        if column in store_inventory_view.columns
                    ]
                ].rename(
                    columns={
                        "stock_level": "available_quantity",
                        "selling_price": "unit_price",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

st.divider()

st.subheader("Build Order")
st.caption("Add one or more products from this store to the current order.")

if store_inventory_view.empty:
    st.info("Orders cannot be placed because this store has no available inventory records.")
    st.stop()

product_options = store_inventory_view["product_id"].astype(str).tolist()
selected_product_id = st.selectbox(
    "Select Product",
    product_options,
    format_func=lambda product_id: store_inventory_view[
        store_inventory_view["product_id"].astype(str).eq(str(product_id))
    ]["product_label"].iloc[0],
)
selected_product = store_inventory_view[
    store_inventory_view["product_id"].astype(str).eq(selected_product_id)
].iloc[0]

available_quantity = int(
    pd.to_numeric(selected_product.get("stock_level", 0), errors="coerce")
)
unit_price = float(
    pd.to_numeric(selected_product.get("selling_price", 0), errors="coerce")
)

order_col, info_col = st.columns([2, 3], gap="large")
with order_col:
    quantity_ordered = st.number_input(
        "Order Quantity",
        min_value=1,
        step=1,
        value=1,
    )
    can_add_to_order = quantity_ordered <= available_quantity
    if not can_add_to_order:
        st.warning("Insufficient stock available")

    if st.button(
        "Add to Order",
        use_container_width=True,
        disabled=not can_add_to_order,
    ):
        success, message = add_to_cart(
            selected_store_id,
            selected_product,
            int(quantity_ordered),
        )
        if success:
            st.success(message)
            st.rerun()
        else:
            st.error(message)

with info_col:
    with st.container(border=True):
        st.markdown(f"**Product:** {selected_product.get('product_name', '')}")
        st.caption(f"Category: {selected_product.get('category', '')}")
        st.caption(f"Available Quantity: {available_quantity}")
        st.caption(f"Unit Price: {unit_price:,.2f}")
        st.caption(f"Projected Total: {(int(quantity_ordered) * unit_price):,.2f}")

st.divider()

st.subheader("Current Order")
cart_items = get_cart(selected_store_id)
if not cart_items:
    st.info("No products have been added to the current order yet.")
else:
    cart_df = pd.DataFrame(cart_items)
    cart_df["line_status"] = cart_df.apply(
        lambda row: (
            "Insufficient stock available"
            if int(row.get("quantity_ordered", 0)) > int(row.get("available_quantity", 0))
            else "Ready"
        ),
        axis=1,
    )
    st.dataframe(
        cart_df[
            [
                "product_id",
                "product_name",
                "category",
                "quantity_ordered",
                "available_quantity",
                "unit_price",
                "total_amount",
                "line_status",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    invalid_items = cart_df[cart_df["line_status"].eq("Insufficient stock available")]
    if not invalid_items.empty:
        st.warning("One or more items in the current order have insufficient stock available.")

    cart_total = float(pd.to_numeric(cart_df["total_amount"], errors="coerce").fillna(0).sum())
    st.caption(f"Current order total: {cart_total:,.2f}")

    action_left, action_right = st.columns(2, gap="large")
    with action_left:
        remove_product_id = st.selectbox(
            "Remove Product",
            [""] + cart_df["product_id"].astype(str).tolist(),
            format_func=lambda product_id: (
                "Select a product"
                if product_id == ""
                else f"{product_id} - {cart_df[cart_df['product_id'].astype(str).eq(product_id)]['product_name'].iloc[0]}"
            ),
        )
        if st.button(
            "Remove Selected Product",
            use_container_width=True,
            disabled=remove_product_id == "",
        ):
            remove_from_cart(selected_store_id, remove_product_id)
            st.rerun()
    with action_right:
        if st.button(
            "Place Full Order",
            use_container_width=True,
            disabled=not invalid_items.empty,
        ):
            success, message = place_order(selected_store, cart_items)
            if success:
                set_cart(selected_store_id, [])
                load_order_page_data.clear()
                st.success(message)
                st.rerun()
            else:
                st.error(message)

st.divider()

st.subheader("Recent Orders")
if orders.empty:
    st.info("No customer orders have been placed yet.")
else:
    recent_orders = orders.sort_values("order_date", ascending=False).head(20)
    st.dataframe(
        recent_orders,
        use_container_width=True,
        hide_index=True,
    )
