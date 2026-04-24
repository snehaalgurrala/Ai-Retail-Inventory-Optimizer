from pathlib import Path
import sys

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from backend.services.order_service import (  # noqa: E402
    ORDERS_FILE,
    STORES_FILE,
    get_available_products_by_store,
    place_order,
    validate_order,
)
from frontend.utils.page_helpers import apply_page_style, render_page_header  # noqa: E402


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
def load_stores() -> pd.DataFrame:
    """Load store data for the orders page."""
    if not STORES_FILE.exists():
        return pd.DataFrame()
    return pd.read_csv(STORES_FILE)


@st.cache_data
def load_orders() -> pd.DataFrame:
    """Load saved customer orders when available."""
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


def clear_order_caches() -> None:
    """Clear cached order page data after a successful write."""
    load_stores.clear()
    load_orders.clear()
    st.cache_data.clear()


render_page_header(
    "Orders",
    "Select a store, choose a product, validate quantity, and place a customer order using the backend order service.",
)

receipt = st.session_state.get("latest_order_receipt")
if receipt:
    st.success(receipt.get("message", "Order placed successfully."))

try:
    stores = load_stores()
except Exception as error:
    st.error("Could not load store data.")
    st.exception(error)
    st.stop()

if stores.empty:
    st.info("No store data is available.")
    st.stop()

store_options = stores["store_id"].astype(str).tolist()
selected_store_id = st.selectbox("1. Select Store", store_options)
selected_store = stores[stores["store_id"].astype(str).eq(selected_store_id)].iloc[0]

store_left, store_right = st.columns([2, 3], gap="large")
with store_left:
    with st.container(border=True):
        st.markdown("**Selected Store**")
        st.caption(f"Store ID: {selected_store.get('store_id', '')}")
        st.caption(f"Store Name: {selected_store.get('store_name', '')}")
        st.caption(f"City / Location: {selected_store.get('city', '')}")
        if "capacity" in selected_store.index:
            st.caption(f"Capacity: {selected_store.get('capacity', '')}")

store_inventory_view = get_available_products_by_store(selected_store_id)
if not store_inventory_view.empty:
    store_inventory_view = store_inventory_view.copy()
    store_inventory_view["product_label"] = (
        store_inventory_view["product_name"]
        .fillna(store_inventory_view["product_id"])
        .astype(str)
        + " ("
        + store_inventory_view["category"].fillna("Unknown").astype(str)
        + ")"
    )

with store_right:
    with st.container(border=True):
        st.markdown("**2. Available Products For This Store**")
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

if store_inventory_view.empty:
    st.info("Orders cannot be placed because this store has no available products.")
    st.stop()

product_options = store_inventory_view["product_id"].astype(str).tolist()
selected_product_id = st.selectbox(
    "3. Select Product",
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

order_left, order_right = st.columns([2, 3], gap="large")
with order_left:
    quantity_ordered = st.number_input(
        "4. Enter Quantity",
        min_value=1,
        step=1,
        value=1,
    )

    validation = validate_order(
        selected_store_id,
        selected_product_id,
        int(quantity_ordered),
    )
    is_valid_order = validation.get("success", False)

    if not is_valid_order:
        st.warning(validation.get("message", "Order validation failed."))
    else:
        st.info("5. Quantity validated successfully.")

    if st.button(
        "6. Place Order",
        use_container_width=True,
        disabled=not is_valid_order,
    ):
        result = place_order(
            selected_store_id,
            selected_product_id,
            int(quantity_ordered),
        )
        if result.get("success", False):
            st.session_state["latest_order_receipt"] = result
            clear_order_caches()
            st.rerun()
        else:
            st.error(result.get("message", "Order could not be placed."))

with order_right:
    with st.container(border=True):
        st.markdown("**Order Preview**")
        st.caption(f"Product: {selected_product.get('product_name', '')}")
        st.caption(f"Category: {selected_product.get('category', '')}")
        st.caption(f"Available Quantity: {available_quantity}")
        st.caption(f"Unit Price: {unit_price:,.2f}")
        st.caption(f"Projected Total: {(int(quantity_ordered) * unit_price):,.2f}")

st.divider()

st.subheader("7. Order Status")
if receipt:
    order_data = receipt.get("order_data", {})
    with st.container(border=True):
        st.markdown("**8. Order Receipt**")
        st.caption(f"Order ID: {order_data.get('order_id', '')}")
        st.caption(f"Order Date: {order_data.get('order_date', '')}")
        st.caption(f"Store: {order_data.get('store_name', '')} ({order_data.get('store_id', '')})")
        st.caption(f"City: {order_data.get('city', '')}")
        st.caption(f"Product: {order_data.get('product_name', '')} ({order_data.get('product_id', '')})")
        st.caption(f"Quantity Ordered: {order_data.get('quantity_ordered', '')}")
        st.caption(f"Unit Price: {float(order_data.get('unit_price', 0)):,.2f}")
        st.caption(f"Total Amount: {float(order_data.get('total_amount', 0)):,.2f}")
        st.caption(f"Status: {order_data.get('status', '')}")
        st.caption(f"Remaining Inventory: {order_data.get('remaining_stock', '')}")

    st.info('9. Go to the Home page and click "Run / Refresh Agents" to refresh recommendations after this order.')
else:
    st.info("Place an order to see the latest receipt here.")

st.divider()

st.subheader("Recent Orders")
orders = load_orders()
if orders.empty:
    st.info("No customer orders have been placed yet.")
else:
    recent_orders = orders.sort_values("order_date", ascending=False).head(20)
    st.dataframe(recent_orders, use_container_width=True, hide_index=True)
