"""Admin Inventory — branch-level stock replenishment.

A back-office sandbox that lets an operator top up on-hand stock after the Order
Simulator (or anything else) has drawn it down. Replenishment is intentionally
**replenish-only** and targets a single **branch + product** at a time, matching
the physical grain of BZ_MOCK_INVENTORY (one row per PRODUCT_ID + BRANCH_ID).

Each replenishment is one atomic Oracle transaction via
``oracle_writer.apply_movements`` (positive delta, validated, ledger-logged to
BZ_MOCK_INVENTORY_TRANSACTION as a PURCHASE). After a successful write every
@st.cache_data loader is cleared so the network-wide figure shown on Customer
Intelligence, the simulator catalogue, the chatbot and recommendations re-sums
and matches — the network total is a live SUM across branches, never stored.
"""

from datetime import datetime
from html import escape
from pathlib import Path
import sys

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from backend.db import oracle_writer, repository  # noqa: E402
from backend.db.config import get_data_backend  # noqa: E402
from backend.services import inventory_scope  # noqa: E402
from backend.services import order_pipeline_service  # noqa: E402
from frontend.utils.page_helpers import apply_page_style, render_page_header  # noqa: E402


st.set_page_config(
    page_title="Admin Inventory",
    page_icon="🏷️",
    layout="wide",
)

apply_page_style()


# Shared demo secret — this is an internal admin surface, not a real auth boundary.
ADMIN_PASSWORD = "Bunzl@123"

# In Oracle mode, cap the cache lifetime so a result captured during a startup
# failure cannot survive indefinitely (mirrors the other pages).
_CACHE_TTL = 60 if get_data_backend() == "oracle" else None


@st.cache_data(ttl=_CACHE_TTL, show_spinner="Loading Bunzl inventory from Oracle...")
def load_admin_data() -> dict[str, pd.DataFrame]:
    return {
        "products": repository.load_products(safe=True),
        "inventory": repository.load_inventory(safe=True),
        "stores": repository.load_stores(safe=True),
    }


def money(value: float) -> str:
    return f"${float(value or 0):,.2f}"


# --------------------------------------------------------------------------
# Session state + login
# --------------------------------------------------------------------------
def _init_state() -> None:
    st.session_state.setdefault("admin_logged_in", False)
    st.session_state.setdefault("admin_last_replenish", None)


def render_login() -> None:
    left, _ = st.columns([1, 1])
    with left:
        with st.container(border=True):
            st.markdown("#### 🔐 Admin Login")
            st.caption("Sign in to replenish branch inventory in Oracle.")
            with st.form("admin_login_form", clear_on_submit=False):
                password = st.text_input("Password", type="password", placeholder="Enter password")
                submitted = st.form_submit_button("Login", use_container_width=True)
            if submitted:
                if password != ADMIN_PASSWORD:
                    st.error("Incorrect password. Please try again.")
                else:
                    st.session_state["admin_logged_in"] = True
                    st.session_state["admin_last_replenish"] = None
                    st.rerun()
            st.markdown(
                '<div style="font-size:0.82rem;color:rgba(10,31,51,0.6);margin-top:0.4rem;">'
                "Admin surface — replenishing stock updates BZ_MOCK_INVENTORY on-hand at the "
                "chosen branch, logs a PURCHASE movement, and refreshes every dependent page."
                "</div>",
                unsafe_allow_html=True,
            )


# --------------------------------------------------------------------------
# Current inventory overview
# --------------------------------------------------------------------------
def _branch_name_map(stores: pd.DataFrame) -> dict[str, str]:
    if stores.empty:
        return {}
    return {
        str(row["store_id"]): str(row.get("store_name") or row["store_id"])
        for _, row in stores.iterrows()
    }


def _product_name_map(products: pd.DataFrame) -> dict[str, str]:
    if products.empty:
        return {}
    return {
        str(row["product_id"]): str(row.get("product_name") or row["product_id"])
        for _, row in products.iterrows()
    }


def render_overview(data: dict[str, pd.DataFrame]) -> None:
    inventory = data["inventory"]
    products = data["products"]
    scope = inventory_scope.get_inventory_scope()
    scope_label = "network-wide (sum across all branches)" if scope == "network" else "per-branch"

    render_page_header(
        "🏷️ Admin Inventory",
        f"Replenish branch stock · active inventory scope: {scope_label}",
    )

    if inventory.empty:
        st.info("No inventory rows are available from BZ_MOCK_INVENTORY.")
        return

    name_map = _product_name_map(products)
    stock = inventory_scope.stock_by_product(inventory)
    reorder = inventory_scope.reorder_by_product(inventory)
    at_risk = inventory_scope.at_risk_product_ids(inventory)

    rows = []
    for pid in sorted(stock, key=lambda p: name_map.get(p, p)):
        on_hand = stock.get(pid, 0)
        rp = reorder.get(pid, 0)
        rows.append({
            "Product": name_map.get(pid, pid),
            "On Hand": on_hand,
            "Reorder Point": rp,
            "Status": "⚠️ At / below reorder" if pid in at_risk else "✅ Healthy",
        })
    summary = pd.DataFrame(rows)

    st.markdown("### 📊 Current Inventory")
    st.caption(
        f"On-hand and reorder figures are {scope_label} — the same numbers the "
        "Customer Intelligence page, simulator catalogue and chatbot show."
    )
    cols = st.columns(3)
    cols[0].metric("Products", f"{len(summary):,}")
    cols[1].metric("At / below reorder", f"{len(at_risk):,}")
    cols[2].metric("Total units on hand", f"{int(summary['On Hand'].sum()):,}")

    st.dataframe(summary, use_container_width=True, hide_index=True)

    with st.expander("🏬 Per-branch detail (BZ_MOCK_INVENTORY rows)"):
        detail = inventory.copy()
        detail["Product"] = detail["product_id"].astype(str).map(name_map).fillna(
            detail["product_id"].astype(str)
        )
        branch_map = _branch_name_map(data["stores"])
        detail["Branch"] = detail["store_id"].astype(str).map(branch_map).fillna(
            detail["store_id"].astype(str)
        )
        detail = detail.rename(columns={
            "stock_level": "On Hand",
            "reorder_threshold": "Reorder Point",
            "last_updated": "Last Updated",
        })
        st.dataframe(
            detail[["Branch", "Product", "On Hand", "Reorder Point", "Last Updated"]],
            use_container_width=True,
            hide_index=True,
        )


# --------------------------------------------------------------------------
# Replenishment
# --------------------------------------------------------------------------
def render_replenish_form(data: dict[str, pd.DataFrame]) -> None:
    inventory = data["inventory"]
    products = data["products"]
    stores = data["stores"]

    st.markdown("### ➕ Replenish Stock")
    if stores.empty or products.empty:
        st.info("Branches or products are unavailable — cannot replenish.")
        return

    branch_map = _branch_name_map(stores)
    product_map = _product_name_map(products)
    branch_label_to_id = {f"{name} (#{bid})": bid for bid, name in branch_map.items()}
    product_label_to_id = {f"{name} (#{pid})": pid for pid, name in product_map.items()}

    with st.form("admin_replenish_form", clear_on_submit=False):
        c1, c2, c3 = st.columns([2, 2, 1])
        branch_label = c1.selectbox("Branch", sorted(branch_label_to_id.keys()))
        product_label = c2.selectbox("Product", sorted(product_label_to_id.keys()))
        qty = c3.number_input("Quantity", min_value=1, max_value=999999, value=100, step=10)
        submitted = st.form_submit_button("📦 Replenish", use_container_width=True, type="primary")

    if not submitted:
        return

    branch_id = branch_label_to_id[branch_label]
    product_id = product_label_to_id[product_label]
    try:
        _replenish(inventory, product_id, branch_id, int(qty))
    except Exception as error:  # surface the failure; nothing was committed
        st.error(f"Could not replenish: {error}")
    else:
        st.rerun()


def _current_branch_reorder(inventory: pd.DataFrame, product_id: str, branch_id: str) -> int:
    """Existing reorder point for the (product, branch) row, or 0 if none yet.

    Only used as the seed REORDER_POINT when ``allow_create`` inserts a brand-new
    inventory row; existing rows keep their own threshold untouched.
    """
    if inventory.empty:
        return 0
    match = inventory[
        (inventory["product_id"].astype(str) == str(product_id))
        & (inventory["store_id"].astype(str) == str(branch_id))
    ]
    if match.empty:
        return 0
    return int(pd.to_numeric(match.iloc[0]["reorder_threshold"], errors="coerce") or 0)


def _replenish(inventory: pd.DataFrame, product_id: str, branch_id: str, qty: int) -> None:
    """Add ``qty`` units of stock to one branch, atomically, then refresh caches.

    Reuses ``oracle_writer.apply_movements`` with a single positive delta so the
    write is validated and logged to BZ_MOCK_INVENTORY_TRANSACTION (PURCHASE).
    ``allow_create`` covers the case where the branch/product pair has no row yet.
    """
    if get_data_backend() != "oracle":
        raise RuntimeError("Inventory replenishment requires the Oracle backend.")

    threshold = _current_branch_reorder(inventory, product_id, branch_id)
    result = oracle_writer.apply_movements([{
        "product_id": product_id,
        "store_id": branch_id,
        "delta": qty,
        "txn_type": "PURCHASE",
        "quantity": qty,
        "source": "admin_replenishment",
        "remarks": f"Admin replenishment: +{qty} units",
        "allow_create": True,
        "threshold": threshold,
    }])
    branch_new_stock = int(result[0]["new_stock"])
    transaction_id = result[0]["transaction_id"]

    # Re-read inventory uncached so the network total reflects the just-written row
    # (the cached loader is cleared below for every other page's next render).
    fresh = repository.load_inventory(safe=True)
    network_total = int(inventory_scope.stock_by_product(fresh).get(str(product_id), branch_new_stock))

    st.session_state["admin_last_replenish"] = {
        "product_id": str(product_id),
        "branch_id": str(branch_id),
        "qty": qty,
        "branch_new_stock": branch_new_stock,
        "network_total": network_total,
        "transaction_id": transaction_id,
        "scope": inventory_scope.get_inventory_scope(),
        "at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

    # Best-effort: let the chatbot see the new stock immediately.
    try:
        order_pipeline_service.refresh_chatbot_context()
    except Exception:
        pass

    # Invalidate every @st.cache_data loader so the catalogue, Customer
    # Intelligence, recommendations and this page all reload fresh from Oracle.
    st.cache_data.clear()


def render_confirmation(data: dict[str, pd.DataFrame]) -> None:
    last = st.session_state.get("admin_last_replenish")
    if not last:
        return
    product_name = _product_name_map(data["products"]).get(last["product_id"], last["product_id"])
    branch_name = _branch_name_map(data["stores"]).get(last["branch_id"], last["branch_id"])

    st.success(
        f"**Replenished {escape(str(product_name))} at {escape(str(branch_name))}** — "
        f"+{last['qty']:,} units at {last['at']}.\n\n"
        f"BZ_MOCK_INVENTORY on-hand at this branch is now **{last['branch_new_stock']:,}** "
        f"(transaction #{last['transaction_id']} logged to BZ_MOCK_INVENTORY_TRANSACTION)."
    )
    if last["scope"] == "network":
        st.info(
            f"📦 **Network-wide on hand for {escape(str(product_name))} is now "
            f"{last['network_total']:,} units** (sum across all branches) — this is the "
            "figure every page, the chatbot and recommendations now show."
        )


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------
def main() -> None:
    _init_state()

    if not st.session_state["admin_logged_in"]:
        render_page_header("🏷️ Admin Inventory", "Branch-level stock replenishment")
        render_login()
        return

    if get_data_backend() != "oracle":
        render_page_header("🏷️ Admin Inventory", "Branch-level stock replenishment")
        st.warning("This page requires the Oracle backend (DATA_BACKEND=oracle).")
        return

    data = load_admin_data()

    with st.sidebar:
        st.markdown("## 🏷️ Admin")
        if st.button("Log out", use_container_width=True):
            st.session_state["admin_logged_in"] = False
            st.session_state["admin_last_replenish"] = None
            st.rerun()

    render_overview(data)
    render_confirmation(data)
    st.divider()
    render_replenish_form(data)


main()
