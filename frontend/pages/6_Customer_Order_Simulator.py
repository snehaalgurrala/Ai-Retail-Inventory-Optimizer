"""Customer Order Simulator.

A self-contained sandbox that lets a Bunzl end-customer "log in" and place a
mock order against the live product catalogue. Placing an order persists the
order, draws down BZ_MOCK_INVENTORY, recalculates Customer Intelligence, re-runs
the Recommendation agent, and emails an Abnormal Order Investigation Alert or a
Low Stock Alert when those conditions are met. All data is sourced from Oracle
(BZ_MOCK_CUSTOMER / BZ_MOCK_PRODUCT / BZ_MOCK_INVENTORY) via the repository.
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
from backend.services import customer_intelligence_service as cis  # noqa: E402
from backend.services import inventory_scope  # noqa: E402
from backend.services import order_pipeline_service  # noqa: E402
from frontend.utils.page_helpers import apply_page_style, render_page_header  # noqa: E402


# Shared session-state key the Customer Intelligence page writes its deviation
# slider into; the simulator reads the same value so its recalculation uses the
# user-configured threshold.
CI_THRESHOLD_KEY = "ci_abnormal_threshold"


st.set_page_config(
    page_title="Customer Order Simulator",
    page_icon="🛒",
    layout="wide",
)

apply_page_style()


# The accepted password is intentionally a shared demo secret — this is a
# simulation surface, not a real authentication boundary.
DEMO_PASSWORD = "Bunzl@123"

# In Oracle mode, cap the cache lifetime so a result captured during a startup
# failure cannot survive indefinitely (mirrors the other pages).
_CACHE_TTL = 60 if get_data_backend() == "oracle" else None


@st.cache_data(ttl=_CACHE_TTL, show_spinner="Loading Bunzl catalogue from Oracle...")
def load_simulator_data() -> dict[str, pd.DataFrame]:
    return {
        "customers": repository.load_customers(safe=True),
        "products": repository.load_products(safe=True),
        "inventory": repository.load_inventory(safe=True),
    }


def money(value: float) -> str:
    return f"${float(value or 0):,.2f}"


SIMULATOR_CSS = """
<style>
.cos-login-note {
    font-size: 0.82rem;
    color: rgba(10, 31, 51, 0.6);
    margin-top: 0.4rem;
}
.cos-welcome {
    display: flex; align-items: center; justify-content: space-between;
    flex-wrap: wrap; gap: 0.6rem;
    border-radius: 14px;
    border: 1px solid var(--airio-border, #D8E2EC);
    border-left: 5px solid var(--airio-green, #6CB33F);
    background: linear-gradient(180deg, #FFFFFF 0%, #F7FBF3 100%);
    padding: 0.85rem 1.1rem;
    margin-bottom: 1.1rem;
    box-shadow: 0 8px 18px rgba(10, 31, 51, 0.05);
}
.cos-welcome-name {
    font-size: 1.12rem; font-weight: 800; color: var(--airio-deep-navy, #0A1F33);
}
.cos-welcome-sub { font-size: 0.82rem; color: rgba(10, 31, 51, 0.62); margin-top: 0.1rem; }
.cos-welcome-badge {
    display: inline-flex; align-items: center; gap: 0.35rem;
    padding: 0.3rem 0.7rem; border-radius: 999px;
    background: rgba(108, 179, 63, 0.16);
    border: 1px solid rgba(108, 179, 63, 0.3);
    color: #285F12; font-weight: 750; font-size: 0.8rem;
}
.cos-prod-cat {
    display: inline-flex; align-self: flex-start;
    padding: 0.16rem 0.55rem; border-radius: 999px;
    background: var(--airio-soft-blue, #EAF1F7);
    color: var(--airio-primary-navy, #183F5F);
    font-size: 0.68rem; font-weight: 800; text-transform: uppercase;
    letter-spacing: 0.04em; margin-bottom: 0.45rem;
}
.cos-prod-name {
    font-size: 1.02rem; font-weight: 800; color: var(--airio-deep-navy, #0A1F33);
    line-height: 1.25; min-height: 2.5em; margin-bottom: 0.55rem;
}
.cos-prod-meta { display: flex; justify-content: space-between; gap: 0.5rem; margin-bottom: 0.2rem; }
.cos-prod-cell .k {
    font-size: 0.64rem; text-transform: uppercase; letter-spacing: 0.04em;
    color: rgba(10, 31, 51, 0.55); font-weight: 700;
}
.cos-prod-cell .v {
    font-size: 1.08rem; font-weight: 800; color: var(--airio-deep-navy, #0A1F33); margin-top: 0.1rem;
}
.cos-prod-cell .v.price { color: var(--airio-primary-navy, #183F5F); }
.cos-prod-cell.stock .v.low { color: var(--airio-warning, #C76A12); }
.cos-prod-cell.stock .v.out { color: var(--airio-risk, #B42318); }
.cos-cart-empty { font-size: 0.86rem; color: rgba(10, 31, 51, 0.6); padding: 0.4rem 0 0.2rem 0; }
.cos-cart-row {
    display: flex; justify-content: space-between; align-items: baseline; gap: 0.5rem;
    padding: 0.4rem 0; border-bottom: 1px dashed var(--airio-border, #D8E2EC);
}
.cos-cart-row .nm { font-size: 0.84rem; font-weight: 700; color: var(--airio-deep-navy, #0A1F33); line-height: 1.3; }
.cos-cart-row .qty { font-size: 0.72rem; color: rgba(10, 31, 51, 0.6); }
.cos-cart-row .amt { font-size: 0.86rem; font-weight: 800; color: var(--airio-primary-navy, #183F5F); white-space: nowrap; }
.cos-cart-total {
    display: flex; justify-content: space-between; align-items: baseline;
    margin-top: 0.7rem; padding-top: 0.6rem; border-top: 2px solid var(--airio-border, #D8E2EC);
}
.cos-cart-total .lbl { font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em; font-weight: 800; color: rgba(10, 31, 51, 0.6); }
.cos-cart-total .val { font-size: 1.4rem; font-weight: 800; color: var(--airio-deep-navy, #0A1F33); }
</style>
"""


# --------------------------------------------------------------------------
# Session state helpers
# --------------------------------------------------------------------------
def _init_state() -> None:
    st.session_state.setdefault("cos_logged_in", False)
    st.session_state.setdefault("cos_customer_id", None)
    st.session_state.setdefault("cos_customer_name", None)
    # Cart: product_id -> {"name", "price", "qty"}
    st.session_state.setdefault("cos_cart", {})
    st.session_state.setdefault("cos_last_order", None)


def _logout() -> None:
    st.session_state["cos_logged_in"] = False
    st.session_state["cos_customer_id"] = None
    st.session_state["cos_customer_name"] = None
    st.session_state["cos_cart"] = {}
    st.session_state["cos_last_order"] = None


@st.cache_data(ttl=_CACHE_TTL)
def fulfillment_branch_id() -> int:
    """The single branch all simulated orders draw stock from (lowest active)."""
    return oracle_writer.default_branch_id()


def _stock_by_product(inventory: pd.DataFrame, branch_id: int) -> dict[str, int]:
    """On-hand stock per product under the platform-wide inventory scope.

    Resolves through the shared ``inventory_scope`` helper so the catalogue's
    "Current Stock", the pre-flight validation, and the order-time draw-down all
    use the same figure the Customer Intelligence page, emails, and chatbot show —
    network-wide totals by default (``branch_id`` only applies under branch scope).
    """
    return inventory_scope.stock_by_product(inventory, branch_id=branch_id)


# --------------------------------------------------------------------------
# Login
# --------------------------------------------------------------------------
def render_login(customers: pd.DataFrame) -> None:
    if customers.empty:
        st.error("No customers are available from BZ_MOCK_CUSTOMER. Cannot start the simulator.")
        return

    active = customers
    if "active_flg" in customers.columns:
        flagged = customers[customers["active_flg"].astype(str).str.upper() == "Y"]
        if not flagged.empty:
            active = flagged
    active = active.sort_values("customer_name")

    name_to_id = dict(zip(active["customer_name"], active["customer_id"].astype(str)))

    left, _ = st.columns([1, 1])
    with left:
        with st.container(border=True):
            st.markdown("#### 🔐 Customer Login")
            st.caption("Sign in as a Bunzl customer to start a simulated order.")
            with st.form("cos_login_form", clear_on_submit=False):
                customer_name = st.selectbox("Customer", list(name_to_id.keys()))
                password = st.text_input("Password", type="password", placeholder="Enter password")
                submitted = st.form_submit_button("Login", use_container_width=True)

            if submitted:
                if password != DEMO_PASSWORD:
                    st.error("Incorrect password. Please try again.")
                else:
                    st.session_state["cos_logged_in"] = True
                    st.session_state["cos_customer_id"] = name_to_id[customer_name]
                    st.session_state["cos_customer_name"] = customer_name
                    st.session_state["cos_cart"] = {}
                    st.session_state["cos_last_order"] = None
                    st.rerun()
            st.markdown(
                '<div class="cos-login-note">Simulation surface — placing an order updates '
                "Oracle inventory, recalculates Customer Intelligence + recommendations, and "
                "emails an abnormal-order or low-stock alert when those conditions are met.</div>",
                unsafe_allow_html=True,
            )


# --------------------------------------------------------------------------
# Product catalogue
# --------------------------------------------------------------------------
def _stock_class(stock: int) -> str:
    if stock <= 0:
        return "out"
    if stock < 50:
        return "low"
    return ""


def render_catalogue(products: pd.DataFrame, stock_map: dict[str, int]) -> None:
    st.markdown("### 🛍️ Product Catalogue")
    if products.empty:
        st.info("No products are available from BZ_MOCK_PRODUCT.")
        return

    catalogue = products.sort_values("product_name").reset_index(drop=True)
    columns_per_row = 3
    for start in range(0, len(catalogue), columns_per_row):
        row = catalogue.iloc[start:start + columns_per_row]
        cols = st.columns(columns_per_row, gap="medium")
        for col, (_, product) in zip(cols, row.iterrows()):
            with col:
                _render_product_card(product, stock_map)


def _render_product_card(product: pd.Series, stock_map: dict[str, int]) -> None:
    product_id = str(product["product_id"])
    name = str(product["product_name"])
    category = str(product.get("category") or "Uncategorised")
    price = float(product.get("selling_price") or 0)
    stock = int(stock_map.get(product_id, 0))
    stock_cls = _stock_class(stock)

    with st.container(border=True):
        st.markdown(
            f'<div class="cos-prod-cat">{escape(category)}</div>'
            f'<div class="cos-prod-name">{escape(name)}</div>'
            '<div class="cos-prod-meta">'
            f'<div class="cos-prod-cell"><div class="k">Price</div>'
            f'<div class="v price">{money(price)}</div></div>'
            f'<div class="cos-prod-cell stock"><div class="k">Current Stock</div>'
            f'<div class="v {stock_cls}">{stock:,}</div></div>'
            "</div>",
            unsafe_allow_html=True,
        )
        qty = st.number_input(
            "Quantity",
            min_value=1,
            max_value=9999,
            value=1,
            step=1,
            key=f"cos_qty_{product_id}",
        )
        if st.button("➕ Add to Cart", key=f"cos_add_{product_id}", use_container_width=True):
            _add_to_cart(product_id, name, price, int(qty))
            st.toast(f"Added {int(qty)} × {name} to cart", icon="🛒")
            st.rerun()


def _add_to_cart(product_id: str, name: str, price: float, qty: int) -> None:
    cart = st.session_state["cos_cart"]
    existing = cart.get(product_id)
    if existing:
        existing["qty"] += qty
    else:
        cart[product_id] = {"name": name, "price": price, "qty": qty}


# --------------------------------------------------------------------------
# Cart (sidebar)
# --------------------------------------------------------------------------
def render_cart(stock_map: dict[str, int], branch_id: int) -> None:
    cart = st.session_state["cos_cart"]
    with st.sidebar:
        st.markdown("## 🧺 Your Cart")
        if not cart:
            st.markdown('<div class="cos-cart-empty">Your cart is empty. Add products to get started.</div>',
                        unsafe_allow_html=True)
            return

        total = 0.0
        rows = []
        for item in cart.values():
            line_total = item["price"] * item["qty"]
            total += line_total
            rows.append(
                '<div class="cos-cart-row">'
                f'<div><div class="nm">{escape(item["name"])}</div>'
                f'<div class="qty">{item["qty"]} × {money(item["price"])}</div></div>'
                f'<div class="amt">{money(line_total)}</div>'
                "</div>"
            )
        rows.append(
            '<div class="cos-cart-total"><span class="lbl">Total Amount</span>'
            f'<span class="val">{money(total)}</span></div>'
        )
        st.markdown("".join(rows), unsafe_allow_html=True)

        total_units = sum(item["qty"] for item in cart.values())
        st.caption(f"{len(cart)} product(s) · {total_units:,} units")

        # Pre-flight stock validation against the fulfillment branch. Authoritative
        # validation still happens transactionally at order time, but this blocks
        # the obvious shortfalls up front with a clear message.
        shortfalls = [
            (item["name"], item["qty"], int(stock_map.get(product_id, 0)))
            for product_id, item in cart.items()
            if item["qty"] > int(stock_map.get(product_id, 0))
        ]
        if shortfalls:
            lines = "\n".join(
                f"- **{name}**: ordered {qty:,}, only {avail:,} in stock"
                for name, qty, avail in shortfalls
            )
            st.warning("Not enough stock to place this order:\n" + lines)

        place = st.button(
            "✅ Place Order",
            use_container_width=True,
            type="primary",
            disabled=bool(shortfalls),
        )
        if place:
            try:
                _place_order(cart, total_units, branch_id)
            except Exception as error:  # surface the failure, keep the cart intact
                st.error(f"Could not place the order: {error}")
            else:
                st.rerun()
        if st.button("🗑️ Clear Cart", use_container_width=True):
            st.session_state["cos_cart"] = {}
            st.rerun()


def _place_order(cart: dict, total_units: int, branch_id: int) -> None:
    """Persist the order, draw down inventory, then run the post-order pipeline.

    Creates ORDER_HEADER + ORDER_LINE and reduces on-hand stock in
    BZ_MOCK_INVENTORY at the fulfillment branch (validated transactionally). Then
    recalculates Customer Intelligence on the new order and re-runs the
    Recommendation agent — no other agents are triggered. Emails are sent only for
    notable conditions: an Abnormal Order Investigation Alert when the order is
    flagged Critical/High/Medium, and a Low Stock Alert when the order drives any
    ordered product to/below its reorder point. On
    success the confirmation details are stashed in session state, every data
    cache is invalidated so the dependent pages reload fresh, and the cart is
    cleared.
    """
    if get_data_backend() != "oracle":
        raise RuntimeError("Order persistence requires the Oracle backend.")

    items = [
        {"product_id": product_id, "quantity": item["qty"]}
        for product_id, item in cart.items()
    ]
    result = oracle_writer.place_customer_order(
        st.session_state["cos_customer_id"],
        items,
        branch_id=branch_id,
    )

    # Join the per-product new stock levels back to display names for the
    # inventory-update confirmation.
    inv_lines = [
        {
            "name": cart.get(entry["product_id"], {}).get("name", entry["product_id"]),
            "qty": entry["quantity"],
            "new_stock": entry["new_stock"],
        }
        for entry in result.get("inventory", [])
    ]

    last_order = {
        "ref": result["order_nbr"],
        "order_id": result["order_id"],
        "customer": st.session_state["cos_customer_name"],
        "lines": result["line_count"],
        "units": total_units,
        "total": result["order_total"],
        "branch_id": result["branch_id"],
        "placed_at": result["order_date"].strftime("%Y-%m-%d %H:%M"),
        "inventory": inv_lines,
    }

    # --- Post-order pipeline: Customer Intelligence -> Recommendation agent ---
    # Stage 1 recomputes the abnormal-order analysis treating this order as the
    # latest, using the threshold configured on the Customer Intelligence page.
    threshold = float(
        st.session_state.get(CI_THRESHOLD_KEY, cis.DEFAULT_ABNORMAL_DEVIATION_PCT)
    )
    try:
        with st.spinner("Recalculating Customer Intelligence on the new order..."):
            last_order["ci"] = order_pipeline_service.recalculate_customer_intelligence(
                result["order_nbr"], threshold, branch_id=result["branch_id"]
            )
    except Exception as error:  # CI recalc is best-effort; the order is committed
        last_order["ci_error"] = str(error)

    # Stage 2 re-runs the Recommendation agent so reorder / transfer / supplier
    # recommendations reflect the new stock levels.
    try:
        with st.spinner("Triggering the Recommendation Agent..."):
            last_order["rec"] = order_pipeline_service.recalculate_recommendations()
    except Exception as error:  # agent run is best-effort; the order is committed
        last_order["rec_error"] = str(error)

    # Stage 3 — after all analysis completes, send the Abnormal Order
    # Investigation Alert email for any Critical / High / Medium order.
    ci_summary = last_order.get("ci")
    if ci_summary and ci_summary.get("is_abnormal"):
        try:
            with st.spinner("Sending abnormal-order investigation alert..."):
                last_order["alert"] = order_pipeline_service.dispatch_abnormal_order_alerts(ci_summary)
        except Exception as error:  # alert is best-effort; the order is committed
            last_order["alert_error"] = str(error)

    # Stage 4 — independent of the abnormal-order check, send a Low Stock Alert
    # email if this order drove any ordered product to/below its reorder point.
    try:
        with st.spinner("Checking reorder points / sending low-stock alert..."):
            last_order["low_stock"] = order_pipeline_service.dispatch_low_stock_alerts(
                result["order_nbr"],
                [it["product_id"] for it in items],
                branch_id=result["branch_id"],
            )
    except Exception as error:  # alert is best-effort; the order is committed
        last_order["low_stock_error"] = str(error)

    # Stage 5 — refresh the chatbot's Oracle context so it can answer questions
    # about this order immediately (abnormal orders, latest order, impact, reorder).
    try:
        last_order["chatbot"] = order_pipeline_service.refresh_chatbot_context()
    except Exception as error:  # best-effort; the order is committed
        last_order["chatbot_error"] = str(error)

    st.session_state["cos_last_order"] = last_order
    st.session_state["cos_cart"] = {}
    # Invalidate every @st.cache_data loader so the catalogue, Customer
    # Intelligence and Recommendations pages all reload fresh from Oracle and the
    # regenerated outputs on their next render.
    st.cache_data.clear()


def render_order_confirmation() -> None:
    order = st.session_state.get("cos_last_order")
    if not order:
        return
    st.success(
        f"**Order {order['ref']} placed and saved to Oracle.** "
        f"{order['lines']} product(s) · {order['units']:,} units · {money(order['total'])} "
        f"for {order['customer']} at {order['placed_at']}.\n\n"
        "Written to BZ_MOCK_ORDER_HEADER + BZ_MOCK_ORDER_LINE and committed."
    )

    inventory = order.get("inventory") or []
    if inventory:
        lines = "\n".join(
            f"- **{escape(str(line['name']))}**: −{line['qty']:,} → "
            f"**{line['new_stock']:,}** units on hand"
            for line in inventory
        )
        if inventory_scope.get_inventory_scope() == "network":
            heading = "📦 **Inventory updated (network-wide)**"
            note = "(BZ_MOCK_INVENTORY on-hand stock reduced; remaining across all branches):"
        else:
            branch = order.get("branch_id", "—")
            heading = f"📦 **Inventory updated at branch {branch}**"
            note = "(BZ_MOCK_INVENTORY on-hand stock reduced):"
        st.info(f"{heading} {note}\n" + lines)

    _render_ci_recalc(order)
    _render_alert_status(order)
    _render_low_stock_status(order)
    _render_recommendation_recalc(order)
    if order.get("chatbot", {}).get("refreshed"):
        st.caption(
            "💬 Chatbot context refreshed — ask it “Any abnormal orders today?”, "
            "“Latest customer order?”, “Inventory impact of recent orders?” or "
            "“What should we reorder?”."
        )


def _render_alert_status(order: dict) -> None:
    """Abnormal Order Investigation Alert email dispatch result."""
    if order.get("alert_error"):
        st.warning(f"Abnormal-order alert email failed: {order['alert_error']}")
        return
    alert = order.get("alert")
    if not alert or not alert.get("attempted"):
        return
    bands = ", ".join(alert.get("bands") or [])
    if alert.get("sent"):
        st.success(
            f"📧 **Abnormal Order Investigation Alert sent** "
            f"({alert['sent']} of {alert['attempted']} · {bands}). {alert.get('message', '')}"
        )
    else:
        st.warning(
            "📧 Abnormal Order Investigation Alert was triggered "
            f"({alert['attempted']} · {bands}) but not delivered: {alert.get('message', '')}"
        )


def _render_low_stock_status(order: dict) -> None:
    """Low Stock Alert email dispatch result for products driven below reorder."""
    if order.get("low_stock_error"):
        st.warning(f"Low-stock alert email failed: {order['low_stock_error']}")
        return
    low_stock = order.get("low_stock")
    if not low_stock or not low_stock.get("attempted"):
        return
    products = ", ".join(low_stock.get("products") or [])
    count = low_stock.get("attempted", 0)
    if low_stock.get("sent"):
        st.success(
            f"📉 **Low Stock Alert email sent** — {count} product(s) now at/below reorder "
            f"point ({products}). {low_stock.get('message', '')}"
        )
    else:
        st.warning(
            f"📉 Low Stock Alert was triggered for {count} product(s) at/below reorder "
            f"point ({products}) but not delivered: {low_stock.get('message', '')}"
        )


def _render_ci_recalc(order: dict) -> None:
    """Customer Intelligence recalculation result for the new order."""
    if order.get("ci_error"):
        st.warning(f"Customer Intelligence recalculation failed: {order['ci_error']}")
        return
    ci = order.get("ci")
    if not ci:
        return

    threshold = int(ci.get("threshold", 0))
    if not ci.get("is_abnormal"):
        st.success(
            f"🧠 **Customer Intelligence recalculated** at the configured "
            f"{threshold}% deviation threshold — this order is within normal demand "
            "patterns, so no abnormal-order investigation was raised. "
            "The order now appears on the Customer Intelligence page."
        )
        return

    entries = ci.get("entries") or []
    blocks = []
    for entry in entries:
        impact = entry.get("inventory_impact_pct")
        impact_str = f"{impact:.0f}% of on-hand stock" if impact is not None else "n/a (stock unknown)"
        inv = entry.get("current_inventory")
        inv_str = f"{inv:,} units" if inv is not None else "unavailable"
        blocks.append(
            f"- **{escape(entry['product_name'])}** — ordered "
            f"**{entry['current_quantity']:,}** vs historical avg "
            f"**{entry['historical_avg']:.0f}** / max **{entry['historical_max']:,}**  \n"
            f"  Deviation **+{entry['deviation_pct']:.0f}%** · Inventory impact "
            f"**{impact_str}** (on hand {inv_str}) · Risk Score "
            f"**{entry['risk_score']}/100 ({entry['risk_band']})**"
        )
    st.error(
        f"🚨 **Abnormal order detected — investigation entry created** "
        f"(threshold {threshold}%). It is now flagged on the Customer Intelligence page:\n"
        + "\n".join(blocks)
    )


def _render_recommendation_recalc(order: dict) -> None:
    """Recommendation agent re-run result."""
    if order.get("rec_error"):
        st.warning(
            "Recommendation Agent re-run failed: "
            f"{order['rec_error']}. Recommendations may be stale until the next refresh."
        )
        return
    rec = order.get("rec")
    if not rec:
        return
    st.info(
        "🤖 **Recommendation Agent re-run** against the new stock levels — "
        f"**{rec['total']:,}** recommendations: "
        f"reorder **{rec['reorder']}** · transfer **{rec['transfer']}** · "
        f"supplier risk **{rec['supplier_risk']}**. "
        "The Recommendations page now reflects this order."
    )


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------
def main() -> None:
    st.markdown(SIMULATOR_CSS, unsafe_allow_html=True)
    _init_state()

    render_page_header(
        "Customer Order Simulator",
        "Sign in as a Bunzl customer and place a simulated order against the live catalogue.",
    )

    data = load_simulator_data()

    if not st.session_state["cos_logged_in"]:
        render_login(data["customers"])
        return

    # Logged in.
    customer_name = st.session_state["cos_customer_name"]
    st.markdown(
        '<div class="cos-welcome">'
        "<div>"
        f'<div class="cos-welcome-name">👋 Welcome, {escape(str(customer_name))}</div>'
        '<div class="cos-welcome-sub">Browse the catalogue, build your cart, and place a simulated order.</div>'
        "</div>"
        '<span class="cos-welcome-badge">Simulation mode</span>'
        "</div>",
        unsafe_allow_html=True,
    )

    top_left, top_right = st.columns([3, 1])
    with top_right:
        if st.button("🔙 Switch Customer / Logout", use_container_width=True):
            _logout()
            st.rerun()

    render_order_confirmation()

    branch_id = fulfillment_branch_id()
    stock_map = _stock_by_product(data["inventory"], branch_id)
    render_catalogue(data["products"], stock_map)
    render_cart(stock_map, branch_id)


main()
