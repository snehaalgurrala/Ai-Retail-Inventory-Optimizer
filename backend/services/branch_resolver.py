"""Resolve a customer's "home" distribution branch.

The Order Simulator locks a logged-in customer's whole session to a single
branch — catalogue, stock display, pre-flight validation and inventory draw-down
all scope to it. This module derives that branch from Oracle data (no hardcoded
city->branch map) using a deterministic three-step rule:

1. Primary  -> the active branch whose CITY+STATE matches the customer's
   CITY+STATE (case-insensitive, trimmed; matched on city+state, not zip).
2. Fallback -> the branch the customer has historically ordered from most
   (mode of ORDER_HEADER.BRANCH_ID, exposed as ``store_id`` on the orders frame).
3. Final    -> the lowest active BRANCH_ID.

For the seven active seed customers this yields branches 1, 2, 3, 1, 5, 5, 4
(see ``tests``/asserts).
"""

from __future__ import annotations

import pandas as pd

from backend.db import repository


def _active_branches(branches: pd.DataFrame) -> pd.DataFrame:
    """Active branches with a numeric ``branch_id_int`` helper column."""
    if branches is None or branches.empty or "branch_id" not in branches.columns:
        return pd.DataFrame()
    active = branches
    if "active_flg" in branches.columns:
        flagged = branches[branches["active_flg"].astype(str).str.upper() == "Y"]
        if not flagged.empty:
            active = flagged
    active = active.copy()
    active["branch_id_int"] = pd.to_numeric(active["branch_id"], errors="coerce")
    return active.dropna(subset=["branch_id_int"])


def resolve_home_branch(
    customer_id,
    *,
    branches: pd.DataFrame | None = None,
    customers: pd.DataFrame | None = None,
    orders: pd.DataFrame | None = None,
) -> int:
    """Return the home ``BRANCH_ID`` (int) for ``customer_id``.

    Frames are loaded from the repository when not injected (injection is for
    tests). Applies the city/state -> order-mode -> lowest-active rule above.
    """
    customer_id = str(customer_id)
    if branches is None:
        branches = repository.load_branches(safe=True)
    if customers is None:
        customers = repository.load_customers(safe=True)

    active = _active_branches(branches)

    # 1. CITY + STATE match (case-insensitive, trimmed).
    if (
        not active.empty
        and {"city", "state"} <= set(active.columns)
        and customers is not None
        and not customers.empty
        and "customer_id" in customers.columns
    ):
        crow = customers[customers["customer_id"].astype(str) == customer_id]
        if not crow.empty:
            city = str(crow.iloc[0].get("city", "")).strip().upper()
            state = str(crow.iloc[0].get("state", "")).strip().upper()
            if city:
                match = active[
                    (active["city"].astype(str).str.strip().str.upper() == city)
                    & (active["state"].astype(str).str.strip().str.upper() == state)
                ]
                if not match.empty:
                    return int(match["branch_id_int"].min())

    # 2. Fallback: most-frequent branch this customer has ordered from.
    if orders is None:
        orders = repository.load_orders(safe=True)
    if (
        orders is not None
        and not orders.empty
        and "customer_id" in orders.columns
        and "store_id" in orders.columns
    ):
        mine = orders[orders["customer_id"].astype(str) == customer_id]
        if not mine.empty:
            modes = pd.to_numeric(mine["store_id"], errors="coerce").dropna().mode()
            if not modes.empty:
                return int(modes.iloc[0])

    # 3. Final fallback: lowest active branch id.
    if not active.empty:
        return int(active["branch_id_int"].min())
    return 1
