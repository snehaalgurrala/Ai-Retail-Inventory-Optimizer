"""Single source of truth for per-product "current inventory".

Every surface that shows or reasons about a product's on-hand stock — Customer
Intelligence cards, the abnormal-order email, the Order Simulator catalogue, the
chatbot, recommendations, risk scores — must resolve stock through these helpers
so the numbers are identical everywhere.

The scope is governed by ``config.get_inventory_scope()``:

  * ``"network"`` (default): a product's current inventory is the SUM of
    ``stock_level`` across ALL branches/warehouses. Reorder point is likewise the
    sum across branches. This is the temporary business decision in force until
    the client confirms their fulfillment model.
  * ``"branch"``: stock/reorder are scoped to a single ``branch_id`` (the order's
    or customer's fulfilling branch). Reserved for the future per-branch model.

Example (network): Warehouse A=100, B=80, C=56  ->  Current Inventory = 236.
Use 236 everywhere — page, email, simulator, chatbot, recommendations.
"""

from __future__ import annotations

import pandas as pd

from backend.db.config import get_inventory_scope

_STOCK_COL = "stock_level"
_REORDER_COL = "reorder_threshold"


def _prepare(inventory: pd.DataFrame) -> pd.DataFrame | None:
    if inventory is None or inventory.empty or "product_id" not in inventory.columns:
        return None
    inv = inventory.copy()
    inv["product_id"] = inv["product_id"].astype(str)
    inv["_stock"] = pd.to_numeric(inv.get(_STOCK_COL), errors="coerce").fillna(0)
    inv["_reorder"] = pd.to_numeric(inv.get(_REORDER_COL), errors="coerce").fillna(0)
    return inv


def _scope(scope: str | None) -> str:
    return scope if scope in ("network", "branch") else get_inventory_scope()


def _apply_scope(inv: pd.DataFrame, branch_id, scope: str) -> pd.DataFrame:
    """Filter to one branch under "branch" scope; otherwise keep every branch."""
    if scope == "branch" and branch_id is not None and "store_id" in inv.columns:
        return inv[inv["store_id"].astype(str) == str(branch_id)]
    return inv


def stock_by_product(
    inventory: pd.DataFrame, *, branch_id=None, scope: str | None = None
) -> dict[str, int]:
    """Map product_id -> current on-hand stock under the active inventory scope.

    Network scope sums ``stock_level`` across all branches; branch scope returns
    only ``branch_id``'s stock (falling back to network if no branch is given).
    """
    inv = _prepare(inventory)
    if inv is None:
        return {}
    inv = _apply_scope(inv, branch_id, _scope(scope))
    return inv.groupby("product_id")["_stock"].sum().round().astype(int).to_dict()


def reorder_by_product(
    inventory: pd.DataFrame, *, branch_id=None, scope: str | None = None
) -> dict[str, int]:
    """Map product_id -> reorder threshold under the active inventory scope.

    Network scope sums the per-branch reorder points into a network-wide
    threshold so it lines up with the network-wide on-hand figure.
    """
    inv = _prepare(inventory)
    if inv is None:
        return {}
    inv = _apply_scope(inv, branch_id, _scope(scope))
    return inv.groupby("product_id")["_reorder"].sum().round().astype(int).to_dict()


def product_inventory(
    inventory: pd.DataFrame, *, branch_id=None, scope: str | None = None
) -> dict[str, dict[str, int]]:
    """Map product_id -> {"stock", "reorder"} under the active inventory scope."""
    stock = stock_by_product(inventory, branch_id=branch_id, scope=scope)
    reorder = reorder_by_product(inventory, branch_id=branch_id, scope=scope)
    return {
        pid: {"stock": stock.get(pid, 0), "reorder": reorder.get(pid, 0)}
        for pid in set(stock) | set(reorder)
    }


def at_risk_product_ids(
    inventory: pd.DataFrame, *, branch_id=None, scope: str | None = None
) -> set[str]:
    """Product ids whose current stock is at or below their reorder threshold.

    Compared at the active scope's granularity: network totals vs network reorder
    (default), or a single branch's stock vs that branch's reorder.
    """
    combined = product_inventory(inventory, branch_id=branch_id, scope=scope)
    return {pid for pid, v in combined.items() if v["stock"] <= v["reorder"]}
