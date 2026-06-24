"""Per-customer order quantity limits for the Order Simulator.

The simulator caps how many units of any single product a customer may add to
their cart. Limits are persisted to a small JSON config file alongside the other
generated data files (next to ``recommendations.csv`` in ``data/processed``) —
Oracle is never touched, so the BZ_MOCK_* schema stays exactly as-is.

JSON shape: ``{ "10001": 2, "10002": 5, ... }`` where keys are CUSTOMER_ID as
strings and values are integer per-product unit caps. Any customer not present in
the file defaults to :data:`DEFAULT_LIMIT`.
"""

from __future__ import annotations

import json

from backend.db.paths import PROCESSED_DIR


# Per-product unit cap applied to any customer without an explicit saved limit.
DEFAULT_LIMIT = 2

# Session-state key both the Customer Intelligence page and the Order Simulator
# mirror the active limits into, so they share one in-session view (mirrors the
# cross-page ``ci_abnormal_threshold`` contract).
SESSION_KEY = "customer_order_limits"

# Stored beside the other generated artifacts (recommendations.csv lives here).
LIMITS_PATH = PROCESSED_DIR / "customer_order_limits.json"


def load_limits() -> dict[int, int]:
    """Read all saved limits as ``{customer_id(int): limit(int)}``; ``{}`` if none."""
    if not LIMITS_PATH.exists():
        return {}
    try:
        raw = json.loads(LIMITS_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, ValueError):
        return {}
    limits: dict[int, int] = {}
    for key, value in (raw or {}).items():
        try:
            limits[int(key)] = int(value)
        except (TypeError, ValueError):
            continue
    return limits


def save_limits(limits: dict) -> None:
    """Bulk-write the limits map (keys stored as JSON string keys, values >= 1)."""
    payload: dict[str, int] = {}
    for key, value in (limits or {}).items():
        try:
            cid = int(key)
            lim = int(value)
        except (TypeError, ValueError):
            continue
        payload[str(cid)] = lim if lim >= 1 else 1
    LIMITS_PATH.parent.mkdir(parents=True, exist_ok=True)
    LIMITS_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )


def get_limit(customer_id) -> int:
    """The per-product unit cap for a customer, or :data:`DEFAULT_LIMIT` if unset."""
    try:
        cid = int(customer_id)
    except (TypeError, ValueError):
        return DEFAULT_LIMIT
    return load_limits().get(cid, DEFAULT_LIMIT)


def set_limit(customer_id, limit) -> None:
    """Validate (integer >= 1) and persist a single customer's limit."""
    cid = int(customer_id)
    lim = int(limit)
    if lim < 1:
        raise ValueError("Order quantity limit must be an integer >= 1.")
    limits = load_limits()
    limits[cid] = lim
    save_limits(limits)
