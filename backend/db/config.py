"""Configuration for the data access layer.

The active backend is read from the ``DATA_BACKEND`` environment variable at call
time (not import time) so tests and runtime can switch it without re-importing.
Only ``csv`` is supported right now; ``oracle`` is reserved for a later phase.
"""

import os
from pathlib import Path

from dotenv import load_dotenv


# Load .env once for the whole data layer so DATA_BACKEND / ORACLE_* take effect
# for both the app and the validation scripts. override=False keeps any explicit
# environment variables (e.g. inline exports) authoritative.
load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)


DEFAULT_BACKEND = "csv"
SUPPORTED_BACKENDS = ("csv", "oracle")

# Inventory scope — how "current inventory" for a product is computed everywhere
# the platform displays or reasons about stock (Customer Intelligence, Order
# Simulator, abnormal-order emails, chatbot, recommendations, risk scores).
#   "network" -> SUM(stock) across ALL branches/warehouses for the product
#   "branch"  -> only the order's / customer's fulfilling branch
# Temporary business decision: until the client confirms their fulfillment model
# (single warehouse, multi-warehouse, pooled, or per-branch customers) the whole
# platform uses one consistent "network" view so figures match across every
# surface. Flip ``INVENTORY_SCOPE=branch`` later to switch without code changes.
DEFAULT_INVENTORY_SCOPE = "network"
SUPPORTED_INVENTORY_SCOPES = ("network", "branch")


def get_data_backend() -> str:
    """Return the currently selected data backend name (lower-cased)."""
    value = os.getenv("DATA_BACKEND", DEFAULT_BACKEND)
    return (value or DEFAULT_BACKEND).strip().lower()


def get_inventory_scope() -> str:
    """Return the active inventory scope ("network" or "branch").

    Read from the ``INVENTORY_SCOPE`` environment variable at call time so it can
    be switched without re-importing. Any unrecognised value falls back to the
    network-wide default.
    """
    value = (os.getenv("INVENTORY_SCOPE", DEFAULT_INVENTORY_SCOPE) or "").strip().lower()
    return value if value in SUPPORTED_INVENTORY_SCOPES else DEFAULT_INVENTORY_SCOPE


def get_oracle_config() -> dict:
    """Read Oracle connection settings from the environment.

    Only consulted when ``DATA_BACKEND=oracle``. Secrets live in the
    environment, never in code.
    """
    return {
        "user": os.getenv("ORACLE_USER", ""),
        "password": os.getenv("ORACLE_PASSWORD", ""),
        "dsn": os.getenv("ORACLE_DSN", ""),
        "pool_min": int(os.getenv("ORACLE_POOL_MIN", "1")),
        "pool_max": int(os.getenv("ORACLE_POOL_MAX", "4")),
    }
