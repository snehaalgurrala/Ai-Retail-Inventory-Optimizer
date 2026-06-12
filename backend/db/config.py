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


def get_data_backend() -> str:
    """Return the currently selected data backend name (lower-cased)."""
    value = os.getenv("DATA_BACKEND", DEFAULT_BACKEND)
    return (value or DEFAULT_BACKEND).strip().lower()


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
