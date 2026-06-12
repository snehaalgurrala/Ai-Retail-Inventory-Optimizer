"""Live-from-Oracle data context shared by all MCP tools.

Every tool reads through this module, which loads the raw Oracle tables once via
``repository`` and lazily builds the derived views the analytics services need.
Nothing here touches processed CSVs: derived datasets (low-stock, predictions,
rankings) are recomputed live so Oracle stays the single source of truth.

Reads are strict: an Oracle failure raises ``OracleUnavailable`` rather than
returning a silent empty frame, so the chatbot reports a real error instead of a
misleading "no data" answer.
"""

from __future__ import annotations

import time
from typing import Any

import pandas as pd

from backend.db import repository
from backend.db.config import get_data_backend
from backend.mcp import config
from backend.services.store_inventory_service import build_store_inventory_view
from backend.services.inventory_prediction_service import build_predictive_inventory_view


RAW_TABLES = ("inventory", "products", "stores", "sales", "suppliers", "transactions")

# Logical dataset name -> Oracle table it is sourced from (for "sources" output).
ORACLE_SOURCE = {
    "inventory": "BZ_MOCK_INVENTORY",
    "products": "BZ_MOCK_PRODUCT",
    "stores": "BZ_MOCK_BRANCH",
    "sales": "BZ_MOCK_SALES_HISTORY",
    "suppliers": "BZ_MOCK_SUPPLIER",
    "transactions": "BZ_MOCK_INVENTORY_TRANSACTION",
}


class OracleUnavailable(RuntimeError):
    """Raised when the raw Oracle tables cannot be read for a tool call."""


class OracleContext:
    """A single snapshot of raw Oracle frames plus lazily-built derived views."""

    def __init__(self, frames: dict[str, pd.DataFrame]):
        self.frames = frames
        self._store_view: pd.DataFrame | None = None
        self._predictive_view: pd.DataFrame | None = None

    # -- raw frames ---------------------------------------------------------
    def raw(self, name: str) -> pd.DataFrame:
        return self.frames.get(name, pd.DataFrame()).copy()

    # -- derived views (built once, reused across tools) --------------------
    def store_inventory_view(self) -> pd.DataFrame:
        if self._store_view is None:
            self._store_view = build_store_inventory_view(
                self.frames["inventory"],
                self.frames["products"],
                self.frames["stores"],
                self.frames["suppliers"],
                self.frames["sales"],
            )
        return self._store_view.copy()

    def predictive_view(self) -> pd.DataFrame:
        if self._predictive_view is None:
            self._predictive_view = build_predictive_inventory_view(
                self.frames["inventory"],
                self.frames["products"],
                self.frames["stores"],
                self.frames["sales"],
                self.frames["suppliers"],
            )
        return self._predictive_view.copy()


def _load_frames() -> dict[str, pd.DataFrame]:
    """Read all raw tables from Oracle. Strict: raises on any read failure."""
    frames: dict[str, pd.DataFrame] = {}
    for name in RAW_TABLES:
        try:
            frames[name] = repository.load_raw(name, safe=False)
        except Exception as error:  # surface, never swallow into empty
            raise OracleUnavailable(
                f"Could not read '{name}' from the {get_data_backend()} backend: "
                f"{type(error).__name__}: {error}"
            ) from error
    return frames


# Module-level short-lived cache so a multi-tool turn reads Oracle once.
_CACHE: dict[str, Any] = {"context": None, "loaded_at": 0.0}


def get_context(force_reload: bool = False) -> OracleContext:
    """Return a warm OracleContext, reloading once the short TTL elapses."""
    now = time.monotonic()
    ttl = config.context_ttl_seconds()
    cached = _CACHE["context"]
    if (
        not force_reload
        and cached is not None
        and (now - _CACHE["loaded_at"]) < ttl
    ):
        return cached

    context = OracleContext(_load_frames())
    _CACHE["context"] = context
    _CACHE["loaded_at"] = now
    return context


def clear_context() -> None:
    """Drop the cached context (used by tests and after data changes)."""
    _CACHE["context"] = None
    _CACHE["loaded_at"] = 0.0


# ---------------------------------------------------------------------------
# Shared helpers for building compact, JSON-safe tool output.
# ---------------------------------------------------------------------------
def num(df: pd.DataFrame, column: str) -> pd.Series:
    """Numeric view of a column, 0-filled, safe when the column is absent."""
    if column not in df.columns:
        return pd.Series(0, index=df.index, dtype="float64")
    return pd.to_numeric(df[column], errors="coerce").fillna(0)


def records(df: pd.DataFrame, columns: list[str], limit: int) -> list[dict]:
    """Return up to ``limit`` compact, JSON-serializable rows."""
    if df.empty:
        return []
    available = [column for column in columns if column in df.columns]
    out = df[available].copy()
    if limit and limit > 0:
        out = out.head(limit)
    out = out.replace([float("inf"), float("-inf")], None)
    out = out.astype(object).where(pd.notnull(out), None)
    return out.to_dict(orient="records")


def sources(*dataset_names: str) -> list[dict]:
    """Build the ``sources`` list, tagging each with its Oracle table."""
    return [
        {"dataset": name, "oracle_table": ORACLE_SOURCE.get(name, "")}
        for name in dataset_names
    ]


def clamp_limit(limit: int) -> int:
    """Clamp a caller-supplied limit to the configured bounds."""
    if not limit or limit <= 0:
        return config.default_record_limit()
    return min(int(limit), config.max_record_limit())
