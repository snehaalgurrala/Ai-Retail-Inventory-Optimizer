"""CSV implementation of the data access backend.

This is the only module in the data layer that calls ``pandas.read_csv``. It
returns DataFrames exactly as ``read_csv`` produces them — no column, dtype, or
shape transformation — so callers get byte-for-byte the same structure they did
before the repository existed.
"""

from pathlib import Path

import pandas as pd

from backend.db import paths


class CsvBackend:
    """Reads datasets from CSV files under ``data/raw`` and ``data/processed``."""

    @staticmethod
    def _read(path: Path, safe: bool) -> pd.DataFrame:
        """Read one CSV.

        strict (``safe=False``): raises if the file is missing, matching the
        historical ``pd.read_csv(path)`` behavior.

        safe (``safe=True``): returns an empty DataFrame if the file is missing
        or cannot be parsed, matching the historical ``_safe_read_csv`` wrappers.
        """
        if safe:
            if not path.exists():
                return pd.DataFrame()
            try:
                return pd.read_csv(path)
            except Exception:
                return pd.DataFrame()
        return pd.read_csv(path)

    def read_raw(self, name: str, *, safe: bool = False) -> pd.DataFrame:
        """Read a raw source table by logical name."""
        return self._read(paths.raw_path(name), safe)

    def read_processed(self, name: str, *, safe: bool = False) -> pd.DataFrame:
        """Read a processed dataset by logical name."""
        return self._read(paths.processed_path(name), safe)
