#!/usr/bin/env python
"""Runtime diagnostic for Oracle loading.

Run this with the SAME interpreter that launches the app (e.g. the one behind
``streamlit``) to confirm the Oracle driver is visible there:

    python scripts/diagnose_runtime.py
    # or, to match Streamlit exactly:
    python -m streamlit  ...   ->  use that interpreter to run this script

Reports the interpreter, DATA_BACKEND / ORACLE_DSN (from .env), and whether
``oracledb`` is importable. Exit code 1 if DATA_BACKEND=oracle but the driver
is missing.
"""

import importlib.util
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from backend.db.config import get_data_backend, get_oracle_config  # loads .env  # noqa: E402


def main() -> int:
    backend = get_data_backend()
    cfg = get_oracle_config()
    oracledb_installed = importlib.util.find_spec("oracledb") is not None

    print("=" * 52)
    print("RUNTIME DIAGNOSTIC")
    print("=" * 52)
    print(f"Python executable : {sys.executable}")
    print(f"Python version    : {sys.version.split()[0]}")
    print(f"DATA_BACKEND      : {backend}")
    print(f"ORACLE_DSN        : {cfg['dsn'] or '(not set)'}")
    print(f"oracledb installed: {'Yes' if oracledb_installed else 'No'}")
    print("=" * 52)

    if backend == "oracle" and not oracledb_installed:
        print("ERROR: Oracle driver not installed. Run: pip install oracledb")
        print(f"       (install into THIS interpreter: {sys.executable})")
        return 1

    print("Runtime OK for backend:", backend)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
