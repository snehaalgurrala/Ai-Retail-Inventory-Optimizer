#!/usr/bin/env python
"""Oracle connectivity probe.

Verifies the Oracle connection works and the core BZ_MOCK_* tables are present
and populated. Read-only: issues COUNT(*) queries only. Run before parity
validation and before cutover.

Usage:
    python scripts/test_oracle_connection.py

Reads ORACLE_USER / ORACLE_PASSWORD / ORACLE_DSN from the environment
(see .env.example). Exit code 0 if connection + all tables PASS, else 1.
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from backend.db.config import get_oracle_config  # noqa: E402


TABLES = [
    "BZ_MOCK_PRODUCT",
    "BZ_MOCK_INVENTORY",
    "BZ_MOCK_BRANCH",
    "BZ_MOCK_SALES_HISTORY",
    "BZ_MOCK_CUSTOMER",
    "BZ_MOCK_ORDER_HEADER",
    "BZ_MOCK_ORDER_LINE",
]


def main() -> int:
    print("=" * 52)
    print("ORACLE CONNECTION VALIDATION")
    print("=" * 52)

    try:
        import oracledb
    except ImportError as error:
        print(f"FAIL: oracle driver not installed ({error})")
        print("      pip install oracledb")
        return 1

    cfg = get_oracle_config()
    if not cfg["user"] or not cfg["dsn"]:
        print("FAIL: ORACLE_USER / ORACLE_DSN are not set (see .env.example)")
        return 1

    try:
        connection = oracledb.connect(
            user=cfg["user"],
            password=cfg["password"],
            dsn=cfg["dsn"],
        )
    except Exception as error:
        print(f"FAIL: connection -> {type(error).__name__}: {error}")
        return 1

    print(f"PASS: connection established (dsn={cfg['dsn']})")
    print("-" * 52)

    failures = 0
    cursor = connection.cursor()
    for table in TABLES:
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            if count > 0:
                print(f"PASS: {table:<24} rows={count}")
            else:
                failures += 1
                print(f"FAIL: {table:<24} rows=0 (empty)")
        except Exception as error:
            failures += 1
            print(f"FAIL: {table:<24} -> {type(error).__name__}: {error}")

    cursor.close()
    connection.close()

    print("-" * 52)
    if failures == 0:
        print("VERDICT: PASS - Oracle reachable and core tables populated.")
        return 0
    print(f"VERDICT: FAIL - {failures} table check(s) failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
