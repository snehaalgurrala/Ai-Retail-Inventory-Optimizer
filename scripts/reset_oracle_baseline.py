#!/usr/bin/env python
"""Reset the Oracle BZ_MOCK_* dataset back to the provided seed baseline.

Restores the 10 baseline tables (BRANCH, CUSTOMER, PRODUCT, PRICE_HISTORY,
CONTRACT_PRICE, INVENTORY, ORDER_HEADER, ORDER_LINE, SALES_HISTORY,
COMPETITOR_PRICE) and the 2 AI views to exactly what the seed script defines, by
wiping their rows and re-inserting strictly from
``sql/bunzl_walmart_like_mock_b2b_inventory_pricing_oracle.sql``.

The three non-seed tables that later development added and that other features
still depend on (BZ_MOCK_SUPPLIER, BZ_MOCK_BRANCH_CAPACITY,
BZ_MOCK_INVENTORY_TRANSACTION) are KEPT. Their reference data (SUPPLIER,
BRANCH_CAPACITY) is left untouched; the INVENTORY_TRANSACTION movement ledger is
cleared so the baseline starts from a clean ledger.

Mechanics (one transaction):
  1. Disable every BZ_MOCK_* foreign key so rows can be deleted in any order.
  2. DELETE the 10 baseline tables + clear BZ_MOCK_INVENTORY_TRANSACTION.
  3. Execute the seed's INSERT statements + CREATE OR REPLACE VIEW statements
     (CREATE TABLE / CREATE INDEX are skipped: the tables already exist).
  4. Re-enable (and validate) the foreign keys, then COMMIT.
  5. Print row counts for verification.

Usage:
    python scripts/reset_oracle_baseline.py            # do the reset
    python scripts/reset_oracle_baseline.py --dry-run  # parse only, no DB writes

Reads ORACLE_* from the environment (.env). Requires DATA_BACKEND=oracle config.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from backend.db.config import get_oracle_config  # noqa: E402


SEED_FILE = PROJECT_ROOT / "sql" / "bunzl_walmart_like_mock_b2b_inventory_pricing_oracle.sql"

# Baseline tables to wipe, in child -> parent order (used even though FKs are
# disabled, so a partial/aborted run still respects referential order).
BASELINE_TABLES = [
    "BZ_MOCK_ORDER_LINE",
    "BZ_MOCK_ORDER_HEADER",
    "BZ_MOCK_SALES_HISTORY",
    "BZ_MOCK_COMPETITOR_PRICE",
    "BZ_MOCK_CONTRACT_PRICE",
    "BZ_MOCK_PRICE_HISTORY",
    "BZ_MOCK_INVENTORY",
    "BZ_MOCK_PRODUCT",
    "BZ_MOCK_CUSTOMER",
    "BZ_MOCK_BRANCH",
]

# Non-seed table whose rows we clear (movement ledger -> empty at baseline).
LEDGER_TABLE = "BZ_MOCK_INVENTORY_TRANSACTION"

# Expected baseline row counts (for the final verification print).
EXPECTED = {
    "BZ_MOCK_BRANCH": 5,
    "BZ_MOCK_CUSTOMER": 12,
    "BZ_MOCK_PRODUCT": 16,
    "BZ_MOCK_PRICE_HISTORY": 80,
    "BZ_MOCK_CONTRACT_PRICE": 15,
    "BZ_MOCK_INVENTORY": 80,
    "BZ_MOCK_ORDER_HEADER": 30,
    "BZ_MOCK_ORDER_LINE": 103,
    "BZ_MOCK_SALES_HISTORY": 450,
    "BZ_MOCK_COMPETITOR_PRICE": 32,
}


def _strip_comments(sql: str) -> str:
    """Remove /* block */ and -- line comments so statements split cleanly."""
    sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
    lines = []
    for line in sql.splitlines():
        stripped = line.strip()
        if stripped.startswith("--"):
            continue
        lines.append(line)
    return "\n".join(lines)


def parse_statements(sql_text: str) -> dict[str, list[str]]:
    """Split the seed into INSERT and CREATE-VIEW statements (others ignored)."""
    cleaned = _strip_comments(sql_text)
    raw = [s.strip() for s in cleaned.split(";")]
    inserts: list[str] = []
    views: list[str] = []
    for stmt in raw:
        if not stmt:
            continue
        head = stmt.lstrip().upper()
        if head.startswith("INSERT"):
            inserts.append(stmt)
        elif head.startswith("CREATE OR REPLACE VIEW"):
            views.append(stmt)
        # CREATE TABLE / CREATE INDEX / SELECT / COMMIT are intentionally skipped.
    return {"inserts": inserts, "views": views}


def main() -> int:
    dry_run = "--dry-run" in sys.argv

    if not SEED_FILE.exists():
        print(f"FAIL: seed file not found: {SEED_FILE}")
        return 1

    statements = parse_statements(SEED_FILE.read_text(encoding="utf-8"))
    inserts, views = statements["inserts"], statements["views"]
    print(f"Parsed seed: {len(inserts)} INSERT statements, {len(views)} views.")
    if len(inserts) != sum(EXPECTED.values()):
        print(
            f"WARN: parsed INSERT count ({len(inserts)}) != expected "
            f"({sum(EXPECTED.values())}). Continuing, but verify counts below."
        )

    if dry_run:
        print("--dry-run: not connecting to Oracle; no changes made.")
        return 0

    try:
        import oracledb
    except ImportError as error:
        print(f"FAIL: oracle driver not installed ({error}). pip install oracledb")
        return 1

    cfg = get_oracle_config()
    connection = oracledb.connect(
        user=cfg["user"], password=cfg["password"], dsn=cfg["dsn"]
    )
    cursor = connection.cursor()
    print(f"Connected (dsn={cfg['dsn']}).")

    # 1. Disable every BZ_MOCK_* foreign key.
    cursor.execute(
        "SELECT table_name, constraint_name FROM user_constraints "
        "WHERE constraint_type = 'R' AND table_name LIKE 'BZ_MOCK%' "
        "AND status = 'ENABLED'"
    )
    fks = cursor.fetchall()
    for table_name, constraint_name in fks:
        cursor.execute(f'ALTER TABLE {table_name} DISABLE CONSTRAINT {constraint_name}')
    print(f"Disabled {len(fks)} foreign key constraint(s).")

    try:
        # 2. Wipe baseline tables + clear the movement ledger.
        for table in BASELINE_TABLES + [LEDGER_TABLE]:
            cursor.execute(f"DELETE FROM {table}")
        print(f"Cleared {len(BASELINE_TABLES)} baseline tables + {LEDGER_TABLE}.")

        # 3. Re-seed: INSERTs then views.
        for stmt in inserts:
            cursor.execute(stmt)
        print(f"Executed {len(inserts)} INSERT statements.")
        for stmt in views:
            cursor.execute(stmt)
        print(f"(Re)created {len(views)} views.")
    except Exception:
        connection.rollback()
        # best-effort: re-enable FKs we disabled before surfacing the error
        for table_name, constraint_name in fks:
            try:
                cursor.execute(
                    f'ALTER TABLE {table_name} ENABLE CONSTRAINT {constraint_name}'
                )
            except Exception:
                pass
        raise

    # 4. Re-enable + validate the foreign keys, then commit.
    for table_name, constraint_name in fks:
        cursor.execute(f'ALTER TABLE {table_name} ENABLE CONSTRAINT {constraint_name}')
    print(f"Re-enabled {len(fks)} foreign key constraint(s).")
    connection.commit()
    print("COMMIT done.")

    # 5. Verify.
    print("-" * 52)
    ok = True
    for table, expected in EXPECTED.items():
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        actual = cursor.fetchone()[0]
        flag = "OK " if actual == expected else "BAD"
        if actual != expected:
            ok = False
        print(f"{flag} {table:<28} {actual:>5} (expected {expected})")
    cursor.execute(f"SELECT COUNT(*) FROM {LEDGER_TABLE}")
    print(f"    {LEDGER_TABLE:<28} {cursor.fetchone()[0]:>5} (ledger cleared)")
    # ORDER_ID range sanity.
    cursor.execute("SELECT MIN(ORDER_ID), MAX(ORDER_ID) FROM BZ_MOCK_ORDER_HEADER")
    lo, hi = cursor.fetchone()
    print(f"    ORDER_ID range: {lo}..{hi} (expected 9001..9030)")
    if (lo, hi) != (9001, 9030):
        ok = False

    cursor.close()
    connection.close()
    print("-" * 52)
    print("VERDICT:", "PASS - baseline restored." if ok else "FAIL - counts off.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
