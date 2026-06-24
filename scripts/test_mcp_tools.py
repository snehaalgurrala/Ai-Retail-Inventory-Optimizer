#!/usr/bin/env python
"""Smoke test for the MCP tool catalog.

Exercises every allow-listed tool two ways and reports row counts:
  1. in-process via the registry (fast, no subprocess)
  2. over the stdio MCP server (the real chatbot transport), unless --in-process

Validation only: reads live Oracle data through the tools and checks each
returns the standard {summary, records, sources} shape. It does not modify data.

Usage:
    python scripts/test_mcp_tools.py
    python scripts/test_mcp_tools.py --in-process   # skip the stdio round-trip

Exit code 0 when every tool returns a well-formed payload, else 1.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.mcp import registry  # noqa: E402
from backend.mcp.context import clear_context  # noqa: E402


SAMPLE_ARGS = {
    "list_stores": {"limit": 5},
    "list_products": {"limit": 5},
    "list_suppliers": {"limit": 5},
    "validate_location": {"location": "Hyderabad"},
    "get_inventory_health": {},
    "get_low_stock_items": {"limit": 5},
    "get_overstock_items": {"limit": 5},
    "get_product_master": {"limit": 5},
    "get_products_below_reorder": {"limit": 5},
    "get_top_products": {"limit": 5, "metric": "revenue"},
    "get_bottom_products": {"limit": 5, "metric": "units"},
    "get_product_performance": {"limit": 5},
    "get_stockout_risk": {"limit": 5},
    "get_demand_forecast": {"limit": 5},
    "get_high_demand_items": {"limit": 5},
    "get_supplier_analysis": {"limit": 5},
    "get_procurement_risk": {"limit": 5},
    "get_top_customers": {"limit": 5, "metric": "revenue"},
    "get_customer_order_analysis": {"limit": 5},
    "get_customer_products": {"customer": "", "metric": "quantity", "limit": 5},
    "detect_abnormal_ordering": {"limit": 5},
    "get_customer_demand_trends": {"trend": "", "limit": 5},
    "get_dormant_accounts": {"limit": 5},
    "get_transfer_opportunities": {"limit": 5},
}


def _well_formed(payload: dict) -> bool:
    return (
        isinstance(payload, dict)
        and "summary" in payload
        and "records" in payload
        and "sources" in payload
    )


def _run_in_process() -> int:
    clear_context()
    failures = 0
    print("IN-PROCESS (registry) -------------------------------------------")
    for name in registry.TOOL_CATALOG:
        args = SAMPLE_ARGS.get(name, {"limit": 5})
        try:
            payload = registry.execute(name, args)
            ok = _well_formed(payload)
            failures += 0 if ok else 1
            flag = "OK  " if ok else "BAD "
            print(f"  {flag} {name:30s} records={len(payload.get('records', []))}")
        except Exception as error:  # noqa: BLE001
            failures += 1
            print(f"  FAIL {name:30s} {type(error).__name__}: {error}")
    return failures


def _run_stdio() -> int:
    from backend.mcp.client import get_client, reset_client

    failures = 0
    print("\nSTDIO MCP (subprocess) ------------------------------------------")
    try:
        client = get_client()
        names = client.list_tool_names()
        catalog = set(registry.TOOL_CATALOG)
        if set(names) != catalog:
            print(f"  ! tool list mismatch: server={sorted(names)} catalog={sorted(catalog)}")
            failures += 1
        for name in registry.TOOL_CATALOG:
            args = SAMPLE_ARGS.get(name, {"limit": 5})
            try:
                payload = client.call_tool(name, args)
                ok = _well_formed(payload)
                failures += 0 if ok else 1
                flag = "OK  " if ok else "BAD "
                print(f"  {flag} {name:30s} records={len(payload.get('records', []))}")
            except Exception as error:  # noqa: BLE001
                failures += 1
                print(f"  FAIL {name:30s} {type(error).__name__}: {error}")
    finally:
        reset_client()
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="MCP tool catalog smoke test.")
    parser.add_argument("--in-process", action="store_true", help="Skip the stdio round-trip.")
    args = parser.parse_args()

    print("=" * 64)
    print(f"MCP TOOL CATALOG SMOKE TEST  ({len(registry.TOOL_CATALOG)} tools)")
    print("=" * 64)
    failures = _run_in_process()
    if not args.in_process:
        failures += _run_stdio()

    print("\n" + "=" * 64)
    print("RESULT:", "PASS" if failures == 0 else f"FAIL ({failures} issue(s))")
    print("=" * 64)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
