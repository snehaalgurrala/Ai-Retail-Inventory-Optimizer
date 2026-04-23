from importlib import import_module
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


REQUIRED_FOLDERS = [
    "backend",
    "backend/agents",
    "backend/services",
    "backend/utils",
    "frontend",
    "frontend/pages",
    "frontend/utils",
    "data",
    "data/raw",
    "data/processed",
]

RAW_FILES = [
    "inventory.csv",
    "products.csv",
    "sales.csv",
    "stores.csv",
    "suppliers.csv",
    "transactions.csv",
]

PROCESSED_FILES = [
    "current_inventory.csv",
    "sales_summary.csv",
    "product_performance.csv",
    "store_inventory_summary.csv",
    "low_stock_items.csv",
    "stockout_risk_items.csv",
    "overstock_items.csv",
    "dead_stock_candidates.csv",
    "high_demand_items.csv",
    "slow_moving_items.csv",
    "recommendations.csv",
]

KEY_MODULES = [
    "backend.utils.data_loader",
    "backend.services.data_processor",
    "backend.services.inventory_analyzer",
    "backend.services.recommendation_engine",
    "backend.agents.demand_agent",
    "backend.agents.pricing_agent",
    "backend.agents.transfer_agent",
    "backend.agents.risk_agent",
    "backend.agents.orchestrator_agent",
    "backend.main",
]

FRONTEND_PAGES = [
    "frontend/app.py",
    "frontend/pages/1_Inventory.py",
    "frontend/pages/2_Sales.py",
    "frontend/pages/3_Recommendations.py",
    "frontend/pages/4_Chatbot.py",
]


def check_path(path: Path, label: str) -> bool:
    """Print a status line for one path."""
    if path.exists():
        print(f"[OK] {label}: {path}")
        return True

    print(f"[MISSING] {label}: {path}")
    return False


def check_csv(path: Path, label: str) -> bool:
    """Check that a CSV exists and can be read."""
    if not path.exists():
        print(f"[MISSING] {label}: {path}")
        return False

    try:
        df = pd.read_csv(path)
    except Exception as error:
        print(f"[ERROR] {label}: could not read {path} ({error})")
        return False

    print(f"[OK] {label}: {path} ({len(df):,} rows)")
    return True


def check_import(module_name: str) -> bool:
    """Check that a Python module imports successfully."""
    try:
        import_module(module_name)
    except Exception as error:
        print(f"[ERROR] import {module_name}: {error}")
        return False

    print(f"[OK] import {module_name}")
    return True


def main() -> int:
    print("AI Retail Inventory Optimizer - Project Status")
    print("=" * 52)

    checks = []

    print("\nRequired folders")
    for folder in REQUIRED_FOLDERS:
        checks.append(check_path(PROJECT_ROOT / folder, folder))

    print("\nRaw CSV files")
    for filename in RAW_FILES:
        checks.append(check_csv(PROJECT_ROOT / "data" / "raw" / filename, filename))

    print("\nProcessed outputs")
    for filename in PROCESSED_FILES:
        checks.append(
            check_csv(PROJECT_ROOT / "data" / "processed" / filename, filename)
        )

    print("\nKey module imports")
    for module_name in KEY_MODULES:
        checks.append(check_import(module_name))

    print("\nFrontend pages")
    for page in FRONTEND_PAGES:
        checks.append(check_path(PROJECT_ROOT / page, page))

    passed = sum(1 for check in checks if check)
    total = len(checks)

    print("\nSummary")
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("Project status: ready")
        return 0

    print("Project status: needs attention")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
