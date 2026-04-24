from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from backend.agents.orchestrator_agent import run_agent_graph  # noqa: E402
from backend.services.order_service import (  # noqa: E402
    INVENTORY_FILE,
    ORDERS_FILE,
    PRODUCTS_FILE,
    SALES_FILE,
    STORES_FILE,
    TRANSACTIONS_FILE,
    get_available_products_by_store,
    place_order,
)


AGENT_OUTPUTS_FILE = PROJECT_ROOT / "data" / "processed" / "agent_outputs.csv"
RECOMMENDATIONS_FILE = PROJECT_ROOT / "data" / "processed" / "recommendations.csv"
ORCHESTRATOR_SUMMARY_FILE = (
    PROJECT_ROOT / "data" / "processed" / "orchestrator_summary.csv"
)

FILES_TO_BACKUP = [
    INVENTORY_FILE,
    SALES_FILE,
    TRANSACTIONS_FILE,
    ORDERS_FILE,
    AGENT_OUTPUTS_FILE,
    RECOMMENDATIONS_FILE,
    ORCHESTRATOR_SUMMARY_FILE,
]


@dataclass
class StepResult:
    step: str
    passed: bool
    detail: str


def read_csv_safe(file_path: Path) -> pd.DataFrame:
    """Read a CSV safely or return an empty dataframe."""
    if not file_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(file_path)
    except Exception:
        return pd.DataFrame()


def backup_files(backup_dir: Path) -> None:
    """Copy target files to a backup directory."""
    backup_dir.mkdir(parents=True, exist_ok=True)
    for file_path in FILES_TO_BACKUP:
        if not file_path.exists():
            continue
        target = backup_dir / file_path.name
        shutil.copy2(file_path, target)


def restore_files(backup_dir: Path) -> None:
    """Restore backed up files and remove files created during the test."""
    for file_path in FILES_TO_BACKUP:
        backup_file = backup_dir / file_path.name
        if backup_file.exists():
            file_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(backup_file, file_path)
        elif file_path.exists():
            file_path.unlink()


def find_test_product() -> tuple[str, str, str]:
    """Pick a store and product that currently has available stock."""
    stores = read_csv_safe(STORES_FILE)
    if stores.empty or "store_id" not in stores.columns:
        raise RuntimeError("stores.csv is missing or does not contain store_id.")

    for store_id in stores["store_id"].astype(str).tolist():
        store_products = get_available_products_by_store(store_id)
        if store_products.empty:
            continue

        stock_levels = pd.to_numeric(
            store_products.get("stock_level", 0),
            errors="coerce",
        ).fillna(0)
        available_rows = store_products[stock_levels >= 1].copy()
        if available_rows.empty:
            continue

        selected_row = available_rows.sort_values(
            ["stock_level", "product_id"],
            ascending=[False, True],
        ).iloc[0]
        return (
            str(store_id),
            str(selected_row.get("product_id", "")),
            str(selected_row.get("product_name", "")),
        )

    raise RuntimeError("No store/product pair with available stock was found.")


def get_inventory_stock(store_id: str, product_id: str) -> int:
    """Read one stock level from inventory.csv."""
    inventory = read_csv_safe(INVENTORY_FILE)
    if inventory.empty:
        return 0

    row = inventory[
        inventory["store_id"].astype(str).eq(str(store_id))
        & inventory["product_id"].astype(str).eq(str(product_id))
    ]
    if row.empty:
        return 0

    return int(
        pd.to_numeric(row["stock_level"], errors="coerce").fillna(0).iloc[0]
    )


def count_matching_rows(df: pd.DataFrame, filters: dict[str, str | int]) -> int:
    """Count rows matching a small filter dictionary."""
    if df.empty:
        return 0

    filtered = df.copy()
    for column, value in filters.items():
        if column not in filtered.columns:
            return 0
        filtered = filtered[filtered[column].astype(str) == str(value)]
    return int(len(filtered))


def run_test_flow(keep_changes: bool) -> list[StepResult]:
    """Execute the full order-to-agent flow and collect step results."""
    results: list[StepResult] = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = PROJECT_ROOT / "data" / "processed" / "test_backups" / timestamp

    inventory_before = pd.DataFrame()
    sales_before = pd.DataFrame()
    transactions_before = pd.DataFrame()
    recommendations_before = pd.DataFrame()

    store_id = ""
    product_id = ""
    product_name = ""
    order_result: dict = {}

    try:
        backup_files(backup_dir)

        inventory_before = read_csv_safe(INVENTORY_FILE)
        results.append(
            StepResult(
                "1. Load current inventory",
                not inventory_before.empty,
                f"Loaded {len(inventory_before)} inventory rows."
                if not inventory_before.empty
                else "inventory.csv could not be loaded.",
            )
        )

        try:
            store_id, product_id, product_name = find_test_product()
            results.append(
                StepResult(
                    "2. Pick a store and product with available stock",
                    True,
                    f"Selected store={store_id}, product={product_id} ({product_name}).",
                )
            )
        except Exception as error:
            results.append(
                StepResult(
                    "2. Pick a store and product with available stock",
                    False,
                    str(error),
                )
            )
            return results

        stock_before = get_inventory_stock(store_id, product_id)
        sales_before = read_csv_safe(SALES_FILE)
        transactions_before = read_csv_safe(TRANSACTIONS_FILE)
        recommendations_before = read_csv_safe(RECOMMENDATIONS_FILE)

        order_result = place_order(store_id, product_id, 1)
        results.append(
            StepResult(
                "3. Place a test order with quantity 1",
                bool(order_result.get("success", False)),
                order_result.get("message", "No response returned."),
            )
        )
        if not order_result.get("success", False):
            return results

        stock_after = get_inventory_stock(store_id, product_id)
        expected_stock = stock_before - 1
        results.append(
            StepResult(
                "4. Verify inventory.csv quantity reduced",
                stock_after == expected_stock,
                f"Stock before={stock_before}, after={stock_after}, expected={expected_stock}.",
            )
        )

        sales_after = read_csv_safe(SALES_FILE)
        sale_id = order_result.get("order_data", {}).get("sale_id", "")
        sale_filters = {
            "sale_id": sale_id,
            "product_id": product_id,
            "store_id": store_id,
            "quantity_sold": 1,
        }
        sales_row_added = len(sales_after) == len(sales_before) + 1
        sales_row_match = count_matching_rows(sales_after, sale_filters) >= 1
        results.append(
            StepResult(
                "5. Verify sales.csv new row added",
                sales_row_added and sales_row_match,
                (
                    f"Rows before={len(sales_before)}, after={len(sales_after)}, "
                    f"sale_id={sale_id}."
                ),
            )
        )

        transactions_after = read_csv_safe(TRANSACTIONS_FILE)
        transaction_id = order_result.get("order_data", {}).get("transaction_id", "")
        transaction_filters = {
            "transaction_id": transaction_id,
            "product_id": product_id,
            "store_id": store_id,
            "transaction_type": "sale",
            "quantity": 1,
        }
        transaction_row_added = len(transactions_after) == len(transactions_before) + 1
        transaction_row_match = (
            count_matching_rows(transactions_after, transaction_filters) >= 1
        )
        results.append(
            StepResult(
                "6. Verify transactions.csv new row added",
                transaction_row_added and transaction_row_match,
                (
                    f"Rows before={len(transactions_before)}, "
                    f"after={len(transactions_after)}, transaction_id={transaction_id}."
                ),
            )
        )

        final_state = run_agent_graph(save_output=True)
        agent_run_ok = bool(final_state.get("combined_output", {}))
        results.append(
            StepResult(
                "7. Run orchestrator agents",
                agent_run_ok,
                (
                    f"Run time={final_state.get('combined_output', {}).get('run_time', '')}."
                    if agent_run_ok
                    else "No orchestrator output was returned."
                ),
            )
        )

        agent_outputs_after = read_csv_safe(AGENT_OUTPUTS_FILE)
        results.append(
            StepResult(
                "8. Verify agent_outputs.csv created",
                AGENT_OUTPUTS_FILE.exists() and not agent_outputs_after.empty,
                f"agent_outputs rows={len(agent_outputs_after)}.",
            )
        )

        recommendations_after = read_csv_safe(RECOMMENDATIONS_FILE)
        recommendations_updated = (
            RECOMMENDATIONS_FILE.exists()
            and not recommendations_after.empty
            and (
                len(recommendations_after) > 0
                or len(recommendations_before) != len(recommendations_after)
            )
        )
        results.append(
            StepResult(
                "9. Verify recommendations.csv updated",
                recommendations_updated,
                (
                    f"Rows before={len(recommendations_before)}, "
                    f"after={len(recommendations_after)}."
                ),
            )
        )

        passed_steps = sum(result.passed for result in results)
        results.append(
            StepResult(
                "10. Print pass/fail result for each step",
                True,
                f"Collected {passed_steps}/{len(results)} passing steps before summary.",
            )
        )

        return results
    finally:
        if keep_changes:
            print(f"[INFO] Kept modified CSV files. Backup remains at: {backup_dir}")
        else:
            restore_files(backup_dir)
            print(f"[INFO] Restored CSV files from backup: {backup_dir}")


def print_results(results: list[StepResult]) -> int:
    """Print readable test results and return shell exit code."""
    print("Order + Agent Flow Test")
    print("=" * 40)

    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"[{status}] {result.step}")
        print(f"       {result.detail}")

    failed = [result for result in results if not result.passed]
    print("=" * 40)
    print(f"Passed: {len(results) - len(failed)}/{len(results)}")
    print(f"Failed: {len(failed)}")
    return 1 if failed else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test the full order placement and agent refresh flow. "
            "By default, all touched CSV files are restored after the test."
        )
    )
    parser.add_argument(
        "--keep-changes",
        action="store_true",
        help="Keep modified CSV files after the test instead of restoring from backup.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results = run_test_flow(keep_changes=args.keep_changes)
    return print_results(results)


if __name__ == "__main__":
    raise SystemExit(main())
