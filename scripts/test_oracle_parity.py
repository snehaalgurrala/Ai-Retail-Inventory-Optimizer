#!/usr/bin/env python
"""Oracle <-> CSV backend parity validation (pre-cutover gate).

Proves that ``OracleBackend`` is a drop-in replacement for ``CsvBackend`` at the
data-access boundary, across products / inventory / sales / stores / suppliers /
transactions. Validation only: reads through the two backends and compares
structure. It does not modify any application code or data.

Usage:
    python scripts/test_oracle_parity.py                 # needs a live Oracle
    python scripts/test_oracle_parity.py --allow-offline # static contract only
    python scripts/test_oracle_parity.py --require-oracle # fail if Oracle is down

Exit codes:
    0  parity validated (live), or static contract OK with --allow-offline
    1  a real parity FAILURE (schema / dtype / null-key / column-count / join)
    2  Oracle unavailable and offline not allowed

Row counts are reported but NOT a pass/fail criterion: the CSV and Oracle
sources legitimately hold different data. Parity is judged on structure.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from backend.db.csv_backend import CsvBackend  # noqa: E402
from backend.db.oracle_backend import COLUMNS as ORACLE_COLUMNS  # noqa: E402
from backend.db.oracle_backend import OracleBackend  # noqa: E402


DATASETS = ["products", "inventory", "sales", "stores", "suppliers", "transactions"]

# Application joins to validate (left_dataset, left_key, right_dataset, right_key).
JOINS = [
    ("inventory", "product_id", "products", "product_id"),
    ("sales", "product_id", "products", "product_id"),
    ("inventory", "store_id", "stores", "store_id"),
    ("transactions", "product_id", "products", "product_id"),
]

# Columns that must never be null/blank in either backend (identifiers / join keys).
KEY_COLUMNS = {
    "products": ["product_id"],
    "inventory": ["product_id", "store_id"],
    "sales": ["product_id", "store_id"],
    "stores": ["store_id"],
    "suppliers": ["supplier_id"],
    "transactions": ["product_id", "store_id"],
}

# Columns the application expects to parse as dates ('YYYY-MM-DD' strings).
DATE_COLUMNS = {
    "sales": ["date"],
    "inventory": ["last_updated"],
    "transactions": ["date"],
}


def _dtype_category(series: pd.Series) -> str:
    """Category that matters for drop-in compatibility (int/float both numeric)."""
    kind = series.dtype.kind
    if kind in "iuf":
        return "numeric"
    if kind == "M":
        return "datetime"
    if kind == "b":
        return "bool"
    return "text"


def _looks_like_iso_date(series: pd.Series) -> bool:
    """True if non-empty values parse as dates (accepts 'YYYY-MM-DD' strings)."""
    values = series.dropna().astype(str)
    values = values[values.str.strip() != ""]
    if values.empty:
        return True
    parsed = pd.to_datetime(values, errors="coerce", format="%Y-%m-%d")
    return parsed.notna().all()


@dataclass
class Result:
    name: str
    csv_shape: tuple
    oracle_shape: tuple | None
    required_ok: bool
    names_ok: bool
    order_ok: bool
    dtype_ok: bool
    date_ok: bool
    null_ok: bool
    colcount_ok: bool
    warnings: list = field(default_factory=list)
    dtype_mismatches: list = field(default_factory=list)
    null_detail: str = ""
    detail: str = ""

    @property
    def status(self) -> str:
        if self.oracle_shape is None:
            return "SKIPPED"
        if all(
            [
                self.required_ok,
                self.names_ok,
                self.order_ok,
                self.dtype_ok,
                self.date_ok,
                self.null_ok,
                self.colcount_ok,
            ]
        ):
            return "PASS"
        return "FAIL"


def _load_all(backend) -> dict[str, pd.DataFrame]:
    return {name: backend.read_raw(name) for name in DATASETS}


def _null_blank_count(series: pd.Series) -> int:
    nulls = int(series.isna().sum())
    blanks = int((series.astype(str).str.strip() == "").sum())
    return nulls + blanks


def _compare(name: str, csv_df: pd.DataFrame, oracle_df: pd.DataFrame | None) -> Result:
    csv_cols = list(csv_df.columns)
    required = ORACLE_COLUMNS.get(name, csv_cols)

    if oracle_df is None:
        # Offline: validate the declared Oracle column contract against CSV.
        contract = ORACLE_COLUMNS.get(name, [])
        return Result(
            name=name,
            csv_shape=csv_df.shape,
            oracle_shape=None,
            required_ok=set(required).issubset(set(contract)),
            names_ok=set(csv_cols) == set(contract),
            order_ok=csv_cols == contract,
            dtype_ok=False,
            date_ok=False,
            null_ok=False,
            colcount_ok=len(csv_cols) == len(contract),
            detail="" if csv_cols == contract else f"contract={contract} csv={csv_cols}",
        )

    oracle_cols = list(oracle_df.columns)
    required_ok = set(required).issubset(set(oracle_cols))
    names_ok = set(csv_cols) == set(oracle_cols)
    order_ok = csv_cols == oracle_cols
    colcount_ok = csv_df.shape[1] == oracle_df.shape[1]

    # Dtype categories over the CSV columns.
    dtype_ok = True
    dtype_mismatches = []
    for column in csv_cols:
        if column not in oracle_df.columns:
            dtype_ok = False
            dtype_mismatches.append((column, str(csv_df[column].dtype), "MISSING"))
            continue
        csv_cat = _dtype_category(csv_df[column])
        ora_cat = _dtype_category(oracle_df[column])
        if csv_cat != ora_cat:
            dtype_ok = False
            dtype_mismatches.append(
                (column, str(csv_df[column].dtype), str(oracle_df[column].dtype))
            )

    # Date-format check on the expected date columns.
    date_ok = True
    for column in DATE_COLUMNS.get(name, []):
        if column in oracle_df.columns and not _looks_like_iso_date(oracle_df[column]):
            date_ok = False

    # Null-safe identifiers / join keys in both backends.
    null_ok = True
    null_notes = []
    for key in KEY_COLUMNS.get(name, []):
        for label, frame in (("csv", csv_df), ("oracle", oracle_df)):
            if key in frame.columns:
                count = _null_blank_count(frame[key])
                if count > 0:
                    null_ok = False
                    null_notes.append(f"{label}.{key}={count}")

    # Soft warnings (do not fail parity).
    warnings = []
    if csv_df.shape[0] != oracle_df.shape[0]:
        warnings.append(
            f"row count differs (CSV {csv_df.shape[0]} vs Oracle {oracle_df.shape[0]}) "
            "- expected for distinct datasets"
        )
    for column in oracle_cols:
        if column not in KEY_COLUMNS.get(name, []) and column in csv_df.columns:
            if _null_blank_count(oracle_df[column]) > 0 and _null_blank_count(csv_df[column]) == 0:
                warnings.append(f"non-key nulls in oracle.{column}")

    return Result(
        name=name,
        csv_shape=csv_df.shape,
        oracle_shape=oracle_df.shape,
        required_ok=required_ok,
        names_ok=names_ok,
        order_ok=order_ok,
        dtype_ok=dtype_ok,
        date_ok=date_ok,
        null_ok=null_ok,
        colcount_ok=colcount_ok,
        warnings=warnings,
        dtype_mismatches=dtype_mismatches,
        null_detail=", ".join(null_notes),
        detail="" if order_ok else f"csv={csv_cols} oracle={oracle_cols}",
    )


def _join_report(frames: dict[str, pd.DataFrame]) -> list[dict]:
    rows = []
    for left, lkey, right, rkey in JOINS:
        left_df, right_df = frames.get(left), frames.get(right)
        label = f"{left}.{lkey} -> {right}.{rkey}"
        if (
            left_df is None
            or right_df is None
            or lkey not in left_df.columns
            or rkey not in right_df.columns
        ):
            rows.append({"label": label, "keys": False, "rate": None, "orphans": None})
            continue
        left_keys = left_df[lkey].astype(str)
        right_keys = set(right_df[rkey].astype(str))
        total = len(left_keys)
        matched = int(left_keys.isin(right_keys).sum())
        orphans = total - matched
        rate = (matched / total * 100.0) if total else 100.0
        rows.append({"label": label, "keys": True, "rate": rate, "orphans": orphans})
    return rows


def _flag(value: bool) -> str:
    return "PASS" if value else "FAIL"


def run(allow_offline: bool, require_oracle: bool) -> int:
    csv_frames = _load_all(CsvBackend())

    oracle_frames: dict[str, pd.DataFrame] | None = None
    oracle_error = ""
    try:
        oracle_frames = _load_all(OracleBackend())
    except Exception as error:
        oracle_error = f"{type(error).__name__}: {error}"

    results = [
        _compare(
            name,
            csv_frames[name],
            None if oracle_frames is None else oracle_frames[name],
        )
        for name in DATASETS
    ]

    print("=" * 52)
    print("ORACLE PARITY VALIDATION REPORT")
    print("=" * 52)
    if oracle_frames is None:
        print(f"Oracle backend: UNAVAILABLE ({oracle_error})")
        print("Mode: OFFLINE - static column-contract checks only.")
    else:
        print("Oracle backend: CONNECTED - full live parity comparison.")

    # ---- Per-dataset blocks -------------------------------------------------
    for r in results:
        print()
        print(f"DATASET: {r.name.upper()}")
        print()
        print(f"  CSV Shape:       {r.csv_shape}")
        print(
            f"  Oracle Shape:    "
            f"{'unavailable' if r.oracle_shape is None else r.oracle_shape}"
        )
        print()
        if r.oracle_shape is None:
            print(f"  Required Cols:   {_flag(r.required_ok)} (contract vs CSV)")
            print(f"  Column Match:    {_flag(r.names_ok)} (contract vs CSV)")
            print(f"  Order Match:     {_flag(r.order_ok)} (contract vs CSV)")
            print(f"  Dtype Match:     SKIP (needs live Oracle)")
            print(f"  Null Check:      SKIP (needs live Oracle)")
        else:
            print(f"  Required Cols:   {_flag(r.required_ok)}")
            print(f"  Column Match:    {_flag(r.names_ok)}")
            print(f"  Order Match:     {_flag(r.order_ok)}")
            print(f"  Dtype Match:     {_flag(r.dtype_ok)}")
            print(f"  Date Format:     {_flag(r.date_ok)}")
            print(f"  Null Check:      {_flag(r.null_ok)}")
            print(f"  Column Count:    {_flag(r.colcount_ok)}")
            row_n = "n/a" if r.oracle_shape is None else r.oracle_shape[0]
            print(f"  Row Count:       CSV={r.csv_shape[0]} Oracle={row_n} (info)")
        if r.dtype_mismatches:
            print(f"  ! dtype issues:  {r.dtype_mismatches}")
        if not r.null_ok and r.null_detail:
            print(f"  ! null keys:     {r.null_detail}")
        if not r.order_ok and r.detail:
            print(f"  ! columns:       {r.detail}")
        for warning in r.warnings:
            print(f"  ~ warning:       {warning}")
        print()
        print(f"  Status: {r.status}")
        print("-" * 52)

    # ---- Sample join validation --------------------------------------------
    print()
    print("SAMPLE JOIN VALIDATION (within each backend)")
    print()
    join_regression = False
    csv_joins = _join_report(csv_frames)
    oracle_joins = _join_report(oracle_frames) if oracle_frames is not None else None
    for i, cj in enumerate(csv_joins):
        csv_rate = "n/a" if cj["rate"] is None else f"{cj['rate']:.1f}%"
        print(f"  {cj['label']}")
        print(f"    CSV    : key={'YES' if cj['keys'] else 'NO'} success={csv_rate} orphans={cj['orphans']}")
        if oracle_joins is None:
            print(f"    Oracle : unavailable")
        else:
            oj = oracle_joins[i]
            ora_rate = "n/a" if oj["rate"] is None else f"{oj['rate']:.1f}%"
            print(f"    Oracle : key={'YES' if oj['keys'] else 'NO'} success={ora_rate} orphans={oj['orphans']}")
            if (
                oj["rate"] is not None
                and cj["rate"] is not None
                and oj["rate"] + 1e-9 < cj["rate"]
            ):
                join_regression = True
                print(f"    -> JOIN REGRESSION (Oracle below CSV integrity)")
    print("-" * 52)

    # ---- Final summary ------------------------------------------------------
    total = len(results)
    passed = sum(1 for r in results if r.status == "PASS")
    failed = sum(1 for r in results if r.status == "FAIL")
    skipped = sum(1 for r in results if r.status == "SKIPPED")
    warnings = sum(len(r.warnings) for r in results)

    print()
    print("=" * 52)
    print("SUMMARY")
    print("=" * 52)
    print(f"  Total Datasets Checked: {total}")
    print(f"  Passed:                 {passed}")
    print(f"  Failed:                 {failed}")
    print(f"  Skipped:                {skipped}")
    print(f"  Warnings:               {warnings}")
    if join_regression:
        print(f"  Join Regression:        YES")

    # ---- Verdict + exit code ------------------------------------------------
    if oracle_frames is None:
        contract_ok = all(r.names_ok and r.order_ok for r in results)
        readiness = round(
            sum(1 for r in results if r.names_ok and r.order_ok) / total * 100
        )
        print(f"  Oracle Cutover Readiness: {readiness}% (static contract only)")
        print("=" * 52)
        if require_oracle:
            print("VERDICT: FAIL - Oracle required but unavailable.")
            return 2
        if allow_offline:
            print(
                f"VERDICT: {'PASS (offline)' if contract_ok else 'FAIL (contract)'} "
                "- live dtype/null/join checks not run."
            )
            return 0 if contract_ok else 1
        print("VERDICT: UNVALIDATED - re-run against a live Oracle, or pass --allow-offline.")
        return 2

    all_pass = failed == 0 and not join_regression
    base = round(passed / total * 100)
    readiness = min(base, 80) if join_regression else base
    print(f"  Oracle Cutover Readiness: {readiness}%")
    print("=" * 52)
    print(
        "VERDICT: "
        + (
            "PASS - OracleBackend is a drop-in replacement for CsvBackend."
            if all_pass
            else "FAIL - resolve the parity differences above before cutover."
        )
    )
    return 0 if all_pass else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Oracle<->CSV backend parity check.")
    parser.add_argument("--allow-offline", action="store_true",
                        help="Pass on static column-contract parity when Oracle is down.")
    parser.add_argument("--require-oracle", action="store_true",
                        help="Fail (exit 2) if a live Oracle backend is not reachable.")
    args = parser.parse_args()
    return run(allow_offline=args.allow_offline, require_oracle=args.require_oracle)


if __name__ == "__main__":
    raise SystemExit(main())
