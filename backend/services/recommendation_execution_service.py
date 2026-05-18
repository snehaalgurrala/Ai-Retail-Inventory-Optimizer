import json
import re
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
from uuid import uuid4

import pandas as pd

from backend.memory.memory_store import save_decision_record, save_outcome_record
from backend.services.data_processor import build_processed_datasets
from backend.services.inventory_analyzer import build_inventory_analysis
from backend.services.llm_reasoner import _chat_json, _extract_json_text, llm_is_configured


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

RECOMMENDATIONS_FILE = PROCESSED_DATA_DIR / "recommendations.csv"
DECISIONS_FILE = PROCESSED_DATA_DIR / "recommendation_decisions.csv"
PRICE_UPDATES_FILE = PROCESSED_DATA_DIR / "price_updates.csv"
TRANSFER_ACTIONS_FILE = PROCESSED_DATA_DIR / "transfer_actions.csv"
PROCUREMENT_ORDERS_FILE = PROCESSED_DATA_DIR / "procurement_orders.csv"
CLEARANCE_ACTIONS_FILE = PROCESSED_DATA_DIR / "clearance_actions.csv"
RISK_ACTIONS_FILE = PROCESSED_DATA_DIR / "risk_actions.csv"
PRODUCTS_FILE = RAW_DATA_DIR / "products.csv"
INVENTORY_FILE = RAW_DATA_DIR / "inventory.csv"
TRANSACTIONS_FILE = RAW_DATA_DIR / "transactions.csv"
STORES_FILE = RAW_DATA_DIR / "stores.csv"
SUPPLIERS_FILE = RAW_DATA_DIR / "suppliers.csv"
CURRENT_INVENTORY_FILE = PROCESSED_DATA_DIR / "current_inventory.csv"
LOW_STOCK_FILE = PROCESSED_DATA_DIR / "low_stock_items.csv"
OVERSTOCK_FILE = PROCESSED_DATA_DIR / "overstock_items.csv"
DEAD_STOCK_FILE = PROCESSED_DATA_DIR / "dead_stock_candidates.csv"
STOCKOUT_FILE = PROCESSED_DATA_DIR / "stockout_risk_items.csv"
PRODUCT_PERFORMANCE_FILE = PROCESSED_DATA_DIR / "product_performance.csv"

DECISION_COLUMNS = [
    "decision_id",
    "recommendation_id",
    "decision",
    "decided_at",
    "recommendation_type",
    "product_id",
    "product_name",
    "store_id",
    "source_agent",
    "manager_note",
    "rejection_reason",
    "edited_values_json",
    "execution_status",
    "action_summary",
]

PRICE_UPDATE_COLUMNS = [
    "update_id",
    "recommendation_id",
    "action_type",
    "product_id",
    "product_name",
    "store_id",
    "old_price",
    "discount_percent",
    "new_price",
    "reason",
    "manager_note",
    "executed_at",
]

TRANSFER_ACTION_COLUMNS = [
    "transfer_id",
    "recommendation_id",
    "product_id",
    "product_name",
    "source_store_id",
    "target_store_id",
    "quantity",
    "reason",
    "manager_note",
    "executed_at",
]

PROCUREMENT_COLUMNS = [
    "procurement_id",
    "recommendation_id",
    "product_id",
    "product_name",
    "store_id",
    "supplier_id",
    "supplier_name",
    "quantity",
    "priority",
    "reason",
    "manager_note",
    "executed_at",
]

CLEARANCE_COLUMNS = [
    "clearance_id",
    "recommendation_id",
    "product_id",
    "product_name",
    "store_id",
    "old_price",
    "clearance_discount_percent",
    "new_price",
    "reason",
    "manager_note",
    "executed_at",
]

RISK_ACTION_COLUMNS = [
    "risk_action_id",
    "recommendation_id",
    "recommendation_type",
    "product_id",
    "product_name",
    "store_id",
    "mitigation_action",
    "priority",
    "manager_note",
    "status",
    "executed_at",
]


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _new_id(prefix: str) -> str:
    return f"{prefix}{datetime.now().strftime('%Y%m%d%H%M%S')}{uuid4().hex[:6].upper()}"


def _read_csv(file_path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if not file_path.exists():
        return pd.DataFrame(columns=columns or [])

    df = pd.read_csv(file_path)
    if columns:
        for column in columns:
            if column not in df.columns:
                df[column] = ""
        return df
    return df


def _atomic_write_csv(df: pd.DataFrame, file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w",
        newline="",
        delete=False,
        dir=str(file_path.parent),
        suffix=".csv",
    ) as temp_file:
        df.to_csv(temp_file.name, index=False)
        temp_path = Path(temp_file.name)
    temp_path.replace(file_path)


def _append_row(file_path: Path, row: dict[str, Any], columns: list[str]) -> None:
    df = _read_csv(file_path, columns)
    for column in columns:
        if column not in row:
            row[column] = ""
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    _atomic_write_csv(df[columns], file_path)


def _to_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _to_float(value: Any, default: float = 0.0) -> float:
    parsed = pd.to_numeric(value, errors="coerce")
    if pd.isna(parsed):
        return default
    return float(parsed)


def _to_int(value: Any, default: int = 0) -> int:
    return int(round(_to_float(value, default)))


def _parse_evidence(evidence: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for part in str(evidence or "").split(","):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def _extract_first_number(text: str, pattern: str) -> float | None:
    match = re.search(pattern, str(text or ""), flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return float(match.group(1))
    except (TypeError, ValueError):
        return None


def _read_lookup(df: pd.DataFrame, keys: list[str]) -> dict[tuple[str, ...], dict[str, Any]]:
    lookup: dict[tuple[str, ...], dict[str, Any]] = {}
    if df.empty:
        return lookup
    for _, row in df.iterrows():
        lookup[tuple(_to_text(row.get(key, "")) for key in keys)] = row.to_dict()
    return lookup


def _resolve_store_identifier(store_value: str, stores_df: pd.DataFrame) -> tuple[str, str]:
    normalized = _to_text(store_value)
    if not normalized or stores_df.empty:
        return normalized, normalized
    if "store_id" not in stores_df.columns or "store_name" not in stores_df.columns:
        return normalized, normalized

    store_id_mask = stores_df["store_id"].astype(str) == normalized
    if store_id_mask.any():
        row = stores_df.loc[store_id_mask].iloc[0]
        return _to_text(row.get("store_id", normalized)), _to_text(row.get("store_name", normalized))

    store_name_mask = stores_df["store_name"].astype(str) == normalized
    if store_name_mask.any():
        row = stores_df.loc[store_name_mask].iloc[0]
        return _to_text(row.get("store_id", normalized)), _to_text(row.get("store_name", normalized))

    return normalized, normalized


def _load_context_data() -> dict[str, pd.DataFrame]:
    return {
        "recommendations": _read_csv(RECOMMENDATIONS_FILE),
        "products": _read_csv(PRODUCTS_FILE),
        "inventory": _read_csv(INVENTORY_FILE),
        "stores": _read_csv(STORES_FILE),
        "suppliers": _read_csv(SUPPLIERS_FILE),
        "current_inventory": _read_csv(CURRENT_INVENTORY_FILE),
        "low_stock": _read_csv(LOW_STOCK_FILE),
        "overstock": _read_csv(OVERSTOCK_FILE),
        "dead_stock": _read_csv(DEAD_STOCK_FILE),
        "stockout": _read_csv(STOCKOUT_FILE),
        "product_performance": _read_csv(PRODUCT_PERFORMANCE_FILE),
    }


def _find_recommendation_row(recommendation_id: str) -> pd.Series:
    recommendations = _read_csv(RECOMMENDATIONS_FILE)
    matches = recommendations[
        recommendations["recommendation_id"].astype(str) == str(recommendation_id)
    ]
    if matches.empty:
        raise ValueError(f"Recommendation '{recommendation_id}' was not found.")
    return matches.iloc[0]


def _recommendation_analysis_row(
    recommendation: pd.Series,
    context: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    recommendation_type = _to_text(recommendation.get("recommendation_type", ""))
    product_id = _to_text(recommendation.get("product_id", ""))
    store_id = _to_text(recommendation.get("store_id", ""))
    key = (product_id, store_id)

    lookup_map = {
        "discount": _read_lookup(context["overstock"], ["product_id", "store_id"]),
        "clearance": _read_lookup(context["dead_stock"], ["product_id", "store_id"]),
        "reorder": _read_lookup(context["low_stock"], ["product_id", "store_id"]),
        "stock_transfer": _read_lookup(context["low_stock"], ["product_id", "store_id"]),
        "transfer": _read_lookup(context["low_stock"], ["product_id", "store_id"]),
        "stockout_prevention_alert": _read_lookup(context["stockout"], ["product_id", "store_id"]),
        "overstock_alert": _read_lookup(context["overstock"], ["product_id", "store_id"]),
    }

    if recommendation_type == "supplier_risk_alert":
        product_lookup = _read_lookup(context["product_performance"], ["product_id"])
        return product_lookup.get((product_id,), {})

    return lookup_map.get(recommendation_type, {}).get(key, {})


def build_recommendation_context(recommendation: pd.Series | dict[str, Any]) -> dict[str, Any]:
    recommendation_row = recommendation if isinstance(recommendation, pd.Series) else pd.Series(recommendation)
    context = _load_context_data()
    product_id = _to_text(recommendation_row.get("product_id", ""))
    store_id = _to_text(recommendation_row.get("store_id", ""))
    recommendation_type = _to_text(recommendation_row.get("recommendation_type", ""))
    evidence_map = _parse_evidence(_to_text(recommendation_row.get("evidence", "")))

    product_lookup = _read_lookup(context["products"], ["product_id"])
    store_lookup = _read_lookup(context["stores"], ["store_id"])
    inventory_lookup = _read_lookup(context["inventory"], ["product_id", "store_id"])
    current_inventory_lookup = _read_lookup(context["current_inventory"], ["product_id", "store_id"])
    supplier_lookup = _read_lookup(context["suppliers"], ["supplier_id"])

    product_row = product_lookup.get((product_id,), {})
    store_row = store_lookup.get((store_id,), {})
    inventory_row = inventory_lookup.get((product_id, store_id), {})
    current_inventory_row = current_inventory_lookup.get((product_id, store_id), {})
    analysis_row = _recommendation_analysis_row(recommendation_row, context)

    supplier_id = _to_text(
        product_row.get("supplier_id")
        or analysis_row.get("supplier_id")
        or evidence_map.get("supplier_id", "")
    )
    supplier_row = supplier_lookup.get((supplier_id,), {})

    suggested_quantity = _to_int(
        recommendation_row.get("suggested_quantity")
        or evidence_map.get("suggested_quantity")
        or evidence_map.get("quantity")
        or 0
    )

    source_store_id, source_store_name = _resolve_store_identifier(
        evidence_map.get("source_store_id", "") or evidence_map.get("source_store", ""),
        context["stores"],
    )
    destination_store_id, destination_store_name = _resolve_store_identifier(
        evidence_map.get("destination_store_id", "") or evidence_map.get("destination_store", "") or store_id,
        context["stores"],
    )
    target_store_name = _to_text(
        destination_store_name
        or store_row.get("store_name")
        or analysis_row.get("store_name")
        or destination_store_id
    )

    current_price = _to_float(product_row.get("selling_price", 0))
    recommended_discount = _extract_first_number(
        _to_text(recommendation_row.get("action", "")),
        r"(\d+(?:\.\d+)?)\s*%",
    )
    if recommended_discount is None:
        recommended_discount = 10.0 if recommendation_type == "discount" else 20.0
    discounted_price = round(current_price * (1 - recommended_discount / 100), 2)

    return {
        "recommendation": recommendation_row.to_dict(),
        "evidence_map": evidence_map,
        "analysis_row": analysis_row,
        "product_row": product_row,
        "inventory_row": inventory_row,
        "current_inventory_row": current_inventory_row,
        "store_row": store_row,
        "supplier_row": supplier_row,
        "store_name": _to_text(store_row.get("store_name", "")),
        "supplier_name": _to_text(supplier_row.get("supplier_name", "")),
        "supplier_id": supplier_id,
        "current_stock": _to_int(
            inventory_row.get("stock_level")
            or current_inventory_row.get("stock_level")
            or analysis_row.get("stock_level")
            or 0
        ),
        "threshold": _to_int(
            inventory_row.get("reorder_threshold")
            or analysis_row.get("effective_reorder_threshold")
            or product_row.get("reorder_threshold")
            or 0
        ),
        "current_price": current_price,
        "recommended_discount_percent": float(recommended_discount),
        "discounted_price": discounted_price,
        "suggested_quantity": suggested_quantity,
        "source_store_id": source_store_id,
        "source_store_name": source_store_name,
        "target_store_id": destination_store_id,
        "target_store_name": target_store_name,
        "recent_daily_velocity": _to_float(
            analysis_row.get("recent_daily_sales_velocity")
            or evidence_map.get("recent_daily_velocity")
            or 0
        ),
        "recent_30_day_sales": _to_int(analysis_row.get("recent_30_day_quantity_sold") or 0),
        "days_of_stock": round(
            _to_float(
                analysis_row.get("days_of_stock_remaining")
                or evidence_map.get("days_of_stock")
                or 0
            ),
            2,
        ),
        "shelf_life_days": _to_int(
            product_row.get("shelf_life_days") or analysis_row.get("shelf_life_days") or 0
        ),
        "reliability_score": _to_float(
            supplier_row.get("reliability_score") or evidence_map.get("reliability_score") or 0
        ),
        "avg_delivery_days": _to_float(
            supplier_row.get("avg_delivery_days") or evidence_map.get("avg_delivery_days") or 0
        ),
        "alternative_product_id": _to_text(recommendation_row.get("alternative_product_id", "")),
        "alternative_product_name": _to_text(recommendation_row.get("alternative_product_name", "")),
        "alternative_store_id": _to_text(recommendation_row.get("alternative_store_id", "")),
        "alternative_store_name": _to_text(recommendation_row.get("alternative_store_name", "")),
        "available_quantity": _to_int(
            recommendation_row.get("available_quantity")
            or evidence_map.get("available_quantity")
            or 0
        ),
        "category": _to_text(
            product_row.get("category")
            or analysis_row.get("category")
            or evidence_map.get("category", "")
        ),
        "is_exclusive": _to_text(evidence_map.get("is_exclusive", "")).lower() == "true",
    }


def _fallback_explanation(recommendation: dict[str, Any], context: dict[str, Any]) -> dict[str, str]:
    recommendation_type = _to_text(recommendation.get("recommendation_type", ""))
    product_name = _to_text(recommendation.get("product_name", "product"))
    store_name = context.get("store_name") or _to_text(recommendation.get("store_id", ""))
    evidence = _to_text(recommendation.get("evidence", ""))
    source_agent = _to_text(recommendation.get("source_agent", "agent"))
    current_stock = context.get("current_stock", 0)
    threshold = context.get("threshold", 0)
    velocity = context.get("recent_daily_velocity", 0)
    impact = ""
    factors = ""
    improvement = ""
    risk_if_ignored = ""

    if recommendation_type == "discount":
        factors = (
            f"Slow movement, overstock pressure, recent sales velocity of {velocity:.2f} units/day, "
            f"stock level of {current_stock}, and shelf life of {context.get('shelf_life_days', 0)} days were considered."
        )
        impact = (
            f"Approving will update [data/raw/products.csv] with a lower selling price and create an audit entry in "
            f"[data/processed/price_updates.csv]."
        )
        improvement = "Lower pricing should improve sell-through and reduce excess holding time."
        risk_if_ignored = "Excess inventory may continue tying up working capital and shelf space."
    elif recommendation_type == "clearance":
        factors = (
            f"Current stock is {current_stock}, recent 30-day movement is {context.get('recent_30_day_sales', 0)} units, "
            f"and shelf life pressure is {context.get('shelf_life_days', 0)} days."
        )
        impact = (
            f"Approving will reduce the selling price in [data/raw/products.csv] and log the clearance action in "
            f"[data/processed/clearance_actions.csv]."
        )
        improvement = "Clearance should accelerate sell-through and reduce expiry or dead-stock exposure."
        risk_if_ignored = "The item may remain stuck in inventory or age out before it sells."
    elif recommendation_type in {"stock_transfer", "transfer"}:
        factors = (
            f"Target stock is {current_stock} against threshold {threshold}. The transfer suggestion is {context.get('suggested_quantity', 0)} units "
            f"from {context.get('source_store_name', context.get('source_store_id', 'source store'))}."
        )
        impact = (
            f"Approving will move inventory between stores in [data/raw/inventory.csv], create transfer transaction rows in "
            f"[data/raw/transactions.csv], and append a transfer log in [data/processed/transfer_actions.csv]."
        )
        improvement = "Balancing stock should reduce lost sales at the target store without waiting for fresh procurement."
        risk_if_ignored = "The destination store may stock out while surplus stock sits elsewhere."
    elif recommendation_type == "exclusive_availability":
        factors = (
            f"{product_name} has {context.get('available_quantity', 0)} units available at "
            f"{store_name or 'the exclusive store'} and no positive stock in other stores."
        )
        impact = "Approving logs this as a manager-reviewed promotional availability option without changing inventory."
        improvement = "The item can be promoted as a store-specific option when nearby alternatives are limited."
        risk_if_ignored = "The business may miss a chance to redirect demand toward available inventory."
    elif recommendation_type == "alternative_option":
        factors = (
            f"{product_name} is low at {store_name or 'the target store'}, while "
            f"{context.get('alternative_product_name', 'an alternative product')} has "
            f"{context.get('available_quantity', 0)} units at "
            f"{context.get('alternative_store_name', context.get('alternative_store_id', 'another store'))}."
        )
        impact = "Approving logs this alternative offer as reviewed without moving stock automatically."
        improvement = "The alternative can protect sales when the original low-stock item cannot be fulfilled quickly."
        risk_if_ignored = "Customers may see an unavailable item without a substitute offer."
    elif recommendation_type == "reorder":
        factors = (
            f"Current stock is {current_stock}, reorder threshold is {threshold}, and the suggested replenishment is "
            f"{context.get('suggested_quantity', 0)} units."
        )
        impact = (
            f"Approving will increase stock in [data/raw/inventory.csv], create procurement logs in "
            f"[data/processed/procurement_orders.csv], and write a restock transaction to [data/raw/transactions.csv]."
        )
        improvement = "Replenishment should restore coverage and lower the chance of missed demand."
        risk_if_ignored = "The store could continue running below safe coverage and move closer to stockout."
    else:
        risk_label = recommendation_type.replace("_", " ")
        factors = f"Priority, supporting evidence, and related product or supplier signals were reviewed. Evidence: {evidence or 'not provided'}"
        impact = (
            f"Approving will mark the alert as mitigated in [data/processed/recommendations.csv] and log the mitigation in "
            f"[data/processed/risk_actions.csv]."
        )
        improvement = "Operational follow-up should reduce escalation risk and improve traceability."
        risk_if_ignored = f"The {risk_label} may worsen without a documented mitigation step."

    return {
        "why_generated": (
            f"{product_name} at {store_name or 'the relevant scope'} was flagged because { _to_text(recommendation.get('reason', '')) or 'the underlying metrics crossed the configured threshold' }."
        ),
        "factors_analyzed": factors,
        "expected_business_impact": impact,
        "sales_inventory_improvement": improvement,
        "risk_if_ignored": risk_if_ignored,
        "confidence_level": _to_text(recommendation.get("priority", "medium")) or "medium",
        "source_agent": source_agent,
        "supporting_evidence": evidence or "No evidence string is available for this recommendation.",
    }


def get_recommendation_explanation(
    recommendation: dict[str, Any] | pd.Series,
    context: dict[str, Any] | None = None,
) -> dict[str, str]:
    recommendation_data = recommendation.to_dict() if isinstance(recommendation, pd.Series) else dict(recommendation)
    context = context or build_recommendation_context(recommendation_data)
    fallback = _fallback_explanation(recommendation_data, context)

    if not llm_is_configured():
        fallback["explanation_source"] = "fallback"
        return fallback

    prompt_payload = {
        "recommendation": recommendation_data,
        "context": {
            "store_name": context.get("store_name", ""),
            "supplier_name": context.get("supplier_name", ""),
            "current_stock": context.get("current_stock", 0),
            "threshold": context.get("threshold", 0),
            "current_price": context.get("current_price", 0),
            "recommended_discount_percent": context.get("recommended_discount_percent", 0),
            "discounted_price": context.get("discounted_price", 0),
            "suggested_quantity": context.get("suggested_quantity", 0),
            "source_store_name": context.get("source_store_name", ""),
            "target_store_name": context.get("target_store_name", ""),
            "recent_daily_velocity": context.get("recent_daily_velocity", 0),
            "recent_30_day_sales": context.get("recent_30_day_sales", 0),
            "days_of_stock": context.get("days_of_stock", 0),
            "shelf_life_days": context.get("shelf_life_days", 0),
            "reliability_score": context.get("reliability_score", 0),
            "avg_delivery_days": context.get("avg_delivery_days", 0),
        },
        "expected_output_schema": {
            "why_generated": "short explanation",
            "factors_analyzed": "short explanation",
            "expected_business_impact": "short explanation",
            "sales_inventory_improvement": "short explanation",
            "risk_if_ignored": "short explanation",
            "confidence_level": "low|medium|high",
            "source_agent": "agent name",
            "supporting_evidence": "grounded evidence summary",
        },
    }
    try:
        raw_response = _chat_json(
            "You explain retail recommendations for a human approver. Use only the given data. Return valid JSON only.",
            json.dumps(prompt_payload, ensure_ascii=True),
        )
        explanation = json.loads(_extract_json_text(raw_response))
        for key in fallback:
            explanation.setdefault(key, fallback[key])
        explanation["explanation_source"] = "llm"
        return explanation
    except Exception:
        fallback["explanation_source"] = "fallback"
        return fallback


def _validate_required_text(value: Any, label: str) -> str:
    text_value = _to_text(value)
    if not text_value:
        raise ValueError(f"{label} is required before approval.")
    return text_value


def _validate_positive_int(value: Any, label: str) -> int:
    parsed = _to_int(value, -1)
    if parsed <= 0:
        raise ValueError(f"{label} must be greater than zero.")
    return parsed


def _validate_discount_percent(value: Any) -> float:
    discount = _to_float(value, -1)
    if discount <= 0 or discount >= 100:
        raise ValueError("Discount percentage must be between 0 and 100.")
    return round(discount, 2)


def _update_recommendation_status(
    recommendation_id: str,
    status: str,
    manager_note: str = "",
    decision_reason: str = "",
) -> dict[str, Any]:
    recommendations = _read_csv(RECOMMENDATIONS_FILE)
    if recommendations.empty:
        raise ValueError("Recommendations file is empty.")
    mask = recommendations["recommendation_id"].astype(str) == str(recommendation_id)
    if not mask.any():
        raise ValueError(f"Recommendation '{recommendation_id}' was not found.")

    for column in ["manager_note", "decision_reason", "status_updated_at"]:
        if column not in recommendations.columns:
            recommendations[column] = ""

    recommendations.loc[mask, "status"] = status
    recommendations.loc[mask, "manager_note"] = manager_note
    recommendations.loc[mask, "decision_reason"] = decision_reason
    recommendations.loc[mask, "status_updated_at"] = _now_iso()
    _atomic_write_csv(recommendations, RECOMMENDATIONS_FILE)
    return recommendations.loc[mask].iloc[0].to_dict()


def _log_decision(
    recommendation: dict[str, Any],
    decision: str,
    execution_status: str,
    edited_values: dict[str, Any] | None = None,
    manager_note: str = "",
    rejection_reason: str = "",
    action_summary: str = "",
) -> None:
    decided_at = _now_iso()
    row = {
        "decision_id": _new_id("DEC"),
        "recommendation_id": _to_text(recommendation.get("recommendation_id", "")),
        "decision": decision,
        "decided_at": decided_at,
        "recommendation_type": _to_text(recommendation.get("recommendation_type", "")),
        "product_id": _to_text(recommendation.get("product_id", "")),
        "product_name": _to_text(recommendation.get("product_name", "")),
        "store_id": _to_text(recommendation.get("store_id", "")),
        "source_agent": _to_text(recommendation.get("source_agent", "")),
        "manager_note": manager_note,
        "rejection_reason": rejection_reason,
        "edited_values_json": json.dumps(edited_values or {}, ensure_ascii=True),
        "execution_status": execution_status,
        "action_summary": action_summary,
    }
    _append_row(DECISIONS_FILE, row, DECISION_COLUMNS)
    save_decision_record(recommendation, decision, decided_at)


def _next_transaction_id(transactions_df: pd.DataFrame) -> str:
    if transactions_df.empty or "transaction_id" not in transactions_df.columns:
        return "TXN0000001"
    numeric_part = (
        transactions_df["transaction_id"]
        .astype(str)
        .str.extract(r"(\d+)", expand=False)
        .fillna("0")
        .astype(int)
        .max()
    )
    return f"TXN{str(int(numeric_part) + 1).zfill(7)}"


def _append_transactions(rows: list[dict[str, Any]]) -> None:
    transactions = _read_csv(TRANSACTIONS_FILE)
    next_id = _next_transaction_id(transactions)
    next_number = int(re.search(r"(\d+)", next_id).group(1))
    timestamp = datetime.now().strftime("%Y-%m-%d")
    normalized_rows = []
    for index, row in enumerate(rows):
        normalized_rows.append(
            {
                "transaction_id": f"TXN{str(next_number + index).zfill(7)}",
                "date": timestamp,
                "transaction_type": row.get("transaction_type", ""),
                "product_id": row.get("product_id", ""),
                "store_id": row.get("store_id", ""),
                "quantity": row.get("quantity", 0),
                "source": row.get("source", "recommendation_execution"),
                "remarks": row.get("remarks", ""),
            }
        )
    transactions = pd.concat([transactions, pd.DataFrame(normalized_rows)], ignore_index=True)
    _atomic_write_csv(transactions, TRANSACTIONS_FILE)


def execute_discount_action(
    recommendation: dict[str, Any],
    edited_values: dict[str, Any],
) -> dict[str, Any]:
    context = build_recommendation_context(recommendation)
    discount_percent = _validate_discount_percent(
        edited_values.get("discount_percent", context["recommended_discount_percent"])
    )
    manager_note = _to_text(edited_values.get("manager_note", ""))
    reason = _validate_required_text(
        edited_values.get("action_text", recommendation.get("action", "")),
        "Action text",
    )

    products = _read_csv(PRODUCTS_FILE)
    product_id = _to_text(recommendation.get("product_id", ""))
    mask = products["product_id"].astype(str) == product_id
    if not mask.any():
        raise ValueError(f"Product '{product_id}' was not found in products.csv.")

    products["selling_price"] = pd.to_numeric(
        products["selling_price"],
        errors="coerce",
    ).fillna(0.0).astype(float)
    old_price = _to_float(products.loc[mask, "selling_price"].iloc[0], 0)
    new_price = round(old_price * (1 - discount_percent / 100), 2)
    if new_price <= 0:
        raise ValueError("Discounted price must remain above zero.")

    products.loc[mask, "selling_price"] = new_price
    _atomic_write_csv(products, PRODUCTS_FILE)

    _append_row(
        PRICE_UPDATES_FILE,
        {
            "update_id": _new_id("PRC"),
            "recommendation_id": recommendation.get("recommendation_id", ""),
            "action_type": "discount",
            "product_id": product_id,
            "product_name": recommendation.get("product_name", ""),
            "store_id": recommendation.get("store_id", ""),
            "old_price": round(old_price, 2),
            "discount_percent": discount_percent,
            "new_price": new_price,
            "reason": reason,
            "manager_note": manager_note,
            "executed_at": _now_iso(),
        },
        PRICE_UPDATE_COLUMNS,
    )
    return {
        "action_summary": f"Updated selling price from {old_price:.2f} to {new_price:.2f} ({discount_percent:.2f}% discount).",
        "old_price": old_price,
        "new_price": new_price,
        "discount_percent": discount_percent,
    }


def execute_clearance_action(
    recommendation: dict[str, Any],
    edited_values: dict[str, Any],
) -> dict[str, Any]:
    result = execute_discount_action(recommendation, edited_values)
    _append_row(
        CLEARANCE_ACTIONS_FILE,
        {
            "clearance_id": _new_id("CLR"),
            "recommendation_id": recommendation.get("recommendation_id", ""),
            "product_id": recommendation.get("product_id", ""),
            "product_name": recommendation.get("product_name", ""),
            "store_id": recommendation.get("store_id", ""),
            "old_price": result["old_price"],
            "clearance_discount_percent": result["discount_percent"],
            "new_price": result["new_price"],
            "reason": _to_text(edited_values.get("action_text", recommendation.get("action", ""))),
            "manager_note": _to_text(edited_values.get("manager_note", "")),
            "executed_at": _now_iso(),
        },
        CLEARANCE_COLUMNS,
    )
    result["action_summary"] = f"Applied clearance pricing to {result['new_price']:.2f} after {result['discount_percent']:.2f}% markdown."
    return result


def execute_transfer_action(
    recommendation: dict[str, Any],
    edited_values: dict[str, Any],
) -> dict[str, Any]:
    context = build_recommendation_context(recommendation)
    quantity = _validate_positive_int(
        edited_values.get("suggested_quantity", context["suggested_quantity"]),
        "Transfer quantity",
    )
    source_store_id = _validate_required_text(
        edited_values.get("source_store", context["source_store_id"]),
        "Source store",
    )
    target_store_id = _validate_required_text(
        edited_values.get("target_store", context["target_store_id"]),
        "Target store",
    )
    if source_store_id == target_store_id:
        raise ValueError("Source and target stores must be different.")

    inventory = _read_csv(INVENTORY_FILE)
    product_id = _to_text(recommendation.get("product_id", ""))
    source_mask = (
        inventory["product_id"].astype(str).eq(product_id)
        & inventory["store_id"].astype(str).eq(source_store_id)
    )
    target_mask = (
        inventory["product_id"].astype(str).eq(product_id)
        & inventory["store_id"].astype(str).eq(target_store_id)
    )
    if not source_mask.any():
        raise ValueError("Source store inventory row is missing for this product.")

    source_stock = _to_int(inventory.loc[source_mask, "stock_level"].iloc[0], 0)
    if source_stock < quantity:
        raise ValueError(
            f"Transfer would create negative stock at the source store. Available: {source_stock}, requested: {quantity}."
        )

    inventory.loc[source_mask, "stock_level"] = source_stock - quantity
    inventory.loc[source_mask, "last_updated"] = datetime.now().strftime("%Y-%m-%d")

    if target_mask.any():
        target_stock = _to_int(inventory.loc[target_mask, "stock_level"].iloc[0], 0)
        inventory.loc[target_mask, "stock_level"] = target_stock + quantity
        inventory.loc[target_mask, "last_updated"] = datetime.now().strftime("%Y-%m-%d")
    else:
        product_context = context["product_row"]
        inventory = pd.concat(
            [
                inventory,
                pd.DataFrame(
                    [
                        {
                            "product_id": product_id,
                            "store_id": target_store_id,
                            "stock_level": quantity,
                            "reorder_threshold": _to_int(product_context.get("reorder_threshold"), 0),
                            "last_updated": datetime.now().strftime("%Y-%m-%d"),
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    _atomic_write_csv(inventory, INVENTORY_FILE)
    _append_transactions(
        [
            {
                "transaction_type": "transfer_out",
                "product_id": product_id,
                "store_id": source_store_id,
                "quantity": quantity,
                "remarks": f"Approved transfer to {target_store_id} for recommendation {recommendation.get('recommendation_id', '')}",
            },
            {
                "transaction_type": "transfer_in",
                "product_id": product_id,
                "store_id": target_store_id,
                "quantity": quantity,
                "remarks": f"Approved transfer from {source_store_id} for recommendation {recommendation.get('recommendation_id', '')}",
            },
        ]
    )
    _append_row(
        TRANSFER_ACTIONS_FILE,
        {
            "transfer_id": _new_id("TRN"),
            "recommendation_id": recommendation.get("recommendation_id", ""),
            "product_id": product_id,
            "product_name": recommendation.get("product_name", ""),
            "source_store_id": source_store_id,
            "target_store_id": target_store_id,
            "quantity": quantity,
            "reason": _to_text(edited_values.get("action_text", recommendation.get("action", ""))),
            "manager_note": _to_text(edited_values.get("manager_note", "")),
            "executed_at": _now_iso(),
        },
        TRANSFER_ACTION_COLUMNS,
    )
    return {
        "action_summary": f"Transferred {quantity} units from {source_store_id} to {target_store_id}.",
        "quantity": quantity,
        "source_store_id": source_store_id,
        "target_store_id": target_store_id,
    }


def execute_reorder_action(
    recommendation: dict[str, Any],
    edited_values: dict[str, Any],
) -> dict[str, Any]:
    context = build_recommendation_context(recommendation)
    quantity = _validate_positive_int(
        edited_values.get("suggested_quantity", context["suggested_quantity"]),
        "Reorder quantity",
    )
    store_id = _validate_required_text(
        edited_values.get("target_store", recommendation.get("store_id", "")),
        "Store",
    )
    inventory = _read_csv(INVENTORY_FILE)
    product_id = _to_text(recommendation.get("product_id", ""))
    mask = (
        inventory["product_id"].astype(str).eq(product_id)
        & inventory["store_id"].astype(str).eq(store_id)
    )

    if mask.any():
        current_stock = _to_int(inventory.loc[mask, "stock_level"].iloc[0], 0)
        inventory.loc[mask, "stock_level"] = current_stock + quantity
        inventory.loc[mask, "last_updated"] = datetime.now().strftime("%Y-%m-%d")
    else:
        inventory = pd.concat(
            [
                inventory,
                pd.DataFrame(
                    [
                        {
                            "product_id": product_id,
                            "store_id": store_id,
                            "stock_level": quantity,
                            "reorder_threshold": context["threshold"],
                            "last_updated": datetime.now().strftime("%Y-%m-%d"),
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    _atomic_write_csv(inventory, INVENTORY_FILE)
    _append_transactions(
        [
            {
                "transaction_type": "procurement",
                "product_id": product_id,
                "store_id": store_id,
                "quantity": quantity,
                "remarks": f"Approved reorder for recommendation {recommendation.get('recommendation_id', '')}",
            }
        ]
    )
    _append_row(
        PROCUREMENT_ORDERS_FILE,
        {
            "procurement_id": _new_id("PO"),
            "recommendation_id": recommendation.get("recommendation_id", ""),
            "product_id": product_id,
            "product_name": recommendation.get("product_name", ""),
            "store_id": store_id,
            "supplier_id": context.get("supplier_id", ""),
            "supplier_name": context.get("supplier_name", ""),
            "quantity": quantity,
            "priority": _to_text(edited_values.get("priority", recommendation.get("priority", ""))),
            "reason": _to_text(edited_values.get("action_text", recommendation.get("action", ""))),
            "manager_note": _to_text(edited_values.get("manager_note", "")),
            "executed_at": _now_iso(),
        },
        PROCUREMENT_COLUMNS,
    )
    return {
        "action_summary": f"Added {quantity} units to store {store_id} through procurement.",
        "quantity": quantity,
        "store_id": store_id,
    }


def execute_risk_action(
    recommendation: dict[str, Any],
    edited_values: dict[str, Any],
) -> dict[str, Any]:
    mitigation_action = _validate_required_text(
        edited_values.get("action_text", recommendation.get("action", "")),
        "Mitigation action",
    )
    _append_row(
        RISK_ACTIONS_FILE,
        {
            "risk_action_id": _new_id("RSK"),
            "recommendation_id": recommendation.get("recommendation_id", ""),
            "recommendation_type": recommendation.get("recommendation_type", ""),
            "product_id": recommendation.get("product_id", ""),
            "product_name": recommendation.get("product_name", ""),
            "store_id": recommendation.get("store_id", ""),
            "mitigation_action": mitigation_action,
            "priority": _to_text(edited_values.get("priority", recommendation.get("priority", ""))),
            "manager_note": _to_text(edited_values.get("manager_note", "")),
            "status": "mitigated",
            "executed_at": _now_iso(),
        },
        RISK_ACTION_COLUMNS,
    )
    return {"action_summary": f"Logged mitigation action: {mitigation_action}"}


def log_decision(
    recommendation: dict[str, Any],
    decision: str,
    edited_values: dict[str, Any] | None = None,
    manager_note: str = "",
    rejection_reason: str = "",
    execution_status: str = "success",
    action_summary: str = "",
) -> None:
    _log_decision(
        recommendation=recommendation,
        decision=decision,
        execution_status=execution_status,
        edited_values=edited_values,
        manager_note=manager_note,
        rejection_reason=rejection_reason,
        action_summary=action_summary,
    )


def _refresh_processed_outputs() -> None:
    build_processed_datasets()
    build_inventory_analysis()


def approve_recommendation(
    recommendation_id: str,
    edited_values: dict[str, Any],
) -> dict[str, Any]:
    recommendation_row = _find_recommendation_row(recommendation_id)
    recommendation = recommendation_row.to_dict()
    recommendation_type = _to_text(recommendation.get("recommendation_type", ""))
    manager_note = _to_text(edited_values.get("manager_note", ""))
    action_map = {
        "discount": execute_discount_action,
        "clearance": execute_clearance_action,
        "stock_transfer": execute_transfer_action,
        "transfer": execute_transfer_action,
        "reorder": execute_reorder_action,
        "supplier_risk_alert": execute_risk_action,
        "overstock_alert": execute_risk_action,
        "stockout_prevention_alert": execute_risk_action,
        "exclusive_availability": execute_risk_action,
        "alternative_option": execute_risk_action,
    }

    if recommendation_type not in action_map:
        raise ValueError(f"Unsupported recommendation type '{recommendation_type}'.")

    result = action_map[recommendation_type](recommendation, edited_values)
    updated_recommendation = _update_recommendation_status(
        recommendation_id=recommendation_id,
        status="approved",
        manager_note=manager_note,
        decision_reason=_to_text(edited_values.get("action_text", recommendation.get("reason", ""))),
    )
    _refresh_processed_outputs()
    log_decision(
        recommendation=updated_recommendation,
        decision="approved",
        edited_values=edited_values,
        manager_note=manager_note,
        execution_status="executed",
        action_summary=result.get("action_summary", ""),
    )
    save_outcome_record(
        {
            "recommendation_id": recommendation_id,
            "recommendation_type": recommendation_type,
            "product_id": recommendation.get("product_id", ""),
            "product_name": recommendation.get("product_name", ""),
            "store_id": recommendation.get("store_id", ""),
            "action_taken": result.get("action_summary", ""),
            "outcome_status": "executed",
            "outcome_note": manager_note,
        }
    )
    return {
        "status": "approved",
        "recommendation_id": recommendation_id,
        "message": result.get("action_summary", "Recommendation approved."),
        "details": result,
    }


def reject_recommendation(recommendation_id: str, reason: str) -> dict[str, Any]:
    rejection_reason = _validate_required_text(reason, "Rejection reason")
    recommendation_row = _find_recommendation_row(recommendation_id)
    recommendation = recommendation_row.to_dict()
    updated_recommendation = _update_recommendation_status(
        recommendation_id=recommendation_id,
        status="rejected",
        decision_reason=rejection_reason,
    )
    log_decision(
        recommendation=updated_recommendation,
        decision="rejected",
        rejection_reason=rejection_reason,
        execution_status="not_executed",
        action_summary="Recommendation rejected without changing product or inventory data.",
    )
    save_outcome_record(
        {
            "recommendation_id": recommendation_id,
            "recommendation_type": recommendation.get("recommendation_type", ""),
            "product_id": recommendation.get("product_id", ""),
            "product_name": recommendation.get("product_name", ""),
            "store_id": recommendation.get("store_id", ""),
            "action_taken": "rejected",
            "outcome_status": "not_executed",
            "outcome_note": rejection_reason,
        }
    )
    return {
        "status": "rejected",
        "recommendation_id": recommendation_id,
        "message": "Recommendation rejected without updating inventory or product data.",
    }
