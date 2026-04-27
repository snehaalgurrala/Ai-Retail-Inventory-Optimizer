from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import get_close_matches
from pathlib import Path
from typing import Any

import pandas as pd

from backend.services.llm_reasoner import humanize_analytics_payload, llm_is_configured
from backend.utils.data_loader import load_all_data


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


@dataclass
class AnalyticsRoute:
    handled: bool
    intent: str
    payload: dict[str, Any]
    supporting_df: pd.DataFrame
    sources: list[dict[str, Any]]


def _empty_payload() -> dict[str, Any]:
    return {
        "answer": "",
        "explanation": "",
        "suggestions": [],
        "follow_up_question": "",
        "confidence": "low",
        "supporting_points": [],
        "cannot_answer": False,
    }


def _normalize(text: Any) -> str:
    cleaned = re.sub(r"[^a-z0-9\s]", " ", str(text or "").lower())
    return " ".join(cleaned.split())


def _read_processed_csv(filename: str) -> pd.DataFrame:
    file_path = PROCESSED_DATA_DIR / filename
    if not file_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(file_path)
    except Exception:
        return pd.DataFrame()


def _load_chatbot_frames() -> dict[str, pd.DataFrame]:
    raw_data = load_all_data()
    return {
        **raw_data,
        "low_stock_items": _read_processed_csv("low_stock_items.csv"),
        "recommendations": _read_processed_csv("recommendations.csv"),
        "product_performance": _read_processed_csv("product_performance.csv"),
    }


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", _normalize(text))


FOLLOW_UP_PATTERNS = (
    "why",
    "how",
    "explain",
    "what about",
    "and",
    "then",
)

ANALYTICAL_KEYWORDS = {
    "least",
    "lowest",
    "worst",
    "most",
    "highest",
    "best",
    "top",
    "low",
    "stock",
    "sales",
    "sold",
    "product",
    "products",
    "store",
    "stores",
    "branch",
    "city",
    "inventory",
    "supplier",
    "suppliers",
    "reorder",
    "performance",
}


def _looks_like_follow_up(question: str) -> bool:
    text = _normalize(question)
    if not text:
        return False
    words = text.split()
    if len(words) > 3:
        return False
    return any(text == item or text.startswith(f"{item} ") for item in FOLLOW_UP_PATTERNS)


def _extract_last_assistant_message(chat_history) -> str:
    if not chat_history:
        return ""
    for message in reversed(chat_history):
        message_type = str(getattr(message, "type", "") or "").lower()
        if "ai" in message_type or "assistant" in message_type:
            return str(getattr(message, "content", "") or "").strip()
    return ""


def _friendly_no_data(message: str, follow_up: str = "") -> dict[str, Any]:
    payload = _empty_payload()
    payload.update(
        {
            "answer": message,
            "explanation": "",
            "suggestions": [],
            "follow_up_question": follow_up,
            "confidence": "low",
            "cannot_answer": True,
        }
    )
    return payload


def _build_sources(dataset_names: list[str]) -> list[dict[str, Any]]:
    return [{"dataset": dataset_name} for dataset_name in dataset_names]


def _maybe_humanize(
    question: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    if not llm_is_configured():
        return payload
    return humanize_analytics_payload(
        question=question,
        payload=payload,
        supporting_points=payload.get("supporting_points", []),
    )


def _store_match_candidates(stores: pd.DataFrame) -> dict[str, set[str]]:
    alias_map: dict[str, set[str]] = {}
    if stores.empty:
        return alias_map
    for _, row in stores.iterrows():
        store_id = str(row.get("store_id", "")).strip()
        aliases = [
            store_id,
            row.get("store_name", ""),
            row.get("city", ""),
        ]
        for alias in aliases:
            normalized = _normalize(alias)
            if not normalized:
                continue
            alias_map.setdefault(normalized, set()).add(store_id)
    return alias_map


def _match_stores(question: str, stores: pd.DataFrame) -> pd.DataFrame:
    if stores.empty:
        return pd.DataFrame()

    normalized_question = _normalize(question)
    alias_map = _store_match_candidates(stores)
    matched_store_ids: set[str] = set()

    for alias, store_ids in alias_map.items():
        if alias and alias in normalized_question:
            matched_store_ids.update(store_ids)

    if not matched_store_ids:
        candidate_texts = [normalized_question] + _tokenize(question)
        matches = []
        for candidate in candidate_texts:
            matches.extend(get_close_matches(candidate, list(alias_map.keys()), n=5, cutoff=0.82))
        for match in matches:
            matched_store_ids.update(alias_map.get(match, set()))

    if not matched_store_ids:
        return pd.DataFrame()

    return stores[stores["store_id"].astype(str).isin(sorted(matched_store_ids))].copy()


def _product_alias_map(products: pd.DataFrame) -> dict[str, set[str]]:
    alias_map: dict[str, set[str]] = {}
    if products.empty:
        return alias_map
    for _, row in products.iterrows():
        product_id = str(row.get("product_id", "")).strip()
        aliases = [product_id, row.get("product_name", "")]
        for alias in aliases:
            normalized = _normalize(alias)
            if not normalized:
                continue
            alias_map.setdefault(normalized, set()).add(product_id)
    return alias_map


def _match_products(question: str, products: pd.DataFrame) -> pd.DataFrame:
    if products.empty:
        return pd.DataFrame()

    normalized_question = _normalize(question)
    alias_map = _product_alias_map(products)
    matched_product_ids: set[str] = set()

    for alias, product_ids in alias_map.items():
        if alias and alias in normalized_question:
            matched_product_ids.update(product_ids)

    if not matched_product_ids:
        candidate_texts = [normalized_question] + _tokenize(question)
        matches = []
        for candidate in candidate_texts:
            matches.extend(get_close_matches(candidate, list(alias_map.keys()), n=5, cutoff=0.84))
        for match in matches:
            matched_product_ids.update(alias_map.get(match, set()))

    if not matched_product_ids:
        return pd.DataFrame()

    return products[products["product_id"].astype(str).isin(sorted(matched_product_ids))].copy()


def _detect_analytical_intent(question: str, matched_stores: pd.DataFrame) -> str:
    text = _normalize(question)

    if any(keyword in text for keyword in ["recommendation", "recommend", "suggest", "agent", "why did"]):
        return "recommendation_question"
    if "supplier" in text or "suppliers" in text or ("risk" in text and "supplier" in text):
        return "supplier_risk"
    if any(keyword in text for keyword in ["least sold", "lowest sold", "worst selling", "least selling"]):
        return "least_sold_product_by_store" if not matched_stores.empty else "product_performance"
    if any(keyword in text for keyword in ["top selling", "best selling", "highest sold", "most sold"]):
        return "top_sold_product_by_store" if not matched_stores.empty else "product_performance"
    if "low stock" in text or "below reorder" in text:
        return "low_stock_by_store"
    if "compare" in text and "store" in text:
        return "sales_summary_by_store"
    if "sales" in text and not ("product" in text and ("performance" in text or "performing" in text)):
        return "sales_summary_by_store"
    if "perform" in text or "performance" in text or "sales of" in text:
        return "product_performance"
    if "inventory" in text or "stock" in text:
        return "general_inventory_question"
    return ""


def _has_analytical_signal(
    question: str,
    matched_stores: pd.DataFrame,
    matched_products: pd.DataFrame,
) -> bool:
    text = _normalize(question)
    tokens = set(_tokenize(text))
    return bool(
        _detect_analytical_intent(question, matched_stores)
        or matched_stores.empty is False
        or matched_products.empty is False
        or tokens.intersection(ANALYTICAL_KEYWORDS)
    )


def _answer_follow_up_from_memory(question: str, chat_history) -> AnalyticsRoute:
    last_assistant = _extract_last_assistant_message(chat_history)
    if not last_assistant or not _looks_like_follow_up(question):
        return AnalyticsRoute(False, "", _empty_payload(), pd.DataFrame(), [])

    explanation_match = re.search(r"explanation:\s*(.+)", last_assistant, flags=re.IGNORECASE)
    answer_match = re.search(r"answer:\s*(.+)", last_assistant, flags=re.IGNORECASE)
    top_record_match = re.search(r"top_record:\s*(.+)", last_assistant, flags=re.IGNORECASE)

    answer = (
        explanation_match.group(1).strip()
        if explanation_match
        else answer_match.group(1).strip()
        if answer_match
        else ""
    )
    if not answer:
        return AnalyticsRoute(False, "", _empty_payload(), pd.DataFrame(), [])

    if top_record_match:
        top_record_text = top_record_match.group(1).strip()
        product_match = re.search(r"product_name:\s*([^,]+)", top_record_text, flags=re.IGNORECASE)
        store_match = re.search(r"store_name:\s*([^,]+)", top_record_text, flags=re.IGNORECASE)
        quantity_match = re.search(r"(quantity_sold|total_units_sold):\s*([^,]+)", top_record_text, flags=re.IGNORECASE)
        if product_match or store_match or quantity_match:
            product_name = product_match.group(1).strip() if product_match else "that product"
            store_name = store_match.group(1).strip() if store_match else "that store"
            quantity_text = quantity_match.group(2).strip() if quantity_match else ""
            detail = f"In the previous result, {product_name}"
            if quantity_text:
                detail += f" had {quantity_text} units"
            detail += f" in {store_name}."
            answer = f"{answer} {detail}"
        else:
            answer = f"{answer} {top_record_text}."

    payload = _empty_payload()
    payload.update(
        {
            "answer": answer,
            "explanation": "That is the reasoning behind the previous answer based on the current data we already discussed.",
            "suggestions": ["If you want, I can break it down by product, store, or supplier."],
            "follow_up_question": "",
            "confidence": "medium",
        }
    )
    return AnalyticsRoute(True, "follow_up", payload, pd.DataFrame(), [])


def _store_label(store_matches: pd.DataFrame) -> str:
    if store_matches.empty:
        return "the matched store"
    if len(store_matches) == 1:
        row = store_matches.iloc[0]
        return str(row.get("store_name", row.get("city", row.get("store_id", "the matched store"))))
    city_values = store_matches["city"].dropna().astype(str).unique().tolist() if "city" in store_matches.columns else []
    if len(city_values) == 1:
        return city_values[0]
    return "the matched stores"


def _sales_with_products(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    sales = frames["sales"].copy()
    products = frames["products"].copy()
    if sales.empty:
        return pd.DataFrame()
    sales["store_id"] = sales["store_id"].astype(str)
    sales["product_id"] = sales["product_id"].astype(str)
    sales["quantity_sold"] = pd.to_numeric(sales["quantity_sold"], errors="coerce").fillna(0)
    if "selling_price" in sales.columns:
        sales["selling_price"] = pd.to_numeric(sales["selling_price"], errors="coerce").fillna(0)
    if not products.empty:
        products["product_id"] = products["product_id"].astype(str)
        sales = sales.merge(
            products[["product_id", "product_name", "category"]],
            on="product_id",
            how="left",
        )
    return sales


def _answer_least_or_top_sold(
    frames: dict[str, pd.DataFrame],
    question: str,
    intent: str,
    matched_stores: pd.DataFrame,
) -> AnalyticsRoute:
    if matched_stores.empty:
        return AnalyticsRoute(
            True,
            intent,
            _friendly_no_data(
                "I couldn’t match that store or city yet. Can you check whether it exists in stores.csv?",
                "You can ask with a city name like Vijayawada or a store name like Hyderabad Central.",
            ),
            pd.DataFrame(),
            _build_sources(["stores"]),
        )

    sales = _sales_with_products(frames)
    if sales.empty:
        return AnalyticsRoute(
            True,
            intent,
            _friendly_no_data(
                f"I couldn’t find sales data for {_store_label(matched_stores)} yet.",
                "Want me to check low stock or recommendations for that branch instead?",
            ),
            pd.DataFrame(),
            _build_sources(["sales"]),
        )

    sales = sales[sales["store_id"].isin(matched_stores["store_id"].astype(str))]
    if sales.empty:
        return AnalyticsRoute(
            True,
            intent,
            _friendly_no_data(
                f"I couldn’t find sales data for {_store_label(matched_stores)} yet.",
                "Want me to check inventory or recommendations for that location?",
            ),
            pd.DataFrame(),
            _build_sources(["sales", "stores"]),
        )

    grouped = (
        sales.groupby(["product_id", "product_name"], as_index=False)["quantity_sold"]
        .sum()
        .sort_values(["quantity_sold", "product_name"], ascending=[True, True])
    )
    if grouped.empty:
        return AnalyticsRoute(True, intent, _friendly_no_data("I couldn’t compute a sales ranking from the current sales data."), pd.DataFrame(), _build_sources(["sales"]))

    result_row = grouped.iloc[0] if intent == "least_sold_product_by_store" else grouped.sort_values(
        ["quantity_sold", "product_name"],
        ascending=[False, True],
    ).iloc[0]
    store_name = _store_label(matched_stores)
    quantity = int(result_row.get("quantity_sold", 0))
    product_name = str(result_row.get("product_name", result_row.get("product_id", "Unknown product")))

    if intent == "least_sold_product_by_store":
        answer = f"The least sold product in {store_name} is {product_name}, with {quantity} units sold in the current sales data."
        explanation = "That suggests demand is weaker there than for the other products sold in that branch."
        suggestion = "You may want to check whether it needs a discount, better placement, or a slower reorder cycle."
    else:
        answer = f"The top selling product in {store_name} is {product_name}, with {quantity} units sold in the current sales data."
        explanation = "That means it is currently leading demand in that branch."
        suggestion = "You may want to make sure its stock cover stays healthy so the store does not lose sales."

    supporting_df = grouped.head(8).copy()
    payload = _empty_payload()
    payload.update(
        {
            "answer": answer,
            "explanation": explanation,
            "suggestions": [suggestion],
            "follow_up_question": "Want the full top and bottom product list for that store?",
            "confidence": "high",
            "supporting_points": [
                f"Matched store scope: {store_name}",
                f"Computed from summed quantity_sold grouped by product.",
            ],
        }
    )
    payload = _maybe_humanize(question, payload)
    return AnalyticsRoute(True, intent, payload, supporting_df, _build_sources(["sales", "products", "stores"]))


def _answer_sales_summary_by_store(
    frames: dict[str, pd.DataFrame],
    question: str,
    matched_stores: pd.DataFrame,
) -> AnalyticsRoute:
    if matched_stores.empty:
        return AnalyticsRoute(
            True,
            "sales_summary_by_store",
            _friendly_no_data(
                "I couldn’t match that store or city yet. Can you check whether it exists in stores.csv?",
                "You can ask with a city name like Hyderabad or a store name like Vijayawada Branch.",
            ),
            pd.DataFrame(),
            _build_sources(["stores"]),
        )

    sales = _sales_with_products(frames)
    if sales.empty:
        return AnalyticsRoute(True, "sales_summary_by_store", _friendly_no_data("I couldn’t find sales data in sales.csv yet."), pd.DataFrame(), _build_sources(["sales"]))

    sales = sales[sales["store_id"].isin(matched_stores["store_id"].astype(str))].copy()
    if sales.empty:
        return AnalyticsRoute(
            True,
            "sales_summary_by_store",
            _friendly_no_data(
                f"I couldn’t find sales data for {_store_label(matched_stores)} yet.",
                "Want me to check inventory or low-stock items there instead?",
            ),
            pd.DataFrame(),
            _build_sources(["sales", "stores"]),
        )

    sales["sales_value"] = sales["quantity_sold"] * sales.get("selling_price", 0)
    by_store = (
        sales.groupby("store_id", as_index=False)
        .agg(
            total_units_sold=("quantity_sold", "sum"),
            total_sales_value=("sales_value", "sum"),
        )
        .merge(
            matched_stores[["store_id", "store_name", "city"]],
            on="store_id",
            how="left",
        )
        .sort_values("total_units_sold", ascending=False)
    )

    top_product = (
        sales.groupby(["product_id", "product_name"], as_index=False)["quantity_sold"]
        .sum()
        .sort_values(["quantity_sold", "product_name"], ascending=[False, True])
        .iloc[0]
    )
    label = _store_label(matched_stores)
    total_units = int(by_store["total_units_sold"].sum())
    total_value = float(by_store["total_sales_value"].sum())
    answer = f"Sales in {label} total {total_units} units in the current sales data, worth about {total_value:,.2f}."
    explanation = f"The strongest selling product there is {top_product['product_name']} with {int(top_product['quantity_sold'])} units sold."
    suggestion = "If you want, I can also compare that branch against another store or show the weakest products there."

    payload = _empty_payload()
    payload.update(
        {
            "answer": answer,
            "explanation": explanation,
            "suggestions": [suggestion],
            "follow_up_question": "Want product-wise detail for that store?",
            "confidence": "high",
            "supporting_points": [
                f"Matched store scope: {label}",
                "Computed directly from sales.csv grouped by store and product.",
            ],
        }
    )
    payload = _maybe_humanize(question, payload)
    return AnalyticsRoute(True, "sales_summary_by_store", payload, by_store.head(8), _build_sources(["sales", "stores", "products"]))


def _answer_low_stock_by_store(
    frames: dict[str, pd.DataFrame],
    matched_stores: pd.DataFrame,
) -> AnalyticsRoute:
    if matched_stores.empty:
        return AnalyticsRoute(
            True,
            "low_stock_by_store",
            _friendly_no_data(
                "I couldn’t match that branch yet. Can you check whether the store or city exists in stores.csv?",
                "You can ask with a city name or exact branch name.",
            ),
            pd.DataFrame(),
            _build_sources(["stores"]),
        )

    low_stock = frames["low_stock_items"].copy()
    if low_stock.empty:
        inventory = frames["inventory"].copy()
        products = frames["products"].copy()
        if inventory.empty:
            return AnalyticsRoute(True, "low_stock_by_store", _friendly_no_data("I couldn’t find inventory data yet."), pd.DataFrame(), _build_sources(["inventory"]))
        inventory["stock_level"] = pd.to_numeric(inventory["stock_level"], errors="coerce").fillna(0)
        inventory["reorder_threshold"] = pd.to_numeric(inventory["reorder_threshold"], errors="coerce").fillna(0)
        low_stock = inventory[inventory["stock_level"] <= inventory["reorder_threshold"]].copy()
        if not products.empty:
            low_stock = low_stock.merge(
                products[["product_id", "product_name", "category"]],
                on="product_id",
                how="left",
            )
    low_stock["store_id"] = low_stock["store_id"].astype(str)
    low_stock = low_stock[low_stock["store_id"].isin(matched_stores["store_id"].astype(str))].copy()

    if low_stock.empty:
        label = _store_label(matched_stores)
        payload = _empty_payload()
        payload.update(
            {
                "answer": f"I don’t see any low-stock items in {label} right now.",
                "explanation": "Based on the current inventory data, nothing there is sitting at or below its reorder threshold.",
                "suggestions": ["If you want, I can still check stockout risk or slow-moving items for that branch."],
                "follow_up_question": "",
                "confidence": "high",
            }
        )
        payload = _maybe_humanize(f"low stock in {label}", payload)
        return AnalyticsRoute(True, "low_stock_by_store", payload, pd.DataFrame(), _build_sources(["low_stock_items", "inventory", "stores"]))

    label = _store_label(matched_stores)
    count = len(low_stock)
    names = ", ".join(low_stock["product_name"].fillna(low_stock["product_id"]).astype(str).head(3).tolist())
    payload = _empty_payload()
    payload.update(
        {
            "answer": f"{label} has {count} low-stock item{'s' if count != 1 else ''} in the current data.",
            "explanation": f"The most immediate ones include {names}." if names else "These items are at or below their reorder thresholds.",
            "suggestions": ["You may want to review reorder or transfer recommendations for that branch next."],
            "follow_up_question": "Want the full low-stock list for that branch?",
            "confidence": "high",
        }
    )
    payload = _maybe_humanize(f"low stock in {label}", payload)
    return AnalyticsRoute(True, "low_stock_by_store", payload, low_stock.head(10), _build_sources(["low_stock_items", "inventory", "stores", "products"]))


def _answer_product_performance(
    frames: dict[str, pd.DataFrame],
    question: str,
    matched_stores: pd.DataFrame,
    matched_products: pd.DataFrame,
) -> AnalyticsRoute:
    if matched_products.empty:
        return AnalyticsRoute(False, "", _empty_payload(), pd.DataFrame(), [])

    sales = _sales_with_products(frames)
    if sales.empty:
        return AnalyticsRoute(True, "product_performance", _friendly_no_data("I couldn’t find sales data for that product yet."), pd.DataFrame(), _build_sources(["sales", "products"]))

    sales = sales[sales["product_id"].isin(matched_products["product_id"].astype(str))].copy()
    if not matched_stores.empty:
        sales = sales[sales["store_id"].isin(matched_stores["store_id"].astype(str))]

    if sales.empty:
        product_name = str(matched_products.iloc[0].get("product_name", matched_products.iloc[0].get("product_id", "that product")))
        return AnalyticsRoute(
            True,
            "product_performance",
            _friendly_no_data(
                f"I couldn’t find sales data for {product_name} in that scope yet.",
                "Want me to check its inventory or recommendation status instead?",
            ),
            pd.DataFrame(),
            _build_sources(["sales", "products", "stores"]),
        )

    summary = (
        sales.groupby(["product_id", "product_name"], as_index=False)
        .agg(total_units_sold=("quantity_sold", "sum"))
        .sort_values("total_units_sold", ascending=False)
    )
    top_row = summary.iloc[0]
    product_name = str(top_row.get("product_name", top_row.get("product_id", "That product")))
    total_units = int(top_row.get("total_units_sold", 0))
    scope = _store_label(matched_stores) if not matched_stores.empty else "the available sales data"

    payload = _empty_payload()
    payload.update(
        {
            "answer": f"{product_name} has sold {total_units} units in {scope}.",
            "explanation": "That is based on summed quantity_sold for the matching product in the current sales data.",
            "suggestions": ["If you want, I can also compare it with another product or store."],
            "follow_up_question": "Want a store-wise split for this product?",
            "confidence": "high",
        }
    )
    payload = _maybe_humanize(question, payload)
    return AnalyticsRoute(True, "product_performance", payload, summary.head(8), _build_sources(["sales", "products", "stores"]))


def _answer_supplier_risk(frames: dict[str, pd.DataFrame], matched_products: pd.DataFrame) -> AnalyticsRoute:
    suppliers = frames["suppliers"].copy()
    products = frames["products"].copy()
    performance = frames["product_performance"].copy()

    if suppliers.empty:
        return AnalyticsRoute(True, "supplier_risk", _friendly_no_data("I couldn’t find supplier data yet."), pd.DataFrame(), _build_sources(["suppliers"]))

    suppliers["reliability_score"] = pd.to_numeric(suppliers["reliability_score"], errors="coerce").fillna(0)
    suppliers["avg_delivery_days"] = pd.to_numeric(suppliers["avg_delivery_days"], errors="coerce").fillna(0)
    risky = suppliers.sort_values(["reliability_score", "avg_delivery_days"], ascending=[True, False]).copy()
    if risky.empty:
        return AnalyticsRoute(True, "supplier_risk", _friendly_no_data("I couldn’t compute supplier risk from the current data."), pd.DataFrame(), _build_sources(["suppliers"]))

    if not matched_products.empty and not products.empty:
        product_supplier_ids = set(
            products[products["product_id"].astype(str).isin(matched_products["product_id"].astype(str))]
            ["supplier_id"]
            .astype(str)
            .tolist()
        )
        risky = risky[risky["supplier_id"].astype(str).isin(product_supplier_ids)].copy()
        if risky.empty:
            risky = suppliers.sort_values(["reliability_score", "avg_delivery_days"], ascending=[True, False]).copy()

    row = risky.iloc[0]
    supplier_name = str(row.get("supplier_name", row.get("supplier_id", "Unknown supplier")))
    reliability = float(row.get("reliability_score", 0))
    delivery_days = float(row.get("avg_delivery_days", 0))

    impacted_products = pd.DataFrame()
    if not performance.empty and "supplier_id" in performance.columns:
        impacted_products = performance[performance["supplier_id"].astype(str) == str(row.get("supplier_id", ""))].copy()

    explanation = f"I’m flagging {supplier_name} because its reliability score is {reliability:.2f} and its average delivery time is {delivery_days:.0f} days."
    if not impacted_products.empty:
        example_products = ", ".join(
            impacted_products["product_name"].fillna(impacted_products["product_id"]).astype(str).head(3).tolist()
        )
        explanation = f"{explanation} It affects products like {example_products}."

    payload = _empty_payload()
    payload.update(
        {
            "answer": f"The riskiest supplier in the current data is {supplier_name}.",
            "explanation": explanation,
            "suggestions": ["You may want to review reorder timing or identify backup supply options for the affected products."],
            "follow_up_question": "Want me to show the products linked to that supplier?",
            "confidence": "high",
        }
    )
    payload = _maybe_humanize("supplier risk", payload)
    supporting_df = impacted_products.head(8) if not impacted_products.empty else risky.head(8)
    return AnalyticsRoute(True, "supplier_risk", payload, supporting_df, _build_sources(["suppliers", "product_performance", "products"]))


def _answer_general_inventory(frames: dict[str, pd.DataFrame], matched_stores: pd.DataFrame, matched_products: pd.DataFrame) -> AnalyticsRoute:
    inventory = frames["inventory"].copy()
    products = frames["products"].copy()
    if inventory.empty:
        return AnalyticsRoute(True, "general_inventory_question", _friendly_no_data("I couldn’t find inventory data yet."), pd.DataFrame(), _build_sources(["inventory"]))

    inventory["store_id"] = inventory["store_id"].astype(str)
    inventory["product_id"] = inventory["product_id"].astype(str)
    inventory["stock_level"] = pd.to_numeric(inventory["stock_level"], errors="coerce").fillna(0)

    if not matched_stores.empty:
        inventory = inventory[inventory["store_id"].isin(matched_stores["store_id"].astype(str))]
    if not matched_products.empty:
        inventory = inventory[inventory["product_id"].isin(matched_products["product_id"].astype(str))]
    if not products.empty:
        inventory = inventory.merge(
            products[["product_id", "product_name", "category"]],
            on="product_id",
            how="left",
        )

    if inventory.empty:
        return AnalyticsRoute(
            True,
            "general_inventory_question",
            _friendly_no_data("I couldn’t find inventory rows for that product or store yet."),
            pd.DataFrame(),
            _build_sources(["inventory", "products", "stores"]),
        )

    total_stock = int(inventory["stock_level"].sum())
    scope = _store_label(matched_stores) if not matched_stores.empty else "the current inventory data"
    payload = _empty_payload()
    payload.update(
        {
            "answer": f"There are {total_stock} units in {scope}.",
            "explanation": "That total comes directly from the current inventory snapshot.",
            "suggestions": ["If you want, I can also break that down by product or highlight low-stock rows."],
            "follow_up_question": "",
            "confidence": "high",
        }
    )
    payload = _maybe_humanize("inventory question", payload)
    supporting_df = inventory.sort_values("stock_level", ascending=False).head(10)
    return AnalyticsRoute(True, "general_inventory_question", payload, supporting_df, _build_sources(["inventory", "products", "stores"]))


def try_answer_analytical_question(
    user_input: str,
    chat_history=None,
) -> AnalyticsRoute:
    frames = _load_chatbot_frames()
    stores = frames["stores"].copy()
    products = frames["products"].copy()
    matched_stores = _match_stores(user_input, stores)
    matched_products = _match_products(user_input, products)
    intent = _detect_analytical_intent(user_input, matched_stores)

    if _has_analytical_signal(user_input, matched_stores, matched_products) and intent:
        if intent == "least_sold_product_by_store":
            return _answer_least_or_top_sold(frames, user_input, intent, matched_stores)
        if intent == "top_sold_product_by_store":
            return _answer_least_or_top_sold(frames, user_input, intent, matched_stores)
        if intent == "low_stock_by_store":
            return _answer_low_stock_by_store(frames, matched_stores)
        if intent == "sales_summary_by_store":
            return _answer_sales_summary_by_store(frames, user_input, matched_stores)
        if intent == "product_performance":
            route = _answer_product_performance(frames, user_input, matched_stores, matched_products)
            if route.handled:
                return route
        if intent == "supplier_risk":
            return _answer_supplier_risk(frames, matched_products)
        if intent == "general_inventory_question":
            return _answer_general_inventory(frames, matched_stores, matched_products)

    follow_up_route = _answer_follow_up_from_memory(user_input, chat_history)
    if follow_up_route.handled:
        return follow_up_route

    if intent == "product_performance":
        route = _answer_product_performance(frames, user_input, matched_stores, matched_products)
        if route.handled:
            return route
    if intent == "supplier_risk":
        return _answer_supplier_risk(frames, matched_products)
    if intent == "general_inventory_question":
        return _answer_general_inventory(frames, matched_stores, matched_products)

    return AnalyticsRoute(False, intent, _empty_payload(), pd.DataFrame(), [])
