from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from backend.db import repository


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
STORES_FILE = RAW_DATA_DIR / "stores.csv"
PRODUCTS_FILE = RAW_DATA_DIR / "products.csv"

TYPO_REPLACEMENTS = {
    "hyderabadd": "hyderabad",
    "vijaywada": "vijayawada",
}

BUSINESS_TERMS = {
    "branch",
    "branches",
    "store",
    "stores",
    "warehouse",
    "warehouses",
    "location",
    "locations",
    "inventory",
    "stock",
    "sales",
    "selling",
    "sold",
    "items",
    "item",
    "products",
    "product",
    "category",
    "categories",
    "top",
    "least",
    "lowest",
    "highest",
    "best",
    "most",
    "show",
    "give",
    "list",
    "compare",
    "comparison",
    "wise",
}

TEMPORAL_TERMS = {
    "today",
    "yesterday",
    "tomorrow",
    "week",
    "month",
    "year",
    "quarter",
    "daily",
    "weekly",
    "monthly",
    "current",
    "latest",
    "recent",
    "last",
    "this",
    "next",
    "past",
}

QUESTION_TERMS = {
    "which",
    "what",
    "where",
    "when",
    "why",
    "how",
    "has",
    "have",
    "with",
    "without",
}

LOCATION_CONTEXT_TERMS = {
    "branch",
    "branches",
    "store",
    "stores",
    "warehouse",
    "warehouses",
    "city",
    "cities",
    "country",
    "countries",
    "location",
    "locations",
    "network",
    "distribution",
}


@dataclass
class LocationValidation:
    requested_location: str = ""
    is_available: bool = True
    available_locations: list[str] | None = None
    payload: dict[str, Any] | None = None


def _normalize(text: Any) -> str:
    cleaned = re.sub(r"[^a-z0-9\s]", " ", str(text or "").lower())
    normalized = " ".join(cleaned.split())
    if not normalized:
        return ""
    words = [TYPO_REPLACEMENTS.get(word, word) for word in normalized.split()]
    return " ".join(words)


def _unique_clean_values(values: list[Any]) -> list[str]:
    seen = set()
    cleaned_values = []
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        key = _normalize(text)
        if not key or key in seen:
            continue
        seen.add(key)
        cleaned_values.append(text)
    return cleaned_values


def _available_cities(stores: pd.DataFrame) -> list[str]:
    if stores.empty or "city" not in stores.columns:
        return []
    cities = _unique_clean_values(stores["city"].tolist())
    preferred_order = {"hyderabad": 0, "guntur": 1, "vijayawada": 2}
    return sorted(
        cities,
        key=lambda city: (preferred_order.get(_normalize(city), 100), cities.index(city)),
    )


def _store_aliases(stores: pd.DataFrame) -> set[str]:
    aliases: set[str] = set()
    if stores.empty:
        return aliases
    for _, row in stores.iterrows():
        for column in ["store_id", "store_name", "city"]:
            normalized = _normalize(row.get(column, ""))
            if normalized:
                aliases.add(normalized)
    return aliases


def _product_aliases(products: pd.DataFrame) -> set[str]:
    aliases: set[str] = set()
    if products.empty:
        return aliases
    for _, row in products.iterrows():
        for column in ["product_id", "product_name", "category"]:
            normalized = _normalize(row.get(column, ""))
            if normalized:
                aliases.add(normalized)
    return aliases


def _has_business_signal(text: str) -> bool:
    tokens = set(text.split())
    return bool(tokens & (BUSINESS_TERMS | LOCATION_CONTEXT_TERMS))


def _candidate_is_noise(candidate: str, product_aliases: set[str]) -> bool:
    tokens = candidate.split()
    if not tokens:
        return True
    token_set = set(tokens)
    if token_set <= (BUSINESS_TERMS | TEMPORAL_TERMS | QUESTION_TERMS):
        return True
    if token_set & TEMPORAL_TERMS and not token_set & LOCATION_CONTEXT_TERMS:
        return True
    if candidate in product_aliases:
        return True
    if any(alias and (candidate == alias or candidate in alias) for alias in product_aliases):
        return True
    return False


def _clean_candidate(raw_candidate: str) -> str:
    candidate = _normalize(raw_candidate)
    if not candidate:
        return ""

    stop_phrases = [
        " instead",
        " today",
        " yesterday",
        " tomorrow",
        " this week",
        " last week",
        " this month",
        " last month",
        " by ",
        " with ",
        " compared ",
        " versus ",
        " vs ",
    ]
    for phrase in stop_phrases:
        index = candidate.find(phrase)
        if index > 0:
            candidate = candidate[:index].strip()

    words = [
        word
        for word in candidate.split()
        if word not in {"branch", "branches", "store", "stores", "warehouse", "warehouses"}
    ]
    return " ".join(words).strip()


def _extract_requested_location(question: str, product_aliases: set[str]) -> str:
    text = _normalize(question)
    if not text or not _has_business_signal(text):
        return ""

    patterns = [
        r"\b(?:in|at|from|near)\s+([a-z][a-z0-9\s]{1,40})$",
        r"\b(?:in|at|from|near)\s+([a-z][a-z0-9\s]{1,40}?)(?:\s+(?:branch|branches|store|stores|warehouse|warehouses|city|country|network))\b",
        r"\b(?:branch|store|warehouse|city|country)\s+(?:in|at|for|of)\s+([a-z][a-z0-9\s]{1,40})",
        r"\b([a-z][a-z0-9\s]{1,30}?)\s+(?:branch|store|warehouse)\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        candidate = _clean_candidate(match.group(1))
        if candidate and not _candidate_is_noise(candidate, product_aliases):
            return candidate
    return ""


def _display_name(normalized_location: str) -> str:
    return " ".join(word.capitalize() for word in normalized_location.split())


def _format_locations(locations: list[str]) -> str:
    if not locations:
        return "the configured branch locations"
    if len(locations) == 1:
        return locations[0]
    if len(locations) == 2:
        return f"{locations[0]} and {locations[1]}"
    return f"{', '.join(locations[:-1])}, and {locations[-1]}"


def _unsupported_location_payload(requested_location: str, available_locations: list[str], question: str) -> dict[str, Any]:
    location_name = _display_name(requested_location)
    location_list = _format_locations(available_locations)
    text = _normalize(question)

    if "inventory" in text or "stock" in text or "warehouse" in text:
        answer = (
            f"We do not currently have a branch or warehouse in {location_name}. "
            f"Available branches in the current network are {location_list}."
        )
        follow_up = f"Would you like me to check inventory in {location_list} instead?"
    else:
        answer = (
            f"We currently operate only across {location_list} branches, so I could not find "
            f"a branch or sales records for {location_name} in the current distribution network."
        )
        follow_up = f"Would you like me to show top-selling items in {location_list} instead?"

    return {
        "answer": answer,
        "explanation": (
            f"The current store master lists only {location_list}, and {location_name} is outside that network."
        ),
        "suggestions": [
            f"Top-selling items in {available_locations[0]}" if available_locations else "Top-selling items by branch",
            "Branch-wise sales comparison",
            "Category-wise sales trends",
        ],
        "follow_up_question": follow_up,
        "confidence": "high",
        "supporting_points": [
            f"Supported branch locations: {location_list}.",
            f"Requested location: {location_name}.",
        ],
        "cannot_answer": True,
        "_debug_answer_path": "location_validation",
        "_debug_retrieval_mode": "pre_retrieval",
    }


def validate_requested_location(question: str) -> LocationValidation:
    """Detect unsupported location requests before analytics or vector retrieval."""
    text = _normalize(question)
    if not text:
        return LocationValidation()

    stores = repository.load_stores(safe=True)
    products = repository.load_products(safe=True)
    available_locations = _available_cities(stores)
    store_aliases = _store_aliases(stores)
    product_aliases = _product_aliases(products)

    if any(alias and re.search(rf"\b{re.escape(alias)}\b", text) for alias in store_aliases):
        return LocationValidation(is_available=True, available_locations=available_locations)

    requested_location = _extract_requested_location(text, product_aliases)
    if not requested_location:
        return LocationValidation(is_available=True, available_locations=available_locations)

    if requested_location in store_aliases:
        return LocationValidation(
            requested_location=requested_location,
            is_available=True,
            available_locations=available_locations,
        )

    return LocationValidation(
        requested_location=requested_location,
        is_available=False,
        available_locations=available_locations,
        payload=_unsupported_location_payload(requested_location, available_locations, question),
    )
