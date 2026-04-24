import hashlib
import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google.genai import Client
from google.genai import types as genai_types
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field

from backend.memory.memory_store import get_system_memory_summary
from backend.services.llm_reasoner import get_llm_settings, llm_is_configured


load_dotenv(override=True)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
RAG_DIR = PROCESSED_DATA_DIR / "rag"
CHROMA_DIR = RAG_DIR / "chroma"
MANIFEST_FILE = RAG_DIR / "manifest.json"
COLLECTION_NAME = "retail_inventory_rag"

SOURCE_FILES = {
    "inventory": RAW_DATA_DIR / "inventory.csv",
    "products": RAW_DATA_DIR / "products.csv",
    "sales": RAW_DATA_DIR / "sales.csv",
    "suppliers": RAW_DATA_DIR / "suppliers.csv",
    "transactions": RAW_DATA_DIR / "transactions.csv",
    "current_inventory": PROCESSED_DATA_DIR / "current_inventory.csv",
    "sales_summary": PROCESSED_DATA_DIR / "sales_summary.csv",
    "product_performance": PROCESSED_DATA_DIR / "product_performance.csv",
    "store_inventory_summary": PROCESSED_DATA_DIR / "store_inventory_summary.csv",
    "low_stock_items": PROCESSED_DATA_DIR / "low_stock_items.csv",
    "dead_stock_candidates": PROCESSED_DATA_DIR / "dead_stock_candidates.csv",
    "overstock_items": PROCESSED_DATA_DIR / "overstock_items.csv",
    "high_demand_items": PROCESSED_DATA_DIR / "high_demand_items.csv",
    "slow_moving_items": PROCESSED_DATA_DIR / "slow_moving_items.csv",
    "stockout_risk_items": PROCESSED_DATA_DIR / "stockout_risk_items.csv",
    "recommendations": PROCESSED_DATA_DIR / "recommendations.csv",
    "agent_outputs": PROCESSED_DATA_DIR / "agent_outputs.csv",
    "orchestrator_summary": PROCESSED_DATA_DIR / "orchestrator_summary.csv",
    "customer_orders": PROCESSED_DATA_DIR / "customer_orders.csv",
}


class RAGAnswer(BaseModel):
    answer: str
    explanation: str = ""
    suggestions: list[str] = Field(default_factory=list)
    follow_up_question: str = ""
    confidence: str = "medium"
    supporting_points: list[str] = Field(default_factory=list)
    cannot_answer: bool = False


class QueryIntent(BaseModel):
    retrieval_query: str
    business_goal: str = ""
    likely_datasets: list[str] = Field(default_factory=list)
    target_entities: list[str] = Field(default_factory=list)
    comparison_requested: bool = False
    clarification_needed: bool = False
    clarification_question: str = ""


class GeminiEmbeddings(Embeddings):
    """LangChain embeddings adapter for Gemini."""

    batch_size = 64

    def __init__(self, api_key: str, model: str):
        self.client = Client(api_key=api_key)
        self.model = model

    @staticmethod
    def _extract_values(response) -> list[float]:
        """Extract embedding values from Gemini response shapes."""
        if hasattr(response, "embeddings") and response.embeddings:
            first = response.embeddings[0]
            if hasattr(first, "values"):
                return list(first.values)
        if hasattr(response, "embedding") and hasattr(response.embedding, "values"):
            return list(response.embedding.values)
        if isinstance(response, dict):
            embeddings = response.get("embeddings")
            if embeddings:
                first = embeddings[0]
                if isinstance(first, dict) and "values" in first:
                    return list(first["values"])
            embedding = response.get("embedding")
            if isinstance(embedding, dict) and "values" in embedding:
                return list(embedding["values"])
        raise ValueError("Unexpected Gemini embedding response format.")

    def _extract_batch_values(self, response) -> list[list[float]]:
        """Extract multiple embedding vectors from Gemini batch responses."""
        if hasattr(response, "embeddings") and response.embeddings:
            values = []
            for embedding in response.embeddings:
                if hasattr(embedding, "values"):
                    values.append(list(embedding.values))
            if values:
                return values
        if isinstance(response, dict):
            embeddings = response.get("embeddings", [])
            values = []
            for embedding in embeddings:
                if isinstance(embedding, dict) and "values" in embedding:
                    values.append(list(embedding["values"]))
            if values:
                return values
        return [self._extract_values(response)]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for index in range(0, len(texts), self.batch_size):
            batch = texts[index:index + self.batch_size]
            response = self.client.models.embed_content(
                model=self.model,
                contents=batch,
                config=genai_types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
            )
            vectors.extend(self._extract_batch_values(response))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        response = self.client.models.embed_content(
            model=self.model,
            contents=text,
            config=genai_types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
        )
        return self._extract_values(response)


def _safe_text(value) -> str:
    """Convert values to readable strings for document building."""
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return str(value)


def _source_fingerprint() -> str:
    """Create a hash of source files and memory summary to know when to rebuild."""
    fingerprint_parts = []
    for name, file_path in SOURCE_FILES.items():
        if not file_path.exists():
            fingerprint_parts.append(f"{name}:missing")
            continue
        stat = file_path.stat()
        fingerprint_parts.append(f"{name}:{stat.st_mtime_ns}:{stat.st_size}")

    memory_summary = get_system_memory_summary(limit=5)
    fingerprint_parts.append(json.dumps(memory_summary, sort_keys=True))
    fingerprint_text = "|".join(fingerprint_parts)
    return hashlib.sha256(fingerprint_text.encode("utf-8")).hexdigest()


def _load_source_frames() -> dict[str, pd.DataFrame]:
    """Load all available source datasets for embeddings."""
    frames = {}
    for name, file_path in SOURCE_FILES.items():
        if not file_path.exists():
            frames[name] = pd.DataFrame()
            continue
        try:
            frames[name] = pd.read_csv(file_path)
        except Exception:
            frames[name] = pd.DataFrame()
    return frames


def _row_to_document(dataset_name: str, row_index: int, row: pd.Series) -> Document:
    """Convert one dataframe row into a retrievable document."""
    row_dict = {}
    content_lines = [f"dataset: {dataset_name}", f"row_index: {row_index}"]

    for column, value in row.items():
        text_value = _safe_text(value)
        if text_value == "":
            continue
        row_dict[column] = text_value
        content_lines.append(f"{column}: {text_value}")

    metadata = {
        "dataset": dataset_name,
        "row_index": row_index,
        "product_id": row_dict.get("product_id", ""),
        "product_name": row_dict.get("product_name", ""),
        "store_id": row_dict.get("store_id", ""),
        "recommendation_type": row_dict.get("recommendation_type", ""),
        "row_json": json.dumps(row_dict, ensure_ascii=True),
    }
    return Document(
        page_content="\n".join(content_lines),
        metadata=metadata,
    )


def _memory_documents() -> list[Document]:
    """Create a compact document from system memory history."""
    memory_summary = get_system_memory_summary(limit=10)
    content_lines = [
        "dataset: memory_summary",
        f"recommendation_history_count: {memory_summary.get('recommendation_history_count', 0)}",
        f"decision_history_count: {memory_summary.get('decision_history_count', 0)}",
        f"outcome_history_count: {memory_summary.get('outcome_history_count', 0)}",
    ]

    for record in memory_summary.get("recent_decisions", [])[:5]:
        content_lines.append(
            "recent_decision: "
            + ", ".join(
                f"{key}={value}"
                for key, value in record.items()
                if value not in ("", None)
            )
        )
    for record in memory_summary.get("recent_outcomes", [])[:5]:
        content_lines.append(
            "recent_outcome: "
            + ", ".join(
                f"{key}={value}"
                for key, value in record.items()
                if value not in ("", None)
            )
        )

    return [
        Document(
            page_content="\n".join(content_lines),
            metadata={
                "dataset": "memory_summary",
                "row_index": 0,
                "product_id": "",
                "product_name": "",
                "store_id": "",
                "recommendation_type": "",
                "row_json": json.dumps(memory_summary, ensure_ascii=True),
            },
        )
    ]


def build_rag_documents() -> list[Document]:
    """Create retrievable documents from products, processed outputs, and recommendations."""
    documents: list[Document] = []
    for dataset_name, df in _load_source_frames().items():
        if df.empty:
            continue
        for row_index, (_, row) in enumerate(df.iterrows()):
            documents.append(_row_to_document(dataset_name, row_index, row))

    documents.extend(_memory_documents())
    return documents


def _embedding_settings() -> dict[str, str]:
    """Load embedding-specific settings."""
    llm_settings = get_llm_settings()
    provider = str(llm_settings.get("provider", "openai") or "openai").strip().lower()
    api_key = str(llm_settings.get("api_key", "") or "").strip()
    base_url = str(llm_settings.get("base_url", "") or "").strip()
    model = str(llm_settings.get("model", "") or "").strip()
    embedding_model = str(llm_settings.get("embedding_model", "") or "").strip()

    return {
        "provider": provider,
        "api_key": api_key,
        "base_url": base_url,
        "embedding_model": embedding_model
        or (
            "gemini-embedding-001"
            if provider == "gemini"
            else "text-embedding-3-small"
        ),
        "chat_model": model,
    }


def rag_is_configured() -> bool:
    """Return True when embeddings and chat config are available."""
    settings = _embedding_settings()
    return bool(settings["api_key"] and settings["chat_model"] and settings["embedding_model"])


def chatbot_config_status() -> dict[str, str | bool]:
    """Return a small config summary for chatbot UI messaging."""
    settings = _embedding_settings()
    missing_fields = []
    if not settings["api_key"]:
        missing_fields.append("LLM_API_KEY")
    if not settings["chat_model"]:
        missing_fields.append("LLM_MODEL")
    if not settings["embedding_model"]:
        missing_fields.append("EMBEDDING_MODEL")

    return {
        "configured": len(missing_fields) == 0,
        "provider": settings["provider"] or "unknown",
        "chat_model": settings["chat_model"] or "",
        "embedding_model": settings["embedding_model"] or "",
        "base_url": settings["base_url"] or "",
        "missing_message": (
            "Missing environment variables: " + ", ".join(missing_fields)
            if missing_fields
            else ""
        ),
    }


def _build_embeddings() -> OpenAIEmbeddings:
    """Create embeddings client for the vector store."""
    settings = _embedding_settings()
    if settings["provider"] == "gemini":
        return GeminiEmbeddings(
            api_key=settings["api_key"],
            model=settings["embedding_model"],
        )

    kwargs = {
        "api_key": settings["api_key"],
        "model": settings["embedding_model"],
    }
    if settings["base_url"]:
        kwargs["base_url"] = settings["base_url"]
    return OpenAIEmbeddings(**kwargs)


def _build_chat_model() -> ChatOpenAI | Client:
    """Create the chat model used for grounded answer generation."""
    settings = get_llm_settings()
    if settings["provider"] == "gemini":
        return Client(api_key=settings["api_key"])

    kwargs = {
        "api_key": settings["api_key"],
        "model": settings["model"],
        "temperature": 0.1,
    }
    if settings["base_url"]:
        kwargs["base_url"] = settings["base_url"]
    return ChatOpenAI(**kwargs)


def _load_manifest() -> dict:
    if not MANIFEST_FILE.exists():
        return {}
    try:
        return json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_manifest(data: dict) -> None:
    RAG_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_FILE.write_text(
        json.dumps(data, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )


def ensure_vector_store(force_rebuild: bool = False) -> Chroma:
    """Build or load the persistent Chroma vector store."""
    if not rag_is_configured():
        raise RuntimeError(
            "RAG is not configured. Set LLM_API_KEY, LLM_MODEL, and EMBEDDING_MODEL."
        )

    RAG_DIR.mkdir(parents=True, exist_ok=True)
    embeddings = _build_embeddings()
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    current_fingerprint = _source_fingerprint()
    manifest = _load_manifest()
    stored_fingerprint = manifest.get("fingerprint", "")

    if force_rebuild or current_fingerprint != stored_fingerprint:
        try:
            existing_ids = vector_store.get().get("ids", [])
            if existing_ids:
                vector_store.delete(ids=existing_ids)
        except Exception:
            pass

        documents = build_rag_documents()
        if documents:
            ids = [f"doc_{index}" for index in range(len(documents))]
            vector_store.add_documents(documents, ids=ids)

        _save_manifest(
            {
                "fingerprint": current_fingerprint,
                "document_count": len(documents),
            }
        )

    return vector_store


def _documents_to_supporting_table(documents: list[Document], limit: int = 8) -> pd.DataFrame:
    """Convert retrieved documents back into a compact supporting dataframe."""
    records = []
    for document in documents[:limit]:
        metadata = document.metadata or {}
        row_json = metadata.get("row_json", "{}")
        try:
            row_data = json.loads(row_json)
        except Exception:
            row_data = {}

        row_data["source_dataset"] = metadata.get("dataset", "")
        row_data["source_product_id"] = metadata.get("product_id", "")
        row_data["source_store_id"] = metadata.get("store_id", "")
        row_data["source_recommendation_type"] = metadata.get("recommendation_type", "")
        records.append(row_data)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    preferred_columns = [
        "source_dataset",
        "source_agent",
        "agent_name",
        "product_id",
        "product_name",
        "store_id",
        "store_name",
        "city",
        "recommendation_type",
        "stock_level",
        "quantity_ordered",
        "remaining_stock",
        "effective_reorder_threshold",
        "quantity_sold",
        "total_quantity_sold",
        "recent_7_day_quantity_sold",
        "recent_30_day_quantity_sold",
        "recent_daily_sales_velocity",
        "days_of_stock_remaining",
        "finding_count",
        "priority_level",
        "priority",
        "action",
        "reason",
        "evidence",
        "suggested_quantity",
        "summary",
        "database_health",
        "high_priority_alerts",
        "last_agent_run_time",
        "run_time",
        "decision",
        "outcome_status",
        "outcome_note",
    ]
    visible_columns = [column for column in preferred_columns if column in df.columns]
    return df[visible_columns].copy()


def _fallback_answer(question: str, documents: list[Document]) -> str:
    """Build a deterministic fallback answer when the LLM cannot be used."""
    if not documents:
        return (
            "I could not find enough relevant project data to answer that question."
        )

    dataset_counts = {}
    for document in documents:
        dataset_name = document.metadata.get("dataset", "unknown")
        dataset_counts[dataset_name] = dataset_counts.get(dataset_name, 0) + 1

    dataset_summary = ", ".join(
        f"{name} ({count})" for name, count in dataset_counts.items()
    )
    return (
        "I found a few relevant signals in the current data and listed the supporting records below. "
        f"The strongest context came from: {dataset_summary}. "
        "I’m keeping this answer grounded in those records while the richer explanation layer is unavailable."
    )


def _fallback_payload(question: str, documents: list[Document]) -> dict:
    """Build a safe structured fallback response."""
    answer_text = _fallback_answer(question, documents)
    dataset_counts = {}
    for document in documents:
        dataset_name = document.metadata.get("dataset", "unknown")
        dataset_counts[dataset_name] = dataset_counts.get(dataset_name, 0) + 1

    explanation = (
        "This response is grounded in the retrieved project records shown below. "
        "I used them directly instead of free-form reasoning because the live LLM "
        "response path was unavailable."
    )
    suggestions = []
    if any(name in dataset_counts for name in {"recommendations", "agent_outputs"}):
        suggestions.append(
            "Review the latest recommendation and agent output tables for the next action."
        )
    if any(name in dataset_counts for name in {"customer_orders", "inventory"}):
        suggestions.append(
            'If you recently placed an order, refresh the Home dashboard so agent outputs stay current.'
        )

    return {
        "answer": answer_text,
        "explanation": explanation,
        "suggestions": suggestions[:3],
        "follow_up_question": "Would you like me to narrow this down by product, store, or recommendation type?",
        "confidence": "medium",
        "supporting_points": [],
        "cannot_answer": False,
    }


def _no_data_payload() -> dict:
    """Return the standard response when no relevant evidence is available."""
    return {
        "answer": "I couldn't find enough data to answer that confidently.",
        "explanation": "Try asking about stock levels, sales trends, recommendations, a specific product, or a specific store.",
        "suggestions": [
            "Ask about stock levels for a product or store.",
            "Ask about sales trends or underperforming stores.",
            "Ask what should be reordered or transferred.",
        ],
        "follow_up_question": "Do you want me to dig deeper into a product, store, or supplier?",
        "confidence": "low",
        "supporting_points": [],
        "cannot_answer": True,
    }


def _clarification_payload(question: str) -> dict:
    """Return a safe clarification prompt when the question is too vague."""
    return {
        "answer": "I need a bit more detail to answer that reliably.",
        "explanation": "Please clarify the product, store, supplier, recommendation type, or time period you want me to check.",
        "suggestions": [
            f"Clarify the question: {question}",
            "Mention a product name or ID.",
            "Mention a store name or store ID.",
        ],
        "follow_up_question": "Do you want inventory, sales, supplier risk, or recommendations?",
        "confidence": "low",
        "supporting_points": [],
        "cannot_answer": True,
    }


def _validate_answer_payload(
    question: str,
    payload: dict,
    supporting_df: pd.DataFrame,
    sources: list[dict],
) -> dict:
    """Validate that the final answer stays grounded in retrieved evidence."""
    validated = {
        "answer": str(payload.get("answer", "") or "").strip(),
        "explanation": str(payload.get("explanation", "") or "").strip(),
        "suggestions": [
            str(item).strip()
            for item in payload.get("suggestions", [])
            if str(item).strip()
        ][:3],
        "follow_up_question": str(payload.get("follow_up_question", "") or "").strip(),
        "confidence": str(payload.get("confidence", "low") or "low").strip().lower(),
        "supporting_points": [
            str(item).strip()
            for item in payload.get("supporting_points", [])
            if str(item).strip()
        ][:4],
        "cannot_answer": bool(payload.get("cannot_answer", False)),
    }

    if supporting_df.empty or not sources:
        return _no_data_payload()

    if not validated["answer"]:
        return _no_data_payload()

    question_tokens = set(_tokenize(question))
    if len(question_tokens) <= 2 and not validated["supporting_points"]:
        return _clarification_payload(question)

    source_text = " ".join(
        " ".join(
            str(value or "")
            for value in source.values()
        )
        for source in sources
    ).lower()
    support_text = " ".join(str(value) for value in supporting_df.fillna("").astype(str).values.flatten()).lower()
    evidence_text = f"{source_text} {support_text}"

    if len(question_tokens) > 2:
        overlap = len(question_tokens & set(_tokenize(evidence_text)))
        if overlap == 0:
            return _no_data_payload()

    if validated["confidence"] == "low" and not validated["supporting_points"]:
        return _no_data_payload()

    if not validated["explanation"]:
        validated["explanation"] = "This answer is based only on the retrieved project records shown below."

    if not validated["supporting_points"]:
        evidence_columns = [
            column
            for column in ["reason", "evidence", "source_agent", "agent_name", "recommendation_type", "priority"]
            if column in supporting_df.columns
        ]
        auto_points = []
        if evidence_columns:
            first_row = supporting_df.iloc[0].fillna("").astype(str).to_dict()
            for column in evidence_columns:
                value = str(first_row.get(column, "")).strip()
                if value:
                    auto_points.append(f"{column.replace('_', ' ').title()}: {value}")
        validated["supporting_points"] = auto_points[:3]

    if not validated["follow_up_question"]:
        if "store_id" in supporting_df.columns and supporting_df["store_id"].astype(str).str.strip().any():
            validated["follow_up_question"] = "Do you want store-wise analysis?"
        elif "product_name" in supporting_df.columns and supporting_df["product_name"].astype(str).str.strip().any():
            validated["follow_up_question"] = "Should I check a specific product in more detail?"
        elif "source_agent" in supporting_df.columns or "agent_name" in supporting_df.columns:
            validated["follow_up_question"] = "Should I check supplier risks or recommendations too?"

    return validated


def _tokenize(text: str) -> list[str]:
    """Convert text into simple lowercase tokens for local retrieval."""
    cleaned = "".join(
        character.lower() if character.isalnum() else " "
        for character in str(text or "")
    )
    return [token for token in cleaned.split() if token]


def _history_to_lines(
    chat_history: list[BaseMessage] | None,
    max_messages: int = 6,
) -> list[str]:
    """Convert recent chat history to short role-tagged lines."""
    if not chat_history:
        return []

    lines = []
    for message in chat_history[-max_messages:]:
        content = str(getattr(message, "content", "") or "").strip()
        if not content:
            continue

        if isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        else:
            role = getattr(message, "type", "message")

        lines.append(f"{role}: {content}")
    return lines


def _looks_like_follow_up(question: str) -> bool:
    """Detect short follow-up questions that need conversation context."""
    cleaned_question = str(question or "").strip().lower()
    if not cleaned_question:
        return False

    follow_up_starts = (
        "why",
        "how",
        "what about",
        "how about",
        "and",
        "then",
        "also",
        "this",
        "that",
        "these",
        "those",
        "it",
        "they",
        "them",
        "which one",
        "tell me more",
        "explain",
    )
    return len(_tokenize(cleaned_question)) <= 6 or cleaned_question.startswith(follow_up_starts)


def _build_retrieval_query(
    question: str,
    chat_history: list[BaseMessage] | None = None,
) -> str:
    """Expand short follow-up questions with recent conversation context."""
    cleaned_question = str(question or "").strip()
    history_lines = _history_to_lines(chat_history, max_messages=4)
    if not history_lines:
        return cleaned_question

    if not _looks_like_follow_up(cleaned_question):
        return cleaned_question

    return (
        f"Current user question: {cleaned_question}\n"
        "Recent conversation context:\n"
        + "\n".join(history_lines)
    )


def _extract_json_answer(raw_text: str) -> RAGAnswer | None:
    """Try to parse JSON answer text into the structured response model."""
    text = str(raw_text or "").strip()
    if not text:
        return None

    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:] if lines else lines
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        parsed = json.loads(text)
        return RAGAnswer.model_validate(parsed)
    except Exception:
        return None


def _extract_json_intent(raw_text: str) -> QueryIntent | None:
    """Try to parse JSON intent text into the structured query-intent model."""
    text = str(raw_text or "").strip()
    if not text:
        return None

    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:] if lines else lines
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        parsed = json.loads(text)
        return QueryIntent.model_validate(parsed)
    except Exception:
        return None


def _heuristic_query_intent(
    question: str,
    chat_history: list[BaseMessage] | None = None,
) -> QueryIntent:
    """Build a lightweight intent summary without relying on fixed question mappings."""
    base_query = _build_retrieval_query(question, chat_history)
    tokens = set(_tokenize(question))
    datasets = []
    if {"store", "stores"} & tokens:
        datasets.extend(["store_inventory_summary", "sales_summary", "recommendations"])
    if {"sales", "trend", "dropping", "underperforming"} & tokens:
        datasets.extend(["sales", "sales_summary", "agent_outputs"])
    if {"supplier", "suppliers", "risky"} & tokens:
        datasets.extend(["suppliers", "recommendations", "agent_outputs"])
    if {"reorder", "procurement", "next"} & tokens:
        datasets.extend(["recommendations", "stockout_risk_items", "low_stock_items"])
    if {"compare", "vs", "versus"} & tokens:
        datasets.extend(["sales_summary", "store_inventory_summary", "recommendations"])

    return QueryIntent(
        retrieval_query=base_query,
        business_goal="Answer a custom retail business question using retrieved project data.",
        likely_datasets=list(dict.fromkeys(datasets))[:5],
        target_entities=[],
        comparison_requested=bool({"compare", "vs", "versus"} & tokens),
        clarification_needed=False,
        clarification_question="",
    )


def _infer_query_intent(
    question: str,
    chat_history: list[BaseMessage] | None = None,
) -> QueryIntent:
    """Use the LLM to interpret user intent before retrieval, with a safe heuristic fallback."""
    fallback_intent = _heuristic_query_intent(question, chat_history)
    if not llm_is_configured():
        return fallback_intent

    available_datasets = list(SOURCE_FILES.keys())
    recent_conversation = "\n".join(_history_to_lines(chat_history, max_messages=6))
    system_prompt = (
        "You interpret retail business questions before retrieval. "
        "Your job is to understand user intent, likely entities, and which datasets matter most. "
        "Do not answer the user's business question. "
        "Return valid JSON only."
    )
    user_prompt = json.dumps(
        {
            "question": question,
            "recent_conversation": recent_conversation,
            "available_datasets": available_datasets,
            "required_output_schema": {
                "retrieval_query": "a concise retrieval-friendly rewrite grounded in the user's meaning",
                "business_goal": "short description of what the user is trying to learn or decide",
                "likely_datasets": ["subset of available_datasets"],
                "target_entities": ["product/store/supplier names or ids when visible"],
                "comparison_requested": False,
                "clarification_needed": False,
                "clarification_question": "",
            },
        },
        ensure_ascii=True,
    )

    try:
        settings = get_llm_settings()
        if settings["provider"] == "gemini":
            client = Client(api_key=settings["api_key"])
            response = client.models.generate_content(
                model=settings["model"],
                contents=user_prompt,
                config=genai_types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.0,
                    response_mime_type="application/json",
                    response_schema=QueryIntent,
                ),
            )
            parsed = json.loads(getattr(response, "text", "") or "{}")
            intent = QueryIntent.model_validate(parsed)
        else:
            chat_model = _build_chat_model()
            raw_response = chat_model.invoke(
                system_prompt + "\n\n" + user_prompt
            )
            intent = _extract_json_intent(
                str(getattr(raw_response, "content", raw_response)).strip()
            )
            if intent is None:
                return fallback_intent

        valid_datasets = [name for name in intent.likely_datasets if name in available_datasets]
        intent.likely_datasets = valid_datasets[:6]
        if not intent.retrieval_query.strip():
            intent.retrieval_query = fallback_intent.retrieval_query
        return intent
    except Exception:
        return fallback_intent


def _rerank_documents_by_intent(
    documents: list[Document],
    intent: QueryIntent,
    question: str,
) -> list[Document]:
    """Re-rank retrieved documents using inferred business intent and entity overlap."""
    if not documents:
        return []

    likely_datasets = set(intent.likely_datasets)
    entity_tokens = {
        token.lower()
        for token in intent.target_entities
        if str(token).strip()
        for token in _tokenize(token)
    }
    question_tokens = set(_tokenize(question))
    rescored = []
    for index, document in enumerate(documents):
        metadata = document.metadata or {}
        row_json = metadata.get("row_json", "{}")
        try:
            row_data = json.loads(row_json)
        except Exception:
            row_data = {}

        score = 0
        dataset_name = str(metadata.get("dataset", ""))
        if dataset_name in likely_datasets:
            score += 6

        doc_text = " ".join(str(value or "") for value in row_data.values()).lower()
        doc_tokens = set(_tokenize(doc_text))
        score += len(question_tokens & doc_tokens)
        if entity_tokens:
            score += 2 * len(entity_tokens & doc_tokens)
        if intent.comparison_requested and dataset_name in {
            "sales_summary",
            "store_inventory_summary",
            "current_inventory",
            "recommendations",
        }:
            score += 3

        rescored.append((score, -index, document))

    rescored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [document for _, _, document in rescored]


def _simple_retrieve(question: str, top_k: int = 6) -> list[Document]:
    """Fallback retrieval without embeddings using token overlap and recency bias."""
    question_tokens = set(_tokenize(question))
    if not question_tokens:
        return []

    agent_query = bool(
        {"agent", "orchestrator", "summary", "risk", "suggested", "responsible"} & question_tokens
    )
    order_query = bool({"order", "orders", "changed", "latest"} & question_tokens)
    reorder_query = bool({"reorder", "procurement", "purchase"} & question_tokens)
    stock_query = bool({"stock", "inventory", "store"} & question_tokens)
    supplier_query = bool({"supplier", "suppliers", "risky"} & question_tokens)
    recommendation_query = bool(
        {"recommendation", "recommendations", "dead", "discount", "transfer", "clearance"} & question_tokens
    )

    scored_documents: list[tuple[int, int, Document]] = []
    for index, document in enumerate(build_rag_documents()):
        content_tokens = set(_tokenize(document.page_content))
        overlap = len(question_tokens & content_tokens)
        if overlap == 0:
            continue

        dataset_name = str(document.metadata.get("dataset", ""))
        recency_bonus = 0
        if dataset_name in {"customer_orders", "orchestrator_summary", "agent_outputs", "recommendations"}:
            recency_bonus = 3
        elif dataset_name in {"inventory", "sales"}:
            recency_bonus = 2

        intent_bonus = 0
        if agent_query and dataset_name in {"agent_outputs", "orchestrator_summary", "recommendations"}:
            intent_bonus += 6
        if order_query and dataset_name in {"customer_orders", "inventory", "sales", "transactions"}:
            intent_bonus += 5
        if reorder_query and dataset_name in {
            "recommendations",
            "low_stock_items",
            "stockout_risk_items",
        }:
            intent_bonus += 4
        if stock_query and dataset_name in {
            "inventory",
            "current_inventory",
            "low_stock_items",
            "store_inventory_summary",
        }:
            intent_bonus += 3
        if supplier_query and dataset_name in {"suppliers", "recommendations", "agent_outputs"}:
            intent_bonus += 5
        if recommendation_query and dataset_name in {
            "recommendations",
            "dead_stock_candidates",
            "agent_outputs",
            "orchestrator_summary",
        }:
            intent_bonus += 4

        scored_documents.append((overlap + recency_bonus + intent_bonus, -index, document))

    scored_documents.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [document for _, _, document in scored_documents[:top_k]]


def _is_quota_exhausted_error(error: Exception) -> bool:
    """Return True when an exception looks like a provider quota/rate-limit issue."""
    error_text = str(error).lower()
    quota_signals = [
        "resource_exhausted",
        "quota",
        "rate limit",
        "rate_limit",
        "429",
        "too many requests",
    ]
    return any(signal in error_text for signal in quota_signals)


def _quota_exhausted_message() -> str:
    """Return a friendly chatbot message for quota exhaustion."""
    return (
        "The chatbot is configured correctly, but the current LLM provider quota is "
        "currently exhausted. Please try again later or use a key with available quota."
    )


def answer_question_with_rag(
    question: str,
    chat_history: list[BaseMessage] | None = None,
    top_k: int = 6,
) -> tuple[dict, pd.DataFrame, list[dict]]:
    """Retrieve relevant project records and answer with grounded LLM output."""
    cleaned_question = str(question or "").strip()
    if not cleaned_question:
        return (
            {
                "answer": "Ask a question about project data, inventory, sales, products, or recommendations.",
                "explanation": "",
                "suggestions": [],
                "follow_up_question": "",
                "confidence": "low",
                "supporting_points": [],
                "cannot_answer": False,
            },
            pd.DataFrame(),
            [],
        )

    intent = _infer_query_intent(cleaned_question, chat_history)
    retrieval_query = intent.retrieval_query.strip() or _build_retrieval_query(cleaned_question, chat_history)
    history_lines = _history_to_lines(chat_history, max_messages=6)
    conversation_context = "\n".join(history_lines)
    retrieval_mode = "simple"
    if rag_is_configured():
        try:
            vector_store = ensure_vector_store()
            retriever = vector_store.as_retriever(search_kwargs={"k": max(top_k * 2, 8)})
            documents = retriever.invoke(retrieval_query)
            retrieval_mode = "embeddings"
        except Exception as error:
            if _is_quota_exhausted_error(error):
                return (
                    {
                        "answer": _quota_exhausted_message(),
                        "explanation": "The configured LLM provider is temporarily rate-limited, so the chatbot cannot generate a conversational explanation right now.",
                        "suggestions": [],
                        "follow_up_question": "",
                        "confidence": "low",
                        "supporting_points": [],
                        "cannot_answer": False,
                    },
                    pd.DataFrame(),
                    [],
                )
            documents = _simple_retrieve(retrieval_query, top_k=max(top_k * 2, 8))
    else:
        documents = _simple_retrieve(retrieval_query, top_k=max(top_k * 2, 8))

    documents = _rerank_documents_by_intent(documents, intent, cleaned_question)[:top_k]

    supporting_df = _documents_to_supporting_table(documents)
    retrieved_context = "\n\n".join(
        f"[{index + 1}] {document.page_content}"
        for index, document in enumerate(documents)
    )
    sources = [
        {
            "dataset": document.metadata.get("dataset", ""),
            "product_id": document.metadata.get("product_id", ""),
            "product_name": document.metadata.get("product_name", ""),
            "store_id": document.metadata.get("store_id", ""),
            "recommendation_type": document.metadata.get("recommendation_type", ""),
            "source_agent": json.loads(document.metadata.get("row_json", "{}")).get("source_agent", ""),
            "agent_name": json.loads(document.metadata.get("row_json", "{}")).get("agent_name", ""),
        }
        for document in documents
    ]

    if not documents:
        return (
            _no_data_payload(),
            pd.DataFrame(),
            [],
        )

    if not llm_is_configured():
        return _fallback_payload(cleaned_question, documents), supporting_df, sources

    prompt = (
        "You are a retail business analyst assistant.\n"
        "Be friendly, natural, professional, and commercially practical.\n"
        "Keep responses short but insightful.\n"
        "Do not over-explain.\n"
        "Use recent conversation only to resolve follow-up references like 'why' or 'what about this store'.\n"
        "Use only the retrieved project data as factual evidence.\n"
        "Do not invent facts, products, stores, or metrics.\n"
        "If the context is not enough, say so clearly.\n"
        "Answer custom business questions dynamically instead of relying on fixed question patterns.\n"
        "Reason like a business analyst about trends, underperformance, comparisons, risks, and next actions, but only from retrieved evidence.\n"
        "When relevant, explicitly name the responsible agent or source_agent from the retrieved data.\n"
        "When relevant, explain the recommendation reasoning using the retrieved reason and evidence fields.\n"
        "When relevant, suggest one of the grounded actions already supported by the data, such as reorder, transfer, discount, or clearance.\n"
        "Occasionally include a short natural follow-up question like 'Do you want store-wise analysis?' or 'Should I check supplier risks too?'\n"
        "Return valid JSON only with fields: answer, explanation, suggestions, follow_up_question, confidence, supporting_points, cannot_answer.\n"
        "Keep suggestions practical and grounded in the retrieved data.\n"
        f"Retrieval mode: {retrieval_mode}.\n\n"
        f"Inferred business goal: {intent.business_goal or 'Use the user question as the business goal.'}\n"
        f"Likely datasets: {', '.join(intent.likely_datasets) or 'Not specified'}\n"
        f"Target entities: {', '.join(intent.target_entities) or 'Not specified'}\n"
        f"Recent conversation:\n{conversation_context or 'No prior conversation.'}\n\n"
        f"User question:\n{cleaned_question}\n\n"
        f"Retrieved context:\n{retrieved_context}"
    )

    try:
        llm_settings = get_llm_settings()
        if llm_settings["provider"] == "gemini":
            client = Client(api_key=llm_settings["api_key"])
            response = client.models.generate_content(
                model=llm_settings["model"],
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    temperature=0.1,
                    response_mime_type="application/json",
                    response_schema=RAGAnswer,
                ),
            )
            parsed = json.loads(getattr(response, "text", "") or "{}")
            rag_answer = RAGAnswer.model_validate(parsed)
        else:
            chat_model = _build_chat_model()
            raw_answer = chat_model.invoke(
                (
                    "You are a retail inventory assistant answering only from retrieved "
                    "project data. Use recent conversation only to resolve references. "
                    "Do not invent facts. If the context is insufficient, say so clearly. "
                    "Return valid JSON only with keys answer, explanation, suggestions, "
                    "follow_up_question, confidence, supporting_points, cannot_answer.\n\n"
                    + prompt
                )
            )
            answer_text = str(getattr(raw_answer, "content", raw_answer)).strip()
            rag_answer = _extract_json_answer(answer_text) or RAGAnswer(
                answer=answer_text or "I could not produce a grounded answer from the retrieved project data.",
                explanation="This answer is based on the retrieved project records shown below.",
                supporting_points=[],
                suggestions=[],
                follow_up_question="Would you like me to dig into a specific product, store, or supplier next?",
                cannot_answer=False,
            )

        answer_payload = rag_answer.model_dump()
    except Exception as error:
        if _is_quota_exhausted_error(error):
            return (
                {
                    "answer": _quota_exhausted_message(),
                    "explanation": "The configured LLM provider is temporarily rate-limited, so this response could not be generated conversationally.",
                    "suggestions": [],
                    "follow_up_question": "",
                    "confidence": "low",
                    "supporting_points": [],
                    "cannot_answer": False,
                },
                supporting_df,
                sources,
            )
        answer_payload = _fallback_payload(cleaned_question, documents)

    answer_payload = _validate_answer_payload(
        cleaned_question,
        answer_payload,
        supporting_df,
        sources,
    )
    return answer_payload, supporting_df, sources
