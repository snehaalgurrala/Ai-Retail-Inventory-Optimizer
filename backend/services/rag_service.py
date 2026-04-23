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
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field

from backend.memory.memory_store import get_system_memory_summary
from backend.services.llm_reasoner import get_llm_settings, llm_is_configured


load_dotenv()


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
RAG_DIR = PROCESSED_DATA_DIR / "rag"
CHROMA_DIR = RAG_DIR / "chroma"
MANIFEST_FILE = RAG_DIR / "manifest.json"
COLLECTION_NAME = "retail_inventory_rag"

SOURCE_FILES = {
    "products": RAW_DATA_DIR / "products.csv",
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
}


class RAGAnswer(BaseModel):
    answer: str
    confidence: str = "medium"
    supporting_points: list[str] = Field(default_factory=list)
    cannot_answer: bool = False


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
    return {
        "provider": llm_settings["provider"],
        "api_key": llm_settings["api_key"],
        "base_url": llm_settings["base_url"],
        "embedding_model": llm_settings.get("embedding_model", "").strip()
        or (
            "gemini-embedding-001"
            if llm_settings["provider"] == "gemini"
            else "text-embedding-3-small"
        ),
        "chat_model": llm_settings["model"],
    }


def rag_is_configured() -> bool:
    """Return True when embeddings and chat config are available."""
    settings = _embedding_settings()
    return bool(settings["api_key"] and settings["chat_model"] and settings["embedding_model"])


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


def _build_chat_model() -> ChatOpenAI:
    """Create chat model for grounded answer generation."""
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
        "product_id",
        "product_name",
        "store_id",
        "store_name",
        "recommendation_type",
        "stock_level",
        "effective_reorder_threshold",
        "total_quantity_sold",
        "recent_7_day_quantity_sold",
        "recent_30_day_quantity_sold",
        "recent_daily_sales_velocity",
        "days_of_stock_remaining",
        "priority",
        "action",
        "reason",
        "evidence",
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
        "I found relevant project records and listed them in the supporting table. "
        f"Retrieved context came from: {dataset_summary}. "
        "Configure the LLM settings to enable natural-language grounded answers."
    )


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
        "The chatbot is configured correctly, but the Gemini API quota is currently "
        "exhausted. Please try again later or use a key with available quota."
    )


def answer_question_with_rag(
    question: str,
    top_k: int = 6,
) -> tuple[str, pd.DataFrame, list[dict]]:
    """Retrieve relevant project records and answer with grounded LLM output."""
    cleaned_question = str(question or "").strip()
    if not cleaned_question:
        return (
            "Ask a question about project data, inventory, sales, products, or recommendations.",
            pd.DataFrame(),
            [],
        )

    if not rag_is_configured():
        return (
            "RAG is not configured yet. Set Gemini or LLM environment variables to enable retrieval-based answers.",
            pd.DataFrame(),
            [],
        )

    try:
        vector_store = ensure_vector_store()
        retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
        documents = retriever.invoke(cleaned_question)
    except Exception as error:
        if _is_quota_exhausted_error(error):
            return _quota_exhausted_message(), pd.DataFrame(), []
        return (
            f"Could not retrieve project context: {error}",
            pd.DataFrame(),
            [],
        )

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
        }
        for document in documents
    ]

    if not documents:
        return (
            "I could not find any relevant project records for that question.",
            pd.DataFrame(),
            [],
        )

    if not llm_is_configured():
        return _fallback_answer(cleaned_question, documents), supporting_df, sources

    prompt = (
        "You are a retail inventory assistant answering only from retrieved project data.\n"
        "Use only the context below.\n"
        "Do not invent facts, products, stores, or metrics.\n"
        "If the context is not enough, say so clearly.\n"
        "Answer the user's question directly, then provide 2-4 short supporting points.\n\n"
        f"User question:\n{cleaned_question}\n\n"
        f"Retrieved context:\n{retrieved_context}"
    )

    try:
        llm_settings = get_llm_settings()
        if llm_settings["provider"] == "gemini":
            client = _build_chat_model()
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
            chat_model = _build_chat_model().with_structured_output(RAGAnswer)
            rag_answer = chat_model.invoke(prompt)

        if rag_answer.cannot_answer:
            answer_text = (
                rag_answer.answer
                or "I could not answer that confidently from the retrieved project data."
            )
        else:
            support_text = "\n".join(
                f"- {point}" for point in rag_answer.supporting_points if point
            )
            answer_text = rag_answer.answer.strip()
            if support_text:
                answer_text = f"{answer_text}\n\n{support_text}"
    except Exception as error:
        if _is_quota_exhausted_error(error):
            answer_text = _quota_exhausted_message()
            return answer_text, supporting_df, sources
        answer_text = _fallback_answer(cleaned_question, documents)

    return answer_text, supporting_df, sources
