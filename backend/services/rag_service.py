import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from google.genai import Client
from google.genai import types as genai_types
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from backend.services import llm_reasoner


load_dotenv(override=True)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
VECTOR_STORE_DIR = PROCESSED_DATA_DIR / "vector_store"
MANIFEST_FILE = VECTOR_STORE_DIR / "manifest.json"

SOURCE_FILES = {
    "inventory": RAW_DATA_DIR / "inventory.csv",
    "products": RAW_DATA_DIR / "products.csv",
    "sales": RAW_DATA_DIR / "sales.csv",
    "stores": RAW_DATA_DIR / "stores.csv",
    "suppliers": RAW_DATA_DIR / "suppliers.csv",
    "transactions": RAW_DATA_DIR / "transactions.csv",
    "recommendations": PROCESSED_DATA_DIR / "recommendations.csv",
    "agent_outputs": PROCESSED_DATA_DIR / "agent_outputs.csv",
    "orchestrator_summary": PROCESSED_DATA_DIR / "orchestrator_summary.csv",
    "customer_orders": PROCESSED_DATA_DIR / "customer_orders.csv",
}

INDEX_FILE = VECTOR_STORE_DIR / "index.faiss"
STORE_FILE = VECTOR_STORE_DIR / "index.pkl"

LAST_RETRIEVAL_STATUS: dict[str, Any] = {
    "answer_path": "idle",
    "retrieval_mode": "fallback",
    "last_error": "",
    "embedding_backend": "",
    "embedding_backend_error": "",
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


class LocalHashEmbeddings(Embeddings):
    """Deterministic offline embeddings when external model loading is unavailable."""

    dimension = 384

    @staticmethod
    def _hash_token(token: str) -> tuple[int, float]:
        digest = hashlib.md5(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:2], "little") % LocalHashEmbeddings.dimension
        sign = 1.0 if digest[2] % 2 == 0 else -1.0
        return index, sign

    def _embed(self, text: str) -> list[float]:
        vector = np.zeros(self.dimension, dtype=np.float32)
        for token in _tokenize(text):
            index, sign = self._hash_token(token)
            vector[index] += sign
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector /= norm
        return vector.astype(float).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def _vector_dependency_details() -> dict[str, Any]:
    """Return dependency-level details for local vector retrieval."""
    details = {
        "faiss_available": False,
        "sentence_transformers_available": False,
        "langchain_community_available": False,
        "langchain_embeddings_available": False,
        "langchain_faiss_available": False,
        "errors": [],
        "faiss_error": "",
        "sentence_transformers_error": "",
        "langchain_community_error": "",
        "langchain_embeddings_error": "",
        "langchain_faiss_error": "",
    }

    try:
        import faiss  # noqa: F401
        details["faiss_available"] = True
    except Exception as error:
        details["faiss_error"] = str(error)
        details["errors"].append(f"faiss import failed: {error}")

    try:
        import sentence_transformers  # noqa: F401
        details["sentence_transformers_available"] = True
    except Exception as error:
        details["sentence_transformers_error"] = str(error)
        details["errors"].append(f"sentence_transformers import failed: {error}")

    try:
        import langchain_community  # noqa: F401
        details["langchain_community_available"] = True
    except Exception as error:
        details["langchain_community_error"] = str(error)
        details["errors"].append(f"langchain_community import failed: {error}")

    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings  # noqa: F401
        details["langchain_embeddings_available"] = True
    except Exception as error:
        details["langchain_embeddings_error"] = str(error)
        details["errors"].append(f"HuggingFaceEmbeddings import failed: {error}")

    try:
        from langchain_community.vectorstores import FAISS  # noqa: F401
        details["langchain_faiss_available"] = True
    except Exception as error:
        details["langchain_faiss_error"] = str(error)
        details["errors"].append(f"LangChain FAISS import failed: {error}")

    details["ready"] = all(
        [
            details["faiss_available"],
            details["sentence_transformers_available"],
            details["langchain_community_available"],
            details["langchain_embeddings_available"],
            details["langchain_faiss_available"],
        ]
    )
    return details


def _vector_dependency_status() -> tuple[bool, str]:
    """Check whether local vector embedding dependencies are available."""
    details = _vector_dependency_details()
    if details["ready"]:
        return True, ""
    if details["errors"]:
        return False, details["errors"][0]
    return False, "Vector RAG dependencies are not available in the current Python runtime."


def _vector_imports():
    """Import FAISS and HuggingFace embedding classes lazily."""
    dependency_ready, message = _vector_dependency_status()
    if not dependency_ready:
        raise RuntimeError(message)

    from langchain_community.vectorstores import FAISS

    return FAISS


def _embedding_imports():
    """Import HuggingFace embeddings lazily without requiring FAISS."""
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings
    except Exception as error:
        raise RuntimeError(f"HuggingFaceEmbeddings import failed: {error}") from error


def _safe_text(value) -> str:
    """Convert values to readable strings for document building."""
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return str(value)


def _index_files_exist() -> bool:
    """Return True when the persisted FAISS files are present."""
    return INDEX_FILE.exists() and STORE_FILE.exists()


def _source_fingerprint() -> str:
    """Create a hash of source files to know when to rebuild the FAISS index."""
    fingerprint_parts = []
    for name, file_path in SOURCE_FILES.items():
        if not file_path.exists():
            fingerprint_parts.append(f"{name}:missing")
            continue
        stat = file_path.stat()
        fingerprint_parts.append(f"{name}:{stat.st_mtime_ns}:{stat.st_size}")

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


def build_rag_documents() -> list[Document]:
    """Create retrievable documents from the latest project CSV files."""
    documents: list[Document] = []
    for dataset_name, df in _load_source_frames().items():
        if df.empty:
            continue
        for row_index, (_, row) in enumerate(df.iterrows()):
            documents.append(_row_to_document(dataset_name, row_index, row))
    return documents


def _embedding_settings() -> dict[str, str]:
    """Load embedding-specific settings."""
    llm_settings = llm_reasoner.get_llm_settings()
    provider = str(llm_settings.get("provider", "openrouter") or "openrouter").strip().lower()
    api_key = str(llm_settings.get("api_key", "") or "").strip()
    base_url = str(llm_settings.get("base_url", "") or "").strip()
    model = str(llm_settings.get("model", "") or "").strip()
    embedding_model = str(llm_settings.get("embedding_model", "") or "").strip()

    return {
        "provider": provider,
        "api_key": api_key,
        "base_url": base_url,
        "embedding_model": embedding_model,
        "chat_model": model,
    }


def rag_is_configured() -> bool:
    """Return True when the chat LLM is configured, even if embeddings are optional."""
    settings = _embedding_settings()
    return bool(settings["api_key"] and settings["chat_model"])


@lru_cache(maxsize=4)
def _probe_embedding_model(embedding_model: str) -> tuple[bool, str]:
    """Probe whether local embedding dependencies are ready for the configured model."""
    try:
        _, backend_name, backend_error = _build_embeddings()
        if backend_name == "local_hash" and backend_error:
            return True, f"Using offline local hash embeddings because the HuggingFace model is unavailable: {backend_error}"
        return True, ""
    except Exception as error:
        return False, str(error)


def _embedding_model_runtime_status() -> tuple[bool, str]:
    """Check whether the configured embedding model can be initialized."""
    settings = _embedding_settings()
    embedding_model = str(settings.get("embedding_model", "") or "").strip()
    return _probe_embedding_model(embedding_model)


def vector_rag_is_configured() -> bool:
    """Return True only when dependencies and embeddings are ready for vector retrieval."""
    settings = _embedding_settings()
    dependency_ready, _ = _vector_dependency_status()
    if not settings["embedding_model"] or not dependency_ready:
        return False
    embedding_ready, _ = _embedding_model_runtime_status()
    return bool(embedding_ready)


def check_vector_rag_environment() -> dict[str, Any]:
    """Return explicit runtime diagnostics for Streamlit vector RAG setup."""
    dependency_details = _vector_dependency_details()
    settings = _embedding_settings()
    embedding_ready, embedding_error = _embedding_model_runtime_status()
    index_exists = _index_files_exist()
    can_build_index = bool(
        dependency_details["ready"]
        and embedding_ready
        and bool(settings.get("embedding_model", ""))
    )
    retrieval_mode = "vector_rag" if can_build_index and index_exists else "fallback"
    fallback_reason = ""
    if not dependency_details["faiss_available"]:
        fallback_reason = f"faiss import failed: {dependency_details.get('faiss_error', '')}"
    elif not dependency_details["sentence_transformers_available"]:
        fallback_reason = f"sentence_transformers import failed: {dependency_details.get('sentence_transformers_error', '')}"
    elif not dependency_details["langchain_community_available"]:
        fallback_reason = f"langchain_community import failed: {dependency_details.get('langchain_community_error', '')}"
    elif not dependency_details["langchain_faiss_available"]:
        fallback_reason = f"LangChain FAISS import failed: {dependency_details.get('langchain_faiss_error', '')}"
    elif not settings.get("embedding_model", ""):
        fallback_reason = "EMBEDDING_MODEL is not configured."
    elif not embedding_ready:
        fallback_reason = f"Embedding model failed to initialize: {embedding_error}"
    elif not index_exists:
        fallback_reason = "FAISS index does not exist yet; it can be built on first vector retrieval or by rebuilding the knowledge index."

    return {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "current_working_directory": os.getcwd(),
        "faiss_available": bool(dependency_details["faiss_available"]),
        "faiss_error": str(dependency_details.get("faiss_error", "")),
        "sentence_transformers_available": bool(dependency_details["sentence_transformers_available"]),
        "sentence_transformers_error": str(dependency_details.get("sentence_transformers_error", "")),
        "langchain_community_available": bool(dependency_details["langchain_community_available"]),
        "langchain_community_error": str(dependency_details.get("langchain_community_error", "")),
        "langchain_faiss_available": bool(dependency_details["langchain_faiss_available"]),
        "langchain_faiss_error": str(dependency_details.get("langchain_faiss_error", "")),
        "embedding_model": str(settings.get("embedding_model", "") or ""),
        "embedding_model_loaded": bool(embedding_ready),
        "embedding_model_error": str(embedding_error or ""),
        "faiss_index_exists": bool(index_exists),
        "can_build_index": can_build_index,
        "retrieval_mode": retrieval_mode,
        "fallback_reason": fallback_reason,
    }


def _vector_manifest_summary() -> dict[str, Any]:
    """Return a small summary from the vector-store manifest."""
    manifest = _load_manifest()
    return {
        "document_count": int(manifest.get("document_count", 0) or 0),
        "embedding_model": str(manifest.get("embedding_model", "") or ""),
        "embedding_backend": str(manifest.get("embedding_backend", "") or ""),
        "fingerprint": str(manifest.get("fingerprint", "") or ""),
    }


def get_vector_debug_status() -> dict[str, Any]:
    """Return detailed runtime status for the chatbot sidebar."""
    settings = _embedding_settings()
    dependency_details = _vector_dependency_details()
    embedding_ready, embedding_error = _embedding_model_runtime_status()
    manifest_summary = _vector_manifest_summary()
    last_error = str(LAST_RETRIEVAL_STATUS.get("last_error", "") or "")
    embedding_backend = str(
        LAST_RETRIEVAL_STATUS.get("embedding_backend", "")
        or manifest_summary.get("embedding_backend", "")
        or ("huggingface" if embedding_ready else "")
    ).strip()
    embedding_backend_error = str(LAST_RETRIEVAL_STATUS.get("embedding_backend_error", "") or "").strip()
    dependency_error = "; ".join(dependency_details["errors"][:2])
    retrieval_mode = str(LAST_RETRIEVAL_STATUS.get("retrieval_mode", "") or "").strip()
    answer_path = str(LAST_RETRIEVAL_STATUS.get("answer_path", "idle") or "idle")
    if (not retrieval_mode or retrieval_mode == "fallback") and answer_path == "idle":
        retrieval_mode = "vector_rag" if vector_rag_is_configured() and _index_files_exist() else "fallback"

    return {
        "llm_active": rag_is_configured(),
        "embedding_model": settings["embedding_model"] or "",
        "embedding_model_loaded": embedding_ready,
        "embedding_model_error": embedding_error,
        "embedding_backend": embedding_backend,
        "embedding_backend_error": embedding_backend_error,
        "faiss_available": bool(dependency_details["faiss_available"]),
        "faiss_index_exists": _index_files_exist(),
        "indexed_documents": manifest_summary["document_count"],
        "retrieval_mode": retrieval_mode,
        "answer_path": answer_path,
        "python_executable": sys.executable,
        "dependency_error": dependency_error,
        "last_error": last_error,
    }


def chatbot_config_status() -> dict[str, str | bool]:
    """Return a small config summary for chatbot UI messaging."""
    settings = _embedding_settings()
    missing_fields = []
    if not settings["api_key"]:
        missing_fields.append("OPENROUTER_API_KEY")
    if not settings["chat_model"]:
        missing_fields.append("OPENROUTER_MODEL")

    status_message_fn = getattr(llm_reasoner, "llm_status_message", None)
    status_message = (
        status_message_fn()
        if callable(status_message_fn)
        else "LLM is not configured. Add OPENROUTER_API_KEY in .env."
    )
    dependency_ready, dependency_message = _vector_dependency_status()
    embedding_ready, embedding_error = _embedding_model_runtime_status()
    debug_status = get_vector_debug_status()
    vector_rag_message = ""
    if not dependency_ready:
        vector_rag_message = (
            f"Vector RAG is disabled in this runtime: {dependency_message}"
        )
    elif not settings["embedding_model"]:
        vector_rag_message = "Set EMBEDDING_MODEL in .env to enable local vector RAG."
    elif MANIFEST_FILE.exists():
        backend_name = str(debug_status.get("embedding_backend", "") or "").strip()
        if backend_name == "local_hash":
            vector_rag_message = "Knowledge index is ready with the offline local embedding fallback."
        else:
            vector_rag_message = "Knowledge index is ready."
    else:
        if embedding_ready and not embedding_error:
            vector_rag_message = "Knowledge index will be built automatically on first retrieval."
        else:
            vector_rag_message = "Knowledge index will be built automatically with the offline local embedding fallback."

    return {
        "configured": len(missing_fields) == 0,
        "provider": settings["provider"] or "unknown",
        "chat_model": settings["chat_model"] or "",
        "embedding_model": settings["embedding_model"] or "",
        "base_url": settings["base_url"] or "",
        "status_message": status_message,
        "vector_rag_configured": vector_rag_is_configured(),
        "vector_rag_message": vector_rag_message,
        "faiss_index_exists": bool(debug_status["faiss_index_exists"]),
        "indexed_documents": int(debug_status["indexed_documents"]),
        "retrieval_mode": str(debug_status["retrieval_mode"]),
        "missing_message": (
            "Missing environment variables: " + ", ".join(missing_fields)
            if missing_fields
            else ""
        ),
    }


@lru_cache(maxsize=2)
def _build_embeddings():
    """Create embeddings for the FAISS store, with an offline-safe fallback."""
    settings = _embedding_settings()
    if not settings["embedding_model"]:
        raise RuntimeError(
            "Set EMBEDDING_MODEL in .env to enable local vector RAG."
        )

    huggingface_error = ""
    try:
        HuggingFaceEmbeddings = _embedding_imports()
        embeddings = HuggingFaceEmbeddings(
            model_name=settings["embedding_model"],
            model_kwargs={
                "device": "cpu",
                "local_files_only": True,
            },
            encode_kwargs={"normalize_embeddings": True},
        )
        embeddings.embed_query("inventory health check")
        return embeddings, "huggingface", ""
    except Exception as error:
        huggingface_error = str(error)

    return LocalHashEmbeddings(), "local_hash", huggingface_error


def _build_chat_model() -> ChatOpenAI | Client:
    """Create the chat model used for grounded answer generation."""
    settings = llm_reasoner.get_llm_settings()
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
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_FILE.write_text(
        json.dumps(data, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )


def ensure_vector_store(force_rebuild: bool = False):
    """Build or load the persistent FAISS vector store."""
    if not vector_rag_is_configured():
        LAST_RETRIEVAL_STATUS["last_error"] = chatbot_config_status().get(
            "vector_rag_message",
            "Vector RAG is not configured.",
        )
        raise RuntimeError(
            LAST_RETRIEVAL_STATUS["last_error"]
        )

    FAISS = _vector_imports()
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    current_fingerprint = _source_fingerprint()
    manifest = _load_manifest()
    stored_fingerprint = manifest.get("fingerprint", "")
    stored_backend = str(manifest.get("embedding_backend", "") or "").strip()
    index_exists = _index_files_exist()
    embeddings, embedding_backend, embedding_backend_error = _build_embeddings()
    LAST_RETRIEVAL_STATUS["embedding_backend"] = embedding_backend
    LAST_RETRIEVAL_STATUS["embedding_backend_error"] = embedding_backend_error

    if index_exists and stored_backend and stored_backend != embedding_backend:
        force_rebuild = True

    if not force_rebuild and index_exists and current_fingerprint == stored_fingerprint:
        return FAISS.load_local(
            str(VECTOR_STORE_DIR),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    if not index_exists and not force_rebuild:
        force_rebuild = True

    if force_rebuild or current_fingerprint != stored_fingerprint:
        documents = build_rag_documents()
        if documents:
            vector_store = FAISS.from_documents(documents, embeddings)
        else:
            vector_store = FAISS.from_texts(["empty knowledge base"], embeddings)
        vector_store.save_local(str(VECTOR_STORE_DIR))
        _save_manifest(
            {
                "fingerprint": current_fingerprint,
                "document_count": len(documents),
                "embedding_model": _embedding_settings().get("embedding_model", ""),
                "embedding_backend": embedding_backend,
            }
        )
        return vector_store

    return FAISS.load_local(
        str(VECTOR_STORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def clear_vector_store() -> None:
    """Remove the persisted vector store safely inside the project workspace."""
    if VECTOR_STORE_DIR.exists():
        shutil.rmtree(VECTOR_STORE_DIR)


def rebuild_knowledge_index() -> dict[str, Any]:
    """Force a rebuild of the local FAISS knowledge index."""
    try:
        clear_vector_store()
        vector_store = ensure_vector_store(force_rebuild=True)
        manifest = _load_manifest()
        LAST_RETRIEVAL_STATUS.update(
            {
                "answer_path": "rag",
                "retrieval_mode": "vector_rag",
                "last_error": "",
            }
        )
        return {
            "success": True,
            "message": (
                f"Knowledge index rebuilt successfully with "
                f"{manifest.get('document_count', 0)} document chunks."
            ),
            "document_count": int(manifest.get("document_count", 0) or 0),
            "embedding_model": str(manifest.get("embedding_model", "")),
            "embedding_backend": str(manifest.get("embedding_backend", "")),
            "vector_store_path": str(VECTOR_STORE_DIR),
            "vector_store_type": type(vector_store).__name__,
        }
    except Exception as error:
        LAST_RETRIEVAL_STATUS.update(
            {
                "answer_path": "rag",
                "retrieval_mode": "fallback",
                "last_error": str(error),
            }
        )
        return {
            "success": False,
            "message": str(error),
            "document_count": 0,
            "embedding_model": _embedding_settings().get("embedding_model", ""),
            "embedding_backend": "",
            "vector_store_path": str(VECTOR_STORE_DIR),
        }


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
        return "I couldn't find a clear answer in the current data yet. Want me to check another area?"

    dataset_counts = {}
    for document in documents:
        dataset_name = document.metadata.get("dataset", "unknown")
        dataset_counts[dataset_name] = dataset_counts.get(dataset_name, 0) + 1

    dataset_summary = ", ".join(
        f"{name} ({count})" for name, count in dataset_counts.items()
    )
    return f"I found relevant data in {dataset_summary}, but I need a clearer question to turn it into a stronger business answer."


def _fallback_payload(question: str, documents: list[Document]) -> dict:
    """Build a safe structured fallback response."""
    answer_text = _fallback_answer(question, documents)
    dataset_counts = {}
    for document in documents:
        dataset_name = document.metadata.get("dataset", "unknown")
        dataset_counts[dataset_name] = dataset_counts.get(dataset_name, 0) + 1

    explanation = "This answer comes from the current project data I could match to your question."
    suggestions = []
    if any(name in dataset_counts for name in {"recommendations", "agent_outputs"}):
        suggestions.append("It may help to review the latest recommendations next.")
    if any(name in dataset_counts for name in {"customer_orders", "inventory"}):
        suggestions.append("If you placed a new order, refresh the agents before acting on it.")

    return {
        "answer": answer_text,
        "explanation": explanation,
        "suggestions": suggestions[:1],
        "follow_up_question": "Want me to narrow it down by product, store, or supplier?",
        "confidence": "medium",
        "supporting_points": [],
        "cannot_answer": False,
    }


def _no_data_payload() -> dict:
    """Return the standard response when no relevant evidence is available."""
    return {
        "answer": "I couldn't find a clear answer in the current data yet. Want me to check another area?",
        "explanation": "",
        "suggestions": [],
        "follow_up_question": "I can check stock levels, sales trends, or recommendations next.",
        "confidence": "low",
        "supporting_points": [],
        "cannot_answer": True,
    }


def _clarification_payload(question: str) -> dict:
    """Return a safe clarification prompt when the question is too vague."""
    return {
        "answer": "I want to make sure I'm looking at the right thing.",
        "explanation": "Please mention the product, store, supplier, or time period you want me to check.",
        "suggestions": [],
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
        ][:1],
        "follow_up_question": str(payload.get("follow_up_question", "") or "").strip(),
        "confidence": str(payload.get("confidence", "low") or "low").strip().lower(),
        "supporting_points": [
            str(item).strip()
            for item in payload.get("supporting_points", [])
            if str(item).strip()
        ][:2],
        "cannot_answer": bool(payload.get("cannot_answer", False)),
    }

    if supporting_df.empty or not sources:
        return _no_data_payload()

    if not validated["answer"]:
        return _no_data_payload()

    question_tokens = set(_tokenize(question))
    if (
        len(question_tokens) <= 2
        and not validated["supporting_points"]
        and not _looks_like_follow_up(question)
    ):
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

    if len(question_tokens) > 2 and not _looks_like_follow_up(question):
        overlap = len(question_tokens & set(_tokenize(evidence_text)))
        if overlap == 0:
            return _no_data_payload()

    if validated["confidence"] == "low" and not validated["supporting_points"]:
        return _no_data_payload()

    if not validated["explanation"]:
        validated["explanation"] = "This is based on the current project data linked to your question."

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
        validated["supporting_points"] = auto_points[:2]

    if not validated["follow_up_question"]:
        if "store_id" in supporting_df.columns and supporting_df["store_id"].astype(str).str.strip().any():
            validated["follow_up_question"] = "Want store-wise detail too?"
        elif "product_name" in supporting_df.columns and supporting_df["product_name"].astype(str).str.strip().any():
            validated["follow_up_question"] = "Want me to go deeper on a specific product?"
        elif "source_agent" in supporting_df.columns or "agent_name" in supporting_df.columns:
            validated["follow_up_question"] = "Should I also check supplier risk or recommendations?"

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
        "explain",
        "and",
        "then",
    )
    if len(_tokenize(cleaned_question)) > 3:
        return False
    return any(
        cleaned_question == item or cleaned_question.startswith(f"{item} ")
        for item in follow_up_starts
    )


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
    if not llm_reasoner.llm_is_configured():
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
        settings = llm_reasoner.get_llm_settings()
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
    top_k: int = 8,
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
    retrieval_mode = "fallback"
    if vector_rag_is_configured():
        try:
            vector_store = ensure_vector_store()
            retriever = vector_store.as_retriever(search_kwargs={"k": max(top_k * 2, 10)})
            documents = retriever.invoke(retrieval_query)
            retrieval_mode = "vector_rag"
        except Exception as error:
            LAST_RETRIEVAL_STATUS["last_error"] = str(error)
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
        documents = _simple_retrieve(retrieval_query, top_k=max(top_k * 2, 10))
        LAST_RETRIEVAL_STATUS["last_error"] = chatbot_config_status().get("vector_rag_message", "")

    LAST_RETRIEVAL_STATUS.update(
        {
            "answer_path": "rag",
            "retrieval_mode": retrieval_mode,
        }
    )

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

    if not llm_reasoner.llm_is_configured():
        return _fallback_payload(cleaned_question, documents), supporting_df, sources

    prompt = (
        "You are a retail business analyst assistant.\n"
        "Speak naturally, like a sharp but calm business analyst.\n"
        "Keep responses concise: 3 to 6 lines unless the user asks for more detail.\n"
        "Use this structure when possible: one direct answer line, one or two short explanation lines, one practical suggestion if needed, and an optional short follow-up question.\n"
        "Do not use headings, labels, or system-style phrases.\n"
        "Do not over-explain.\n"
        "Use recent conversation only to resolve follow-up references like 'why' or 'what about this store'.\n"
        "Use only the retrieved project data as factual evidence.\n"
        "Do not invent facts, products, stores, or metrics.\n"
        "If the context is not enough, say: 'I couldn't find anything significant in the current data. Want me to check another area?'\n"
        "Answer custom business questions dynamically instead of relying on fixed question patterns.\n"
        "Reason like a business analyst about trends, underperformance, comparisons, risks, and next actions, but only from retrieved evidence.\n"
        "When relevant, explicitly name the responsible agent or source_agent from the retrieved data.\n"
        "When relevant, explain the recommendation reasoning using the retrieved reason and evidence fields.\n"
        "When relevant, suggest one of the grounded actions already supported by the data, such as reorder, transfer, discount, or clearance.\n"
        "Use at most one suggestion and one follow-up question.\n"
        "Occasionally include a short natural follow-up question like 'Want store-wise detail too?' or 'Should I also check supplier risk?'\n"
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
        llm_settings = llm_reasoner.get_llm_settings()
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
    answer_payload = llm_reasoner.humanize_chatbot_payload(
        cleaned_question,
        answer_payload,
        supporting_points=answer_payload.get("supporting_points", []),
        answer_style="retrieval",
    )
    answer_payload["_debug_retrieval_mode"] = retrieval_mode
    answer_payload["_debug_answer_path"] = "rag"
    return answer_payload, supporting_df, sources

