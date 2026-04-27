from pathlib import Path
import sys
import importlib
import time

import pandas as pd
import streamlit as st
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from backend.services import rag_service  # noqa: E402
from backend.services.chatbot_router import route_chatbot_request  # noqa: E402
from frontend.utils.page_helpers import apply_page_style, render_page_header  # noqa: E402


MAX_CHAT_MEMORY_MESSAGES = 6
SAMPLE_QUESTIONS = [
    "Which products are low in stock?",
    "What should we reorder?",
    "Which supplier is risky?",
    "Which store needs transfer?",
    "Which products are high demand?",
    "Which store has excess stock?",
    "What changed after the latest order?",
    "Which agent found the highest risk?",
    "What should we reorder now?",
    "Why did the procurement agent suggest reorder?",
    "Which agent suggested this?",
    "Why is this product dead stock?",
    "Why is product P012 not selling?",
    "What is the best way to increase sales based on our trends?",
]


def _welcome_payload() -> dict:
    """Return the first-time greeting for new chat sessions."""
    return {
        "answer": "Hi there. I can help with inventory, sales, supplier risk, and recommendations.",
        "explanation": "Ask naturally and I'll keep it short, practical, and grounded in the data.",
        "suggestions": [],
        "follow_up_question": "Want store-wise detail or supplier risk first?",
        "confidence": "high",
        "supporting_points": [],
        "cannot_answer": False,
    }


def _get_rag_service():
    """Return a usable rag_service module, reloading if Streamlit holds a stale copy."""
    service = rag_service
    required_helpers = (
        "chatbot_config_status",
        "get_vector_debug_status",
        "rebuild_knowledge_index",
    )
    if all(callable(getattr(service, helper, None)) for helper in required_helpers):
        return service

    try:
        reloaded_service = importlib.reload(rag_service)
        if all(callable(getattr(reloaded_service, helper, None)) for helper in required_helpers):
            return reloaded_service
    except Exception:
        pass
    return service


def _chatbot_config_status() -> dict:
    """Read chatbot config safely, even during partial Streamlit reloads."""
    service = _get_rag_service()
    config_fn = getattr(service, "chatbot_config_status", None)
    if callable(config_fn):
        return config_fn()
    return {
        "configured": False,
        "provider": "unknown",
        "chat_model": "",
        "embedding_model": "",
        "base_url": "",
        "status_message": "LLM is not configured. Add OPENROUTER_API_KEY in .env.",
        "vector_rag_configured": False,
        "vector_rag_message": "Chatbot configuration helper is temporarily unavailable. Please refresh or restart Streamlit.",
        "missing_message": "Chatbot configuration helper is temporarily unavailable. Please refresh the app.",
    }


def _rag_is_configured() -> bool:
    """Check RAG config safely, even during partial Streamlit reloads."""
    service = _get_rag_service()
    configured_fn = getattr(service, "rag_is_configured", None)
    return bool(callable(configured_fn) and configured_fn())


def _rebuild_knowledge_index() -> dict:
    """Rebuild the local FAISS knowledge index safely."""
    service = importlib.reload(rag_service)
    rebuild_fn = getattr(service, "rebuild_knowledge_index", None)
    if callable(rebuild_fn):
        return rebuild_fn()
    return {
        "success": False,
        "message": "Knowledge index rebuild helper is temporarily unavailable. Please refresh the app.",
    }


def _vector_debug_status() -> dict:
    """Return detailed vector/RAG status for the sidebar debug panel."""
    service = _get_rag_service()
    debug_fn = getattr(service, "get_vector_debug_status", None)
    if callable(debug_fn):
        return debug_fn()
    return {
        "llm_active": False,
        "embedding_model_loaded": False,
        "faiss_index_exists": False,
        "indexed_documents": 0,
        "retrieval_mode": "fallback",
        "answer_path": "idle",
        "dependency_error": "Vector debug helper is unavailable. Refresh the page or restart Streamlit to load the latest chatbot service.",
        "last_error": "",
    }


def _get_chat_memory() -> InMemoryChatMessageHistory:
    """Return a short in-session conversation memory."""
    if "chatbot_memory" not in st.session_state:
        st.session_state["chatbot_memory"] = InMemoryChatMessageHistory()
    return st.session_state["chatbot_memory"]


def _trim_chat_memory(memory: InMemoryChatMessageHistory) -> None:
    """Keep only the last few messages in memory."""
    if len(memory.messages) > MAX_CHAT_MEMORY_MESSAGES:
        memory.messages = memory.messages[-MAX_CHAT_MEMORY_MESSAGES:]


def _get_chat_transcript() -> list[dict]:
    """Return UI transcript state for rendering."""
    if "chatbot_transcript" not in st.session_state:
        st.session_state["chatbot_transcript"] = []
    return st.session_state["chatbot_transcript"]


def _reset_chat() -> None:
    """Clear the current conversation."""
    st.session_state["chatbot_memory"] = InMemoryChatMessageHistory()
    st.session_state["chatbot_transcript"] = []
    st.session_state["chatbot_question"] = ""


def _display_supporting_data(df: pd.DataFrame) -> None:
    """Render supporting records beneath the answer."""
    if df.empty:
        return

    st.caption("Supporting data")
    st.dataframe(df, use_container_width=True, hide_index=True)


def _stream_text(text: str):
    """Yield text in small chunks for a light typing effect."""
    words = str(text or "").split()
    for word in words:
        yield word + " "
        time.sleep(0.01)


def _assistant_memory_text(
    payload: dict,
    supporting_records: list[dict] | None = None,
) -> str:
    """Create a compact assistant memory entry for follow-up questions."""
    parts = []

    answer = str(payload.get("answer", "") or "").strip()
    if answer:
        parts.append(f"answer: {answer}")

    explanation = str(payload.get("explanation", "") or "").strip()
    if explanation:
        parts.append(f"explanation: {explanation}")

    supporting_points = [
        str(point).strip()
        for point in payload.get("supporting_points", [])
        if str(point).strip()
    ]
    if supporting_points:
        parts.append("evidence: " + " | ".join(supporting_points[:2]))

    records = supporting_records or []
    if records:
        first_record = records[0]
        summary_bits = []
        for column in [
            "product_name",
            "store_name",
            "city",
            "store_id",
            "recommendation_type",
            "total_units_sold",
            "total_sales_value",
            "quantity_sold",
            "stock_level",
            "reason",
            "evidence",
            "source_agent",
            "agent_name",
        ]:
            value = str(first_record.get(column, "") or "").strip()
            if value:
                summary_bits.append(f"{column}: {value}")
        if summary_bits:
            parts.append("top_record: " + ", ".join(summary_bits[:4]))

    return "\n".join(parts).strip() or answer


def _render_assistant_message(
    payload: dict,
    supporting_records: list[dict] | None = None,
    sources: list[dict] | None = None,
    stream_answer: bool = False,
) -> None:
    """Render one assistant response in a clear, grounded format."""
    supporting_df = pd.DataFrame(supporting_records or [])
    answer_text = str(payload.get("answer", "") or "")
    if stream_answer and answer_text:
        st.write_stream(_stream_text(answer_text))
    else:
        st.write(answer_text)

    explanation = str(payload.get("explanation", "") or "").strip()
    if explanation:
        st.write(explanation)

    supporting_points = [
        point for point in payload.get("supporting_points", []) if str(point).strip()
    ]
    if supporting_points:
        st.caption("Key evidence")
        for point in supporting_points[:2]:
            st.write(f"- {point}")

    suggestions = [
        suggestion
        for suggestion in payload.get("suggestions", [])
        if str(suggestion).strip()
    ]
    if suggestions:
        st.caption(f"Suggestion: {suggestions[0]}")

    follow_up_question = str(payload.get("follow_up_question", "") or "").strip()
    if follow_up_question:
        st.caption(follow_up_question)

    _display_supporting_data(supporting_df)

    if sources:
        unique_sources = []
        seen = set()
        for source in sources:
            source_key = (
                source.get("dataset", ""),
                source.get("product_id", ""),
                source.get("store_id", ""),
                source.get("recommendation_type", ""),
            )
            if source_key in seen:
                continue
            seen.add(source_key)
            unique_sources.append(source)

        if unique_sources:
            with st.expander("Sources", expanded=False):
                st.dataframe(
                    pd.DataFrame(unique_sources),
                    use_container_width=True,
                    hide_index=True,
                )


st.set_page_config(
    page_title="Chatbot",
    page_icon="C",
    layout="wide",
)

apply_page_style()

render_page_header(
    "Chatbot",
    "A grounded retail assistant for inventory, sales, recommendations, and agent insights.",
)

with st.container(border=True):
    left_col, right_col = st.columns([3, 1])
    with left_col:
        st.markdown("**Retail Decision Assistant**")
        st.caption(
            "Ask naturally and I'll keep the answer short, practical, and grounded in your latest data."
        )
    with right_col:
        if st.button("Clear Chat", use_container_width=True):
            _reset_chat()

config_status = _chatbot_config_status()
debug_status = _vector_debug_status()
last_retrieval_mode = st.session_state.get("chatbot_retrieval_mode", "")
last_answer_path = st.session_state.get("chatbot_answer_path", "")
if last_retrieval_mode:
    debug_status["retrieval_mode"] = last_retrieval_mode
if last_answer_path:
    debug_status["answer_path"] = last_answer_path
rebuild_message = st.session_state.pop("chatbot_rebuild_message", "")
rebuild_success = st.session_state.pop("chatbot_rebuild_success", False)
if rebuild_message:
    if rebuild_success:
        st.success(rebuild_message)
    else:
        st.warning(rebuild_message)

if not config_status["configured"]:
    st.warning(str(config_status.get("status_message", "LLM is not configured. Add OPENROUTER_API_KEY in .env.")))
elif not bool(config_status.get("vector_rag_configured", False)):
    st.info(str(config_status.get("vector_rag_message", "Vector RAG is not configured yet.")))
else:
    if str(config_status.get("embedding_model", "")).strip():
        st.caption(
            f"Ready with `{config_status['chat_model']}` and `{config_status['embedding_model']}`."
        )
    else:
        st.caption(f"Ready with `{config_status['chat_model']}`.")
    if str(config_status.get("vector_rag_message", "")).strip():
        st.caption(str(config_status.get("vector_rag_message", "")))

with st.sidebar:
    st.header("Sample Questions")
    status_text = str(config_status.get("status_message", "") or "")
    if status_text:
        if str(config_status.get("configured", False)).lower() == "true" or config_status.get("configured", False):
            st.caption(status_text)
        else:
            st.warning(status_text)
    if st.button("Rebuild Knowledge Index", use_container_width=True):
        with st.spinner("Rebuilding the knowledge index..."):
            result = _rebuild_knowledge_index()
        st.cache_data.clear()
        st.cache_resource.clear()
        st.session_state["chatbot_rebuild_message"] = str(result.get("message", "Knowledge index rebuild finished."))
        st.session_state["chatbot_rebuild_success"] = bool(result.get("success", False))
        st.rerun()
    vector_message = str(config_status.get("vector_rag_message", "") or "").strip()
    if vector_message:
        st.caption(vector_message)
    with st.expander("Chatbot Debug", expanded=False):
        st.write(f"LLM active: {'yes' if debug_status.get('llm_active') else 'no'}")
        st.write(f"Embedding model loaded: {'yes' if debug_status.get('embedding_model_loaded') else 'no'}")
        st.write(f"Vector index exists: {'yes' if debug_status.get('faiss_index_exists') else 'no'}")
        st.write(f"Number of documents indexed: {int(debug_status.get('indexed_documents', 0) or 0)}")
        st.write(f"Retrieval mode: {debug_status.get('retrieval_mode', 'fallback')}")
        st.write(f"Answer path: {debug_status.get('answer_path', 'idle')}")
        embedding_model = str(config_status.get("embedding_model", "") or "").strip()
        if embedding_model:
            st.write(f"Embedding model: {embedding_model}")
        embedding_backend = str(debug_status.get("embedding_backend", "") or "").strip()
        if embedding_backend:
            st.write(f"Embedding backend: {embedding_backend}")
        dependency_error = str(debug_status.get("dependency_error", "") or "").strip()
        if dependency_error:
            st.caption(dependency_error)
        embedding_backend_error = str(debug_status.get("embedding_backend_error", "") or "").strip()
        if embedding_backend_error:
            st.caption(f"Embedding fallback detail: {embedding_backend_error}")
        last_error = str(debug_status.get("last_error", "") or "").strip()
        if last_error:
            st.caption(f"Last vector error: {last_error}")
    st.caption("Choose a starter prompt or ask your own question below.")
    st.divider()
    for sample_question in SAMPLE_QUESTIONS:
        if st.button(sample_question, use_container_width=True):
            st.session_state["chatbot_question"] = sample_question

question = st.chat_input(
    "Ask me anything about inventory, sales, or recommendations..."
)

if "chatbot_question" not in st.session_state:
    st.session_state["chatbot_question"] = ""

memory = _get_chat_memory()
transcript = _get_chat_transcript()

if question:
    st.session_state["chatbot_question"] = question

for message in transcript:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            _render_assistant_message(
                message.get("payload", {}),
                message.get("supporting_records", []),
                message.get("sources", []),
            )
        else:
            st.write(message.get("content", ""))

if not st.session_state["chatbot_question"] and not transcript:
    with st.chat_message("assistant"):
        _render_assistant_message(_welcome_payload())

    st.caption("Here are a few good places to start.")
    prompt_cols = st.columns(2)
    for index, sample_question in enumerate(SAMPLE_QUESTIONS[:6]):
        with prompt_cols[index % 2]:
            if st.button(sample_question, key=f"main_sample_{index}", use_container_width=True):
                st.session_state["chatbot_question"] = sample_question
                st.rerun()
elif st.session_state["chatbot_question"]:
    user_question = st.session_state["chatbot_question"]
    with st.chat_message("user"):
        st.write(user_question)

    transcript.append({"role": "user", "content": user_question})
    intent, answer_payload, supporting_df, sources = route_chatbot_request(
        user_question,
        chat_history=list(memory.messages),
    )
    st.session_state["chatbot_retrieval_mode"] = str(
        answer_payload.get("_debug_retrieval_mode", "fallback")
    )
    st.session_state["chatbot_answer_path"] = str(
        answer_payload.get("_debug_answer_path", "idle")
    )
    supporting_records = supporting_df.to_dict(orient="records")

    with st.chat_message("assistant"):
        _render_assistant_message(
            answer_payload,
            supporting_records,
            sources,
            stream_answer=True,
        )

    transcript.append(
        {
            "role": "assistant",
            "payload": answer_payload,
            "supporting_records": supporting_records,
            "sources": sources,
        }
    )
    memory.add_message(HumanMessage(content=user_question))
    memory.add_message(
        AIMessage(
            content=_assistant_memory_text(
                answer_payload,
                supporting_records,
            )
        )
    )
    _trim_chat_memory(memory)
    st.session_state["chatbot_question"] = ""

