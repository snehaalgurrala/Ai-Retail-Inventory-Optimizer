from datetime import datetime
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

from backend.mcp import config as mcp_config  # noqa: E402
from backend.services import llm_reasoner  # noqa: E402
from backend.services import rag_service  # noqa: E402
from backend.services.chatbot_router import route_chatbot_request  # noqa: E402
from frontend.utils.page_helpers import (  # noqa: E402
    apply_page_style,
    render_page_header,
    render_table,
)


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

# Compact suggestion chips shown directly above the chat input.
# Each entry is (short label shown on the chip, full question sent to the assistant).
SUGGESTION_CHIPS = [
    ("Low Stock", "Which products are low in stock?"),
    ("Top Products", "What are the top selling products?"),
    ("Demand Insights", "Show me customer demand insights and any high-demand orders."),
    ("Stockout Risk", "Which products are at risk of stockout?"),
    ("Supplier Risk", "Which supplier is risky?"),
    ("Transfer Opportunities", "Which store needs a stock transfer?"),
]


def _is_mcp_mode() -> bool:
    """True when the Oracle-grounded MCP engine is active (the default)."""
    try:
        return mcp_config.chatbot_engine() == "mcp"
    except Exception:
        return True


def _now_label() -> str:
    """Return a short, human-friendly timestamp for chat bubbles."""
    return datetime.now().strftime("%I:%M %p").lstrip("0")


# Page-scoped styling: compact suggestion chips and a tidy clear-chat button.
CHATBOT_CSS = """
<style>
/* Compact, pill-shaped suggestion chips (scoped to the chips container). */
.st-key-chatbot_chips div[data-testid="stHorizontalBlock"] { gap: 0.4rem; }
.st-key-chatbot_chips .stButton > button {
    padding: 0.22rem 0.8rem;
    min-height: 0;
    font-size: 0.78rem;
    font-weight: 650;
    border-radius: 999px;
    background: var(--airio-soft-blue);
    color: var(--airio-primary-navy);
    border: 1px solid var(--airio-border);
    box-shadow: none;
}
.st-key-chatbot_chips .stButton > button:hover {
    background: var(--airio-primary-navy);
    color: #ffffff;
    border-color: var(--airio-primary-navy);
    transform: none;
    box-shadow: 0 4px 10px rgba(24, 63, 95, 0.18);
}
/* Small, quiet clear-chat button in the top-right of the chat area. */
.st-key-chatbot_clear .stButton > button {
    padding: 0.22rem 0.7rem;
    min-height: 0;
    font-size: 0.78rem;
    font-weight: 650;
    border-radius: 10px;
    background: var(--airio-card);
    color: var(--airio-muted);
    border: 1px solid var(--airio-border);
    box-shadow: none;
}
.st-key-chatbot_clear .stButton > button:hover {
    background: var(--airio-soft-red);
    color: var(--airio-risk);
    border-color: var(--airio-soft-red);
    transform: none;
    box-shadow: none;
}
/* User bubble: right-aligned Bunzl-blue; assistant: left-aligned white card. */
div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    flex-direction: row-reverse;
    text-align: left;
}
.airio-chat-time {
    font-size: 0.7rem;
    color: rgba(10, 31, 51, 0.45);
    margin-top: 0.2rem;
}
</style>
"""


def _welcome_payload() -> dict:
    """Return a single, concise first-time greeting for new chat sessions."""
    return {
        "answer": (
            "Hi! I'm your Bunzl AI Assistant. Ask me about inventory, sales, "
            "customers, suppliers, recommendations, or stockout and supplier risks "
            "— every answer is grounded in your live data."
        ),
        "explanation": "",
        "suggestions": [],
        "follow_up_question": "",
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


def _vector_rag_environment() -> dict:
    """Return explicit Python/runtime diagnostics for vector RAG."""
    service = _get_rag_service()
    check_fn = getattr(service, "check_vector_rag_environment", None)
    if callable(check_fn):
        return check_fn()
    return {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "current_working_directory": str(PROJECT_ROOT),
        "faiss_available": False,
        "faiss_error": "Vector diagnostics helper is unavailable. Refresh the page or restart Streamlit.",
        "sentence_transformers_available": False,
        "langchain_community_available": False,
        "retrieval_mode": "fallback",
        "fallback_reason": "Vector diagnostics helper is unavailable.",
    }


def _is_raw_vector_error(message: str) -> bool:
    """Return True for dependency errors that should not be shown in the UI."""
    lowered = str(message or "").lower()
    return "faiss import failed" in lowered or "no module named 'faiss'" in lowered


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
    render_table(df, max_height=360)


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
                render_table(pd.DataFrame(unique_sources), max_height=320)


st.set_page_config(
    page_title="Chatbot",
    page_icon="C",
    layout="wide",
)

apply_page_style()
st.markdown(CHATBOT_CSS, unsafe_allow_html=True)

mcp_mode = _is_mcp_mode()

# --- Single clean page header (merges old copilot banner + assistant card) ----
render_page_header(
    "🤖 Bunzl AI Assistant",
    "Ask questions about inventory, sales, customers, suppliers, recommendations, "
    "stockout risks, customer demand insights, and demand trends.",
)

def _mcp_lightweight_status() -> tuple[dict, dict, dict]:
    """Config/debug status for MCP mode WITHOUT loading the legacy vector stack.

    The MCP engine never uses embeddings or FAISS, but the full status helpers
    (`chatbot_config_status` / `get_vector_debug_status` / `check_vector_rag_environment`)
    each initialize the local sentence-transformers embedding model on first call,
    adding ~10s to the first page render for values that are unused in MCP mode.
    Here we derive only what the page actually shows, from the LLM config alone.
    """
    llm_ok = bool(llm_reasoner.llm_is_configured())
    status_fn = getattr(llm_reasoner, "llm_status_message", None)
    status_message = (
        status_fn() if callable(status_fn)
        else ("LLM is configured." if llm_ok
              else "LLM is not configured. Add OPENROUTER_API_KEY in .env.")
    )
    config = {"configured": llm_ok, "status_message": status_message}
    debug = {"llm_active": llm_ok, "retrieval_mode": "mcp_oracle", "answer_path": "mcp"}
    return config, debug, {}


# --- Status / diagnostics reads (kept; surfaced only where genuinely useful) --
# In MCP mode skip the embedding/FAISS probes entirely (see helper above): they
# load the local sentence-transformers model on first render and are unused here.
if mcp_mode:
    config_status, debug_status, vector_env = _mcp_lightweight_status()
else:
    config_status = _chatbot_config_status()
    debug_status = _vector_debug_status()
    vector_env = _vector_rag_environment()
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

# Only the one status that blocks usage is surfaced on the page; everything else
# lives in the sidebar Admin/Developer section to keep the chat the focus.
if not config_status["configured"]:
    st.warning(
        str(config_status.get(
            "status_message",
            "LLM is not configured. Add OPENROUTER_API_KEY in .env.",
        ))
    )
elif not mcp_mode and not bool(config_status.get("vector_rag_configured", False)):
    vector_message = str(config_status.get("vector_rag_message", "") or "").strip()
    if not _is_raw_vector_error(vector_message):
        st.info(vector_message or "Knowledge search is using fallback retrieval.")

# --- Sidebar: AI Assistant + Suggested Questions (+ collapsed admin tools) -----
with st.sidebar:
    st.header("Bunzl AI Assistant")
    st.caption("Ask about inventory, sales, customers, suppliers, recommendations, and risks.")

    st.subheader("Suggested Questions")
    for sample_question in SAMPLE_QUESTIONS:
        if st.button(sample_question, use_container_width=True, key=f"side_{sample_question}"):
            st.session_state["chatbot_question"] = sample_question

    with st.expander("Admin / Developer", expanded=False):
        # Knowledge-index rebuild and vector diagnostics only apply to the legacy
        # RAG engine; they are hidden entirely when the MCP engine is active.
        if not mcp_mode:
            if st.button("Rebuild Knowledge Index", use_container_width=True):
                with st.spinner("Rebuilding the knowledge index..."):
                    result = _rebuild_knowledge_index()
                st.cache_data.clear()
                st.cache_resource.clear()
                st.session_state["chatbot_rebuild_message"] = str(
                    result.get("message", "Knowledge index rebuild finished.")
                )
                st.session_state["chatbot_rebuild_success"] = bool(result.get("success", False))
                st.rerun()

        st.markdown("**Chatbot Debug**")
        st.write(f"Engine: {'mcp' if mcp_mode else 'legacy'}")
        st.write(f"LLM active: {'yes' if debug_status.get('llm_active') else 'no'}")
        st.write(f"Retrieval mode: {debug_status.get('retrieval_mode', 'fallback')}")
        st.write(f"Answer path: {debug_status.get('answer_path', 'idle')}")
        if not mcp_mode:
            st.write(f"Embedding model loaded: {'yes' if debug_status.get('embedding_model_loaded') else 'no'}")
            st.write(f"Vector index exists: {'yes' if debug_status.get('faiss_index_exists') else 'no'}")
            st.write(f"Number of documents indexed: {int(debug_status.get('indexed_documents', 0) or 0)}")
            embedding_model = str(config_status.get("embedding_model", "") or "").strip()
            if embedding_model:
                st.write(f"Embedding model: {embedding_model}")
            embedding_backend = str(debug_status.get("embedding_backend", "") or "").strip()
            if embedding_backend:
                st.write(f"Embedding backend: {embedding_backend}")
            embedding_backend_error = str(debug_status.get("embedding_backend_error", "") or "").strip()
            if embedding_backend_error:
                st.caption(f"Embedding fallback detail: {embedding_backend_error}")
        last_error = str(debug_status.get("last_error", "") or "").strip()
        if last_error and not _is_raw_vector_error(last_error):
            st.caption(f"Last vector error: {last_error}")

        # Vector RAG diagnostics are legacy-only; hidden in MCP mode.
        if not mcp_mode:
            st.divider()
            st.markdown("**Vector RAG Diagnostics**")
            st.write(f"Python executable: `{vector_env.get('python_executable', '')}`")
            st.write(f"Python version: `{vector_env.get('python_version', '')}`")
            st.write(f"Current working directory: `{vector_env.get('current_working_directory', '')}`")
            st.write(
                "sentence_transformers import works: "
                f"{'yes' if vector_env.get('sentence_transformers_available') else 'no'}"
            )
            sentence_error = str(vector_env.get("sentence_transformers_error", "") or "").strip()
            if sentence_error:
                st.error(f"sentence_transformers import failed: {sentence_error}")
            st.write(
                "langchain_community import works: "
                f"{'yes' if vector_env.get('langchain_community_available') else 'no'}"
            )
            langchain_error = str(vector_env.get("langchain_community_error", "") or "").strip()
            if langchain_error:
                st.error(f"langchain_community import failed: {langchain_error}")
            st.write(f"Embedding model: `{vector_env.get('embedding_model', '')}`")
            st.write(f"Embedding model loaded: {'yes' if vector_env.get('embedding_model_loaded') else 'no'}")
            embedding_error = str(vector_env.get("embedding_model_error", "") or "").strip()
            if embedding_error:
                st.error(f"Embedding model failed to load: {embedding_error}")
            st.write(f"FAISS index exists: {'yes' if vector_env.get('faiss_index_exists') else 'no'}")
            st.write(f"Can build FAISS index: {'yes' if vector_env.get('can_build_index') else 'no'}")
            st.write(f"Retrieval mode: `{vector_env.get('retrieval_mode', 'fallback')}`")
            fallback_reason = str(vector_env.get("fallback_reason", "") or "").strip()
            if fallback_reason and not _is_raw_vector_error(fallback_reason):
                st.warning(f"Fallback reason: {fallback_reason}")
            if (
                not vector_env.get("faiss_available")
                or not vector_env.get("sentence_transformers_available")
                or not vector_env.get("langchain_community_available")
            ):
                st.code(
                    "\n".join(
                        [
                            "python -m pip install faiss-cpu",
                            "python -m pip install langchain-community sentence-transformers",
                            "python -m streamlit run frontend/app.py",
                        ]
                    ),
                    language="powershell",
                )

# --- State ---------------------------------------------------------------------
if "chatbot_question" not in st.session_state:
    st.session_state["chatbot_question"] = ""

memory = _get_chat_memory()
transcript = _get_chat_transcript()

# Chat input is pinned to the bottom of the page by Streamlit regardless of where
# it is called, so we read it early and render the chat window + chips above it.
question = st.chat_input(
    "Ask me anything about inventory, sales, customers, or recommendations..."
)
if question:
    st.session_state["chatbot_question"] = question

# --- Clear control: small, quiet button at the top-right of the chat area ------
head_left, head_right = st.columns([6, 1])
with head_left:
    st.markdown("##### Conversation")
with head_right:
    with st.container(key="chatbot_clear"):
        if st.button("🗑 Clear", key="clear_chat", use_container_width=True):
            _reset_chat()
            st.rerun()

# --- Large chat window (scrolls internally so it owns most of the screen) -------
with st.container(height=560):
    if not transcript and not st.session_state["chatbot_question"]:
        with st.chat_message("assistant"):
            _render_assistant_message(_welcome_payload())

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
            timestamp = str(message.get("timestamp", "") or "")
            if timestamp:
                st.markdown(f"<div class='airio-chat-time'>{timestamp}</div>", unsafe_allow_html=True)

    if st.session_state["chatbot_question"]:
        user_question = st.session_state["chatbot_question"]
        asked_at = _now_label()
        with st.chat_message("user"):
            st.write(user_question)
            st.markdown(f"<div class='airio-chat-time'>{asked_at}</div>", unsafe_allow_html=True)

        transcript.append({"role": "user", "content": user_question, "timestamp": asked_at})
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

        answered_at = _now_label()
        with st.chat_message("assistant"):
            _render_assistant_message(
                answer_payload,
                supporting_records,
                sources,
                stream_answer=True,
            )
            st.markdown(f"<div class='airio-chat-time'>{answered_at}</div>", unsafe_allow_html=True)

        transcript.append(
            {
                "role": "assistant",
                "payload": answer_payload,
                "supporting_records": supporting_records,
                "sources": sources,
                "timestamp": answered_at,
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

# --- Compact suggestion chips, placed directly above the chat input ------------
with st.container(key="chatbot_chips"):
    st.caption("Try one of these:")
    chip_cols = st.columns(len(SUGGESTION_CHIPS))
    for index, (chip_label, chip_question) in enumerate(SUGGESTION_CHIPS):
        with chip_cols[index]:
            if st.button(chip_label, key=f"chip_{index}", use_container_width=True):
                st.session_state["chatbot_question"] = chip_question
                st.rerun()

