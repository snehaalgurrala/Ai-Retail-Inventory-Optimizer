from pathlib import Path
import sys

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from backend.services.rag_service import (  # noqa: E402
    answer_question_with_rag,
    rag_is_configured,
)
from frontend.utils.page_helpers import apply_page_style, render_page_header  # noqa: E402


st.set_page_config(
    page_title="Chatbot",
    page_icon="C",
    layout="wide",
)

apply_page_style()


SAMPLE_QUESTIONS = [
    "Which items are low in stock?",
    "What are the dead stock items?",
    "Which store has excess stock?",
    "Which store needs transfer?",
    "Which products are high demand?",
    "Which suppliers are risky?",
    "Why is product P012 not selling?",
    "What is the best way to increase sales based on our trends?",
]


def _display_supporting_data(df: pd.DataFrame) -> None:
    """Render supporting records beneath the answer."""
    if df.empty:
        st.caption("No supporting records were retrieved for this answer.")
        return

    st.caption("Supporting project data")
    st.dataframe(df, use_container_width=True, hide_index=True)


render_page_header(
    "💬 Chatbot",
    "RAG-based project assistant grounded in product data, processed outputs, recommendations, and memory.",
)

if not rag_is_configured():
    st.warning(
        "RAG is not configured yet. Set Gemini or LLM environment variables to enable retrieval-based chatbot answers."
    )

with st.sidebar:
    st.header("Sample Questions")
    st.caption("Choose a starter prompt or ask your own question below.")
    st.divider()
    for sample_question in SAMPLE_QUESTIONS:
        if st.button(sample_question, use_container_width=True):
            st.session_state["chatbot_question"] = sample_question

question = st.chat_input(
    "Ask about products, inventory, sales trends, recommendations, or why an item is not selling"
)

if "chatbot_question" not in st.session_state:
    st.session_state["chatbot_question"] = ""

if question:
    st.session_state["chatbot_question"] = question

if not st.session_state["chatbot_question"]:
    st.subheader("💡 Ask a Data Question")
    st.write(
        "Questions are answered from retrieved project records only. "
        "The supporting table below each answer shows the data used."
    )
else:
    user_question = st.session_state["chatbot_question"]
    with st.chat_message("user"):
        st.write(user_question)

    answer, supporting_df, sources = answer_question_with_rag(user_question)
    with st.chat_message("assistant"):
        st.write(answer)
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
                st.caption("Retrieved sources")
                st.dataframe(
                    pd.DataFrame(unique_sources),
                    use_container_width=True,
                    hide_index=True,
                )
