import pandas as pd

from backend.services.chatbot_intent import classify_user_intent
from backend.services.rag_service import answer_question_with_rag


def _looks_like_follow_up(user_input: str) -> bool:
    """Detect short follow-up prompts that should stay in the business flow."""
    text = str(user_input or "").strip().lower()
    if not text:
        return False

    follow_up_starts = (
        "why",
        "how",
        "what about",
        "and what about",
        "what else",
        "tell me more",
        "explain more",
        "explain this",
        "go deeper",
        "show me more",
    )
    return len(text.split()) <= 6 or text.startswith(follow_up_starts)


def _has_assistant_context(chat_history) -> bool:
    """Return True when the conversation already includes assistant context."""
    if not chat_history:
        return False

    for message in reversed(chat_history):
        content = str(getattr(message, "content", "") or "").strip()
        if not content:
            continue
        message_type = str(getattr(message, "type", "") or "").lower()
        if "ai" in message_type or "assistant" in message_type:
            return True
    return False


def _build_intent_payload(intent: str) -> dict:
    """Return a friendly assistant response for non-business intents."""
    if intent == "greeting":
        return {
            "answer": "Hi there. How can I help you today?",
            "explanation": "I can help with inventory, sales, supplier risk, and recommendations.",
            "suggestions": [],
            "follow_up_question": "Want me to check stock, sales, or recommendations?",
            "confidence": "high",
            "supporting_points": [],
            "cannot_answer": False,
        }

    if intent == "irrelevant":
        return {
            "answer": "I can help with inventory, sales, or business insights.",
            "explanation": "Try asking about stock levels, supplier risk, store performance, or recommendations.",
            "suggestions": [],
            "follow_up_question": "For example, do you want to check low stock or what should be reordered?",
            "confidence": "high",
            "supporting_points": [],
            "cannot_answer": False,
        }

    return {
        "answer": "Can you tell me a bit more?",
        "explanation": "I can help with inventory, sales, store comparison, supplier risk, or recommendations.",
        "suggestions": [],
        "follow_up_question": "Do you want inventory, sales, or recommendations?",
        "confidence": "medium",
        "supporting_points": [],
        "cannot_answer": False,
    }


def route_chatbot_request(
    user_input: str,
    chat_history=None,
) -> tuple[str, dict, pd.DataFrame, list[dict]]:
    """Route a chatbot request by intent and only run RAG for business queries."""
    classified = classify_user_intent(user_input)
    intent = classified.get("intent", "unclear")
    if (
        intent == "unclear"
        and _looks_like_follow_up(user_input)
        and _has_assistant_context(chat_history)
    ):
        intent = "business_query"

    if intent == "business_query":
        payload, supporting_df, sources = answer_question_with_rag(
            user_input,
            chat_history=chat_history,
        )
        return intent, payload, supporting_df, sources

    return intent, _build_intent_payload(intent), pd.DataFrame(), []
