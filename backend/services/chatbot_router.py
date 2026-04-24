import pandas as pd

from backend.services.chatbot_intent import classify_user_intent
from backend.services.rag_service import answer_question_with_rag


def _build_intent_payload(intent: str) -> dict:
    """Return a friendly assistant response for non-business intents."""
    if intent == "greeting":
        return {
            "answer": "Hi! How can I help you with your inventory today?",
            "explanation": "I can help with stock, sales, suppliers, recommendations, and store-level business insights.",
            "suggestions": [
                "Ask which products are low in stock.",
                "Ask what should be reordered now.",
                "Ask which supplier is risky.",
            ],
            "confidence": "high",
            "supporting_points": [],
            "cannot_answer": False,
        }

    if intent == "irrelevant":
        return {
            "answer": "Sorry, I can only help with inventory, sales, and business insights.",
            "explanation": "Try asking about stock levels, store performance, supplier risk, or recommendations.",
            "suggestions": [
                "Which products are low in stock?",
                "What should we reorder now?",
                "Which store needs transfer?",
            ],
            "confidence": "high",
            "supporting_points": [],
            "cannot_answer": False,
        }

    return {
        "answer": "Can you clarify what you mean?",
        "explanation": "Do you want help with inventory, sales, suppliers, store comparison, or recommendations?",
        "suggestions": [
            "Ask about inventory levels.",
            "Ask about sales trends.",
            "Ask about recommendations or supplier risk.",
        ],
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

    if intent == "business_query":
        payload, supporting_df, sources = answer_question_with_rag(
            user_input,
            chat_history=chat_history,
        )
        return intent, payload, supporting_df, sources

    return intent, _build_intent_payload(intent), pd.DataFrame(), []
