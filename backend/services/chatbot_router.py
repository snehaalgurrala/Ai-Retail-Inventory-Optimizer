import pandas as pd

from backend.services.chatbot_analytics import try_answer_analytical_question
from backend.services.chatbot_intent import classify_user_intent
from backend.services.rag_service import answer_question_with_rag


def _looks_like_transfer_analytics(user_input: str) -> bool:
    text = str(user_input or "").strip().lower()
    transfer_terms = (
        "transfer",
        "stock balancing",
        "shortage",
        "excess stock",
        "surplus",
        "extra stock",
        "more stock",
        "surplus stock",
        "redistribution",
        "redistribute",
        "source store",
        "target store",
        "low stock branch",
        "stockout risk",
        "stock out risk",
        "exclusive",
        "only available",
        "alternative",
        "alternatives",
        "offer instead",
    )
    return any(term in text for term in transfer_terms)


def _looks_like_follow_up(user_input: str) -> bool:
    """Detect short follow-up prompts that should stay in the business flow."""
    text = str(user_input or "").strip().lower()
    if not text:
        return False

    follow_up_starts = (
        "why",
        "how",
        "what about",
        "explain",
        "and",
        "then",
    )
    if len(text.split()) > 3:
        return False
    return any(text == item or text.startswith(f"{item} ") for item in follow_up_starts)


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
            "answer": "Hello mate. How can I help you with inventory insights today?",
            "explanation": "I can help with stock movement, transfers, sales, supplier risk, and recommendations.",
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
    if _looks_like_transfer_analytics(user_input):
        analytics_route = try_answer_analytical_question(
            user_input,
            chat_history=chat_history,
        )
        if analytics_route.handled and analytics_route.intent != "follow_up":
            analytics_route.payload["_debug_answer_path"] = "analytics_first"
            analytics_route.payload["_debug_retrieval_mode"] = "direct_dataframe"
            print("[chatbot] detected_intent=analytical_query is_follow_up=False")
            return (
                "analytical_query",
                analytics_route.payload,
                analytics_route.supporting_df,
                analytics_route.sources,
            )

    classified = classify_user_intent(user_input)
    intent = classified.get("intent", "unclear")
    analytics_route = try_answer_analytical_question(
        user_input,
        chat_history=chat_history,
    )
    has_follow_up_context = _looks_like_follow_up(user_input) and _has_assistant_context(chat_history)

    if analytics_route.handled and analytics_route.intent != "follow_up":
        routed_intent = "analytical_query"
        is_follow_up = False
        analytics_route.payload["_debug_answer_path"] = "analytics_first"
        analytics_route.payload["_debug_retrieval_mode"] = "fallback"
        print(f"[chatbot] detected_intent={routed_intent} is_follow_up={is_follow_up}")
        return (
            routed_intent,
            analytics_route.payload,
            analytics_route.supporting_df,
            analytics_route.sources,
        )

    if intent == "greeting":
        routed_intent = "greeting"
        is_follow_up = False
        print(f"[chatbot] detected_intent={routed_intent} is_follow_up={is_follow_up}")
        return routed_intent, _build_intent_payload(routed_intent), pd.DataFrame(), []

    if analytics_route.handled and analytics_route.intent == "follow_up" and intent in {"follow_up", "unclear"}:
        routed_intent = "follow_up"
        is_follow_up = True
        analytics_route.payload["_debug_answer_path"] = "analytics_follow_up"
        analytics_route.payload["_debug_retrieval_mode"] = "memory"
        print(f"[chatbot] detected_intent={routed_intent} is_follow_up={is_follow_up}")
        return (
            routed_intent,
            analytics_route.payload,
            analytics_route.supporting_df,
            analytics_route.sources,
        )

    if intent in {"analytical_query", "business_query", "unclear"}:
        routed_intent = "business_query"
        is_follow_up = False
        print(f"[chatbot] detected_intent={routed_intent} is_follow_up={is_follow_up}")
        payload, supporting_df, sources = answer_question_with_rag(
            user_input,
            chat_history=chat_history,
        )
        return routed_intent, payload, supporting_df, sources

    if analytics_route.handled and analytics_route.intent == "follow_up":
        routed_intent = "follow_up"
        is_follow_up = True
        analytics_route.payload["_debug_answer_path"] = "analytics_follow_up"
        analytics_route.payload["_debug_retrieval_mode"] = "memory"
        print(f"[chatbot] detected_intent={routed_intent} is_follow_up={is_follow_up}")
        return (
            routed_intent,
            analytics_route.payload,
            analytics_route.supporting_df,
            analytics_route.sources,
        )

    if intent == "irrelevant":
        routed_intent = "irrelevant"
        is_follow_up = False
        print(f"[chatbot] detected_intent={routed_intent} is_follow_up={is_follow_up}")
        return routed_intent, _build_intent_payload(routed_intent), pd.DataFrame(), []

    if has_follow_up_context and analytics_route.handled:
        routed_intent = "follow_up"
        is_follow_up = True
        analytics_route.payload["_debug_answer_path"] = "analytics_follow_up"
        analytics_route.payload["_debug_retrieval_mode"] = "memory"
        print(f"[chatbot] detected_intent={routed_intent} is_follow_up={is_follow_up}")
        return (
            routed_intent,
            analytics_route.payload,
            analytics_route.supporting_df,
            analytics_route.sources,
        )

    routed_intent = "unclear"
    is_follow_up = False
    print(f"[chatbot] detected_intent={routed_intent} is_follow_up={is_follow_up}")
    return routed_intent, _build_intent_payload(routed_intent), pd.DataFrame(), []
