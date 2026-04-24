import json

from google.genai import Client
from google.genai import types as genai_types
from openai import OpenAI
from pydantic import BaseModel

from backend.services.llm_reasoner import get_llm_settings, llm_is_configured


ALLOWED_INTENTS = {"greeting", "business_query", "irrelevant", "unclear"}


class IntentClassification(BaseModel):
    intent: str = "unclear"


def _fallback_intent(user_input: str) -> dict[str, str]:
    """Return a safe lightweight fallback when the LLM is unavailable."""
    text = str(user_input or "").strip().lower()
    if not text:
        return {"intent": "unclear"}

    greeting_words = {"hi", "hello", "hai", "hey", "hii", "good morning", "good evening"}
    business_words = {
        "inventory",
        "stock",
        "sales",
        "supplier",
        "suppliers",
        "recommendation",
        "recommendations",
        "store",
        "stores",
        "product",
        "products",
        "reorder",
        "transfer",
        "discount",
    }
    irrelevant_words = {
        "movie",
        "movies",
        "weather",
        "song",
        "songs",
        "cricket",
        "football",
        "actor",
        "actress",
        "joke",
    }

    if text in greeting_words or any(text.startswith(word) for word in greeting_words):
        return {"intent": "greeting"}
    if any(word in text for word in business_words):
        return {"intent": "business_query"}
    if any(word in text for word in irrelevant_words):
        return {"intent": "irrelevant"}
    return {"intent": "unclear"}


def _normalize_intent(intent: str) -> dict[str, str]:
    """Normalize LLM output to the allowed set."""
    cleaned = str(intent or "").strip().lower()
    if cleaned not in ALLOWED_INTENTS:
        cleaned = "unclear"
    return {"intent": cleaned}


def classify_user_intent(user_input: str) -> dict[str, str]:
    """Classify user input into greeting, business_query, irrelevant, or unclear."""
    text = str(user_input or "").strip()
    if not text:
        return {"intent": "unclear"}

    if not llm_is_configured():
        return _fallback_intent(text)

    settings = get_llm_settings()
    system_prompt = (
        "Classify the user message into exactly one intent label. "
        "Allowed labels: greeting, business_query, irrelevant, unclear. "
        "Use business_query for retail inventory, stock, sales, suppliers, stores, products, or recommendations. "
        "Use irrelevant for unrelated topics like entertainment, weather, or random chat. "
        "Use unclear when the message is too ambiguous to classify confidently. "
        "Return valid JSON only."
    )
    user_prompt = json.dumps(
        {
            "message": text,
            "required_output_schema": {
                "intent": "greeting | business_query | irrelevant | unclear"
            },
        },
        ensure_ascii=True,
    )

    try:
        if settings["provider"] == "gemini":
            client = Client(api_key=settings["api_key"])
            response = client.models.generate_content(
                model=settings["model"],
                contents=user_prompt,
                config=genai_types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.0,
                    response_mime_type="application/json",
                    response_schema=IntentClassification,
                ),
            )
            parsed = json.loads(getattr(response, "text", "") or "{}")
            result = IntentClassification.model_validate(parsed)
            return _normalize_intent(result.intent)

        client_kwargs = {"api_key": settings["api_key"], "timeout": settings["timeout"]}
        if settings["base_url"]:
            client_kwargs["base_url"] = settings["base_url"]
        client = OpenAI(**client_kwargs)
        response = client.chat.completions.create(
            model=settings["model"],
            temperature=0.0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response.choices[0].message.content or "{}"
        parsed = json.loads(content)
        result = IntentClassification.model_validate(parsed)
        return _normalize_intent(result.intent)
    except Exception:
        return _fallback_intent(text)
