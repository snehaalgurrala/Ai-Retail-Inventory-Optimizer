import json
import os
from typing import Any

from dotenv import load_dotenv
from google.genai import Client
from google.genai import types as genai_types
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError


load_dotenv()


DEFAULT_LLM_TIMEOUT_SECONDS = 45
DEFAULT_LLM_BATCH_SIZE = 20
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_MODEL = "openrouter/free"
DEFAULT_LOCAL_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class LLMDecision(BaseModel):
    candidate_id: str
    selected_strategy: str
    keep_recommendation: bool = True
    priority: str = "medium"
    action: str = ""
    reason: str = ""
    evidence: str = ""
    confidence: str = "medium"


class LLMDecisionBatch(BaseModel):
    decisions: list[LLMDecision] = Field(default_factory=list)


class LLMOrchestrationSummary(BaseModel):
    overall_strategy: str = ""
    priority_focus: list[str] = Field(default_factory=list)
    reasoning: str = ""


class LLMToolSelection(BaseModel):
    tool_names: list[str] = Field(default_factory=list)


class LLMLearningInsight(BaseModel):
    recommendation_type: str = ""
    product_id: str = ""
    product_name: str = ""
    store_id: str = ""
    action_taken: str = ""
    outcome_status: str = ""
    learning_label: str = ""
    insight: str = ""
    recommended_adjustment: str = ""


class LLMLearningInsightBatch(BaseModel):
    insights: list[LLMLearningInsight] = Field(default_factory=list)


def get_llm_settings() -> dict[str, Any]:
    """Load shared provider settings for Gemini or OpenRouter/OpenAI-compatible chat."""
    gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip()
    gemini_model = os.getenv("GEMINI_MODEL", "").strip()
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    if not provider:
        provider = "gemini" if gemini_api_key else "openrouter"

    if provider == "gemini":
        api_key = (
            gemini_api_key
            or os.getenv("LLM_API_KEY", "").strip()
            or os.getenv("OPENROUTER_API_KEY", "").strip()
        )
        model = gemini_model or os.getenv("LLM_MODEL", "").strip()
        embedding_model = (
            os.getenv("GEMINI_EMBEDDING_MODEL", "").strip()
            or os.getenv("EMBEDDING_MODEL", "").strip()
            or DEFAULT_LOCAL_EMBEDDING_MODEL
        )
        base_url = os.getenv("LLM_BASE_URL", "").strip()
    else:
        provider = "openrouter"
        api_key = (
            os.getenv("OPENROUTER_API_KEY", "").strip()
            or os.getenv("LLM_API_KEY", "").strip()
        )
        model = (
            os.getenv("OPENROUTER_MODEL", "").strip()
            or os.getenv("LLM_MODEL", "").strip()
            or DEFAULT_OPENROUTER_MODEL
        )
        embedding_model = (
            os.getenv("OPENROUTER_EMBEDDING_MODEL", "").strip()
            or os.getenv("EMBEDDING_MODEL", "").strip()
            or DEFAULT_LOCAL_EMBEDDING_MODEL
        )
        base_url = (
            os.getenv("OPENROUTER_BASE_URL", "").strip()
            or os.getenv("LLM_BASE_URL", "").strip()
            or DEFAULT_OPENROUTER_BASE_URL
        )

    return {
        "provider": provider,
        "api_key": api_key,
        "model": model,
        "embedding_model": embedding_model,
        "base_url": base_url,
        "timeout": int(
            os.getenv("LLM_TIMEOUT_SECONDS", DEFAULT_LLM_TIMEOUT_SECONDS)
        ),
        "batch_size": int(os.getenv("LLM_BATCH_SIZE", DEFAULT_LLM_BATCH_SIZE)),
    }


def llm_is_configured() -> bool:
    """Return True when the app has enough shared settings to call the chat LLM."""
    settings = get_llm_settings()
    return bool(settings["api_key"] and settings["model"])


def llm_status_message() -> str:
    """Return a short debug-friendly status message for UI surfaces."""
    settings = get_llm_settings()
    if not settings["api_key"]:
        return "LLM is not configured. Add OPENROUTER_API_KEY in .env."
    if settings["provider"] == "openrouter":
        return "LLM: OpenRouter active"
    if settings["provider"] == "gemini":
        return "LLM: Gemini active"
    return f"LLM: {settings['provider']} active"


def _build_client() -> OpenAI:
    """Create an OpenAI-compatible client from shared environment settings."""
    settings = get_llm_settings()
    client_kwargs: dict[str, Any] = {
        "api_key": settings["api_key"],
        "timeout": settings["timeout"],
    }
    if settings["base_url"]:
        client_kwargs["base_url"] = settings["base_url"]
    return OpenAI(**client_kwargs)


def _build_gemini_client() -> Client:
    """Create a Gemini client from environment settings."""
    settings = get_llm_settings()
    return Client(api_key=settings["api_key"])


def _chunk_records(records: list[dict], batch_size: int) -> list[list[dict]]:
    """Split candidate payloads into smaller batches for the LLM."""
    if batch_size <= 0:
        batch_size = DEFAULT_LLM_BATCH_SIZE
    return [
        records[index:index + batch_size]
        for index in range(0, len(records), batch_size)
    ]


def _extract_json_text(content: str) -> str:
    """Strip markdown fences if the model wrapped the JSON output."""
    text = str(content or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _chat_json(system_prompt: str, user_prompt: str) -> str:
    """Call the configured LLM and return the raw text response."""
    settings = get_llm_settings()
    if settings["provider"] == "gemini":
        client = _build_gemini_client()
        response = client.models.generate_content(
            model=settings["model"],
            contents=user_prompt,
            config=genai_types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.1,
                response_mime_type="application/json",
            ),
        )
        return getattr(response, "text", "") or ""

    client = _build_client()
    response = client.chat.completions.create(
        model=settings["model"],
        temperature=0.1,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content or ""


def humanize_analytics_payload(
    question: str,
    payload: dict[str, Any],
    supporting_points: list[str] | None = None,
) -> dict[str, Any]:
    """Use the shared chat LLM to make analytical answers sound more human."""
    if not llm_is_configured():
        return payload

    system_prompt = (
        "You are a retail business analyst assistant. "
        "Rewrite the given structured answer so it sounds natural, concise, and helpful. "
        "Keep it factual and grounded in the supplied answer content only. "
        "Do not invent new numbers, stores, products, or conclusions. "
        "Avoid robotic template wording. "
        "Return valid JSON only with keys: answer, explanation, suggestions, follow_up_question, confidence, supporting_points, cannot_answer."
    )
    user_prompt = json.dumps(
        {
            "question": question,
            "current_payload": payload,
            "supporting_points": supporting_points or payload.get("supporting_points", []),
            "style_rules": [
                "Direct answer first",
                "One short explanation",
                "At most one practical suggestion",
                "Optional short follow-up question",
            ],
        },
        ensure_ascii=True,
    )

    try:
        raw_content = _chat_json(system_prompt, user_prompt)
        parsed = json.loads(_extract_json_text(raw_content))
        if isinstance(parsed, dict):
            merged = dict(payload)
            merged.update(parsed)
            return merged
    except Exception:
        return payload
    return payload


def humanize_chatbot_payload(
    question: str,
    payload: dict[str, Any],
    supporting_points: list[str] | None = None,
    answer_style: str = "chatbot",
) -> dict[str, Any]:
    """Make structured chatbot answers sound more natural without changing facts."""
    if not llm_is_configured():
        return payload

    system_prompt = (
        "You are a retail business analyst assistant. "
        "Rewrite the supplied structured answer so it sounds friendly, direct, and human. "
        "Keep every fact grounded in the supplied payload only. "
        "Do not add new numbers, products, stores, or conclusions. "
        "Do not use robotic phrases, retrieval language, or system/debug wording. "
        "Avoid phrases like 'retrieved records', 'strongest evidence came from', "
        "'No supporting records were retrieved', or 'based only on latest records'. "
        "Return valid JSON only with keys: answer, explanation, suggestions, "
        "follow_up_question, confidence, supporting_points, cannot_answer."
    )
    user_prompt = json.dumps(
        {
            "question": question,
            "current_payload": payload,
            "supporting_points": supporting_points or payload.get("supporting_points", []),
            "style_rules": [
                "Direct answer first",
                "One short reason from the data",
                "At most one practical suggestion",
                "Optional short follow-up question",
                f"Answer style: {answer_style}",
            ],
        },
        ensure_ascii=True,
    )

    try:
        raw_content = _chat_json(system_prompt, user_prompt)
        parsed = json.loads(_extract_json_text(raw_content))
        if isinstance(parsed, dict):
            merged = dict(payload)
            merged.update(parsed)
            return merged
    except Exception:
        return payload
    return payload


def reason_over_recommendations(
    agent_name: str,
    agent_goal: str,
    candidates: list[dict],
    allowed_strategies: list[str],
    shared_context: dict | None = None,
) -> dict[str, dict]:
    """Ask the LLM to review candidate actions and return structured decisions."""
    if not llm_is_configured() or not candidates:
        return {}

    settings = get_llm_settings()
    decisions: dict[str, dict] = {}
    batches = _chunk_records(candidates, settings["batch_size"])

    system_prompt = (
        "You are a retail inventory decision-support agent. "
        "Use only the structured data given in the prompt. "
        "Do not invent products, stores, sales values, or thresholds. "
        "Use memory from past recommendations, decisions, and outcomes only when it "
        "is explicitly present in the prompt payload. "
        "Keep your output as valid JSON only. "
        "When data is missing, say that clearly in the reason or evidence. "
        "Only choose strategies from the allowed list."
    )

    for batch in batches:
        user_prompt = json.dumps(
            {
                "agent_name": agent_name,
                "agent_goal": agent_goal,
                "allowed_strategies": allowed_strategies,
                "shared_context": shared_context or {},
                "expected_output_schema": {
                    "decisions": [
                        {
                            "candidate_id": "string",
                            "selected_strategy": "one allowed strategy",
                            "keep_recommendation": True,
                            "priority": "low|medium|high",
                            "action": "short action sentence grounded in the data",
                            "reason": "short reason grounded in the data",
                            "evidence": "brief evidence string using given facts only",
                            "confidence": "low|medium|high",
                        }
                    ]
                },
                "candidates": batch,
            },
            ensure_ascii=True,
        )

        try:
            raw_content = _chat_json(system_prompt, user_prompt)
            parsed = json.loads(_extract_json_text(raw_content))
            batch_response = LLMDecisionBatch.model_validate(parsed)
        except (
            json.JSONDecodeError,
            ValidationError,
            IndexError,
            KeyError,
            TypeError,
            ValueError,
        ):
            continue
        except Exception:
            continue

        for decision in batch_response.decisions:
            if decision.selected_strategy not in allowed_strategies:
                continue
            decisions[decision.candidate_id] = decision.model_dump()

    return decisions


def summarize_orchestration(summary_payload: dict) -> dict:
    """Ask the LLM for a compact orchestration summary."""
    if not llm_is_configured():
        return {}

    system_prompt = (
        "You are an orchestration agent for retail recommendations. "
        "Summarize only the data given to you. "
        "Use memory history only when it appears in the payload. "
        "Return valid JSON only. "
        "Do not invent counts, trends, or recommendations."
    )
    user_prompt = json.dumps(
        {
            "task": (
                "Summarize the overall recommendation strategy and the main areas "
                "that should be addressed first."
            ),
            "expected_output_schema": {
                "overall_strategy": "short summary",
                "priority_focus": ["short focus item"],
                "reasoning": "one grounded paragraph",
            },
            "summary_payload": summary_payload,
        },
        ensure_ascii=True,
    )

    try:
        raw_content = _chat_json(system_prompt, user_prompt)
        parsed = json.loads(_extract_json_text(raw_content))
        return LLMOrchestrationSummary.model_validate(parsed).model_dump()
    except Exception:
        return {}


def select_tools_for_agent(
    agent_name: str,
    agent_goal: str,
    available_tools: list[dict[str, str]],
    context: dict | None = None,
    default_tools: list[str] | None = None,
) -> list[str]:
    """Ask the LLM which tools the agent should use for the current task."""
    default_tools = default_tools or []
    if not llm_is_configured():
        return default_tools

    system_prompt = (
        "You are selecting tools for a retail inventory agent. "
        "Use only the tools listed in the prompt. "
        "Return valid JSON only. "
        "Choose the smallest set of tools that can complete the agent goal."
    )
    user_prompt = json.dumps(
        {
            "agent_name": agent_name,
            "agent_goal": agent_goal,
            "available_tools": available_tools,
            "context": context or {},
            "expected_output_schema": {
                "tool_names": ["tool_name"]
            },
        },
        ensure_ascii=True,
    )

    try:
        raw_content = _chat_json(system_prompt, user_prompt)
        parsed = json.loads(_extract_json_text(raw_content))
        selection = LLMToolSelection.model_validate(parsed)
        valid_tool_names = {tool_data["name"] for tool_data in available_tools}
        chosen_tools = [
            tool_name
            for tool_name in selection.tool_names
            if tool_name in valid_tool_names
        ]
        return chosen_tools or default_tools
    except Exception:
        return default_tools


def summarize_learning_feedback(feedback_rows: list[dict]) -> list[dict]:
    """Ask the LLM to turn past outcomes into compact learning insights."""
    if not llm_is_configured() or not feedback_rows:
        return []

    system_prompt = (
        "You are analyzing retail recommendation outcomes. "
        "Use only the feedback rows provided. "
        "Return valid JSON only. "
        "Summarize what worked, what failed, and what future adjustment should be made. "
        "Do not invent results beyond the feedback rows."
    )
    user_prompt = json.dumps(
        {
            "task": "Create short learning insights from past recommendation feedback.",
            "expected_output_schema": {
                "insights": [
                    {
                        "recommendation_type": "string",
                        "product_id": "string",
                        "product_name": "string",
                        "store_id": "string",
                        "action_taken": "string",
                        "outcome_status": "string",
                        "learning_label": "worked|failed|neutral",
                        "insight": "short grounded insight",
                        "recommended_adjustment": "short next-step guidance",
                    }
                ]
            },
            "feedback_rows": feedback_rows,
        },
        ensure_ascii=True,
    )

    try:
        raw_content = _chat_json(system_prompt, user_prompt)
        parsed = json.loads(_extract_json_text(raw_content))
        batch = LLMLearningInsightBatch.model_validate(parsed)
        return [insight.model_dump() for insight in batch.insights]
    except Exception:
        return []
