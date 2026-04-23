import json
import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError


load_dotenv()


DEFAULT_LLM_TIMEOUT_SECONDS = 45
DEFAULT_LLM_BATCH_SIZE = 20


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


def get_llm_settings() -> dict[str, Any]:
    """Load provider settings for an OpenAI-compatible chat model."""
    return {
        "api_key": os.getenv("LLM_API_KEY", "").strip(),
        "model": os.getenv("LLM_MODEL", "").strip(),
        "base_url": os.getenv("LLM_BASE_URL", "").strip(),
        "timeout": int(
            os.getenv("LLM_TIMEOUT_SECONDS", DEFAULT_LLM_TIMEOUT_SECONDS)
        ),
        "batch_size": int(os.getenv("LLM_BATCH_SIZE", DEFAULT_LLM_BATCH_SIZE)),
    }


def llm_is_configured() -> bool:
    """Return True when the app has enough settings to call the LLM."""
    settings = get_llm_settings()
    return bool(settings["api_key"] and settings["model"])


def _build_client() -> OpenAI:
    """Create an OpenAI-compatible client from environment settings."""
    settings = get_llm_settings()
    client_kwargs: dict[str, Any] = {
        "api_key": settings["api_key"],
        "timeout": settings["timeout"],
    }
    if settings["base_url"]:
        client_kwargs["base_url"] = settings["base_url"]
    return OpenAI(**client_kwargs)


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
