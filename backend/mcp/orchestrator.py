"""Chatbot-facing MCP orchestrator.

Runs the tool-calling loop: the LLM *chooses* an allow-listed tool and arguments
(as JSON — never SQL), the tool executes against Oracle through MCP, and the LLM
*explains* the returned data. Tools are executed via the stdio MCP session when
available, otherwise in-process through the same registry (identical results,
still Oracle-grounded).

Returns the same ``(payload, supporting_df, sources)`` contract the chatbot page
already renders, so the Streamlit UI is unchanged.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable

import pandas as pd

from backend.mcp import config, registry
from backend.services import llm_reasoner


_LLM_RETRIES = 3


logger = logging.getLogger(__name__)

ToolExecutor = Callable[[str, dict], dict]


# ---------------------------------------------------------------------------
# Tool execution backend selection (stdio MCP, with in-process fallback)
# ---------------------------------------------------------------------------
def _resolve_executor() -> tuple[ToolExecutor, str]:
    """Return (executor, mode). Prefers the stdio MCP session."""
    try:
        from backend.mcp.client import get_client

        client = get_client()
        return (lambda name, args: client.call_tool(name, args)), "stdio_mcp"
    except Exception as error:  # noqa: BLE001
        if not config.use_in_process_fallback():
            raise
        logger.warning("MCP stdio unavailable (%s); using in-process tools.", error)
        return (lambda name, args: registry.execute(name, args)), "in_process_mcp"


# ---------------------------------------------------------------------------
# LLM helpers (provider-agnostic JSON output)
# ---------------------------------------------------------------------------
def _strip_fences(text: str) -> str:
    text = str(text or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _parse_json(text: str) -> dict | None:
    """Parse a JSON object, tolerating models that wrap it in prose."""
    cleaned = _strip_fences(text)
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else None
    except (ValueError, TypeError):
        pass
    # Salvage the first balanced {...} block from a noisy response.
    start = cleaned.find("{")
    while start != -1:
        depth = 0
        for index in range(start, len(cleaned)):
            char = cleaned[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = cleaned[start:index + 1]
                    try:
                        parsed = json.loads(candidate)
                        if isinstance(parsed, dict):
                            return parsed
                    except (ValueError, TypeError):
                        break
        start = cleaned.find("{", start + 1)
    return None


def _llm_json(system_prompt: str, user_prompt: str) -> dict | None:
    """Call the LLM and return parsed JSON, retrying transient failures.

    Free-tier providers rate-limit bursty sequential calls; a single transient
    failure must not abort the whole tool-calling loop.
    """
    last_error = ""
    for attempt in range(_LLM_RETRIES):
        result = _llm_json_once(system_prompt, user_prompt)
        if result is not None:
            return result
        last_error = "empty/unparseable or call error"
        time.sleep(0.8 * (attempt + 1))
    logger.warning("LLM returned no usable JSON after %d attempts (%s).", _LLM_RETRIES, last_error)
    return None


def _llm_json_once(system_prompt: str, user_prompt: str) -> dict | None:
    """One LLM call returning parsed JSON (or None on failure)."""
    settings = llm_reasoner.get_llm_settings()
    try:
        if settings["provider"] == "gemini":
            from google.genai import Client
            from google.genai import types as genai_types

            client = Client(api_key=settings["api_key"])
            response = client.models.generate_content(
                model=settings["model"],
                contents=user_prompt,
                config=genai_types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.0,
                    response_mime_type="application/json",
                ),
            )
            return _parse_json(getattr(response, "text", "") or "")

        from openai import OpenAI

        # max_retries=0: this module already retries via _llm_json. Leaving the SDK
        # default (2) compounds with our loop AND the OpenRouter free-tier queueing,
        # turning one transient 429 into minutes of backoff. Bound it here.
        kwargs = {"api_key": settings["api_key"], "timeout": settings["timeout"], "max_retries": 0}
        if settings["base_url"]:
            kwargs["base_url"] = settings["base_url"]
        client = OpenAI(**kwargs)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            response = client.chat.completions.create(
                model=settings["model"], temperature=0.0, messages=messages,
                response_format={"type": "json_object"},
            )
        except Exception:  # model may not support response_format
            response = client.chat.completions.create(
                model=settings["model"], temperature=0.0, messages=messages,
            )
        return _parse_json(response.choices[0].message.content or "")
    except Exception as error:  # noqa: BLE001
        logger.warning("LLM call failed: %s", error)
        return None


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
def _catalog_text() -> str:
    lines = []
    for spec in registry.function_specs():
        params = ", ".join(spec["parameters"].get("properties", {}).keys()) or "none"
        lines.append(f"- {spec['name']}({params}): {spec['description'].splitlines()[0]}")
    return "\n".join(lines)


_SYSTEM_PROMPT = (
    "You are a retail inventory analyst for Bunzl. You answer ONLY by calling the "
    "predefined tools below, which read strictly from the Oracle database. You must "
    "never write SQL and never invent data.\n\n"
    "Available tools:\n{catalog}\n\n"
    "Protocol: respond with a single JSON object and nothing else.\n"
    "To call a tool: {{\"action\":\"call_tool\",\"tool\":\"<name>\",\"arguments\":{{...}}}}\n"
    "When you have enough tool data to answer, respond with:\n"
    "{{\"action\":\"final\",\"answer\":\"...\",\"explanation\":\"...\","
    "\"suggestions\":[\"...\"],\"follow_up_question\":\"...\","
    "\"confidence\":\"high|medium|low\",\"supporting_points\":[\"...\"]}}\n\n"
    "Rules: pick the single most relevant tool per step; use at most {max_iter} tool calls; "
    "base every fact on the returned tool data; keep the answer concise (3-6 lines); "
    "if a tool reports a data caveat in its 'notes', respect and surface it.\n"
    "Most tools take OPTIONAL filters (store_id, category, product_id, metric). Omitting a "
    "filter is normal and means 'all stores / all products'; e.g. ranking questions about "
    "top or bottom sellers across the whole business are answered by get_top_products / "
    "get_bottom_products with no store_id. Product master questions — a product list with "
    "reorder point, current/available inventory, category, unit price, supplier or lead time, "
    "the reorder point of a named product, or whether a product is below its reorder point — "
    "are answered by get_product_master (omit arguments to return the whole catalogue). "
    "Use get_products_below_reorder for 'which products are below their reorder point'. "
    "Customer questions: 'which products did <customer> order' -> get_customer_products; "
    "'who is growing/declining' or 'demand trends' -> get_customer_demand_trends; "
    "'dormant/inactive customers' or 'who hasn't ordered' -> get_dormant_accounts; "
    "'abnormal/unusual orders' -> detect_abnormal_ordering. "
    "Always attempt the most relevant tool with sensible default arguments before concluding. "
    "Only answer that you cannot help if genuinely no listed tool is relevant — never claim a "
    "tool is missing when one above can answer."
)


def _build_user_prompt(question: str, history_text: str, transcript: list[dict]) -> str:
    parts = [f"User question: {question}"]
    if history_text:
        parts.append(f"Recent conversation:\n{history_text}")
    if transcript:
        parts.append("Tool results so far:")
        for step in transcript:
            parts.append(
                f"[{step['tool']}({json.dumps(step['arguments'])})] -> "
                + json.dumps(step["result"], default=str)[:3500]
            )
    parts.append("Respond with the next JSON action.")
    return "\n\n".join(parts)


def _history_text(chat_history) -> str:
    if not chat_history:
        return ""
    lines = []
    for message in list(chat_history)[-4:]:
        content = str(getattr(message, "content", "") or "").strip()
        if not content:
            continue
        role = "assistant" if "ai" in str(getattr(message, "type", "")).lower() else "user"
        lines.append(f"{role}: {content[:300]}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Payload helpers
# ---------------------------------------------------------------------------
def _empty_payload() -> dict:
    return {
        "answer": "", "explanation": "", "suggestions": [], "follow_up_question": "",
        "confidence": "low", "supporting_points": [], "cannot_answer": False,
    }


def _final_payload(decision: dict, last_result: dict | None, mode: str) -> tuple[dict, pd.DataFrame, list[dict]]:
    payload = _empty_payload()
    payload.update({
        "answer": str(decision.get("answer", "") or "").strip(),
        "explanation": str(decision.get("explanation", "") or "").strip(),
        "suggestions": [str(s).strip() for s in decision.get("suggestions", []) if str(s).strip()][:2],
        "follow_up_question": str(decision.get("follow_up_question", "") or "").strip(),
        "confidence": str(decision.get("confidence", "medium") or "medium").strip().lower(),
        "supporting_points": [str(s).strip() for s in decision.get("supporting_points", []) if str(s).strip()][:3],
    })
    note = str((last_result or {}).get("notes", "") or "").strip()
    if note and note not in payload["supporting_points"]:
        payload["supporting_points"] = (payload["supporting_points"] + [note])[:3]
    payload["_debug_answer_path"] = "mcp"
    payload["_debug_retrieval_mode"] = mode
    supporting_df = pd.DataFrame((last_result or {}).get("records", []) or [])
    sources = (last_result or {}).get("sources", []) or []
    return payload, supporting_df, sources


def _config_message() -> tuple[dict, pd.DataFrame, list[dict]]:
    payload = _empty_payload()
    payload.update({
        "answer": "The MCP chatbot needs an LLM to choose tools, but none is configured.",
        "explanation": "Add OPENROUTER_API_KEY (or GEMINI_API_KEY) in .env to enable tool selection.",
        "confidence": "low", "cannot_answer": True,
        "_debug_answer_path": "mcp", "_debug_retrieval_mode": "no_llm",
    })
    return payload, pd.DataFrame(), []


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def answer_with_mcp(question: str, chat_history: Any = None) -> tuple[dict, pd.DataFrame, list[dict]]:
    """Answer a question strictly from Oracle via the MCP tool-calling loop."""
    cleaned = str(question or "").strip()
    if not cleaned:
        payload = _empty_payload()
        payload["answer"] = "Ask me about inventory, sales, suppliers, procurement, or branch orders."
        return payload, pd.DataFrame(), []

    if not llm_reasoner.llm_is_configured():
        return _config_message()

    executor, mode = _resolve_executor()
    system_prompt = _SYSTEM_PROMPT.format(
        catalog=_catalog_text(), max_iter=config.max_tool_iterations()
    )
    history_text = _history_text(chat_history)
    transcript: list[dict] = []
    last_result: dict | None = None

    for _ in range(config.max_tool_iterations()):
        user_prompt = _build_user_prompt(cleaned, history_text, transcript)
        decision = _llm_json(system_prompt, user_prompt)
        if decision is None:
            break

        action = str(decision.get("action", "")).strip().lower()
        if action == "final":
            return _final_payload(decision, last_result, mode)

        if action == "call_tool":
            tool_name = str(decision.get("tool", "")).strip()
            arguments = decision.get("arguments", {}) or {}
            spec = registry.get_tool(tool_name)
            if spec is None:
                transcript.append({
                    "tool": tool_name, "arguments": arguments,
                    "result": {"error": f"'{tool_name}' is not an allow-listed tool."},
                })
                continue
            try:
                result = executor(tool_name, spec.coerce_arguments(arguments))
                last_result = result
            except Exception as error:  # noqa: BLE001
                logger.exception("Tool '%s' failed", tool_name)
                result = {"error": f"{type(error).__name__}: {error}"}
            transcript.append({"tool": tool_name, "arguments": arguments, "result": result})
            continue

        # Unknown action shape: stop and summarize what we have.
        break

    # Loop exhausted or LLM stalled: synthesize a final answer from tool data.
    return _synthesize_final(cleaned, system_prompt, transcript, last_result, mode)


def _synthesize_final(
    question: str, system_prompt: str, transcript: list[dict],
    last_result: dict | None, mode: str,
) -> tuple[dict, pd.DataFrame, list[dict]]:
    """Ask the LLM once more for a final answer, or fall back deterministically."""
    if transcript:
        forced = _llm_json(
            system_prompt,
            _build_user_prompt(question, "", transcript)
            + "\n\nYou have used all tool calls. Respond now with a 'final' action only.",
        )
        if forced and str(forced.get("action", "")).lower() == "final":
            return _final_payload(forced, last_result, mode)

    payload = _empty_payload()
    if last_result and last_result.get("records"):
        payload.update({
            "answer": "Here is what the Oracle data shows for your question.",
            "explanation": f"Returned by the {last_result.get('tool', 'tool')} tool.",
            "confidence": "medium",
        })
    else:
        payload.update({
            "answer": "I couldn't find a matching tool answer in the Oracle data for that question.",
            "explanation": "Try asking about stock, sales, suppliers, procurement, demand, or branch orders.",
            "confidence": "low", "cannot_answer": True,
        })
    payload["_debug_answer_path"] = "mcp"
    payload["_debug_retrieval_mode"] = mode
    supporting_df = pd.DataFrame((last_result or {}).get("records", []) or [])
    sources = (last_result or {}).get("sources", []) or []
    return payload, supporting_df, sources
