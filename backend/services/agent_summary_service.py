import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path

import pandas as pd
from google.genai import Client
from google.genai import types as genai_types
from openai import OpenAI
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from backend.services.llm_reasoner import get_llm_settings, llm_is_configured  # noqa: E402
from backend.services.low_stock_service import get_low_stock_items  # noqa: E402
from backend.services.stock_alternative_service import (  # noqa: E402
    get_alternative_availability_for_low_stock,
    get_surplus_stock_items,
)


PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
AGENT_OUTPUTS_PATH = PROCESSED_DATA_DIR / "agent_outputs.csv"
RECOMMENDATIONS_PATH = PROCESSED_DATA_DIR / "recommendations.csv"
ORCHESTRATOR_SUMMARY_PATH = PROCESSED_DATA_DIR / "orchestrator_summary.csv"
AGENT_CARD_SUMMARIES_PATH = PROCESSED_DATA_DIR / "agent_card_summaries.csv"
SUMMARY_LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_SUMMARY_TIMEOUT_SECONDS", "6"))
SUMMARY_LLM_EXECUTOR = ThreadPoolExecutor(max_workers=2)


AGENT_META = {
    "inventory_agent": {
        "display_name": "Inventory",
        "role_label": "Stock & demand",
        "accent": "blue",
    },
    "pricing_agent": {
        "display_name": "Pricing",
        "role_label": "Discount strategy",
        "accent": "purple",
    },
    "transfer_agent": {
        "display_name": "Transfer / Supply",
        "role_label": "Store balancing",
        "accent": "teal",
    },
    "risk_agent": {
        "display_name": "Risk",
        "role_label": "Risk monitoring",
        "accent": "orange",
    },
    "procurement_agent": {
        "display_name": "Procurement",
        "role_label": "Reorder planning",
        "accent": "green",
    },
}


class AgentSummaryResponse(BaseModel):
    summary: str = ""
    recommended_action: str = ""


class AgentBatchSummaryItem(BaseModel):
    agent_name: str = ""
    summary: str = ""
    recommended_action: str = ""


class AgentBatchSummaryResponse(BaseModel):
    agents: list[AgentBatchSummaryItem] = Field(default_factory=list)


class OrchestratorSummaryResponse(BaseModel):
    executive_summary: str = ""
    executive_recommendation: str = ""
    top_risk: str = ""
    top_opportunity: str = ""
    low_stock_alert: str = ""


class DashboardSummaryResponse(BaseModel):
    agents: list[AgentBatchSummaryItem] = Field(default_factory=list)
    orchestrator: OrchestratorSummaryResponse = Field(default_factory=OrchestratorSummaryResponse)


def safe_get(record, key, fallback=""):
    """Safely read a key from dict, Series, tuple/list, or scalar values."""
    if record is None:
        return fallback
    if isinstance(record, dict):
        return record.get(key, fallback)
    if isinstance(record, pd.Series):
        return record.get(key, fallback)
    if isinstance(record, (list, tuple)):
        if isinstance(key, int) and 0 <= key < len(record):
            return record[key]
        for item in record:
            if isinstance(item, (dict, pd.Series)) and key in item:
                return item.get(key, fallback)
        return fallback
    return getattr(record, str(key), fallback)


def safe_text(value, fallback="") -> str:
    """Return a clean, non-crashing text value for dashboard summaries."""
    if value is None:
        return fallback
    if isinstance(value, pd.Series):
        value = value.dropna().to_dict()
    if isinstance(value, dict):
        parts = []
        for key, item in value.items():
            text = safe_text(item, "")
            if text:
                parts.append(f"{str(key).replace('_', ' ').title()}: {text}")
        return "; ".join(parts) or fallback
    if isinstance(value, (list, tuple, set)):
        parts = [safe_text(item, "") for item in value]
        parts = [part for part in parts if part]
        return "; ".join(parts) or fallback
    text = " ".join(str(value or "").split()).strip()
    return text or fallback


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _build_openai_client() -> OpenAI:
    settings = get_llm_settings()
    client_kwargs = {
        "api_key": settings["api_key"],
        "timeout": max(1, min(int(settings.get("timeout", 45) or 45), SUMMARY_LLM_TIMEOUT_SECONDS)),
    }
    if settings["base_url"]:
        client_kwargs["base_url"] = settings["base_url"]
    return OpenAI(**client_kwargs)


def _chat_json_direct(system_prompt: str, user_payload: dict) -> str:
    settings = get_llm_settings()
    if settings["provider"] == "gemini":
        client = Client(api_key=settings["api_key"])
        response = client.models.generate_content(
            model=settings["model"],
            contents=json.dumps(user_payload, ensure_ascii=True),
            config=genai_types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.1,
                response_mime_type="application/json",
            ),
        )
        return getattr(response, "text", "") or ""

    client = _build_openai_client()
    response = client.chat.completions.create(
        model=settings["model"],
        temperature=0.1,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=True)},
        ],
    )
    return response.choices[0].message.content or ""


def _chat_json(system_prompt: str, user_payload: dict) -> str:
    """Call the LLM with a short dashboard-summary timeout."""
    settings = get_llm_settings()
    timeout = max(1, min(int(settings.get("timeout", 45) or 45), SUMMARY_LLM_TIMEOUT_SECONDS))
    future = SUMMARY_LLM_EXECUTOR.submit(_chat_json_direct, system_prompt, user_payload)
    try:
        return future.result(timeout=timeout)
    except TimeoutError as error:
        future.cancel()
        raise TimeoutError(f"LLM summary timed out after {timeout}s") from error


def _extract_json_text(content: str) -> str:
    text = str(content or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:] if lines else lines
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _clean_sentence(text: str, fallback: str) -> str:
    cleaned = safe_text(text, fallback)
    if not cleaned:
        return fallback
    if cleaned[-1] not in ".!?":
        cleaned += "."
    return cleaned


def _normalize_priority(value: str) -> str:
    text = str(value or "").strip().title()
    return text or "Info"


def _first_non_empty(series: pd.Series) -> str:
    if series.empty:
        return ""
    values = series.fillna("").astype(str)
    values = values[values.str.strip() != ""]
    if values.empty:
        return ""
    return values.iloc[0].strip()


def _recommendation_rows_for_agent(
    recommendations: pd.DataFrame,
    agent_name: str,
) -> pd.DataFrame:
    if recommendations.empty or "source_agent" not in recommendations.columns:
        return pd.DataFrame()
    return recommendations[
        recommendations["source_agent"].fillna("").astype(str).eq(agent_name)
    ].copy()


def _priority_sorted(df: pd.DataFrame, limit: int = 3) -> pd.DataFrame:
    if df.empty:
        return df
    ranked = df.copy()
    if "priority" in ranked.columns:
        ranked["_priority_rank"] = (
            ranked["priority"]
            .fillna("")
            .astype(str)
            .str.lower()
            .str.title()
            .map({"High": 0, "Medium": 1, "Low": 2})
            .fillna(3)
        )
        ranked = ranked.sort_values("_priority_rank")
    return ranked.head(limit).drop(columns=["_priority_rank"], errors="ignore")


def _compact_records(df: pd.DataFrame, columns: list[str], limit: int) -> list[dict]:
    if df.empty:
        return []
    available = [column for column in columns if column in df.columns]
    if not available:
        return []
    return df[available].head(limit).fillna("").astype(str).to_dict(orient="records")


def _supplier_risk_summary(recommendations: pd.DataFrame) -> dict:
    if recommendations.empty or "recommendation_type" not in recommendations.columns:
        return {"count": 0, "top_suppliers": []}
    risk_rows = recommendations[
        recommendations["recommendation_type"]
        .fillna("")
        .astype(str)
        .eq("supplier_risk_alert")
    ].copy()
    if risk_rows.empty:
        return {"count": 0, "top_suppliers": []}
    return {
        "count": int(len(risk_rows)),
        "top_suppliers": _compact_records(
            _priority_sorted(risk_rows, 5),
            ["product_name", "priority", "action", "reason", "evidence"],
            5,
        ),
    }


def build_compact_llm_context(
    agent_outputs_df: pd.DataFrame,
    recommendations_df: pd.DataFrame,
    low_stock_df: pd.DataFrame,
) -> dict:
    """Build the only dashboard context sent to summary LLM calls."""
    stockout_df = _safe_read_csv(PROCESSED_DATA_DIR / "stockout_risk_items.csv")
    overstock_df = _safe_read_csv(PROCESSED_DATA_DIR / "overstock_items.csv")

    recommendation_counts = {}
    if not recommendations_df.empty and "recommendation_type" in recommendations_df.columns:
        recommendation_counts = (
            recommendations_df["recommendation_type"]
            .fillna("unknown")
            .astype(str)
            .value_counts()
            .to_dict()
        )

    top_recommendations_by_agent = {}
    for agent_name in AGENT_META:
        agent_rows = _recommendation_rows_for_agent(recommendations_df, agent_name)
        top_recommendations_by_agent[agent_name] = _compact_records(
            _priority_sorted(agent_rows, 3),
            [
                "recommendation_type",
                "product_name",
                "store_id",
                "priority",
                "action",
                "reason",
            ],
            3,
        )

    return {
        "agent_outputs": agent_outputs_df.fillna("").astype(str).to_dict(orient="records"),
        "low_stock_count": int(len(low_stock_df)),
        "stockout_risk_count": int(len(stockout_df)),
        "overstock_count": int(len(overstock_df)),
        "top_low_stock_items": _compact_records(
            _priority_sorted(low_stock_df, 5),
            [
                "product_name",
                "store_name",
                "store_id",
                "current_quantity",
                "reorder_threshold",
                "suggested_reorder_quantity",
                "priority",
            ],
            5,
        ),
        "top_stockout_risks": _compact_records(
            _priority_sorted(stockout_df, 5),
            [
                "product_name",
                "store_id",
                "stock_level",
                "days_of_stock_remaining",
                "recent_daily_sales_velocity",
                "reason",
            ],
            5,
        ),
        "top_overstock_rows": _compact_records(
            _priority_sorted(overstock_df, 5),
            [
                "product_name",
                "store_id",
                "stock_level",
                "days_of_stock_remaining",
                "reason",
            ],
            5,
        ),
        "recommendation_counts_by_type": recommendation_counts,
        "top_3_recommendations_per_agent": top_recommendations_by_agent,
        "supplier_risk_summary": _supplier_risk_summary(recommendations_df),
        "top_risk": _top_risk_text(recommendations_df),
        "top_opportunity": _top_opportunity_text(recommendations_df),
    }


def _top_risk_text(recommendations: pd.DataFrame) -> str:
    if recommendations.empty:
        return "No major risk stands out in the latest run."

    risk_types = {"supplier_risk_alert", "stockout_prevention_alert", "overstock_alert"}
    risk_rows = recommendations[
        recommendations.get("recommendation_type", pd.Series(dtype=object))
        .fillna("")
        .astype(str)
        .isin(risk_types)
    ].copy()
    if risk_rows.empty:
        return "No major risk stands out in the latest run."

    if "priority" in risk_rows.columns:
        risk_rows["_priority_rank"] = (
            risk_rows["priority"]
            .fillna("")
            .astype(str)
            .str.lower()
            .map({"high": 0, "medium": 1, "low": 2})
            .fillna(3)
        )
        risk_rows = risk_rows.sort_values("_priority_rank")

    row = risk_rows.iloc[0]
    product_name = str(row.get("product_name", "") or "").strip()
    reason = str(row.get("reason", "") or "").strip()
    recommendation_type = str(row.get("recommendation_type", "") or "").replace("_", " ").strip()
    if product_name and reason:
        return _clean_sentence(f"{product_name}: {reason}", "No major risk stands out in the latest run.")
    if reason:
        return _clean_sentence(reason, "No major risk stands out in the latest run.")
    return _clean_sentence(f"{recommendation_type.title()} needs attention", "No major risk stands out in the latest run.")


def _top_opportunity_text(recommendations: pd.DataFrame) -> str:
    if recommendations.empty:
        return "No standout commercial opportunity is available yet."

    opportunity_types = {
        "reorder",
        "discount",
        "stock_transfer",
        "transfer",
        "clearance",
        "exclusive_availability",
        "alternative_option",
    }
    opportunity_rows = recommendations[
        recommendations.get("recommendation_type", pd.Series(dtype=object))
        .fillna("")
        .astype(str)
        .isin(opportunity_types)
    ].copy()
    if opportunity_rows.empty:
        return "No standout commercial opportunity is available yet."

    if "priority" in opportunity_rows.columns:
        opportunity_rows["_priority_rank"] = (
            opportunity_rows["priority"]
            .fillna("")
            .astype(str)
            .str.lower()
            .map({"high": 0, "medium": 1, "low": 2})
            .fillna(3)
        )
        opportunity_rows = opportunity_rows.sort_values("_priority_rank")

    row = opportunity_rows.iloc[0]
    product_name = str(row.get("product_name", "") or "").strip()
    action = str(row.get("action", "") or "").strip()
    recommendation_type = str(row.get("recommendation_type", "") or "").replace("_", " ").strip()
    if product_name and action:
        return _clean_sentence(f"{product_name}: {action}", "No standout commercial opportunity is available yet.")
    if action:
        return _clean_sentence(action, "No standout commercial opportunity is available yet.")
    return _clean_sentence(f"{recommendation_type.title()} is the strongest current opportunity", "No standout commercial opportunity is available yet.")


def _fallback_agent_summary(agent_row: pd.Series, recommendations: pd.DataFrame) -> tuple[str, str]:
    finding_count = int(pd.to_numeric(agent_row.get("finding_count", 0), errors="coerce") or 0)
    latest_insight = str(agent_row.get("latest_insight", "") or "").strip()
    reason = _first_non_empty(recommendations.get("reason", pd.Series(dtype=object)))
    action = _first_non_empty(recommendations.get("action", pd.Series(dtype=object)))

    if finding_count == 0:
        return (
            "No new findings are available yet.",
            "Run the agents to refresh the latest view.",
        )

    agent_name = str(agent_row.get("agent_name", "") or "").strip()
    if agent_name == "transfer_agent" and not recommendations.empty:
        recommendation_type = recommendations.get("recommendation_type", pd.Series(dtype=object)).fillna("").astype(str)
        transfer_count = int(recommendation_type.isin(["stock_transfer", "transfer"]).sum())
        exclusive_count = int(recommendation_type.eq("exclusive_availability").sum())
        surplus_df = get_surplus_stock_items()
        alternatives_df = get_alternative_availability_for_low_stock()
        top_source_store = (
            str(surplus_df.iloc[0].get("store_name", ""))
            if not surplus_df.empty
            else "no source store"
        )
        alternative_rows = recommendations[recommendation_type.eq("alternative_option")].copy()
        top_alternative = _first_non_empty(alternative_rows.get("action", pd.Series(dtype=object)))
        latest_transfer = _first_non_empty(
            recommendations[recommendation_type.isin(["stock_transfer", "transfer"])].get("action", pd.Series(dtype=object))
        )
        summary = (
            f"{transfer_count} transfer opportunities detected. "
            f"{len(surplus_df)} surplus stock items and {len(alternatives_df)} alternative availability options are active. "
            f"Top source store: {top_source_store}."
        )
        action_parts = [part for part in [top_alternative, latest_transfer] if part]
        return (
            _clean_sentence(summary, "Transfer and alternative availability findings are ready."),
            _clean_sentence(
                " ".join(action_parts[:2]),
                "Review exclusive availability and transfer options together.",
            ),
        )

    summary = latest_insight or f"{finding_count} findings were flagged in the latest run."
    if finding_count > 0 and str(finding_count) not in summary:
        summary = f"{finding_count} findings were flagged. {summary}"

    recommended_action = action or reason or "Review the latest items and act on the highest priority rows first."
    return (
        _clean_sentence(summary, "The latest run produced new findings."),
        _clean_sentence(recommended_action, "Review the highest priority rows first."),
    )


def _llm_agent_summary(agent_row: pd.Series, recommendations: pd.DataFrame) -> tuple[str, str]:
    summary, action = _fallback_agent_summary(agent_row, recommendations)
    if not llm_is_configured():
        return summary, action

    sample_columns = [
        column
        for column in [
            "recommendation_type",
            "product_name",
            "store_id",
            "priority",
            "action",
            "reason",
            "evidence",
            "suggested_quantity",
        ]
        if column in recommendations.columns
    ]
    sample_rows = recommendations[sample_columns].head(5).to_dict(orient="records")
    payload = {
        "agent_name": str(agent_row.get("agent_name", "")),
        "finding_count": int(pd.to_numeric(agent_row.get("finding_count", 0), errors="coerce") or 0),
        "priority_level": str(agent_row.get("priority_level", "")),
        "latest_insight": str(agent_row.get("latest_insight", "")),
        "sample_recommendations": sample_rows,
        "required_output_schema": {
            "summary": "one or two short human-like lines",
            "recommended_action": "one short action line",
        },
    }
    system_prompt = (
        "You summarize retail inventory agent output for a premium dashboard. "
        "Use only the provided data. "
        "Keep it compact, managerial, and concrete. "
        "Do not invent products, counts, or trends. "
        "Return valid JSON only."
    )
    try:
        raw = _chat_json(system_prompt, payload)
        parsed = AgentSummaryResponse.model_validate(json.loads(_extract_json_text(raw)))
        return (
            _clean_sentence(parsed.summary, summary),
            _clean_sentence(parsed.recommended_action, action),
        )
    except Exception:
        return summary, action


def _llm_agent_summaries_batch(
    agent_outputs_df: pd.DataFrame,
    recommendations_df: pd.DataFrame,
    compact_context: dict,
) -> tuple[dict[str, tuple[str, str]], bool]:
    """Summarize all five agent cards with a single compact LLM call."""
    fallback_by_agent = {}
    for _, agent_row in agent_outputs_df.iterrows():
        agent_name = str(agent_row.get("agent_name", "") or "").strip()
        fallback_by_agent[agent_name] = _fallback_agent_summary(
            agent_row,
            _recommendation_rows_for_agent(recommendations_df, agent_name),
        )

    if not llm_is_configured():
        return fallback_by_agent, False

    system_prompt = (
        "You summarize five retail inventory agent cards for a dashboard. "
        "Use only the compact data provided. Do not invent products, counts, or trends. "
        "Return valid JSON only."
    )
    payload = {
        "compact_context": compact_context,
        "required_output_schema": {
            "agents": [
                {
                    "agent_name": "inventory_agent|pricing_agent|transfer_agent|risk_agent|procurement_agent",
                    "summary": "one short concrete line",
                    "recommended_action": "one short action line",
                }
            ]
        },
    }

    try:
        raw = _chat_json(system_prompt, payload)
        parsed = AgentBatchSummaryResponse.model_validate(json.loads(_extract_json_text(raw)))
        output = dict(fallback_by_agent)
        for item in parsed.agents:
            agent_name = str(item.agent_name or "").strip()
            if agent_name not in output:
                continue
            fallback_summary, fallback_action = output[agent_name]
            output[agent_name] = (
                _clean_sentence(item.summary, fallback_summary),
                _clean_sentence(item.recommended_action, fallback_action),
            )
        return output, True
    except Exception as error:
        print(f"[agent_refresh] LLM summary skipped, using data summary: {error}")
        return fallback_by_agent, False


def _fallback_orchestrator_summary(
    orchestrator_row: pd.Series,
    agent_summaries_df: pd.DataFrame,
    recommendations: pd.DataFrame,
    low_stock_df: pd.DataFrame,
) -> dict[str, str]:
    top_risk = _top_risk_text(recommendations)
    top_opportunity = _top_opportunity_text(recommendations)
    total_recommendations = int(
        pd.to_numeric(orchestrator_row.get("total_recommendations", 0), errors="coerce") or 0
    )
    high_priority_alerts = int(
        pd.to_numeric(orchestrator_row.get("high_priority_alerts", 0), errors="coerce") or 0
    )

    low_stock_alert = build_low_stock_alert_text(low_stock_df)
    executive_summary = (
        f"{total_recommendations} recommendations are active, with {high_priority_alerts} high-priority alerts. "
        f"Top risk: {top_risk}"
    )
    executive_recommendation = (
        f"Top opportunity: {top_opportunity} Prioritize the highest-risk items first, then move on the strongest commercial opportunity."
    )
    return {
        "top_risk": _clean_sentence(top_risk, "No major risk stands out in the latest run."),
        "top_opportunity": _clean_sentence(top_opportunity, "No standout commercial opportunity is available yet."),
        "low_stock_alert": _clean_sentence(low_stock_alert, "No critical low-stock alerts right now."),
        "executive_summary": _clean_sentence(
            executive_summary,
            "The latest run is ready for review.",
        ),
        "executive_recommendation": _clean_sentence(
            executive_recommendation,
            "Review the latest recommendations and act on the highest-priority items first.",
        ),
    }


def _llm_orchestrator_summary(
    orchestrator_row: pd.Series,
    agent_summaries_df: pd.DataFrame,
    recommendations: pd.DataFrame,
    low_stock_df: pd.DataFrame,
    compact_context: dict | None = None,
) -> tuple[dict[str, str], bool]:
    fallback = _fallback_orchestrator_summary(
        orchestrator_row,
        agent_summaries_df,
        recommendations,
        low_stock_df,
    )
    if not llm_is_configured():
        return fallback, False

    payload = {
        "orchestrator_summary": orchestrator_row.fillna("").astype(str).to_dict(),
        "agent_summaries": agent_summaries_df.fillna("").astype(str).to_dict(orient="records"),
        "compact_context": compact_context or {},
        "required_output_schema": {
            "low_stock_alert": "one short alert line",
            "top_risk": "one short line",
            "top_opportunity": "one short line",
            "executive_summary": "two or three short lines combined into one compact paragraph",
            "executive_recommendation": "one concise manager-level action line",
        },
    }
    system_prompt = (
        "You write executive dashboard summaries for a retail operations command center. "
        "Use only the data provided. "
        "Be concise, human, and manager-friendly. "
        "Do not invent metrics or trends. "
        "Return valid JSON only."
    )
    try:
        raw = _chat_json(system_prompt, payload)
        parsed = OrchestratorSummaryResponse.model_validate(json.loads(_extract_json_text(raw)))
        return {
            "low_stock_alert": _clean_sentence(parsed.low_stock_alert, fallback["low_stock_alert"]),
            "top_risk": _clean_sentence(parsed.top_risk, fallback["top_risk"]),
            "top_opportunity": _clean_sentence(parsed.top_opportunity, fallback["top_opportunity"]),
            "executive_summary": _clean_sentence(parsed.executive_summary, fallback["executive_summary"]),
            "executive_recommendation": _clean_sentence(
                parsed.executive_recommendation,
                fallback["executive_recommendation"],
            ),
        }, True
    except Exception as error:
        print(f"[agent_refresh] LLM summary skipped, using data summary: {error}")
        return fallback, False


def _fallback_dashboard_summaries(
    agent_outputs_df: pd.DataFrame,
    recommendations_df: pd.DataFrame,
    low_stock_df: pd.DataFrame,
    orchestrator_row: pd.Series,
) -> tuple[dict[str, tuple[str, str]], dict[str, str]]:
    agent_summaries = {}
    card_rows = []
    for _, agent_row in agent_outputs_df.iterrows():
        agent_name = safe_text(safe_get(agent_row, "agent_name"), "")
        summary, action = _fallback_agent_summary(
            agent_row,
            _recommendation_rows_for_agent(recommendations_df, agent_name),
        )
        agent_summaries[agent_name] = (summary, action)
        card_rows.append(
            {
                "agent_name": agent_name,
                "summary": summary,
                "recommended_action": action,
            }
        )

    fallback_cards_df = pd.DataFrame(card_rows)
    orchestrator_summary = _fallback_orchestrator_summary(
        orchestrator_row,
        fallback_cards_df,
        recommendations_df,
        low_stock_df,
    )
    return agent_summaries, orchestrator_summary


def _llm_dashboard_summary(
    agent_outputs_df: pd.DataFrame,
    recommendations_df: pd.DataFrame,
    low_stock_df: pd.DataFrame,
    orchestrator_row: pd.Series,
    compact_context: dict,
) -> tuple[dict[str, tuple[str, str]], dict[str, str], bool]:
    """Generate all card and orchestrator text in one bounded LLM call."""
    fallback_agents, fallback_orchestrator = _fallback_dashboard_summaries(
        agent_outputs_df,
        recommendations_df,
        low_stock_df,
        orchestrator_row,
    )
    if not llm_is_configured():
        return fallback_agents, fallback_orchestrator, False

    system_prompt = (
        "You write compact dashboard summaries for a retail inventory command center. "
        "Use only the compact context provided. Do not invent counts, products, stores, or trends. "
        "Return valid JSON only."
    )
    payload = {
        "compact_context": compact_context,
        "orchestrator_summary": orchestrator_row.fillna("").astype(str).to_dict(),
        "required_output_schema": {
            "agents": [
                {
                    "agent_name": "inventory_agent|pricing_agent|transfer_agent|risk_agent|procurement_agent",
                    "summary": "one short concrete line",
                    "recommended_action": "one short action line",
                }
            ],
            "orchestrator": {
                "low_stock_alert": "one short alert line",
                "top_risk": "one short line",
                "top_opportunity": "one short line",
                "executive_summary": "one compact paragraph",
                "executive_recommendation": "one concise manager-level action line",
            },
        },
    }

    try:
        raw = _chat_json(system_prompt, payload)
        parsed = DashboardSummaryResponse.model_validate(json.loads(_extract_json_text(raw)))
        agent_output = dict(fallback_agents)
        for item in parsed.agents:
            agent_name = safe_text(item.agent_name, "")
            if agent_name not in agent_output:
                continue
            fallback_summary, fallback_action = agent_output[agent_name]
            agent_output[agent_name] = (
                _clean_sentence(item.summary, fallback_summary),
                _clean_sentence(item.recommended_action, fallback_action),
            )

        orchestrator = parsed.orchestrator
        orchestrator_output = {
            "low_stock_alert": _clean_sentence(
                orchestrator.low_stock_alert,
                fallback_orchestrator["low_stock_alert"],
            ),
            "top_risk": _clean_sentence(orchestrator.top_risk, fallback_orchestrator["top_risk"]),
            "top_opportunity": _clean_sentence(
                orchestrator.top_opportunity,
                fallback_orchestrator["top_opportunity"],
            ),
            "executive_summary": _clean_sentence(
                orchestrator.executive_summary,
                fallback_orchestrator["executive_summary"],
            ),
            "executive_recommendation": _clean_sentence(
                orchestrator.executive_recommendation,
                fallback_orchestrator["executive_recommendation"],
            ),
        }
        return agent_output, orchestrator_output, True
    except Exception as error:
        print(f"[agent_refresh] LLM summary skipped, using data summary: {error}")
        return fallback_agents, fallback_orchestrator, False


def build_low_stock_alert_text(low_stock_df: pd.DataFrame) -> str:
    """Build a concise manager-facing low-stock alert line."""
    if low_stock_df.empty:
        return "No critical low-stock alerts right now."

    product_count = (
        low_stock_df["product_id"].astype(str).replace("", pd.NA).dropna().nunique()
        if "product_id" in low_stock_df.columns
        else len(low_stock_df)
    )
    store_count = (
        low_stock_df["store_id"].astype(str).replace("", pd.NA).dropna().nunique()
        if "store_id" in low_stock_df.columns
        else 0
    )
    high_priority_rows = low_stock_df[
        low_stock_df.get("priority", pd.Series(dtype=object)).fillna("").astype(str).eq("High")
    ].copy()
    focus_df = high_priority_rows if not high_priority_rows.empty else low_stock_df
    focus_row = focus_df.iloc[0]
    product_name = str(focus_row.get("product_name", focus_row.get("product_id", ""))).strip()
    store_name = str(focus_row.get("store_name", focus_row.get("store_id", ""))).strip()
    current_quantity = int(pd.to_numeric(focus_row.get("current_quantity", 0), errors="coerce") or 0)
    return (
        f"{product_count} product{'s' if product_count != 1 else ''} are below reorder threshold across "
        f"{store_count} store{'s' if store_count != 1 else ''}. "
        f"Highest priority: {product_name} at {store_name} has only {current_quantity} units left."
    )


def generate_agent_card_summaries(
    agent_outputs_df: pd.DataFrame | None = None,
    recommendations_df: pd.DataFrame | None = None,
    orchestrator_summary_df: pd.DataFrame | None = None,
    save_output: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate compact card summaries for all agents and the orchestrator."""
    agent_outputs_df = (
        _safe_read_csv(AGENT_OUTPUTS_PATH) if agent_outputs_df is None else agent_outputs_df.copy()
    )
    recommendations_df = (
        _safe_read_csv(RECOMMENDATIONS_PATH) if recommendations_df is None else recommendations_df.copy()
    )
    orchestrator_summary_df = (
        _safe_read_csv(ORCHESTRATOR_SUMMARY_PATH)
        if orchestrator_summary_df is None
        else orchestrator_summary_df.copy()
    )

    if agent_outputs_df.empty:
        empty_cards = pd.DataFrame(
            columns=[
                "run_time",
                "agent_name",
                "display_name",
                "role_label",
                "accent",
                "finding_count",
                "priority_level",
                "summary",
                "recommended_action",
                "summary_source",
            ]
        )
        return empty_cards, orchestrator_summary_df

    low_stock_df = get_low_stock_items(save_output=True)
    compact_context = build_compact_llm_context(
        agent_outputs_df,
        recommendations_df,
        low_stock_df,
    )
    if orchestrator_summary_df.empty:
        orchestrator_summary_df = pd.DataFrame(
            [
                {
                    "run_time": "",
                    "database_health": "Unknown",
                    "total_recommendations": int(len(recommendations_df)),
                    "high_priority_alerts": 0,
                    "last_agent_run_time": "",
                    "summary": "Run agents to generate the latest summary.",
                }
            ]
        )
    orchestrator_row = orchestrator_summary_df.iloc[0].copy()
    batch_summaries, executive, llm_used = _llm_dashboard_summary(
        agent_outputs_df,
        recommendations_df,
        low_stock_df,
        orchestrator_row,
        compact_context,
    )
    summary_source = "llm" if llm_used else "fallback"
    card_rows = []

    for _, agent_row in agent_outputs_df.iterrows():
        agent_name = str(agent_row.get("agent_name", "") or "").strip()
        meta = AGENT_META.get(
            agent_name,
            {
                "display_name": agent_name.replace("_", " ").title(),
                "role_label": "Agent summary",
                "accent": "blue",
            },
        )
        agent_recommendations = _recommendation_rows_for_agent(recommendations_df, agent_name)
        summary, recommended_action = batch_summaries.get(
            agent_name,
            _fallback_agent_summary(agent_row, agent_recommendations),
        )
        card_rows.append(
            {
                "run_time": str(agent_row.get("run_time", "")),
                "agent_name": agent_name,
                "display_name": meta["display_name"],
                "role_label": meta["role_label"],
                "accent": meta["accent"],
                "finding_count": int(pd.to_numeric(agent_row.get("finding_count", 0), errors="coerce") or 0),
                "priority_level": _normalize_priority(agent_row.get("priority_level", "Info")),
                "summary": safe_text(summary, "The latest run produced new findings."),
                "recommended_action": safe_text(
                    recommended_action,
                    "Review the highest priority rows first.",
                ),
                "summary_source": summary_source,
            }
        )

    agent_card_summaries_df = pd.DataFrame(card_rows)

    orchestrator_summary_df = orchestrator_summary_df.copy()
    orchestrator_summary_df.loc[0, "low_stock_alert"] = safe_text(
        safe_get(executive, "low_stock_alert"),
        "No critical low-stock alerts right now.",
    )
    orchestrator_summary_df.loc[0, "top_risk"] = safe_text(
        safe_get(executive, "top_risk"),
        "No major risk stands out in the latest run.",
    )
    orchestrator_summary_df.loc[0, "top_opportunity"] = safe_text(
        safe_get(executive, "top_opportunity"),
        "No standout commercial opportunity is available yet.",
    )
    orchestrator_summary_df.loc[0, "executive_summary"] = safe_text(
        safe_get(executive, "executive_summary"),
        "The latest run is ready for review.",
    )
    orchestrator_summary_df.loc[0, "executive_recommendation"] = safe_text(
        safe_get(executive, "executive_recommendation"),
        "Review the latest recommendations and act on the highest-priority items first.",
    )
    orchestrator_summary_df.loc[0, "summary_source"] = summary_source

    if save_output:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        agent_card_summaries_df.to_csv(AGENT_CARD_SUMMARIES_PATH, index=False)
        orchestrator_summary_df.to_csv(ORCHESTRATOR_SUMMARY_PATH, index=False)

    return agent_card_summaries_df, orchestrator_summary_df


def ensure_agent_card_summaries() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create card summary files when the upstream outputs exist."""
    if AGENT_CARD_SUMMARIES_PATH.exists() and ORCHESTRATOR_SUMMARY_PATH.exists():
        return _safe_read_csv(AGENT_CARD_SUMMARIES_PATH), _safe_read_csv(ORCHESTRATOR_SUMMARY_PATH)
    return generate_agent_card_summaries(save_output=True)
