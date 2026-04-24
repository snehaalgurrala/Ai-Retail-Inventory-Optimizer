import json
import sys
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


PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
AGENT_OUTPUTS_PATH = PROCESSED_DATA_DIR / "agent_outputs.csv"
RECOMMENDATIONS_PATH = PROCESSED_DATA_DIR / "recommendations.csv"
ORCHESTRATOR_SUMMARY_PATH = PROCESSED_DATA_DIR / "orchestrator_summary.csv"
AGENT_CARD_SUMMARIES_PATH = PROCESSED_DATA_DIR / "agent_card_summaries.csv"


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


class OrchestratorSummaryResponse(BaseModel):
    executive_summary: str = ""
    executive_recommendation: str = ""
    top_risk: str = ""
    top_opportunity: str = ""
    low_stock_alert: str = ""


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
        "timeout": settings["timeout"],
    }
    if settings["base_url"]:
        client_kwargs["base_url"] = settings["base_url"]
    return OpenAI(**client_kwargs)


def _chat_json(system_prompt: str, user_payload: dict) -> str:
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
    cleaned = " ".join(str(text or "").split()).strip()
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

    opportunity_types = {"reorder", "discount", "stock_transfer", "clearance"}
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
) -> dict[str, str]:
    fallback = _fallback_orchestrator_summary(
        orchestrator_row,
        agent_summaries_df,
        recommendations,
        low_stock_df,
    )
    if not llm_is_configured():
        return fallback

    sample_columns = [
        column
        for column in [
            "recommendation_type",
            "product_name",
            "store_id",
            "priority",
            "action",
            "reason",
            "source_agent",
        ]
        if column in recommendations.columns
    ]
    payload = {
        "orchestrator_summary": orchestrator_row.fillna("").astype(str).to_dict(),
        "agent_summaries": agent_summaries_df.fillna("").astype(str).to_dict(orient="records"),
        "low_stock_alert": low_stock_df.head(5).fillna("").astype(str).to_dict(orient="records"),
        "sample_recommendations": recommendations[sample_columns].head(8).to_dict(orient="records"),
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
        }
    except Exception:
        return fallback


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

    summary_source = "llm" if llm_is_configured() else "fallback"
    low_stock_df = get_low_stock_items(save_output=True)
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
        summary, recommended_action = _llm_agent_summary(agent_row, agent_recommendations)
        card_rows.append(
            {
                "run_time": str(agent_row.get("run_time", "")),
                "agent_name": agent_name,
                "display_name": meta["display_name"],
                "role_label": meta["role_label"],
                "accent": meta["accent"],
                "finding_count": int(pd.to_numeric(agent_row.get("finding_count", 0), errors="coerce") or 0),
                "priority_level": _normalize_priority(agent_row.get("priority_level", "Info")),
                "summary": summary,
                "recommended_action": recommended_action,
                "summary_source": summary_source,
            }
        )

    agent_card_summaries_df = pd.DataFrame(card_rows)

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
    executive = _llm_orchestrator_summary(
        orchestrator_row,
        agent_card_summaries_df,
        recommendations_df,
        low_stock_df,
    )

    orchestrator_summary_df = orchestrator_summary_df.copy()
    orchestrator_summary_df.loc[0, "low_stock_alert"] = executive["low_stock_alert"]
    orchestrator_summary_df.loc[0, "top_risk"] = executive["top_risk"]
    orchestrator_summary_df.loc[0, "top_opportunity"] = executive["top_opportunity"]
    orchestrator_summary_df.loc[0, "executive_summary"] = executive["executive_summary"]
    orchestrator_summary_df.loc[0, "executive_recommendation"] = executive["executive_recommendation"]
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
