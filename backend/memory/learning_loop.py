from pathlib import Path

import pandas as pd

from backend.memory.memory_store import (
    MEMORY_DIR,
    get_memory_context,
    load_decision_memory,
    load_outcome_memory,
    load_recommendation_memory,
)
from backend.services.llm_reasoner import summarize_learning_feedback


LEARNING_INSIGHTS_FILE = MEMORY_DIR / "learning_insights.csv"
LEARNING_INSIGHT_COLUMNS = [
    "insight_id",
    "recorded_at",
    "recommendation_type",
    "product_id",
    "product_name",
    "store_id",
    "action_taken",
    "outcome_status",
    "learning_label",
    "insight",
    "recommended_adjustment",
    "source",
]


def _read_learning_insights() -> pd.DataFrame:
    if not LEARNING_INSIGHTS_FILE.exists():
        return pd.DataFrame(columns=LEARNING_INSIGHT_COLUMNS)

    try:
        df = pd.read_csv(LEARNING_INSIGHTS_FILE)
    except Exception:
        return pd.DataFrame(columns=LEARNING_INSIGHT_COLUMNS)

    for column in LEARNING_INSIGHT_COLUMNS:
        if column not in df.columns:
            df[column] = ""
    return df[LEARNING_INSIGHT_COLUMNS].copy()


def _write_learning_insights(df: pd.DataFrame) -> Path:
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    writable_df = df.copy()
    for column in LEARNING_INSIGHT_COLUMNS:
        if column not in writable_df.columns:
            writable_df[column] = ""
    writable_df[LEARNING_INSIGHT_COLUMNS].to_csv(LEARNING_INSIGHTS_FILE, index=False)
    return LEARNING_INSIGHTS_FILE


def _combined_feedback_rows() -> pd.DataFrame:
    """Join recommendations, decisions, and outcomes into feedback rows."""
    recommendation_memory = load_recommendation_memory()
    decision_memory = load_decision_memory()
    outcome_memory = load_outcome_memory()

    if recommendation_memory.empty:
        return pd.DataFrame()

    feedback = recommendation_memory.copy()

    if not decision_memory.empty:
        latest_decisions = (
            decision_memory.sort_values("recorded_at")
            .drop_duplicates("recommendation_id", keep="last")
            .rename(columns={"recorded_at": "decision_recorded_at"})
        )
        feedback = feedback.merge(
            latest_decisions[
                [
                    "recommendation_id",
                    "decision_recorded_at",
                    "decision",
                ]
            ],
            on="recommendation_id",
            how="left",
        )

    if not outcome_memory.empty:
        latest_outcomes = (
            outcome_memory.sort_values("recorded_at")
            .drop_duplicates("recommendation_id", keep="last")
            .rename(columns={"recorded_at": "outcome_recorded_at"})
        )
        feedback = feedback.merge(
            latest_outcomes[
                [
                    "recommendation_id",
                    "outcome_recorded_at",
                    "action_taken",
                    "outcome_status",
                    "outcome_note",
                ]
            ],
            on="recommendation_id",
            how="left",
        )

    for column in [
        "decision",
        "action_taken",
        "outcome_status",
        "outcome_note",
    ]:
        if column not in feedback.columns:
            feedback[column] = ""
        feedback[column] = feedback[column].fillna("").astype(str)

    return feedback


def _heuristic_learning_insight(row: pd.Series) -> dict:
    """Create a simple deterministic learning insight from one feedback row."""
    outcome_status = str(row.get("outcome_status", "")).strip().lower()
    recommendation_type = str(row.get("recommendation_type", "")).strip()
    action_taken = str(row.get("action_taken", "")).strip() or str(
        row.get("decision", "")
    ).strip()
    outcome_note = str(row.get("outcome_note", "")).strip()

    positive_terms = {"worked", "success", "improved", "good", "effective"}
    negative_terms = {"failed", "poor", "bad", "declined", "ineffective"}

    learning_label = "neutral"
    recommended_adjustment = "monitor future results before changing strategy"
    insight = "No strong learning signal is available yet."

    if any(term in outcome_status for term in positive_terms):
        learning_label = "worked"
        recommended_adjustment = (
            f"keep using {recommendation_type or 'this action'} when similar signals appear"
        )
        insight = (
            f"{recommendation_type or 'Recommendation'} appears to have worked"
            + (f" after action '{action_taken}'" if action_taken else "")
            + "."
        )
    elif any(term in outcome_status for term in negative_terms):
        learning_label = "failed"
        recommended_adjustment = (
            f"avoid repeating {recommendation_type or 'this action'} in the same context"
        )
        insight = (
            f"{recommendation_type or 'Recommendation'} appears to have worked poorly"
            + (f" after action '{action_taken}'" if action_taken else "")
            + "."
        )

    if outcome_note:
        insight = f"{insight} Outcome note: {outcome_note}"

    return {
        "recommendation_type": recommendation_type,
        "product_id": row.get("product_id", ""),
        "product_name": row.get("product_name", ""),
        "store_id": row.get("store_id", ""),
        "action_taken": action_taken,
        "outcome_status": row.get("outcome_status", ""),
        "learning_label": learning_label,
        "insight": insight,
        "recommended_adjustment": recommended_adjustment,
        "source": "heuristic",
    }


def build_learning_insights(save_output: bool = True) -> pd.DataFrame:
    """Analyze outcomes and decisions and save compact learning insights."""
    feedback = _combined_feedback_rows()
    if feedback.empty:
        empty_df = pd.DataFrame(columns=LEARNING_INSIGHT_COLUMNS)
        if save_output:
            _write_learning_insights(empty_df)
        return empty_df

    candidate_rows = feedback[
        (feedback["decision"] != "") | (feedback["outcome_status"] != "")
    ].copy()
    if candidate_rows.empty:
        empty_df = pd.DataFrame(columns=LEARNING_INSIGHT_COLUMNS)
        if save_output:
            _write_learning_insights(empty_df)
        return empty_df

    heuristic_insights = [
        _heuristic_learning_insight(row)
        for _, row in candidate_rows.iterrows()
    ]
    insights_df = pd.DataFrame(heuristic_insights)

    llm_insights = summarize_learning_feedback(
        candidate_rows.head(30).to_dict(orient="records")
    )
    if llm_insights:
        llm_df = pd.DataFrame(llm_insights)
        if not llm_df.empty:
            llm_df["source"] = "llm"
            insights_df = pd.concat([insights_df, llm_df], ignore_index=True)

    if insights_df.empty:
        insights_df = pd.DataFrame(columns=LEARNING_INSIGHT_COLUMNS)
    else:
        insights_df["recorded_at"] = pd.Timestamp.now().isoformat(timespec="seconds")
        insights_df["insight_id"] = [
            f"LEARN{str(index + 1).zfill(6)}"
            for index in range(len(insights_df))
        ]
        for column in LEARNING_INSIGHT_COLUMNS:
            if column not in insights_df.columns:
                insights_df[column] = ""
        insights_df = insights_df[LEARNING_INSIGHT_COLUMNS]

    if save_output:
        _write_learning_insights(insights_df)

    return insights_df


def load_learning_insights() -> pd.DataFrame:
    return _read_learning_insights()


def get_learning_context(
    product_id: str = "",
    store_id: str = "",
    recommendation_type: str = "",
    limit: int = 5,
) -> dict:
    """Return memory plus learning signals for a given product/store/type."""
    memory_context = get_memory_context(
        product_id=product_id,
        store_id=store_id,
        recommendation_type=recommendation_type,
        limit=limit,
    )
    learning_df = load_learning_insights()
    if not learning_df.empty:
        if product_id:
            learning_df = learning_df[
                learning_df["product_id"].astype(str) == str(product_id)
            ]
        if store_id:
            learning_df = learning_df[
                learning_df["store_id"].astype(str) == str(store_id)
            ]
        if recommendation_type:
            learning_df = learning_df[
                learning_df["recommendation_type"].astype(str)
                == str(recommendation_type)
            ]

    learning_df = learning_df.head(limit)
    learning_hint = ""
    if not learning_df.empty:
        latest = learning_df.iloc[0]
        learning_hint = str(latest.get("insight", "")).strip()
        adjustment = str(latest.get("recommended_adjustment", "")).strip()
        if adjustment:
            learning_hint = (
                f"{learning_hint} Recommended adjustment: {adjustment}".strip()
            )

    memory_context["recent_learning_insights"] = learning_df.to_dict(orient="records")
    memory_context["learning_hint"] = learning_hint
    return memory_context


def get_system_learning_summary(limit: int = 10) -> dict:
    """Return a compact global learning summary for the orchestrator."""
    learning_df = load_learning_insights()
    if learning_df.empty:
        return {
            "learning_insight_count": 0,
            "learning_by_label": {},
            "recent_learning_insights": [],
        }

    return {
        "learning_insight_count": int(len(learning_df)),
        "learning_by_label": learning_df["learning_label"].astype(str).value_counts().to_dict(),
        "recent_learning_insights": learning_df.head(limit).to_dict(orient="records"),
    }
