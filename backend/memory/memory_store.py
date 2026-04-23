from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MEMORY_DIR = PROJECT_ROOT / "data" / "processed" / "memory"
RECOMMENDATION_MEMORY_FILE = MEMORY_DIR / "recommendation_memory.csv"
DECISION_MEMORY_FILE = MEMORY_DIR / "decision_memory.csv"
OUTCOME_MEMORY_FILE = MEMORY_DIR / "outcome_memory.csv"


RECOMMENDATION_MEMORY_COLUMNS = [
    "memory_id",
    "recorded_at",
    "recommendation_id",
    "recommendation_type",
    "product_id",
    "product_name",
    "store_id",
    "priority",
    "action",
    "reason",
    "evidence",
    "suggested_quantity",
    "source_agent",
    "status",
]

DECISION_MEMORY_COLUMNS = [
    "memory_id",
    "recorded_at",
    "recommendation_id",
    "recommendation_type",
    "product_id",
    "product_name",
    "store_id",
    "decision",
    "source_agent",
]

OUTCOME_MEMORY_COLUMNS = [
    "memory_id",
    "recorded_at",
    "recommendation_id",
    "recommendation_type",
    "product_id",
    "product_name",
    "store_id",
    "action_taken",
    "outcome_status",
    "outcome_note",
]


def _ensure_memory_dir() -> None:
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)


def _read_memory_csv(file_path: Path, columns: list[str]) -> pd.DataFrame:
    """Read one memory CSV safely."""
    _ensure_memory_dir()
    if not file_path.exists():
        return pd.DataFrame(columns=columns)

    try:
        df = pd.read_csv(file_path)
    except Exception:
        return pd.DataFrame(columns=columns)

    for column in columns:
        if column not in df.columns:
            df[column] = ""

    return df[columns].copy()


def _write_memory_csv(df: pd.DataFrame, file_path: Path, columns: list[str]) -> None:
    """Write one memory CSV with the expected schema."""
    _ensure_memory_dir()
    writable_df = df.copy()
    for column in columns:
        if column not in writable_df.columns:
            writable_df[column] = ""
    writable_df[columns].to_csv(file_path, index=False)


def _next_memory_id(prefix: str, existing_df: pd.DataFrame) -> str:
    """Generate a simple sequential memory record id."""
    return f"{prefix}{str(len(existing_df) + 1).zfill(6)}"


def load_recommendation_memory() -> pd.DataFrame:
    return _read_memory_csv(
        RECOMMENDATION_MEMORY_FILE,
        RECOMMENDATION_MEMORY_COLUMNS,
    )


def load_decision_memory() -> pd.DataFrame:
    return _read_memory_csv(
        DECISION_MEMORY_FILE,
        DECISION_MEMORY_COLUMNS,
    )


def load_outcome_memory() -> pd.DataFrame:
    return _read_memory_csv(
        OUTCOME_MEMORY_FILE,
        OUTCOME_MEMORY_COLUMNS,
    )


def save_recommendation_batch(recommendations_df: pd.DataFrame) -> Path:
    """Append the latest recommendation batch to recommendation memory."""
    existing_df = load_recommendation_memory()
    if recommendations_df.empty:
        _write_memory_csv(existing_df, RECOMMENDATION_MEMORY_FILE, RECOMMENDATION_MEMORY_COLUMNS)
        return RECOMMENDATION_MEMORY_FILE

    recorded_at = datetime.now().isoformat(timespec="seconds")
    new_rows = recommendations_df.copy()
    for column in RECOMMENDATION_MEMORY_COLUMNS:
        if column not in new_rows.columns:
            new_rows[column] = ""

    new_rows["recorded_at"] = recorded_at
    base_count = len(existing_df)
    new_rows["memory_id"] = [
        f"MEMREC{str(base_count + index + 1).zfill(6)}"
        for index in range(len(new_rows))
    ]

    combined_df = pd.concat([existing_df, new_rows], ignore_index=True)
    _write_memory_csv(
        combined_df,
        RECOMMENDATION_MEMORY_FILE,
        RECOMMENDATION_MEMORY_COLUMNS,
    )
    return RECOMMENDATION_MEMORY_FILE


def save_decision_record(
    recommendation: pd.Series | dict,
    decision: str,
    decided_at: str | None = None,
) -> Path:
    """Append one approved or rejected decision to memory."""
    existing_df = load_decision_memory()
    recommendation_data = (
        recommendation.to_dict()
        if isinstance(recommendation, pd.Series)
        else dict(recommendation)
    )
    record = {
        "memory_id": _next_memory_id("MEMDEC", existing_df),
        "recorded_at": decided_at or datetime.now().isoformat(timespec="seconds"),
        "recommendation_id": recommendation_data.get("recommendation_id", ""),
        "recommendation_type": recommendation_data.get("recommendation_type", ""),
        "product_id": recommendation_data.get("product_id", ""),
        "product_name": recommendation_data.get("product_name", ""),
        "store_id": recommendation_data.get("store_id", ""),
        "decision": decision,
        "source_agent": recommendation_data.get("source_agent", ""),
    }

    combined_df = pd.concat([existing_df, pd.DataFrame([record])], ignore_index=True)
    _write_memory_csv(combined_df, DECISION_MEMORY_FILE, DECISION_MEMORY_COLUMNS)
    return DECISION_MEMORY_FILE


def save_outcome_record(outcome: dict) -> Path:
    """Append one outcome record when outcome data is available."""
    existing_df = load_outcome_memory()
    record = {
        "memory_id": _next_memory_id("MEMOUT", existing_df),
        "recorded_at": outcome.get(
            "recorded_at",
            datetime.now().isoformat(timespec="seconds"),
        ),
        "recommendation_id": outcome.get("recommendation_id", ""),
        "recommendation_type": outcome.get("recommendation_type", ""),
        "product_id": outcome.get("product_id", ""),
        "product_name": outcome.get("product_name", ""),
        "store_id": outcome.get("store_id", ""),
        "action_taken": outcome.get("action_taken", ""),
        "outcome_status": outcome.get("outcome_status", ""),
        "outcome_note": outcome.get("outcome_note", ""),
    }

    combined_df = pd.concat([existing_df, pd.DataFrame([record])], ignore_index=True)
    _write_memory_csv(combined_df, OUTCOME_MEMORY_FILE, OUTCOME_MEMORY_COLUMNS)
    return OUTCOME_MEMORY_FILE


def _filter_memory(
    df: pd.DataFrame,
    product_id: str = "",
    store_id: str = "",
    recommendation_type: str = "",
) -> pd.DataFrame:
    """Filter a memory dataframe by product, store, or recommendation type."""
    filtered = df.copy()
    if product_id and "product_id" in filtered.columns:
        filtered = filtered[filtered["product_id"].astype(str) == str(product_id)]
    if store_id and "store_id" in filtered.columns:
        filtered = filtered[filtered["store_id"].astype(str) == str(store_id)]
    if recommendation_type and "recommendation_type" in filtered.columns:
        filtered = filtered[
            filtered["recommendation_type"].astype(str) == str(recommendation_type)
        ]
    return filtered


def get_memory_context(
    product_id: str = "",
    store_id: str = "",
    recommendation_type: str = "",
    limit: int = 5,
) -> dict:
    """Return recent recommendation, decision, and outcome history."""
    recommendation_memory = _filter_memory(
        load_recommendation_memory(),
        product_id,
        store_id,
        recommendation_type,
    )
    decision_memory = _filter_memory(
        load_decision_memory(),
        product_id,
        store_id,
        recommendation_type,
    )
    outcome_memory = _filter_memory(
        load_outcome_memory(),
        product_id,
        store_id,
        recommendation_type,
    )

    recommendation_memory = recommendation_memory.sort_values(
        "recorded_at",
        ascending=False,
    ).head(limit)
    decision_memory = decision_memory.sort_values(
        "recorded_at",
        ascending=False,
    ).head(limit)
    outcome_memory = outcome_memory.sort_values(
        "recorded_at",
        ascending=False,
    ).head(limit)

    decision_summary = {}
    if not decision_memory.empty and "decision" in decision_memory.columns:
        decision_summary = decision_memory["decision"].astype(str).value_counts().to_dict()

    outcome_summary = {}
    if not outcome_memory.empty and "outcome_status" in outcome_memory.columns:
        outcome_summary = (
            outcome_memory["outcome_status"].astype(str).value_counts().to_dict()
        )

    memory_hint_parts = []
    if decision_summary:
        memory_hint_parts.append(
            "decision_history=" + ", ".join(
                f"{key}:{value}" for key, value in decision_summary.items()
            )
        )
    if outcome_summary:
        memory_hint_parts.append(
            "outcome_history=" + ", ".join(
                f"{key}:{value}" for key, value in outcome_summary.items()
            )
        )
    if not outcome_memory.empty:
        latest_outcome = outcome_memory.iloc[0]
        outcome_status = str(latest_outcome.get("outcome_status", "")).strip()
        outcome_note = str(latest_outcome.get("outcome_note", "")).strip()
        if outcome_status or outcome_note:
            memory_hint_parts.append(
                f"latest_outcome={outcome_status or 'unknown'}"
                + (f" ({outcome_note})" if outcome_note else "")
            )
    if not decision_memory.empty:
        latest_decision = decision_memory.iloc[0]
        latest_decision_value = str(latest_decision.get("decision", "")).strip()
        latest_decision_time = str(latest_decision.get("recorded_at", "")).strip()
        if latest_decision_value:
            memory_hint_parts.append(
                f"latest_decision={latest_decision_value}"
                + (f" at {latest_decision_time}" if latest_decision_time else "")
            )

    return {
        "recent_recommendations": recommendation_memory.to_dict(orient="records"),
        "recent_decisions": decision_memory.to_dict(orient="records"),
        "recent_outcomes": outcome_memory.to_dict(orient="records"),
        "decision_summary": decision_summary,
        "outcome_summary": outcome_summary,
        "memory_hint": "; ".join(memory_hint_parts),
    }


def get_system_memory_summary(limit: int = 10) -> dict:
    """Return a compact global memory summary for the orchestrator."""
    recommendation_memory = load_recommendation_memory()
    decision_memory = load_decision_memory()
    outcome_memory = load_outcome_memory()

    return {
        "recommendation_history_count": int(len(recommendation_memory)),
        "decision_history_count": int(len(decision_memory)),
        "outcome_history_count": int(len(outcome_memory)),
        "recent_decisions": decision_memory.sort_values(
            "recorded_at",
            ascending=False,
        ).head(limit).to_dict(orient="records"),
        "recent_outcomes": outcome_memory.sort_values(
            "recorded_at",
            ascending=False,
        ).head(limit).to_dict(orient="records"),
    }
