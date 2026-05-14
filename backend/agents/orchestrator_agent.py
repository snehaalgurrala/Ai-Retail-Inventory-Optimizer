from pathlib import Path
import sys
from datetime import datetime
from time import perf_counter
from typing import TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from backend.agents import pricing_agent, risk_agent, transfer_agent
from backend.agents.procurement_agent import run as run_procurement_agent
from backend.memory.memory_store import (
    get_system_memory_summary,
    save_recommendation_batch,
)
from backend.memory.learning_loop import (
    build_learning_insights,
    get_system_learning_summary,
)
from backend.services.data_processor import build_processed_datasets
from backend.services.agent_summary_service import generate_agent_card_summaries
from backend.services.inventory_analyzer import build_inventory_analysis
from backend.services.recommendation_engine import (
    PROCESSED_DATA_DIR,
    RECOMMENDATION_COLUMNS,
    get_config,
    load_recommendation_inputs,
)


class AgentGraphState(TypedDict, total=False):
    config: dict
    save_output: bool
    inputs: dict[str, pd.DataFrame]
    data_summary: dict
    memory_context: dict
    learning_context: dict
    inventory_output: dict
    pricing_output: dict
    transfer_output: dict
    risk_output: dict
    procurement_output: dict
    recommendations_df: pd.DataFrame
    agent_outputs_df: pd.DataFrame
    orchestrator_summary_df: pd.DataFrame
    combined_output: dict
    unified_recommendations: list[dict]
    timing_log: dict[str, float]
    agent_analysis_started: float
    graph_started: float
    inventory_agent_seconds: float
    pricing_agent_seconds: float
    transfer_agent_seconds: float
    risk_agent_seconds: float
    procurement_agent_seconds: float
    inventory_agent_error: str
    pricing_agent_error: str
    transfer_agent_error: str
    risk_agent_error: str
    procurement_agent_error: str


AGENT_OUTPUT_COLUMNS = [
    "run_time",
    "agent_name",
    "finding_count",
    "priority_level",
    "latest_insight",
]

ORCHESTRATOR_SUMMARY_COLUMNS = [
    "run_time",
    "database_health",
    "total_recommendations",
    "high_priority_alerts",
    "last_agent_run_time",
    "summary",
    "inventory_agent_seconds",
    "pricing_agent_seconds",
    "transfer_agent_seconds",
    "risk_agent_seconds",
    "procurement_agent_seconds",
    "total_graph_seconds",
]


def _standardize_recommendations(recommendations: list[dict]) -> pd.DataFrame:
    """Return one clean dataframe with the recommendation schema."""
    if not recommendations:
        return pd.DataFrame(columns=RECOMMENDATION_COLUMNS)

    recommendations_df = pd.DataFrame(recommendations)

    for column in RECOMMENDATION_COLUMNS:
        if column not in recommendations_df.columns:
            recommendations_df[column] = ""

    recommendations_df["recommendation_id"] = [
        f"REC{str(index + 1).zfill(5)}"
        for index in range(len(recommendations_df))
    ]
    recommendations_df["status"] = recommendations_df["status"].replace("", "pending")
    recommendations_df["status"] = recommendations_df["status"].fillna("pending")

    return recommendations_df[RECOMMENDATION_COLUMNS]


def save_recommendations(recommendations_df: pd.DataFrame) -> Path:
    """Save standardized recommendations for the Streamlit page."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DATA_DIR / "recommendations.csv"
    recommendations_df.to_csv(output_path, index=False)
    return output_path


def _database_health_label() -> str:
    """Build a simple health label from expected raw and processed files."""
    required_paths = [
        PROJECT_ROOT / "data" / "raw" / "inventory.csv",
        PROJECT_ROOT / "data" / "raw" / "products.csv",
        PROJECT_ROOT / "data" / "raw" / "sales.csv",
        PROJECT_ROOT / "data" / "raw" / "stores.csv",
        PROJECT_ROOT / "data" / "raw" / "suppliers.csv",
        PROJECT_ROOT / "data" / "raw" / "transactions.csv",
        PROCESSED_DATA_DIR / "low_stock_items.csv",
        PROCESSED_DATA_DIR / "stockout_risk_items.csv",
        PROCESSED_DATA_DIR / "overstock_items.csv",
        PROCESSED_DATA_DIR / "dead_stock_candidates.csv",
        PROCESSED_DATA_DIR / "high_demand_items.csv",
        PROCESSED_DATA_DIR / "slow_moving_items.csv",
    ]
    available_count = sum(path.exists() for path in required_paths)
    total_count = len(required_paths)

    if available_count == total_count:
        return f"Healthy ({available_count}/{total_count})"
    if available_count >= total_count - 2:
        return f"Watch ({available_count}/{total_count})"
    return f"Needs Attention ({available_count}/{total_count})"


def _highest_priority_label(recommendations: list[dict]) -> str:
    """Return the highest priority present in a recommendation list."""
    if not recommendations:
        return "info"

    priorities = pd.Series(
        [str(row.get("priority", "")).lower() for row in recommendations]
    )
    if priorities.eq("high").any():
        return "high"
    if priorities.eq("medium").any():
        return "medium"
    if priorities.eq("low").any():
        return "low"
    return "info"


def _first_non_empty_text(recommendations: list[dict], field_name: str) -> str:
    """Return the first non-empty text field from recommendation rows."""
    for recommendation in recommendations:
        value = str(recommendation.get(field_name, "")).strip()
        if value:
            return value
    return ""


def _build_agent_outputs_df(state: AgentGraphState, run_time: str) -> pd.DataFrame:
    """Create one dashboard-friendly row per agent."""
    inventory_output = state.get("inventory_output", {})
    pricing_output = state.get("pricing_output", {})
    transfer_output = state.get("transfer_output", {})
    risk_output = state.get("risk_output", {})
    procurement_output = state.get("procurement_output", {})

    inventory_counts = [
        int(inventory_output.get("low_stock_count", 0)),
        int(inventory_output.get("stockout_risk_count", 0)),
        int(inventory_output.get("overstock_count", 0)),
        int(inventory_output.get("high_demand_count", 0)),
    ]
    inventory_finding_count = sum(inventory_counts)
    inventory_priority = (
        "high"
        if (
            int(inventory_output.get("stockout_risk_count", 0)) > 0
            or int(inventory_output.get("low_stock_count", 0)) > 0
        )
        else "medium" if inventory_finding_count > 0 else "info"
    )

    inventory_insight_parts = []
    if int(inventory_output.get("high_demand_count", 0)) > 0:
        inventory_insight_parts.append(
            f"{inventory_output.get('high_demand_count', 0)} high demand items"
        )
    if int(inventory_output.get("low_stock_count", 0)) > 0:
        inventory_insight_parts.append(
            f"{inventory_output.get('low_stock_count', 0)} low stock items"
        )
    if int(inventory_output.get("stockout_risk_count", 0)) > 0:
        inventory_insight_parts.append(
            f"{inventory_output.get('stockout_risk_count', 0)} stockout risks"
        )
    inventory_latest_insight = (
        ". ".join(inventory_insight_parts[:3]) + "."
        if inventory_insight_parts
        else "Run agents to generate latest analysis."
    )

    agent_rows = [
        {
            "run_time": run_time,
            "agent_name": "inventory_agent",
            "finding_count": inventory_finding_count,
            "priority_level": inventory_priority,
            "latest_insight": inventory_latest_insight,
        }
    ]

    for output in [
        pricing_output,
        transfer_output,
        risk_output,
        procurement_output,
    ]:
        recommendations = output.get("recommendations", [])
        error = str(output.get("error", "") or "").strip()
        agent_rows.append(
            {
                "run_time": run_time,
                "agent_name": output.get("agent_name", ""),
                "finding_count": int(output.get("recommendation_count", 0)),
                "priority_level": "error" if error else _highest_priority_label(recommendations),
                "latest_insight": (
                    f"Agent failed: {error}"
                    if error
                    else
                    _first_non_empty_text(recommendations, "reason")
                    or _first_non_empty_text(recommendations, "action")
                    or "Run agents to generate latest analysis."
                ),
            }
        )

    return pd.DataFrame(agent_rows, columns=AGENT_OUTPUT_COLUMNS)


def _build_orchestrator_summary_df(
    state: AgentGraphState,
    recommendations_df: pd.DataFrame,
    run_time: str,
) -> pd.DataFrame:
    """Create a one-row orchestrator summary dataset for the dashboard."""
    inventory_output = state.get("inventory_output", {})
    summary_parts = []
    if int(inventory_output.get("low_stock_count", 0)) > 0:
        summary_parts.append(
            f"{inventory_output.get('low_stock_count', 0)} low stock items"
        )
    if int(inventory_output.get("stockout_risk_count", 0)) > 0:
        summary_parts.append(
            f"{inventory_output.get('stockout_risk_count', 0)} stockout risks"
        )
    if int(inventory_output.get("overstock_count", 0)) > 0:
        summary_parts.append(
            f"{inventory_output.get('overstock_count', 0)} overstock rows"
        )
    if int(inventory_output.get("high_demand_count", 0)) > 0:
        summary_parts.append(
            f"{inventory_output.get('high_demand_count', 0)} high demand items"
        )
    if len(recommendations_df) > 0:
        summary_parts.append(f"{len(recommendations_df)} active recommendations")

    summary_text = (
        ". ".join(summary_parts) + "."
        if summary_parts
        else "Run agents to generate latest analysis."
    )
    high_priority_alerts = 0
    if "priority" in recommendations_df.columns:
        high_priority_alerts = int(
            recommendations_df["priority"].fillna("").astype(str).str.lower().eq("high").sum()
        )

    summary_row = {
        "run_time": run_time,
        "database_health": _database_health_label(),
        "total_recommendations": int(len(recommendations_df)),
        "high_priority_alerts": high_priority_alerts,
        "last_agent_run_time": run_time,
        "summary": summary_text,
        "inventory_agent_seconds": float(state.get("inventory_agent_seconds", 0)),
        "pricing_agent_seconds": float(state.get("pricing_agent_seconds", 0)),
        "transfer_agent_seconds": float(state.get("transfer_agent_seconds", 0)),
        "risk_agent_seconds": float(state.get("risk_agent_seconds", 0)),
        "procurement_agent_seconds": float(state.get("procurement_agent_seconds", 0)),
        "total_graph_seconds": float(state.get("total_graph_seconds", 0)),
    }
    return pd.DataFrame([summary_row], columns=ORCHESTRATOR_SUMMARY_COLUMNS)


def save_agent_outputs(agent_outputs_df: pd.DataFrame) -> Path:
    """Save per-agent dashboard output rows."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DATA_DIR / "agent_outputs.csv"
    agent_outputs_df.to_csv(output_path, index=False)
    return output_path


def save_orchestrator_summary(orchestrator_summary_df: pd.DataFrame) -> Path:
    """Save the latest orchestrator summary row."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DATA_DIR / "orchestrator_summary.csv"
    orchestrator_summary_df.to_csv(output_path, index=False)
    return output_path


def _load_latest_data_summary_node(state: AgentGraphState) -> AgentGraphState:
    """Load fresh processed data and build a shared summary for the graph run."""
    started = perf_counter()
    config = get_config(state.get("config"))

    build_processed_datasets()
    build_inventory_analysis(config)
    inputs = load_recommendation_inputs()
    memory_context = get_system_memory_summary()
    build_learning_insights(save_output=True, use_llm=False)
    learning_context = get_system_learning_summary()
    inputs["memory_context"] = memory_context
    inputs["learning_context"] = learning_context
    data_summary = {
        "current_inventory_rows": int(len(inputs.get("current_inventory", pd.DataFrame()))),
        "product_performance_rows": int(len(inputs.get("product_performance", pd.DataFrame()))),
        "low_stock_rows": int(len(inputs.get("low_stock_items", pd.DataFrame()))),
        "stockout_risk_rows": int(len(inputs.get("stockout_risk_items", pd.DataFrame()))),
        "overstock_rows": int(len(inputs.get("overstock_items", pd.DataFrame()))),
        "dead_stock_rows": int(len(inputs.get("dead_stock_candidates", pd.DataFrame()))),
        "high_demand_rows": int(len(inputs.get("high_demand_items", pd.DataFrame()))),
        "slow_moving_rows": int(len(inputs.get("slow_moving_items", pd.DataFrame()))),
    }

    timing_log = dict(state.get("timing_log", {}))
    timing_log["data_analysis_seconds"] = round(perf_counter() - started, 3)
    print(f"[agent_refresh] data analysis time: {timing_log['data_analysis_seconds']}s")

    return {
        "config": config,
        "save_output": state.get("save_output", True),
        "inputs": inputs,
        "data_summary": data_summary,
        "memory_context": memory_context,
        "learning_context": learning_context,
        "timing_log": timing_log,
        "agent_analysis_started": perf_counter(),
        "graph_started": state.get("graph_started", perf_counter()),
    }


def _inventory_agent_node(state: AgentGraphState) -> AgentGraphState:
    """Summarize inventory and demand signals for the graph state."""
    started = perf_counter()
    try:
        inputs = state.get("inputs", {})
        low_stock_items = inputs.get("low_stock_items", pd.DataFrame())
        stockout_risk_items = inputs.get("stockout_risk_items", pd.DataFrame())
        overstock_items = inputs.get("overstock_items", pd.DataFrame())
        high_demand_items = inputs.get("high_demand_items", pd.DataFrame())

        finding_count = (
            int(len(low_stock_items))
            + int(len(stockout_risk_items))
            + int(len(overstock_items))
            + int(len(high_demand_items))
        )
        inventory_output = {
            "agent_name": "inventory_agent",
            "recommendations": [],
            "finding_count": finding_count,
            "low_stock_count": int(len(low_stock_items)),
            "stockout_risk_count": int(len(stockout_risk_items)),
            "overstock_count": int(len(overstock_items)),
            "high_demand_count": int(len(high_demand_items)),
            "tool_context": {
                "inventory_rows": int(len(inputs.get("current_inventory", pd.DataFrame()))),
                "low_stock_rows": int(len(low_stock_items)),
                "stockout_risk_rows": int(len(stockout_risk_items)),
                "overstock_rows": int(len(overstock_items)),
            },
            "focus_product_ids": sorted(
                {
                    str(product_id)
                    for product_id in pd.concat(
                        [
                            low_stock_items.get("product_id", pd.Series(dtype=object)),
                            high_demand_items.get("product_id", pd.Series(dtype=object)),
                        ],
                        ignore_index=True,
                    )
                    .dropna()
                    .astype(str)
                    if product_id
                }
            ),
        }
        error = ""
    except Exception as exc:
        error = str(exc)
        inventory_output = {
            "agent_name": "inventory_agent",
            "recommendations": [],
            "finding_count": 0,
            "low_stock_count": 0,
            "stockout_risk_count": 0,
            "overstock_count": 0,
            "high_demand_count": 0,
            "error": error,
        }

    elapsed = round(perf_counter() - started, 3)
    print(f"[agent_refresh] inventory_agent time: {elapsed}s")
    if error:
        print(f"[agent_refresh] inventory_agent error: {error}")
    return {
        "inventory_output": inventory_output,
        "inventory_agent_seconds": elapsed,
        "inventory_agent_error": error,
    }


def _pricing_agent_node(state: AgentGraphState) -> AgentGraphState:
    """Run pricing recommendations from shared graph inputs."""
    started = perf_counter()
    error = ""
    try:
        recommendations = pricing_agent.run(
            state.get("inputs", {}),
            state.get("config"),
        )
    except Exception as exc:
        error = str(exc)
        recommendations = []
    elapsed = round(perf_counter() - started, 3)
    print(f"[agent_refresh] pricing_agent time: {elapsed}s")
    if error:
        print(f"[agent_refresh] pricing_agent error: {error}")
    return {
        "pricing_output": {
            "agent_name": "pricing_agent",
            "recommendations": recommendations,
            "recommendation_count": len(recommendations),
            "error": error,
        },
        "pricing_agent_seconds": elapsed,
        "pricing_agent_error": error,
    }


def _transfer_agent_node(state: AgentGraphState) -> AgentGraphState:
    """Run transfer recommendations from shared graph inputs."""
    started = perf_counter()
    error = ""
    try:
        recommendations = transfer_agent.run(
            state.get("inputs", {}),
            state.get("config"),
        )
    except Exception as exc:
        error = str(exc)
        recommendations = []
    elapsed = round(perf_counter() - started, 3)
    print(f"[agent_refresh] transfer_agent time: {elapsed}s")
    if error:
        print(f"[agent_refresh] transfer_agent error: {error}")
    return {
        "transfer_output": {
            "agent_name": "transfer_agent",
            "recommendations": recommendations,
            "recommendation_count": len(recommendations),
            "error": error,
        },
        "transfer_agent_seconds": elapsed,
        "transfer_agent_error": error,
    }


def _risk_agent_node(state: AgentGraphState) -> AgentGraphState:
    """Run risk recommendations from shared graph inputs."""
    started = perf_counter()
    error = ""
    try:
        recommendations = risk_agent.run(
            state.get("inputs", {}),
            state.get("config"),
        )
    except Exception as exc:
        error = str(exc)
        recommendations = []
    elapsed = round(perf_counter() - started, 3)
    print(f"[agent_refresh] risk_agent time: {elapsed}s")
    if error:
        print(f"[agent_refresh] risk_agent error: {error}")
    return {
        "risk_output": {
            "agent_name": "risk_agent",
            "recommendations": recommendations,
            "recommendation_count": len(recommendations),
            "error": error,
        },
        "risk_agent_seconds": elapsed,
        "risk_agent_error": error,
    }


def _procurement_agent_node(state: AgentGraphState) -> AgentGraphState:
    """Run procurement recommendations from shared graph inputs."""
    started = perf_counter()
    error = ""
    try:
        recommendations = run_procurement_agent(
            state.get("inputs", {}),
            state.get("config"),
        )
    except Exception as exc:
        error = str(exc)
        recommendations = []
    elapsed = round(perf_counter() - started, 3)
    print(f"[agent_refresh] procurement_agent time: {elapsed}s")
    if error:
        print(f"[agent_refresh] procurement_agent error: {error}")
    return {
        "procurement_output": {
            "agent_name": "procurement_agent",
            "recommendations": recommendations,
            "recommendation_count": len(recommendations),
            "error": error,
        },
        "procurement_agent_seconds": elapsed,
        "procurement_agent_error": error,
    }


def _combine_results_node(state: AgentGraphState) -> AgentGraphState:
    """Merge parallel agent outputs into one orchestrator summary."""
    started = perf_counter()
    agent_output_keys = [
        "inventory_output",
        "pricing_output",
        "transfer_output",
        "risk_output",
        "procurement_output",
    ]

    recommendations: list[dict] = []
    agent_counts: dict[str, int] = {}
    agent_errors: dict[str, str] = {}
    for key in agent_output_keys:
        agent_output = state.get(key, {})
        agent_name = agent_output.get("agent_name", key)
        agent_recommendations = agent_output.get("recommendations", [])
        error = str(agent_output.get("error", "") or "").strip()
        if error:
            agent_errors[agent_name] = error
        if key == "inventory_output":
            agent_counts[agent_name] = int(agent_output.get("finding_count", 0))
        elif isinstance(agent_recommendations, list):
            recommendations.extend(agent_recommendations)
            agent_counts[agent_name] = len(agent_recommendations)
        else:
            agent_counts[agent_name] = 0

    recommendations_df = _standardize_recommendations(recommendations)
    run_time = datetime.now().isoformat(timespec="seconds")
    agent_outputs_df = _build_agent_outputs_df(state, run_time)
    orchestrator_summary_df = _build_orchestrator_summary_df(
        state,
        recommendations_df,
        run_time,
    )

    combined_output = {
        "agent_name": "orchestrator",
        "data_summary": state.get("data_summary", {}),
        "inventory_summary": state.get("inventory_output", {}),
        "memory_context": state.get("memory_context", {}),
        "learning_context": state.get("learning_context", {}),
        "agent_counts": agent_counts,
        "agent_errors": agent_errors,
        "total_recommendations": int(len(recommendations_df)),
        "run_time": run_time,
    }
    timing_log = dict(state.get("timing_log", {}))
    agent_started = float(state.get("agent_analysis_started", started))
    for timing_key in [
        "inventory_agent_seconds",
        "pricing_agent_seconds",
        "transfer_agent_seconds",
        "risk_agent_seconds",
        "procurement_agent_seconds",
    ]:
        timing_log[timing_key] = float(state.get(timing_key, 0))
    timing_log["agent_analysis_seconds"] = round(perf_counter() - agent_started, 3)
    print(f"[agent_refresh] agent analysis time: {timing_log['agent_analysis_seconds']}s")

    return {
        "recommendations_df": recommendations_df,
        "agent_outputs_df": agent_outputs_df,
        "orchestrator_summary_df": orchestrator_summary_df,
        "combined_output": combined_output,
        "unified_recommendations": recommendations_df.to_dict(orient="records"),
        "timing_log": timing_log,
    }


def _save_outputs_node(state: AgentGraphState) -> AgentGraphState:
    """Persist the final orchestrator outputs to processed CSV files."""
    started = perf_counter()
    timing_log = dict(state.get("timing_log", {}))
    graph_started = float(state.get("graph_started", started))
    timing_log["total_graph_seconds"] = round(perf_counter() - graph_started, 3)
    recommendations_df = state.get(
        "recommendations_df",
        pd.DataFrame(columns=RECOMMENDATION_COLUMNS),
    )
    agent_outputs_df = state.get(
        "agent_outputs_df",
        pd.DataFrame(columns=AGENT_OUTPUT_COLUMNS),
    )
    orchestrator_summary_df = state.get(
        "orchestrator_summary_df",
        pd.DataFrame(columns=ORCHESTRATOR_SUMMARY_COLUMNS),
    )
    if not orchestrator_summary_df.empty:
        for timing_key in [
            "inventory_agent_seconds",
            "pricing_agent_seconds",
            "transfer_agent_seconds",
            "risk_agent_seconds",
            "procurement_agent_seconds",
            "total_graph_seconds",
        ]:
            if timing_key not in orchestrator_summary_df.columns:
                orchestrator_summary_df[timing_key] = 0.0
            orchestrator_summary_df.loc[:, timing_key] = timing_log.get(timing_key, 0)

    output_path = ""
    agent_outputs_path = ""
    orchestrator_summary_path = ""
    if state.get("save_output", True):
        output_path = str(save_recommendations(recommendations_df))
        agent_outputs_path = str(save_agent_outputs(agent_outputs_df))
        orchestrator_summary_path = str(
            save_orchestrator_summary(orchestrator_summary_df)
        )
        summary_started = perf_counter()
        card_summaries_df, orchestrator_summary_df = generate_agent_card_summaries(
            agent_outputs_df=agent_outputs_df,
            recommendations_df=recommendations_df,
            orchestrator_summary_df=orchestrator_summary_df,
            save_output=True,
        )
        save_recommendation_batch(recommendations_df)
        timing_log["llm_summary_seconds"] = round(perf_counter() - summary_started, 3)
        timing_log["llm_seconds"] = timing_log["llm_summary_seconds"]

    combined_output = state.get("combined_output", {}).copy()
    combined_output["output_path"] = output_path
    combined_output["agent_outputs_path"] = agent_outputs_path
    combined_output["orchestrator_summary_path"] = orchestrator_summary_path
    timing_log["file_save_seconds"] = round(perf_counter() - started, 3)
    timing_log["total_graph_seconds"] = round(perf_counter() - graph_started, 3)
    if not orchestrator_summary_df.empty:
        orchestrator_summary_df.loc[:, "total_graph_seconds"] = timing_log["total_graph_seconds"]
        if state.get("save_output", True):
            orchestrator_summary_path = str(
                save_orchestrator_summary(orchestrator_summary_df)
            )
            combined_output["orchestrator_summary_path"] = orchestrator_summary_path
    combined_output["timing_log"] = timing_log
    print(f"[agent_refresh] LLM summary time: {timing_log.get('llm_summary_seconds', 0)}s")
    print(f"[agent_refresh] file save time: {timing_log['file_save_seconds']}s")

    return {
        "combined_output": combined_output,
        "orchestrator_summary_df": orchestrator_summary_df,
        "timing_log": timing_log,
    }


def build_orchestrator_graph():
    """Create the LangGraph workflow with parallel agent fan-out/fan-in."""
    graph = StateGraph(AgentGraphState)

    graph.add_node("load_latest_data_summary", _load_latest_data_summary_node)
    graph.add_node("inventory_agent", _inventory_agent_node)
    graph.add_node("pricing_agent", _pricing_agent_node)
    graph.add_node("transfer_agent", _transfer_agent_node)
    graph.add_node("risk_agent", _risk_agent_node)
    graph.add_node("procurement_agent", _procurement_agent_node)
    graph.add_node("combine_results", _combine_results_node)
    graph.add_node("save_outputs", _save_outputs_node)

    graph.add_edge(START, "load_latest_data_summary")
    graph.add_edge("load_latest_data_summary", "inventory_agent")
    graph.add_edge("load_latest_data_summary", "pricing_agent")
    graph.add_edge("load_latest_data_summary", "transfer_agent")
    graph.add_edge("load_latest_data_summary", "risk_agent")
    graph.add_edge("load_latest_data_summary", "procurement_agent")
    graph.add_edge(
        [
            "inventory_agent",
            "pricing_agent",
            "transfer_agent",
            "risk_agent",
            "procurement_agent",
        ],
        "combine_results",
    )
    graph.add_edge("combine_results", "save_outputs")
    graph.add_edge("save_outputs", END)

    return graph.compile()


def run_agent_graph(
    config: dict | None = None,
    save_output: bool = True,
) -> AgentGraphState:
    """Run the LangGraph workflow and return the final shared state."""
    started = perf_counter()
    orchestrator_graph = build_orchestrator_graph()
    initial_state: AgentGraphState = {
        "config": get_config(config),
        "save_output": save_output,
        "graph_started": started,
    }
    final_state = orchestrator_graph.invoke(initial_state)
    timing_log = dict(final_state.get("timing_log", {}))
    timing_log["total_refresh_seconds"] = round(perf_counter() - started, 3)
    timing_log["total_graph_seconds"] = timing_log["total_refresh_seconds"]
    timing_log["total_seconds"] = timing_log["total_refresh_seconds"]
    combined_output = dict(final_state.get("combined_output", {}))
    combined_output["timing_log"] = timing_log
    final_state["combined_output"] = combined_output
    final_state["timing_log"] = timing_log
    print(f"[agent_refresh] total refresh time: {timing_log['total_refresh_seconds']}s")
    return final_state


def run_all_agents(
    config: dict | None = None,
    save_output: bool = True,
) -> pd.DataFrame:
    """Run the LangGraph orchestrator and return the unified recommendations."""
    final_state = run_agent_graph(config=config, save_output=save_output)
    recommendation_records = final_state.get("unified_recommendations", [])

    if not recommendation_records:
        return pd.DataFrame(columns=RECOMMENDATION_COLUMNS)

    recommendations_df = pd.DataFrame(recommendation_records)
    for column in RECOMMENDATION_COLUMNS:
        if column not in recommendations_df.columns:
            recommendations_df[column] = ""

    return recommendations_df[RECOMMENDATION_COLUMNS]


if __name__ == "__main__":
    run_all_agents()
