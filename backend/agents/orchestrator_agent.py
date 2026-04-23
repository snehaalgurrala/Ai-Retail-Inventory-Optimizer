from pathlib import Path
import sys
from typing import TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from backend.agents import pricing_agent, risk_agent, transfer_agent
from backend.agents.procurement_agent import run as run_procurement_agent
from backend.services.data_processor import build_processed_datasets
from backend.services.inventory_analyzer import build_inventory_analysis
from backend.services.llm_reasoner import summarize_orchestration
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
    inventory_output: dict
    pricing_output: dict
    transfer_output: dict
    risk_output: dict
    procurement_output: dict
    combined_output: dict
    unified_recommendations: list[dict]


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


def _orchestrator_node(state: AgentGraphState) -> AgentGraphState:
    """Prepare the shared state used by all downstream agent nodes."""
    config = get_config(state.get("config"))

    build_processed_datasets()
    build_inventory_analysis(config)
    inputs = load_recommendation_inputs()

    return {
        "config": config,
        "save_output": state.get("save_output", True),
        "inputs": inputs,
    }


def _inventory_agent_node(state: AgentGraphState) -> AgentGraphState:
    """Summarize inventory and demand signals for the graph state."""
    inputs = state.get("inputs", {})
    low_stock_items = inputs.get("low_stock_items", pd.DataFrame())
    stockout_risk_items = inputs.get("stockout_risk_items", pd.DataFrame())
    overstock_items = inputs.get("overstock_items", pd.DataFrame())
    high_demand_items = inputs.get("high_demand_items", pd.DataFrame())

    inventory_output = {
        "agent_name": "inventory_agent",
        "low_stock_count": int(len(low_stock_items)),
        "stockout_risk_count": int(len(stockout_risk_items)),
        "overstock_count": int(len(overstock_items)),
        "high_demand_count": int(len(high_demand_items)),
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

    return {"inventory_output": inventory_output}


def _pricing_agent_node(state: AgentGraphState) -> AgentGraphState:
    """Run pricing recommendations from shared graph inputs."""
    recommendations = pricing_agent.run(
        state.get("inputs", {}),
        state.get("config"),
    )
    return {
        "pricing_output": {
            "agent_name": "pricing_agent",
            "recommendations": recommendations,
            "recommendation_count": len(recommendations),
        }
    }


def _transfer_agent_node(state: AgentGraphState) -> AgentGraphState:
    """Run transfer recommendations from shared graph inputs."""
    recommendations = transfer_agent.run(
        state.get("inputs", {}),
        state.get("config"),
    )
    return {
        "transfer_output": {
            "agent_name": "transfer_agent",
            "recommendations": recommendations,
            "recommendation_count": len(recommendations),
        }
    }


def _risk_agent_node(state: AgentGraphState) -> AgentGraphState:
    """Run risk recommendations from shared graph inputs."""
    recommendations = risk_agent.run(
        state.get("inputs", {}),
        state.get("config"),
    )
    return {
        "risk_output": {
            "agent_name": "risk_agent",
            "recommendations": recommendations,
            "recommendation_count": len(recommendations),
        }
    }


def _procurement_agent_node(state: AgentGraphState) -> AgentGraphState:
    """Run procurement recommendations from shared graph inputs."""
    recommendations = run_procurement_agent(
        state.get("inputs", {}),
        state.get("config"),
    )
    return {
        "procurement_output": {
            "agent_name": "procurement_agent",
            "recommendations": recommendations,
            "recommendation_count": len(recommendations),
        }
    }


def _combine_results_node(state: AgentGraphState) -> AgentGraphState:
    """Merge all agent outputs into the final recommendation list."""
    agent_output_keys = [
        "pricing_output",
        "transfer_output",
        "risk_output",
        "procurement_output",
    ]

    recommendations = []
    agent_counts = {}
    for key in agent_output_keys:
        agent_output = state.get(key, {})
        agent_name = agent_output.get("agent_name", key)
        agent_recommendations = agent_output.get("recommendations", [])
        recommendations.extend(agent_recommendations)
        agent_counts[agent_name] = len(agent_recommendations)

    recommendations_df = _standardize_recommendations(recommendations)
    output_path = ""
    if state.get("save_output", True):
        output_path = str(save_recommendations(recommendations_df))

    combined_output = {
        "agent_name": "orchestrator",
        "inventory_summary": state.get("inventory_output", {}),
        "agent_counts": agent_counts,
        "total_recommendations": int(len(recommendations_df)),
        "output_path": output_path,
    }
    llm_summary = summarize_orchestration(
        {
            "inventory_summary": state.get("inventory_output", {}),
            "agent_counts": agent_counts,
            "total_recommendations": int(len(recommendations_df)),
            "sample_recommendations": recommendations_df.head(10).to_dict(
                orient="records"
            ),
        }
    )
    if llm_summary:
        combined_output["llm_summary"] = llm_summary

    return {
        "combined_output": combined_output,
        "unified_recommendations": recommendations_df.to_dict(orient="records"),
    }


def build_orchestrator_graph():
    """Create the LangGraph workflow for agent orchestration."""
    graph = StateGraph(AgentGraphState)

    graph.add_node("orchestrator", _orchestrator_node)
    graph.add_node("inventory_agent", _inventory_agent_node)
    graph.add_node("pricing_agent", _pricing_agent_node)
    graph.add_node("transfer_agent", _transfer_agent_node)
    graph.add_node("risk_agent", _risk_agent_node)
    graph.add_node("procurement_agent", _procurement_agent_node)
    graph.add_node("combine_results", _combine_results_node)

    graph.add_edge(START, "orchestrator")
    graph.add_edge("orchestrator", "inventory_agent")
    graph.add_edge("inventory_agent", "pricing_agent")
    graph.add_edge("inventory_agent", "transfer_agent")
    graph.add_edge("inventory_agent", "risk_agent")
    graph.add_edge("inventory_agent", "procurement_agent")
    graph.add_edge("pricing_agent", "combine_results")
    graph.add_edge("transfer_agent", "combine_results")
    graph.add_edge("risk_agent", "combine_results")
    graph.add_edge("procurement_agent", "combine_results")
    graph.add_edge("combine_results", END)

    return graph.compile()


def run_agent_graph(
    config: dict | None = None,
    save_output: bool = True,
) -> AgentGraphState:
    """Run the LangGraph workflow and return the final shared state."""
    orchestrator_graph = build_orchestrator_graph()
    initial_state: AgentGraphState = {
        "config": get_config(config),
        "save_output": save_output,
    }
    return orchestrator_graph.invoke(initial_state)


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
