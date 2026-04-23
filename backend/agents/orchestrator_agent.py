from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from backend.agents import demand_agent, pricing_agent, risk_agent, transfer_agent
from backend.services.data_processor import build_processed_datasets
from backend.services.inventory_analyzer import build_inventory_analysis
from backend.services.recommendation_engine import (
    PROCESSED_DATA_DIR,
    RECOMMENDATION_COLUMNS,
    get_config,
    load_recommendation_inputs,
)


AGENTS = [
    demand_agent,
    transfer_agent,
    pricing_agent,
    risk_agent,
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


def run_all_agents(
    config: dict | None = None,
    save_output: bool = True,
) -> pd.DataFrame:
    """Run all internal agents and combine their recommendation outputs."""
    config = get_config(config)

    build_processed_datasets()
    build_inventory_analysis(config)
    inputs = load_recommendation_inputs()

    recommendations = []
    for agent in AGENTS:
        recommendations.extend(agent.run(inputs, config))

    recommendations_df = _standardize_recommendations(recommendations)

    if save_output:
        save_recommendations(recommendations_df)

    return recommendations_df


if __name__ == "__main__":
    run_all_agents()
