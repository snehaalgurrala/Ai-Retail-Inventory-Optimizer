# AI Retail Inventory Optimizer

AI-powered retail decision support system built with:
- `Streamlit` for the operations dashboard
- `FastAPI` for backend APIs
- `LangGraph` and `LangChain` for multi-agent recommendation orchestration
- `OpenRouter` or `Gemini` compatible LLM support
- CSV-based raw and processed data storage

## Current Working Model

This project currently works as a retail operations copilot over CSV data.

It can:
- load raw retail data from `data/raw/*.csv`
- build processed datasets for inventory, sales, and product performance
- run inventory analysis for:
  - low stock
  - stockout risk
  - overstock
  - dead stock
  - slow-moving items
  - high-demand items
- run a multi-agent recommendation pipeline with:
  - pricing agent
  - transfer agent
  - risk agent
  - procurement agent
  - orchestrator agent
- show recommendations in a Streamlit review queue
- support human-in-the-loop approval and rejection of recommendations
- execute approved actions safely against source CSV files
- log decisions, action history, and memory/outcome data
- place customer orders and update inventory
- send low-stock alert emails after agent refresh
- answer grounded chatbot questions using RAG over raw, processed, and recommendation data

## Main Features

### 1. Agent Command Center
The home page in [frontend/app.py](/abs/path/c:/Users/TS6206_SNEHAAL/Desktop/Ai-Retail-Inventory-Optimizer/frontend/app.py) provides:
- orchestrator health summary
- specialist agent cards
- latest recommendation queue preview
- low-stock alert preview
- sales and inventory KPI cards
- sales trend and inventory distribution charts
- `Run / Refresh Agents` action to regenerate recommendations and summaries

### 2. Inventory Analysis
The inventory pipeline in [backend/services/inventory_analyzer.py](/abs/path/c:/Users/TS6206_SNEHAAL/Desktop/Ai-Retail-Inventory-Optimizer/backend/services/inventory_analyzer.py) generates:
- `low_stock_items.csv`
- `stockout_risk_items.csv`
- `overstock_items.csv`
- `dead_stock_candidates.csv`
- `high_demand_items.csv`
- `slow_moving_items.csv`

### 3. Multi-Agent Recommendations
The orchestrator in [backend/agents/orchestrator_agent.py](/abs/path/c:/Users/TS6206_SNEHAAL/Desktop/Ai-Retail-Inventory-Optimizer/backend/agents/orchestrator_agent.py) combines agent outputs into:
- `data/processed/recommendations.csv`
- `data/processed/agent_outputs.csv`
- `data/processed/orchestrator_summary.csv`

Supported recommendation types:
- `discount`
- `clearance`
- `stock_transfer`
- `reorder`
- `supplier_risk_alert`
- `overstock_alert`
- `stockout_prevention_alert`

### 4. Human-in-the-Loop Recommendation Execution
The Recommendations page in [frontend/pages/3_Recommendations.py](/abs/path/c:/Users/TS6206_SNEHAAL/Desktop/Ai-Retail-Inventory-Optimizer/frontend/pages/3_Recommendations.py) now supports:
- compact recommendation cards
- type-specific summaries
- expandable AI reasoning and impact view
- editable approval fields
- approve
- reject
- manager notes
- execution logging

Execution logic is handled by [backend/services/recommendation_execution_service.py](/abs/path/c:/Users/TS6206_SNEHAAL/Desktop/Ai-Retail-Inventory-Optimizer/backend/services/recommendation_execution_service.py).

Approval behavior currently updates data as follows:

- `discount`
  - updates `data/raw/products.csv`
  - logs to `data/processed/price_updates.csv`
- `clearance`
  - updates `data/raw/products.csv`
  - logs to `data/processed/clearance_actions.csv`
- `stock_transfer`
  - updates `data/raw/inventory.csv`
  - logs movements to `data/raw/transactions.csv`
  - logs action to `data/processed/transfer_actions.csv`
- `reorder`
  - updates `data/raw/inventory.csv`
  - logs procurement to `data/raw/transactions.csv`
  - logs order to `data/processed/procurement_orders.csv`
- `supplier_risk_alert`, `overstock_alert`, `stockout_prevention_alert`
  - logs mitigation to `data/processed/risk_actions.csv`
- all decisions
  - update `data/processed/recommendations.csv`
  - append to `data/processed/recommendation_decisions.csv`
  - write memory/outcome records for feedback loops

Safety checks currently include:
- missing field validation before approval
- quantity and price validation
- negative stock prevention
- audit logging before or alongside data mutation
- atomic CSV writes for critical service paths

### 5. Order Flow
The Orders page in [frontend/pages/5_Orders.py](/abs/path/c:/Users/TS6206_SNEHAAL/Desktop/Ai-Retail-Inventory-Optimizer/frontend/pages/5_Orders.py) supports:
- store selection
- store-specific product availability
- quantity validation
- customer order placement
- inventory update after order placement
- order receipt preview
- recent order history

Backend order handling lives in [backend/services/order_service.py](/abs/path/c:/Users/TS6206_SNEHAAL/Desktop/Ai-Retail-Inventory-Optimizer/backend/services/order_service.py).

### 6. Chatbot with RAG
The Chatbot page in [frontend/pages/4_Chatbot.py](/abs/path/c:/Users/TS6206_SNEHAAL/Desktop/Ai-Retail-Inventory-Optimizer/frontend/pages/4_Chatbot.py) supports:
- analytics-first answers for store, product, stock, sales, and supplier questions
- local vector RAG with HuggingFace embeddings and FAISS
- grounded question answering over current retail data
- retrieval from raw, processed, recommendation, and agent datasets
- automatic knowledge-index build plus manual rebuild from the chatbot sidebar
- direct answer style with optional supporting table
- follow-up chat memory
- sample question prompts

RAG and embedding logic lives in [backend/services/rag_service.py](/abs/path/c:/Users/TS6206_SNEHAAL/Desktop/Ai-Retail-Inventory-Optimizer/backend/services/rag_service.py).

### 7. Email Alerts
Low-stock alert emailing is handled by [backend/services/email_service.py](/abs/path/c:/Users/TS6206_SNEHAAL/Desktop/Ai-Retail-Inventory-Optimizer/backend/services/email_service.py).

Current behavior:
- sends low-stock email after agent refresh
- suppresses duplicate emails when the alert signature has not changed
- logs email activity to `data/processed/email_alert_log.csv`

## Project Structure

```text
backend/
  agents/                 Multi-agent recommendation logic
  memory/                 Recommendation, decision, and outcome memory
  services/               Processing, analysis, execution, RAG, chatbot, orders, email
  tools/                  Shared tool wrappers for agents
  utils/                  Data loading helpers

frontend/
  app.py                  Home command center
  pages/
    1_Inventory.py
    2_Sales.py
    3_Recommendations.py
    4_Chatbot.py
    5_Orders.py
  components/             Shared Streamlit UI helpers

data/
  raw/                    Source CSV files
  processed/              Derived datasets, recommendations, logs, and memory files
```

## Data Flow

The current end-to-end flow is:

1. Raw CSVs in `data/raw` are loaded.
2. Processed datasets are built by `data_processor.py`.
3. Inventory analysis outputs are generated.
4. Specialist agents produce recommendation candidates.
5. The orchestrator combines and saves recommendations.
6. The dashboard displays agent outputs and recommendations.
7. A user approves or rejects recommendations.
8. Approved actions update raw CSVs and processed logs.
9. Memory and outcome records are saved for future reasoning context.

## Tech Stack

From `requirements.txt`, the current app uses:
- `fastapi`
- `uvicorn`
- `pandas`
- `numpy`
- `streamlit`
- `plotly`
- `langchain`
- `langgraph`
- `langchain-community`
- `langchain-openai`
- `chromadb`
- `sentence-transformers`
- `faiss-cpu`
- `google-genai`
- `openai`
- `python-dotenv`
- `pydantic`

## Environment Variables

Create a `.env` file with the values you want to use.

Current code supports:

### LLM / RAG
- `LLM_PROVIDER=openrouter`
- `OPENROUTER_API_KEY`
- `OPENROUTER_MODEL=openrouter/free`
- `OPENROUTER_BASE_URL=https://openrouter.ai/api/v1`
- `EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2`
- `OPENROUTER_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2` (optional compatibility alias)
- `GEMINI_API_KEY`
- `GEMINI_MODEL`
- `GEMINI_EMBEDDING_MODEL`
- `LLM_TIMEOUT_SECONDS`
- `LLM_BATCH_SIZE`

The chatbot stores its local FAISS knowledge index in `data/processed/vector_store/` and rebuilds it automatically when source CSV files change.

### Email
- `SMTP_EMAIL`
- `SMTP_APP_PASSWORD`
- `MANAGER_EMAIL`

## How To Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the Streamlit app

```bash
streamlit run frontend/app.py
```

### 3. Start the FastAPI server

```bash
uvicorn backend.main:app --reload
```

## FastAPI Endpoints

The current backend in [backend/main.py](/abs/path/c:/Users/TS6206_SNEHAAL/Desktop/Ai-Retail-Inventory-Optimizer/backend/main.py) exposes:
- `GET /`
- `GET /health`
- `GET /data/summary`
- `GET /inventory/current`
- `GET /sales/summary`
- `GET /recommendations`
- `POST /agents/run`
- `GET /agents/latest-summary`
- `GET /agents/latest-outputs`
- `GET /orders/stores`
- `GET /orders/products/{store_id}`
- `POST /orders/place`

## Important Output Files

Key processed outputs currently include:
- `data/processed/current_inventory.csv`
- `data/processed/sales_summary.csv`
- `data/processed/product_performance.csv`
- `data/processed/store_inventory_summary.csv`
- `data/processed/recommendations.csv`
- `data/processed/recommendation_decisions.csv`
- `data/processed/agent_outputs.csv`
- `data/processed/orchestrator_summary.csv`
- `data/processed/email_alert_log.csv`

Action logs created during execution include:
- `data/processed/price_updates.csv`
- `data/processed/transfer_actions.csv`
- `data/processed/procurement_orders.csv`
- `data/processed/clearance_actions.csv`
- `data/processed/risk_actions.csv`

Memory files include:
- `data/processed/memory/recommendation_memory.csv`
- `data/processed/memory/decision_memory.csv`
- `data/processed/memory/outcome_memory.csv`
- `data/processed/memory/learning_insights.csv`

## Current Status

This is no longer just in setup phase. The current working model already includes:
- dashboard pages for inventory, sales, recommendations, chatbot, and orders
- working multi-agent recommendation generation
- recommendation approval and execution
- CSV mutation and logging
- RAG-backed chatbot
- low-stock email alerting
- FastAPI endpoints for summary, recommendations, agents, and orders

## Notes

- Storage is currently CSV-based, not database-backed.
- Recommendation execution updates shared files directly, so this should be treated as a local single-user or controlled environment workflow unless stronger locking is added.
- Processed analytics refresh after recommendation approval so downstream views stay aligned with changed raw data.
