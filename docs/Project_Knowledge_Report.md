# AI Retail Inventory Optimizer — Project Knowledge Report

> **Audience & purpose:** This document is an internal knowledge base. It is **not** a client-facing artifact. Its purpose is to give another AI system (or a new team member) a complete, accurate understanding of the platform so it can generate a professional, executive-level client PowerPoint from an existing template **without inspecting the source code**. Section 18 contains explicit slide-by-slide guidance for that PPT.

---

# 1. Executive Summary

**What the platform is.** The AI Retail Inventory Optimizer is an AI-powered **supply-chain decision-support platform** for a Bunzl-style B2B distribution business. It sits on top of a live **Oracle database** (the single source of truth) and turns raw inventory, sales, product, customer, and order data into a continuously-updated stream of operational intelligence: inventory health, demand insights, replenishment recommendations, customer analytics, executive alerts, and a natural-language assistant. It is delivered as a multi-page **Streamlit** operations dashboard backed by a **Python** service layer, a **LangGraph multi-agent** recommendation engine, and an **MCP (Model Context Protocol)** chatbot.

**Who uses it.**
- **Inventory / supply-chain planners** — monitor stock health, low-stock and overstock, act on reorder/transfer advice.
- **Sales & category managers** — track revenue, top/bottom products, demand trends.
- **Account managers / commercial leadership** — read Customer Intelligence: who matters, who is growing or declining, which orders are abnormal, who is dormant.
- **Operations managers** — receive automated email briefings (low-stock and abnormal-order investigation alerts) and branch-wise reports.
- **Sales engineers / solution consultants** — drive the live Order Simulator demo for prospects.
- **End customers (demo persona)** — log into the Order Simulator and place real orders that ripple through the whole platform.

**Business objectives.** Reduce stockouts and overstock, shorten the order-to-decision cycle, surface revenue concentration and churn risk, catch abnormal demand before it causes a stockout, and replace manual spreadsheet reporting with automated, consistent, AI-generated intelligence.

**Problems it solves.** Stockouts and lost service levels; overstock and dead capital; demand uncertainty; slow, manual procurement decisions; lack of customer-level demand visibility; and labor-intensive reporting.

**Key value proposition.** *One order, fully understood.* Every number across the platform — dashboard card, email, simulator, chatbot — agrees because they all read **one inventory figure** from **one Oracle source**. A single customer order is automatically committed against live stock, re-judged against its own demand history, scored for risk, converted into replenishment advice, escalated by email when it matters, and made instantly queryable by the assistant — in seconds.

---

# 2. Business Problem Statement

Distribution businesses run on thin margins where availability is everything. The platform targets six concrete pain points:

- **Inventory stockouts.** Running out of a product breaks delivery promises, forces emergency procurement, and damages every customer's service level — often triggered silently by a single outsized order.
- **Overstock situations.** Excess and dead stock tie up working capital, occupy warehouse space, and risk obsolescence. Without systematic detection, overstock accumulates unnoticed.
- **Demand uncertainty.** Demand is volatile and customer-specific. Planners lack an early-warning signal that distinguishes a normal fluctuation from a meaningful spike.
- **Procurement inefficiencies.** Reorder and transfer decisions are made manually and reactively, often too late, without a consolidated view of risk, lead times, and supplier reliability.
- **Customer demand visibility challenges.** Revenue in distribution is concentrated and fragile — a handful of accounts drive most revenue, dormant accounts leak opportunity, and declining accounts churn quietly. This is invisible in standard sales reports (a dormant customer has *no* sales rows to appear in).
- **Manual reporting challenges.** Inventory and sales reporting is spreadsheet-driven, slow, inconsistent between teams, and stale by the time it reaches leadership.

---

# 3. Solution Overview

The platform is a single integrated operations cockpit composed of seven cooperating capabilities:

- **Inventory Optimization** — continuous classification of stock into low-stock, stockout-risk, overstock, dead-stock, slow-moving, and high-demand; predictive depletion windows; network-wide stock and reorder logic.
- **Sales Intelligence** — revenue, units, top/bottom products, category mix, branch comparison, and demand trends with AI-written narrative insights.
- **Customer Intelligence** — top customers and revenue concentration, demand trend bucketing (growing/stable/declining), dormant-account detection, inventory-pressure attribution, and the flagship **Customer Demand Insights / Abnormal Order Detection** engine.
- **AI Recommendation Engine** — a LangGraph multi-agent pipeline (inventory, pricing, transfer, risk, procurement, orchestrator) that produces prioritized reorder / transfer / discount / clearance / supplier-risk / overstock / stockout-prevention recommendations with reasoning, plus a human-in-the-loop approval & execution queue.
- **AI Chatbot** — an MCP-based natural-language assistant that answers strictly from Oracle through a fixed, allow-listed tool catalog.
- **Reporting** — branch-wise inventory and sales reports (CSV + PDF) and an executive Customer Demand Intelligence report, emailed on demand.
- **Alerting** — automated low-stock alert emails and abnormal-order investigation alert emails.
- **Order Simulation** — a live sandbox where a customer logs in and places a real, committed order that drives the entire cause-and-effect chain end-to-end.

The connective tissue is two design invariants: **Oracle as the single source of truth** and a **single inventory scope** (`inventory_scope.py`, network-wide by default) so every surface reports the same numbers.

---

# 4. System Architecture

## 4.1 High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                    PRESENTATION LAYER (Streamlit)                       │
│  app.py (Agent Command Center)                                          │
│  pages/ 1_Inventory · 2_Sales · 3_Customer_Intelligence ·              │
│         4_Recommendations · 5_Chatbot · 6_Customer_Order_Simulator ·   │
│         7_Admin_Inventory                                               │
│  components/ (cards, theme, ui_components)  utils/ (page_helpers)       │
└───────────────┬──────────────────────────────────────┬───────────────┘
                │                                        │
┌───────────────▼───────────────┐      ┌────────────────▼────────────────┐
│   AI / AGENT LAYER             │      │   MCP CHATBOT LAYER              │
│  LangGraph orchestrator_agent  │      │  FastMCP stdio server           │
│  inventory/pricing/transfer/   │      │  26 allow-listed Oracle tools   │
│  risk/procurement agents       │      │  orchestrator (LLM tool-calling)│
│  memory + learning_loop        │      │  client (persistent stdio)      │
└───────────────┬───────────────┘      └────────────────┬────────────────┘
                │                                        │
┌───────────────▼────────────────────────────────────────▼──────────────┐
│                      SERVICE LAYER (Python)                             │
│  customer_intelligence · abnormal_order_intelligence · inventory_scope │
│  inventory_analyzer · recommendation_engine · order_pipeline_service   │
│  sales_analytics · low_stock_service · report/pdf_report · email svcs  │
│  oracle_writer · branch_resolver · customer_order_limits · ...         │
└───────────────────────────────┬───────────────────────────────────────┘
                                 │
┌───────────────────────────────▼───────────────────────────────────────┐
│                   DATA ACCESS (backend/db/repository.py)                │
│        OracleBackend  ◄── primary ──►  CSVBackend (fallback/legacy)     │
└───────────────────────────────┬───────────────────────────────────────┘
                                 │
┌───────────────────────────────▼───────────────────────────────────────┐
│            ORACLE DATABASE  (BZ_MOCK_* tables + 2 views)                │
│  single source of truth — customers, orders, inventory, products, etc.  │
└────────────────────────────────────────────────────────────────────────┘

Cross-cutting: REPORTING/EMAIL (SMTP), FastAPI service (backend/main.py)
```

## 4.2 Component Roles

- **Frontend (Streamlit).** Multi-page dashboard. Each page is self-contained, imports backend services, and caches data via `st.cache_data`. Shared UI lives in `frontend/components/` and `frontend/utils/page_helpers.py`. The home page (`app.py`) is the **Agent Command Center** with orchestrator health, specialist agent cards, low-stock alerts, the Report Email Center, latest recommendations, KPIs, and trend charts.
- **Backend (Python service layer).** Business logic in `backend/services/`. Data access is abstracted behind `backend/db/repository.py`, which selects an **OracleBackend** (primary) or **CSVBackend** (legacy/fallback) depending on configuration.
- **Database (Oracle).** All live data is `BZ_MOCK_*` tables. Writes (order placement, restock) go through `backend/db/oracle_writer.py` in atomic transactions.
- **AI Layer (LangGraph agents).** `backend/agents/` runs a stateful multi-agent graph that produces recommendations and an executive orchestrator summary.
- **MCP Layer.** `backend/mcp/` exposes a fixed catalog of Oracle-reading tools over a FastMCP stdio server; the chatbot LLM selects tools (never SQL) and explains the results.
- **Reporting Layer.** `report_service.py`, `pdf_report_service.py`, `abnormal_order_report.py`, and the email services build and dispatch reports/alerts via SMTP.

## 4.3 How Components Communicate

- Streamlit pages call service functions directly (in-process Python).
- Services read/write Oracle through `repository` / `oracle_writer`.
- The recommendation pipeline is invoked synchronously (`run_agent_graph`) and writes processed CSV artifacts (`recommendations.csv`, `agent_outputs.csv`, `orchestrator_summary.csv`) that the dashboard reads.
- The chatbot page calls `chatbot_router`, which (per the `CHATBOT_ENGINE` flag) routes to the MCP orchestrator; the orchestrator launches/holds a persistent stdio subprocess (`client.py`) running the FastMCP server.
- When an order is placed in the simulator, `order_pipeline_service` fans out to Customer Intelligence recalculation, recommendation re-run, abnormal-order email, and chatbot context reset, then clears Streamlit caches so all pages refresh.
- A **FastAPI** service (`backend/main.py`) exposes REST endpoints (summary, inventory, sales, recommendations, agent run, orders) for programmatic/integration access.

---

# 5. Technology Stack

**Frontend**
- **Streamlit** — primary UI framework (multi-page app, caching, session state).
- **Plotly** — interactive charts (trends, bars, donuts).
- Custom HTML/CSS theming via `components/theme.py`, `ui_components.py`, `page_helpers.py` (Bunzl-style navy/green premium theme). No separate React app; "components" are Streamlit/HTML render helpers.

**Backend**
- **Python** service layer.
- **FastAPI + Uvicorn** — REST API (`backend/main.py`).
- **pandas / numpy** — all analytics and transformations.
- **Pydantic** — typed config/validation.

**Database**
- **Oracle** — single source of truth (`BZ_MOCK_*` schema). Accessed via `python-oracledb`-style backend in `backend/db/oracle_backend.py` / `oracle_writer.py`, abstracted by `repository.py`. A **CSV backend** remains as a legacy/fallback path.

**AI**
- **MCP (Model Context Protocol)** via **FastMCP** — the chatbot tool server (stdio).
- **LLM integration** — OpenRouter (default `openrouter/free` auto-router) and Google **Gemini** are both supported through a provider abstraction; the LLM is used for (a) MCP **tool calling**, (b) agent/orchestrator **executive summaries**, and (c) AI narrative generation.
- **LangChain / LangGraph** — multi-agent orchestration graph.
- **Tool calling** — the MCP orchestrator runs an LLM tool-selection loop over the allow-listed catalog (JSON tool calls, never raw SQL).
- **RAG (legacy path)** — `sentence-transformers` embeddings + **FAISS** / **ChromaDB** vector store powered the older chatbot; retained behind `CHATBOT_ENGINE=legacy`. The default MCP path does **not** use RAG/embeddings.

**Reporting**
- **Email engine** — SMTP (Gmail app-password) via `email_service.py` and `abnormal_order_email.py`; premium HTML email templates.
- **Excel / CSV / PDF reports** — `report_service.py`, `pdf_report_service.py` (and openpyxl-style Excel outputs such as `Low_Stock_Report.xlsx` and the Abnormal Order Report `.xlsx`).

---

# 6. Oracle Database Structure

Oracle is the authoritative store. The canonical seed is `sql/bunzl_walmart_like_mock_b2b_inventory_pricing_oracle.sql` — **10 baseline tables + 2 views**. Baseline row counts: 5 branches, 12 customers, 16 products, 80 price-history, 15 contract-price, 80 inventory, 30 orders, 103 order-lines, 450 sales-history, 32 competitor-price. The live database also carries **3 extra tables** that some features depend on: `BZ_MOCK_SUPPLIER`, `BZ_MOCK_BRANCH_CAPACITY`, `BZ_MOCK_INVENTORY_TRANSACTION`. The reset tool (`scripts/reset_oracle_baseline.py`) restores baseline data while keeping these three.

> **Naming note for slides:** internal tables are prefixed `BZ_MOCK_`. In a client deck, refer to them by business name (Customers, Orders, Inventory, etc.).

## 6.1 Tables

### `BZ_MOCK_BRANCH` — Branches / Warehouses
- **Purpose:** distribution locations that hold inventory and fulfil orders.
- **Key columns:** `BRANCH_ID` (PK), `BRANCH_NBR`, `BRANCH_NAME`, `CITY`, `STATE`, `ZIP_CODE`, `BRANCH_TYPE`, `ACTIVE_FLG`.
- **Relationships:** referenced by Inventory, Order Header, Sales History, Price History.

### `BZ_MOCK_CUSTOMER` — Customers
- **Purpose:** the real end-customer dimension (B2B accounts).
- **Key columns:** `CUSTOMER_ID` (PK), `CUSTOMER_NBR`, `CUSTOMER_NAME`, `INDUSTRY`, `CUSTOMER_SEGMENT`, `CITY/STATE/ZIP`, `CONTRACT_TIER`, `CREDIT_LIMIT_AMT`, `SIGNUP_DATE`, `CONTRACT_NBR`, `ACTIVE_FLG`.
- **Relationships:** parent of Order Header; referenced by Contract Price. 12 active customers; ~5–7 have orders, the rest surface as **Dormant Accounts**.

### `BZ_MOCK_PRODUCT` — Product Master / Catalogue
- **Purpose:** SKU catalogue and pricing.
- **Key columns:** `PRODUCT_ID` (PK), `SKU` (unique), `UPC`, `PRODUCT_NAME`, `BRAND_NAME`, `CATEGORY`, `SUB_CATEGORY`, `PACK_SIZE_DESC`, `UOM`, `COST_PRICE`, `CURRENT_SELLING_PRICE`, `ACTIVE_FLG`.
- **Relationships:** referenced by Inventory, Order Line, Sales History, Price History, Contract Price, Competitor Price.

### `BZ_MOCK_INVENTORY` — Inventory (stock per product per branch)
- **Purpose:** on-hand stock position and replenishment parameters.
- **Key columns:** `INVENTORY_ID` (PK), `BRANCH_ID` (FK), `PRODUCT_ID` (FK), `ON_HAND_QTY`, `RESERVED_QTY`, `AVAILABLE_QTY`, `REORDER_POINT`, `SAFETY_STOCK`, `REORDER_QTY`, `AISLE`, `BIN_LOCATION`, `LAST_RESTOCK_DATE`, `LAST_SOLD_DATE`, `INVENTORY_STATUS`.
- **Relationships:** child of Branch and Product. One row per product-branch. **Network scope** sums `ON_HAND_QTY` and `REORDER_POINT` across all branches for a product.
- *Honest data gap:* in the read path, `AVAILABLE_QTY` is treated as on-hand; `SAFETY_STOCK` is present but not used as a distinct driver.

### `BZ_MOCK_ORDER_HEADER` — Order Header
- **Purpose:** a customer order (one per purchase event).
- **Key columns:** `ORDER_ID` (PK), `ORDER_NBR` (unique, `ORD-YYYYMMDD-NNNN`), `CUSTOMER_ID` (FK), `BRANCH_ID` (FK), `ORDER_DATE`, `ORDER_CHANNEL`, `ORDER_STATUS`, `PAYMENT_METHOD`, `ORDER_TOTAL_AMT`.
- **Relationships:** child of Customer and Branch; parent of Order Line.

### `BZ_MOCK_ORDER_LINE` — Order Lines
- **Purpose:** line items of an order (product + quantity + price).
- **Key columns:** `ORDER_LINE_ID` (PK), `ORDER_ID` (FK), `PRODUCT_ID` (FK), `QUANTITY`, `UNIT_SELLING_PRICE`, `UNIT_COST_PRICE`, `DISCOUNT_AMT`, **`LINE_TOTAL_AMT`**.
- **Relationships:** child of Order Header and Product.
- **Critical rule:** **revenue = `LINE_TOTAL_AMT`** (authoritative). Some lines do not equal qty×price−discount, so revenue is **never recomputed** — always read directly.

### `BZ_MOCK_SALES_HISTORY` — Sales History
- **Purpose:** historical point-of-sale movement used for velocity, trends, demand.
- **Key columns:** `SALES_HIST_ID` (PK), `SALES_DATE`, `BRANCH_ID` (FK), `PRODUCT_ID` (FK), `UNITS_SOLD`, `GROSS_SALES_AMT`, `DISCOUNT_AMT`, `NET_SALES_AMT`, `AVG_SELLING_PRICE`.
- **Relationships:** child of Branch and Product. Powers the Sales page and demand velocity.

### `BZ_MOCK_PRICE_HISTORY` — Price History
- **Purpose:** time-phased branch-level pricing (regular/selling/cost, promos).
- **Key columns:** `PRICE_HIST_ID` (PK), `PRODUCT_ID`, `BRANCH_ID`, `REGULAR_PRICE`, `SELLING_PRICE`, `COST_PRICE`, `PRICE_TYPE`, `EFFECTIVE/EXPIRATION_DATE`, `PROMO_DESC`.

### `BZ_MOCK_CONTRACT_PRICE` — Contract Pricing
- **Purpose:** customer-specific negotiated prices.
- **Key columns:** `CONTRACT_PRICE_ID` (PK), `CUSTOMER_ID` (FK), `PRODUCT_ID` (FK), `CONTRACT_PRICE`, effective/expiration dates.

### `BZ_MOCK_COMPETITOR_PRICE` — Competitor Pricing
- **Purpose:** external price benchmarks for pricing intelligence.
- **Key columns:** `COMP_PRICE_ID` (PK), `PRODUCT_ID` (FK), `COMPETITOR_NAME`, `COMPETITOR_PRICE`, `PRICE_DATE`, `ZIP_CODE`, `SOURCE_DESC`.

### Supporting (non-seed) tables
- **`BZ_MOCK_SUPPLIER`** — supplier master incl. `AVG_DELIVERY_DAYS` (lead time) and reliability; feeds procurement risk and product-master lead-time.
- **`BZ_MOCK_BRANCH_CAPACITY`** — per-branch capacity (joined into store loading).
- **`BZ_MOCK_INVENTORY_TRANSACTION`** — movement ledger for restocks/adjustments (Admin Inventory, recommendation execution).

### Views
- **`BZ_MOCK_AI_ORDER_DETAIL_VW`** — denormalized order detail for analytics.
- **`BZ_MOCK_AI_PRICING_INTEL_VW`** — pricing intelligence (product vs contract vs competitor).

## 6.2 Relationship Diagram (text)

```
BZ_MOCK_BRANCH ──────┐ (1:N)
                     ├──< BZ_MOCK_INVENTORY >── BZ_MOCK_PRODUCT
                     ├──< BZ_MOCK_SALES_HISTORY >── BZ_MOCK_PRODUCT
                     ├──< BZ_MOCK_PRICE_HISTORY >── BZ_MOCK_PRODUCT
                     │
BZ_MOCK_CUSTOMER ──< BZ_MOCK_ORDER_HEADER >── BZ_MOCK_BRANCH
        │                    │ (1:N)
        │                    ▼
        │            BZ_MOCK_ORDER_LINE ──> BZ_MOCK_PRODUCT   (revenue = LINE_TOTAL_AMT)
        │
        └──< BZ_MOCK_CONTRACT_PRICE >── BZ_MOCK_PRODUCT

BZ_MOCK_PRODUCT ──< BZ_MOCK_COMPETITOR_PRICE

Supporting: BZ_MOCK_SUPPLIER, BZ_MOCK_BRANCH_CAPACITY, BZ_MOCK_INVENTORY_TRANSACTION
Views:      BZ_MOCK_AI_ORDER_DETAIL_VW, BZ_MOCK_AI_PRICING_INTEL_VW
```

---

# 7. Inventory Intelligence Module

**Page:** `frontend/pages/1_Inventory.py`. **Core services:** `inventory_analyzer.py`, `low_stock_service.py`, `inventory_scope.py`, `transfer_analysis_service.py`, `depletion_formatter.py`.

**Purpose.** Give planners a continuously-classified, predictive view of stock health so they can prevent stockouts and clear overstock before either hurts the business.

**Features.**
- Network-wide inventory snapshot and store-level drill-down (store selector excluded from network-scope standardization on purpose).
- Automatic classification of every product into: **low-stock, stockout-risk, overstock, dead-stock, slow-moving, high-demand**.
- **Predictive depletion windows** and urgency labels (Critical / High / Medium / Healthy) from sales velocity.
- Understock and overstock tables with suggested reorder quantities and suggested transfer source branches.
- **AI Inventory Insight** narrative and store-specific AI recommendations.
- Category-wise charts and store comparison.

**KPIs (Inventory page).** Inventory Qty (total units), Products (unique SKUs), Low Stock count, Overstock count, Dead/Slow indicators, **Inventory Value** (qty × selling price).

**Inventory Health.** Driven by `inventory_scope.py`: a product's "current inventory" = SUM of on-hand across all branches; its reorder point = SUM of per-branch reorder points. A product is **at-risk** when network stock ≤ network reorder point. This single helper guarantees the same figure appears on the Inventory page, Customer Intelligence cards, the simulator catalogue, emails, the chatbot, and the recommendation engine.

**Low Stock Detection.** `low_stock_service.get_low_stock_items` scans inventory vs reorder point, computes sales velocity and `predicted_days_remaining`, assigns priority (stock == reorder point is treated as **Medium**, not Low), and produces the alert table consumed by the dashboard and the low-stock email.

**Overstock Detection.** Identifies products with stock far above demand/days-of-cover thresholds (`overstock_items.csv`, `dead_stock_candidates.csv`, `slow_moving_items.csv`), enabling clearance/discount action.

**Transfer Recommendations.** `transfer_analysis_service.py` finds alternative products and surplus branches to cover a shortfall — e.g. move stock from a low-demand branch to a branch nearing stockout.

**Inventory Reports.** Branch-wise inventory reports (CSV + PDF) generated by `report_service.py` / `pdf_report_service.py` and emailed from the Report Email Center.

**Low Stock Email Alerts.** Premium HTML email built by `email_service._build_premium_html_email`, fed by two paths: the scheduled forecast scan (real velocity) and the order-triggered path (`order_pipeline_service.dispatch_low_stock_alerts`). Status/depletion is **inventory-position based** via `depletion_formatter.inventory_position_status`, so a product at/below reorder never falsely reads "Healthy."

**Business value.** Fewer stockouts and emergency buys; less dead capital; planners act on prediction, not after the fact; consistent numbers eliminate "which screen is right?" debates.

---

# 8. Sales Intelligence Module

**Page:** `frontend/pages/2_Sales.py`. **Service:** `sales_analytics_service.py`. **Source:** `BZ_MOCK_SALES_HISTORY`.

**Purpose.** Show how the business is performing commercially and where demand is heading, with filters by branch, category, and date.

**Features.**
- Filterable sales dataset (branch / category / date range).
- Overview KPI cards, trend chart, category mix donut, branch comparison.
- Top and bottom product performance.
- Inventory-vs-sales comparison.
- AI-generated sales insights narrative.

**KPIs.** Total revenue, total units sold, average selling price, order/line counts, branch and category breakdowns.

**Revenue Analytics.** Revenue and units by product, branch, and category over time.

**Top Products / Bottom Products.** Ranked by revenue and quantity to reveal which SKUs carry the business and which underperform.

**Demand Trends.** Trend lines over the available sales window plus branch comparison to spot momentum and laggards.

**Sales Reports.** Branch-wise sales reports (CSV) generated and emailed from the Report Email Center on the home page.

**Business value.** Protects availability of revenue-leading SKUs, exposes underperformers for action, and gives leadership a fast, automated commercial pulse without manual spreadsheets.

---

# 9. Customer Intelligence Module

**Page:** `frontend/pages/3_Customer_Intelligence.py`. **Services:** `customer_intelligence_service.py` (detection + aggregation), `abnormal_order_intelligence.py` (cards, risk scoring, AI narrative — the shared single source of truth), `inventory_scope.py`. **Source:** the real `CUSTOMER → ORDER_HEADER → ORDER_LINE` model.

> **Client rebrand note (presentation only):** the abnormal-order feature is shown to users as **"Customer Demand Insights"**; the card is **"Demand Opportunity Review"**; the email/report is **"Customer Demand Intelligence Alert/Report."** Internal band keys (Low/Medium/High/Critical) are never renamed; only display labels map via `RISK_DISPLAY_LABEL` (Critical→*Significant Opportunity*, High→*High Demand Activity*, Medium→*Moderate Demand Activity*, Low→*Normal Demand Activity*). **Use the rebranded language in client slides.**

**Business objective.** Turn raw order history into a live answer to four leadership questions: *Who are my most valuable customers? Whose demand is growing or fading? Which orders are abnormal/risky right now? Which customers are quietly draining my most fragile inventory?*

**Customer demand analysis.** Live from Oracle, with a header showing order window, order count, active buyers, and total customers on file.

**Executive KPIs (5 flip cards).** Top Customer (highest revenue), Highest Growth (largest positive revenue change, split-window), Abnormal Orders (count above deviation threshold), Stockout-Risk Customers (ordering at-risk products), Dormant Accounts (active customers with zero orders).

**Top customers.** Customer Spotlight (top 5 by revenue as cards) and a Top Customers leaderboard table (orders, units, revenue, contribution %). Drives the concentration talking point ("top 3 = X% of revenue").

**Top products.** Top 10 by revenue and by quantity (two charts), grouped from `LINE_TOTAL_AMT` and quantity.

**Customer demand insights / Demand opportunity detection (the hero section — "AI Order Intelligence Center").**
- A **deviation-threshold control** (10–200%, default 50%) re-derives every anomaly figure live; the same threshold is shared with the simulator (`ci_abnormal_threshold` in session state).
- **AI Executive Summary** panel: abnormal orders detected, customers/products impacted, **revenue exposure** (sum of flagged line revenue), highest-risk customer/product.
- A risk-band summary (Critical / High / Medium / Low counts).
- **One full-width investigation panel per abnormal order**, ordered Critical→Low, each leading with a plain-English Executive Summary and a risk badge — no formulas on the face.
- An expandable **AI Investigation Report** per order: a full order-history chart (every order as a bar, the abnormal one highlighted red at its true position), then narrative blocks — *What Happened, Inventory Impact, Product Demand Context, Customer Behaviour Assessment* (deliberately hedged: "may indicate", "could suggest"), *Business Recommendation*, and a collapsed **Technical Details** drill-down with the exact math.
- A detailed table of every flagged line.

**Historical average / maximum calculations.** For each product, a **prior-orders-only baseline** is built: mean/max/min/std computed from that product's orders dated **strictly before** the order being judged. The shared helper `prior_order_baseline()` adds `hist_count/hist_mean/hist_max/hist_min/hist_std`; the first order of a product has no prior history and is never flagged. **Only the most recent order per product is evaluated**, so the flagged bar always sits at the far right of the trend chart.

**Why prior-only.** Including the current order in its own average would inflate the baseline and hide the spike. Judging each order only against what was known *before* it keeps the question honest: *given everything we knew before today, is the newest order abnormal enough to act on?* The page and the simulator share this helper, so they can never disagree.

**Deviation percentage calculation.** `deviation = (current − prior_mean) / prior_mean × 100`. An order qualifies as abnormal only if it is **both** ≥ the configured threshold (default 50%) above the prior mean **and** at least **5 units** above it (an absolute floor so tiny-mean products don't flag on noise). Severity is graded "High" at **3×** the configured threshold (≥150% at the 50% default), otherwise Medium.

**Inventory impact calculation.** Share of on-hand network stock a single order consumes; a product is "at-risk" when network stock ≤ network reorder point.

**Demand trend calculation.** The order window is split at its midpoint; each customer's second-half revenue is compared to the first half: **> +10% = Growing, < −10% = Declining, else Stable** (brand-new buyers count as Growing). Positioned honestly as a directional signal, not a forecast (the order window is short).

**Risk scoring methodology — composite 0–100 score** (drives bands and emails), blending five business factors:

| Component | Weight | Business meaning |
|---|---|---|
| Deviation from average | 30% | How far above its own normal this order is |
| Increase above historical max | 25% | Whether it breaks the product's all-time peak |
| Inventory impact | 20% | Share of on-hand stock this one order consumes |
| Reorder-point pressure | 15% | Whether the product is already at/below reorder |
| Recent demand trend | 10% | Whether demand was already accelerating |

**Bands:** 0–30 Low · 31–60 Medium · 61–85 High · 86–100 Critical. The same 200-unit order can be Medium or Critical depending on inventory — **risk is demand *and* inventory, not deviation alone.**

**Inventory Impact Analysis section.** Ranks customers driving inventory pressure (ordering at-risk products), with a 0–100 pressure score relative to the heaviest contributor and the affected products — connecting *customer behaviour* to *stockout risk*.

**Dormant Accounts.** Active (`active_flg='Y'`) customers with zero orders — a ready-made re-engagement list (segment, tier, industry, city, credit limit, signup date). Framed as "an opportunity, not an error."

**AI-generated recommendations / AI Insights.** Concise, data-grounded sentences (top customer + concentration, top products, largest abnormal order with real numbers, dormant count, biggest inventory-pressure customer, growing vs declining counts). Every claim is backed by a number shown elsewhere on the page — nothing is invented.

**Business value.** Protect and grow the accounts that matter, investigate risky orders before fulfilment, pre-empt stockouts at the customer level, and re-engage dormant revenue — all automatically, from live Oracle data, with no analyst or spreadsheet.

---

# 10. Customer Order Simulator

**Page:** `frontend/pages/6_Customer_Order_Simulator.py`. **Write path:** `oracle_writer.place_customer_order`. **Pipeline:** `order_pipeline_service.py`. **Helpers:** `branch_resolver.py`, `customer_order_limits.py`.

**Purpose.** A live sandbox that demonstrates the entire platform in motion, on demand. A customer logs in, builds a basket, and places an order — and the platform reacts exactly as it would to a genuine order: stock committed, demand re-evaluated, risk scored, recommendations regenerated, alerts emailed, chatbot updated. It turns an abstract "AI supply-chain platform" into a tangible cause-and-effect story the client can trigger themselves.

**Customer login flow.** Customer "logs in" (password `Bunzl@123`). The login dropdown lists **only customers with ≥1 order** (derived, not hardcoded). On login, the session **locks to the customer's home branch** via `branch_resolver.resolve_home_branch` (active branch matching the customer's city+state → else mode of their order history → else lowest active). Catalogue, stock display, pre-flight checks, and draw-down all run **branch-scoped** for that session (`SESSION_SCOPE="branch"`), independent of the global network-scope flag.

**Order placement flow.** The catalogue shows each product's price and current stock, with low-stock (<50) and out-of-stock (0) flagged. A **per-customer per-product quantity cap** (`customer_order_limits.py`, default 2, configurable on the CI page) limits how many units of one product can be added. As items are added, a **pre-flight stock check** disables Place Order if any line exceeds available stock.

**Oracle updates (the transactional commit).** `place_customer_order(..., single_branch=True)` does everything in **one transaction**: prices every line from the authoritative catalogue, inserts one `ORDER_HEADER` (`order_id = MAX+1`, `order_nbr = ORD-YYYYMMDD-NNNN`) + its `ORDER_LINE` rows, and decrements `BZ_MOCK_INVENTORY` on-hand/available at the home branch with a **validated, locking decrement**. If any line would drive stock below zero, the **entire transaction rolls back** — no partial order, no inventory drift. This is the real overselling guarantee.

**Inventory updates.** On-hand stock is reduced at the home branch; the confirmation shows, per product, `−ordered qty → new units on hand`.

**Sales / revenue updates.** The committed order and its authoritative revenue (`LINE_TOTAL_AMT`) become part of the order model that Customer Intelligence reads.

**Customer Intelligence updates.** The just-placed order is treated as the **latest** order for each of its products and scored against that product's prior-only baseline at the user's threshold — producing the abnormal verdict, deviation %, inventory impact %, and composite risk band.

**Email workflows.** If the order is genuinely concerning (band Critical/High/Medium **or** the product is now at/below reorder), a manager receives a narrative **Abnormal Order Investigation Alert** email. Routine orders send nothing (no alert fatigue).

**The post-order pipeline (4 best-effort stages over an already-committed order):**
1. **Customer Intelligence recalculation** — re-derive abnormal detection treating the new order as latest; threshold from the CI page slider; risk mirrors the CI formula.
2. **Recommendation agent re-run** — `orchestrator_agent.run_all_agents` regenerates reorder/transfer/supplier-risk recommendations on the new stock.
3. **Abnormal Order Investigation Alert email** — sent per the rule above (`abnormal_order_email.py`).
4. **Chatbot context refresh** — `context.clear_context()` + `client.reset_client()` so the next chatbot query reflects the order immediately.
Then `st.cache_data.clear()` invalidates every cached loader so the catalogue, CI, and Recommendations pages reload fresh.

**Why it was built.** To make the platform's value visible and interactive — compressing the order-to-decision cycle into a few seconds the client can trigger with their own hands.

**How it is used during demos.** See Section 15 — open Customer Intelligence, place a deliberately large order, watch the confirmation cascade, return to CI to see the new Critical/High panel, show the email, ask the chatbot, and close on the business decision.

---

# 11. AI Recommendation Engine

**Service:** `recommendation_engine.py`. **Agents:** `backend/agents/` (LangGraph). **Execution:** `recommendation_execution_service.py`. **Page:** `frontend/pages/4_Recommendations.py`.

**Purpose.** Convert the current data picture into a prioritized, explainable action queue, and let a human approve/reject before anything changes.

**Architecture.** A **LangGraph StateGraph** (`orchestrator_agent.py`) runs specialist agents and an orchestrator:
- **Inventory agent** — surfaces low-stock/stockout/overstock/dead-stock signals.
- **Pricing agent** — discount/clearance opportunities (uses cost, selling, competitor, contract prices).
- **Transfer agent** — inter-branch stock-transfer opportunities to cover shortfalls.
- **Risk agent** — supplier risk, stockout-prevention, overstock alerts.
- **Procurement agent** — reorder/purchase recommendations using reorder point, lead time, supplier reliability.
- **Orchestrator agent** — combines all outputs, writes the executive summary (database health, total recommendations, high-priority alerts, top risk, top opportunity), and persists results. Includes **memory** (`memory_store`) and a **learning loop** (`learning_loop`) that record decisions/outcomes for feedback context.

**Inputs.** Processed analyzer outputs (low/stockout/overstock/dead/slow/high-demand), current inventory, product performance, suppliers, sales, products, stores — all loaded via `repository`. Key config (`DEFAULT_CONFIG`): reorder cover days (7), stockout high/medium priority days (3/7), transfer source buffer multiplier (1.5), discount days-of-stock (45), supplier reliability threshold (0.90), supplier delivery-days threshold (4).

**Analysis logic & recommendation generation.** Each agent emits typed candidates; the orchestrator unifies, deduplicates, prioritizes (High/Medium/Low), attaches reasoning/evidence, depletion window, urgency, risk score, confidence, suggested quantities/branches, and writes `recommendations.csv`, `agent_outputs.csv`, `orchestrator_summary.csv`, plus per-agent card summaries.

**Supported recommendation types.** `reorder`, `stock_transfer`, `discount`, `clearance`, `supplier_risk_alert`, `overstock_alert`, `stockout_prevention_alert`.

**Human-in-the-loop execution.** The Recommendations page renders compact cards with type-specific summaries, expandable AI reasoning/impact, and editable approval fields. On approve, `recommendation_execution_service.py` applies the action with safety checks (field validation, quantity/price validation, negative-stock prevention, audit logging, atomic writes): e.g. `reorder`/`stock_transfer` update inventory and log to the transaction ledger; `discount`/`clearance` update product pricing; risk alerts log mitigation. All decisions append to `recommendation_decisions.csv` and write memory/outcome records.

**Business value.** Replaces reactive, manual procurement with a prioritized, explainable, audited queue — faster, safer replenishment and clearance decisions with a human always in control.

---

# 12. MCP Chatbot

**Page:** `frontend/pages/5_Chatbot.py`. **Engine:** `backend/mcp/` (FastMCP). **Router:** `chatbot_router.py` (`CHATBOT_ENGINE` flag, `mcp` default / `legacy` = old RAG).

**Architecture.** The chatbot answers **strictly from Oracle** via a fixed, allow-listed tool catalog. The LLM only **selects tools** (JSON tool calls, never SQL) and explains the results — no RAG/FAISS/embeddings on this path. Key pieces:
- **`context.py`** — loads the raw Oracle tables once (live, strict), builds shared store-inventory + predictive views, and loads the order model; exposes `customer_order_facts()`. `clear_context()` drops the warm snapshot (TTL ~30s).
- **`registry.py` + `tools/`** — the **26 allow-listed tools** (single source of truth). Each tool's JSON schema is derived from its typed signature, so the catalog and the executed function can never drift. A tool not in the catalog cannot be called by anyone.
- **`server.py`** — FastMCP stdio server; launched by `scripts/run_mcp_server.py`.
- **`client.py`** — persistent stdio session in a background asyncio thread, sync bridge, singleton reused across Streamlit reruns (server subprocess pinned to `sys.executable`).
- **`orchestrator.py`** — the LLM tool-calling loop; falls back to the in-process registry if stdio can't start; system prompt routes phrasing to the right tools.

**Tool catalog (26 allow-listed tools).**
- *Reference:* `list_stores`, `list_products`, `list_suppliers`, `validate_location`.
- *Inventory health:* `get_inventory_health`, `get_low_stock_items`, `get_overstock_items`.
- *Product master & performance:* `get_product_master`, `get_products_below_reorder`, `get_top_products`, `get_bottom_products`, `get_product_performance`.
- *Forecasting / risk:* `get_stockout_risk`, `get_demand_forecast`, `get_high_demand_items`.
- *Supplier / procurement:* `get_supplier_analysis`, `get_procurement_risk`.
- *Customers & orders (real customer model):* `get_top_customers`, `get_customer_order_analysis`, `get_customer_products`, `get_recent_orders`, `detect_abnormal_ordering`, `get_customer_demand_trends`, `get_dormant_accounts`, `get_order_inventory_impact`.
- *Transfer:* `get_transfer_opportunities`.

**Oracle integration.** Tools reuse the existing `*_service.py` business logic and resolve inventory through `inventory_scope` (network-wide), so chatbot answers match every other surface. Customer/order tools use the real `BZ_MOCK_CUSTOMER → ORDER_HEADER → ORDER_LINE` model, so the chatbot reflects orders placed in the simulator (its context is reset right after each order).

**How questions are answered (MCP workflow).** User question → orchestrator presents the catalog to the LLM → LLM emits a JSON tool call → registry validates/coerces args and executes the tool against the warm Oracle snapshot → tool returns structured data → LLM explains it in natural language. Retries + tolerant JSON salvage mitigate flaky free-tier models.

**Example queries.** "Give me the product list with their reorder points." · "What products did [customer] order, and which is their highest-revenue product?" · "Any abnormal orders today?" · "What should we reorder?" · "Which products are below reorder point?" · "Who are our top customers?" · "Which customers are growing or declining?" · "Which customers are dormant?" · "What's the stockout risk for [product]?"

**Business value.** Self-serve, always-current analytics in plain English — no SQL, no analyst — grounded only in real Oracle data, with answers that agree with every dashboard, email, and the simulator.

---

# 13. Email Automation Framework

All emails are premium HTML in the Bunzl theme, sent over SMTP (`SMTP_EMAIL` / `SMTP_APP_PASSWORD` / `MANAGER_EMAIL`). Set `EMAIL_DRY_RUN=1` to run all analysis while suppressing actual SMTP sends. **SMTP is configured in this environment — sends are real; test sparingly.**

### 13.1 Low Stock Alert
- **Trigger:** after an agent refresh (when requested) and on the order-triggered path when an order pushes stock to/below reorder. Duplicate alerts are suppressed when the alert signature is unchanged.
- **Recipients:** manager (`MANAGER_EMAIL`).
- **Content:** products at/below reorder with inventory-position status (Critical / Reorder Required / Monitor / Healthy), reorder point, avg daily sales (or "No recent demand"), depletion window, suggested reorder/transfer. Status is inventory-position based, never falsely "Healthy."
- **Service:** `email_service.py` (`_build_premium_html_email`), `low_stock_service.py`, `depletion_formatter.py`.
- **Business purpose:** prompt, contradiction-free replenishment signal that reads like an inventory planning report.

### 13.2 Inventory Reports
- **Trigger:** on demand from the Report Email Center on the home page (branch + date range).
- **Recipients:** manager.
- **Content:** branch-wise inventory report as **CSV + PDF** attachments with an HTML summary (latest stock snapshot).
- **Service:** `report_service.py`, `pdf_report_service.py`.
- **Business purpose:** instant, consistent branch reporting without spreadsheets.

### 13.3 Sales Reports
- **Trigger:** on demand from the Report Email Center (branch + date range).
- **Recipients:** manager.
- **Content:** branch-wise sales report (CSV) with HTML summary.
- **Service:** `report_service.py`.
- **Business purpose:** fast commercial reporting to leadership.

### 13.4 Customer Demand Intelligence Reports (Abnormal Order)
- **Trigger:** on demand ("Today / Yesterday / Selected Date" buttons in the Report Email Center), and per-order via the simulator pipeline (**Abnormal Order Investigation Alert** for bands Critical/High/Medium or at-reorder).
- **Recipients:** manager.
- **Content:** an executive briefing (~10 sections): executive summary, the order-history pattern with the spike highlighted, deviation %, inventory impact %, stockout-risk level, reasoning bullets, possible business reasons, and recommended actions — same risk bands, scores, and inventory scope as the CI page.
- **Service:** `abnormal_order_report.py` (on-demand report) and `abnormal_order_email.py` (per-order alert), both importing the shared `abnormal_order_intelligence.py` logic.
- **Business purpose:** get the right manager to investigate a risky order *before* fulfilment causes a stockout.

---

# 14. End-to-End Workflow

```
Customer Login (Order Simulator)
        │  session locks to home branch; branch-scoped catalogue & stock
        ▼
Order Placement
        │  per-product cap + pre-flight stock check; Place Order
        ▼
Oracle Update  (ONE atomic transaction)
        │  ORDER_HEADER + ORDER_LINE inserted; rolls back entirely on shortfall
        ▼
Inventory Update
        │  on-hand decremented at home branch (validated, locking)
        ▼
Sales / Revenue Update
        │  committed order + authoritative LINE_TOTAL_AMT now in the order model
        ▼
Customer Intelligence Update (Stage 1)
        │  new order = latest; scored vs prior-only baseline; deviation %,
        │  inventory impact %, composite 0–100 risk score + band
        ▼
Recommendation Update (Stage 2)
        │  multi-agent re-run → new reorder / transfer / supplier-risk advice
        ▼
Chatbot Update (Stage 4)
        │  context cleared + stdio client reset → order answerable immediately
        ▼
Email Generation (Stage 3)
        │  Abnormal Order Investigation Alert if Critical/High/Medium or at-reorder
        ▼
Cache clear → CI page, Recommendations page, catalogue all reload fresh
        ▼
Business decision: approve · investigate · expedite · transfer · re-engage
```

Every figure in this chain is reconciled through **one inventory scope** and sourced live from **one Oracle database**, so the simulator's "new stock," the CI card's "current inventory," the email's number, and the chatbot's answer are identical. Stages 1–4 are best-effort over an already-committed order, so analytics failures never corrupt the order or inventory.

---

# 15. Demo Walkthrough

A complete client demo, step by step:

1. **Inventory Page** — show network-wide stock health, KPIs (Inventory Qty, Products, Low Stock, Overstock, Inventory Value), understock/overstock tables, and AI inventory insight. Talking point: "one consistent stock figure everywhere."
2. **Sales Page** — revenue and units KPIs, sales trend, category mix, top/bottom products, branch comparison. Talking point: "automated commercial pulse, no spreadsheets."
3. **Customer Intelligence Page (set the scene first)** — top customers and revenue concentration ("top 3 = X%"), demand trends (growing/declining), dormant accounts. Note the current Abnormal Orders count and the deviation threshold dial.
4. **Order Simulator** — log in as a customer (locked to home branch), show the catalogue with live stock, then **place a deliberately large order** (~2–3× normal, but ≤ available so it commits). Watch the **confirmation cascade**: order saved → inventory reduced (−qty → new on hand) → "Customer Intelligence recalculated… demand opportunity detected" with deviation, inventory impact, risk score → "Recommendation Agent re-run: reorder N · transfer N" → "Investigation Alert sent" → "Chatbot context refreshed."
5. **Back to Customer Intelligence** — the new Significant-Opportunity/High-Demand panel is now at the top. Open the AI Investigation Report; walk the plain-English narrative, then expand Technical Details to prove the math is real.
6. **Email Alerts** — show the manager inbox: the narrative Customer Demand Intelligence / Abnormal Order Investigation Alert email, plus a low-stock alert if triggered. Also demo the Report Email Center (branch-wise inventory & sales reports).
7. **Chatbot** — ask "Any abnormal orders today?" and "What should we reorder?" — it answers using the order just placed, with numbers matching every screen.
8. **Close** — "From one order, your team now has a prioritized investigation, replenishment advice, an alerted manager, and a queryable assistant — automatically." Demonstrate the threshold dial (lower to flag a borderline order, raise to clear it) to show sensitivity is the client's to tune.

---

# 16. Business Benefits

**Operational benefits.** Compresses the order-to-decision cycle from days to seconds; one order automatically produces an investigation, replenishment advice, a manager alert, and a queryable assistant; eliminates manual reporting; consistent numbers remove cross-team reconciliation friction.

**Financial benefits.** Fewer stockouts (protected revenue and service levels), less overstock and dead capital, fewer emergency procurement buys, and recovered revenue from re-engaging dormant accounts.

**Inventory benefits.** Predictive depletion windows, network-wide stock visibility, automated low-stock/overstock detection, and transfer recommendations that rebalance stock before a shortfall bites.

**Customer benefits.** Customer-level demand visibility — who matters, who's growing or declining, who's draining fragile inventory, and who's dormant — enabling proactive account management and protected service levels for the accounts that matter most.

**Management benefits.** Executive AI summaries and narrative email briefings; a prioritized, explainable, audited recommendation queue with human-in-the-loop control; and a tunable sensitivity dial that puts judgment in the business's hands.

---

# 17. Future Enhancements

- **Customer-specific quantity controls** — extend the per-customer per-product cap into richer contract-aware ordering policies and approval thresholds.
- **Advanced demand forecasting** — move from directional split-window trends to true time-series forecasting once a longer order/sales history accrues.
- **Procurement optimization** — multi-supplier, lead-time-aware, cost-optimized reorder planning.
- **Supplier intelligence** — deeper supplier scorecards (reliability, lead-time variance, risk) feeding procurement.
- **Predictive stockout prevention** — proactive, forecast-driven stockout alerts ahead of the reorder point.
- **Approval workflows** — multi-step, role-based approval and audit for recommendation execution and large/abnormal orders.
- **Executive dashboards** — consolidated leadership KPI dashboards and scheduled executive reporting.
- **Mobile support** — responsive/mobile access for managers and field users.
- **Per-branch fulfilment scope** — flip `INVENTORY_SCOPE` from network to per-branch once the client confirms their fulfilment model (already supported by one flag).
- **Pinned JSON-capable LLM** — pin a specific OpenRouter/Gemini model for reliable chatbot tool selection.

---

# 18. PowerPoint Generation Guidance

> The deck is executive/client-facing. Use the **rebranded** language: "Customer Demand Insights" (not "Abnormal Order Detection"), "Demand Opportunity Review," "Customer Demand Intelligence Report." Refer to tables by business name, not `BZ_MOCK_*`. Lead each module slide with the business message, support with one or two visuals, and keep math in an appendix/backup slide. Recommended flow: Title → Executive Summary → Problem → Solution Overview → Architecture → each Module → End-to-End Workflow → Demo → Benefits → Roadmap → Closing.

**Slide 1 — Title**
- *Title:* "AI Retail Inventory Optimizer — Intelligent Supply-Chain Decision Support."
- *Visual:* clean hero with platform name, Bunzl-style navy/green theme, tagline "One order, fully understood."
- *Message:* premium AI platform over live Oracle data.

**Slide 2 — Executive Summary**
- *Title:* "From Raw Orders to Real-Time Decisions."
- *Visuals:* 5 value icons (inventory, customer intelligence, recommendations, chatbot, alerts).
- *Message:* live Oracle source of truth; one order ripples through the whole platform in seconds; every number agrees.
- *Charts:* none (icon row).

**Slide 3 — Business Problem**
- *Title:* "Six Costs of Flying Blind."
- *Visual:* 2×3 grid (stockouts, overstock, demand uncertainty, procurement inefficiency, customer visibility, manual reporting).
- *Message:* availability and visibility gaps cost revenue and capital.

**Slide 4 — Solution Overview**
- *Title:* "One Integrated Operations Cockpit."
- *Visual:* hub-and-spoke (platform center; 7 capability spokes).
- *Message:* seven cooperating capabilities, two invariants (single Oracle source, single inventory scope).

**Slide 5 — System Architecture**
- *Title:* "How It Fits Together."
- *Visual:* layered architecture diagram from Section 4.1 (Presentation → AI/MCP → Services → Data Access → Oracle).
- *Message:* clean separation; Oracle as single source of truth; consistent numbers by design.

**Slide 6 — Technology Stack**
- *Title:* "Built on Proven, Modern Foundations."
- *Visual:* logos/badges grouped by layer (Streamlit/Plotly · Python/FastAPI · Oracle · LangGraph/MCP/LLM · SMTP/PDF).
- *Message:* enterprise data (Oracle) + modern AI (LangGraph, MCP, LLM tool-calling).

**Slide 7 — Oracle Data Model**
- *Title:* "A Real Relational Backbone."
- *Visual:* the entity-relationship diagram (Section 6.2) in business names (Customers→Orders→Order Lines→Products; Branches; Inventory; Sales; Suppliers).
- *Message:* a genuine B2B order model, not a proxy — revenue is authoritative (`LINE_TOTAL_AMT`).

**Slide 8 — Inventory Intelligence**
- *Title:* "Predictive Inventory Health."
- *Visuals:* Inventory page KPI cards + understock/overstock tables + category donut.
- *Charts:* category-wise stock bar/donut; low-stock table with depletion windows.
- *Message:* prevent stockouts and clear overstock with predictive, network-wide visibility.
- *Screenshots:* Inventory page (KPIs, understock/overstock, AI insight).

**Slide 9 — Sales Intelligence**
- *Title:* "The Commercial Pulse."
- *Visuals:* sales trend line, category mix donut, top/bottom products.
- *Charts:* revenue trend; top-10 products bar.
- *Message:* automated revenue and demand insight; protect revenue-leading SKUs.
- *Screenshots:* Sales page (KPIs, trend, top products).

**Slide 10 — Customer Intelligence (anchor module — consider 2 slides)**
- *Title:* "Customer Demand Insights."
- *Visuals:* 5 executive flip cards; Customer Spotlight; order-history chart with the highlighted spike; risk-band summary.
- *Charts:* top customers leaderboard (contribution %); demand-trend split (growing/stable/declining); order-history bar with anomaly highlighted red.
- *Message:* who matters, who's growing/declining, which orders are risky, who's dormant — automatically, from live data.
- *Screenshots:* CI page header KPIs, an investigation panel (plain-English summary), the AI Investigation Report chart.
- *Backup slide:* the 5-factor risk score table and prior-only baseline math (Section 9).

**Slide 11 — Customer Order Simulator**
- *Title:* "Watch the Platform React — Live."
- *Visual:* the End-to-End Workflow chain (Section 14) as a vertical cascade.
- *Charts:* the confirmation cascade screenshot.
- *Message:* place one order, watch stock commit, demand re-judged, recommendations regenerate, manager alerted, chatbot updated — in seconds.
- *Screenshots:* login, catalogue with live stock, order confirmation cascade.

**Slide 12 — AI Recommendation Engine**
- *Title:* "From Signal to Prioritized Action."
- *Visual:* multi-agent diagram (inventory/pricing/transfer/risk/procurement → orchestrator → recommendation queue).
- *Charts:* recommendation queue with priority badges; recommendation-type breakdown.
- *Message:* prioritized, explainable, audited recommendations with human-in-the-loop approval.
- *Screenshots:* Recommendations page cards (reasoning expanded).

**Slide 13 — MCP Chatbot**
- *Title:* "Ask Your Data, in Plain English."
- *Visual:* MCP flow (question → tool selection → Oracle → answer) + a tool-catalog tag cloud.
- *Charts:* none; show a chat transcript.
- *Message:* self-serve analytics grounded only in real Oracle data; answers agree with every screen.
- *Screenshots:* Chatbot page with an example Q&A ("Any abnormal orders today?").

**Slide 14 — Email Automation**
- *Title:* "The Right Briefing, at the Right Moment."
- *Visual:* 4 email types (Low Stock, Inventory Report, Sales Report, Customer Demand Intelligence) with trigger/recipient/purpose.
- *Charts:* none; show email screenshots.
- *Message:* contradiction-free alerts and one-click branch reports; investigate risky orders before fulfilment.
- *Screenshots:* the Abnormal Order Investigation Alert email and a low-stock alert email.

**Slide 15 — End-to-End Workflow**
- *Title:* "One Order, Fully Understood."
- *Visual:* the full cascade (Section 14).
- *Message:* every figure reconciled through one scope, one Oracle source; analytics never corrupt the order.

**Slide 16 — Demo Walkthrough**
- *Title:* "See It Live."
- *Visual:* numbered 8-step demo flow (Section 15).
- *Message:* a repeatable, hands-on story the client can trigger themselves.

**Slide 17 — Business Benefits**
- *Title:* "What It's Worth."
- *Visual:* 5 columns (Operational, Financial, Inventory, Customer, Management).
- *Charts:* optional before/after (stockouts down, overstock down, reporting time down).
- *Message:* protected revenue, less dead capital, faster decisions, recovered dormant revenue.

**Slide 18 — Future Roadmap**
- *Title:* "Where We Go Next."
- *Visual:* roadmap timeline (forecasting, procurement optimization, supplier intelligence, approval workflows, executive dashboards, mobile, per-branch scope).
- *Message:* a clear, credible growth path.

**Slide 19 — Closing**
- *Title:* "AI Retail Inventory Optimizer."
- *Visual:* tagline + key talking points (every number agrees; no overselling; risk is demand *and* inventory; one order → full action list; sensitivity is yours to tune).
- *Message:* call to action.

**Screenshots checklist (capture before building the deck):** Agent Command Center (home), Inventory page, Sales page, Customer Intelligence (KPIs + investigation panel + AI report chart), Order Simulator (login, catalogue, confirmation cascade), Recommendations page, Chatbot Q&A, and the two key emails (Abnormal Order Investigation Alert + Low Stock Alert).
