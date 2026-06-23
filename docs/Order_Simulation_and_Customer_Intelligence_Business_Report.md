# Order Simulation & Customer Intelligence — Business Logic & Demo Guide

*A business-first deep dive for product owners, sales engineers, and anyone presenting these modules to a client. Code is referenced only where it changes the business story.*

---

## Quick Orientation

The platform runs on **Oracle as the single source of truth**. Two object families matter here:

- **Customer / Order model** — `BZ_MOCK_CUSTOMER → BZ_MOCK_ORDER_HEADER → BZ_MOCK_ORDER_LINE`. This is a real relational order model: a customer places orders, each order has line items, each line is a product + quantity + price.
- **Inventory model** — `BZ_MOCK_INVENTORY` (on-hand stock per product per branch), `BZ_MOCK_PRODUCT` (catalogue + pricing), `BZ_MOCK_BRANCH` (warehouses).

One concept underpins every number on screen: **inventory scope**. By default the platform runs in **network scope** — a product's "current inventory" is the *sum of stock across all branches*, and its reorder point is the *sum of per-branch reorder points*. So if Warehouse A has 100, B has 80, C has 56, the product's Current Inventory is **236 everywhere** — the Customer Intelligence card, the abnormal-order email, the simulator catalogue, the chatbot, and the recommendation engine all read that same figure. This consistency is deliberate and is enforced through one helper (`backend/services/inventory_scope.py`). When you demo, you never have to explain "why does this screen say 236 and that one say 100" — they always agree.

The two modules in this report are:

1. **Order Simulation** (`frontend/pages/6_Customer_Order_Simulator.py`) — the *cause*. A customer places an order.
2. **Customer Intelligence** (`frontend/pages/3_Customer_Intelligence.py`) — the *effect and the insight*. The business sees what that order means.

The bridge between them is the **post-order pipeline** (`backend/services/order_pipeline_service.py`).

---

# PART 1 — ORDER SIMULATION

## 1.1 Business Purpose

### Why would a customer use Order Simulation?

Order Simulation is a **live sandbox that lets you demonstrate the entire supply-chain intelligence platform in motion, on demand, in front of a client**. Instead of waiting for a real customer order to arrive and ripple through the system, the presenter (or the client themselves) logs in as a Bunzl end-customer, builds a basket, and places an order — and the platform reacts exactly as it would to a genuine order: stock is committed, demand patterns are re-evaluated, risk is scored, recommendations are regenerated, alerts are emailed, and the chatbot becomes aware of it.

In one sentence: **it turns an abstract "AI supply chain platform" into a tangible cause-and-effect story the client can trigger with their own hands.**

### What real-world business process does it represent?

It represents the **order-to-fulfilment commitment** moment in a distribution business — the instant a customer's purchase order is accepted and the warehouse commits stock to fulfil it. In a real Bunzl-style operation this is the trigger point where:

- inventory becomes *spoken for* (committed, then drawn down),
- demand signal is updated,
- exceptions (unusually large orders) need a human decision,
- replenishment planning may need to change.

The simulator compresses that whole chain into a few seconds so it can be seen.

### Who are the users of this module?

- **Primary (demo context):** the sales engineer / solution consultant presenting to a prospect, and the prospect themselves "test-driving" the system.
- **Business analogue:** a customer service rep / order desk, an account manager, or the customer's own procurement user placing an order through a portal.

### What decisions can be made using it?

The simulator itself doesn't *make* decisions — it *generates the events that drive decisions* everywhere else. After placing an order, the business can decide whether to:

- **Approve or investigate** an unusually large order before fulfilling it.
- **Expedite replenishment** for a product the order just pushed toward stockout.
- **Transfer stock** between locations to cover a shortfall.
- **Engage an account manager** when a customer's buying behaviour shifts.

> **Important nuance for the demo:** there are actually *two* order surfaces in the codebase. The login/catalogue page header and one helper still carry a legacy "simulation only — does not update inventory or trigger agents" caption, but the **actual order-placement path (`_place_order`) is fully live**: it writes to Oracle, draws down stock, recalculates Customer Intelligence, re-runs the recommendation agent, and sends the abnormal-order email. When presenting, treat **placing an order as a real, committed event** — because it is.

---

## 1.2 End-to-End Flow — What Happens When a Customer Places an Order

Below is every step from "customer clicks Place Order" to "the whole platform has reacted," with the **business meaning** of each step first.

### Step 0 — Login & catalogue browsing
**Business meaning:** A specific customer is now "in session." Everything that follows is attributed to *this* customer, which is what makes the downstream intelligence (their baseline, their behaviour, their risk) meaningful. The catalogue shows each product's price and **current network-wide stock**, with low-stock (<50) and out-of-stock (0) visually flagged — so the user is shopping against the real stock position, not a fiction.

### Step 1 — Cart building & pre-flight stock check
**Business meaning:** Before the order is even submitted, the system protects the business from the obvious mistake of ordering more than exists. As items are added, the cart sidebar compares requested quantity against available stock and, if any line exceeds stock, shows *"Not enough stock to place this order"* and **disables the Place Order button**. This is the first line of defence against overselling.

### Step 2 — Order placement (the transactional commit)
**Business meaning:** The warehouse formally commits to fulfil the order. Stock is no longer just "on hand" — it is now allocated to this customer.

*Technical reality (why it's trustworthy):* `oracle_writer.place_customer_order` does everything in **one database transaction**:
- prices every line from the authoritative catalogue,
- inserts one `ORDER_HEADER` + its `ORDER_LINE` rows,
- draws down inventory line by line with a **validated, locking decrement**.

If *any* line would drive stock below zero, the **entire transaction rolls back** — no partial order, no inventory drift. This is the real overselling guarantee (the pre-flight check in Step 1 is just a courtesy that blocks the obvious cases early).

### Step 3 — Inventory deduction
**Business meaning:** On-hand stock is reduced because the warehouse has now committed those units. Under network scope the order draws from the pooled stock across all branches so the deduction matches the network figure the customer saw. The confirmation screen shows, per product, `−ordered quantity → new units on hand`.

### Step 4 — Customer Intelligence recalculation (Stage 1 of the pipeline)
**Business meaning:** The business immediately re-asks the question *"given everything we knew before this order, is this order normal?"* The just-placed order is treated as the **latest** order for each of its products and is scored against that product's prior-only baseline. (Full mechanics in Part 2.) This is what produces the abnormal-order verdict, deviation %, inventory impact, and a 0–100 composite risk score.

### Step 5 — Risk evaluation
**Business meaning:** Each flagged line gets a severity band — **Low / Medium / High / Critical** — so a human knows whether to act now, review later, or ignore. The score blends five business factors (deviation, breaking the historical peak, inventory impact, reorder pressure, demand trend).

### Step 6 — Recommendation agent re-run (Stage 2 of the pipeline)
**Business meaning:** Because stock just changed, the platform regenerates its **reorder / transfer / supplier-risk** recommendations against the new levels. The Recommendations page now reflects the consequences of this order. The confirmation reports the new counts (e.g. "reorder 4 · transfer 2 · supplier risk 1").

### Step 7 — Email notification (Stage 3)
**Business meaning:** If the order is genuinely concerning (risk band Critical / High / Medium, **or** the product is now at/below reorder), a manager receives a narrative **Abnormal Order Investigation Alert** email — a business briefing that explains *why* the order is unusual and what to do, not a raw metric dump. Routine orders send nothing.

### Step 8 — Chatbot context refresh (Stage 4)
**Business meaning:** The order becomes *immediately* answerable by the assistant. The platform drops the chatbot's cached Oracle snapshot so the next question — *"Any abnormal orders today?"*, *"What was the latest customer order?"*, *"What should we reorder?"* — reflects this order at once, instead of waiting for a cache to expire.

### Step 9 — Dashboard impact
**Business meaning:** Every dependent page reloads fresh. The code calls `st.cache_data.clear()`, which invalidates every cached data loader, so the catalogue, Customer Intelligence, and Recommendations pages all re-read Oracle and the regenerated outputs on their next render. The order's effect is now visible platform-wide.

> **Resilience detail worth mentioning:** Stages 1–4 are all **best-effort**. The order is already committed to Oracle before the pipeline runs, and each stage is wrapped so that if (say) the email fails, the order and inventory remain correct and the failure is surfaced on screen rather than lost. The business never ends up with a committed order and corrupted analytics.

---

## 1.3 Order Validation Logic

**Why inventory is checked:** A distribution business lives or dies on not promising stock it doesn't have. Overselling means broken delivery promises, emergency procurement, and lost trust.

**Why an order can fail:** An order fails if a line's quantity exceeds available stock at the moment of commit.

**What happens if inventory is insufficient:**
- *Up front (pre-flight):* the cart warns and the Place Order button is disabled.
- *At commit (authoritative):* the locking decrement raises an error and the **whole transaction rolls back** — header, lines, and every decrement are undone together.

**How the system protects against overselling:** Validation happens **at the moment of decrement, under a database lock**, not as a separate "check then write" (which could race). Because the check and the deduction are the same atomic operation inside one transaction, two simultaneous orders can't both consume the last units.

*Business language first:* the warehouse will never commit stock it doesn't have, and it will never leave you with half an order. *Technical second:* atomic, validated, locking decrement inside a single rolled-back-on-failure transaction.

---

## 1.4 Abnormal Order Detection During Order Placement

**Why the system evaluates abnormal orders:** A sudden, far-larger-than-normal order is the single most useful early-warning signal in distribution. It can mean a great thing (a customer is expanding) or a dangerous thing (a one-off spike that will drain stock and starve other customers). Either way, **a human should look before the warehouse blindly fulfils it.**

**The business risk being identified:** silent stockouts caused by one outsized order, and undetected shifts in a customer's buying pattern.

**How the deviation threshold affects this:** The threshold (default **+50%** over the product's prior baseline, configurable 10–200% on the Customer Intelligence page) is the sensitivity dial. A lower threshold flags more orders (more caution, more noise); a higher threshold flags only dramatic spikes. The simulator reads the *same* threshold the user set on the Customer Intelligence page, so the two surfaces always agree.

**How high-risk orders/customers are identified:** Every flagged line gets a 0–100 composite risk score → band. The email fires for Critical/High/Medium, or whenever the product is now at/below reorder. A customer with *multiple* flagged lines is called out as showing "recurring abnormal ordering behaviour."

**Why alerts are generated:** So the right manager investigates *before* fulfilment causes a problem — confirm the order with the account manager, expedite replenishment, or plan a transfer.

**Examples (product baseline ≈ 100 units, threshold 50%):**
- Order **120** → +20%. Below threshold → **not abnormal**, no alert.
- Order **160** → +60%. Above 50% but below 3× threshold (150%) → **Medium/High**, investigation entry + email.
- Order **300** → +200%. Above the High line and likely above the historical max → **Critical**, urgent investigation, email, and very likely a stockout-risk flag.

---

## 1.5 Business Outcome — After an Order Is Placed

**New information now available:**
- The committed order and its exact revenue (from `LINE_TOTAL_AMT`, the authoritative figure).
- Updated, network-wide on-hand stock per product.
- A verdict on whether the order was normal, with deviation %, inventory impact %, and a risk score/band.
- Regenerated reorder/transfer/supplier-risk recommendations.
- An investigation email in the manager's inbox (if warranted).
- A chatbot that can now discuss the order.

**Actions the business can take:** approve and fulfil; hold and investigate an outsized order; expedite replenishment; transfer stock to cover a shortfall; flag the customer's account for a behaviour shift; re-plan procurement if demand looks structural.

---

# PART 2 — CUSTOMER INTELLIGENCE

## 2.1 Executive-Level Explanation (the 2-minute CEO pitch)

> **What it is:** Customer Intelligence turns your raw order history into a live answer to four questions every distribution leader cares about: *Who are my most valuable customers? Whose demand is growing or fading? Which orders are abnormal and risky right now? And which customers are quietly draining my most fragile inventory?*
>
> **Why leadership should care:** Revenue in distribution is concentrated and fragile. A handful of accounts drive most of the revenue, a single abnormal order can cause a stockout that damages every other customer's service level, and dormant or declining accounts are revenue leaking away unnoticed. This page surfaces all of that automatically, in plain English, from live Oracle data — no analyst, no spreadsheet.
>
> **What decisions it supports:** which accounts to protect and grow, which orders to investigate before fulfilling, where to pre-empt stockouts, and which customers to re-engage.

---

## 2.2 Business Logic of Every Section

The page is sourced live from Oracle and carries a header caption showing the **order window, order count, active buyers, and total customers on file** so the user knows the data's scope at a glance.

### 2.2.1 Executive KPIs (5 flip cards)

Five headline cards; hover/tap flips each to reveal supporting detail.

**1. Top Customer**
- *What it shows:* the single highest-revenue customer and their revenue.
- *How it's calculated:* sum of `LINE_TOTAL_AMT` per customer, ranked; back face adds contribution %, orders, units, segment/tier.
- *Why it matters:* identifies the account you can least afford to lose.
- *Action:* prioritise retention, service quality, and executive relationship.

**2. Highest Growth**
- *What it shows:* the customer with the largest positive revenue change.
- *How it's calculated:* order window split in half; second-half revenue vs first-half.
- *Why it matters:* spotlights momentum and expansion accounts.
- *Action:* invest in the relationship while it's accelerating; ensure supply can keep up.

**3. Abnormal Orders**
- *What it shows:* count of order lines flagged above the deviation threshold.
- *How it's calculated:* per-product prior-only baseline (see 2.3); severity graded High vs Medium.
- *Why it matters:* the day's exception queue — the orders a human should look at.
- *Action:* investigate the largest deviations first.

**4. Stockout-Risk Customers**
- *What it shows:* how many customers are ordering products that are at/below reorder point.
- *How it's calculated:* customers whose orders touch "at-risk" products, scored 0–100 by at-risk volume.
- *Why it matters:* connects *customer behaviour* to *inventory fragility* — the customers most likely to trigger a stockout.
- *Action:* prioritise replenishment/transfers for the products these customers are draining.

**5. Dormant Accounts**
- *What it shows:* active customers with zero orders.
- *How it's calculated:* `active_flg='Y'` customers with no rows in the order table.
- *Why it matters:* revenue opportunity sitting idle — *not* an error.
- *Action:* targeted re-engagement / win-back outreach.

### 2.2.2 Customer Spotlight
- *What it shows:* top 5 customers by order revenue, as cards (revenue, orders, segment, tier).
- *How it's calculated:* same revenue ranking as Top Customer, top 5.
- *Why it matters:* a fast visual of your strategic-account roster.
- *Action:* ensure each has an owner and a retention plan.

### 2.2.3 Abnormal Order Detection — the "AI Order Intelligence Center" (the hero section)
This is the centrepiece and the most demo-worthy part. It contains:

- **A deviation-threshold control** (10–200%, default 50) that re-derives every anomaly figure on the page.
- **AI Executive Summary panel:** abnormal orders detected, customers impacted, products impacted, **revenue exposure** (sum of flagged line revenue), and the highest-risk customer/product.
- **A risk-band summary** (Critical / High / Medium / Low counts).
- **One full-width investigation panel per abnormal order**, ordered Critical→High→Medium→Low. Each leads with a **plain-English Executive Summary** ("X typically orders ~A units; the latest order was B units, making it [qualifier] previous orders…") and a risk badge — *no formulas on the face*.
- **An expandable AI Investigation Report** per order: a complete order-history chart (every order as a bar, the abnormal one highlighted red at its true position), then narrative blocks — *What Happened, Inventory Impact, Product Demand Context, Customer Behaviour Assessment* (deliberately hedged hypotheses: "may indicate", "could suggest"), *Business Recommendation*, and a collapsed **Technical Details** drill-down (risk score, ratios, coverage, the exact formula).
- **A detailed table** of every flagged line.

*Why it matters:* it converts a statistical anomaly into a briefing a manager can act on in seconds, with the maths available but never in the way.
*Action:* work the panels top-down (Critical first); confirm the order, check inventory impact, expedite/transfer as advised.

### 2.2.4 Top Products
- *What it shows:* top 10 products by revenue and by quantity (two charts).
- *How it's calculated:* grouped sums of revenue (`line_total`) and quantity.
- *Why it matters:* reveals which SKUs carry the business and which move in volume.
- *Action:* protect availability of the revenue-leading SKUs; watch margin on the volume leaders.

### 2.2.5 Top Customers (table)
- *What it shows:* leaderboard — orders, units, revenue, contribution %.
- *How it's calculated:* per-customer aggregation; contribution = customer revenue ÷ total revenue.
- *Why it matters:* quantifies revenue concentration (the demo talking point: "top 3 = X% of revenue").
- *Action:* tier service levels by contribution.

### 2.2.6 Customer Demand Trends
- *What it shows:* every customer bucketed **Growing / Stable / Declining**.
- *How it's calculated:* order window split at its midpoint; second-half revenue vs first-half; **>+10% = Growing, <−10% = Declining, else Stable**; brand-new customers (no first-half activity) count as Growing.
- *Why it matters:* directional early-warning on the book of business.
- *Action:* see 2.5.

### 2.2.7 Inventory Impact Analysis
- *What it shows:* customers driving inventory pressure by ordering at-risk products, with a 0–100 pressure score and the affected products.
- *How it's calculated:* "at-risk" = stock at/below reorder point (network scope); sum of at-risk units/revenue per customer; score relative to the heaviest contributor.
- *Why it matters:* directly links a *customer's* behaviour to a *stockout* risk — the join most analytics tools miss.
- *Action:* prioritise replenishment/transfers on the products these top-pressure customers consume; pre-warn affected customers.

### 2.2.8 Dormant Accounts (table)
- *What it shows:* active customers who have never ordered (segment, tier, industry, city, credit limit, signup date).
- *Why it matters:* a ready-made re-engagement list.
- *Action:* prioritise by credit limit / segment for win-back.

### 2.2.9 AI Insights
- *What it shows:* a handful of concise, data-grounded sentences (top customer + concentration, top products, the largest abnormal order, dormant count, biggest inventory-pressure customer, growing vs declining counts).
- *How it's calculated:* generated directly from the same computed frames — every claim is backed by a real number, nothing invented.
- *Why it matters:* an instant executive narrative of the whole page.
- *Action:* use as the talk-track / email summary to leadership.

---

## 2.3 Abnormal Order Detection — In Depth

This is the analytical heart of the platform, so it's worth understanding precisely.

**How the baseline is established:** For each product, the system builds a **prior-orders-only baseline** — for any given order, the mean/max/min/std are computed from that product's orders that occurred **strictly before** it in time. The order being judged is **never part of its own baseline.**

**Why prior order history is used:** If you included the current order in its own average, a huge spike would inflate the baseline and partly hide itself. By judging each order only against what was known *before* it, the question stays honest: *"Given everything we knew before today, is the newest order abnormal enough to need attention?"* This logic lives in a shared `prior_order_baseline` helper used by both the page and the post-order pipeline, so the page and the simulator can never disagree.

**Only the most recent order per product is evaluated.** Older spikes remain part of the history (they're not re-flagged on their own); they become context for judging the latest order. A product with only a single order has no prior history and is **never flagged**.

**What qualifies as abnormal:** the latest order must be **both**:
1. at least `min_deviation_pct` (default 50%) above the prior-only mean, **and**
2. at least **5 units** above the mean (an absolute floor so tiny-mean products don't flag on noise).

**How deviation percentage works:** `deviation = (current − prior_mean) / prior_mean × 100`.

**Why different thresholds matter / severity grading:** "High" risk is graded at **3× the configured threshold** (so at the 50% default, ≥150% deviation = High, otherwise Medium). Severity scales with the chosen sensitivity rather than a fixed cut-off.

**The composite 0–100 risk score** (used for the bands and the email) blends five business factors:
| Component | Weight | Business meaning |
|---|---|---|
| Deviation from average | 30% | How far above its own normal this order is |
| Increase above historical max | 25% | Whether it breaks the product's all-time peak |
| Inventory impact | 20% | Share of on-hand stock this one order consumes |
| Reorder-point pressure | 15% | Whether the product is already at/below reorder |
| Recent demand trend | 10% | Whether demand was already accelerating |

Bands: **0–30 Low · 31–60 Medium · 61–85 High · 86–100 Critical.** The simulator's pipeline mirrors this exact formula, so the number shown after placing an order equals the number on the page.

**Worked examples — normal pattern 100 units/week, threshold 50%:**

| Scenario | Deviation | +5-unit floor? | Verdict | Why |
|---|---|---|---|---|
| Orders **120** | +20% | yes | **Normal** | Below the 50% threshold |
| Orders **150** | +50% | yes | **Abnormal — Medium** | Exactly meets 50%; below the 150% High line |
| Orders **200** | +100% | yes | **Abnormal — Medium/High** | Well over threshold; High if ≥150% |
| Orders **300** | +200% | yes | **Abnormal — High/Critical** | Over the 3× High line; Critical if it also breaks the max / drains stock |

The same 200-unit order can land **Medium or Critical** depending on inventory: if 200 units is 90% of remaining stock or the product is already at reorder, the inventory-impact and reorder components push the composite score into Critical even at the same deviation. **Risk is demand *and* inventory, not deviation alone** — a key talking point.

---

## 2.4 Inventory Impact Analysis — In Depth

**Why customer ordering behaviour affects inventory planning:** Inventory pressure isn't random — it's *driven by specific customers ordering specific products*. If you only watch stock levels, you see *that* a product is draining; you don't see *who* is draining it or whether it will continue. Joining customer behaviour to inventory tells you both.

**How at-risk products are identified:** a product is at-risk when its **network-wide stock is at or below its network-wide reorder point** (same scope as everywhere else).

**Why stockout-risk customers matter:** these are the customers whose orders are actively consuming fragile stock. They are simultaneously your *risk* (most likely to cause a stockout) and your *signal* (where to direct replenishment). 

**Example:** Product X is at reorder. Customers A, B, C all buy it; A has bought 800 at-risk units this window, B 300, C 100. A scores 100 (relative to the heaviest), B ~38, C ~13. The business now knows: expedite Product X *because of A*, and proactively talk to A about lead times.

---

## 2.5 Demand Trends — In Depth

The order window is split at its midpoint and each customer's second-half revenue is compared to the first half.

| Category | Rule | Business meaning | Action |
|---|---|---|---|
| **Growing** | change **> +10%** (or brand-new buyer) | account expanding | invest in the relationship, secure supply for rising demand, explore upsell |
| **Stable** | within **±10%** | steady, predictable | maintain service; low-touch retention |
| **Declining** | change **< −10%** | account cooling — churn risk | account manager outreach to find the cause before it's lost |

Because order histories are short, this is positioned honestly as a **directional signal, not a forecast.**

---

## 2.6 Dormant Accounts — In Depth

**Why they matter:** an active, credit-approved customer who has never ordered is *latent revenue* — onboarding effort already spent, nothing returned. They're easy to miss because they don't appear in any sales report (they have no sales).

**How they're identified:** active (`active_flg='Y'`) customers with zero rows in the order table.

**Actions:** prioritise outreach by credit limit and segment; treat as a win-back / activation campaign list. The page explicitly frames this as "an opportunity, not an error."

---

## 2.7 AI Insights — In Depth

**What's generated:** concise sentences covering the top customer and revenue concentration, the top revenue products, the single largest abnormal order (with real numbers), the dormant-account count and examples, the biggest inventory-pressure customer, and demand momentum (growing vs declining counts).

**What data they use:** the exact same computed frames that drive the rest of the page — there is **no separate model inventing claims**; every insight is grounded in a number you can see elsewhere on the page.

**How to interpret them:** as the executive summary / talk-track for the page. If an insight says "the top 3 customers account for 62% of revenue," that's your concentration-risk headline.

---

# PART 3 — HOW THE TWO MODULES WORK TOGETHER

This is the core story. Order Simulation is the **cause**; Customer Intelligence is the **effect and the insight**. The post-order pipeline (`order_pipeline_service.py`) is the wiring.

## 3.1 The Chain

```
Customer places order (Order Simulator)
        │
        ▼
ORDER_HEADER + ORDER_LINE written  ──►  Inventory drawn down (atomic, validated)
        │
        ▼
Stage 1: Customer Intelligence recalculated
   • new order treated as the LATEST order for its products
   • scored vs prior-only baseline at the user's threshold
   • deviation %, inventory impact %, composite risk score + band
        │
        ▼
Stage 2: Recommendation agent re-run
   • reorder / transfer / supplier-risk recomputed on new stock
        │
        ▼
Stage 3: Abnormal Order Investigation Alert email
   • sent if band ∈ {Critical, High, Medium} OR product now at/below reorder
        │
        ▼
Stage 4: Chatbot context refreshed (order answerable immediately)
        │
        ▼
All caches cleared ──► Customer Intelligence page, Recommendations page,
                       catalogue all reload fresh
        │
        ▼
Business decision: approve / investigate / expedite / transfer / re-engage
```

**How one module influences the other, precisely:**
- The order **changes the data** Customer Intelligence reads (a new latest order + lower stock).
- Customer Intelligence **re-judges** that order against history → risk verdict.
- That verdict **drives** the email (alert or silence) and feeds the recommendation re-run (lower stock → new reorder/transfer advice).
- Cache-clear ensures the **dashboard** shows it all on next render.
- Every figure is reconciled through **one inventory scope**, so the simulator's "new stock," the CI card's "current inventory," the email's "current inventory," and the chatbot's answer are the **same number**.

## 3.2 Real-World Scenario

**Setup:** Customer "Northwind Facilities" normally orders ~100 units of *Blue Nitrile Gloves (Box)* each week (recent history `90 → 110 → 95 → 105`). The product has 230 units on hand network-wide, reorder point 120. Deviation threshold is set to 50%.

**Today Northwind orders 250 units.**

1. **Order Simulation** prices the line, writes the order, and draws stock down: **230 → 0… wait — 250 > 230.** The pre-flight check flags the shortfall and the Place Order button is disabled; at commit it would roll back. *Demo variant:* make it **200** units so it commits, leaving **30 on hand**.

   With **200 units committed**, stock is now **30** (below the 120 reorder point).

2. **What the system detects (Stage 1):**
   - Prior mean ≈ 100 → deviation **+100%** (≥50% threshold and ≥5-unit floor → abnormal).
   - It's the largest order ever recorded for the product (breaks the historical max).
   - Inventory impact: 200 of the pre-order 230 = **~87% of on-hand stock**.
   - Product is now **at/below reorder**.
   - Composite score → high deviation (30%) + breaks max (25%) + huge inventory impact (20%) + reorder pressure (15%) → **Critical**.

3. **What changes in Customer Intelligence:** a new **Critical** investigation panel appears at the top: *"Northwind typically orders around 100 units of Blue Nitrile Gloves. The latest order was for 200 units, making it significantly larger than previous orders — the largest order recorded for this product. Because the product is already near its reorder threshold, this order is significantly outside normal behaviour and needs immediate attention to avoid stockout risk."* The Abnormal Orders KPI ticks up; Northwind appears in Inventory Impact Analysis; revenue exposure rises.

4. **What recommendations are generated (Stage 2):** the agent now sees 30 units vs reorder 120 and emits a **reorder** (and possibly a **transfer** from a lower-demand location) for the gloves.

5. **What alerts are triggered (Stage 3):** a manager receives the **Abnormal Order Investigation Alert** email — executive summary, the `90 → 110 → 95 → 105 → 200` pattern with the spike highlighted, deviation +100%, inventory impact ~87%, stockout risk Elevated, reasoning bullets, possible business reasons, and recommended actions.

6. **Chatbot (Stage 4):** *"Any abnormal orders today?"* now returns Northwind's gloves order.

7. **Business actions:** confirm with Northwind whether this is a one-off or a new contract; expedite glove replenishment; transfer stock to cover other customers; monitor Northwind's next 7–14 days for a pattern shift.

---

# PART 4 — CLIENT DEMONSTRATION GUIDE

## 4.1 The Story to Tell

> *"Your order history is full of signals you can't act on fast enough — the abnormal order that quietly causes a stockout, the customer whose demand is shifting, the fragile product that one big order can drain. We'll place a single order live, and you'll watch the entire platform react — committing stock, re-judging the order against its own history, regenerating replenishment advice, emailing the manager, and updating the assistant — in seconds."*

## 4.2 Ideal Demo Flow

1. **Open Customer Intelligence first.** Set the scene: top customers, revenue concentration ("top 3 = X%"), demand trends, dormant accounts. Note the current Abnormal Orders count and the deviation threshold.
2. **Show the catalogue & current stock** in the Order Simulator — pick a product and note its on-hand units and that it has a clear order history.
3. **Place a deliberately large order** for that product (≈2–3× its normal size; large enough to flag, but ≤ available stock so it commits).
4. **Watch the confirmation cascade:** order saved → inventory reduced (−qty → new on hand) → "Customer Intelligence recalculated… abnormal order detected" with deviation, inventory impact, and risk score → "Recommendation Agent re-run: reorder N · transfer N" → "Abnormal Order Investigation Alert sent" → "Chatbot context refreshed."
5. **Return to Customer Intelligence** — the new Critical/High panel is now at the top. Open the AI Investigation Report; walk the plain-English narrative, then expand Technical Details to show the maths is real.
6. **Show the email** in the manager inbox — the narrative briefing.
7. **Ask the chatbot** *"Any abnormal orders today?"* / *"What should we reorder?"* — it answers using the order you just placed.
8. **Close on the business decision:** "From one order, your team now has a prioritised investigation, replenishment advice, an alerted manager, and a queryable assistant — automatically."

**Tip:** pre-set the deviation threshold to demonstrate sensitivity — drop it to flag a borderline order, raise it to clear it — to show the dial is the client's to tune.

---

# PART 5 — PRESENTATION PREPARATION

## 5.1 Executive Summary
Order Simulation and Customer Intelligence together demonstrate a closed loop: a customer order is captured and committed against live inventory, then automatically re-judged against the product's own demand history, scored for risk, turned into replenishment recommendations, escalated by email when it matters, and made queryable by an assistant — all reconciled to one consistent set of inventory numbers, sourced live from Oracle. It compresses the order-to-decision cycle of a distribution business into seconds, and makes it visible.

## 5.2 Business Summary
- **Order Simulation** = the live trigger: place an order, commit stock safely (no overselling, no partial orders).
- **Customer Intelligence** = the insight layer: who matters, who's growing/declining, which orders are abnormal and risky, who's draining fragile inventory, who's dormant.
- **Together** they turn a single order into a prioritised action list. Value: fewer stockouts, faster exception handling, protected revenue, and re-engagement opportunities surfaced automatically.

## 5.3 Technical Summary
- **Single source of truth:** Oracle (`BZ_MOCK_*`). **Single inventory scope** (`inventory_scope.py`, network by default) keeps every surface's stock figure identical.
- **Order placement:** one atomic transaction (header + lines + validated locking inventory decrement); rolls back entirely on insufficient stock.
- **Abnormal detection:** per-product **prior-orders-only** baseline (current order excluded from its own baseline), latest-order-only evaluation, deviation + 5-unit floor, configurable threshold, 3× High grading, 5-factor 0–100 composite score with 4 bands.
- **Post-order pipeline:** Stage 1 CI recalc → Stage 2 recommendation agent re-run → Stage 3 abnormal-order email (Critical/High/Medium or at-reorder) → Stage 4 chatbot context refresh → cache clear. Stages are **best-effort over an already-committed order**.

## 5.4 Key Client Talking Points
- "Every number agrees — page, email, simulator, chatbot — because they all read one inventory figure."
- "We never oversell: orders are validated as stock is decremented, inside one transaction that rolls back on failure."
- "An order is never judged against itself — we score it on everything we knew *before* it arrived."
- "Risk is demand *and* inventory: the same 200-unit order can be Medium or Critical depending on stock."
- "One order → investigation, replenishment advice, manager alert, and an updated assistant — automatically."
- "The sensitivity dial is yours: tune how aggressively the system flags."

## 5.5 Frequently Asked Client Questions & Suggested Answers

**Q: Is the data real or hard-coded?**
A: Live from Oracle. Placing an order writes real rows and draws down real stock; every downstream number is recomputed from that data.

**Q: Could this oversell stock?**
A: No. Stock is validated as it's decremented inside a single transaction; if any line is short, the whole order rolls back. Nothing partial is ever persisted.

**Q: Why was this order flagged but that one wasn't?**
A: Each order is compared only to that product's *prior* orders. If it's at least [threshold]% above the prior average and at least 5 units higher, it's flagged. You control the threshold.

**Q: Why is the same deviation sometimes Medium and sometimes Critical?**
A: The risk score blends deviation with inventory impact, whether it breaks the historical peak, reorder pressure, and demand trend. An order that drains nearly all stock or hits a product already at reorder scores far higher than the same percentage on a well-stocked product.

**Q: Does the current order distort its own baseline?**
A: Never. The baseline is computed strictly from orders that came before it.

**Q: What triggers an email vs not?**
A: Critical, High, or Medium risk, or the product dropping to/below its reorder point. Routine orders send nothing — no alert fatigue.

**Q: What if the email or recommendation step fails?**
A: The order and inventory are already safely committed first; the analytics stages are best-effort and surface any failure on screen, so your data is never left inconsistent.

**Q: "Network-wide stock" — what does that mean?**
A: A temporary, configurable business rule: a product's stock is the total across all branches. It flips to per-branch with one setting when you confirm your fulfilment model.

**Q: Why are some customers shown as dormant?**
A: They're active, credit-approved accounts that haven't ordered — a re-engagement opportunity, not a data error.

**Q: How current is the chatbot?**
A: Immediately current — placing an order refreshes its data snapshot, so it can answer about that order at once.

**Q: Can we tune the sensitivity?**
A: Yes, live, from 10% to 200% deviation; the whole page (and the simulator's evaluation) re-derives instantly.
