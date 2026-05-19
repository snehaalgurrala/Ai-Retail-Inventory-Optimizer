from __future__ import annotations

from datetime import date, datetime
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROCESSED_DATA_DIR / "reports"
REPORT_LOG_PATH = REPORTS_DIR / "report_log.csv"
AGENT_LABELS = {
    "inventory_agent": "Inventory Agent",
    "pricing_agent": "Pricing Agent",
    "transfer_agent": "Transfer/Supply Agent",
    "risk_agent": "Risk Agent",
    "procurement_agent": "Procurement Agent",
}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _load_report_data() -> dict[str, pd.DataFrame]:
    return {
        "inventory": _read_csv(RAW_DATA_DIR / "inventory.csv"),
        "sales": _read_csv(RAW_DATA_DIR / "sales.csv"),
        "products": _read_csv(RAW_DATA_DIR / "products.csv"),
        "stores": _read_csv(RAW_DATA_DIR / "stores.csv"),
        "suppliers": _read_csv(RAW_DATA_DIR / "suppliers.csv"),
        "recommendations": _read_csv(PROCESSED_DATA_DIR / "recommendations.csv"),
        "agent_outputs": _read_csv(PROCESSED_DATA_DIR / "agent_outputs.csv"),
        "orchestrator_summary": _read_csv(PROCESSED_DATA_DIR / "orchestrator_summary.csv"),
        "overstock": _read_csv(PROCESSED_DATA_DIR / "overstock_items.csv"),
        "low_stock": _read_csv(PROCESSED_DATA_DIR / "low_stock_alerts.csv"),
    }


def _normalize_date(value: Any) -> pd.Timestamp | None:
    if value in (None, ""):
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.normalize()


def _format_date(value: Any) -> str:
    parsed = _normalize_date(value)
    if parsed is None:
        return "All dates"
    return parsed.strftime("%d %b %Y")


def _filter_date_range(
    df: pd.DataFrame,
    date_column: str,
    start_date: Any,
    end_date: Any,
) -> pd.DataFrame:
    if df.empty or date_column not in df.columns:
        return df.copy()

    filtered = df.copy()
    filtered["_report_date"] = pd.to_datetime(filtered[date_column], errors="coerce")
    start = _normalize_date(start_date)
    end = _normalize_date(end_date)
    if start is not None:
        filtered = filtered[filtered["_report_date"] >= start]
    if end is not None:
        filtered = filtered[filtered["_report_date"] <= end]
    return filtered.drop(columns=["_report_date"], errors="ignore")


def _clean_text(value: Any, fallback: str = "") -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return fallback
    text = str(value or "").strip()
    return text or fallback


def _norm_text(value: Any) -> str:
    return _clean_text(value).casefold()


def _resolve_store_id(branch_filter: str, stores: pd.DataFrame) -> str:
    branch_text = _clean_text(branch_filter)
    if not branch_text or _norm_text(branch_text) == "all branches":
        return "All Branches"
    if stores.empty:
        return branch_text

    stores_view = stores.copy()
    for column in ["store_id", "store_name", "city"]:
        if column not in stores_view.columns:
            stores_view[column] = ""
        stores_view[column] = stores_view[column].fillna("").astype(str).str.strip()

    normalized = _norm_text(branch_text)
    for column in ["store_id", "store_name", "city"]:
        match = stores_view[stores_view[column].map(_norm_text).eq(normalized)]
        if not match.empty:
            return str(match.iloc[0].get("store_id", "")).strip()

    contains_match = stores_view[
        stores_view["store_name"].map(lambda value: _norm_text(value) in normalized)
        | stores_view["city"].map(lambda value: bool(_norm_text(value)) and _norm_text(value) in normalized)
    ]
    if not contains_match.empty:
        return str(contains_match.iloc[0].get("store_id", "")).strip()

    return branch_text


def _filter_branch(
    df: pd.DataFrame,
    branch_filter: str,
    stores: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if df.empty or "store_id" not in df.columns:
        return df.copy()
    resolved_store_id = _resolve_store_id(str(branch_filter), stores if stores is not None else pd.DataFrame())
    if _norm_text(resolved_store_id) == "all branches":
        return df.copy()
    return df[df["store_id"].fillna("").astype(str).str.strip().eq(str(resolved_store_id).strip())].copy()


def _store_label(branch_filter: str, stores: pd.DataFrame) -> str:
    resolved_store_id = _resolve_store_id(str(branch_filter), stores)
    if _norm_text(resolved_store_id) == "all branches":
        return "All Branches"
    if stores.empty or not {"store_id", "store_name"}.issubset(stores.columns):
        return str(branch_filter)
    match = stores[stores["store_id"].fillna("").astype(str).str.strip().eq(str(resolved_store_id).strip())]
    if match.empty:
        return str(branch_filter)
    row = match.iloc[0]
    city = str(row.get("city", "") or "").strip()
    suffix = f", {city}" if city else ""
    return f"{row.get('store_name', branch_filter)} ({resolved_store_id}{suffix})"


def _enrich(df: pd.DataFrame, products: pd.DataFrame, stores: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    if not products.empty and "product_id" in enriched.columns and "product_id" in products.columns:
        product_columns = [
            column
            for column in ["product_id", "product_name", "category", "cost_price", "selling_price"]
            if column in products.columns and (column == "product_id" or column not in enriched.columns)
        ]
        enriched = enriched.merge(products[product_columns], on="product_id", how="left")
    if not stores.empty and "store_id" in enriched.columns and "store_id" in stores.columns:
        store_columns = [
            column
            for column in ["store_id", "store_name", "city"]
            if column in stores.columns
        ]
        enriched = enriched.merge(stores[store_columns], on="store_id", how="left")
    return enriched


def _numeric(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(series, errors="coerce").fillna(0)


def _money(value: float) -> str:
    return f"Rs. {float(value):,.2f}"


def _save_attachment(df: pd.DataFrame, report_type: str) -> Path | None:
    if df.empty:
        return None
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = REPORTS_DIR / f"{report_type}_report_{timestamp}.csv"
    df.to_csv(path, index=False)
    return path


def _write_report_log(row: dict[str, Any]) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    log_df = _read_csv(REPORT_LOG_PATH)
    output = pd.concat([log_df, pd.DataFrame([row])], ignore_index=True)
    output.to_csv(REPORT_LOG_PATH, index=False)


def _table_html(df: pd.DataFrame, columns: list[str], limit: int = 10) -> str:
    visible_columns = [column for column in columns if column in df.columns]
    if df.empty or not visible_columns:
        return "<p style='margin:0;color:#64748b;'>No rows available for this section.</p>"

    header = "".join(f"<th>{escape(str(column).replace('_', ' ').title())}</th>" for column in visible_columns)
    rows = []
    for _, row in df.head(limit).iterrows():
        cells = "".join(f"<td>{escape(str(row.get(column, '')))}</td>" for column in visible_columns)
        rows.append(f"<tr>{cells}</tr>")
    return f"""
    <table class="report-table">
      <thead><tr>{header}</tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
    """


def _summary_table(df: pd.DataFrame, group_columns: list[str], value_column: str) -> pd.DataFrame:
    if df.empty or value_column not in df.columns:
        return pd.DataFrame()
    usable_groups = [column for column in group_columns if column in df.columns]
    if not usable_groups:
        return pd.DataFrame()
    summary = df.copy()
    summary[value_column] = _numeric(summary[value_column])
    return (
        summary.groupby(usable_groups, dropna=False, as_index=False)[value_column]
        .sum()
        .sort_values(value_column, ascending=False)
    )


def _recommendations_for_branch(recommendations: pd.DataFrame, branch_filter: str) -> pd.DataFrame:
    if recommendations.empty:
        return pd.DataFrame()
    filtered = recommendations.copy()
    if branch_filter and branch_filter != "All Branches" and "store_id" in filtered.columns:
        filtered = filtered[filtered["store_id"].astype(str).eq(str(branch_filter))]
    if "priority" in filtered.columns:
        rank = {"high": 0, "medium": 1, "low": 2}
        filtered["_priority_rank"] = filtered["priority"].fillna("").astype(str).str.lower().map(rank).fillna(3)
        filtered = filtered.sort_values(["_priority_rank"]).drop(columns=["_priority_rank"], errors="ignore")
    return filtered


def _inventory_date_column(inventory: pd.DataFrame) -> str:
    for column in ["date", "last_updated", "inventory_date"]:
        if column in inventory.columns:
            parsed = pd.to_datetime(inventory[column], errors="coerce")
            if parsed.notna().any():
                return column
    return ""


def _priority_count(recommendations: pd.DataFrame, priority: str = "high") -> int:
    if recommendations.empty or "priority" not in recommendations.columns:
        return 0
    return int(recommendations["priority"].fillna("").astype(str).str.casefold().eq(priority.casefold()).sum())


def _low_stock_by_branch(low_stock: pd.DataFrame) -> pd.DataFrame:
    if low_stock.empty or "store_name" not in low_stock.columns:
        return pd.DataFrame(columns=["store_name", "low_stock_count"])
    return (
        low_stock.groupby("store_name", dropna=False)
        .size()
        .reset_index(name="low_stock_count")
        .sort_values("low_stock_count", ascending=False)
    )


def _overstock_by_branch(overstock: pd.DataFrame) -> pd.DataFrame:
    if overstock.empty or "store_name" not in overstock.columns:
        return pd.DataFrame(columns=["store_name", "overstock_count"])
    return (
        overstock.groupby("store_name", dropna=False)
        .size()
        .reset_index(name="overstock_count")
        .sort_values("overstock_count", ascending=False)
    )


def _email_count(value: Any) -> str:
    return f"{int(value or 0):,}"


def _email_badge(text: Any, color: str = "#2563eb", background: str = "#dbeafe") -> str:
    return (
        f"<span style='display:inline-block;padding:5px 10px;border-radius:999px;"
        f"font-size:11px;font-weight:700;color:{color};background:{background};'>"
        f"{escape(_clean_text(text, 'Info').title())}</span>"
    )


def _email_dashboard_card(
    title: str,
    count: Any,
    insight: str,
    accent: str,
    background: str,
) -> str:
    return f"""
    <td style="width:25%;padding:6px;vertical-align:top;">
      <div style="background:{background};border:1px solid rgba(15,23,42,0.08);border-left:5px solid {accent};border-radius:14px;padding:14px;min-height:112px;">
        <div style="font-size:12px;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:0.03em;">{escape(title)}</div>
        <div style="font-size:28px;line-height:34px;font-weight:800;color:#0f172a;margin-top:8px;">{_email_count(count)}</div>
        <div style="font-size:12px;line-height:17px;color:#475569;margin-top:6px;">{escape(insight)}</div>
      </div>
    </td>
    """


def _email_table(
    df: pd.DataFrame,
    columns: list[str],
    labels: dict[str, str] | None = None,
    limit: int = 6,
) -> str:
    labels = labels or {}
    visible_columns = [column for column in columns if column in df.columns]
    if df.empty or not visible_columns:
        return (
            "<div style='background:#f8fafc;border:1px dashed #cbd5e1;border-radius:12px;"
            "padding:14px;color:#64748b;font-size:13px;'>No rows available for this section.</div>"
        )

    header_cells = "".join(
        f"<th style='padding:10px 9px;text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:0.03em;color:#475569;background:#f8fafc;border-bottom:1px solid #e2e8f0;'>{escape(labels.get(column, column.replace('_', ' ').title()))}</th>"
        for column in visible_columns
    )
    body_rows = []
    for _, row in df.head(limit).iterrows():
        cells = []
        for column in visible_columns:
            value = _clean_text(row.get(column))
            if column == "priority":
                priority = value.casefold()
                color, background = {
                    "high": ("#991b1b", "#fee2e2"),
                    "medium": ("#92400e", "#fef3c7"),
                    "low": ("#1d4ed8", "#dbeafe"),
                }.get(priority, ("#334155", "#e2e8f0"))
                value_html = _email_badge(value, color, background)
            else:
                value_html = escape(value)
            cells.append(
                f"<td style='padding:10px 9px;font-size:12px;line-height:17px;color:#0f172a;border-bottom:1px solid #edf2f7;vertical-align:top;'>{value_html}</td>"
            )
        body_rows.append(f"<tr>{''.join(cells)}</tr>")

    return f"""
    <div style="overflow:hidden;border:1px solid #e2e8f0;border-radius:14px;background:#ffffff;">
      <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="border-collapse:collapse;width:100%;">
        <thead><tr>{header_cells}</tr></thead>
        <tbody>{''.join(body_rows)}</tbody>
      </table>
    </div>
    """


def _email_section(title: str, subtitle: str, body: str) -> str:
    return f"""
    <div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:16px;padding:18px;margin-top:16px;box-shadow:0 8px 22px rgba(15,23,42,0.05);">
      <div style="font-size:17px;font-weight:800;color:#0f172a;margin-bottom:4px;">{escape(title)}</div>
      <div style="font-size:12px;line-height:18px;color:#64748b;margin-bottom:12px;">{escape(subtitle)}</div>
      {body}
    </div>
    """


def _agent_priority(agent_name: str, recommendations: pd.DataFrame) -> str:
    if recommendations.empty or "source_agent" not in recommendations.columns or "priority" not in recommendations.columns:
        return "Info"
    match = recommendations[
        recommendations["source_agent"].fillna("").astype(str).str.casefold().eq(agent_name.casefold())
    ]
    if match.empty:
        return "Info"
    priority_order = {"high": 0, "medium": 1, "low": 2}
    ranked = match.copy()
    ranked["_rank"] = ranked["priority"].fillna("").astype(str).str.casefold().map(priority_order).fillna(3)
    return _clean_text(ranked.sort_values("_rank").iloc[0].get("priority"), "Info")


def _agent_recommendation(agent_name: str, recommendations: pd.DataFrame) -> str:
    if recommendations.empty or "source_agent" not in recommendations.columns:
        return "Review the latest inventory signals and prioritize rows with operational risk."
    match = recommendations[
        recommendations["source_agent"].fillna("").astype(str).str.casefold().eq(agent_name.casefold())
    ]
    if match.empty:
        return "Review the latest inventory signals and prioritize rows with operational risk."
    row = match.iloc[0]
    return _clean_text(row.get("action"), _clean_text(row.get("reason"), "Review the recommended action queue."))


def _agent_cards_html(report: dict[str, Any]) -> str:
    suggestions = report.get("agent_suggestions", pd.DataFrame())
    recommendations = report.get("branch_recommendations", pd.DataFrame())
    cards = []
    for agent_name, label in AGENT_LABELS.items():
        insight = ""
        if not suggestions.empty and "agent" in suggestions.columns:
            match = suggestions[suggestions["agent"].astype(str).eq(label)]
            if not match.empty:
                insight = _clean_text(match.iloc[0].get("suggestion"))
        priority = _agent_priority(agent_name, recommendations)
        priority_color, priority_bg = {
            "high": ("#991b1b", "#fee2e2"),
            "medium": ("#92400e", "#fef3c7"),
            "low": ("#1d4ed8", "#dbeafe"),
        }.get(priority.casefold(), ("#334155", "#e2e8f0"))
        cards.append(
            f"""
            <div style="border:1px solid #e2e8f0;border-radius:14px;padding:14px;margin-top:10px;background:#fbfdff;">
              <table role="presentation" width="100%" cellspacing="0" cellpadding="0">
                <tr>
                  <td style="font-size:15px;font-weight:800;color:#0f172a;">{escape(label)}</td>
                  <td style="text-align:right;">{_email_badge(priority, priority_color, priority_bg)}</td>
                </tr>
              </table>
              <div style="font-size:12px;font-weight:700;color:#64748b;text-transform:uppercase;margin-top:10px;">Latest Insight</div>
              <div style="font-size:13px;line-height:19px;color:#0f172a;margin-top:3px;">{escape(insight or 'No agent output is available; using inventory fallback logic.')}</div>
              <div style="font-size:12px;font-weight:700;color:#64748b;text-transform:uppercase;margin-top:10px;">Recommendation</div>
              <div style="font-size:13px;line-height:19px;color:#0f172a;margin-top:3px;">{escape(_agent_recommendation(agent_name, recommendations))}</div>
            </div>
            """
        )
    return "".join(cards)


def build_inventory_report_email_html(report: dict[str, Any]) -> str:
    generated_at = _clean_text(
        report.get("generated_at"),
        datetime.now().strftime("%d %b %Y, %I:%M %p"),
    )
    low_count = int(report.get("low_stock_count", 0) or 0)
    over_count = int(report.get("overstock_count", 0) or 0)
    transfer_count = int(report.get("transfer_opportunity_count", 0) or 0)
    high_count = int(report.get("high_priority_recommendations_count", 0) or 0)
    low_insight = "critical products below threshold" if low_count else "no products below reorder threshold"
    over_insight = "products need optimization" if over_count else "no material overstock detected"
    transfer_insight = "branch transfer or alternative options" if transfer_count else "no transfer opportunities found"
    high_insight = "recommendations need priority attention" if high_count else "no high-priority recommendations"

    kpi_cards = f"""
    <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="border-collapse:collapse;">
      <tr>
        {_email_dashboard_card("Low Stock Items", low_count, low_insight, "#dc2626", "#fff7f7")}
        {_email_dashboard_card("Overstock Items", over_count, over_insight, "#d97706", "#fffbeb")}
        {_email_dashboard_card("Transfer Opportunities", transfer_count, transfer_insight, "#0f766e", "#ecfdf5")}
        {_email_dashboard_card("High Priority", high_count, high_insight, "#2563eb", "#eff6ff")}
      </tr>
    </table>
    """

    low_stock_table = _email_table(
        report.get("low_stock_table", pd.DataFrame()),
        ["product_name", "store_name", "stock_level", "reorder_threshold", "suggested_reorder_quantity", "priority"],
        {
            "product_name": "Product",
            "store_name": "Branch",
            "stock_level": "Qty",
            "reorder_threshold": "Threshold",
            "suggested_reorder_quantity": "Suggested Reorder",
        },
        limit=7,
    )
    overstock_table = _email_table(
        report.get("overstock_table", pd.DataFrame()),
        ["product_name", "store_name", "surplus_quantity", "suggested_action"],
        {
            "product_name": "Product",
            "store_name": "Branch",
            "surplus_quantity": "Surplus Qty",
            "suggested_action": "Suggested Action",
        },
        limit=7,
    )
    transfer_table = _email_table(
        report.get("transfer_opportunities", pd.DataFrame()),
        ["source_branch", "target_branch", "product", "suggested_transfer_quantity", "reason"],
        {
            "source_branch": "Source Branch",
            "target_branch": "Target Branch",
            "product": "Product",
            "suggested_transfer_quantity": "Qty",
            "reason": "Transfer Suggestion",
        },
        limit=6,
    )
    branch_recs = _email_table(
        report.get("branch_recommendations", pd.DataFrame()),
        ["store_name", "source_agent", "priority", "product_name", "action"],
        {
            "store_name": "Branch",
            "source_agent": "Agent",
            "product_name": "Product",
        },
        limit=8,
    )

    return f"""
    <html>
      <body style="margin:0;padding:0;background:#eef3f8;font-family:Arial,Helvetica,sans-serif;color:#0f172a;">
        <div style="display:none;max-height:0;overflow:hidden;">Inventory dashboard report with PDF and CSV attachments.</div>
        <div style="max-width:920px;margin:0 auto;padding:28px 18px;">
          <div style="background:#0f172a;border-radius:22px 22px 10px 10px;padding:28px;color:#ffffff;">
            <div style="font-size:12px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:#93c5fd;">AI Retail Inventory Optimizer</div>
            <div style="font-size:28px;line-height:35px;font-weight:800;margin-top:8px;">AI Retail Inventory Intelligence Report</div>
            <div style="font-size:14px;line-height:21px;color:#dbeafe;margin-top:10px;">A manager-ready stock view with low-stock risk, overstock optimization, transfer opportunities, and AI agent recommendations.</div>
            <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="margin-top:20px;border-collapse:collapse;">
              <tr>
                <td style="padding:9px 12px;background:rgba(255,255,255,0.08);border-radius:12px;font-size:12px;color:#e2e8f0;">
                  <strong style="color:#ffffff;">Branch:</strong> {escape(str(report.get('branch_label', 'All Branches')))}
                </td>
                <td style="padding:9px 12px;background:rgba(255,255,255,0.08);border-radius:12px;font-size:12px;color:#e2e8f0;">
                  <strong style="color:#ffffff;">Generated:</strong> {escape(generated_at)}
                </td>
                <td style="padding:9px 12px;background:rgba(255,255,255,0.08);border-radius:12px;font-size:12px;color:#e2e8f0;">
                  <strong style="color:#ffffff;">Range:</strong> {escape(str(report.get('date_range_label', 'Latest snapshot')))}
                </td>
              </tr>
            </table>
          </div>

          <div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:0 0 18px 18px;padding:20px;box-shadow:0 14px 36px rgba(15,23,42,0.10);">
            <div style="font-size:14px;line-height:21px;color:#334155;margin-bottom:16px;">
              Hello Inventory Manager, this visual report summarizes the latest inventory intelligence. The full visual PDF and filtered inventory CSV are attached for offline review.
            </div>
            {kpi_cards}
            <div style="font-size:12px;line-height:18px;color:#64748b;background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;padding:12px;margin-top:14px;">
              {escape(str(report.get('snapshot_note', 'Inventory report is based on the latest stock snapshot.')))} Charts are included in the attached PDF.
            </div>

            {_email_section("Low Stock Items", "Products below threshold with suggested reorder quantities.", low_stock_table)}
            {_email_section("Overstock Items", "Surplus inventory candidates for transfer, discount, or clearance.", overstock_table)}
            {_email_section("Transfer Opportunities", "Branch movement and alternative availability opportunities.", transfer_table)}
            {_email_section("Branch-wise Recommendations", "Latest branch-filtered AI recommendation queue.", branch_recs)}
            {_email_section("AI Agent Recommendations", "Five specialist agent perspectives for the selected inventory scope.", _agent_cards_html(report))}

            <div style="margin-top:18px;background:#f8fafc;border:1px solid #e2e8f0;border-radius:14px;padding:15px;">
              <div style="font-size:13px;font-weight:800;color:#0f172a;margin-bottom:6px;">Attachments Included</div>
              <div style="font-size:13px;line-height:20px;color:#334155;">1. Visual PDF executive report<br>2. Filtered inventory CSV data</div>
            </div>

            <div style="font-size:12px;line-height:18px;color:#64748b;text-align:center;margin-top:22px;">
              This report was generated automatically by AI Retail Inventory Optimizer.
            </div>
          </div>
        </div>
      </body>
    </html>
    """


def _inventory_email_body(report: dict[str, Any]) -> str:
    return build_inventory_report_email_html(report)


def _prepare_transfer_opportunities(
    enriched_inventory: pd.DataFrame,
    products: pd.DataFrame,
    stores: pd.DataFrame,
    low_stock_table: pd.DataFrame,
    branch_filter: str,
) -> pd.DataFrame:
    try:
        from backend.services.stock_alternative_service import get_alternative_availability_for_low_stock
    except Exception:
        return pd.DataFrame()

    try:
        alternatives = get_alternative_availability_for_low_stock(
            inventory=enriched_inventory[
                [column for column in ["product_id", "store_id", "stock_level", "reorder_threshold", "last_updated"] if column in enriched_inventory.columns]
            ].copy(),
            products=products,
            stores=stores,
            low_stock_items=low_stock_table.rename(
                columns={"stock_level": "current_quantity"}
            ),
        )
    except Exception:
        return pd.DataFrame()

    if alternatives.empty:
        return pd.DataFrame()

    rows = []
    for _, row in alternatives.iterrows():
        product = _clean_text(row.get("low_stock_product"))
        source_branch = _clean_text(row.get("alternative_store"))
        target_branch = _clean_text(row.get("low_stock_store"))
        if _norm_text(branch_filter) != "all branches":
            selected_label = _store_label(branch_filter, stores)
            selected_store_name = selected_label.split(" (")[0]
            if _norm_text(selected_store_name) not in {_norm_text(source_branch), _norm_text(target_branch)}:
                continue
        rows.append(
            {
                "product": product,
                "source_branch": source_branch,
                "target_branch": target_branch,
                "suggested_transfer_quantity": int(pd.to_numeric(row.get("available_quantity", 0), errors="coerce") or 0),
                "alternative_product": _clean_text(row.get("alternative_product"), product),
                "reason": _clean_text(row.get("reason"), _clean_text(row.get("suggested_action"))),
            }
        )
    return pd.DataFrame(rows)


def _agent_suggestions_for_branches(
    branches: pd.DataFrame,
    agent_outputs: pd.DataFrame,
    branch_inventory: pd.DataFrame,
    branch_low_stock: pd.DataFrame,
    branch_overstock: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    latest_agent = pd.DataFrame()
    if not agent_outputs.empty and {"agent_name", "latest_insight"}.issubset(agent_outputs.columns):
        latest_agent = agent_outputs.copy()
        if "run_time" in latest_agent.columns:
            latest_agent = latest_agent.sort_values("run_time").drop_duplicates("agent_name", keep="last")

    for _, branch in branches.iterrows():
        store_name = _clean_text(branch.get("store_name"), _clean_text(branch.get("store_id"), "All Branches"))
        store_id = _clean_text(branch.get("store_id"))
        stock_qty = int(_numeric(branch_inventory.get("stock_level")).sum()) if not branch_inventory.empty else 0
        low_count = len(branch_low_stock)
        over_count = len(branch_overstock)
        fallback = {
            "inventory_agent": f"{store_name} has {stock_qty:,} units on hand across the current stock snapshot.",
            "pricing_agent": f"{over_count:,} overstock rows can be reviewed for discount, bundle, or clearance action.",
            "transfer_agent": f"{low_count:,} shortage rows and {over_count:,} surplus rows should be checked for transfer fit.",
            "risk_agent": f"{low_count:,} low-stock rows need priority review to reduce stockout risk.",
            "procurement_agent": f"Reorder planning should start with the {low_count:,} rows below threshold.",
        }
        for agent_name, label in AGENT_LABELS.items():
            suggestion = ""
            source = "fallback"
            if not latest_agent.empty:
                match = latest_agent[latest_agent["agent_name"].astype(str).eq(agent_name)]
                if not match.empty:
                    suggestion = _clean_text(match.iloc[0].get("latest_insight"))
                    source = "agent_outputs.csv"
            rows.append(
                {
                    "store_id": store_id,
                    "store_name": store_name,
                    "agent": label,
                    "suggestion": suggestion or fallback[agent_name],
                    "source": source if suggestion else "inventory fallback",
                }
            )
    return pd.DataFrame(rows)


def _email_shell(title: str, subtitle: str, cards: str, sections: str) -> str:
    return f"""
    <html>
      <head>
        <style>
          body {{ margin:0; padding:0; background:#f4f7fb; color:#0f172a; font-family:Arial, Helvetica, sans-serif; }}
          .wrap {{ max-width:980px; margin:0 auto; padding:26px; }}
          .hero {{ background:linear-gradient(135deg,#0f766e,#2563eb); border-radius:18px 18px 8px 8px; padding:28px; color:#ffffff; }}
          .hero h1 {{ margin:0 0 8px; font-size:26px; }}
          .hero p {{ margin:0; opacity:0.92; font-size:14px; }}
          .panel {{ background:#ffffff; border:1px solid #e2e8f0; border-radius:8px; padding:20px; margin-top:14px; box-shadow:0 10px 28px rgba(15,23,42,0.08); }}
          .cards {{ display:grid; grid-template-columns:repeat(3,1fr); gap:10px; margin-top:14px; }}
          .metric {{ background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:14px; }}
          .metric-label {{ color:#64748b; font-size:12px; text-transform:uppercase; font-weight:700; }}
          .metric-value {{ margin-top:5px; font-size:20px; font-weight:800; }}
          h2 {{ margin:0 0 12px; font-size:18px; }}
          .section {{ margin-top:20px; }}
          .report-table {{ width:100%; border-collapse:collapse; font-size:13px; }}
          .report-table th {{ background:#eff6ff; color:#1e3a8a; text-align:left; padding:10px; border:1px solid #dbeafe; }}
          .report-table td {{ padding:10px; border:1px solid #e5e7eb; vertical-align:top; }}
          .badge {{ display:inline-block; padding:4px 9px; border-radius:999px; color:#ffffff; font-weight:700; font-size:12px; }}
          .badge-red {{ background:#dc2626; }}
          .badge-amber {{ background:#d97706; }}
          .badge-blue {{ background:#2563eb; }}
          .footer {{ text-align:center; color:#64748b; font-size:12px; padding:20px 0 4px; }}
          @media (max-width:720px) {{ .cards {{ display:block; }} .metric {{ margin-bottom:10px; }} }}
        </style>
      </head>
      <body>
        <div class="wrap">
          <div class="hero">
            <h1>{escape(title)}</h1>
            <p>{escape(subtitle)}</p>
          </div>
          <div class="panel">
            <div class="cards">{cards}</div>
            {sections}
          </div>
          <div class="footer">AI Retail Inventory Optimizer</div>
        </div>
      </body>
    </html>
    """


def _metric_card(label: str, value: Any) -> str:
    return f"""
    <div class="metric">
      <div class="metric-label">{escape(str(label))}</div>
      <div class="metric-value">{escape(str(value))}</div>
    </div>
    """


def _section(title: str, body: str) -> str:
    return f"<div class='section'><h2>{escape(title)}</h2>{body}</div>"


def build_inventory_report_html(report: dict[str, Any]) -> str:
    cards = "".join(
        [
            _metric_card("Total Products", f"{report['total_products']:,}"),
            _metric_card("Inventory Qty", f"{report['total_inventory_quantity']:,}"),
            _metric_card("Inventory Value", report["estimated_inventory_value"]),
            _metric_card("Low Stock Items", f"{report['low_stock_count']:,}"),
            _metric_card("Overstock Items", f"{report['overstock_count']:,}"),
            _metric_card("Branch", report["branch_label"]),
        ]
    )
    sections = "".join(
        [
            _section("Snapshot Note", f"<p style='margin:0;line-height:1.6;color:#475569;'>{escape(report['snapshot_note'])}</p>"),
            _section("Category-wise Inventory Summary", _table_html(report["category_summary"], ["category", "stock_level"])),
            _section("Store-wise Inventory Summary", _table_html(report["store_summary"], ["store_name", "city", "stock_level"]) if report["include_store_summary"] else "<p style='margin:0;color:#64748b;'>Single branch selected.</p>"),
            _section("Low Stock Items", _table_html(report["low_stock_table"], ["product_name", "store_name", "stock_level", "reorder_threshold", "shortage_quantity", "suggested_reorder_quantity", "priority", "ai_recommendation"])),
            _section("Overstock Items", _table_html(report["overstock_table"], ["product_name", "store_name", "stock_level", "reorder_threshold", "surplus_quantity", "suggested_action"])),
            _section("Transfer / Alternative Availability", _table_html(report["transfer_opportunities"], ["product", "source_branch", "target_branch", "suggested_transfer_quantity", "alternative_product", "reason"])),
            _section("Branch-wise Recommendations", _table_html(report["branch_recommendations"], ["store_name", "source_agent", "priority", "recommendation_type", "product_name", "action"], limit=12)),
            _section("5 Agent Suggestions per Branch", _table_html(report["agent_suggestions"], ["store_name", "agent", "suggestion", "source"], limit=20)),
        ]
    )
    subtitle = f"{report['branch_label']} | {report['date_range_label']}"
    return _email_shell("Inventory Report", subtitle, cards, sections)


def build_sales_report_html(report: dict[str, Any]) -> str:
    cards = "".join(
        [
            _metric_card("Sales Quantity", f"{report['total_sales_quantity']:,}"),
            _metric_card("Total Revenue", report["total_revenue"]),
            _metric_card("Branch", report["branch_label"]),
            _metric_card("Top Product", report["top_product"]),
            _metric_card("Sales Days", f"{report['sales_days']:,}"),
            _metric_card("Date Range", report["date_range_label"]),
        ]
    )
    sections = "".join(
        [
            _section("Top Selling Products", _table_html(report["top_selling"], ["product_name", "category", "quantity_sold", "revenue"])),
            _section("Least Selling Products", _table_html(report["least_selling"], ["product_name", "category", "quantity_sold", "revenue"])),
            _section("Category-wise Sales Summary", _table_html(report["category_summary"], ["category", "quantity_sold", "revenue"])),
            _section("Branch-wise Sales Summary", _table_html(report["branch_summary"], ["store_name", "city", "quantity_sold", "revenue"]) if report["include_branch_summary"] else "<p style='margin:0;color:#64748b;'>Single branch selected.</p>"),
            _section("Sales Trend Summary", _table_html(report["trend_summary"], ["date", "quantity_sold", "revenue"], limit=12)),
            _section("AI Insight", f"<p style='margin:0;line-height:1.6;'>{escape(report['ai_insight'])}</p>"),
        ]
    )
    subtitle = f"{report['branch_label']} | {report['date_range_label']}"
    return _email_shell("Sales Report", subtitle, cards, sections)


def generate_inventory_report(branch_filter, start_date, end_date) -> dict[str, Any]:
    data = _load_report_data()
    resolved_branch_filter = _resolve_store_id(str(branch_filter), data["stores"])
    inventory = data["inventory"].copy()
    report_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    if inventory.empty:
        return {"success": False, "message": "No inventory records found."}

    inventory = _filter_branch(inventory, resolved_branch_filter, data["stores"])
    if inventory.empty:
        return {"success": False, "message": "No inventory records found for selected branch."}

    snapshot_note = "Inventory report is based on the latest stock snapshot."
    date_column = _inventory_date_column(inventory)
    if date_column:
        date_filtered_inventory = _filter_date_range(inventory, date_column, start_date, end_date)
        if not date_filtered_inventory.empty:
            inventory = date_filtered_inventory
            snapshot_note = (
                f"Inventory report is based on the stock snapshot rows in "
                f"{date_column} between {_format_date(start_date)} and {_format_date(end_date)}."
            )

    enriched = _enrich(inventory, data["products"], data["stores"])

    if enriched.empty:
        return {"success": False, "message": "No inventory records found for selected branch."}

    stock = _numeric(enriched.get("stock_level"))
    thresholds = _numeric(enriched.get("reorder_threshold"))
    enriched["stock_level"] = stock
    enriched["reorder_threshold"] = thresholds
    if "selling_price" in enriched.columns:
        enriched["inventory_value"] = stock * _numeric(enriched.get("selling_price"))
    else:
        enriched["inventory_value"] = 0

    low_stock = enriched[(thresholds > 0) & (stock <= thresholds)].copy()
    low_stock["shortage_quantity"] = (low_stock["reorder_threshold"] - low_stock["stock_level"]).clip(lower=0).round().astype(int)
    low_stock["suggested_reorder_quantity"] = (low_stock["shortage_quantity"] + low_stock["reorder_threshold"]).round().astype(int)
    low_stock["priority"] = low_stock["shortage_quantity"].map(lambda value: "High" if value >= 10 else "Medium")
    low_stock["ai_recommendation"] = low_stock.apply(
        lambda row: (
            f"Reorder {int(row.get('suggested_reorder_quantity', 0))} units for "
            f"{row.get('product_name', row.get('product_id', 'this product'))} at "
            f"{row.get('store_name', row.get('store_id', 'this branch'))}."
        ),
        axis=1,
    )

    overstock = enriched[(thresholds > 0) & (stock >= (thresholds * 2))].copy()
    overstock["surplus_quantity"] = stock - thresholds
    overstock["suggested_action"] = overstock["surplus_quantity"].map(
        lambda value: "transfer / discount" if value < 50 else "transfer / discount / clearance"
    )

    category_summary = _summary_table(enriched, ["category"], "stock_level")
    store_summary = _summary_table(enriched, ["store_name", "city"], "stock_level")
    recommendations = _recommendations_for_branch(data["recommendations"], str(resolved_branch_filter))
    branch_recommendations = recommendations.copy()
    if not branch_recommendations.empty and "store_name" not in branch_recommendations.columns:
        branch_recommendations = _enrich(branch_recommendations, data["products"], data["stores"])
    transfer_opportunities = _prepare_transfer_opportunities(
        enriched,
        data["products"],
        data["stores"],
        low_stock,
        str(resolved_branch_filter),
    )

    if _norm_text(resolved_branch_filter) == "all branches":
        branch_rows = data["stores"].copy()
    else:
        branch_rows = data["stores"][
            data["stores"]["store_id"].fillna("").astype(str).str.strip().eq(str(resolved_branch_filter).strip())
        ].copy()
    if branch_rows.empty:
        branch_rows = pd.DataFrame(
            [{"store_id": resolved_branch_filter, "store_name": branch_label if "branch_label" in locals() else resolved_branch_filter}]
        )

    branch_label = _store_label(str(resolved_branch_filter), data["stores"])
    date_range_label = f"{_format_date(start_date)} to {_format_date(end_date)}"
    agent_suggestions = []
    for _, branch_row in branch_rows.iterrows():
        branch_store_id = _clean_text(branch_row.get("store_id"))
        branch_inventory = enriched[enriched["store_id"].fillna("").astype(str).str.strip().eq(branch_store_id)] if "store_id" in enriched.columns else enriched
        branch_low_stock = low_stock[low_stock["store_id"].fillna("").astype(str).str.strip().eq(branch_store_id)] if "store_id" in low_stock.columns else low_stock
        branch_overstock = overstock[overstock["store_id"].fillna("").astype(str).str.strip().eq(branch_store_id)] if "store_id" in overstock.columns else overstock
        agent_suggestions.append(
            _agent_suggestions_for_branches(
                pd.DataFrame([branch_row]),
                data["agent_outputs"],
                branch_inventory,
                branch_low_stock,
                branch_overstock,
            )
        )
    agent_suggestions_df = pd.concat(agent_suggestions, ignore_index=True) if agent_suggestions else pd.DataFrame()

    low_stock_by_branch = _low_stock_by_branch(low_stock)
    overstock_by_branch = _overstock_by_branch(overstock)
    healthy_stock_count = max(len(enriched) - len(low_stock) - len(overstock), 0)
    csv_path = REPORTS_DIR / f"inventory_report_{report_timestamp}.csv"
    pdf_path = REPORTS_DIR / f"inventory_report_{report_timestamp}.pdf"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(csv_path, index=False)

    report = {
        "success": True,
        "message": "Inventory report generated.",
        "report_type": "inventory",
        "branch_label": branch_label,
        "date_range_label": date_range_label,
        "generated_at": datetime.now().strftime("%d %b %Y, %I:%M %p"),
        "snapshot_note": snapshot_note,
        "total_products": enriched["product_id"].nunique() if "product_id" in enriched.columns else len(enriched),
        "total_inventory_quantity": int(stock.sum()),
        "low_stock_count": int(len(low_stock)),
        "overstock_count": int(len(overstock)),
        "transfer_opportunity_count": int(len(transfer_opportunities)),
        "high_priority_recommendations_count": _priority_count(recommendations, "high"),
        "healthy_stock_count": int(healthy_stock_count),
        "estimated_inventory_value": _money(float(enriched["inventory_value"].sum())),
        "category_summary": category_summary,
        "store_summary": store_summary,
        "low_stock_by_branch": low_stock_by_branch,
        "overstock_by_branch": overstock_by_branch,
        "include_store_summary": _norm_text(resolved_branch_filter) == "all branches",
        "low_stock_table": low_stock.sort_values("stock_level").head(25),
        "overstock_table": overstock.sort_values("surplus_quantity", ascending=False).head(25),
        "transfer_opportunities": transfer_opportunities.head(25),
        "branch_recommendations": branch_recommendations.head(50),
        "agent_suggestions": agent_suggestions_df,
        "top_low_stock": low_stock.sort_values("stock_level").head(10),
        "top_overstock": overstock.sort_values("surplus_quantity", ascending=False).head(10),
        "recommendations": recommendations.head(8),
        "filtered_data": enriched,
        "html": "",
        "email_html": "",
        "attachment_path": csv_path,
        "attachment_paths": [csv_path],
        "csv_path": csv_path,
        "pdf_path": None,
        "pdf_warning": "",
    }
    report["html"] = build_inventory_report_html(report)
    report["email_html"] = _inventory_email_body(report)

    try:
        from backend.services.pdf_report_service import generate_inventory_pdf_report

        generated_pdf = generate_inventory_pdf_report(report, pdf_path)
        report["pdf_path"] = generated_pdf
        report["attachment_paths"].append(generated_pdf)
    except Exception as error:
        report["pdf_warning"] = f"PDF creation failed, so only the CSV report will be attached: {error}"

    _write_report_log(
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "report_type": "inventory",
            "branch_filter": branch_label,
            "date_range": date_range_label,
            "csv_path": str(csv_path),
            "pdf_path": str(report.get("pdf_path") or ""),
            "pdf_status": "created" if report.get("pdf_path") else "failed",
            "message": report.get("pdf_warning", ""),
        }
    )
    return report


def generate_sales_report(branch_filter, start_date, end_date) -> dict[str, Any]:
    data = _load_report_data()
    sales = _filter_branch(data["sales"], str(branch_filter))
    sales = _filter_date_range(sales, "date", start_date, end_date)
    enriched = _enrich(sales, data["products"], data["stores"])

    if enriched.empty:
        return {"success": False, "message": "No sales data exists for the selected filters."}

    enriched["quantity_sold"] = _numeric(enriched.get("quantity_sold"))
    if "selling_price" in enriched.columns:
        enriched["revenue"] = enriched["quantity_sold"] * _numeric(enriched.get("selling_price"))
    else:
        enriched["revenue"] = 0

    product_summary = (
        enriched.groupby(["product_id", "product_name", "category"], dropna=False, as_index=False)[["quantity_sold", "revenue"]]
        .sum()
        .sort_values("quantity_sold", ascending=False)
    )
    category_summary = (
        enriched.groupby(["category"], dropna=False, as_index=False)[["quantity_sold", "revenue"]]
        .sum()
        .sort_values("quantity_sold", ascending=False)
    )
    branch_summary = (
        enriched.groupby(["store_name", "city"], dropna=False, as_index=False)[["quantity_sold", "revenue"]]
        .sum()
        .sort_values("quantity_sold", ascending=False)
    )
    trend_summary = (
        enriched.groupby(["date"], dropna=False, as_index=False)[["quantity_sold", "revenue"]]
        .sum()
        .sort_values("date")
    )
    recommendations = _recommendations_for_branch(data["recommendations"], str(branch_filter))
    insight = "Review top movers for replenishment and watch least-selling products for discount or transfer decisions."
    if not data["orchestrator_summary"].empty:
        row = data["orchestrator_summary"].iloc[0]
        insight = str(row.get("executive_summary", row.get("summary", insight)) or insight)
    elif not recommendations.empty:
        first = recommendations.iloc[0]
        insight = str(first.get("action", insight) or insight)

    branch_label = _store_label(str(branch_filter), data["stores"])
    date_range_label = f"{_format_date(start_date)} to {_format_date(end_date)}"
    attachment_path = _save_attachment(enriched, "sales")
    top_product = "No product"
    if not product_summary.empty:
        top_product = str(product_summary.iloc[0].get("product_name", "No product"))

    report = {
        "success": True,
        "message": "Sales report generated.",
        "report_type": "sales",
        "branch_label": branch_label,
        "date_range_label": date_range_label,
        "total_sales_quantity": int(enriched["quantity_sold"].sum()),
        "total_revenue": _money(float(enriched["revenue"].sum())),
        "top_product": top_product,
        "sales_days": enriched["date"].nunique() if "date" in enriched.columns else 0,
        "top_selling": product_summary.head(10),
        "least_selling": product_summary.sort_values("quantity_sold", ascending=True).head(10),
        "category_summary": category_summary,
        "branch_summary": branch_summary,
        "include_branch_summary": str(branch_filter) == "All Branches",
        "trend_summary": trend_summary,
        "ai_insight": insight,
        "filtered_data": enriched,
        "html": "",
        "attachment_path": attachment_path,
    }
    report["html"] = build_sales_report_html(report)
    return report
