"""Premium executive-level inventory report formatter with HTML email generation."""

from datetime import datetime
from pathlib import Path
from typing import Optional
import pandas as pd
from html import escape
from backend.services.report_service import _email_shell, _metric_card, _section, _table_html

# Theme colors matching the Streamlit application
THEME = {
    "primary_navy": "#183F5F",
    "deep_navy": "#0A1F33",
    "fresh_green": "#6CB33F",
    "soft_green": "#A6D96A",
    "light_bg": "#F5F8FB",
    "white": "#FFFFFF",
    "soft_border": "#D8E2EC",
    "muted_text": "#476C8B",
    "warning": "#C76A12",
    "danger": "#B42318",
    "soft_blue": "#EAF1F7",
    "soft_amber": "#FFF7E8",
    "soft_red": "#FFF1F2",
    "navy_hover": "#102F49",
}


def _get_risk_color(risk_level: str) -> tuple[str, str]:
    """Get background and text color for risk level."""
    risk_level = str(risk_level or "").lower().strip()
    if risk_level == "critical":
        return THEME["danger"], "#FFFFFF"
    if risk_level == "high":
        return THEME["warning"], "#FFFFFF"
    if risk_level == "medium":
        return THEME["soft_amber"], THEME["deep_navy"]
    return THEME["soft_green"], THEME["deep_navy"]


def _get_urgency_badge_html(days_remaining: float, urgency: str) -> str:
    """Generate HTML badge for urgency level."""
    bg_color, text_color = _get_risk_color(urgency)
    return f"""
    <span style="
        display: inline-block;
        padding: 6px 12px;
        border-radius: 6px;
        background-color: {bg_color};
        color: {text_color};
        font-weight: 600;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    ">{escape(str(urgency))}</span>
    """


def _generate_header_html(
    report_date: Optional[str] = None,
    branch_scope: str = "All Branches",
) -> str:
    """Generate professional HTML header with theme branding."""
    if report_date is None:
        report_date = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    
    return f"""
    <!-- Executive Header -->
    <div style="
        background: linear-gradient(135deg, {THEME['primary_navy']}, {THEME['deep_navy']});
        color: white;
        padding: 40px 30px;
        text-align: center;
        border-radius: 12px 12px 0 0;
        margin: 0;
    ">
        <h1 style="
            margin: 0;
            font-size: 32px;
            font-weight: 800;
            letter-spacing: -0.5px;
            margin-bottom: 8px;
        ">Inventory Intelligence</h1>
        <p style="
            margin: 0 0 12px 0;
            font-size: 20px;
            font-weight: 600;
            opacity: 0.95;
        ">Low Stock Alert Report</p>
        
        <div style="
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-top: 20px;
            font-size: 13px;
            opacity: 0.9;
        ">
            <div>
                <div style="opacity: 0.8; margin-bottom: 4px;">Report Generated</div>
                <div style="font-weight: 600;">{escape(str(report_date))}</div>
            </div>
            <div>
                <div style="opacity: 0.8; margin-bottom: 4px;">Scope</div>
                <div style="font-weight: 600;">{escape(str(branch_scope))}</div>
            </div>
        </div>
    </div>
    """


def _generate_kpi_cards_html(
    low_stock_count: int,
    affected_branches: int,
    highest_risk_product: str,
    reorder_recommendations: int,
) -> str:
    """Generate KPI cards section."""
    cards = [
        ("Low Stock Items", low_stock_count, THEME["danger"]),
        ("Affected Branches", affected_branches, THEME["warning"]),
        ("Critical Products", reorder_recommendations, THEME["fresh_green"]),
    ]
    
    cards_html = ""
    for title, value, color in cards:
        cards_html += f"""
        <div style="
            background: white;
            border: 2px solid {THEME['soft_border']};
            border-left: 4px solid {color};
            border-radius: 8px;
            padding: 20px;
            flex: 1;
            min-width: 150px;
            box-shadow: 0 4px 12px rgba(10, 31, 51, 0.08);
        ">
            <div style="
                font-size: 13px;
                color: {THEME['muted_text']};
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                margin-bottom: 8px;
            ">{escape(str(title))}</div>
            <div style="
                font-size: 36px;
                font-weight: 800;
                color: {THEME['primary_navy']};
            ">{value}</div>
        </div>
        """
    
    # Add highest risk product card
    cards_html += f"""
    <div style="
        background: {THEME['soft_red']};
        border: 2px solid {THEME['danger']};
        border-left: 4px solid {THEME['danger']};
        border-radius: 8px;
        padding: 20px;
        flex: 1;
        min-width: 150px;
        box-shadow: 0 4px 12px rgba(10, 31, 51, 0.08);
    ">
        <div style="
            font-size: 13px;
            color: {THEME['danger']};
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
        ">Highest Risk Product</div>
        <div style="
            font-size: 16px;
            font-weight: 800;
            color: {THEME['deep_navy']};
            line-height: 1.4;
            word-wrap: break-word;
        ">{escape(str(highest_risk_product))}</div>
    </div>
    """
    
    return f"""
    <!-- KPI Cards Section -->
    <div style="
        padding: 30px 30px;
        background: {THEME['light_bg']};
    ">
        <h2 style="
            color: {THEME['deep_navy']};
            font-size: 18px;
            font-weight: 700;
            margin: 0 0 20px 0;
            padding-bottom: 12px;
            border-bottom: 2px solid {THEME['primary_navy']};
        ">Executive Summary</h2>
        <div style="
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 16px;
            margin-bottom: 0;
        " class="kpi-grid">
            {cards_html}
        </div>
    </div>
    """


def _generate_inventory_table_html(low_stock_df: pd.DataFrame) -> str:
    """Generate professionally styled inventory risk table."""
    if low_stock_df.empty:
        return f"""
        <div style="
            padding: 30px 30px;
            text-align: center;
            color: {THEME['muted_text']};
            font-size: 14px;
        ">
            <p>No low stock items at this time.</p>
        </div>
        """
    
    # Prepare table rows
    table_rows = ""
    for idx, row in low_stock_df.iterrows():
        bg_color = THEME["white"] if idx % 2 == 0 else THEME["light_bg"]
        risk_level = str(row.get("risk_category", "Medium")).lower()
        badge_html = _get_urgency_badge_html(
            float(row.get("predicted_days_remaining", 0)),
            row.get("risk_category", "Medium")
        )
        
        # Get values with safe defaults
        product_name = escape(str(row.get("product_name", "N/A")))
        store_name = escape(str(row.get("store_name", "N/A")))
        current_stock = f"{int(float(row.get('current_quantity', 0)))}"
        threshold = f"{int(float(row.get('reorder_threshold', 0)))}"
        daily_sales = f"{float(row.get('recent_daily_sales_velocity', 0)):.1f}"
        days_remaining = f"{float(row.get('predicted_days_remaining', 0)):.1f}"
        reorder_qty = f"{int(float(row.get('suggested_reorder_quantity', 0)))}"
        ai_reasoning = escape(str(row.get("ai_alert_message", "Inventory below optimal level"))[:100])
        
        table_rows += f"""
        <tr style="background-color: {bg_color};">
            <td style="padding: 12px 15px; border-bottom: 1px solid {THEME['soft_border']}; font-weight: 500; color: {THEME['deep_navy']};">{product_name}</td>
            <td style="padding: 12px 15px; border-bottom: 1px solid {THEME['soft_border']}; color: {THEME['muted_text']};">{store_name}</td>
            <td style="padding: 12px 15px; border-bottom: 1px solid {THEME['soft_border']}; text-align: center; font-weight: 600; color: {THEME['primary_navy']};">{current_stock}</td>
            <td style="padding: 12px 15px; border-bottom: 1px solid {THEME['soft_border']}; text-align: center; color: {THEME['muted_text']};">{threshold}</td>
            <td style="padding: 12px 15px; border-bottom: 1px solid {THEME['soft_border']}; text-align: center; color: {THEME['muted_text']};">{daily_sales}</td>
            <td style="padding: 12px 15px; border-bottom: 1px solid {THEME['soft_border']}; text-align: center;">{badge_html}</td>
            <td style="padding: 12px 15px; border-bottom: 1px solid {THEME['soft_border']}; text-align: center; font-weight: 600; color: {THEME['danger']};">{reorder_qty}</td>
            <td style="padding: 12px 15px; border-bottom: 1px solid {THEME['soft_border']}; font-size: 12px; color: {THEME['muted_text']};">{ai_reasoning}</td>
        </tr>
        """
    
    return f"""
    <!-- Inventory Risk Table -->
    <div style="padding: 30px 30px;">
        <h2 style="
            color: {THEME['deep_navy']};
            font-size: 18px;
            font-weight: 700;
            margin: 0 0 20px 0;
            padding-bottom: 12px;
            border-bottom: 2px solid {THEME['primary_navy']};
        ">Inventory Risk Overview</h2>
        
        <div style="overflow-x: auto; margin-top: 16px; border-radius: 8px; border: 1px solid {THEME['soft_border']}; box-shadow: 0 4px 12px rgba(10, 31, 51, 0.08);">
            <table style="width: 100%; border-collapse: collapse; background: white;">
                <thead>
                    <tr style="background: {THEME['primary_navy']}; color: white;">
                        <th style="padding: 14px 15px; text-align: left; font-weight: 700; font-size: 13px; text-transform: uppercase; letter-spacing: 0.5px;">Product</th>
                        <th style="padding: 14px 15px; text-align: left; font-weight: 700; font-size: 13px; text-transform: uppercase; letter-spacing: 0.5px;">Store</th>
                        <th style="padding: 14px 15px; text-align: center; font-weight: 700; font-size: 13px; text-transform: uppercase; letter-spacing: 0.5px;">Current Stock</th>
                        <th style="padding: 14px 15px; text-align: center; font-weight: 700; font-size: 13px; text-transform: uppercase; letter-spacing: 0.5px;">Threshold</th>
                        <th style="padding: 14px 15px; text-align: center; font-weight: 700; font-size: 13px; text-transform: uppercase; letter-spacing: 0.5px;">Daily Sales</th>
                        <th style="padding: 14px 15px; text-align: center; font-weight: 700; font-size: 13px; text-transform: uppercase; letter-spacing: 0.5px;">Status</th>
                        <th style="padding: 14px 15px; text-align: center; font-weight: 700; font-size: 13px; text-transform: uppercase; letter-spacing: 0.5px;">Reorder Qty</th>
                        <th style="padding: 14px 15px; text-align: left; font-weight: 700; font-size: 13px; text-transform: uppercase; letter-spacing: 0.5px;">AI Reasoning</th>
                    </tr>
                </thead>
                <tbody>
                    {table_rows}
                </tbody>
            </table>
        </div>
    </div>
    """


def _generate_ai_analysis_html(low_stock_df: pd.DataFrame) -> str:
    """Generate AI analysis section with insights for top items."""
    if low_stock_df.empty:
        return ""
    
    # Get top critical items
    critical_items = low_stock_df.head(3)
    
    analysis_items = ""
    for idx, row in critical_items.iterrows():
        product_name = escape(str(row.get("product_name", "N/A")))
        current_qty = int(float(row.get("current_quantity", 0)))
        daily_sales = float(row.get("recent_daily_sales_velocity", 0))
        days_remaining = float(row.get("predicted_days_remaining", 0))
        risk_category = str(row.get("risk_category", "Medium"))
        demand_trend = str(row.get("demand_trend", "Stable")).strip().title()
        
        bg_color, _ = _get_risk_color(risk_category)
        
        analysis_items += f"""
        <div style="
            background: white;
            border: 1px solid {THEME['soft_border']};
            border-left: 4px solid {bg_color};
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 16px;
        ">
            <div style="
                font-size: 16px;
                font-weight: 700;
                color: {THEME['deep_navy']};
                margin-bottom: 12px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            ">
                <span>{product_name}</span>
                {_get_urgency_badge_html(days_remaining, risk_category)}
            </div>
            
            <div style="
                display: grid;
                grid-template-columns: 1fr 1fr 1fr;
                gap: 16px;
                margin-bottom: 16px;
                font-size: 13px;
            ">
                <div>
                    <div style="color: {THEME['muted_text']}; font-weight: 600; margin-bottom: 4px;">Current Inventory</div>
                    <div style="color: {THEME['primary_navy']}; font-size: 18px; font-weight: 700;">{current_qty} units</div>
                </div>
                <div>
                    <div style="color: {THEME['muted_text']}; font-weight: 600; margin-bottom: 4px;">Avg Daily Sales</div>
                    <div style="color: {THEME['primary_navy']}; font-size: 18px; font-weight: 700;">{daily_sales:.1f}</div>
                </div>
                <div>
                    <div style="color: {THEME['muted_text']}; font-weight: 600; margin-bottom: 4px;">Depletion Window</div>
                    <div style="color: {THEME['danger']}; font-size: 18px; font-weight: 700;">{days_remaining:.1f} days</div>
                </div>
            </div>
            
            <div style="
                background: {THEME['light_bg']};
                border-radius: 6px;
                padding: 12px;
                margin-bottom: 12px;
                font-size: 13px;
                color: {THEME['deep_navy']};
                line-height: 1.6;
            ">
                <strong style="color: {THEME['primary_navy']};">Factors Analyzed:</strong><br>
                ✓ Current Inventory Level: {current_qty} units<br>
                ✓ Sales Velocity: {daily_sales:.1f} units/day<br>
                ✓ Demand Trend: {demand_trend}<br>
                ✓ Demand Spike Detection: Active monitoring<br>
                ✓ Reorder Threshold: Applied<br>
                ✓ Historical Consumption Pattern: {escape(str(row.get('demand_trend', 'Stable')))}<br>
            </div>
            
            <div style="
                font-size: 13px;
                color: {THEME['muted_text']};
                line-height: 1.6;
                background: {THEME['soft_blue']};
                padding: 12px;
                border-radius: 6px;
            ">
                <strong style="color: {THEME['primary_navy']};">AI Reasoning:</strong><br>
                {escape(str(row.get('ai_alert_message', 'Inventory analysis indicates stock levels below reorder threshold.')))}
            </div>
        </div>
        """
    
    return f"""
    <!-- AI Analysis Section -->
    <div style="padding: 30px 30px;">
        <h2 style="
            color: {THEME['deep_navy']};
            font-size: 18px;
            font-weight: 700;
            margin: 0 0 20px 0;
            padding-bottom: 12px;
            border-bottom: 2px solid {THEME['primary_navy']};
        ">AI Analysis - Critical Items</h2>
        <div style="margin-top: 16px;">
            {analysis_items}
        </div>
    </div>
    """


def _generate_recommended_actions_html(low_stock_df: pd.DataFrame) -> str:
    """Generate recommended actions section."""
    if low_stock_df.empty:
        return ""
    
    critical_count = len(low_stock_df[low_stock_df.get("risk_category", "").str.lower().isin(["critical", "high"])])
    medium_count = len(low_stock_df[low_stock_df.get("risk_category", "").str.lower() == "medium"])
    total_reorder = int(low_stock_df["suggested_reorder_quantity"].sum())
    
    actions_html = f"""
    <div style="
        background: white;
        border: 1px solid {THEME['danger']};
        border-left: 4px solid {THEME['danger']};
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 16px;
    ">
        <h3 style="
            margin: 0 0 12px 0;
            color: {THEME['danger']};
            font-size: 16px;
            font-weight: 700;
        ">⚠️ CRITICAL ATTENTION REQUIRED</h3>
        <p style="margin: 0; color: {THEME['deep_navy']}; font-size: 14px; line-height: 1.6;">
            <strong>{critical_count} product(s)</strong> require immediate action. Inventory depletion within 2-5 days.
            <br><strong>Recommended Action:</strong> Process reorder immediately to avoid stockouts.
        </p>
    </div>
    
    <div style="
        background: white;
        border: 1px solid {THEME['warning']};
        border-left: 4px solid {THEME['warning']};
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 16px;
    ">
        <h3 style="
            margin: 0 0 12px 0;
            color: {THEME['warning']};
            font-size: 16px;
            font-weight: 700;
        ">📋 Standard Reorder Recommendations</h3>
        <p style="margin: 0; color: {THEME['deep_navy']}; font-size: 14px; line-height: 1.6;">
            <strong>{medium_count} product(s)</strong> in medium/low risk category.
            <br><strong>Total Reorder Quantity:</strong> {total_reorder} units across {len(low_stock_df.groupby('store_id'))} locations.
        </p>
    </div>
    """
    
    return f"""
    <!-- Recommended Actions Section -->
    <div style="padding: 30px 30px;">
        <h2 style="
            color: {THEME['deep_navy']};
            font-size: 18px;
            font-weight: 700;
            margin: 0 0 20px 0;
            padding-bottom: 12px;
            border-bottom: 2px solid {THEME['primary_navy']};
        ">Recommended Actions</h2>
        <div style="margin-top: 16px;">
            {actions_html}
        </div>
    </div>
    """


def _generate_branch_summary_html(low_stock_df: pd.DataFrame) -> str:
    """Generate branch-wise summary section."""
    if low_stock_df.empty:
        return ""
    
    branch_summary = low_stock_df.groupby("store_name").agg({
        "product_name": "count",
        "risk_category": lambda x: (x.str.lower().isin(["critical", "high"])).sum(),
        "suggested_reorder_quantity": "sum",
    }).rename(columns={
        "product_name": "low_stock_count",
        "risk_category": "critical_count",
        "suggested_reorder_quantity": "total_reorder_qty"
    }).reset_index()
    
    branch_cards = ""
    for _, row in branch_summary.iterrows():
        store_name = escape(str(row["store_name"]))
        low_stock = int(row["low_stock_count"])
        critical = int(row["critical_count"])
        reorder_qty = int(row["total_reorder_qty"])
        
        branch_cards += f"""
        <div style="
            background: white;
            border: 1px solid {THEME['soft_border']};
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 12px;
        ">
            <div style="
                font-size: 14px;
                font-weight: 700;
                color: {THEME['deep_navy']};
                margin-bottom: 12px;
            ">{store_name}</div>
            <div style="
                display: grid;
                grid-template-columns: 1fr 1fr 1fr;
                gap: 12px;
                font-size: 12px;
            ">
                <div>
                    <div style="color: {THEME['muted_text']}; margin-bottom: 4px;">Low Stock Items</div>
                    <div style="color: {THEME['primary_navy']}; font-weight: 700; font-size: 16px;">{low_stock}</div>
                </div>
                <div>
                    <div style="color: {THEME['muted_text']}; margin-bottom: 4px;">Critical</div>
                    <div style="color: {THEME['danger']}; font-weight: 700; font-size: 16px;">{critical}</div>
                </div>
                <div>
                    <div style="color: {THEME['muted_text']}; margin-bottom: 4px;">Reorder Qty</div>
                    <div style="color: {THEME['fresh_green']}; font-weight: 700; font-size: 16px;">{reorder_qty}</div>
                </div>
            </div>
        </div>
        """
    
    return f"""
    <!-- Branch Summary Section -->
    <div style="padding: 30px 30px;">
        <h2 style="
            color: {THEME['deep_navy']};
            font-size: 18px;
            font-weight: 700;
            margin: 0 0 20px 0;
            padding-bottom: 12px;
            border-bottom: 2px solid {THEME['primary_navy']};
        ">Branch-Wise Summary</h2>
        <div style="margin-top: 16px;">
            {branch_cards}
        </div>
    </div>
    """


def _generate_footer_html() -> str:
    """Generate professional footer."""
    return f"""
    <!-- Footer -->
    <div style="
        background: {THEME['deep_navy']};
        color: white;
        padding: 30px 30px;
        text-align: center;
        border-radius: 0 0 12px 12px;
        font-size: 12px;
        line-height: 1.8;
    ">
        <p style="margin: 0 0 12px 0; opacity: 0.9;">
            This report was generated automatically by the <strong>AI Retail Inventory Optimization Platform</strong>.
        </p>
        <p style="margin: 0; opacity: 0.7; font-size: 11px;">
            For questions or to disable these alerts, please contact your inventory management team.
        </p>
    </div>
    """


def generate_premium_html_email(
    low_stock_df: pd.DataFrame,
    report_date: Optional[str] = None,
    branch_scope: str = "All Branches",
) -> str:
    """
    Generate a complete premium HTML email for low stock alerts.
    
    Args:
        low_stock_df: DataFrame with low stock items
        report_date: Optional custom report date (default: now)
        branch_scope: Branch scope description
        
    Returns:
        Complete HTML email content
    """
    if report_date is None:
        report_date = datetime.now().strftime("%B %d, %Y at %I:%M %p")

    def _safe_number(value, fallback: float = 0.0) -> float:
        parsed = pd.to_numeric(pd.Series([value]), errors="coerce").fillna(fallback)
        return float(parsed.iloc[0])

    report_df = low_stock_df.copy()
    detail_rows = []
    for _, row in report_df.iterrows():
        risk = str(row.get("risk_category", row.get("priority", "Medium")) or "Medium").title()
        detail_rows.append(
            {
                "product_name": row.get("product_name", row.get("product_id", "N/A")),
                "branch": row.get("store_name", row.get("store_id", "N/A")),
                "current_stock": int(_safe_number(row.get("current_quantity", 0))),
                "threshold": int(_safe_number(row.get("reorder_threshold", 0))),
                "avg_daily_sales": f"{_safe_number(row.get('recent_daily_sales_velocity', 0)):.1f}",
                "days_remaining": f"{_safe_number(row.get('predicted_days_remaining', 0)):.1f}",
                "urgency_label": risk,
                "suggested_reorder": int(_safe_number(row.get("suggested_reorder_quantity", 0))),
                "ai_reasoning": str(row.get("ai_alert_message", "Review for reorder"))[:120],
            }
        )
    detail_df = pd.DataFrame(detail_rows)

    low_stock_count = len(report_df)
    affected_branches = report_df["store_id"].nunique() if "store_id" in report_df.columns else 0
    risk_series = (
        detail_df["urgency_label"].fillna("").astype(str).str.casefold()
        if "urgency_label" in detail_df.columns
        else pd.Series(dtype=str)
    )
    reorder_recommendations = int(risk_series.isin(["critical", "high"]).sum())
    highest_risk_product = (
        str(report_df.iloc[0].get("product_name", report_df.iloc[0].get("product_id", "None")))
        if not report_df.empty
        else "None"
    )
    total_reorder = int(
        pd.to_numeric(
            report_df.get("suggested_reorder_quantity", pd.Series(dtype=float)),
            errors="coerce",
        )
        .fillna(0)
        .sum()
    )

    cards = "".join(
        [
            _metric_card("Low Stock Items", f"{low_stock_count:,}"),
            _metric_card("Affected Branches", f"{affected_branches:,}"),
            _metric_card("Requires Action", f"{reorder_recommendations:,}"),
        ]
    )
    summary = (
        "<p style='margin:0;line-height:1.6;color:#476C8B;'>"
        f"The latest inventory scan found <strong style='color:#0A1F33;'>{low_stock_count:,}</strong> low-stock item(s) "
        f"across <strong style='color:#0A1F33;'>{affected_branches:,}</strong> branch(es). "
        f"Start with <strong style='color:#0A1F33;'>{escape(highest_risk_product)}</strong> and review "
        f"<strong style='color:#0A1F33;'>{total_reorder:,}</strong> total suggested reorder units in the attached workbook."
        "</p>"
    )

    branch_summary = pd.DataFrame()
    if "store_name" in report_df.columns:
        branch_summary = (
            report_df.assign(
                suggested_reorder_quantity=pd.to_numeric(
                    report_df.get("suggested_reorder_quantity", pd.Series(0, index=report_df.index)),
                    errors="coerce",
                ).fillna(0)
            )
            .groupby("store_name", dropna=False)
            .agg(
                low_stock_items=("store_name", "size"),
                suggested_reorder_quantity=("suggested_reorder_quantity", "sum"),
            )
            .reset_index()
            .sort_values("low_stock_items", ascending=False)
        )
        branch_summary["suggested_reorder_quantity"] = branch_summary["suggested_reorder_quantity"].astype(int)

    sections = "".join(
        [
            _section("Executive Summary", summary),
            _section(
                "Branch-wise Low Stock Summary",
                _table_html(
                    branch_summary,
                    ["store_name", "low_stock_items", "suggested_reorder_quantity"],
                    limit=12,
                ),
            ),
            _section(
                "Detailed Low Stock Items",
                _table_html(
                    detail_df,
                    [
                        "product_name",
                        "branch",
                        "current_stock",
                        "threshold",
                        "avg_daily_sales",
                        "days_remaining",
                        "urgency_label",
                        "suggested_reorder",
                        "ai_reasoning",
                    ],
                    limit=20,
                ),
            ),
            _section(
                "Attachments Included",
                "<p style='margin:0;line-height:1.6;color:#476C8B;'>1. Low_Stock_Report.xlsx</p>",
            ),
        ]
    )

    return _email_shell(
        "Low Stock Alert Report",
        f"{branch_scope} | Generated {report_date}",
        cards,
        sections,
    )

    if low_stock_df.empty:
        low_stock_count = 0
        affected_branches = 0
        highest_risk_product = "None"
        reorder_recommendations = 0
    else:
        low_stock_count = len(low_stock_df)
        affected_branches = low_stock_df["store_id"].nunique()
        highest_risk_product = str(low_stock_df.iloc[0]["product_name"])
        reorder_recommendations = len(
            low_stock_df[
                low_stock_df.get("risk_category", "").str.lower().isin(["critical", "high"])
            ]
        )
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Low Stock Alert Report</title>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 0;
                background-color: {THEME['light_bg']};
                color: {THEME['deep_navy']};
            }}
            table {{
                border-collapse: collapse;
            }}
            .kpi-grid {{
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 16px;
            }}
            @media (max-width: 600px) {{
                .kpi-grid {{
                    grid-template-columns: 1fr;
                }}
            }}
        </style>
    </head>
    <body style="margin: 0; padding: 20px; background-color: {THEME['light_bg']};">
        <div style="
            max-width: 900px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 12px 32px rgba(10, 31, 51, 0.15);
        ">
            {_generate_header_html(report_date, branch_scope)}
            {_generate_kpi_cards_html(low_stock_count, affected_branches, highest_risk_product, reorder_recommendations)}
            {_generate_inventory_table_html(low_stock_df)}
            {_generate_ai_analysis_html(low_stock_df)}
            {_generate_recommended_actions_html(low_stock_df)}
            {_generate_branch_summary_html(low_stock_df)}
            {_generate_footer_html()}
        </div>
    </body>
    </html>
    """
    
    return html_content


def generate_excel_report(
    low_stock_df: pd.DataFrame,
    output_path: Path,
) -> Path:
    """
    Generate a formatted Excel report with multiple sheets.
    
    Args:
        low_stock_df: DataFrame with low stock items
        output_path: Path to save the Excel file
        
    Returns:
        Path to the generated Excel file
    """
    try:
        from openpyxl import Workbook
        from openpyxl.styles import (
            Font, PatternFill, Alignment, Border, Side, ProtectedCell
        )
        from openpyxl.utils import get_column_letter
    except ImportError:
        raise ImportError("openpyxl is required for Excel export. Install with: pip install openpyxl")
    
    wb = Workbook()
    ws = wb.active
    ws.title = "Low Stock Alerts"
    
    # Define styles
    header_fill = PatternFill(start_color="183F5F", end_color="183F5F", fill_type="solid")
    header_font = Font(bold=True, color="FFFFFF", size=11)
    header_alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    
    border = Border(
        left=Side(style="thin", color="D8E2EC"),
        right=Side(style="thin", color="D8E2EC"),
        top=Side(style="thin", color="D8E2EC"),
        bottom=Side(style="thin", color="D8E2EC"),
    )
    
    # Summary sheet
    summary_ws = wb.create_sheet("Summary", 0)
    
    # Add summary data
    summary_ws["A1"] = "Inventory Intelligence - Low Stock Alert Report"
    summary_ws["A1"].font = Font(bold=True, size=14, color="0A1F33")
    summary_ws.merge_cells("A1:D1")
    
    summary_ws["A3"] = "Report Generated"
    summary_ws["B3"] = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    
    summary_ws["A4"] = "Total Low Stock Items"
    summary_ws["B4"] = len(low_stock_df)
    summary_ws["B4"].font = Font(bold=True, color="183F5F", size=12)
    
    summary_ws["A5"] = "Affected Branches"
    summary_ws["B5"] = low_stock_df["store_id"].nunique() if not low_stock_df.empty else 0
    summary_ws["B5"].font = Font(bold=True, color="183F5F", size=12)
    
    if not low_stock_df.empty:
        critical_count = len(low_stock_df[low_stock_df.get("risk_category", "").str.lower().isin(["critical", "high"])])
        summary_ws["A6"] = "Critical Items"
        summary_ws["B6"] = critical_count
        summary_ws["B6"].font = Font(bold=True, color="B42318", size=12)
    
    summary_ws.column_dimensions["A"].width = 25
    summary_ws.column_dimensions["B"].width = 30
    
    # Main data sheet
    columns = [
        "Product Name",
        "Store Name",
        "Current Stock",
        "Reorder Threshold",
        "Avg Daily Sales",
        "Days Remaining",
        "Risk Level",
        "Suggested Reorder Qty",
        "AI Analysis",
    ]
    
    for col_num, column_title in enumerate(columns, 1):
        cell = ws.cell(row=1, column=col_num)
        cell.value = column_title
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = header_alignment
        cell.border = border
    
    # Add data rows
    for row_num, (_, row_data) in enumerate(low_stock_df.iterrows(), 2):
        ws.cell(row=row_num, column=1).value = str(row_data.get("product_name", "N/A"))
        ws.cell(row=row_num, column=2).value = str(row_data.get("store_name", "N/A"))
        ws.cell(row=row_num, column=3).value = int(float(row_data.get("current_quantity", 0)))
        ws.cell(row=row_num, column=4).value = int(float(row_data.get("reorder_threshold", 0)))
        ws.cell(row=row_num, column=5).value = float(row_data.get("recent_daily_sales_velocity", 0))
        ws.cell(row=row_num, column=6).value = float(row_data.get("predicted_days_remaining", 0))
        ws.cell(row=row_num, column=7).value = str(row_data.get("risk_category", "Medium"))
        ws.cell(row=row_num, column=8).value = int(float(row_data.get("suggested_reorder_quantity", 0)))
        ws.cell(row=row_num, column=9).value = str(row_data.get("ai_alert_message", ""))
        
        for col_num in range(1, len(columns) + 1):
            ws.cell(row=row_num, column=col_num).border = border
    
    # Adjust column widths
    widths = [25, 20, 14, 16, 16, 14, 14, 18, 40]
    for col_num, width in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(col_num)].width = width
    
    # Save file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(output_path)
    
    return output_path


def generate_pdf_report(
    low_stock_df: pd.DataFrame,
    output_path: Path,
) -> Path:
    """
    Generate a professional PDF report with branding and charts.
    
    Args:
        low_stock_df: DataFrame with low stock items
        output_path: Path to save the PDF file
        
    Returns:
        Path to the generated PDF file
    """
    try:
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    except ImportError:
        raise ImportError("reportlab is required for PDF export. Install with: pip install reportlab")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
    )
    
    story = []
    styles = getSampleStyleSheet()
    
    # Title
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Heading1"],
        fontSize=24,
        textColor=colors.HexColor(THEME["deep_navy"]),
        spaceAfter=6,
        alignment=TA_CENTER,
        fontName="Helvetica-Bold",
    )
    story.append(Paragraph("Inventory Intelligence", title_style))
    story.append(Paragraph("Low Stock Alert Report", title_style))
    story.append(Spacer(1, 0.2 * inch))
    
    # Report info
    info_style = ParagraphStyle(
        "Info",
        parent=styles["Normal"],
        fontSize=10,
        textColor=colors.HexColor(THEME["muted_text"]),
        alignment=TA_CENTER,
    )
    report_date = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    story.append(Paragraph(f"Report Generated: {report_date}", info_style))
    story.append(Spacer(1, 0.3 * inch))
    
    # Executive Summary
    summary_title = ParagraphStyle(
        "SummaryTitle",
        parent=styles["Heading2"],
        fontSize=14,
        textColor=colors.HexColor(THEME["primary_navy"]),
        spaceAfter=12,
        fontName="Helvetica-Bold",
    )
    story.append(Paragraph("Executive Summary", summary_title))
    
    summary_data = [
        ["Metric", "Value"],
        ["Total Low Stock Items", str(len(low_stock_df))],
        ["Affected Branches", str(low_stock_df["store_id"].nunique() if not low_stock_df.empty else 0)],
        ["Critical Items", str(len(low_stock_df[low_stock_df.get("risk_category", "").str.lower().isin(["critical", "high"])]) if not low_stock_df.empty else 0)],
        ["Total Reorder Quantity", str(int(low_stock_df["suggested_reorder_quantity"].sum()) if not low_stock_df.empty else 0)],
    ]
    
    summary_table = Table(summary_data, colWidths=[3 * inch, 2 * inch])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(THEME["primary_navy"])),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 11),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
        ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
        ("GRID", (0, 0), (-1, -1), 1, colors.HexColor(THEME["soft_border"])),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor(THEME["light_bg"])]),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 0.3 * inch))
    
    # Detailed inventory table
    if not low_stock_df.empty:
        story.append(PageBreak())
        story.append(Paragraph("Inventory Risk Details", summary_title))
        story.append(Spacer(1, 0.15 * inch))
        
        table_data = [
            ["Product", "Store", "Current", "Threshold", "Daily Sales", "Days", "Risk", "Reorder Qty"],
        ]
        
        for _, row in low_stock_df.head(20).iterrows():
            table_data.append([
                str(row.get("product_name", "N/A"))[:20],
                str(row.get("store_name", "N/A"))[:12],
                str(int(float(row.get("current_quantity", 0)))),
                str(int(float(row.get("reorder_threshold", 0)))),
                f"{float(row.get('recent_daily_sales_velocity', 0)):.1f}",
                f"{float(row.get('predicted_days_remaining', 0)):.1f}",
                str(row.get("risk_category", "Medium")),
                str(int(float(row.get("suggested_reorder_quantity", 0)))),
            ])
        
        inventory_table = Table(table_data, colWidths=[1.2 * inch, 0.9 * inch, 0.7 * inch, 0.8 * inch, 0.8 * inch, 0.65 * inch, 0.7 * inch, 0.9 * inch])
        inventory_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(THEME["primary_navy"])),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 9),
            ("FONTSIZE", (0, 1), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 10),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor(THEME["soft_border"])),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor(THEME["light_bg"])]),
        ]))
        story.append(inventory_table)
    
    # Footer
    story.append(Spacer(1, 0.5 * inch))
    footer_style = ParagraphStyle(
        "Footer",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.HexColor(THEME["muted_text"]),
        alignment=TA_CENTER,
    )
    story.append(Paragraph(
        "This report was generated automatically by the AI Retail Inventory Optimization Platform.",
        footer_style
    ))
    
    doc.build(story)
    return output_path
