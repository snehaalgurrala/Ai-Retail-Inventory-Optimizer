import hashlib
import mimetypes
import os
import smtplib
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from email.message import EmailMessage
from html import escape
from pathlib import Path
from time import perf_counter

import pandas as pd
from dotenv import load_dotenv

from backend.services.depletion_formatter import (
    depletion_urgency_label,
    exact_depletion_tooltip,
    format_depletion_window,
    inventory_position_status,
    urgency_rank,
)
from backend.services.report_service import _email_shell, _metric_card, _section, _table_html

load_dotenv()

# Premium theme colors matching the Streamlit application
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
}


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
EMAIL_LOG_PATH = PROCESSED_DATA_DIR / "email_alert_log.csv"
EMAIL_LOG_COLUMNS = [
    "timestamp",
    "alert_key",
    "product_id",
    "store_id",
    "current_quantity",
    "suggested_reorder_quantity",
    "priority",
    "email_sent",
    "delivery_status",
    "error_message",
]
EMAIL_EXECUTOR = ThreadPoolExecutor(max_workers=1)


def _safe_read_log() -> pd.DataFrame:
    if not EMAIL_LOG_PATH.exists():
        return pd.DataFrame(columns=EMAIL_LOG_COLUMNS)
    try:
        log_df = pd.read_csv(EMAIL_LOG_PATH)
    except Exception:
        return pd.DataFrame(columns=EMAIL_LOG_COLUMNS)

    for column in EMAIL_LOG_COLUMNS:
        if column not in log_df.columns:
            log_df[column] = ""
    return log_df[EMAIL_LOG_COLUMNS].copy()


def _write_log(log_df: pd.DataFrame) -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    writable_df = log_df.copy()
    for column in EMAIL_LOG_COLUMNS:
        if column not in writable_df.columns:
            writable_df[column] = ""
    writable_df[EMAIL_LOG_COLUMNS].to_csv(EMAIL_LOG_PATH, index=False)


def _row_alert_key(row: pd.Series | dict) -> str:
    return "|".join(
        str((row.get(column, "") if hasattr(row, "get") else "") or "").strip()
        for column in ["product_id", "store_id", "current_quantity", "priority"]
    )


def _append_log_rows(
    low_stock_df: pd.DataFrame,
    email_sent: bool,
    delivery_status: str = "",
    error_message: str = "",
) -> None:
    if low_stock_df.empty:
        return
    log_df = _safe_read_log()
    timestamp = datetime.now().isoformat(timespec="seconds")
    rows = low_stock_df.copy()
    rows["timestamp"] = timestamp
    rows["alert_key"] = rows.apply(_row_alert_key, axis=1)
    rows["email_sent"] = email_sent
    rows["delivery_status"] = delivery_status or ("sent" if email_sent else "skipped")
    rows["error_message"] = error_message
    rows = rows[
        [
            column
            for column in EMAIL_LOG_COLUMNS
            if column in rows.columns
        ]
    ]
    _write_log(pd.concat([log_df, rows], ignore_index=True))


def _alert_signature(low_stock_df: pd.DataFrame) -> str:
    if low_stock_df.empty:
        return ""

    signature_df = low_stock_df[
        [
            column
            for column in [
                "product_id",
                "store_id",
                "current_quantity",
                "suggested_reorder_quantity",
                "priority",
            ]
            if column in low_stock_df.columns
        ]
    ].copy()
    signature_df = signature_df.fillna("").astype(str).sort_values(signature_df.columns.tolist())
    signature_text = signature_df.to_csv(index=False)
    return hashlib.sha256(signature_text.encode("utf-8")).hexdigest()


def _latest_sent_signature(log_df: pd.DataFrame) -> str:
    if log_df.empty or "email_sent" not in log_df.columns:
        return ""

    sent_rows = log_df[log_df["email_sent"].astype(str).str.lower().eq("true")].copy()
    if sent_rows.empty:
        return ""

    latest_timestamp = sent_rows["timestamp"].astype(str).max()
    latest_rows = sent_rows[sent_rows["timestamp"].astype(str).eq(latest_timestamp)].copy()
    if latest_rows.empty:
        return ""
    return _alert_signature(latest_rows)


def _has_new_or_changed_alerts(low_stock_df: pd.DataFrame, log_df: pd.DataFrame) -> bool:
    if low_stock_df.empty:
        return False
    if log_df.empty:
        return True
    if "alert_key" not in log_df.columns:
        log_df = log_df.copy()
        log_df["alert_key"] = log_df.apply(_row_alert_key, axis=1)
    sent_keys = set(
        log_df[
            log_df["email_sent"].astype(str).str.lower().eq("true")
        ]["alert_key"].fillna("").astype(str)
    )
    current_keys = {
        _row_alert_key(row)
        for _, row in low_stock_df.iterrows()
    }
    return bool(current_keys - sent_keys)


def _smtp_settings() -> dict[str, str]:
    return {
        "smtp_email": os.getenv("SMTP_EMAIL", "").strip(),
        "smtp_password": os.getenv("SMTP_APP_PASSWORD", "").strip(),
        "manager_email": os.getenv("MANAGER_EMAIL", "").strip(),
    }


def send_report_email(
    subject: str,
    html_body: str,
    attachment_path: str | Path | None = None,
    attachment_paths: list[str | Path] | tuple[str | Path, ...] | None = None,
) -> dict:
    """Send a professional HTML report email to the configured manager."""
    settings = _smtp_settings()
    if not settings["smtp_email"] or not settings["smtp_password"] or not settings["manager_email"]:
        return {
            "success": False,
            "email_sent": False,
            "warning": "Report email not sent because SMTP credentials are not configured.",
            "message": "SMTP credentials are missing. Please set SMTP_EMAIL, SMTP_APP_PASSWORD, and MANAGER_EMAIL in .env.",
        }

    message = EmailMessage()
    message["Subject"] = str(subject)
    message["From"] = settings["smtp_email"]
    message["To"] = settings["manager_email"]
    message.set_content(
        "Hello Inventory Manager,\n\n"
        "Your AI Retail Inventory Optimizer report is attached and also available in HTML format.\n\n"
        "Regards,\nAI Retail Inventory Optimizer"
    )
    message.add_alternative(str(html_body or "<p>No report details available.</p>"), subtype="html")

    attachment_warning = ""
    paths: list[str | Path] = []
    if attachment_paths:
        paths.extend(attachment_paths)
    elif attachment_path:
        paths.append(attachment_path)

    for attachment in paths:
        path = Path(attachment)
        try:
            if path.exists() and path.is_file():
                content_type, _ = mimetypes.guess_type(path.name)
                maintype, subtype = (content_type or "text/csv").split("/", 1)
                message.add_attachment(
                    path.read_bytes(),
                    maintype=maintype,
                    subtype=subtype,
                    filename=path.name,
                )
            else:
                attachment_warning = "One attachment file was not found, so the email was sent with available files."
        except Exception as error:
            attachment_warning = f"One attachment could not be added, so the email was sent with available files: {error}"

    try:
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as server:
            server.starttls()
            server.login(settings["smtp_email"], settings["smtp_password"])
            server.send_message(message)
    except Exception as error:
        return {
            "success": False,
            "email_sent": False,
            "warning": "",
            "message": f"Report email could not be sent: {error}",
        }

    return {
        "success": True,
        "email_sent": True,
        "warning": attachment_warning,
        "message": f"Report email sent to {settings['manager_email']}.",
    }


def _priority_badge(priority: str) -> str:
    priority_text = str(priority or "Medium").strip().title()
    color = {
        "Critical": "#dc2626",
        "High": "#ea580c",
        "High Risk": "#ea580c",
        "Medium": "#d97706",
        "Low": "#2563eb",
        "Healthy": "#16a34a",
    }.get(priority_text, "#475569")
    return (
        f'<span style="display:inline-block;padding:4px 10px;border-radius:999px;'
        f'background:{color};color:#ffffff;font-size:12px;font-weight:700;">'
        f'{priority_text}</span>'
    )


def _depletion_display(row: pd.Series | dict) -> tuple[str, str, str]:
    """Resolve (urgency, window, exact) for a low-stock row.

    A demand forecast is only trusted when there is genuine recent velocity and a
    finite day estimate; otherwise the row is classified purely by its inventory
    position (stock vs. reorder point). Even when a forecast exists, the inventory
    position overrides it whenever the position is the more urgent of the two —
    this is what stops a near-zero velocity from reporting "Inventory stable" for
    an item that has actually reached its reorder threshold.
    """
    days = pd.to_numeric(
        pd.Series([row.get("predicted_days_remaining", row.get("days_of_stock_remaining", 999))]),
        errors="coerce",
    ).fillna(999.0).iloc[0]
    velocity = pd.to_numeric(
        pd.Series([row.get("recent_daily_sales_velocity", 0)]), errors="coerce"
    ).fillna(0.0).iloc[0]
    position = inventory_position_status(
        row.get("current_quantity"), row.get("reorder_threshold")
    )
    has_forecast = velocity > 0 and days < 999

    if has_forecast:
        urgency = str(row.get("urgency_label") or depletion_urgency_label(days))
        window = str(row.get("depletion_window") or format_depletion_window(days))
        exact = exact_depletion_tooltip(days)
        # Only resolve the *contradiction*: a forecast that reads "Healthy/stable"
        # purely because velocity rounds low, while stock is actually at/below the
        # reorder point. A genuinely urgent day-based forecast (e.g. "2-5 days
        # remaining") is more actionable than a position label, so it is kept.
        is_healthy_forecast = urgency_rank(urgency) >= urgency_rank("Healthy")
        if position and is_healthy_forecast and urgency_rank(position[0]) < urgency_rank(urgency):
            urgency, window = position
        return urgency, window, exact

    # No reliable demand signal — classify on inventory position alone and do not
    # emit a placeholder day estimate.
    if position:
        return position[0], position[1], ""
    return "Monitor", "Insufficient sales history", ""


def _get_risk_badge_html(urgency: str) -> str:
    """Generate HTML badge for urgency level with theme colors."""
    urgency_lower = str(urgency or "Medium").lower().strip()
    if urgency_lower in ["critical", "high"]:
        bg_color = THEME["danger"]
        text_color = "#FFFFFF"
    elif urgency_lower == "medium":
        bg_color = THEME["warning"]
        text_color = "#FFFFFF"
    else:
        bg_color = THEME["soft_green"]
        text_color = THEME["deep_navy"]
    
    return f"""<span style="
        display: inline-block;
        padding: 6px 12px;
        border-radius: 6px;
        background-color: {bg_color};
        color: {text_color};
        font-weight: 600;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    ">{escape(str(urgency))}</span>"""


def _build_premium_html_email(low_stock_df: pd.DataFrame) -> str:
    """Build a premium HTML low-stock alert email using the shared report layout."""
    if low_stock_df.empty:
        return "<p>No low-stock items found.</p>"

    def _safe_number(value, fallback: float = 0.0) -> float:
        parsed = pd.to_numeric(pd.Series([value]), errors="coerce").fillna(fallback)
        return float(parsed.iloc[0])

    report_df = low_stock_df.copy()
    if "priority" in report_df.columns:
        priority_rank = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        report_df["_priority_rank"] = (
            report_df["priority"].fillna("").astype(str).str.casefold().map(priority_rank).fillna(4)
        )
        report_df = report_df.sort_values("_priority_rank").drop(columns=["_priority_rank"], errors="ignore")

    detail_rows = []
    for _, row in report_df.iterrows():
        urgency, window, _exact = _depletion_display(row)
        velocity = _safe_number(row.get("recent_daily_sales_velocity", 0))
        reorder_point = int(_safe_number(row.get("reorder_threshold", 0)))
        detail_rows.append(
            {
                "product_name": row.get("product_name", row.get("product_id", "N/A")),
                "branch": row.get("store_name", row.get("store_id", "N/A")),
                "current_stock": int(_safe_number(row.get("current_quantity", 0))),
                "reorder_point": reorder_point,
                # No fabricated demand: an item flagged purely by an order draw-down
                # has no recent sales velocity, so say so plainly rather than "0.0".
                "avg_daily_sales": f"{velocity:.1f}/day" if velocity > 0 else "No recent demand",
                "inventory_position": window,
                "status": urgency,
                "suggested_reorder": int(_safe_number(row.get("suggested_reorder_quantity", 0))),
                "ai_reasoning": str(row.get("ai_alert_message", "Review for reorder"))[:160],
            }
        )
    detail_df = pd.DataFrame(detail_rows)

    low_stock_count = len(report_df)
    affected_branches = report_df["store_id"].nunique() if "store_id" in report_df.columns else 0
    action_states = ["critical", "high", "reorder required"]
    status_series = (
        detail_df["status"].fillna("").astype(str).str.casefold()
        if "status" in detail_df.columns
        else pd.Series(dtype=str)
    )
    priority_series = (
        report_df["priority"].fillna("").astype(str).str.casefold()
        if "priority" in report_df.columns
        else pd.Series(dtype=str)
    )
    critical_count = int(
        status_series.isin(action_states).sum()
        if not status_series.empty
        else priority_series.isin(action_states).sum()
    )
    top_item = str(report_df.iloc[0].get("product_name", report_df.iloc[0].get("product_id", "N/A"))).strip()
    total_reorder = int(
        pd.to_numeric(
            report_df.get("suggested_reorder_quantity", pd.Series(dtype=float)),
            errors="coerce",
        )
        .fillna(0)
        .sum()
    )
    report_date = datetime.now().strftime("%B %d, %Y at %I:%M %p")

    cards = "".join(
        [
            _metric_card("Low Stock Items", f"{low_stock_count:,}"),
            _metric_card("Affected Branches", f"{affected_branches:,}"),
            _metric_card("Requires Action", f"{critical_count:,}"),
        ]
    )
    summary = (
        "<p style='margin:0;line-height:1.6;color:#476C8B;'>"
        f"The latest inventory scan found <strong style='color:#0A1F33;'>{low_stock_count:,}</strong> low-stock item(s) "
        f"across <strong style='color:#0A1F33;'>{affected_branches:,}</strong> branch(es). "
        f"Start with <strong style='color:#0A1F33;'>{escape(top_item)}</strong> and review "
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
                        "reorder_point",
                        "status",
                        "inventory_position",
                        "avg_daily_sales",
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
        f"All Branches | Generated {report_date}",
        cards,
        sections,
    )
    
    # Calculate KPI metrics
    low_stock_count = len(low_stock_df)
    affected_branches = low_stock_df["store_id"].nunique() if "store_id" in low_stock_df.columns else 0
    highest_risk = str(low_stock_df.iloc[0].get("product_name", "Product")) if not low_stock_df.empty else "N/A"
    critical_count = len(low_stock_df[low_stock_df.get("priority", "").str.lower().isin(["critical", "high"])])
    
    # Generate table rows
    table_rows = ""
    for idx, row in low_stock_df.iterrows():
        urgency, window, _ = _depletion_display(row)
        product = escape(str(row.get("product_name", "N/A")))
        store = escape(str(row.get("store_name", "N/A")))
        current_stock = int(float(row.get("current_quantity", 0)))
        avg_daily_sales = f"{float(row.get('recent_daily_sales_velocity', 0)):.1f}"
        reorder_qty = int(float(row.get("suggested_reorder_quantity", 0)))
        ai_reasoning = escape(str(row.get("ai_alert_message", "Review for reorder"))[:80])
        
        row_bg = THEME["white"] if idx % 2 == 0 else THEME["light_bg"]
        badge_html = _get_risk_badge_html(urgency)
        
        table_rows += f"""
        <tr style="background-color: {row_bg};">
            <td style="padding: 12px 15px; border-bottom: 1px solid {THEME['soft_border']}; font-weight: 500; color: {THEME['deep_navy']};">{product}</td>
            <td style="padding: 12px 15px; border-bottom: 1px solid {THEME['soft_border']}; color: {THEME['muted_text']};">{store}</td>
            <td style="padding: 12px 15px; border-bottom: 1px solid {THEME['soft_border']}; text-align: center; font-weight: 600; color: {THEME['primary_navy']};">{current_stock}</td>
            <td style="padding: 12px 15px; border-bottom: 1px solid {THEME['soft_border']}; text-align: center; color: {THEME['muted_text']};">{avg_daily_sales}</td>
            <td style="padding: 12px 15px; border-bottom: 1px solid {THEME['soft_border']}; text-align: center;">{badge_html}</td>
            <td style="padding: 12px 15px; border-bottom: 1px solid {THEME['soft_border']}; text-align: center; font-weight: 600; color: {THEME['danger']};">{reorder_qty}</td>
            <td style="padding: 12px 15px; border-bottom: 1px solid {THEME['soft_border']}; font-size: 12px; color: {THEME['muted_text']};">{ai_reasoning}</td>
        </tr>
        """
    
    # Generate branch summary
    branch_summary = ""
    if "store_name" in low_stock_df.columns:
        branch_data = low_stock_df.groupby("store_name").size().reset_index(name="count")
        for _, b_row in branch_data.iterrows():
            store = escape(str(b_row["store_name"]))
            count = b_row["count"]
            branch_summary += f"""
            <div style="
                background: white;
                border: 1px solid {THEME['soft_border']};
                border-left: 4px solid {THEME['fresh_green']};
                border-radius: 6px;
                padding: 12px;
                margin-bottom: 8px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            ">
                <span style="color: {THEME['deep_navy']}; font-weight: 600;">{store}</span>
                <span style="background: {THEME['soft_green']}; color: {THEME['deep_navy']}; padding: 4px 10px; border-radius: 20px; font-weight: 600; font-size: 12px;">{count} items</span>
            </div>
            """
    
    report_date = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Low Stock Alert</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Helvetica Neue', Arial, sans-serif; }}
            @media (max-width: 600px) {{
                .kpi-grid {{ grid-template-columns: 1fr !important; }}
                .table-scroll {{ overflow-x: auto; }}
            }}
        </style>
    </head>
    <body style="margin: 0; padding: 20px; background-color: {THEME['light_bg']}; font-family: Arial, sans-serif; color: {THEME['deep_navy']};">
        <div style="max-width: 1000px; margin: 0 auto; background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 8px 24px rgba(10, 31, 51, 0.12);">
            <!-- Header -->
            <div style="background: linear-gradient(135deg, {THEME['primary_navy']}, {THEME['deep_navy']}); color: white; padding: 40px 30px; text-align: center;">
                <h1 style="margin: 0; font-size: 32px; font-weight: 800; letter-spacing: -0.5px; margin-bottom: 8px;">Inventory Intelligence</h1>
                <p style="margin: 0 0 12px 0; font-size: 18px; font-weight: 600; opacity: 0.95;">Low Stock Alert Report</p>
                <div style="font-size: 13px; opacity: 0.85;">Report Generated: {report_date}</div>
            </div>
            
            <!-- Executive Summary & KPI Cards -->
            <div style="padding: 30px 30px; background: {THEME['light_bg']};">
                <h2 style="color: {THEME['deep_navy']}; font-size: 18px; font-weight: 700; margin: 0 0 20px 0; padding-bottom: 12px; border-bottom: 2px solid {THEME['primary_navy']};">Executive Summary</h2>
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; margin-bottom: 0;" class="kpi-grid">
                    <div style="background: white; border: 2px solid {THEME['soft_border']}; border-left: 4px solid {THEME['danger']}; border-radius: 8px; padding: 20px; box-shadow: 0 4px 12px rgba(10, 31, 51, 0.08);">
                        <div style="font-size: 13px; color: {THEME['muted_text']}; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px;">Low Stock Items</div>
                        <div style="font-size: 36px; font-weight: 800; color: {THEME['primary_navy']};">{low_stock_count}</div>
                    </div>
                    <div style="background: white; border: 2px solid {THEME['soft_border']}; border-left: 4px solid {THEME['warning']}; border-radius: 8px; padding: 20px; box-shadow: 0 4px 12px rgba(10, 31, 51, 0.08);">
                        <div style="font-size: 13px; color: {THEME['muted_text']}; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px;">Affected Branches</div>
                        <div style="font-size: 36px; font-weight: 800; color: {THEME['primary_navy']};">{affected_branches}</div>
                    </div>
                    <div style="background: white; border: 2px solid {THEME['soft_border']}; border-left: 4px solid {THEME['fresh_green']}; border-radius: 8px; padding: 20px; box-shadow: 0 4px 12px rgba(10, 31, 51, 0.08);">
                        <div style="font-size: 13px; color: {THEME['muted_text']}; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px;">Requires Action</div>
                        <div style="font-size: 36px; font-weight: 800; color: {THEME['primary_navy']};">{critical_count}</div>
                    </div>
                    <div style="background: {THEME['soft_red']}; border: 2px solid {THEME['danger']}; border-left: 4px solid {THEME['danger']}; border-radius: 8px; padding: 20px; box-shadow: 0 4px 12px rgba(10, 31, 51, 0.08);">
                        <div style="font-size: 13px; color: {THEME['danger']}; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px;">Highest Risk</div>
                        <div style="font-size: 16px; font-weight: 800; color: {THEME['deep_navy']}; word-wrap: break-word;">{escape(highest_risk)}</div>
                    </div>
                </div>
            </div>
            
            <!-- Low Stock Items Table -->
            <div style="padding: 30px 30px;">
                <h2 style="color: {THEME['deep_navy']}; font-size: 18px; font-weight: 700; margin: 0 0 20px 0; padding-bottom: 12px; border-bottom: 2px solid {THEME['primary_navy']};">Low Stock Items</h2>
                <div style="overflow-x: auto; margin-top: 16px; border-radius: 8px; border: 1px solid {THEME['soft_border']}; box-shadow: 0 4px 12px rgba(10, 31, 51, 0.08);" class="table-scroll">
                    <table style="width: 100%; border-collapse: collapse; background: white;">
                        <thead>
                            <tr style="background: {THEME['primary_navy']}; color: white;">
                                <th style="padding: 14px 15px; text-align: left; font-weight: 700; font-size: 13px; text-transform: uppercase; letter-spacing: 0.5px;">Product</th>
                                <th style="padding: 14px 15px; text-align: left; font-weight: 700; font-size: 13px; text-transform: uppercase; letter-spacing: 0.5px;">Branch</th>
                                <th style="padding: 14px 15px; text-align: center; font-weight: 700; font-size: 13px; text-transform: uppercase; letter-spacing: 0.5px;">Current Stock</th>
                                <th style="padding: 14px 15px; text-align: center; font-weight: 700; font-size: 13px; text-transform: uppercase; letter-spacing: 0.5px;">Avg Daily Sales</th>
                                <th style="padding: 14px 15px; text-align: center; font-weight: 700; font-size: 13px; text-transform: uppercase; letter-spacing: 0.5px;">Depletion Window</th>
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
            
            <!-- Branch Summary -->
            <div style="padding: 30px 30px;">
                <h2 style="color: {THEME['deep_navy']}; font-size: 18px; font-weight: 700; margin: 0 0 20px 0; padding-bottom: 12px; border-bottom: 2px solid {THEME['primary_navy']};">Branch-Wise Summary</h2>
                <div style="margin-top: 16px;">
                    {branch_summary}
                </div>
            </div>
            
            <!-- Action Items -->
            <div style="padding: 30px 30px;">
                <h2 style="color: {THEME['deep_navy']}; font-size: 18px; font-weight: 700; margin: 0 0 20px 0; padding-bottom: 12px; border-bottom: 2px solid {THEME['primary_navy']};">Recommended Actions</h2>
                <div style="background: {THEME['soft_red']}; border: 1px solid {THEME['danger']}; border-left: 4px solid {THEME['danger']}; border-radius: 8px; padding: 20px; margin-bottom: 16px;">
                    <h3 style="margin: 0 0 12px 0; color: {THEME['danger']}; font-size: 16px; font-weight: 700;">⚠️ IMMEDIATE ACTION REQUIRED</h3>
                    <p style="margin: 0; color: {THEME['deep_navy']}; font-size: 14px; line-height: 1.6;">Review the {critical_count} critical item(s) above. Process reorders immediately to prevent stockouts.</p>
                </div>
                <div style="background: {THEME['soft_blue']}; border: 1px solid {THEME['primary_navy']}; border-left: 4px solid {THEME['fresh_green']}; border-radius: 8px; padding: 20px;">
                    <h3 style="margin: 0 0 12px 0; color: {THEME['primary_navy']}; font-size: 16px; font-weight: 700;">✓ Next Steps</h3>
                    <ul style="margin: 0; color: {THEME['deep_navy']}; font-size: 14px; line-height: 1.8; padding-left: 20px;">
                        <li>Review the low stock items table above</li>
                        <li>Prioritize reorders for high-risk items first</li>
                        <li>Update supplier based on suggested reorder quantities</li>
                        <li>Monitor for demand spikes</li>
                    </ul>
                </div>
            </div>
            
            <!-- Footer -->
            <div style="background: {THEME['deep_navy']}; color: white; padding: 30px 30px; text-align: center; font-size: 12px; line-height: 1.8;">
                <p style="margin: 0 0 12px 0; opacity: 0.9;">This report was generated automatically by the <strong>AI Retail Inventory Optimization Platform</strong>.</p>
                <p style="margin: 0; opacity: 0.7; font-size: 11px;">For questions or to disable these alerts, please contact your inventory management team.</p>
            </div>
        </div>
    </body>
    </html>
    """


def _generate_low_stock_excel(low_stock_df: pd.DataFrame) -> Path:
    """Generate an Excel file with low stock items for email attachment."""
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    except ImportError:
        raise ImportError("openpyxl is required. Install with: pip install openpyxl")
    
    # Create workbook
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
    
    # Define columns
    columns = [
        "Product",
        "Branch",
        "Current Stock",
        "Reorder Point",
        "Status",
        "Inventory Position",
        "Avg Daily Sales",
        "Suggested Reorder Qty",
        "AI Reasoning",
    ]

    # Add header row
    for col_num, column_title in enumerate(columns, 1):
        cell = ws.cell(row=1, column=col_num)
        cell.value = column_title
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = header_alignment
        cell.border = border

    # Add data rows
    for row_num, (_, row_data) in enumerate(low_stock_df.iterrows(), 2):
        urgency, window, _ = _depletion_display(row_data)
        velocity = float(pd.to_numeric(row_data.get("recent_daily_sales_velocity", 0), errors="coerce") or 0)

        ws.cell(row=row_num, column=1).value = str(row_data.get("product_name", "N/A"))
        ws.cell(row=row_num, column=2).value = str(row_data.get("store_name", "N/A"))
        ws.cell(row=row_num, column=3).value = int(float(row_data.get("current_quantity", 0)))
        ws.cell(row=row_num, column=4).value = int(float(row_data.get("reorder_threshold", 0)))
        ws.cell(row=row_num, column=5).value = urgency
        ws.cell(row=row_num, column=6).value = window
        ws.cell(row=row_num, column=7).value = f"{velocity:.2f}/day" if velocity > 0 else "No recent demand"
        ws.cell(row=row_num, column=8).value = int(float(row_data.get("suggested_reorder_quantity", 0)))
        ws.cell(row=row_num, column=9).value = str(row_data.get("ai_alert_message", "Review for reorder"))

    # Auto-adjust column widths
    ws.column_dimensions["A"].width = 25
    ws.column_dimensions["B"].width = 20
    ws.column_dimensions["C"].width = 14
    ws.column_dimensions["D"].width = 14
    ws.column_dimensions["E"].width = 16
    ws.column_dimensions["F"].width = 24
    ws.column_dimensions["G"].width = 16
    ws.column_dimensions["H"].width = 18
    ws.column_dimensions["I"].width = 40
    
    # Save file
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DATA_DIR / "Low_Stock_Report.xlsx"
    wb.save(output_path)
    
    return output_path


def _build_email_text_body(low_stock_df: pd.DataFrame) -> str:
    store_count = (
        low_stock_df["store_id"].astype(str).replace("", pd.NA).dropna().nunique()
        if "store_id" in low_stock_df.columns
        else 0
    )
    product_count = (
        low_stock_df["product_id"].astype(str).replace("", pd.NA).dropna().nunique()
        if "product_id" in low_stock_df.columns
        else len(low_stock_df)
    )
    top_row = low_stock_df.iloc[0] if not low_stock_df.empty else {}
    top_item = str(top_row.get("product_name", top_row.get("product_id", ""))).strip() if hasattr(top_row, "get") else ""
    lines = [
        "Hello Inventory Manager,",
        "",
        "The AI Inventory Orchestrator detected low-stock items that require attention.",
        "",
        "Summary:",
        f"- Total low-stock products: {product_count}",
        f"- Stores affected: {store_count}",
        f"- Highest priority item: {top_item}",
        "- Recommended action: Review and initiate procurement for high-priority items first.",
        "",
        "Low Stock Details:",
    ]

    for index, (_, row) in enumerate(low_stock_df.iterrows(), start=1):
        product_name = str(row.get("product_name", row.get("product_id", ""))).strip()
        store_name = str(row.get("store_name", row.get("store_id", ""))).strip()
        city = str(row.get("city", "")).strip()
        urgency, window, exact = _depletion_display(row)
        window_line = f"   Inventory Position: {window}" + (f" ({exact})" if exact else "")
        lines.extend(
            [
                f"{index}. Product: {product_name}",
                f"   Store: {store_name}",
                f"   City: {city}",
                f"   Current Stock: {row.get('current_quantity', '')}",
                f"   Reorder Threshold: {row.get('reorder_threshold', '')}",
                f"   Suggested Reorder Quantity: {row.get('suggested_reorder_quantity', '')}",
                f"   Status: {urgency}",
                window_line,
                f"   Priority: {row.get('priority', '')}",
                "   Agent Recommendation: Reorder immediately if priority is High; otherwise queue replenishment in the next purchase cycle.",
                "",
            ]
        )

    lines.extend(
        [
            "Why this matters:",
            "These items are below reorder threshold and may cause stockout if not replenished on time.",
            "",
            "Suggested Next Step:",
            "Please review and initiate procurement for the high-priority items first.",
            "",
            "Regards,",
            "AI Retail Inventory Optimizer",
        ]
    )
    return "\n".join(lines)


def _build_email_body(low_stock_df: pd.DataFrame) -> str:
    """Build a professional HTML low-stock alert email."""
    if low_stock_df.empty:
        return "<p>No low-stock items found.</p>"

    product_count = (
        low_stock_df["product_id"].astype(str).replace("", pd.NA).dropna().nunique()
        if "product_id" in low_stock_df.columns
        else len(low_stock_df)
    )
    store_count = (
        low_stock_df["store_id"].astype(str).replace("", pd.NA).dropna().nunique()
        if "store_id" in low_stock_df.columns
        else 0
    )
    priority_rank = {"High": 0, "Medium": 1, "Low": 2}
    sorted_df = low_stock_df.copy()
    if "priority" in sorted_df.columns:
        sorted_df["_priority_rank"] = sorted_df["priority"].map(priority_rank).fillna(3)
    else:
        sorted_df["_priority_rank"] = 3
    sorted_df = sorted_df.sort_values("_priority_rank").drop(columns=["_priority_rank"], errors="ignore")
    top_row = sorted_df.iloc[0]
    top_item = str(top_row.get("product_name", top_row.get("product_id", ""))).strip()
    recommended_action = "Review and initiate procurement for high-priority items first."

    detail_rows = []
    for index, (_, row) in enumerate(sorted_df.iterrows(), start=1):
        urgency, window, exact = _depletion_display(row)
        detail_rows.append(
            "<tr>"
            f"<td style='padding:10px;border-bottom:1px solid #e5e7eb;'>{index}</td>"
            f"<td style='padding:10px;border-bottom:1px solid #e5e7eb;'><strong>{row.get('product_name', row.get('product_id', ''))}</strong></td>"
            f"<td style='padding:10px;border-bottom:1px solid #e5e7eb;'>{row.get('store_name', row.get('store_id', ''))}</td>"
            f"<td style='padding:10px;border-bottom:1px solid #e5e7eb;'>{row.get('city', '')}</td>"
            f"<td style='padding:10px;border-bottom:1px solid #e5e7eb;text-align:right;'>{row.get('current_quantity', '')}</td>"
            f"<td style='padding:10px;border-bottom:1px solid #e5e7eb;text-align:right;'>{row.get('reorder_threshold', '')}</td>"
            f"<td style='padding:10px;border-bottom:1px solid #e5e7eb;text-align:right;'>{row.get('suggested_reorder_quantity', '')}</td>"
            f"<td style='padding:10px;border-bottom:1px solid #e5e7eb;'>{_priority_badge(urgency)}</td>"
            f"<td style='padding:10px;border-bottom:1px solid #e5e7eb;' title='{exact}'>{window}<br><span style='font-size:11px;color:#64748b;'>{exact}</span></td>"
            f"<td style='padding:10px;border-bottom:1px solid #e5e7eb;'>{_priority_badge(row.get('priority', ''))}</td>"
            "<td style='padding:10px;border-bottom:1px solid #e5e7eb;'>Reorder immediately if priority is High; otherwise queue replenishment in the next purchase cycle.</td>"
            "</tr>"
        )

    return f"""
    <html>
      <body style="margin:0;padding:0;background:#f8fafc;font-family:Arial,sans-serif;color:#111827;">
        <div style="max-width:900px;margin:0 auto;padding:24px;">
          <div style="background:#ffffff;border:1px solid #e5e7eb;border-radius:8px;padding:24px;">
            <h2 style="margin:0 0 12px;color:#b91c1c;">Low Stock Alert</h2>
            <p>Hello Inventory Manager,</p>
            <p>The AI Inventory Orchestrator detected low-stock items that require attention.</p>
            <h3 style="margin-top:24px;">Summary</h3>
            <ul style="line-height:1.7;">
              <li><strong>Total low-stock products:</strong> {product_count}</li>
              <li><strong>Stores affected:</strong> {store_count}</li>
              <li><strong>Highest priority item:</strong> {top_item}</li>
              <li><strong>Recommended action:</strong> {recommended_action}</li>
            </ul>
            <h3 style="margin-top:24px;">Low Stock Details</h3>
            <table style="width:100%;border-collapse:collapse;font-size:14px;">
              <thead>
                <tr style="background:#f1f5f9;text-align:left;">
                  <th style="padding:10px;">#</th>
                  <th style="padding:10px;">Product</th>
                  <th style="padding:10px;">Store</th>
                  <th style="padding:10px;">City</th>
                  <th style="padding:10px;text-align:right;">Current Stock</th>
                  <th style="padding:10px;text-align:right;">Threshold</th>
                  <th style="padding:10px;text-align:right;">Suggested Qty</th>
                  <th style="padding:10px;">Urgency</th>
                  <th style="padding:10px;">Depletion Window</th>
                  <th style="padding:10px;">Priority</th>
                  <th style="padding:10px;">Agent Recommendation</th>
                </tr>
              </thead>
              <tbody>{''.join(detail_rows)}</tbody>
            </table>
            <h3 style="margin-top:24px;">Why this matters</h3>
            <p>These items are below reorder threshold and may cause stockout if not replenished on time.</p>
            <h3 style="margin-top:20px;">Suggested Next Step</h3>
            <p>Please review and initiate procurement for the high-priority items first.</p>
            <p style="margin-top:28px;">Regards,<br><strong>AI Retail Inventory Optimizer</strong></p>
          </div>
        </div>
      </body>
    </html>
    """


def _legacy_send_low_stock_alert_email(low_stock_df: pd.DataFrame) -> dict:
    """Legacy synchronous sender kept for compatibility with older imports."""
    low_stock_df = low_stock_df.copy()
    if low_stock_df.empty:
        return {
            "success": False,
            "email_sent": False,
            "warning": "",
            "message": "No low-stock items found.",
        }

    settings = _smtp_settings()
    if not settings["smtp_email"] or not settings["smtp_password"] or not settings["manager_email"]:
        _append_log_rows(low_stock_df, email_sent=False)
        return {
            "success": False,
            "email_sent": False,
            "warning": "Low-stock email not sent because SMTP credentials are not configured.",
            "message": "SMTP credentials are missing.",
        }

    # Generate premium HTML email
    html_body = _build_premium_html_email(low_stock_df)
    
    # Generate Excel attachment
    excel_path = _generate_low_stock_excel(low_stock_df)

    message = EmailMessage()
    message["Subject"] = "🚨 Low Stock Alert - Immediate Reorder Required"
    message["From"] = settings["smtp_email"]
    message["To"] = settings["manager_email"]
    message.set_content(_build_email_text_body(low_stock_df))
    message.add_alternative(html_body, subtype="html")
    
    # Attach Excel file
    try:
        excel_content = excel_path.read_bytes()
        message.add_attachment(
            excel_content,
            maintype="application",
            subtype="vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            filename="Low_Stock_Report.xlsx",
        )
    except Exception as error:
        pass  # Continue without attachment if it fails

    try:
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as server:
            server.starttls()
            server.login(settings["smtp_email"], settings["smtp_password"])
            server.send_message(message)
    except Exception as error:
        _append_log_rows(low_stock_df, email_sent=False)
        return {
            "success": False,
            "email_sent": False,
            "warning": "",
            "message": f"Low-stock email could not be sent: {error}",
        }

    _append_log_rows(low_stock_df, email_sent=True)

    return {
        "success": True,
        "email_sent": True,
        "warning": "",
        "message": f"Low-stock alert email sent to {settings['manager_email']} with attachment.",
    }


def send_low_stock_alert_email(low_stock_df: pd.DataFrame) -> dict:
    """Send a low-stock alert email with premium formatting and Excel attachment."""
    started = perf_counter()
    low_stock_df = low_stock_df.copy()
    try:
        if low_stock_df.empty:
            return {
                "success": False,
                "email_sent": False,
                "warning": "",
                "message": "No low-stock items found.",
            }

        settings = _smtp_settings()
        if not settings["smtp_email"] or not settings["smtp_password"] or not settings["manager_email"]:
            _append_log_rows(low_stock_df, email_sent=False, delivery_status="missing_smtp")
            return {
                "success": False,
                "email_sent": False,
                "warning": "Low-stock email not sent because SMTP credentials are not configured.",
                "message": "SMTP credentials are missing.",
            }

        # Generate premium HTML email
        html_body = _build_premium_html_email(low_stock_df)
        
        # Generate Excel attachment
        excel_path = _generate_low_stock_excel(low_stock_df)

        message = EmailMessage()
        message["Subject"] = "🚨 Low Stock Alert - Immediate Reorder Required"
        message["From"] = settings["smtp_email"]
        message["To"] = settings["manager_email"]
        message.set_content(_build_email_text_body(low_stock_df))
        message.add_alternative(html_body, subtype="html")
        
        # Attach Excel file
        try:
            excel_content = excel_path.read_bytes()
            message.add_attachment(
                excel_content,
                maintype="application",
                subtype="vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                filename="Low_Stock_Report.xlsx",
            )
        except Exception as error:
            return {
                "success": False,
                "email_sent": False,
                "warning": "",
                "message": f"Could not attach Excel file: {error}",
            }

        try:
            with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as server:
                server.starttls()
                server.login(settings["smtp_email"], settings["smtp_password"])
                server.send_message(message)
        except Exception as error:
            _append_log_rows(
                low_stock_df,
                email_sent=False,
                delivery_status="failed",
                error_message=str(error),
            )
            return {
                "success": False,
                "email_sent": False,
                "warning": "",
                "message": f"Low-stock email could not be sent: {error}",
            }

        _append_log_rows(low_stock_df, email_sent=True, delivery_status="sent")
        return {
            "success": True,
            "email_sent": True,
            "warning": "",
            "message": f"Low-stock alert email sent to {settings['manager_email']} with Low_Stock_Report.xlsx attachment.",
        }
    finally:
        print(f"[agent_refresh] email time: {round(perf_counter() - started, 3)}s")


def queue_low_stock_alert_email(low_stock_df: pd.DataFrame) -> dict:
    """Queue a low-stock alert email without blocking dashboard refresh."""
    low_stock_df = low_stock_df.copy()
    if low_stock_df.empty:
        return {
            "success": False,
            "email_sent": False,
            "queued": False,
            "warning": "",
            "message": "No low-stock items found.",
        }

    settings = _smtp_settings()
    if not settings["smtp_email"] or not settings["smtp_password"] or not settings["manager_email"]:
        _append_log_rows(low_stock_df, email_sent=False, delivery_status="missing_smtp")
        return {
            "success": False,
            "email_sent": False,
            "queued": False,
            "warning": "Low-stock email not queued because SMTP credentials are not configured.",
            "message": "SMTP credentials are missing.",
        }

    future: Future = EMAIL_EXECUTOR.submit(send_low_stock_alert_email, low_stock_df)
    return {
        "success": True,
        "email_sent": False,
        "queued": True,
        "future": future,
        "warning": "",
        "message": f"Low-stock alert email queued for {settings['manager_email']}.",
    }
