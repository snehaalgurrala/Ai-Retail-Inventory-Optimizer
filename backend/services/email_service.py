import hashlib
import os
import smtplib
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path
from time import perf_counter

import pandas as pd
from dotenv import load_dotenv


load_dotenv()


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


def _priority_badge(priority: str) -> str:
    priority_text = str(priority or "Medium").strip().title()
    color = {
        "High": "#dc2626",
        "Medium": "#d97706",
        "Low": "#2563eb",
    }.get(priority_text, "#475569")
    return (
        f'<span style="display:inline-block;padding:4px 10px;border-radius:999px;'
        f'background:{color};color:#ffffff;font-size:12px;font-weight:700;">'
        f'{priority_text}</span>'
    )


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
        lines.extend(
            [
                f"{index}. Product: {product_name}",
                f"   Store: {store_name}",
                f"   City: {city}",
                f"   Current Stock: {row.get('current_quantity', '')}",
                f"   Reorder Threshold: {row.get('reorder_threshold', '')}",
                f"   Suggested Reorder Quantity: {row.get('suggested_reorder_quantity', '')}",
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
        detail_rows.append(
            "<tr>"
            f"<td style='padding:10px;border-bottom:1px solid #e5e7eb;'>{index}</td>"
            f"<td style='padding:10px;border-bottom:1px solid #e5e7eb;'><strong>{row.get('product_name', row.get('product_id', ''))}</strong></td>"
            f"<td style='padding:10px;border-bottom:1px solid #e5e7eb;'>{row.get('store_name', row.get('store_id', ''))}</td>"
            f"<td style='padding:10px;border-bottom:1px solid #e5e7eb;'>{row.get('city', '')}</td>"
            f"<td style='padding:10px;border-bottom:1px solid #e5e7eb;text-align:right;'>{row.get('current_quantity', '')}</td>"
            f"<td style='padding:10px;border-bottom:1px solid #e5e7eb;text-align:right;'>{row.get('reorder_threshold', '')}</td>"
            f"<td style='padding:10px;border-bottom:1px solid #e5e7eb;text-align:right;'>{row.get('suggested_reorder_quantity', '')}</td>"
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

    log_df = _safe_read_log()
    current_signature = _alert_signature(low_stock_df)
    previous_signature = _latest_sent_signature(log_df)

    if current_signature and current_signature == previous_signature:
        _append_log_rows(low_stock_df, email_sent=False)
        return {
            "success": True,
            "email_sent": False,
            "warning": "",
            "message": "Low-stock alert email skipped because the alert has not changed.",
        }

    message = EmailMessage()
    message["Subject"] = "🚨 Low Stock Alert - Immediate Reorder Required"
    message["From"] = settings["smtp_email"]
    message["To"] = settings["manager_email"]
    message.set_content(_build_email_text_body(low_stock_df))
    message.add_alternative(_build_email_body(low_stock_df), subtype="html")

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
        "message": f"Low-stock alert email sent to {settings['manager_email']}.",
    }


def send_low_stock_alert_email(low_stock_df: pd.DataFrame) -> dict:
    """Send a low-stock alert email with timing and duplicate suppression."""
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

        log_df = _safe_read_log()
        if not _has_new_or_changed_alerts(low_stock_df, log_df):
            _append_log_rows(low_stock_df, email_sent=False, delivery_status="duplicate_skipped")
            return {
                "success": True,
                "email_sent": False,
                "warning": "",
                "message": "Low-stock alert email skipped because the alert has not changed.",
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

        message = EmailMessage()
        message["Subject"] = "🚨 Low Stock Alert - Immediate Reorder Required"
        message["From"] = settings["smtp_email"]
        message["To"] = settings["manager_email"]
        message.set_content(_build_email_text_body(low_stock_df))
        message.add_alternative(_build_email_body(low_stock_df), subtype="html")

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
            "message": f"Low-stock alert email sent to {settings['manager_email']}.",
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

    log_df = _safe_read_log()
    if not _has_new_or_changed_alerts(low_stock_df, log_df):
        _append_log_rows(low_stock_df, email_sent=False, delivery_status="duplicate_skipped")
        return {
            "success": True,
            "email_sent": False,
            "queued": False,
            "warning": "",
            "message": "Low-stock alert email skipped because the alert has not changed.",
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
