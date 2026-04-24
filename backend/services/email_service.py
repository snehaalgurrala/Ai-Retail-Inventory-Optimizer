import hashlib
import os
import smtplib
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv


load_dotenv()


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
EMAIL_LOG_PATH = PROCESSED_DATA_DIR / "email_alert_log.csv"
EMAIL_LOG_COLUMNS = [
    "timestamp",
    "product_id",
    "store_id",
    "current_quantity",
    "suggested_reorder_quantity",
    "priority",
    "email_sent",
]


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


def _append_log_rows(low_stock_df: pd.DataFrame, email_sent: bool) -> None:
    if low_stock_df.empty:
        return
    log_df = _safe_read_log()
    timestamp = datetime.now().isoformat(timespec="seconds")
    rows = low_stock_df.copy()
    rows["timestamp"] = timestamp
    rows["email_sent"] = email_sent
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


def _smtp_settings() -> dict[str, str]:
    return {
        "smtp_email": os.getenv("SMTP_EMAIL", "").strip(),
        "smtp_password": os.getenv("SMTP_APP_PASSWORD", "").strip(),
        "manager_email": os.getenv("MANAGER_EMAIL", "").strip(),
    }


def _build_email_body(low_stock_df: pd.DataFrame) -> str:
    store_count = (
        low_stock_df["store_id"].astype(str).replace("", pd.NA).dropna().nunique()
        if "store_id" in low_stock_df.columns
        else 0
    )
    lines = [
        "Hello Inventory Manager,",
        "",
        (
            f"{len(low_stock_df)} low-stock rows need attention across "
            f"{store_count} store(s)."
        ),
        "",
        "Low-stock details:",
        "",
    ]

    for _, row in low_stock_df.iterrows():
        product_name = str(row.get("product_name", row.get("product_id", ""))).strip()
        store_name = str(row.get("store_name", row.get("store_id", ""))).strip()
        city = str(row.get("city", "")).strip()
        lines.extend(
            [
                f"Product: {product_name}",
                f"Store: {store_name}" + (f" ({city})" if city else ""),
                f"Current Stock: {row.get('current_quantity', '')}",
                f"Threshold: {row.get('reorder_threshold', '')}",
                f"Suggested Reorder Qty: {row.get('suggested_reorder_quantity', '')}",
                f"Priority: {row.get('priority', '')}",
                "Recommendation: Reorder immediately if priority is High, otherwise queue replenishment in the next purchase cycle.",
                "",
            ]
        )

    lines.extend(
        [
            "Please review these items and trigger replenishment where needed.",
            "",
            "Regards,",
            "AI Retail Inventory Optimizer",
        ]
    )
    return "\n".join(lines)


def send_low_stock_alert_email(low_stock_df: pd.DataFrame) -> dict:
    """Send a low-stock alert email after refresh, with duplicate suppression."""
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
    message["Subject"] = "🚨 Low Stock Alert - Inventory Reorder Required"
    message["From"] = settings["smtp_email"]
    message["To"] = settings["manager_email"]
    message.set_content(_build_email_body(low_stock_df))

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
