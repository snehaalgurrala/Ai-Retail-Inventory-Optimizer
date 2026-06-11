"""Premium low stock email handler with enhanced HTML formatting and Excel attachments."""

import mimetypes
import os
import smtplib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

from backend.services.premium_report_formatter import (
    generate_premium_html_email,
    generate_excel_report,
)

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
ATTACHMENTS_DIR = PROCESSED_DATA_DIR / "attachments"

# Email configuration
SENDER_EMAIL = os.getenv("EMAIL_FROM", "inventory-alerts@retailcompany.com")
SENDER_NAME = "AI Retail Inventory Optimizer"
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
EMAIL_USERNAME = os.getenv("EMAIL_USERNAME")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

PREMIUM_EMAIL_EXECUTOR = ThreadPoolExecutor(max_workers=1)


def _ensure_attachments_dir() -> Path:
    """Ensure attachments directory exists."""
    ATTACHMENTS_DIR.mkdir(parents=True, exist_ok=True)
    return ATTACHMENTS_DIR


def _generate_unique_filename(base_name: str, extension: str) -> str:
    """Generate unique filename with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{timestamp}{extension}"


def send_premium_low_stock_email(
    low_stock_df: pd.DataFrame,
    recipient_email: str,
    recipient_name: str = "Inventory Manager",
    branch_scope: str = "All Branches",
    include_attachments: bool = True,
    async_send: bool = False,
) -> dict:
    """
    Send a premium low stock alert email with HTML formatting and attachments.
    
    Args:
        low_stock_df: DataFrame with low stock items
        recipient_email: Recipient email address
        recipient_name: Recipient name for greeting
        branch_scope: Branch scope description
        include_attachments: Whether to include the Excel attachment
        async_send: Whether to send asynchronously
        
    Returns:
        Dictionary with send status and details
    """
    try:
        # Validate inputs
        if not recipient_email or not isinstance(recipient_email, str):
            return {
                "success": False,
                "error": "Invalid recipient email",
                "email_sent": False,
            }
        
        # Generate report date
        report_date = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        
        # Generate premium HTML email
        html_content = generate_premium_html_email(
            low_stock_df,
            report_date=report_date,
            branch_scope=branch_scope,
        )
        
        # Create email message
        msg = EmailMessage()
        msg["Subject"] = f"🔔 Inventory Alert: {len(low_stock_df)} Low Stock Items Detected"
        msg["From"] = f"{SENDER_NAME} <{SENDER_EMAIL}>"
        msg["To"] = f"{recipient_name} <{recipient_email}>"
        
        # Add HTML content
        msg.set_content(
            f"Low Stock Alert Report\n\n"
            f"This email contains {len(low_stock_df)} low stock items. "
            f"Please open in an HTML-capable email client to view the formatted report.",
            subtype="plain"
        )
        msg.add_alternative(html_content, subtype="html")
        
        # Generate and attach reports if requested
        attachments_info = []
        if include_attachments and not low_stock_df.empty:
            attachments_dir = _ensure_attachments_dir()
            
            try:
                # Generate Excel report
                excel_filename = "Low_Stock_Report.xlsx"
                excel_path = attachments_dir / excel_filename
                generate_excel_report(low_stock_df, excel_path)
                
                with open(excel_path, "rb") as attachment:
                    msg.add_attachment(
                        attachment.read(),
                        maintype="application",
                        subtype="vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        filename=excel_filename,
                    )
                attachments_info.append(f"Excel: {excel_filename}")
                
            except Exception as e:
                print(f"Warning: Failed to generate Excel report: {e}")
        
        # Send email
        if async_send:
            PREMIUM_EMAIL_EXECUTOR.submit(_send_email_async, msg, recipient_email)
            return {
                "success": True,
                "email_sent": True,
                "message": "Email queued for sending",
                "attachments": attachments_info,
            }
        else:
            _send_email_sync(msg)
            return {
                "success": True,
                "email_sent": True,
                "message": "Email sent successfully",
                "attachments": attachments_info,
            }
    
    except Exception as e:
        return {
            "success": False,
            "email_sent": False,
            "error": str(e),
            "message": f"Failed to send email: {str(e)}",
        }


def _send_email_sync(msg: EmailMessage) -> None:
    """Send email synchronously."""
    if not EMAIL_USERNAME or not EMAIL_PASSWORD:
        raise ValueError("Email credentials not configured in environment variables")
    
    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.starttls()
        server.login(EMAIL_USERNAME, EMAIL_PASSWORD)
        server.send_message(msg)


def _send_email_async(msg: EmailMessage, recipient_email: str) -> None:
    """Send email asynchronously."""
    try:
        _send_email_sync(msg)
    except Exception as e:
        print(f"Error sending async email to {recipient_email}: {e}")


def send_bulk_premium_emails(
    low_stock_df: pd.DataFrame,
    recipients: list[dict],
    branch_scope: str = "All Branches",
    include_attachments: bool = True,
) -> dict:
    """
    Send premium low stock emails to multiple recipients.
    
    Args:
        low_stock_df: DataFrame with low stock items
        recipients: List of dicts with 'email' and optional 'name' keys
        branch_scope: Branch scope description
        include_attachments: Whether to include attachments
        
    Returns:
        Dictionary with aggregated send results
    """
    results = {
        "total_recipients": len(recipients),
        "successful": 0,
        "failed": 0,
        "errors": [],
        "details": [],
    }
    
    for recipient in recipients:
        email = recipient.get("email", "").strip()
        name = recipient.get("name", "Inventory Manager")
        
        if not email:
            results["failed"] += 1
            results["errors"].append("Missing email address")
            continue
        
        result = send_premium_low_stock_email(
            low_stock_df,
            email,
            name,
            branch_scope=branch_scope,
            include_attachments=include_attachments,
            async_send=True,  # Use async for bulk sends
        )
        
        if result["success"]:
            results["successful"] += 1
            results["details"].append({
                "email": email,
                "status": "sent",
                "attachments": result.get("attachments", []),
            })
        else:
            results["failed"] += 1
            results["errors"].append(f"{email}: {result.get('error', 'Unknown error')}")
            results["details"].append({
                "email": email,
                "status": "failed",
                "error": result.get("error"),
            })
    
    return results


def generate_and_save_report_files(
    low_stock_df: pd.DataFrame,
    output_dir: Optional[Path] = None,
) -> dict:
    """
    Generate and save the Excel report without sending email.
    
    Args:
        low_stock_df: DataFrame with low stock items
        output_dir: Output directory for reports (default: attachments dir)
        
    Returns:
        Dictionary with file paths and status
    """
    if output_dir is None:
        output_dir = _ensure_attachments_dir()
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "success": True,
        "files": {},
        "errors": [],
    }
    
    try:
        # Generate Excel
        excel_filename = "Low_Stock_Report.xlsx"
        excel_path = output_dir / excel_filename
        generate_excel_report(low_stock_df, excel_path)
        results["files"]["excel"] = str(excel_path)
    except Exception as e:
        results["errors"].append(f"Excel generation failed: {str(e)}")
    
    results["success"] = len(results["errors"]) == 0
    return results
