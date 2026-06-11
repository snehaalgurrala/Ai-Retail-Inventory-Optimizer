"""
Usage guide and integration examples for Premium Low Stock Report Formatter.

This module shows how to use the new premium report formatting system
to upgrade low stock alerts from basic text to executive-level reports
with professional HTML emails, Excel spreadsheets, and PDF documents.

The new system maintains full backward compatibility with existing
inventory logic and prediction calculations.
"""

from pathlib import Path
import pandas as pd

from backend.services.low_stock_service import get_low_stock_items
from backend.services.premium_report_formatter import (
    generate_premium_html_email,
    generate_excel_report,
    generate_pdf_report,
)
from backend.services.premium_low_stock_email_handler import (
    send_premium_low_stock_email,
    send_bulk_premium_emails,
    generate_and_save_report_files,
)


# ============================================================================
# EXAMPLE 1: Generate and send premium email to single recipient
# ============================================================================

def example_send_premium_email_single_recipient():
    """
    Example: Send a premium low stock alert email to a single recipient
    with HTML formatting, Excel, and PDF attachments.
    """
    # Get low stock items (existing logic unchanged)
    low_stock_df = get_low_stock_items(save_output=True)
    
    if low_stock_df.empty:
        print("No low stock items to report")
        return
    
    # Send premium email
    result = send_premium_low_stock_email(
        low_stock_df=low_stock_df,
        recipient_email="manager@retailcompany.com",
        recipient_name="Inventory Manager",
        branch_scope="All Branches",
        include_attachments=True,
        async_send=False,
    )
    
    print(f"Email send result: {result}")
    print(f"Success: {result['success']}")
    print(f"Email sent: {result['email_sent']}")
    if result.get('attachments'):
        print(f"Attachments: {', '.join(result['attachments'])}")


# ============================================================================
# EXAMPLE 2: Send bulk premium emails to multiple recipients
# ============================================================================

def example_send_premium_emails_bulk():
    """
    Example: Send premium low stock alert emails to multiple recipients
    (e.g., different managers for different branches).
    """
    # Get low stock items (existing logic unchanged)
    low_stock_df = get_low_stock_items(save_output=True)
    
    if low_stock_df.empty:
        print("No low stock items to report")
        return
    
    # Define recipients
    recipients = [
        {"email": "manager1@retailcompany.com", "name": "Regional Manager - North"},
        {"email": "manager2@retailcompany.com", "name": "Regional Manager - South"},
        {"email": "cfo@retailcompany.com", "name": "Chief Financial Officer"},
    ]
    
    # Send to all recipients
    results = send_bulk_premium_emails(
        low_stock_df=low_stock_df,
        recipients=recipients,
        branch_scope="All Branches",
        include_attachments=True,
    )
    
    print(f"\nBulk email results:")
    print(f"Total recipients: {results['total_recipients']}")
    print(f"Successful: {results['successful']}")
    print(f"Failed: {results['failed']}")
    if results.get('errors'):
        print(f"Errors: {results['errors']}")


# ============================================================================
# EXAMPLE 3: Generate report files without sending email
# ============================================================================

def example_generate_report_files_only():
    """
    Example: Generate Excel and PDF reports without sending email.
    Useful for scheduled batch processing or integration with other systems.
    """
    # Get low stock items (existing logic unchanged)
    low_stock_df = get_low_stock_items(save_output=True)
    
    if low_stock_df.empty:
        print("No low stock items to report")
        return
    
    # Define output directory
    output_dir = Path("./data/processed/reports")
    
    # Generate report files
    result = generate_and_save_report_files(
        low_stock_df=low_stock_df,
        output_dir=output_dir,
    )
    
    print(f"\nReport generation result:")
    print(f"Success: {result['success']}")
    if result.get('files'):
        print("Generated files:")
        for file_type, file_path in result['files'].items():
            print(f"  {file_type.upper()}: {file_path}")
    if result.get('errors'):
        print(f"Errors: {result['errors']}")


# ============================================================================
# EXAMPLE 4: Generate just the HTML email content (for preview/debugging)
# ============================================================================

def example_generate_html_email_only():
    """
    Example: Generate HTML email content for preview or debugging.
    Useful for testing email rendering before sending.
    """
    # Get low stock items (existing logic unchanged)
    low_stock_df = get_low_stock_items(save_output=True)
    
    if low_stock_df.empty:
        print("No low stock items to report")
        return
    
    # Generate HTML email
    html_content = generate_premium_html_email(
        low_stock_df=low_stock_df,
        report_date=None,  # Uses current date/time if not provided
        branch_scope="All Branches",
    )
    
    # Save to file for preview
    output_file = Path("./data/processed/email_preview.html")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(html_content, encoding="utf-8")
    print(f"HTML email preview saved to: {output_file}")
    
    # You can now open the file in a browser to see how it renders


# ============================================================================
# EXAMPLE 5: Integrate with FastAPI backend for API endpoint
# ============================================================================

def example_fastapi_integration():
    """
    Example FastAPI endpoint for triggering premium report generation.
    
    This can be integrated into your FastAPI backend (backend/main.py).
    """
    
    # This is pseudocode showing how to integrate into FastAPI
    # Add this to your backend/main.py or create a new endpoints file
    
    fastapi_example = """
from fastapi import APIRouter, HTTPException
from backend.services.low_stock_service import get_low_stock_items
from backend.services.premium_low_stock_email_handler import (
    send_premium_low_stock_email,
    generate_and_save_report_files,
)

router = APIRouter(prefix="/api/reports", tags=["reports"])

@router.post("/low-stock-premium")
async def send_low_stock_premium_report(
    recipient_email: str,
    branch_scope: str = "All Branches",
    include_attachments: bool = True,
):
    '''Generate and send premium low stock alert report.'''
    try:
        low_stock_df = get_low_stock_items(save_output=True)
        
        result = send_premium_low_stock_email(
            low_stock_df=low_stock_df,
            recipient_email=recipient_email,
            branch_scope=branch_scope,
            include_attachments=include_attachments,
        )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/low-stock-reports/generate")
async def generate_low_stock_reports(branch_scope: str = "All Branches"):
    '''Generate Excel and PDF reports without sending email.'''
    try:
        low_stock_df = get_low_stock_items(save_output=True)
        
        result = generate_and_save_report_files(
            low_stock_df=low_stock_df,
            output_dir=None,  # Uses default attachments directory
        )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    """
    
    print("FastAPI Integration Example:")
    print(fastapi_example)


# ============================================================================
# EXAMPLE 6: Scheduled batch processing (Cron job style)
# ============================================================================

def example_scheduled_report_generation():
    """
    Example: Scheduled daily report generation and distribution.
    
    This can be run as a cron job or scheduled task:
    - Linux/Mac: Add to crontab
    - Windows: Add to Task Scheduler
    - Python: Use APScheduler or similar
    """
    
    import logging
    from datetime import datetime
    
    logger = logging.getLogger(__name__)
    
    def daily_low_stock_report():
        """Daily low stock report generation and distribution."""
        logger.info(f"Starting daily low stock report at {datetime.now()}")
        
        try:
            # Get low stock items
            low_stock_df = get_low_stock_items(save_output=True)
            
            if low_stock_df.empty:
                logger.info("No low stock items to report")
                return
            
            # Distribution list
            recipients = [
                {"email": "inventory@retailcompany.com", "name": "Inventory Team"},
                {"email": "cfo@retailcompany.com", "name": "CFO"},
                {"email": "operations@retailcompany.com", "name": "Operations Manager"},
            ]
            
            # Send to all recipients
            results = send_bulk_premium_emails(
                low_stock_df=low_stock_df,
                recipients=recipients,
                branch_scope="All Branches",
                include_attachments=True,
            )
            
            logger.info(f"Daily report completed. Sent to {results['successful']} recipients.")
            
            if results.get('failed') > 0:
                logger.warning(f"Failed to send to {results['failed']} recipients")
                if results.get('errors'):
                    logger.error(f"Errors: {results['errors']}")
        
        except Exception as e:
            logger.error(f"Failed to generate daily report: {e}", exc_info=True)
    
    logger.info("Scheduled daily report generation function defined")


# ============================================================================
# EXAMPLE 7: Custom report generation with filtered data
# ============================================================================

def example_filtered_report():
    """
    Example: Generate reports for specific stores or product categories.
    Useful for branch managers who only care about their location.
    """
    # Get all low stock items
    all_low_stock = get_low_stock_items(save_output=True)
    
    if all_low_stock.empty:
        print("No low stock items to report")
        return
    
    # Filter by store (example: only Store 5)
    store_5_items = all_low_stock[all_low_stock['store_id'] == 5]
    
    if store_5_items.empty:
        print("No low stock items for Store 5")
        return
    
    # Send filtered report
    result = send_premium_low_stock_email(
        low_stock_df=store_5_items,
        recipient_email="store5-manager@retailcompany.com",
        recipient_name="Store 5 Manager",
        branch_scope="Store 5",
        include_attachments=True,
    )
    
    print(f"Filtered report result: {result}")


# ============================================================================
# Theme Color Reference
# ============================================================================

THEME_COLORS_REFERENCE = {
    "primary_navy": "#183F5F",       # Main navigation and headers
    "deep_navy": "#0A1F33",          # Dark text and backgrounds
    "fresh_green": "#6CB33F",        # Success indicators and accents
    "soft_green": "#A6D96A",         # Light positive indicators
    "light_bg": "#F5F8FB",           # Page background
    "white": "#FFFFFF",              # Card backgrounds
    "soft_border": "#D8E2EC",        # Subtle borders
    "muted_text": "#476C8B",         # Secondary text
    "warning": "#C76A12",            # Warning indicators
    "danger": "#B42318",             # Critical/alert indicators
    "soft_blue": "#EAF1F7",          # Information backgrounds
    "soft_amber": "#FFF7E8",         # Warning backgrounds
    "soft_red": "#FFF1F2",           # Error backgrounds
}


if __name__ == "__main__":
    print("Premium Low Stock Report Formatter - Usage Examples")
    print("=" * 60)
    print("\nAvailable examples:")
    print("1. Send premium email to single recipient")
    print("2. Send bulk emails to multiple recipients")
    print("3. Generate report files without sending email")
    print("4. Generate HTML email for preview")
    print("5. FastAPI integration example")
    print("6. Scheduled batch processing example")
    print("7. Filtered report example")
    print("\nTo run an example, call the corresponding function:")
    print("  from backend.services.premium_report_usage import example_*")
    print("  example_send_premium_email_single_recipient()")
