# Premium Low Stock Report - Quick Start Guide

## ⚡ 5-Minute Setup

### Step 1: Update Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Configure Email
Add to `.env`:
```env
EMAIL_FROM=inventory-alerts@retailcompany.com
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
EMAIL_USERNAME=your-email@gmail.com
EMAIL_PASSWORD=your-app-password
```

### Step 3: Run Tests
```bash
python backend/services/test_premium_reports.py
```

### Step 4: Send Your First Premium Email
```python
from backend.services.low_stock_service import get_low_stock_items
from backend.services.premium_low_stock_email_handler import send_premium_low_stock_email

# Get low stock items
low_stock_df = get_low_stock_items()

# Send premium email
result = send_premium_low_stock_email(
    low_stock_df=low_stock_df,
    recipient_email="manager@company.com",
    recipient_name="Inventory Manager",
    include_attachments=True,
)

print(f"✓ Email sent! Status: {result['success']}")
```

## 📁 Files Added

| File | Purpose | Size |
|------|---------|------|
| `backend/services/premium_report_formatter.py` | Core HTML/Excel/PDF generation | 600+ lines |
| `backend/services/premium_low_stock_email_handler.py` | Email delivery & attachments | 350+ lines |
| `backend/services/premium_report_usage.py` | Usage examples & integration | 400+ lines |
| `backend/services/test_premium_reports.py` | Comprehensive test suite | 500+ lines |
| `PREMIUM_REPORT_IMPLEMENTATION.md` | Full documentation | 500+ lines |
| `PREMIUM_QUICK_START.md` | This file | Quick reference |

## 🎨 What You Get

### Email Features
- ✓ Professional HTML with project branding
- ✓ Executive summary KPI cards
- ✓ Inventory risk overview table
- ✓ AI analysis section
- ✓ Recommended actions
- ✓ Branch-wise breakdown
- ✓ Excel & PDF attachments
- ✓ Mobile responsive
- ✓ Compatible with Gmail/Outlook

### Excel Features
- ✓ Summary sheet with key metrics
- ✓ Detailed low stock items sheet
- ✓ Professional formatting
- ✓ Color-coded headers
- ✓ Proper column widths
- ✓ Alternating row colors

### PDF Features
- ✓ Professional branding
- ✓ Executive summary
- ✓ KPI cards
- ✓ Detailed inventory table
- ✓ Print-friendly layout

## 🚀 Common Use Cases

### Send to Single Manager
```python
send_premium_low_stock_email(
    low_stock_df=low_stock_df,
    recipient_email="manager@company.com",
    include_attachments=True,
)
```

### Send to Multiple Managers
```python
recipients = [
    {"email": "manager1@company.com", "name": "North Region"},
    {"email": "manager2@company.com", "name": "South Region"},
]
send_bulk_premium_emails(
    low_stock_df=low_stock_df,
    recipients=recipients,
    include_attachments=True,
)
```

### Generate Reports Only
```python
result = generate_and_save_report_files(
    low_stock_df=low_stock_df,
    output_dir=Path("./reports")
)
```

### Preview Email HTML
```python
html = generate_premium_html_email(low_stock_df)
Path("preview.html").write_text(html)
# Open in browser
```

## 🔧 Integration Options

### Option 1: Replace Existing Email
Update your low stock alert code:
```python
# Old
from backend.services.email_service import send_low_stock_alert_email
send_low_stock_alert_email(low_stock_df)

# New
from backend.services.premium_low_stock_email_handler import send_premium_low_stock_email
send_premium_low_stock_email(low_stock_df, "manager@company.com")
```

### Option 2: Keep Both Systems
Send both quick alerts and executive reports:
```python
# Quick internal alert
send_low_stock_alert_email(low_stock_df)

# Executive report
send_premium_low_stock_email(low_stock_df, "cfo@company.com")
```

### Option 3: Schedule Daily Reports
```python
from apscheduler.schedulers.background import BackgroundScheduler

def daily_report():
    low_stock_df = get_low_stock_items()
    send_bulk_premium_emails(
        low_stock_df=low_stock_df,
        recipients=[{"email": "team@company.com"}],
    )

scheduler = BackgroundScheduler()
scheduler.add_job(daily_report, 'cron', hour=8, minute=0)
scheduler.start()
```

## 📊 Example Report Output

### Email Preview
- **Header**: Professional branded banner with date and scope
- **KPI Cards**: 4 cards showing key metrics
- **Risk Table**: Full inventory details with color-coded status
- **AI Analysis**: Critical items with detailed breakdown
- **Actions**: Recommended steps with visual highlighting
- **Summary**: Branch-wise breakdown
- **Attachments**: Excel and PDF files

### Excel Report
- **Sheet 1 - Summary**: Key metrics and generation timestamp
- **Sheet 2 - Data**: Full inventory details with formatting

### PDF Report
- **Page 1**: Cover with metrics
- **Pages 2+**: Detailed inventory tables

## ✅ Testing

Run the comprehensive test suite:
```bash
python backend/services/test_premium_reports.py
```

Tests include:
- ✓ Module imports
- ✓ Dependency verification
- ✓ HTML generation
- ✓ Excel generation
- ✓ PDF generation
- ✓ Email handler
- ✓ Empty data handling
- ✓ Theme colors
- ✓ Special characters
- ✓ Large datasets

## 🎨 Customization

### Change Colors
Edit `THEME` in `premium_report_formatter.py`:
```python
THEME = {
    "primary_navy": "#YOUR_COLOR",
    "fresh_green": "#YOUR_COLOR",
    # ... other colors
}
```

### Add Custom Fields
1. Add columns to DataFrame
2. Update table headers in `generate_excel_report()`
3. Update PDF table in `generate_pdf_report()`

### Customize Email Template
Edit `_generate_header_html()` and other sections in `premium_report_formatter.py`

## 🔍 Troubleshooting

### Email Not Sending
1. Verify `.env` has EMAIL_USERNAME and EMAIL_PASSWORD
2. Enable "Less secure apps" for Gmail
3. Use app-specific passwords for Gmail
4. Check firewall allows SMTP

### PDF Generation Fails
- Install: `pip install reportlab`
- Check disk space
- Verify file permissions

### Excel Generation Fails
- Install: `pip install openpyxl`
- Check for duplicate sheet names
- Verify data types

### Email Rendering Issues
- Preview in browser: `generate_premium_html_email()` → save as `.html`
- Test in target email client
- Check CSS support (some clients strip CSS)

## 📚 Documentation

- **PREMIUM_REPORT_IMPLEMENTATION.md** - Full technical guide
- **backend/services/premium_report_usage.py** - 7 usage examples
- **backend/services/test_premium_reports.py** - Test cases with examples

## 🎯 Best Practices

1. **Use Async for Bulk Sends**: Prevents blocking
   ```python
   send_premium_low_stock_email(low_stock_df, async_send=True)
   ```

2. **Schedule During Off-Peak**: Generate reports at night
   ```python
   scheduler.add_job(daily_report, 'cron', hour=2, minute=0)
   ```

3. **Archive Old Reports**: Clean up `data/processed/attachments/` periodically

4. **Monitor Email Logs**: Check `data/processed/email_alert_log.csv`

5. **Test in Staging**: Send test emails before production

6. **Verify Rendering**: Open emails in target clients

## 📞 Support

### Check Logs
```bash
# View email logs
cat data/processed/email_alert_log.csv

# View generated files
ls -la data/processed/attachments/
```

### Test Components
```python
# Test HTML only
html = generate_premium_html_email(low_stock_df)

# Test Excel only
generate_excel_report(low_stock_df, Path("test.xlsx"))

# Test PDF only
generate_pdf_report(low_stock_df, Path("test.pdf"))
```

### Debug Email Sending
```python
result = send_premium_low_stock_email(low_stock_df, async_send=False)
print(result)  # Shows success/error details
```

## 🔄 Migration Path

### Week 1: Testing
- [ ] Deploy new files
- [ ] Run tests
- [ ] Send test emails
- [ ] Verify rendering

### Week 2-3: Pilot
- [ ] Send to select managers
- [ ] Gather feedback
- [ ] Verify attachments

### Week 4+: Production
- [ ] Update distribution lists
- [ ] Schedule recurring reports
- [ ] Archive old system if desired

## ✨ What's Next

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Configure email**: Add EMAIL_* to `.env`
3. **Run tests**: `python backend/services/test_premium_reports.py`
4. **Send test email**: Use example code above
5. **Integrate**: Update your email code
6. **Deploy**: Push to production
7. **Monitor**: Check generated reports

---

**Status**: Production Ready  
**Compatibility**: Python 3.8+, All platforms  
**Support**: See PREMIUM_REPORT_IMPLEMENTATION.md for detailed documentation
