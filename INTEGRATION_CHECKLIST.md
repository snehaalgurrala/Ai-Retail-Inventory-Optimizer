# Premium Report Implementation - Integration Checklist

## Pre-Integration Review
- [ ] Read `PREMIUM_QUICK_START.md` (5 min)
- [ ] Review `IMPLEMENTATION_SUMMARY.md` (10 min)
- [ ] Check `PREMIUM_REPORT_IMPLEMENTATION.md` if needed (reference)
- [ ] Understand the new files and their purpose (10 min)

## Dependencies Installation
- [ ] Run: `pip install -r requirements.txt`
- [ ] Verify openpyxl is installed: `python -c "import openpyxl"`
- [ ] Verify reportlab is installed: `python -c "import reportlab"`
- [ ] Verify pandas is installed: `python -c "import pandas"`

## Configuration Setup
### Email Settings
- [ ] Create or update `.env` file
- [ ] Add `EMAIL_FROM=inventory-alerts@retailcompany.com`
- [ ] Add `SMTP_SERVER=smtp.gmail.com` (or your SMTP server)
- [ ] Add `SMTP_PORT=587` (or appropriate port)
- [ ] Add `EMAIL_USERNAME=your-email@gmail.com`
- [ ] Add `EMAIL_PASSWORD=your-app-password` (use Gmail app password)
- [ ] Verify `.env` is not committed to version control
- [ ] Double-check all email settings are correct

### Directory Setup
- [ ] Verify `data/processed/` directory exists
- [ ] Check write permissions on `data/processed/`
- [ ] Create `data/processed/attachments/` if it doesn't exist
- [ ] Verify all directories are writable

## Testing Phase
### Unit Tests
- [ ] Run test suite: `python backend/services/test_premium_reports.py`
- [ ] Verify all 10 tests pass
- [ ] Check test output directory: `data/processed/test_reports/`
- [ ] Review generated test files:
  - [ ] `email_preview.html` - open in browser to verify rendering
  - [ ] `test_report.xlsx` - open in Excel/spreadsheet app
  - [ ] `test_report.pdf` - open in PDF viewer

### HTML Email Preview
- [ ] Generate HTML preview: `python -c "from backend.services.premium_report_formatter import generate_premium_html_email; import pandas as pd; df = pd.read_csv('data/processed/low_stock_alerts.csv'); print(generate_premium_html_email(df)[:500]...)"`
- [ ] Or use example code to generate: `Path('preview.html').write_text(html)`
- [ ] Open preview in:
  - [ ] Chrome/Firefox/Safari
  - [ ] Email client (Gmail, Outlook, Apple Mail)
  - [ ] Mobile browser
- [ ] Verify rendering quality
- [ ] Check colors appear correct
- [ ] Verify tables render properly
- [ ] Test responsive design on mobile

### Manual Integration Test
- [ ] Create test DataFrame with sample low stock data
- [ ] Run: `send_premium_low_stock_email(test_df, "your-test-email@gmail.com")`
- [ ] Monitor for errors
- [ ] Check email inbox
- [ ] Verify email received within 5 seconds
- [ ] Open email in Gmail
- [ ] Verify HTML rendering in Gmail
- [ ] Open email in Outlook (if available)
- [ ] Verify HTML rendering in Outlook
- [ ] Check attachments exist (Excel and PDF)
- [ ] Download and verify Excel file
- [ ] Download and verify PDF file
- [ ] Check email logs: `data/processed/email_alert_log.csv`
- [ ] Verify log entry was created

### Production Data Test
- [ ] Generate actual low stock items: `low_stock_df = get_low_stock_items()`
- [ ] Send test email to manager: `send_premium_low_stock_email(low_stock_df, "manager@company.com")`
- [ ] Verify email with production data looks correct
- [ ] Get manager feedback
- [ ] Note any changes needed

## Integration into Existing Code

### Identify Integration Points
- [ ] Find where `send_low_stock_alert_email()` is currently called
- [ ] List all locations: `grep -r "send_low_stock_alert_email" --include="*.py"`
- [ ] Decide integration approach:
  - [ ] Option A: Replace completely
  - [ ] Option B: Run alongside existing system
  - [ ] Option C: Selective replacement (some recipients get premium)

### Integration - Option A: Complete Replacement
- [ ] Update import in your calling code:
  ```python
  # Remove:
  from backend.services.email_service import send_low_stock_alert_email
  
  # Add:
  from backend.services.premium_low_stock_email_handler import send_premium_low_stock_email
  ```
- [ ] Replace function call:
  ```python
  # Old:
  result = send_low_stock_alert_email(low_stock_df)
  
  # New:
  result = send_premium_low_stock_email(
      low_stock_df=low_stock_df,
      recipient_email="manager@company.com",
      include_attachments=True,
  )
  ```
- [ ] Test code still works
- [ ] Run your test suite if you have one

### Integration - Option B: Run Both Systems
- [ ] Keep existing `send_low_stock_alert_email()` calls as-is
- [ ] Add new `send_premium_low_stock_email()` calls for executives
- [ ] Example:
  ```python
  # Quick internal alert
  send_low_stock_alert_email(low_stock_df)
  
  # Executive report
  send_premium_low_stock_email(low_stock_df, "cfo@company.com")
  ```
- [ ] Test both systems work together

### Integration - Option C: Selective Replacement
- [ ] Create a configuration dict of recipients:
  ```python
  recipients = {
      "internal": ("quick-alert", email_service.send_low_stock_alert_email),
      "executives": ("premium-report", premium_handler.send_premium_low_stock_email),
  }
  ```
- [ ] Route emails accordingly
- [ ] Test routing logic

## Scheduled Reports Setup (Optional)

### If Using APScheduler
- [ ] Install: `pip install apscheduler`
- [ ] Create scheduled task:
  ```python
  from apscheduler.schedulers.background import BackgroundScheduler
  scheduler = BackgroundScheduler()
  scheduler.add_job(daily_report, 'cron', hour=8, minute=0)
  scheduler.start()
  ```
- [ ] Test scheduler works
- [ ] Verify daily reports generate

### If Using Cron (Linux/Mac)
- [ ] Create Python script for daily report
- [ ] Add to crontab:
  ```bash
  0 8 * * * /path/to/python /path/to/daily_report.py
  ```
- [ ] Verify cron job executes

### If Using Windows Task Scheduler
- [ ] Create Python script for daily report
- [ ] Create Windows Task that runs script daily at 8 AM
- [ ] Verify task executes

## Email Distribution List Setup

### Create Distribution Groups (if needed)
- [ ] Identify all recipients who should get reports
- [ ] Create groups:
  - [ ] Inventory Managers
  - [ ] Regional Managers
  - [ ] Executive Leadership
  - [ ] Operations Team
- [ ] Collect email addresses for each group

### Update Code with Recipients
- [ ] Add recipient lists to your configuration or .env:
  ```python
  PREMIUM_REPORT_RECIPIENTS = {
      "inventory": ["inventory@company.com", "manager@company.com"],
      "executives": ["cfo@company.com", "coo@company.com"],
      "operations": ["ops@company.com"],
  }
  ```
- [ ] Or update in code directly
- [ ] Test sending to multiple recipients

## File Generation Setup

### Verify Output Directories
- [ ] Check `data/processed/attachments/` is created
- [ ] Verify write permissions
- [ ] Set up log rotation if needed (optional)
- [ ] Document file cleanup policy:
  - [ ] Delete files after X days
  - [ ] Archive files annually
  - [ ] Or keep indefinitely

### File Archival (Optional)
- [ ] Create archival script if needed
- [ ] Set up automated cleanup:
  ```python
  import shutil
  import os
  from datetime import datetime, timedelta
  
  # Delete reports older than 30 days
  cutoff = datetime.now() - timedelta(days=30)
  for f in os.listdir("data/processed/attachments"):
      if os.path.getmtime(f) < cutoff.timestamp():
          os.remove(f)
  ```

## Customization (Optional)

### Brand Customization
- [ ] Edit colors in `premium_report_formatter.py` THEME if desired
- [ ] Update logo/footer text if needed
- [ ] Test changes in preview email
- [ ] Regenerate reports to verify changes

### Add Custom Fields
- [ ] Identify additional fields to display
- [ ] Update DataFrame columns before sending
- [ ] Update HTML generation if needed
- [ ] Update Excel/PDF generation
- [ ] Test all formats

### Email Template Changes
- [ ] Edit `_generate_header_html()` if needed
- [ ] Edit `_generate_*_section_html()` functions
- [ ] Test changes with preview
- [ ] Send test emails

## Quality Assurance

### Functional Testing
- [ ] Email sends successfully ✓
- [ ] Email arrives within reasonable time ✓
- [ ] HTML renders correctly in Gmail ✓
- [ ] HTML renders correctly in Outlook ✓
- [ ] HTML renders correctly on mobile ✓
- [ ] Attachments download successfully ✓
- [ ] Excel file opens correctly ✓
- [ ] PDF file opens correctly ✓
- [ ] Data in files is accurate ✓
- [ ] No data errors or truncation ✓

### Edge Cases
- [ ] Empty data (no low stock items) ✓
- [ ] Single item (minimal data) ✓
- [ ] Large dataset (100+ items) ✓
- [ ] Special characters in product names ✓
- [ ] Very long product names ✓
- [ ] Very long store names ✓
- [ ] Missing/null data fields ✓

### Performance
- [ ] HTML generation is fast (< 200ms) ✓
- [ ] Excel generation completes (< 1s) ✓
- [ ] PDF generation completes (< 2s) ✓
- [ ] Email delivery is reliable ✓
- [ ] No timeouts or hangs ✓

### Monitoring
- [ ] Check email logs regularly ✓
- [ ] Monitor for failed deliveries ✓
- [ ] Track attachment generation ✓
- [ ] Alert on errors ✓

## Deployment to Production

### Pre-Deployment Review
- [ ] All tests pass ✓
- [ ] QA testing complete ✓
- [ ] Code reviewed ✓
- [ ] Configuration verified ✓
- [ ] Backup of current system ✓
- [ ] Rollback plan documented ✓

### Deployment
- [ ] Create feature branch
- [ ] Commit changes to version control
- [ ] Push to staging environment
- [ ] Test in staging
- [ ] Get approval for production
- [ ] Deploy to production
- [ ] Monitor for issues

### Post-Deployment
- [ ] Monitor email logs for 24 hours
- [ ] Check for delivery failures
- [ ] Verify reports look correct
- [ ] Get user feedback
- [ ] Document any issues
- [ ] Keep backup of current system for 30 days

## Training & Documentation

### Team Training
- [ ] Document new email process for team
- [ ] Show examples of new report format
- [ ] Demonstrate how to access/download attachments
- [ ] Explain any new features
- [ ] Answer questions

### Update Documentation
- [ ] Update internal wiki/docs
- [ ] Add links to QUICK_START guide
- [ ] Add troubleshooting section
- [ ] Document support contact
- [ ] Archive old email documentation if applicable

### Support
- [ ] Create support playbook for common issues
- [ ] Document troubleshooting steps
- [ ] Create FAQ
- [ ] Set up monitoring alerts

## Maintenance Plan

### Regular Tasks
- [ ] Weekly: Check email logs for errors
- [ ] Weekly: Verify reports are generating
- [ ] Monthly: Verify attachment files are generating
- [ ] Monthly: Check disk usage for reports
- [ ] Quarterly: Backup email configuration

### Ongoing Improvements
- [ ] Gather user feedback
- [ ] Identify improvements needed
- [ ] Plan customizations
- [ ] Test improvements
- [ ] Deploy updates

## Rollback Plan (If Needed)

### If Something Goes Wrong
- [ ] Stop new email system
- [ ] Revert code changes
- [ ] Update configuration
- [ ] Restore email routing to old system
- [ ] Test old system works
- [ ] Communicate with team
- [ ] Debug issue
- [ ] Plan fix

### Rollback Commands
```bash
# Restore from git
git revert <commit-hash>

# Stop new system
# Update email calling code back to old function

# Verify old system works
python -c "from backend.services.email_service import send_low_stock_alert_email"
```

## Success Criteria Met

- [ ] All new files in place and working
- [ ] Dependencies installed successfully
- [ ] Email configuration complete
- [ ] Tests pass successfully
- [ ] Integration code updated and tested
- [ ] Production emails sending successfully
- [ ] Team trained and ready
- [ ] Monitoring in place
- [ ] Rollback plan documented
- [ ] Documentation complete

## Sign-Off

- [ ] Implementation Lead: _________________ Date: ________
- [ ] QA Lead: _________________ Date: ________
- [ ] Operations Lead: _________________ Date: ________
- [ ] Project Manager: _________________ Date: ________

---

**Notes & Issues Found During Implementation:**

_____________________________________________________________________

_____________________________________________________________________

_____________________________________________________________________

**Estimated Time to Complete Entire Checklist: 2-4 hours**

- Pre-integration review: 15 min
- Dependencies & configuration: 15 min
- Testing: 30 min
- Integration: 30-60 min
- Customization (if needed): 30-60 min
- QA & deployment: 30 min
- Training: 30 min

**Total: 2.5-4 hours**

---

**Document Created**: June 1, 2026  
**Version**: 1.0  
**Status**: Ready for Implementation
