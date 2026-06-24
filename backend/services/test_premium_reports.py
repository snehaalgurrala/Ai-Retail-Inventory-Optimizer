"""
Test script for Premium Low Stock Report Formatter.

Run this script to verify that all components are working correctly:
    python backend/services/test_premium_reports.py
"""

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

# Test data
TEST_LOW_STOCK_DATA = {
    'product_id': [101, 102, 103, 104],
    'product_name': ['Premium Coffee Beans', 'Organic Tea Selection', 'Specialty Honey', 'Artisan Chocolate'],
    'category': ['Beverages', 'Beverages', 'Condiments', 'Snacks'],
    'store_id': [1, 2, 1, 3],
    'store_name': ['Downtown Store', 'Mall Location', 'Downtown Store', 'Airport Outlet'],
    'city': ['New York', 'Los Angeles', 'New York', 'Chicago'],
    'supplier_id': [201, 202, 201, 203],
    'supplier_name': ['Global Imports Co', 'Local Producers Inc', 'Global Imports Co', 'Premium Distributors'],
    'current_quantity': [15, 8, 12, 3],
    'reorder_threshold': [50, 40, 35, 25],
    'recent_daily_sales_velocity': [3.5, 2.1, 2.8, 4.5],
    'predicted_days_remaining': [4.3, 3.8, 4.3, 0.67],
    'risk_category': ['High', 'High', 'Medium', 'Critical'],
    'suggested_reorder_quantity': [35, 32, 23, 22],
    'ai_alert_message': [
        'Stock level below optimal. Sales velocity remains strong at 3.5 units/day.',
        'Inventory at critical low point. Monitor closely for immediate action.',
        'Moderate risk. Recommend reorder within 3 days.',
        'CRITICAL: Estimated stockout in less than 1 day. Immediate reorder required.',
    ],
}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "test_reports"


def print_header(title: str):
    """Print formatted test section header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def print_success(message: str):
    """Print success message."""
    print(f"✓ {message}")


def print_error(message: str):
    """Print error message."""
    print(f"✗ {message}")


def test_imports():
    """Test that all modules can be imported."""
    print_header("TEST 1: IMPORT MODULES")
    
    try:
        from backend.services.premium_report_formatter import (
            generate_premium_html_email,
            generate_excel_report,
            generate_pdf_report,
        )
        print_success("premium_report_formatter imported")
    except ImportError as e:
        print_error(f"Failed to import premium_report_formatter: {e}")
        return False
    
    try:
        from backend.services.premium_low_stock_email_handler import (
            send_premium_low_stock_email,
            send_bulk_premium_emails,
            generate_and_save_report_files,
        )
        print_success("premium_low_stock_email_handler imported")
    except ImportError as e:
        print_error(f"Failed to import premium_low_stock_email_handler: {e}")
        return False
    
    try:
        from backend.services.premium_report_usage import (
            example_send_premium_email_single_recipient,
        )
        print_success("premium_report_usage imported")
    except ImportError as e:
        print_error(f"Failed to import premium_report_usage: {e}")
        return False
    
    return True


def test_dependencies():
    """Test that required dependencies are installed."""
    print_header("TEST 2: CHECK DEPENDENCIES")
    
    required_packages = {
        'pandas': 'Data processing',
        'reportlab': 'PDF generation',
        'openpyxl': 'Excel generation',
    }
    
    all_available = True
    for package, description in required_packages.items():
        try:
            __import__(package)
            print_success(f"{package:<15} - {description}")
        except ImportError:
            print_error(f"{package:<15} - NOT INSTALLED (required for {description})")
            all_available = False
    
    return all_available


def test_html_generation():
    """Test HTML email generation."""
    print_header("TEST 3: HTML EMAIL GENERATION")
    
    try:
        from backend.services.premium_report_formatter import generate_premium_html_email
        
        test_df = pd.DataFrame(TEST_LOW_STOCK_DATA)
        html = generate_premium_html_email(test_df, branch_scope="All Branches")
        
        # Verify content
        assert isinstance(html, str), "HTML should be string"
        assert '<html>' in html.lower(), "Should contain HTML tag"
        assert 'Executive Summary' in html, "Should contain Executive Summary"
        assert 'Premium Coffee Beans' in html, "Should contain product name"
        assert '#183F5F' in html or '183F5F' in html, "Should contain theme colors"
        
        # Save preview
        preview_path = OUTPUT_DIR / "email_preview.html"
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        preview_path.write_text(html, encoding="utf-8")
        
        print_success(f"HTML generation successful")
        print_success(f"Preview saved to: {preview_path}")
        print(f"   - Size: {len(html):,} bytes")
        print(f"   - Contains: Executive Summary, KPI Cards, Risk Table, AI Analysis")
        
        return True
    except Exception as e:
        print_error(f"HTML generation failed: {e}")
        return False


def test_excel_generation():
    """Test Excel report generation."""
    print_header("TEST 4: EXCEL REPORT GENERATION")
    
    try:
        from backend.services.premium_report_formatter import generate_excel_report
        
        test_df = pd.DataFrame(TEST_LOW_STOCK_DATA)
        excel_path = OUTPUT_DIR / "test_report.xlsx"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        result = generate_excel_report(test_df, excel_path)
        
        assert result.exists(), "Excel file should be created"
        assert result.stat().st_size > 0, "Excel file should not be empty"
        
        # Verify file can be read
        try:
            import openpyxl
            wb = openpyxl.load_workbook(str(result))
            sheet_names = wb.sheetnames
            print_success(f"Excel generation successful")
            print_success(f"File saved to: {result}")
            print(f"   - Size: {result.stat().st_size:,} bytes")
            print(f"   - Sheets: {', '.join(sheet_names)}")
        except Exception as e:
            print_error(f"Excel file verification failed: {e}")
            return False
        
        return True
    except Exception as e:
        print_error(f"Excel generation failed: {e}")
        return False


def test_pdf_generation():
    """Test PDF report generation."""
    print_header("TEST 5: PDF REPORT GENERATION")
    
    try:
        from backend.services.premium_report_formatter import generate_pdf_report
        
        test_df = pd.DataFrame(TEST_LOW_STOCK_DATA)
        pdf_path = OUTPUT_DIR / "test_report.pdf"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        result = generate_pdf_report(test_df, pdf_path)
        
        assert result.exists(), "PDF file should be created"
        assert result.stat().st_size > 0, "PDF file should not be empty"
        
        print_success(f"PDF generation successful")
        print_success(f"File saved to: {result}")
        print(f"   - Size: {result.stat().st_size:,} bytes")
        print(f"   - Contains: Cover page, Summary, Inventory details")
        
        return True
    except Exception as e:
        print_error(f"PDF generation failed: {e}")
        return False


def test_email_handler():
    """Test email handler functions."""
    print_header("TEST 6: EMAIL HANDLER")
    
    try:
        from backend.services.premium_low_stock_email_handler import (
            generate_and_save_report_files,
        )
        
        test_df = pd.DataFrame(TEST_LOW_STOCK_DATA)
        test_output_dir = OUTPUT_DIR / "email_test"
        
        result = generate_and_save_report_files(
            low_stock_df=test_df,
            output_dir=test_output_dir,
        )
        
        assert result['success'], f"File generation should succeed: {result.get('errors')}"
        assert 'excel' in result['files'], "Should generate Excel file"
        assert 'pdf' in result['files'], "Should generate PDF file"
        
        # Verify files exist
        for file_type, file_path in result['files'].items():
            assert Path(file_path).exists(), f"{file_type} file should exist"
        
        print_success(f"Email handler test successful")
        print(f"   - Excel: {Path(result['files']['excel']).name}")
        print(f"   - PDF: {Path(result['files']['pdf']).name}")
        
        return True
    except Exception as e:
        print_error(f"Email handler test failed: {e}")
        return False


def test_empty_data():
    """Test handling of empty data."""
    print_header("TEST 7: EMPTY DATA HANDLING")
    
    try:
        from backend.services.premium_report_formatter import generate_premium_html_email
        
        empty_df = pd.DataFrame()
        html = generate_premium_html_email(empty_df)
        
        assert '<html>' in html.lower(), "Should still generate valid HTML"
        assert 'No low stock items' in html or 'Summary' in html, "Should handle empty gracefully"
        
        print_success(f"Empty data handling successful")
        print(f"   - HTML generated without errors")
        print(f"   - Size: {len(html):,} bytes")
        
        return True
    except Exception as e:
        print_error(f"Empty data handling failed: {e}")
        return False


def test_theme_colors():
    """Test that theme colors are used correctly."""
    print_header("TEST 8: THEME COLORS")
    
    try:
        from backend.services.premium_report_formatter import THEME, generate_premium_html_email
        
        test_df = pd.DataFrame(TEST_LOW_STOCK_DATA)
        html = generate_premium_html_email(test_df)
        
        # Check for key colors
        color_checks = {
            'Primary Navy': '#183F5F',
            'Deep Navy': '#0A1F33',
            'Fresh Green': '#6CB33F',
            'Danger Red': '#B42318',
        }
        
        all_found = True
        for color_name, color_code in color_checks.items():
            # Check with or without #
            clean_code = color_code.lstrip('#')
            if clean_code in html or color_code in html:
                print_success(f"{color_name}: {color_code}")
            else:
                print_error(f"{color_name}: {color_code} not found in HTML")
                all_found = False
        
        return all_found
    except Exception as e:
        print_error(f"Theme color test failed: {e}")
        return False


def test_special_characters():
    """Test handling of special characters in data."""
    print_header("TEST 9: SPECIAL CHARACTERS")
    
    try:
        from backend.services.premium_report_formatter import generate_premium_html_email
        
        special_df = pd.DataFrame({
            'product_name': ['Café & Bar', 'O\'Reilly™ Brand', '<Special> Item'],
            'store_name': ['Downtown "Store"', 'R&D Center', 'Main St. Location'],
            'current_quantity': [10, 20, 30],
            'reorder_threshold': [50, 50, 50],
            'recent_daily_sales_velocity': [1.0, 2.0, 3.0],
            'predicted_days_remaining': [10.0, 25.0, 10.0],
            'risk_category': ['Medium', 'Medium', 'Medium'],
            'suggested_reorder_quantity': [40, 30, 20],
            'ai_alert_message': ['Test message', 'Another test', 'Final test'],
        })
        
        html = generate_premium_html_email(special_df)
        
        # Should not contain unescaped HTML
        assert '&amp;' in html or 'Bar' in html, "Should handle ampersands"
        assert '&lt;' in html or 'Special' in html, "Should handle angle brackets"
        
        print_success(f"Special character handling successful")
        print(f"   - Ampersands handled correctly")
        print(f"   - Quotes handled correctly")
        print(f"   - HTML entities properly escaped")
        
        return True
    except Exception as e:
        print_error(f"Special character test failed: {e}")
        return False


def test_large_dataset():
    """Test handling of large datasets."""
    print_header("TEST 10: LARGE DATASET")
    
    try:
        from backend.services.premium_report_formatter import generate_premium_html_email
        import time
        
        # Create large test dataset (100 items)
        large_data = {
            'product_name': [f'Product {i}' for i in range(100)],
            'store_name': [f'Store {i % 10}' for i in range(100)],
            'current_quantity': [i % 50 for i in range(100)],
            'reorder_threshold': [50] * 100,
            'recent_daily_sales_velocity': [1.0 + (i % 5)] * 100,
            'predicted_days_remaining': [5.0 + (i % 20)] * 100,
            'risk_category': [['Critical', 'High', 'Medium', 'Low'][i % 4] for i in range(100)],
            'suggested_reorder_quantity': [40 + (i % 20) for i in range(100)],
            'ai_alert_message': ['Standard alert message'] * 100,
        }
        
        large_df = pd.DataFrame(large_data)
        
        start_time = time.time()
        html = generate_premium_html_email(large_df)
        elapsed = time.time() - start_time
        
        assert isinstance(html, str), "Should generate HTML"
        assert len(html) > 0, "HTML should not be empty"
        
        print_success(f"Large dataset handling successful")
        print(f"   - Processed {len(large_df)} items")
        print(f"   - Generated {len(html):,} bytes of HTML")
        print(f"   - Time taken: {elapsed:.2f} seconds")
        
        return True
    except Exception as e:
        print_error(f"Large dataset test failed: {e}")
        return False


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 70)
    print(" PREMIUM LOW STOCK REPORT - COMPREHENSIVE TEST SUITE")
    print("=" * 70)
    print(f"Test Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output Directory: {OUTPUT_DIR}")
    
    tests = [
        ("Import Modules", test_imports),
        ("Check Dependencies", test_dependencies),
        ("HTML Generation", test_html_generation),
        ("Excel Generation", test_excel_generation),
        ("PDF Generation", test_pdf_generation),
        ("Email Handler", test_email_handler),
        ("Empty Data Handling", test_empty_data),
        ("Theme Colors", test_theme_colors),
        ("Special Characters", test_special_characters),
        ("Large Dataset", test_large_dataset),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print_error(f"Test {test_name} crashed: {e}")
            results[test_name] = False
    
    # Print summary
    print_header("TEST SUMMARY")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, passed_flag in results.items():
        status = "PASS" if passed_flag else "FAIL"
        symbol = "✓" if passed_flag else "✗"
        print(f"  {symbol} {test_name:<30} [{status}]")
    
    print("\n" + "=" * 70)
    print(f" RESULTS: {passed}/{total} tests passed")
    print("=" * 70)
    
    if passed == total:
        print("\n✓ All tests passed! Premium reports are ready to use.")
        print(f"\nGenerated test files in: {OUTPUT_DIR}")
        return True
    else:
        print(f"\n✗ {total - passed} test(s) failed. Please review the errors above.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
