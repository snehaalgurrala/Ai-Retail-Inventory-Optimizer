from streamlit.testing.v1 import AppTest

pages = [
    "frontend/app.py",
    "frontend/pages/1_Inventory.py",
    "frontend/pages/2_Sales.py",
    "frontend/pages/3_Customer_Intelligence.py",
    "frontend/pages/4_Recommendations.py",
    "frontend/pages/5_Chatbot.py",
    "frontend/pages/6_Orders.py",
]
for p in pages:
    try:
        at = AppTest.from_file(p, default_timeout=120).run()
        errs = [e for e in at.exception]
        if errs:
            print(f"[FAIL] {p}")
            for e in errs:
                print("   ", str(e.value)[:300])
        else:
            print(f"[OK]   {p}  (markdown blocks: {len(at.markdown)})")
    except Exception as ex:
        print(f"[ERROR] {p} -> {type(ex).__name__}: {str(ex)[:300]}")
