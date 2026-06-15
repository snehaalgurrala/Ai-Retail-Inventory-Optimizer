# Archived pages

Streamlit auto-discovers every `*.py` file directly inside `frontend/pages/` and
turns it into a sidebar navigation entry. Files in this folder are intentionally
kept **out** of `pages/` so they stay in the codebase but are **not shown in the
UI navigation**.

- `6_Orders.py` — Orders page. Removed from the sidebar as a UI/navigation change.
  The page code is preserved here unchanged. All Orders backend code, services,
  Oracle tables, APIs, and order-related analytics remain fully intact and in use
  elsewhere (e.g. `backend/services/order_service.py`).

To restore a page to the navigation, move its file back into `frontend/pages/`.
The directory depth here matches `frontend/pages/`, so each script's
`Path(__file__).resolve().parents[2]` PROJECT_ROOT calculation keeps working.
