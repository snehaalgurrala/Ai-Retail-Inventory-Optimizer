"""Data access layer for the AI Retail Inventory Optimizer.

This package is the single I/O boundary for reading datasets. Application code
should call ``backend.db.repository`` instead of reading CSV files directly, so
that the physical data source can later change (e.g. to Oracle) without touching
business logic, agents, pages, reports, or the chatbot.

Backend selection is controlled by the ``DATA_BACKEND`` environment variable.
Only ``csv`` is implemented today.
"""

from backend.db import repository

__all__ = ["repository"]
