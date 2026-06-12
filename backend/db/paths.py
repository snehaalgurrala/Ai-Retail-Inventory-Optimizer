"""Single source of truth for dataset file locations.

Every physical path the data access layer reads from is declared here. Modules
that still own their own write paths are unaffected; this registry only governs
reads routed through the repository.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MEMORY_DIR = PROCESSED_DIR / "memory"


# Raw source tables (system of record). Maps logical name -> file name.
RAW_FILES = {
    "products": "products.csv",
    "sales": "sales.csv",
    "stores": "stores.csv",
    "suppliers": "suppliers.csv",
    "inventory": "inventory.csv",
    "transactions": "transactions.csv",
}


def _with_csv_suffix(name: str) -> str:
    """Return ``name`` with a single .csv suffix."""
    return name if name.endswith(".csv") else f"{name}.csv"


def raw_path(name: str) -> Path:
    """Resolve a raw dataset name (e.g. ``"inventory"``) to its file path."""
    filename = RAW_FILES.get(name, _with_csv_suffix(name))
    return RAW_DIR / filename


def processed_path(name: str) -> Path:
    """Resolve a processed dataset name (e.g. ``"recommendations"``) to its path.

    Supports nested names such as ``"memory/decision_memory"``.
    """
    return PROCESSED_DIR / _with_csv_suffix(name)
