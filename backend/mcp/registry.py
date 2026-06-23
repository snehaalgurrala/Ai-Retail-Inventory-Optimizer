"""The fixed, allow-listed MCP tool catalog.

This is the single source of truth for which tools exist. The FastMCP server,
the in-process executor, and the LLM tool-selection prompt all read from here.
A tool name not present in ``TOOL_CATALOG`` cannot be called by anyone.

Each tool's JSON schema is derived from its function signature, so the catalog
and the executed function can never drift apart.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Callable, get_type_hints

from backend.mcp.tools import (
    customers,
    forecasting,
    inventory,
    products,
    reference,
    supplier,
    transfer,
)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    handler: Callable[..., dict]

    @property
    def description(self) -> str:
        doc = inspect.getdoc(self.handler) or ""
        return doc.strip()

    def _hints(self) -> dict:
        """Resolved parameter types (handles `from __future__ import annotations`,
        which would otherwise leave annotations as strings)."""
        try:
            return get_type_hints(self.handler)
        except Exception:
            return {}

    def parameters_schema(self) -> dict:
        """JSON Schema for the tool's arguments, derived from the signature."""
        hints = self._hints()
        properties: dict = {}
        required: list[str] = []
        for pname, param in inspect.signature(self.handler).parameters.items():
            annotation = hints.get(pname, str)
            if annotation is int:
                json_type = "integer"
            elif annotation is float:
                json_type = "number"
            elif annotation is bool:
                json_type = "boolean"
            else:
                json_type = "string"
            properties[pname] = {"type": json_type}
            if param.default is inspect._empty:
                required.append(pname)
        schema: dict = {"type": "object", "properties": properties}
        if required:
            schema["required"] = required
        return schema

    def coerce_arguments(self, arguments: dict | None) -> dict:
        """Keep only known params and coerce them to the annotated types."""
        arguments = dict(arguments or {})
        hints = self._hints()
        clean: dict = {}
        for pname in inspect.signature(self.handler).parameters:
            if pname not in arguments:
                continue
            value = arguments[pname]
            annotation = hints.get(pname, str)
            try:
                if annotation is int:
                    value = int(value)
                elif annotation is float:
                    value = float(value)
                elif annotation is bool:
                    value = bool(value)
                else:
                    value = "" if value is None else str(value)
            except (TypeError, ValueError):
                continue  # drop un-coercible arg; handler default applies
            clean[pname] = value
        return clean


# -- The catalog. Order is the order shown to the LLM. ----------------------
_TOOLS: list[Callable[..., dict]] = [
    # Reference
    reference.list_stores,
    reference.list_products,
    reference.list_suppliers,
    reference.validate_location,
    # Inventory health
    inventory.get_inventory_health,
    inventory.get_low_stock_items,
    inventory.get_overstock_items,
    # Products (master catalogue + sales performance)
    products.get_product_master,
    products.get_products_below_reorder,
    products.get_top_products,
    products.get_bottom_products,
    products.get_product_performance,
    # Forecasting / risk
    forecasting.get_stockout_risk,
    forecasting.get_demand_forecast,
    forecasting.get_high_demand_items,
    # Supplier / procurement
    supplier.get_supplier_analysis,
    supplier.get_procurement_risk,
    # Customers & orders (real BZ_MOCK_CUSTOMER -> ORDER_HEADER -> ORDER_LINE)
    customers.get_top_customers,
    customers.get_customer_order_analysis,
    customers.get_customer_products,
    customers.get_recent_orders,
    customers.detect_abnormal_ordering,
    customers.get_order_inventory_impact,
    # Transfer
    transfer.get_transfer_opportunities,
]

TOOL_CATALOG: dict[str, ToolSpec] = {
    fn.__name__: ToolSpec(name=fn.__name__, handler=fn) for fn in _TOOLS
}


def get_tool(name: str) -> ToolSpec | None:
    """Return the spec for an allow-listed tool, or None if not allow-listed."""
    return TOOL_CATALOG.get(name)


def list_tool_specs() -> list[ToolSpec]:
    return list(TOOL_CATALOG.values())


def function_specs() -> list[dict]:
    """OpenAI/Gemini-style function specs for every allow-listed tool."""
    return [
        {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.parameters_schema(),
        }
        for spec in TOOL_CATALOG.values()
    ]


def execute(name: str, arguments: dict | None = None) -> dict:
    """Execute an allow-listed tool in-process. Raises on unknown tool."""
    spec = get_tool(name)
    if spec is None:
        raise KeyError(f"Tool '{name}' is not in the allow-listed catalog.")
    return spec.handler(**spec.coerce_arguments(arguments))
