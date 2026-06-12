"""FastMCP server exposing the fixed, allow-listed tool catalog over stdio.

The server registers exactly the tools in ``backend.mcp.registry.TOOL_CATALOG``
and nothing else. Each tool reads live from Oracle and returns a JSON-safe dict.
There is no SQL surface and no generic query tool: the LLM can only invoke these
named tools with typed arguments.

Run it via ``scripts/run_mcp_server.py`` (which sets up sys.path first).
"""

from __future__ import annotations

import logging

from mcp.server.fastmcp import FastMCP

from backend.mcp import registry


logger = logging.getLogger(__name__)


def build_server() -> FastMCP:
    """Create the FastMCP app with every allow-listed tool registered."""
    mcp = FastMCP(
        "bunzl-inventory",
        instructions=(
            "Answer retail inventory, sales, supplier, procurement, and "
            "branch-ordering questions strictly by calling these tools. The data "
            "comes only from Oracle. Never request or generate SQL."
        ),
    )
    for spec in registry.list_tool_specs():
        # FastMCP derives the input schema from each handler's typed signature.
        mcp.add_tool(spec.handler, name=spec.name, description=spec.description)
    logger.info("Registered %d MCP tools", len(registry.TOOL_CATALOG))
    return mcp


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    build_server().run(transport="stdio")


if __name__ == "__main__":
    main()
