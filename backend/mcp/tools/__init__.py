"""MCP tool implementations.

Each tool is a plain function returning a JSON-serializable dict with the shape:

    {
        "tool": <name>,
        "summary": {...},          # scalar highlights
        "records": [...],          # compact rows (<= limit)
        "sources": [{"dataset", "oracle_table"}],
        "notes": "<optional caveat>",
    }

Tools are read-only, compute live from Oracle (via backend.mcp.context), and
reuse existing analytics services. They never accept or build SQL.
"""
