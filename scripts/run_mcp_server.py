#!/usr/bin/env python
"""Launch the Bunzl inventory MCP server over stdio.

This is the entrypoint the chatbot's MCP client spawns as a subprocess. It sets
up the project import path, then hands control to the FastMCP server, which
communicates over stdin/stdout — so this script must not print anything to
stdout itself.
"""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.mcp.server import main  # noqa: E402


if __name__ == "__main__":
    main()
