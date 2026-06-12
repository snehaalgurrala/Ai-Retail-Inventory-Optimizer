"""Configuration for the MCP server, client, and chatbot orchestrator.

All values are read from the environment at call time so the runtime can change
them without re-importing. Defaults are production-safe.
"""

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def chatbot_engine() -> str:
    """Which chatbot engine to use: 'mcp' (default) or 'legacy' (old RAG path)."""
    return (os.getenv("CHATBOT_ENGINE", "mcp") or "mcp").strip().lower()


def mcp_transport() -> str:
    """MCP transport for the server: 'stdio' (default) or 'http' (future)."""
    return (os.getenv("MCP_TRANSPORT", "stdio") or "stdio").strip().lower()


def mcp_server_python() -> str:
    """Interpreter used to launch the stdio MCP server subprocess.

    Pinned to the *current* interpreter by default. This host has several Python
    installs on PATH (3.10 / 3.11 / 3.14); a bare ``python`` spawn could pick the
    wrong one and silently lack the Oracle driver, so we never rely on PATH.
    """
    return os.getenv("MCP_SERVER_PYTHON", sys.executable)


def server_command() -> tuple[str, list[str]]:
    """Return (executable, args) that launch the stdio MCP server."""
    script = str(PROJECT_ROOT / "scripts" / "run_mcp_server.py")
    return mcp_server_python(), [script]


def max_tool_iterations() -> int:
    """Maximum number of tool calls the LLM may chain for one user question."""
    return int(os.getenv("MCP_MAX_TOOL_ITERATIONS", "3"))


def default_record_limit() -> int:
    """Default number of records a tool returns when the caller does not specify."""
    return int(os.getenv("MCP_DEFAULT_LIMIT", "10"))


def max_record_limit() -> int:
    """Hard cap on records any tool may return (guards token bloat)."""
    return int(os.getenv("MCP_MAX_LIMIT", "25"))


def context_ttl_seconds() -> float:
    """How long a loaded Oracle frame bundle stays warm within a request burst.

    Short by design: a multi-tool turn reads Oracle once, but the next user
    question re-reads fresh data. Avoids the stale-cache class of bug.
    """
    return float(os.getenv("MCP_CONTEXT_TTL_SECONDS", "30"))


def use_in_process_fallback() -> bool:
    """If True, the orchestrator may execute tools in-process when the stdio
    MCP session is unavailable. Tools are identical either way (same registry),
    so answers stay Oracle-grounded. Default True for resilience."""
    return (os.getenv("MCP_INPROCESS_FALLBACK", "true") or "true").strip().lower() == "true"
