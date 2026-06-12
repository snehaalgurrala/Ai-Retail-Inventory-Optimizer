"""Synchronous bridge to the stdio MCP server for the (sync) Streamlit app.

The MCP SDK is asyncio-based; Streamlit reruns are synchronous. This module runs
one persistent ``ClientSession`` inside a dedicated background thread with its own
event loop, and exposes blocking ``list_tools`` / ``call_tool`` methods. A single
warm session is reused across reruns (module-level singleton), so we spawn the
server subprocess once, not per question.

The server subprocess is launched with a pinned interpreter (see config) so the
multi-Python PATH on this host cannot select the wrong one.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from typing import Any

from backend.mcp import config


logger = logging.getLogger(__name__)


def _extract_payload(result: Any) -> dict:
    """Pull the tool's JSON dict out of an MCP CallToolResult."""
    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, dict):
        # FastMCP wraps a bare dict return under {"result": ...}.
        if set(structured.keys()) == {"result"} and isinstance(structured["result"], dict):
            return structured["result"]
        return structured
    for item in getattr(result, "content", []) or []:
        text = getattr(item, "text", None)
        if text:
            try:
                return json.loads(text)
            except (ValueError, TypeError):
                continue
    return {}


class MCPClient:
    """A persistent stdio MCP session driven from a background event loop."""

    def __init__(self):
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._session: Any = None
        self._ready = threading.Event()
        self._stop: asyncio.Event | None = None
        self._error: str = ""
        self._lock = threading.Lock()

    # -- lifecycle ----------------------------------------------------------
    def start(self, timeout: float = 30.0) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive() and self._session is not None:
                return
            self._ready.clear()
            self._error = ""
            self._thread = threading.Thread(
                target=self._run, name="mcp-client-loop", daemon=True
            )
            self._thread.start()
            if not self._ready.wait(timeout):
                raise RuntimeError(self._error or "MCP client did not start in time.")
            if self._session is None:
                raise RuntimeError(self._error or "MCP client failed to open a session.")

    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._main())
        except Exception as error:  # noqa: BLE001
            self._error = f"{type(error).__name__}: {error}"
            self._ready.set()
        finally:
            self._loop.close()

    async def _main(self) -> None:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        command, args = config.server_command()
        params = StdioServerParameters(command=command, args=args, env=os.environ.copy())
        self._stop = asyncio.Event()
        try:
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self._session = session
                    logger.info("MCP stdio session initialized (%s)", command)
                    self._ready.set()
                    await self._stop.wait()
        except Exception as error:  # noqa: BLE001
            self._error = f"{type(error).__name__}: {error}"
            self._session = None
            self._ready.set()

    def stop(self) -> None:
        if self._loop and self._stop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._stop.set)
        self._session = None

    @property
    def last_error(self) -> str:
        return self._error

    # -- calls (blocking) ---------------------------------------------------
    def _submit(self, coro, timeout: float):
        if self._session is None or self._loop is None:
            raise RuntimeError("MCP session is not available.")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    def list_tool_names(self, timeout: float = 30.0) -> list[str]:
        result = self._submit(self._session.list_tools(), timeout)
        return [tool.name for tool in result.tools]

    def call_tool(self, name: str, arguments: dict | None = None, timeout: float = 60.0) -> dict:
        result = self._submit(self._session.call_tool(name, arguments or {}), timeout)
        return _extract_payload(result)


# Module-level singleton: one warm session reused across Streamlit reruns.
_CLIENT: MCPClient | None = None
_CLIENT_LOCK = threading.Lock()


def get_client() -> MCPClient:
    """Return a started MCP client singleton. Raises if the server can't start."""
    global _CLIENT
    with _CLIENT_LOCK:
        if _CLIENT is None:
            _CLIENT = MCPClient()
        _CLIENT.start()
        return _CLIENT


def reset_client() -> None:
    """Tear down the singleton (tests / explicit restart)."""
    global _CLIENT
    with _CLIENT_LOCK:
        if _CLIENT is not None:
            _CLIENT.stop()
        _CLIENT = None
