"""A minimal client for the UE 5.8 editor's MCP server (standard library only), shared by the tools.

Start the server in the editor console with ``ModelContextProtocol.StartServer`` (or launch the
editor with ``-ExecCmds="ModelContextProtocol.StartServer"``); it serves http://127.0.0.1:8000/mcp.
Tools are called through the ``call_tool`` meta tool with a toolset and a tool name, e.g.
``Mcp().tool("editor_toolset.toolsets.asset.AssetTools", "save_assets", {...})``.
"""
from __future__ import annotations

import json
import urllib.request

URL = "http://127.0.0.1:8000/mcp"


class Mcp:
    def __init__(self, url: str = URL):
        self.url, self.sid = url, None
        self._post({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "vivid2-tools", "version": "1"}}})
        self._post({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def _post(self, body):
        headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
        if self.sid:
            headers["Mcp-Session-Id"] = self.sid
        req = urllib.request.Request(self.url, json.dumps(body).encode(), headers)
        with urllib.request.urlopen(req, timeout=900) as r:
            self.sid = r.headers.get("Mcp-Session-Id") or self.sid
            text = r.read().decode("utf-8", "replace")
            return json.loads(text) if text.strip() else None

    def call(self, name: str, args: dict):
        """Call a top-level MCP tool (e.g. ``list_toolsets``, ``describe_toolset``); the raw text."""
        r = self._post({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": name, "arguments": args}})
        if "error" in r:
            raise RuntimeError(f"{name}: {r['error']}")
        return "\n".join(c.get("text", "") for c in r["result"]["content"])

    def tool(self, toolset: str, name: str, args: dict):
        """Call one toolset tool; its return value (parsed), or RuntimeError with the server's message."""
        text = self.call("call_tool", {"toolset_name": toolset, "tool_name": name, "arguments": args})
        try:
            return json.loads(text)["returnValue"]
        except (ValueError, KeyError):
            raise RuntimeError(f"{name}: {text[:400]}") from None
