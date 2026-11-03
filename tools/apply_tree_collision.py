"""Give the Forest Fire pack's trees full collision (trunk + crown), through the editor's MCP server.

The pack ships its birches with one small capsule on the lower trunk, so a drone flying among
the trees passes through upper trunks and crowns. VIVID 2 replaces that with a 24-hull convex
decomposition: a thin trunk hull up to the crown base (about 4-6 m) and stacked hulls 2-8 m wide
for the crown, which leaves the space under the canopy and between trunks open.

The pack's assets are not in git (Fab licence), so every map author runs this once after
installing the pack:

    1. open unreal/VIVID2Env in UE 5.8 and run `ModelContextProtocol.StartServer` in the console
    2. python tools/apply_tree_collision.py            (add --dry-run to only report current collision)

Standard library only. The forest graph (PCG_VIVID2_Forest) already spawns these meshes with
BlockAll collision; regenerate and save the maps afterwards if they were generated before.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request

URL = "http://127.0.0.1:8000/mcp"
FOLIAGE = "/Game/ForestFire/Models/Foliage"
TREES = ["SM_Birch_Large_01", "SM_Birch_Large_02", "SM_Birch_Small_01"]
HULLS, HULL_VERTS, PRECISION = 24, 16, 500000


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

    def tool(self, toolset: str, name: str, args: dict):
        r = self._post({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
            "name": "call_tool", "arguments": {"toolset_name": toolset, "tool_name": name, "arguments": args}}})
        text = "\n".join(c.get("text", "") for c in r["result"]["content"])
        try:
            return json.loads(text)["returnValue"]
        except (ValueError, KeyError):
            raise RuntimeError(f"{name}: {text[:400]}") from None


def collision(mcp: Mcp, mesh: str) -> str:
    obj = "editor_toolset.toolsets.object.ObjectTools"
    ref = {"refPath": f"{FOLIAGE}/{mesh}.{mesh}"}
    body = json.loads(mcp.tool(obj, "get_properties", {"instance": ref, "properties": ["BodySetup"]}))["BodySetup"]
    geom = json.loads(mcp.tool(obj, "get_properties", {"instance": body, "properties": ["AggGeom"]}))["AggGeom"]
    parts = {k: len(v) for k, v in geom.items() if isinstance(v, list) and v}
    return ", ".join(f"{n} {k}" for k, n in parts.items()) or "none"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true", help="only report the current collision")
    args = ap.parse_args(argv)
    try:
        mcp = Mcp()
    except OSError as e:
        print(f"cannot reach the editor's MCP server at {URL} ({e}); start it with ModelContextProtocol.StartServer",
              file=sys.stderr)
        return 1
    for mesh in TREES:
        print(f"{mesh}: {collision(mcp, mesh)}")
        if args.dry_run:
            continue
        ok = mcp.tool("editor_toolset.toolsets.static_mesh.StaticMeshTools", "generate_convex_collisions",
                      {"mesh": {"refPath": f"{FOLIAGE}/{mesh}.{mesh}"}, "hull_count": HULLS,
                       "max_hull_verts": HULL_VERTS, "hull_precision": PRECISION})
        if not ok:
            print(f"  failed to generate collision for {mesh}", file=sys.stderr)
            return 1
        print(f"  -> {collision(mcp, mesh)}")
    if not args.dry_run:
        mcp.tool("editor_toolset.toolsets.asset.AssetTools", "save_assets",
                 {"asset_paths": [f"{FOLIAGE}/{m}" for m in TREES]})
        print("saved")
    return 0


if __name__ == "__main__":
    sys.exit(main())
