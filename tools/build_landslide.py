"""Build the Landslide maps (LandslideBefore, Landslide) in the open editor, through its MCP server (UE 5.8).

Both maps are ForestSAR's site with a mountain road cut across its south-east slope;
Landslide adds a debris slide across the road. One layout,
unreal/VIVID2Env/SourceData/Landslide/layout.json, gives the terrain of each map
(``vivid2 terrain build --like ForestSAR --layout ... --stage pre|post``), the damage ground truth
(``vivid2 damage build --map Landslide``) and, through this tool, what is placed in the levels.

Steps (run some with --steps; each can be rerun):
    level   create /Game/VIVID2/Maps/LandslideBefore and /Game/VIVID2/Maps/Landslide if missing:
            copies of ForestSAR with their own markers (BP_VIVID2Map_<map>). Then import each map's
            heightmap into its landscape by hand (Landscape mode can't be driven by script in UE
            5.8): the tool prints the settings.
    check   after the imports: trace the landscape at 400 points (on its vertices) and compare with
            the map's terrain model (sim_config/maps/<map>/terrain_dtm.npz); hits above the ground
            (trees, rocks, logs) are skipped, ground below the model is an error.
    masks   each map's clear mask, SourceData/<map>/clear_mask.png (0.5 + cleared / 2, the layout's
            mask grid): no trees, undergrowth or clutter on the road and its cut and fill faces
            (wherever the ground moved), the house pads and, in Landslide, the scar and the debris;
            imported as /Game/VIVID2/Textures/T_VIVID2_<map>_ClearMask
    forest  PCG_VIVID2_<map>: a copy of PCG_VIVID2_Forest that drops every point (trees, undergrowth,
            grass, clutter) where the clear mask is over half; set on the map's PCG volume, the
            forest regenerated on the new ground, the level saved
    ground  M_VIVID2_Landscape_<map>: a copy of M_VIVID2_Landscape that lays, through one mask
            (SourceData/<map>/ground_mask.png: R road, G bare soil, B scar, A debris), bare soil on the
            cut and fill faces and house yards, asphalt on the road, and in Landslide raw rocky soil on
            the scar and dark churned soil over the debris (which buries the road); Forest Fire pack
            dirt and rock textures; set on the map's landscape, the level saved
    houses  box houses on their pads (placeholders until a building pack; TestYard's materials), the
            floor at the ground truth's base and the walls down to the lowest ground: all intact in
            LandslideBefore; in Landslide as graded (B02's roof caved in at the scar end, rubble and
            boulders against that wall; B03 and the shed B05 buried, slabs and fragments showing)
    debris  Landslide only: boulders (big ones at the toe and edges), rocks, broken trunks lying with
            the flow and uprooted stumps over the debris, a few blocks in the scar; Forest Fire pack
            meshes as tagged actors (VIVID2LandslideDebris), partly sunk into the ground; nothing
            within keep_clear_m of the layout's scatter spots, on the houses or where the cars are
    cars    the layout's vehicles (Velocity Cars Pack static blueprints; no driver or passengers, lights
            off, paint and plate set) on the ground under them, tilted to its slope: parked by the
            houses in LandslideBefore; in Landslide also carried by the slide (on its side, or sunk
            into the debris with rocks against it) and stopped either side of the blocked road

    1. vivid2 terrain build --map LandslideBefore --like ForestSAR --layout unreal/VIVID2Env/SourceData/Landslide/layout.json --stage pre
       vivid2 terrain build --map Landslide --like ForestSAR --layout unreal/VIVID2Env/SourceData/Landslide/layout.json --stage post
       vivid2 damage build --map Landslide
    2. open unreal/VIVID2Env in UE 5.8 with the MCP server on, stop Play-In-Editor
    3. python tools/build_landslide.py --steps level, import the heightmaps, then --steps check
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from _mcp import URL, Mcp

ROOT = Path(__file__).resolve().parents[1]
LAYOUT = ROOT / "unreal/VIVID2Env/SourceData/Landslide/layout.json"
MAPS = ("LandslideBefore", "Landslide")
BASE = "ForestSAR"
MAP_DIR = "/Game/VIVID2/Maps"
FOREST_SRC = "/Game/VIVID2/PCG/PCG_VIVID2_Forest"
A, SCENE = "editor_toolset.toolsets.asset.AssetTools", "editor_toolset.toolsets.scene.SceneTools"
OBJ, PCG = "editor_toolset.toolsets.object.ObjectTools", "PCGToolset.PCGToolset"
TRACE_TOP_CM = 60000.0
GROUND_TOL_M = 0.003
CANOPY_M = 1.0
ROAD_CLEAR_M = 11.0     # cleared corridor along the road: the 8 m bench and a metre and a half each side
PAD_MARGIN_M = 3.0      # yard around each house pad
MOVED_M = 0.15          # ground the edits moved by more than this is cleared (cut and fill faces)
CLEAR_SOFT_M = 1.5      # soft edge of the cleared ground
FOREST_LAYERS = (("Tree", "TreePose", "TreeSpawner", 0), ("Under", "UnderPose", "UnderSpawner", 2),
                 ("Grass", "GrassPose", "GrassSpawner", 4), ("Clutter", "ClutterPose", "ClutterSpawner", 6))


def obj(path: str):
    name = path.rsplit("/", 1)[-1]
    return {"refPath": path if "." in name else f"{path}.{name}"}


def setp(m: Mcp, target, **values) -> None:
    for k, v in values.items():
        if not m.tool(OBJ, "set_properties", {"instance": target, "values": json.dumps({k: v})}):
            raise RuntimeError(f"could not set {k} on {target['refPath']}")


def clear_tex(name: str) -> str:
    return f"/Game/VIVID2/Textures/T_VIVID2_{name}_ClearMask"


def forest_graph(name: str) -> str:
    return f"/Game/VIVID2/PCG/PCG_VIVID2_{name}"


def clear_field(name: str) -> np.ndarray:
    """The map's cleared ground in [0, 1] on the layout's mask grid ([row = east, col = north])."""
    import cv2
    from vivid2.damage import regions as rg
    from vivid2.damage.layout import _footprint
    from vivid2.terrain import TerrainModel

    lay = json.loads(LAYOUT.read_text(encoding="utf-8"))
    size, extent = int(lay["mask"]["size_px"]), float(lay["mask"]["extent_m"])
    px_m = extent / size
    soft = {"soft_m": CLEAR_SOFT_M}
    parts = [rg.region_field({"polylines_ne": [r["polyline_ne"] for r in lay["roads"]], "width_m": ROAD_CLEAR_M,
                              "edge": soft}, size, extent)]
    for b in lay["buildings"]:  # the footprint plus its yard
        fp, _ = _footprint(b, {})
        sd = rg.region_sdf({"polygon_ne": fp}, size, extent) + PAD_MARGIN_M
        t = np.clip(sd / CLEAR_SOFT_M + 0.5, 0.0, 1.0)
        parts.append((t * t * (3 - 2 * t)).astype(np.float32))
    # wherever the edits moved the ground: cut and fill faces, pads, the slide
    base = TerrainModel.load(ROOT / f"sim_config/maps/{BASE}/terrain_dtm.npz")
    here = TerrainModel.load(ROOT / f"sim_config/maps/{name}/terrain_dtm.npz")
    c = (np.arange(size) + 0.5) * px_m - extent / 2.0
    E, N = np.meshgrid(c, c, indexing="ij")  # rows = east, cols = north
    moved = np.abs(here.down_at(N.ravel(), E.ravel()) - base.down_at(N.ravel(), E.ravel())).reshape(E.shape) > MOVED_M
    k = int(round(CLEAR_SOFT_M / px_m)) * 2 + 1
    moved = cv2.dilate(moved.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    parts.append(np.clip(cv2.GaussianBlur(moved.astype(np.float32), (0, 0), CLEAR_SOFT_M / px_m / 2), 0.0, 1.0))
    if name == "Landslide":  # the slide itself
        f = rg.fields(lay)
        parts += [f["scar"], f["deposit"]]
    return np.max(parts, axis=0).astype(np.float32)


def _actors(m: Mcp, name: str):
    return m.tool(SCENE, "find_actors", {"name": name, "tag": "", "collision_channels": []})


def _open(m: Mcp, level: str) -> None:
    current = m.tool(SCENE, "get_current_level", {})
    if current != level:
        # load_level drops the open level's unsaved changes without asking (e.g. a heightmap import)
        if m.tool(A, "is_dirty", {"asset_path": current}):
            raise SystemExit(f"{current} has unsaved changes: save it (File > Save Current Level) first")
        m.tool(SCENE, "load_level", {"level_path": level})


def step_level(m: Mcp) -> None:
    for name in MAPS:
        level, marker = f"{MAP_DIR}/{name}", f"{MAP_DIR}/BP_VIVID2Map_{name}"
        if not m.tool(A, "exists", {"path": marker}):
            m.tool(A, "duplicate", {"path": f"{MAP_DIR}/BP_VIVID2Map_{BASE}", "new_path": marker})
            m.tool(A, "save_assets", {"asset_paths": [marker]})
        made = not m.tool(A, "exists", {"path": level})
        if made:
            m.tool(A, "duplicate", {"path": f"{MAP_DIR}/{BASE}", "new_path": level})
            m.tool(A, "save_assets", {"asset_paths": [level]})
        _open(m, level)
        for a in _actors(m, f"BP_VIVID2Map_{BASE}"):
            if a["refPath"].rsplit(".", 1)[-1].startswith(f"BP_VIVID2Map_{BASE}_C"):
                m.tool(SCENE, "remove_from_scene", {"actor": a})
        if not _actors(m, f"BP_VIVID2Map_{name}"):
            m.tool(SCENE, "add_to_scene_from_asset", {"asset_path": marker, "name": f"BP_VIVID2Map_{name}",
                                                      "xform": {"location": {"x": 0, "y": 0, "z": 0}}})
        m.tool(A, "save_assets", {"asset_paths": [level]})
        info = json.loads((ROOT / f"sim_config/maps/{name}/terrain.json").read_text(encoding="utf-8"))["landscape"]
        print(f"{level}: {'created from ' + BASE if made else 'exists'}; marker BP_VIVID2Map_{name}")
        print(f"  import unreal/VIVID2Env/SourceData/{name}/heightmap.png into its landscape: resolution "
              f"{info['resolution'][0]}x{info['resolution'][1]}, location {info['landscape_centre_cm']} cm, "
              f"scale {info['scale']} (unchanged from {BASE})")


def step_check(m: Mcp, maps=MAPS) -> int:
    from vivid2.terrain import TerrainModel

    bad = 0
    for name in maps:
        level = f"{MAP_DIR}/{name}"
        _open(m, level)
        t = TerrainModel.load(ROOT / f"sim_config/maps/{name}/terrain_dtm.npz")
        rng = np.random.default_rng(1)
        # half the points on the edited ground (road, pads, slide), half anywhere
        layout = json.loads((ROOT / "unreal/VIVID2Env/SourceData/Landslide/layout.json").read_text(encoding="utf-8"))
        road = np.asarray(layout["roads"][0]["polyline_ne"], float)
        pts = []
        while len(pts) < 200:
            k = rng.integers(len(road) - 1)
            p = road[k] + rng.uniform() * (road[k + 1] - road[k]) + rng.uniform(-12, 12, 2)
            pts.append(p)
        pts += list(rng.uniform(-500, 500, (200, 2)))
        # on landscape vertices (the 1 m grid), where the landscape and the terrain model hold the same
        # heightmap value; between them triangles and bilinear interpolation differ at kinks (road cuts)
        pts = [np.round(p) for p in pts]
        script = ("import json\n"
                  f"PTS = json.loads({json.dumps(json.dumps([[float(a), float(b)] for a, b in pts]))})\n"
                  "def run():\n"
                  "    out = []\n"
                  "    for n, e in PTS:\n"
                  "        r = execute_tool('editor_toolset.toolsets.scene.SceneTools.trace_world', json.dumps({\n"
                  f"            'start': {{'x': n * 100, 'y': e * 100, 'z': {TRACE_TOP_CM}}}, 'end': {{'x': n * 100, 'y': e * 100, 'z': -1000}}}}))\n"
                  "        out.append(r.get('returnValue'))\n"
                  "    return {'d': out}\n")
        r = m.tool("editor_toolset.toolsets.programmatic.ProgrammaticToolset", "execute_tool_script", {"script": script})
        dists = (json.loads(r) if isinstance(r, str) else r)["d"]
        d = np.asarray([(TRACE_TOP_CM - dist) / 100.0 + float(t.down_at(n, e))
                        for (n, e), dist in zip(pts, dists) if dist is not None])
        # above the ground: trees (over 1 m) and clutter (rocks, logs, stumps) are skipped; only ground
        # below the terrain model can be a landscape error (a wrong or misplaced import)
        match, below = int((np.abs(d) <= GROUND_TOL_M).sum()), d[d < -GROUND_TOL_M]
        clutter, trees = int(((d > GROUND_TOL_M) & (d <= CANOPY_M)).sum()), int((d > CANOPY_M).sum())
        ok = match / max(match + len(below), 1)
        worst = float(below.min()) if len(below) else 0.0
        print(f"{name}: {match} of {match + len(below)} ground hits within {GROUND_TOL_M * 1000:.0f} mm ({ok:.1%}); "
              f"{len(below)} below the terrain model (worst {worst:.3f} m); skipped {trees} trees, {clutter} clutter")
        bad += ok < 0.98 or worst < -0.05
    return bad


def step_masks(m: Mcp, maps=MAPS) -> None:
    import cv2
    from vivid2.damage import regions as rg

    lay = json.loads(LAYOUT.read_text(encoding="utf-8"))
    for name in maps:
        field = clear_field(name)
        png = ROOT / f"unreal/VIVID2Env/SourceData/{name}/clear_mask.png"
        cv2.imwrite(str(png), rg.to_png(field))
        tex = clear_tex(name)
        if m.tool(A, "exists", {"path": tex}):
            m.tool(A, "delete", {"path": tex})
        folder, asset = tex.rsplit("/", 1)
        m.tool("editor_toolset.toolsets.texture.TextureTools", "import_file",
               {"folder_path": folder, "asset_name": asset, "source_file": str(png)})
        setp(m, obj(tex), SRGB=False, CompressionSettings="TC_Grayscale", MipGenSettings="TMGS_NoMipmaps",
             NeverStream=True, AddressX="TA_Clamp", AddressY="TA_Clamp")
        m.tool(A, "save_assets", {"asset_paths": [tex]})
        ha = (field >= 0.5).sum() * (float(lay["mask"]["extent_m"]) / field.shape[0]) ** 2 / 1e4
        print(f"{name}: clear mask {png.name} ({ha:.1f} ha cleared) -> {tex}")


def step_forest(m: Mcp, maps=MAPS) -> None:
    import time

    lay = json.loads(LAYOUT.read_text(encoding="utf-8"))
    half_cm = float(lay["mask"]["extent_m"]) * 50.0
    for name in maps:
        graph = forest_graph(name)
        if m.tool(A, "exists", {"path": graph}):
            m.tool(A, "delete", {"path": graph})  # rebuilt, never patched: PCG reuses cached node results
        m.tool(A, "duplicate", {"path": FOREST_SRC, "new_path": graph})
        g, gname = obj(graph), graph.rsplit("/", 1)[-1]

        def node(n):
            return {"refPath": f"{graph}.{gname}:{n}"}

        def add(kind, n, params, title, x, y):
            m.tool(PCG, "AddNode", {"graph": g, "nativeNodeType": kind, "nodeName": n, "jsonParams": json.dumps(params),
                                    "nodeTitle": title, "nodeComment": "", "xPositionIdx": x, "yPositionIdx": y})

        def connect(a, ap, b, bp):
            m.tool(PCG, "ConnectNodePins", {"fromNode": node(a), "fromPinLabel": ap, "toNode": node(b), "toPinLabel": bp})

        add("Get Texture Data", "ClearMask", {
            "texture": obj(clear_tex(name)), "bUseAbsoluteTransform": True, "colorChannel": "Red", "filter": "Bilinear",
            "transform": {"location": {"x": 0, "y": 0, "z": 0}, "rotation": {"pitch": 0, "yaw": 0, "roll": 0},
                          "scale": {"x": half_cm, "y": half_cm, "z": 1}}},
            f"Clear Mask (SourceData/{name}/clear_mask.png)", 6, -2)
        for key, pose, spawner, row in FOREST_LAYERS:
            m.tool(PCG, "DisconnectNodePins", {"fromNode": node(pose), "fromPinLabel": "Out", "toNode": node(spawner),
                                               "toPinLabel": "In"})
            add("Attribute Maths Op", f"{key}DensityOne", {"operation": "Ceil", "inputSource1": "$Density",
                                                           "outputTarget": "$Density"},
                "Density = 1 (the mask sample multiplies it)", 6, row)
            add("Sample Texture", f"{key}ClearSample", {"densityMergeFunction": "Set"},
                "Sample Clear Mask (0.5 forest .. 1 cleared)", 7, row)
            add("Density Filter", f"{key}Kept", {"lowerBound": 0.0, "upperBound": 0.749}, "Not cleared", 8, row)
            connect(pose, "Out", f"{key}DensityOne", "In")
            connect(f"{key}DensityOne", "Out", f"{key}ClearSample", "Point")
            connect("ClearMask", "Out", f"{key}ClearSample", "BaseTexture")
            connect(f"{key}ClearSample", "Out", f"{key}Kept", "In")
            connect(f"{key}Kept", "Out", spawner, "In")
        m.tool(A, "save_assets", {"asset_paths": [graph]})

        level = f"{MAP_DIR}/{name}"
        _open(m, level)
        vols = m.tool(PCG, "ListGraphInstances", {})
        if len(vols) != 1:
            raise RuntimeError(f"expected one PCG volume in {level}, found {len(vols)}")
        vol = vols[0]["actor"]
        setp(m, {"refPath": vol["refPath"] + ".PCG Component.PCGGraphInstance"}, Graph=obj(graph)["refPath"])
        for _ in range(12):  # the first call can collide with the refresh the graph change starts
            try:
                msgs = m.tool(PCG, "ExecuteGraphInstance", {"pCGVolume": vol})
                break
            except RuntimeError:
                time.sleep(5)
        else:
            raise RuntimeError("the forest did not regenerate")
        time.sleep(15)
        m.tool(A, "save_assets", {"asset_paths": [level]})
        if m.tool(A, "is_dirty", {"asset_path": level}):
            raise RuntimeError(f"{level} did not save")
        print(f"{level}: forest {gname} regenerated ({msgs or 'no messages'}), saved")


# -- ground ----------------------------------------------------------------------------------------------
GROUND_SRC = "/Game/VIVID2/Materials/M_VIVID2_Landscape"
ROAD_SURFACE_M = 6.5   # asphalt: the 6 m road and a little shoulder
PACK_TEX = "/Game/ForestFire/Textures"
MT = "editor_toolset.toolsets.material.MaterialTools"


def ground_tex(name: str) -> str:
    return f"/Game/VIVID2/Textures/T_VIVID2_{name}_GroundMask"


def ground_material(name: str) -> str:
    return f"/Game/VIVID2/Materials/M_VIVID2_Landscape_{name}"


def ground_mask(name: str) -> np.ndarray:
    """The map's surfaces in [0, 1] on the layout's mask grid, as BGRA: R road, G bare soil (cut and fill
    faces, house yards), B scar, A debris ([row = east, col = north], as the clear mask)."""
    import cv2
    from vivid2.damage import regions as rg
    from vivid2.damage.layout import _footprint
    from vivid2.terrain import TerrainModel

    lay = json.loads(LAYOUT.read_text(encoding="utf-8"))
    size, extent = int(lay["mask"]["size_px"]), float(lay["mask"]["extent_m"])
    px_m = extent / size
    road = rg.region_field({"polylines_ne": [r["polyline_ne"] for r in lay["roads"]], "width_m": ROAD_SURFACE_M,
                            "edge": {"soft_m": 0.5}}, size, extent)
    base = TerrainModel.load(ROOT / f"sim_config/maps/{BASE}/terrain_dtm.npz")
    pre = TerrainModel.load(ROOT / "sim_config/maps/LandslideBefore/terrain_dtm.npz")
    c = (np.arange(size) + 0.5) * px_m - extent / 2.0
    E, N = np.meshgrid(c, c, indexing="ij")
    moved = (np.abs(pre.down_at(N.ravel(), E.ravel()) - base.down_at(N.ravel(), E.ravel())).reshape(E.shape)
             > MOVED_M).astype(np.float32)  # the road's and pads' cut and fill (the slide is its own layer)
    bare = [cv2.GaussianBlur(moved, (0, 0), 1.0 / px_m)]
    for b in lay["buildings"]:
        fp, _ = _footprint(b, {})
        sd = rg.region_sdf({"polygon_ne": fp}, size, extent) + PAD_MARGIN_M
        bare.append(np.clip(sd / 1.0 + 0.5, 0.0, 1.0).astype(np.float32))
    bare = np.clip(np.max(bare, axis=0), 0, 1)
    if name == "Landslide":
        f = rg.fields(lay)
        scar, debris = f["scar"], f["deposit"]
    else:
        scar = debris = np.zeros_like(road)
    to8 = lambda a: np.clip(np.round(a * 255.0), 0, 255).astype(np.uint8)
    return np.dstack([to8(scar), to8(bare), to8(road), to8(debris)])  # cv2 writes B, G, R, A


class _Mat:
    """Adds expressions to a material through MCP."""

    def __init__(self, m: Mcp, path: str):
        self.m, self.mat = m, obj(path)

    def add(self, cls: str, **props):
        e = self.m.tool(MT, "add_expression", {"material_or_function": self.mat,
                                                "expression_class": {"refPath": f"/Script/Engine.{cls}"}, "x": 0, "y": 0})
        if props:
            setp(self.m, e, **props)
        return e

    def link(self, a, a_out, b, b_in):
        self.m.tool(MT, "connect_expressions", {"from_expression": a, "from_output_name": a_out, "to_expression": b,
                                                "to_input_name": b_in})

    def op(self, cls, a, b, a_out="", b_out=""):
        e = self.add(cls)
        self.link(a, a_out, e, "A")
        self.link(b, b_out, e, "B")
        return e

    def const3(self, r, g, b):
        return self.add("MaterialExpressionConstant3Vector", Constant={"R": r, "G": g, "B": b, "A": 1.0})

    def const(self, v):
        return self.add("MaterialExpressionConstant", R=v)

    def tex(self, texture, uv, normal=False):
        e = self.add("MaterialExpressionTextureSample", Texture=obj(texture)["refPath"],
                     SamplerType="SAMPLERTYPE_Normal" if normal else ("SAMPLERTYPE_Masks" if texture.endswith("_M")
                                                                     else "SAMPLERTYPE_Color"),
                     SamplerSource="SSM_Wrap_WorldGroupSettings")
        self.link(uv, "", e, "UVs")
        return e

    def lerp(self, a, b, alpha, a_out="", b_out="", alpha_out=""):
        e = self.add("MaterialExpressionLinearInterpolate")
        self.link(a, a_out, e, "A")
        self.link(b, b_out, e, "B")
        self.link(alpha, alpha_out, e, "Alpha")
        return e


def _save_settled(m: Mcp, level: str, tries: int = 6) -> None:
    """Save the level until it stays saved: the landscape keeps updating for a while after its
    material changes and marks the level dirty again."""
    import time

    for _ in range(tries):
        m.tool(A, "save_assets", {"asset_paths": [level]})
        time.sleep(10)
        if not m.tool(A, "is_dirty", {"asset_path": level}):
            return
    raise RuntimeError(f"{level} keeps changing after saving")


def step_ground(m: Mcp, maps=MAPS) -> None:
    import cv2

    lay = json.loads(LAYOUT.read_text(encoding="utf-8"))
    half_cm = float(lay["mask"]["extent_m"]) * 50.0
    for name in maps:
        png = ROOT / f"unreal/VIVID2Env/SourceData/{name}/ground_mask.png"
        cv2.imwrite(str(png), ground_mask(name))
        tex = ground_tex(name)
        if m.tool(A, "exists", {"path": tex}):
            m.tool(A, "delete", {"path": tex})
        folder, asset = tex.rsplit("/", 1)
        m.tool("editor_toolset.toolsets.texture.TextureTools", "import_file",
               {"folder_path": folder, "asset_name": asset, "source_file": str(png)})
        setp(m, obj(tex), SRGB=False, CompressionSettings="TC_VectorDisplacementmap", MipGenSettings="TMGS_NoMipmaps",
             NeverStream=True, AddressX="TA_Clamp", AddressY="TA_Clamp")
        m.tool(A, "save_assets", {"asset_paths": [tex]})

        path = ground_material(name)
        if m.tool(A, "exists", {"path": path}):
            m.tool(A, "delete", {"path": path})
        m.tool(A, "duplicate", {"path": GROUND_SRC, "new_path": path})
        g = _Mat(m, path)
        base = m.tool(MT, "get_property_input", {"material": g.mat, "material_property": "MP_BaseColor"})
        rough = m.tool(MT, "get_property_input", {"material": g.mat, "material_property": "MP_Roughness"})
        nrm = m.tool(MT, "get_property_input", {"material": g.mat, "material_property": "MP_Normal"})
        xy = g.add("MaterialExpressionComponentMask", R=True, G=True, B=False, A=False)
        g.link(g.add("MaterialExpressionWorldPosition"), "XYZ", xy, "")
        add_half = g.add("MaterialExpressionAdd", ConstB=half_cm)
        g.link(xy, "", add_half, "A")
        muv = g.add("MaterialExpressionDivide", ConstB=2 * half_cm)
        g.link(add_half, "", muv, "A")
        mask = g.add("MaterialExpressionTextureSample", Texture=obj(tex)["refPath"], SamplerType="SAMPLERTYPE_LinearColor",
                     SamplerSource="SSM_Clamp_WorldGroupSettings")
        g.link(muv, "", mask, "UVs")

        def uv(tile_cm):
            d = g.add("MaterialExpressionDivide", ConstB=tile_cm)
            g.link(xy, "", d, "A")
            return d

        dirt1, dirt2 = f"{PACK_TEX}/Ground/T_Ground_Dirt_01", f"{PACK_TEX}/Ground/T_Ground_Dirt_02"
        rock1, rock2 = f"{PACK_TEX}/Rock/T_Rock_01", f"{PACK_TEX}/Rock/T_Rock_02"
        # bare soil: the second dirt, a touch lighter and yellower (fresh cut)
        u_bare = uv(400.0)
        bare = g.op("MaterialExpressionMultiply", g.tex(dirt2 + "_D", u_bare), g.const3(0.80, 0.68, 0.56), "RGB")
        # asphalt: the fine rock, desaturated and dark (aggregate in tar)
        u_road = uv(160.0)
        asph = g.add("MaterialExpressionDesaturation")
        g.link(g.tex(rock2 + "_D", u_road), "RGB", asph, "")
        g.link(g.const(1.0), "", asph, "Fraction")
        asphalt = g.op("MaterialExpressionMultiply", asph, g.const3(0.20, 0.20, 0.21))
        # scar: reddish raw soil broken by rock where the rock mask says so
        u_scar, u_scar_r = uv(500.0), uv(700.0)
        soil = g.op("MaterialExpressionMultiply", g.tex(dirt2 + "_D", u_scar), g.const3(0.72, 0.60, 0.50), "RGB")
        rock_mix = g.tex(rock1 + "_M", u_scar_r)
        scar = g.lerp(soil, g.tex(rock1 + "_D", u_scar_r), rock_mix, "", "RGB", "R")
        # debris: dark churned soil with rock speckles
        u_deb, u_deb_r = uv(350.0), uv(220.0)
        mud = g.op("MaterialExpressionMultiply", g.tex(dirt1 + "_D", u_deb), g.const3(0.42, 0.33, 0.25), "RGB")
        # large-scale variation (a 25 m tile of the other dirt) so the debris doesn't repeat every few metres
        macro = g.add("MaterialExpressionLinearInterpolate", ConstA=0.6, ConstB=1.25)
        g.link(g.tex(dirt2 + "_D", uv(2500.0)), "R", macro, "Alpha")
        mud = g.op("MaterialExpressionMultiply", mud, macro)
        speck = g.add("MaterialExpressionSaturate")
        g.link(g.op("MaterialExpressionSubtract", g.tex(dirt1 + "_M", u_deb_r), g.const(0.55), "R"), "", speck, "")
        debris = g.lerp(mud, g.op("MaterialExpressionMultiply", g.tex(rock2 + "_D", u_deb_r), g.const(0.8), "RGB"),
                        g.op("MaterialExpressionMultiply", speck, g.const(2.5)))
        # layered: base -> bare soil (G) -> asphalt (R) -> scar (B) -> debris (A)
        col = g.lerp(base["expression"], bare, mask, base["output_name"], "", "G")
        col = g.lerp(col, asphalt, mask, "", "", "R")
        col = g.lerp(col, scar, mask, "", "", "B")
        col = g.lerp(col, debris, mask, "", "", "A")
        m.tool(MT, "connect_to_output", {"expression": col, "output_name": "", "material_property": "MP_BaseColor"})
        r = g.lerp(rough["expression"], g.const(0.95), mask, rough["output_name"], "", "G")
        r = g.lerp(r, g.const(0.72), mask, "", "", "R")
        r = g.lerp(r, g.const(0.93), mask, "", "", "B")
        r = g.lerp(r, g.const(0.78), mask, "", "", "A")
        m.tool(MT, "connect_to_output", {"expression": r, "output_name": "", "material_property": "MP_Roughness"})
        n = g.lerp(nrm["expression"], g.tex(dirt2 + "_N", u_bare, normal=True), mask, nrm["output_name"], "RGB", "G")
        n = g.lerp(n, g.const3(0.0, 0.0, 1.0), mask, "", "", "R")
        n = g.lerp(n, g.tex(rock1 + "_N", u_scar_r, normal=True), mask, "", "RGB", "B")
        n = g.lerp(n, g.tex(dirt1 + "_N", u_deb, normal=True), mask, "", "RGB", "A")
        m.tool(MT, "connect_to_output", {"expression": n, "output_name": "", "material_property": "MP_Normal"})
        m.tool(MT, "layout_expressions", {"material_or_function": g.mat})
        m.tool(MT, "recompile", {"material_or_function": g.mat})
        m.tool(A, "save_assets", {"asset_paths": [path]})

        level = f"{MAP_DIR}/{name}"
        _open(m, level)
        land = [a for a in m.tool(SCENE, "find_actors", {"name": "Landscape", "tag": "", "collision_channels": []})
                if a["refPath"].rsplit(".", 1)[-1].startswith("Landscape_")]
        for a in land:
            setp(m, a, LandscapeMaterial=obj(path)["refPath"])
        _save_settled(m, level)
        print(f"{level}: ground {path.rsplit('/', 1)[-1]} (mask {png.name}) set, saved")


# -- houses ----------------------------------------------------------------------------------------------
HOUSE_TAG = "VIVID2LandslideHouses"
PROXY_MI = "/Game/VIVID2/TestYard/MI_VIVID2_TestYard_"  # placeholder looks (TestYard's): Wall, Roof, RoofGrey, Rubble, Dark
CUBE, CYLINDER = "/Engine/BasicShapes/Cube", "/Engine/BasicShapes/Cylinder"
ROOF_LOOK = {"B01": "Roof", "B02": "RoofGrey", "B03": "Roof", "B04": "RoofGrey", "B05": "RoofGrey"}


def house_actors(name: str, m: Mcp) -> list:
    """Box houses on their pads: intact in LandslideBefore; in Landslide as the ground truth grades them."""
    from vivid2.damage.layout import _footprint
    from vivid2.geo import points_in_polygon
    from vivid2.terrain import TerrainModel

    lay = json.loads(LAYOUT.read_text(encoding="utf-8"))
    pre = TerrainModel.load(ROOT / "sim_config/maps/LandslideBefore/terrain_dtm.npz")
    post = TerrainModel.load(ROOT / f"sim_config/maps/{name}/terrain_dtm.npz")
    grades = {}
    if name == "Landslide":
        gt = json.loads((ROOT / "sim_config/maps/Landslide/damage_gt.json").read_text(encoding="utf-8"))
        grades = {b["id"]: b for b in gt["buildings"]}
    rng = np.random.default_rng(31)
    rock_m = _mesh_path(ROCK_M[0], m)
    out = []

    def part(nm, look, c, yaw, x, y, z, sx, sy, sz, pitch=0.0, roll=0.0, mesh=CUBE):
        """A box in the house's frame: x along its heading ``yaw``, y to its right, z up (centre, metres)."""
        cy, sy_ = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
        n, e = c[0] + x * cy - y * sy_, c[1] + x * sy_ + y * cy
        out.append({"name": nm, "mesh": mesh, "material": None if look is None else f"{PROXY_MI}{look}",
                    "loc": [n * 100, e * 100, z * 100], "rot": [pitch, yaw, roll], "scale": [sx, sy, sz]})

    for b in lay["buildings"]:
        fp, h = _footprint(b, {})
        fp = np.asarray(fp)
        c, yaw = np.asarray(b["ne"], float), float(b.get("yaw_deg", 0.0))
        L, W = float(b["size_m"][0]), float(b["size_m"][1])
        lo, hi = fp.min(axis=0), fp.max(axis=0)
        grid = np.stack(np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1)), -1).reshape(-1, 2)
        inside = grid[points_in_polygon(grid, fp.tolist())]
        ground = -pre.down_at(inside[:, 0], inside[:, 1]) if len(inside) else np.array([-pre.down_at(*c)])
        floor = float(np.mean(ground))  # the ground truth's base: the mean ground over the footprint
        bottom = float(np.min(ground)) - 0.3
        grade = grades.get(b["id"], {}).get("grade", "none")
        bid, roof = b["id"], ROOF_LOOK.get(b["id"], "RoofGrey")
        top = floor + h
        cy_, sy_ = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))

        def world(x, y):
            return c[0] + x * cy_ - y * sy_, c[1] + x * sy_ + y * cy_

        def surf(x, y):
            return -float(post.down_at(*world(x, y)))

        if grade == "none":
            part(f"LS_House_{bid}_Walls", "Wall", c, yaw, 0, 0, (bottom + top - 0.3) / 2, L, W, top - 0.3 - bottom)
            part(f"LS_House_{bid}_Roof", roof, c, yaw, 0, 0, top - 0.15, L + 0.6, W + 0.6, 0.3)
            if L * W > 50:  # a rooftop water tank, as on most Taiwanese houses
                part(f"LS_House_{bid}_Tank", "Wall", c, yaw, -L * 0.25, W * 0.2, top + 0.75, 1.3, 1.3, 1.5, mesh=CYLINDER)
            continue
        if grade == "partial":
            # the end facing the slide (+x) is broken open: no roof, walls broken down, slabs fallen inside
            keep = L * 0.58
            xk = -L / 2 + keep / 2
            part(f"LS_House_{bid}_Walls", "Wall", c, yaw, xk, 0, (bottom + top - 0.3) / 2, keep, W, top - 0.3 - bottom)
            part(f"LS_House_{bid}_Roof", roof, c, yaw, xk - 0.15, 0, top - 0.15, keep + 0.3, W + 0.6, 0.3)
            part(f"LS_House_{bid}_Tank", "Wall", c, yaw, -L * 0.3, W * 0.2, top + 0.75, 1.3, 1.3, 1.5, mesh=CYLINDER)
            open_l, x0 = L - keep, -L / 2 + keep
            part(f"LS_House_{bid}_Inside", "Dark", c, yaw, x0 + open_l / 2, 0, floor + 0.05, open_l, W - 0.4, 0.1)
            for side, y in (("L", -W / 2 + 0.15), ("R", W / 2 - 0.15)):  # side walls stepping down, jagged
                for k in range(3):
                    seg = open_l / 3
                    wall_h = h * (0.75 - 0.22 * k) * rng.uniform(0.85, 1.1)
                    part(f"LS_House_{bid}_Side{side}{k}", "Wall", c, yaw, x0 + seg * (k + 0.5), y,
                         (bottom + floor + wall_h) / 2, seg, 0.3, floor + wall_h - bottom)
            part(f"LS_House_{bid}_EndStub", "Wall", c, yaw, L / 2 - 0.15, -W / 4, (bottom + floor + 1.2) / 2, 0.3, W / 2,
                 floor + 1.2 - bottom, roll=rng.uniform(-8, 8))
            for i in range(3):  # roof slabs fallen in, tilted
                part(f"LS_House_{bid}_Fallen_{i}", roof, c, yaw + rng.uniform(-15, 15), x0 + open_l * rng.uniform(0.2, 0.8),
                     rng.uniform(-W / 4, W / 4), floor + rng.uniform(0.8, 2.2), rng.uniform(2.0, 3.0), rng.uniform(1.8, 2.8), 0.3,
                     pitch=rng.uniform(-35, 35), roll=rng.uniform(-30, 30))
            for i in range(34):  # rubble and boulders piled against the open end and in it
                s = rng.uniform(0.5, 1.6)
                x = L / 2 + rng.uniform(-open_l * 0.8, 4.0)
                y = rng.uniform(-W / 2 - 1, W / 2 + 1)
                g = max(surf(x, y), floor) if x < L / 2 else surf(x, y)
                pile = max(0.0, 1.6 - abs(x - L / 2) * 0.35)  # heaped against the wall line
                if i % 3 == 0:
                    n, e = world(x, y)
                    out.append({"name": f"LS_House_{bid}_Boulder_{i:02d}", "mesh": rock_m, "material": None,
                                "loc": [n * 100, e * 100, (g + pile * 0.6 + 0.3 * s) * 100],
                                "rot": [rng.uniform(-20, 20), rng.uniform(0, 360), rng.uniform(-20, 20)],
                                "scale": [s * 0.8, s * 0.8, s * 0.7]})
                else:
                    part(f"LS_House_{bid}_Rubble_{i:02d}", "Rubble" if i % 4 else roof, c, yaw + rng.uniform(0, 360), x, y,
                         g + pile * 0.6 + 0.2 * s, s, s * rng.uniform(0.5, 1.2), s * 0.55,
                         pitch=rng.uniform(-35, 35), roll=rng.uniform(-35, 35))
            continue
        # collapsed and buried: what pokes out of the debris surface
        shed = L * W < 30
        if not shed:
            for i in range(5):  # roof slabs, broken and tilted, half out of the debris
                x, y = rng.uniform(-L / 3, L / 3), rng.uniform(-W / 3, W / 3)
                part(f"LS_House_{bid}_Slab_{i}", roof, c, yaw + rng.uniform(-40, 40), x, y, surf(x, y) + 0.5,
                     rng.uniform(3.0, 5.0), rng.uniform(2.2, 3.5), 0.3, pitch=rng.uniform(-30, 30), roll=rng.uniform(-30, 30))
            for i in range(8):  # wall stubs and big concrete fragments
                x, y = rng.uniform(-L / 2, L / 2), rng.uniform(-W / 2, W / 2)
                part(f"LS_House_{bid}_Wall_{i}", "Wall", c, yaw + rng.uniform(-25, 25), x, y, surf(x, y) + 0.8,
                     rng.uniform(2.0, 4.0), 0.3, rng.uniform(1.5, 2.6), pitch=rng.uniform(-25, 25), roll=rng.uniform(-35, 35))
            part(f"LS_House_{bid}_Tank", "Wall", c, yaw, L * 0.3, -W * 0.3, surf(L * 0.3, -W * 0.3) + 0.4, 1.3, 1.3, 1.5,
                 pitch=75.0, roll=rng.uniform(-20, 20), mesh=CYLINDER)  # the water tank, rolled off
            for i in range(45):
                x, y = rng.uniform(-L / 2 - 3, L / 2 + 3), rng.uniform(-W / 2 - 3, W / 2 + 3)
                s = rng.uniform(0.5, 1.8)
                part(f"LS_House_{bid}_Rubble_{i:02d}", "Rubble" if i % 5 else roof, c, yaw + rng.uniform(0, 360), x, y,
                     surf(x, y) + 0.2 * s, s, s * rng.uniform(0.5, 1.3), s * 0.55,
                     pitch=rng.uniform(-35, 35), roll=rng.uniform(-35, 35))
        else:
            for i in range(5):  # the shed's metal sheets, bent and half buried
                x, y = rng.uniform(-L / 2, L / 2), rng.uniform(-W / 2, W / 2)
                part(f"LS_House_{bid}_Sheet_{i}", roof, c, yaw + rng.uniform(-50, 50), x, y, surf(x, y) + 0.4,
                     rng.uniform(2.0, 3.2), rng.uniform(1.0, 1.6), 0.06, pitch=rng.uniform(-45, 45), roll=rng.uniform(-45, 45))
            for i in range(2):
                part(f"LS_House_{bid}_Post_{i}", "Dark", c, yaw, (i - 0.5) * L * 0.8, W / 2 - 0.3, surf((i - 0.5) * L * 0.8, W / 2) + 0.6,
                     0.14, 0.14, 1.4, pitch=rng.uniform(-30, 30), roll=rng.uniform(-30, 30))
    return out


_SPAWN_MAT = """
import json

def run():
    made = 0
    for a in json.loads(SPECS):
        act = execute_tool("editor_toolset.toolsets.scene.SceneTools.add_to_scene_from_asset", json.dumps({
            "asset_path": a["mesh"], "name": a["name"],
            "xform": {"location": {"x": a["loc"][0], "y": a["loc"][1], "z": a["loc"][2]},
                      "rotation": {"pitch": a["rot"][0], "yaw": a["rot"][1], "roll": a["rot"][2]},
                      "scale": {"x": a["scale"][0], "y": a["scale"][1], "z": a["scale"][2]}}}))["returnValue"]
        execute_tool("editor_toolset.toolsets.actor.ActorTools.add_tag", json.dumps({"actor": act, "tag": TAG}))
        if a.get("material"):
            comp = execute_tool("editor_toolset.toolsets.actor.ActorTools.get_root_component",
                                json.dumps({"actor": act}))["returnValue"]
            mat = a["material"] + "." + a["material"].rsplit("/", 1)[-1]
            execute_tool("editor_toolset.toolsets.object.ObjectTools.set_properties",
                         json.dumps({"instance": comp, "values": json.dumps({"OverrideMaterials": [mat]})}))
        made += 1
    return {"made": made}
"""


def step_houses(m: Mcp, maps=MAPS) -> None:
    for name in maps:
        level = f"{MAP_DIR}/{name}"
        _open(m, level)
        old = m.tool(SCENE, "find_actors", {"name": "", "tag": HOUSE_TAG, "collision_channels": []})
        for act in old:
            m.tool(SCENE, "remove_from_scene", {"actor": act})
        specs = house_actors(name, m)
        made = 0
        for i in range(0, len(specs), 60):
            script = f"SPECS = {json.dumps(json.dumps(specs[i:i + 60]))}\nTAG = {json.dumps(HOUSE_TAG)}\n" + _SPAWN_MAT
            r = m.tool("editor_toolset.toolsets.programmatic.ProgrammaticToolset", "execute_tool_script", {"script": script})
            made += (json.loads(r) if isinstance(r, str) else r)["made"]
        _save_settled(m, level)
        print(f"{level}: removed {len(old)} old house actors, placed {made}; saved")


# -- debris ----------------------------------------------------------------------------------------------
# Meshes from the Forest Fire pack (centre pivots unless noted); native size in metres.
ROCK_L, ROCK_M, ROCK_S = ("/Game/ForestFire/Meshes/Rocks/SM_Rock_Large_01", 5.2), \
    ("/Game/ForestFire/Meshes/Rocks/SM_Rock_Medium_01", 2.0), ("/Game/ForestFire/Meshes/Rocks/SM_Rock_Small_01", 1.05)
STUMP = "/Game/ForestFire/Meshes/Stumps/SM_Stump_01"   # an uprooted stump, pivot at its base, 1.44 m tall
LOG = "/Game/ForestFire/Meshes/Stumps/SM_Stump_02"     # a fallen trunk 11 m long (local Y), 1.8 m thick
DEBRIS_TAG = "VIVID2LandslideDebris"
FLOW = ((-232.0, 285.0), (-392.0, 428.0))  # the slide's axis: road crossing -> toe


def _mesh_path(name: str, m: Mcp) -> str:
    """The full path of a pack mesh by name (the pack's folders vary)."""
    found = m.tool(A, "find_assets", {"folder_path": "/Game", "name": name.rsplit("/", 1)[-1],
                                       "asset_type": {"refPath": "/Script/Engine.StaticMesh"}})
    exact = [f for f in found if f.rsplit("/", 1)[-1] == name.rsplit("/", 1)[-1]]
    if not exact:
        raise RuntimeError(f"mesh {name} not in the project (the Forest Fire pack)")
    return exact[0]


def debris_actors(m: Mcp) -> list:
    """Rocks, logs and stumps on the slide, from the layout's regions and the Landslide terrain."""
    from vivid2.damage import regions as rg
    from vivid2.damage.layout import _footprint
    from vivid2.geo import points_in_polygon
    from vivid2.terrain import TerrainModel

    lay = json.loads(LAYOUT.read_text(encoding="utf-8"))
    sc = lay["scatter"]
    rng = np.random.default_rng(int(sc["seed"]))
    ext = float(lay["mask"]["extent_m"])
    fields = rg.fields(lay)
    t = TerrainModel.load(ROOT / "sim_config/maps/Landslide/terrain_dtm.npz")
    houses = [np.asarray(_footprint(b, {})[0]) for b in lay["buildings"]]
    keep = np.asarray(sc["keep_clear_ne"], float)
    cars = np.asarray([st["ne"] for _, st in car_states("Landslide")], float).reshape(-1, 2)
    paths = {k: _mesh_path(v, m) for k, v in (("L", ROCK_L[0]), ("M", ROCK_M[0]), ("S", ROCK_S[0]),
                                               ("stump", STUMP), ("log", LOG))}
    a, b = np.asarray(FLOW[0]), np.asarray(FLOW[1])
    axis = (b - a) / np.linalg.norm(b - a)
    flow_yaw = float(np.degrees(np.arctan2(axis[1], axis[0])))  # from north towards east

    def field(name, n, e):
        return float(rg.sample(fields[name], np.array([n]), np.array([e]), ext)[0])

    def along(n, e):
        return float(np.clip(np.dot(np.array([n, e]) - a, axis) / np.linalg.norm(b - a), 0, 1))

    def ground(n, e):
        return -float(t.down_at(n, e))

    placed = []  # (n, e, radius)

    def free(n, e, r):
        if np.any(np.hypot(keep[:, 0] - n, keep[:, 1] - e) < float(sc["keep_clear_m"]) + r):
            return False
        if np.any(np.hypot(cars[:, 0] - n, cars[:, 1] - e) < CAR_CLEAR_M + r):
            return False
        for fp in houses:
            c = fp.mean(axis=0)
            if np.hypot(*(c - [n, e])) < np.ptp(fp, axis=0).max() / 2 + 1.5 + r:
                return False
        return all(np.hypot(pn - n, pe - e) > 0.85 * (pr + r) for pn, pe, pr in placed)

    region_box = {"deposit": ((-410, 238), (-215, 448)), "scar": ((-240, 224), (-176, 316))}

    def pick(region, r, weight):
        (n0, e0), (n1, e1) = region_box[region]
        for _ in range(4000):
            n, e = rng.uniform(n0, n1), rng.uniform(e0, e1)
            f = field(region, n, e)
            if f < 0.55 or rng.uniform() > weight(f, along(n, e)) or not free(n, e, r):
                continue
            placed.append((n, e, r))
            return n, e
        return None

    out = []

    def rock(kind, region, size, embed, weight, k):
        mesh, native = {"L": ROCK_L, "M": ROCK_M, "S": ROCK_S}[kind]
        s = size / native
        spot = pick(region, size / 2, weight)
        if spot is None:
            return
        n, e = spot
        h = native * s * 0.75  # the rocks are about 3/4 as tall as wide
        z = ground(n, e) + h * (0.5 - embed)
        sx, sy = s * rng.uniform(0.85, 1.15), s * rng.uniform(0.85, 1.15)
        out.append({"name": f"LS_Debris_Rock{kind}_{region}_{k:03d}", "mesh": paths[kind],
                    "loc": [n * 100, e * 100, z * 100], "rot": [rng.uniform(-20, 20), rng.uniform(0, 360), rng.uniform(-20, 20)],
                    "scale": [sx, sy, s * rng.uniform(0.7, 1.0)]})

    # the toe and the edges (levees) collect the biggest blocks; the scar keeps a few bedrock blocks
    big = lambda f, u: (0.25 + 0.75 * u) * (1.0 if f < 0.85 else 0.6)
    for k in range(90):
        rock("L", "deposit", rng.uniform(2.5, 5.5), rng.uniform(0.25, 0.4), big, k)
    for k in range(260):
        rock("M", "deposit", rng.uniform(1.0, 2.5), rng.uniform(0.2, 0.35), lambda f, u: 0.5 + 0.5 * u, k)
    for k in range(450):
        rock("S", "deposit", rng.uniform(0.4, 1.1), rng.uniform(0.15, 0.3), lambda f, u: 1.0, k)
    for k in range(16):
        rock("L", "scar", rng.uniform(2.5, 4.5), rng.uniform(0.4, 0.55), lambda f, u: 1.0, k)
    for k in range(40):
        rock("M", "scar", rng.uniform(0.8, 2.0), rng.uniform(0.3, 0.45), lambda f, u: 1.0, k)
    for k in range(60):  # broken trunks lying with the flow, on the slope
        length, dia = rng.uniform(4.0, 9.0), rng.uniform(0.5, 0.9)
        spot = pick("deposit", length / 3, lambda f, u: 0.6 + 0.4 * u)
        if spot is None:
            continue
        n, e = spot
        yaw = flow_yaw + rng.uniform(-40, 40) + (180 if rng.uniform() < 0.5 else 0)
        d = np.array([np.cos(np.radians(yaw)), np.sin(np.radians(yaw))])
        p0, p1 = np.array([n, e]) - d * length / 2, np.array([n, e]) + d * length / 2
        slope = np.degrees(np.arctan2(ground(*p1) - ground(*p0), length))
        z = (ground(*p0) + ground(*p1)) / 2 + dia * 0.25
        out.append({"name": f"LS_Debris_Log_{k:03d}", "mesh": paths["log"], "loc": [n * 100, e * 100, z * 100],
                    "rot": [0.0, yaw - 90.0, -slope], "scale": [dia / 1.8, length / 11.0, dia / 1.8]})
    for k in range(35):  # uprooted stumps on their sides
        s = rng.uniform(0.7, 1.2)
        spot = pick("deposit", 1.0 * s, lambda f, u: 0.6 + 0.4 * u)
        if spot is None:
            continue
        n, e = spot
        out.append({"name": f"LS_Debris_Stump_{k:03d}", "mesh": paths["stump"],
                    "loc": [n * 100, e * 100, (ground(n, e) + 0.55 * s) * 100],
                    "rot": [-90.0 + rng.uniform(-15, 15), rng.uniform(0, 360), rng.uniform(-30, 30)], "scale": [s, s, s]})
    return out


_SPAWN = """
import json

def run():
    made = 0
    for a in json.loads(SPECS):
        act = execute_tool("editor_toolset.toolsets.scene.SceneTools.add_to_scene_from_asset", json.dumps({
            "asset_path": a["mesh"], "name": a["name"],
            "xform": {"location": {"x": a["loc"][0], "y": a["loc"][1], "z": a["loc"][2]},
                      "rotation": {"pitch": a["rot"][0], "yaw": a["rot"][1], "roll": a["rot"][2]},
                      "scale": {"x": a["scale"][0], "y": a["scale"][1], "z": a["scale"][2]}}}))["returnValue"]
        execute_tool("editor_toolset.toolsets.actor.ActorTools.add_tag", json.dumps({"actor": act, "tag": TAG}))
        made += 1
    return {"made": made}
"""


def step_debris(m: Mcp) -> None:
    level = f"{MAP_DIR}/Landslide"
    _open(m, level)
    old = m.tool(SCENE, "find_actors", {"name": "", "tag": DEBRIS_TAG, "collision_channels": []})
    for act in old:
        m.tool(SCENE, "remove_from_scene", {"actor": act})
    specs = debris_actors(m)
    made = 0
    for i in range(0, len(specs), 60):
        script = f"SPECS = {json.dumps(json.dumps(specs[i:i + 60]))}\nTAG = {json.dumps(DEBRIS_TAG)}\n" + _SPAWN
        r = m.tool("editor_toolset.toolsets.programmatic.ProgrammaticToolset", "execute_tool_script", {"script": script})
        made += (json.loads(r) if isinstance(r, str) else r)["made"]
    m.tool(A, "save_assets", {"asset_paths": [level]})
    if m.tool(A, "is_dirty", {"asset_path": level}):
        raise RuntimeError(f"{level} did not save")
    kinds = {}
    for s in specs:
        k = s["name"].split("_")[2]
        kinds[k] = kinds.get(k, 0) + 1
    print(f"{level}: removed {len(old)} old debris actors, placed {made}: {kinds}; saved")


# -- cars ------------------------------------------------------------------------------------------------
CAR_TAG = "VIVID2LandslideCars"
ACTOR, SMESH = "editor_toolset.toolsets.actor.ActorTools", "editor_toolset.toolsets.static_mesh.StaticMeshTools"
CAR_BODY = "/Game/Velocity_Cars_Pack/Meshes/Velocity_{model}/Static_Meshes/SM_Body"
CAR_CLEAR_M = 3.2  # the debris step keeps its rocks this far from where the cars are in Landslide
PAINT = {  # linear paint colours (most cars in Taiwan are white, silver, grey or black)
    "white": [0.80, 0.80, 0.78], "silver": [0.62, 0.63, 0.65], "black": [0.015, 0.015, 0.015],
    "dark_blue": [0.015, 0.03, 0.09], "red": [0.45, 0.02, 0.01], "dark_red": [0.16, 0.012, 0.01],
}


def car_states(name: str) -> list:
    """The layout's vehicles in this map: [(vehicle, state)], state = {ne, yaw_deg, [pitch_deg, roll_deg,
    sink_m, doors, rocks]}."""
    lay = json.loads(LAYOUT.read_text(encoding="utf-8"))
    key = "pre" if name == "LandslideBefore" else "post"
    out = []
    for v in lay.get("vehicles", {}).get("items", []):
        st = v.get(key)
        if st == "same":
            st = v["pre"]
        if st:
            out.append((v, st))
    return out


def _rot_z(pitch: float, roll: float, x, y, z):
    """World height (relative to the root) of local points under an Unreal rotator's pitch and roll
    (FRotationMatrix: pitch raises +X, roll lowers +Y)."""
    sp, cp = np.sin(np.radians(pitch)), np.cos(np.radians(pitch))
    sr, cr = np.sin(np.radians(roll)), np.cos(np.radians(roll))
    return x * sp - y * sr * cp + z * cr * cp


def car_actors(name: str, m: Mcp) -> tuple:
    """(cars, rocks): each car's blueprint, transform and options; rocks piled against the cars the
    slide carried."""
    from vivid2.terrain import TerrainModel

    lay = json.loads(LAYOUT.read_text(encoding="utf-8"))
    pack = lay["vehicles"]["pack"]
    t = TerrainModel.load(ROOT / f"sim_config/maps/{name}/terrain_dtm.npz")
    rng = np.random.default_rng(41)
    rock_paths = {"M": _mesh_path(ROCK_M[0], m), "S": _mesh_path(ROCK_S[0], m)}
    cars, rocks = [], []
    for v, st in car_states(name):
        b = m.tool(SMESH, "get_bounds", {"mesh": obj(CAR_BODY.format(model=v["model"]))})
        x0, x1 = b["min"]["x"] / 100, b["max"]["x"] / 100
        hw, h = max(-b["min"]["y"], b["max"]["y"]) / 100, b["max"]["z"] / 100
        yaw = float(st["yaw_deg"])
        c, s = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
        # the plane under the wheels: its height and slope along and across the car
        xs, ys = np.meshgrid(np.linspace(x0 * 0.85, x1 * 0.85, 7), np.linspace(-hw * 0.85, hw * 0.85, 4))
        xs, ys = xs.ravel(), ys.ravel()
        n0, e0 = st["ne"]
        g = -t.down_at(n0 + xs * c - ys * s, e0 + xs * s + ys * c)
        (ga, gb, gz), *_ = np.linalg.lstsq(np.c_[xs, ys, np.ones_like(xs)], g, rcond=None)
        pitch_x, roll_x = float(st.get("pitch_deg", 0.0)), float(st.get("roll_deg", 0.0))
        corners = np.array([[x, y, z] for x in (x0, x1) for y in (-hw, hw) for z in (0.0, h)])
        lift = -float(_rot_z(pitch_x, roll_x, *corners.T).min())  # the lowest corner on the plane
        z = float(gz) + lift - float(st.get("sink_m", 0.0))
        pitch = float(np.degrees(np.arctan(ga))) + pitch_x
        roll = -float(np.degrees(np.arctan(gb))) + roll_x
        doors = st.get("doors", {})
        cars.append({"name": f"LS_Car_{v['id']}_{v['model']}", "bp": pack + v["model"],
                     "loc": [n0 * 100, e0 * 100, z * 100], "rot": [pitch, yaw, roll],
                     "props": {"lights": False, "lights_DRL": False, "driver": False, "passengers": 0,
                               "percent of cars with driver": 0, "randomize Car": False, "randomize Paint": False,
                               "randomize RegPlate Text": False, "american Reg Plates": False,
                               "color": dict(zip("rgba", PAINT[v["color"]] + [1.0])), "metallic": 0.4, "roughness": 0.2,
                               "text": v["plate"], "country Code": "",
                               **{f"door {d} Open": float(doors.get(d, 0.0)) for d in ("FL", "FR", "RL", "RR")}}})
        # rocks against the car, most on its uphill side
        up = np.degrees(np.arctan2(*(np.array([np.cos(np.radians(yaw)), np.sin(np.radians(yaw))]) * ga +
                                     np.array([-np.sin(np.radians(yaw)), np.cos(np.radians(yaw))]) * gb)))
        k = 0
        while k < int(st.get("rocks", 0)):
            ang = rng.uniform(0, 360)
            if rng.uniform() > 0.3 + 0.7 * max(0.0, np.cos(np.radians(ang - up))):
                continue
            kind = "M" if k % 3 == 0 else "S"
            size = rng.uniform(1.2, 2.2) if kind == "M" else rng.uniform(0.5, 1.1)
            native = ROCK_M[1] if kind == "M" else ROCK_S[1]
            lx, ly = np.cos(np.radians(ang - yaw)), np.sin(np.radians(ang - yaw))  # direction in the car's frame
            r = 1.0 / np.hypot(lx / (x1 - x0) * 2, ly / hw) + size * 0.3 + rng.uniform(0.0, 0.8)
            n, e = n0 + r * np.cos(np.radians(ang)), e0 + r * np.sin(np.radians(ang))
            sc = size / native
            rocks.append({"name": f"LS_Car_{v['id']}_Rock_{k:02d}", "mesh": rock_paths[kind],
                          "loc": [n * 100, e * 100, (-float(t.down_at(n, e)) + size * 0.75 * 0.25) * 100],
                          "rot": [rng.uniform(-20, 20), rng.uniform(0, 360), rng.uniform(-20, 20)],
                          "scale": [sc * rng.uniform(0.85, 1.15), sc * rng.uniform(0.85, 1.15), sc * rng.uniform(0.7, 1.0)]})
            k += 1
    return cars, rocks


def step_cars(m: Mcp, maps=MAPS) -> None:
    for name in maps:
        level = f"{MAP_DIR}/{name}"
        _open(m, level)
        old = m.tool(SCENE, "find_actors", {"name": "", "tag": CAR_TAG, "collision_channels": []})
        for act in old:
            m.tool(SCENE, "remove_from_scene", {"actor": act})
        cars, rocks = car_actors(name, m)
        for cspec in cars:
            act = m.tool(SCENE, "add_to_scene_from_asset", {"asset_path": cspec["bp"], "name": cspec["name"], "xform": {
                "location": dict(zip("xyz", cspec["loc"])), "rotation": dict(zip(("pitch", "yaw", "roll"), cspec["rot"])),
                "scale": {"x": 1, "y": 1, "z": 1}}})
            m.tool(ACTOR, "add_tag", {"actor": act, "tag": CAR_TAG})
            if not m.tool(OBJ, "set_properties", {"instance": act, "values": json.dumps(cspec["props"])}):
                raise RuntimeError(f"could not set the options of {cspec['name']}")
        if rocks:
            script = f"SPECS = {json.dumps(json.dumps(rocks))}\nTAG = {json.dumps(CAR_TAG)}\n" + _SPAWN
            m.tool("editor_toolset.toolsets.programmatic.ProgrammaticToolset", "execute_tool_script", {"script": script})
        _save_settled(m, level)
        print(f"{level}: removed {len(old)} old car actors, placed {len(cars)} cars and {len(rocks)} rocks; saved")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--steps", default="level", help="comma-separated: level, check, masks, forest, ground, debris, houses, cars")
    ap.add_argument("--maps", default=",".join(MAPS), help="which maps (default both)")
    args = ap.parse_args(argv)
    maps = [s.strip() for s in args.maps.split(",") if s.strip()]
    if any(s not in MAPS for s in maps):
        ap.error(f"--maps takes {', '.join(MAPS)}")
    try:
        m = Mcp()
    except OSError as e:
        print(f"no MCP server at {URL} ({e}): open unreal/VIVID2Env and run ModelContextProtocol.StartServer")
        return 1
    if m.tool("EditorToolset.EditorAppToolset", "IsPIERunning", {}):
        print("stop Play-In-Editor first")
        return 1
    bad = 0
    for s in [s.strip() for s in args.steps.split(",") if s.strip()]:
        if s == "level":
            step_level(m)
        elif s == "check":
            bad += step_check(m, maps)
        elif s == "masks":
            step_masks(m, maps)
        elif s == "forest":
            step_forest(m, maps)
        elif s == "debris":
            step_debris(m)
        elif s == "ground":
            step_ground(m, maps)
        elif s == "houses":
            step_houses(m, maps)
        elif s == "cars":
            step_cars(m, maps)
        else:
            ap.error(f"unknown step {s}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
