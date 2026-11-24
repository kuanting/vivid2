"""Build the Avalanche map in the open editor, through its MCP server (UE 5.8).

A slab avalanche in the Raichosawa valley below Tsurugi-gozen, Tateyama (Northern Alps, Japan), on its
own terrain (Copernicus GLO-30 at 36.5860 N, 137.6102 E). One layout,
unreal/VIVID2Env/SourceData/Avalanche/layout.json, gives the terrain (``vivid2 terrain build --map
Avalanche --lat 36.5860 --lon 137.6102 --layout ... --stage post``), the damage ground truth
(``vivid2 damage build --map Avalanche``) and, through this tool, what is placed in the level.

Steps (run some with --steps; each can be rerun):
    level     create /Game/VIVID2/Maps/Avalanche if missing: a copy of ForestSAR without its forest,
              with its own marker (BP_VIVID2Map_Avalanche), the landscape's height datum and scale set
              to the new terrain's, and the sun at the site (SunSky latitude, longitude, UTC+9). Then
              import the heightmap into the landscape by hand (Landscape mode can't be driven by script
              in UE 5.8): the tool prints the settings.
    check     after the import: trace the landscape at 400 points (on its vertices; half on the
              avalanche) and compare with the terrain model (sim_config/maps/Avalanche/terrain_dtm.npz)
    textures  the Megascans snow surfaces (Fab downloads, not committed: --fab-dir, default
              unreal/fab) at 2048 px, imported to /Game/Megascans/AvalancheSnow (outside /Game/VIVID2,
              so not committed either)
    ground    M_VIVID2_Landscape_Avalanche through two masks (SourceData/Avalanche/ground_mask.png: R
              the slab's bed, G the track, B the debris, A rock; ground_mask2.png: R melted-out ground,
              G dirt in the debris, B the party's skin track): windswept snow varying over tens of metres,
              an icy bed where the slab broke away, faint flow streaks down the track, rough grey snow
              over the debris with specks of dirt, rock on ground steeper than about 48 deg, dry grass and
              soil on wind-scoured knolls; set on the landscape, level saved
    clues     the ski gear (tools/ski_gear_blender.py's FBX in SourceData/Clues) as static meshes in
              /Game/VIVID2/Clues and one blueprint per coloured clue (VIVID2_Clue_<Item>_<Colour>), its
              slots filled with the decoys' colour materials
    debris    about 1,900 snow blocks over the debris, bigger towards the toe and the edges (slow: one
              MCP call per actor, about 20 minutes)
    scatter   rock fragments and broken branches in the debris; rock outcrops, rock bands on steep
              ridges and boulder fields, with patchy snow on their tops; dark dwarf pine mats on the
              melted-out ground; and the layout's scatter.features beside the scenarios' people (about
              1,300 actors). The snow blocks come first in one seeded stream, so they stay put when the
              rest changes. Meshes made from Forest Fire pack meshes (blocks, snowy rocks, dwarf pine)
              go to /Game/Megascans/AvalancheSnow too: pack content, not committed, remade here.

    1. vivid2 terrain build --map Avalanche --lat 36.5860 --lon 137.6102 \\
           --layout unreal/VIVID2Env/SourceData/Avalanche/layout.json --stage post
       vivid2 damage build --map Avalanche
    2. open unreal/VIVID2Env in UE 5.8 with the MCP server on, stop Play-In-Editor
    3. python tools/build_avalanche.py --steps level, import the heightmap, then --steps check
    4. --steps textures,ground,clues,debris,scatter
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from _mcp import URL, Mcp

ROOT = Path(__file__).resolve().parents[1]
NAME = "Avalanche"
LAYOUT = ROOT / f"unreal/VIVID2Env/SourceData/{NAME}/layout.json"
SOURCE = ROOT / f"unreal/VIVID2Env/SourceData/{NAME}"
BASE = "ForestSAR"
MAP_DIR = "/Game/VIVID2/Maps"
LEVEL = f"{MAP_DIR}/{NAME}"
A, SCENE = "editor_toolset.toolsets.asset.AssetTools", "editor_toolset.toolsets.scene.SceneTools"
OBJ, ACTOR = "editor_toolset.toolsets.object.ObjectTools", "editor_toolset.toolsets.actor.ActorTools"
MT, TEX = "editor_toolset.toolsets.material.MaterialTools", "editor_toolset.toolsets.texture.TextureTools"
SITE = {"Latitude": 36.586, "Longitude": 137.6102, "TimeZone": 9, "Month": 4, "Day": 25}
TRACE_TOP_CM = 60000.0
GROUND_TOL_M = 0.003


def obj(path: str):
    name = path.rsplit("/", 1)[-1]
    return {"refPath": path if "." in name else f"{path}.{name}"}


def setp(m: Mcp, target, **values) -> None:
    for k, v in values.items():
        if not m.tool(OBJ, "set_properties", {"instance": target, "values": json.dumps({k: v})}):
            raise RuntimeError(f"could not set {k} on {target['refPath']}")


def getp(m: Mcp, target, *names):
    r = m.tool(OBJ, "get_properties", {"instance": target, "properties": list(names)})
    return json.loads(r) if isinstance(r, str) else r


def layout() -> dict:
    return json.loads(LAYOUT.read_text(encoding="utf-8"))


def terrain_info() -> dict:
    return json.loads((ROOT / f"sim_config/maps/{NAME}/terrain.json").read_text(encoding="utf-8"))["landscape"]


def _actors(m: Mcp, name: str = ""):
    return m.tool(SCENE, "find_actors", {"name": name, "tag": "", "collision_channels": []})


def _named(m: Mcp, prefix: str):
    return [a for a in _actors(m) if a["refPath"].rsplit(".", 1)[-1].startswith(prefix)]


def _open(m: Mcp, level: str = LEVEL) -> None:
    current = m.tool(SCENE, "get_current_level", {})
    if current != level:
        # load_level drops the open level's unsaved changes without asking (e.g. a heightmap import)
        if m.tool(A, "is_dirty", {"asset_path": current}):
            raise SystemExit(f"{current} has unsaved changes: save it (File > Save Current Level) first")
        m.tool(SCENE, "load_level", {"level_path": level})


def _save_settled(m: Mcp, level: str = LEVEL, tries: int = 6) -> None:
    """Save the level until it stays saved: the landscape keeps updating for a while after its
    material changes and marks the level dirty again."""
    for _ in range(tries):
        m.tool(A, "save_assets", {"asset_paths": [level]})
        time.sleep(10)
        if not m.tool(A, "is_dirty", {"asset_path": level}):
            return
    raise RuntimeError(f"{level} keeps changing after saving")


# -- level ---------------------------------------------------------------------------------------------

def step_level(m: Mcp) -> None:
    marker = f"{MAP_DIR}/BP_VIVID2Map_{NAME}"
    if not m.tool(A, "exists", {"path": marker}):
        m.tool(A, "duplicate", {"path": f"{MAP_DIR}/BP_VIVID2Map_{BASE}", "new_path": marker})
        m.tool(A, "save_assets", {"asset_paths": [marker]})
    made = not m.tool(A, "exists", {"path": LEVEL})
    if made:
        m.tool(A, "duplicate", {"path": f"{MAP_DIR}/{BASE}", "new_path": LEVEL})
        m.tool(A, "save_assets", {"asset_paths": [LEVEL]})
    _open(m)
    for a in _named(m, f"BP_VIVID2Map_{BASE}_C"):
        m.tool(SCENE, "remove_from_scene", {"actor": a})
    if not _actors(m, f"BP_VIVID2Map_{NAME}"):
        m.tool(SCENE, "add_to_scene_from_asset", {"asset_path": marker, "name": f"BP_VIVID2Map_{NAME}",
                                                  "xform": {"location": {"x": 0, "y": 0, "z": 0}}})
    for a in _named(m, "PCG_Forest"):  # ForestSAR's forest (its trees go with the PCG actor)
        m.tool(SCENE, "remove_from_scene", {"actor": a})
    for sun in _named(m, "SunSky"):
        setp(m, sun, **SITE)
    info = terrain_info()
    land = _named(m, "Landscape_")
    if len(land) != 1:
        raise SystemExit(f"expected one landscape in {LEVEL}, found {len(land)}")
    xf = m.tool(ACTOR, "get_actor_transform", {"actor": land[0]})
    loc, scale = xf["location"], xf["scale"]
    want_z, want_sz = float(info["landscape_centre_cm"][2]), float(info["scale"][2])
    if abs(loc["z"] - want_z) > 0.5 or abs(scale["z"] - want_sz) > 1e-6:
        xf["location"]["z"], xf["scale"]["z"] = want_z, want_sz
        m.tool(ACTOR, "set_actor_transform", {"actor": land[0], "xform": xf})
    m.tool(A, "save_assets", {"asset_paths": [LEVEL]})
    xf = m.tool(ACTOR, "get_actor_transform", {"actor": land[0]})
    print(f"{LEVEL}: {'created from ' + BASE if made else 'exists'}; marker BP_VIVID2Map_{NAME}; forest removed; "
          f"sun at {SITE['Latitude']}, {SITE['Longitude']}, UTC+{SITE['TimeZone']}")
    print(f"  landscape location {xf['location']}, scale {xf['scale']}")
    print(f"  now import unreal/VIVID2Env/SourceData/{NAME}/heightmap.png into it (Landscape mode > Manage > "
          f"Import, into the existing landscape): resolution {info['resolution'][0]}x{info['resolution'][1]}, "
          f"keep its location and scale (centre {info['landscape_centre_cm']} cm, scale {info['scale']}), "
          f"then save the level")


# -- check ---------------------------------------------------------------------------------------------

def step_check(m: Mcp) -> int:
    from vivid2.terrain import TerrainModel

    _open(m)
    t = TerrainModel.load(ROOT / f"sim_config/maps/{NAME}/terrain_dtm.npz")
    lay = layout()
    rng = np.random.default_rng(1)
    pts = []
    polys = [np.asarray(r["polygon_ne"], float) for r in lay["regions"].values()]
    while len(pts) < 200:  # half on the avalanche (where the edits are), half anywhere
        p = polys[rng.integers(len(polys))]
        pts.append(rng.uniform(p.min(axis=0), p.max(axis=0)))
    pts += list(rng.uniform(-500, 500, (200, 2)))
    pts = [np.round(p) for p in pts]  # landscape vertices (1 m grid)
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
    match, below, above = int((np.abs(d) <= GROUND_TOL_M).sum()), d[d < -GROUND_TOL_M], d[d > GROUND_TOL_M]
    ok = match / max(match + len(below), 1)
    worst = float(below.min()) if len(below) else 0.0
    print(f"{NAME}: {match} of {match + len(below)} ground hits within {GROUND_TOL_M * 1000:.0f} mm ({ok:.1%}); "
          f"{len(below)} below the terrain model (worst {worst:.3f} m); {len(above)} above it (rocks, blocks, "
          f"shrubs) skipped; {len(pts) - len(d)} missed")
    return int(ok < 0.98 or worst < -0.05)


# -- capture (for checking) ----------------------------------------------------------------------------

def capture(m: Mcp, path: Path, n: float, e: float, z_m: float, pitch: float = -90.0, yaw: float = 0.0) -> Path:
    """The level viewport from (n, e) at Unreal Z ``z_m`` (pitch -90: straight down, north up), as a PNG."""
    import base64

    r = m.tool("EditorToolset.EditorAppToolset", "CaptureViewport", {
        "captureTransform": {"location": {"x": n * 100, "y": e * 100, "z": z_m * 100},
                             "rotation": {"pitch": pitch, "yaw": yaw, "roll": 0}, "scale": {"x": 1, "y": 1, "z": 1}},
        "annotations": {"gridSpacing": 1, "gridExtent": 1, "gridHeight": -1e7, "maxLabelDistance": 1,
                        "classFilter": [], "maxLabels": 0}, "bShowUI": False})
    path.write_bytes(base64.b64decode(r["image"]["data"]))
    return path


# -- textures ------------------------------------------------------------------------------------------

SNOW_DIR = "/Game/Megascans/AvalancheSnow"
SNOW = {  # surface -> the Fab download's name stem (unreal/fab/<stem>_8k.zip)
    "WindsweptSnow": "fresh_windswept_snow", "SnowCliff": "snow_cliff", "TrampledSnow": "trampled_snow",
    "SnowyRoad": "snowy_road", "FreshSnow": "fresh_snow"}
SNOW_PX = 2048  # the cook caps textures at 2048 (Megascans downloads are 8K)


def snow_tex(surface: str, kind: str) -> str:
    return f"{SNOW_DIR}/T_{surface}_{kind}"


def step_textures(m: Mcp, fab_dir: Path) -> None:
    """Megascans BaseColor and Normal maps at 2048 px into SNOW_DIR (Fab content: never committed)."""
    import io
    import zipfile

    import cv2

    cache = fab_dir / "2k"
    cache.mkdir(exist_ok=True)
    for surface, stem in SNOW.items():
        zips = sorted(fab_dir.glob(f"{stem}_*.zip"))
        if not zips:
            raise SystemExit(f"no {stem}_*.zip in {fab_dir}: download '{surface}' from Fab (Megascans) first")
        zf = zipfile.ZipFile(zips[0])
        for kind in ("BaseColor", "Normal"):
            member = next(n for n in zf.namelist() if n.endswith(f"_{kind}.jpg"))
            png = cache / f"T_{surface}_{kind}.png"
            if not png.exists():
                img = cv2.imdecode(np.frombuffer(zf.read(member), np.uint8), cv2.IMREAD_COLOR)
                cv2.imwrite(str(png), cv2.resize(img, (SNOW_PX, SNOW_PX), interpolation=cv2.INTER_AREA))
            tex = snow_tex(surface, kind)
            if not m.tool(A, "exists", {"path": tex}):
                folder, asset = tex.rsplit("/", 1)
                m.tool(TEX, "import_file", {"folder_path": folder, "asset_name": asset, "source_file": str(png)})
            if kind == "Normal":  # Megascans normal maps are OpenGL (green up)
                setp(m, obj(tex), CompressionSettings="TC_Normalmap", SRGB=False, bFlipGreenChannel=True)
            m.tool(A, "save_assets", {"asset_paths": [tex]})
        print(f"{surface}: {zips[0].name} -> {snow_tex(surface, 'BaseColor')}, _Normal")


# -- ground --------------------------------------------------------------------------------------------

GROUND_MAT = f"/Game/VIVID2/Materials/M_VIVID2_Landscape_{NAME}"
GROUND_TEX = f"/Game/VIVID2/Textures/T_VIVID2_{NAME}_GroundMask"
GROUND_TEX2 = f"/Game/VIVID2/Textures/T_VIVID2_{NAME}_GroundMask2"
PACK_DIRT = "/Game/ForestFire/Textures/Ground/T_Ground_Dirt_0"
ROCK_SLOPE_DEG = (44.0, 52.0)  # snow gives way to rock between these slopes
PACK_ROCK = "/Game/ForestFire/Textures/Rock/T_Rock_01"


def ground_mask() -> np.ndarray:
    """The map's surfaces in [0, 1] on the layout's mask grid ([row = east, col = north]) as BGRA for cv2:
    R the slab's bed, G the track, B the debris, A bare rock (the steepest ground, not the crown)."""
    import cv2
    from vivid2.damage import regions as rg
    from vivid2.terrain import TerrainModel

    lay = layout()
    size, extent = int(lay["mask"]["size_px"]), float(lay["mask"]["extent_m"])
    px_m = extent / size
    f = rg.fields(lay)
    t = TerrainModel.load(ROOT / f"sim_config/maps/{NAME}/terrain_dtm.npz")
    c = (np.arange(size) + 0.5) * px_m - extent / 2.0
    E, N = np.meshgrid(c, c, indexing="ij")
    z = -t.down_at(N.ravel(), E.ravel()).reshape(E.shape)
    z = cv2.GaussianBlur(z.astype(np.float32), (0, 0), 1.5 / px_m)  # slope over a few metres, not per vertex
    ge, gn = np.gradient(z, px_m)
    slope = np.degrees(np.arctan(np.hypot(ge, gn)))
    lo, hi = ROCK_SLOPE_DEG
    rock = np.clip((slope - lo) / (hi - lo), 0, 1)
    k = int(round(6.0 / px_m)) * 2 + 1  # the crown and the slab's flanks are snow walls, not rock
    near_slab = cv2.dilate((f["release"] > 0.05).astype(np.uint8), np.ones((k, k), np.uint8)).astype(np.float32)
    rock = rock * (1 - near_slab)
    to8 = lambda a: np.clip(np.round(a * 255.0), 0, 255).astype(np.uint8)
    return np.dstack([to8(f["deposit"]), to8(f["track"]), to8(f["release"]), to8(rock)])


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

    def mul(self, a, b, a_out="", b_out=""):
        return self.op("MaterialExpressionMultiply", a, b, a_out, b_out)

    def const3(self, r, g, b):
        return self.add("MaterialExpressionConstant3Vector", Constant={"R": r, "G": g, "B": b, "A": 1.0})

    def const(self, v):
        return self.add("MaterialExpressionConstant", R=v)

    def tex(self, texture, uv, sampler="SAMPLERTYPE_Color", source="SSM_Wrap_WorldGroupSettings"):
        e = self.add("MaterialExpressionTextureSample", Texture=obj(texture)["refPath"], SamplerType=sampler,
                     SamplerSource=source)
        self.link(uv, "", e, "UVs")
        return e

    def lerp(self, a, b, alpha, a_out="", b_out="", alpha_out=""):
        e = self.add("MaterialExpressionLinearInterpolate")
        self.link(a, a_out, e, "A")
        self.link(b, b_out, e, "B")
        self.link(alpha, alpha_out, e, "Alpha")
        return e

    def out(self, e, prop, out_name=""):
        self.m.tool(MT, "connect_to_output", {"expression": e, "output_name": out_name, "material_property": prop})


def step_ground(m: Mcp) -> None:
    import cv2

    lay = layout()
    half_cm = float(lay["mask"]["extent_m"]) * 50.0
    png, png2 = SOURCE / "ground_mask.png", SOURCE / "ground_mask2.png"
    mask, mask2 = ground_mask(), ground_mask2()
    cv2.imwrite(str(png), mask)
    cv2.imwrite(str(png2), mask2)
    if m.tool(A, "exists", {"path": GROUND_MAT}):  # the material goes first: it references the masks
        m.tool(A, "delete", {"path": GROUND_MAT})
    for tex, src in ((GROUND_TEX, png), (GROUND_TEX2, png2)):
        if m.tool(A, "exists", {"path": tex}):
            m.tool(A, "delete", {"path": tex})
        folder, asset = tex.rsplit("/", 1)
        m.tool(TEX, "import_file", {"folder_path": folder, "asset_name": asset, "source_file": str(src)})
        setp(m, obj(tex), SRGB=False, CompressionSettings="TC_VectorDisplacementmap", MipGenSettings="TMGS_NoMipmaps",
             NeverStream=True, AddressX="TA_Clamp", AddressY="TA_Clamp")
        m.tool(A, "save_assets", {"asset_paths": [tex]})

    if m.tool(A, "exists", {"path": GROUND_MAT}):
        m.tool(A, "delete", {"path": GROUND_MAT})
    folder, asset = GROUND_MAT.rsplit("/", 1)
    m.tool(MT, "create_material", {"folder_path": folder, "asset_name": asset})
    g = _Mat(m, GROUND_MAT)
    xy = g.add("MaterialExpressionComponentMask", R=True, G=True, B=False, A=False)
    g.link(g.add("MaterialExpressionWorldPosition"), "XYZ", xy, "")
    add_half = g.add("MaterialExpressionAdd", ConstB=half_cm)
    g.link(xy, "", add_half, "A")
    muv = g.add("MaterialExpressionDivide", ConstB=2 * half_cm)
    g.link(add_half, "", muv, "A")
    msk = g.tex(GROUND_TEX, muv, "SAMPLERTYPE_LinearColor", "SSM_Clamp_WorldGroupSettings")
    msk2 = g.tex(GROUND_TEX2, muv, "SAMPLERTYPE_LinearColor", "SSM_Clamp_WorldGroupSettings")

    def uv(tile_cm):
        d = g.add("MaterialExpressionDivide", ConstB=tile_cm)
        g.link(xy, "", d, "A")
        return d

    def snow(surface, tile_cm):
        u = uv(tile_cm)
        return g.tex(snow_tex(surface, "BaseColor"), u), g.tex(snow_tex(surface, "Normal"), u, "SAMPLERTYPE_Normal")

    def macro(tile_cm, lo, hi):
        """Broad brightness variation (one channel of a snow texture at a large scale), lo..hi."""
        e = g.add("MaterialExpressionLinearInterpolate", ConstA=lo, ConstB=hi)
        g.link(g.tex(snow_tex("WindsweptSnow", "BaseColor"), uv(tile_cm)), "R", e, "Alpha")
        return e

    # undisturbed snow: windswept, varying over tens of metres
    base_c, base_n = snow("WindsweptSnow", 300.0)
    base = g.mul(g.mul(base_c, macro(3100.0, 0.80, 0.93), "RGB"), macro(870.0, 0.96, 1.02))
    # the slab's bed: older, icier snow, a touch blue
    bed_c, bed_n = snow("SnowCliff", 250.0)
    bed = g.mul(bed_c, g.const3(0.92, 0.95, 1.0), "RGB")
    # the track: the base snow with faint flow streaks (the snowy road's ruts, run along the flow, east-west)
    streak = g.add("MaterialExpressionLinearInterpolate", ConstA=1.0)
    g.link(g.mul(g.tex(snow_tex("SnowyRoad", "BaseColor"), uv(1400.0)), g.const(1.3), "R"), "", streak, "B")
    g.link(g.const(0.22), "", streak, "Alpha")
    track = g.mul(base, streak)
    # the debris: trampled snow, broken and greyer, varying in patches
    deb_c, deb_n = snow("TrampledSnow", 400.0)
    debris = g.mul(g.mul(deb_c, g.const3(1.08, 1.08, 1.1), "RGB"), macro(2300.0, 0.85, 1.0))
    lumps = g.tex(snow_tex("TrampledSnow", "Normal"), uv(1600.0), "SAMPLERTYPE_Normal")  # metre-scale lumps
    deb_n = g.lerp(deb_n, lumps, g.const(0.5), "RGB", "RGB")
    # rock on the steepest ground
    rock_c = g.mul(g.tex(PACK_ROCK + "_D", uv(600.0)), g.const3(0.55, 0.55, 0.57), "RGB")
    rock_n = g.tex(PACK_ROCK + "_N", uv(600.0), "SAMPLERTYPE_Normal")

    # dirty patches in the debris (soil and rock the avalanche scraped up)
    dirty = g.mul(g.tex(PACK_DIRT + "1_D", uv(500.0)), g.const3(0.55, 0.47, 0.40), "RGB")
    debris = g.lerp(debris, dirty, g.mul(msk2, g.const(0.3), "G"))
    # melted-out ground: dry grass and soil
    u_bare = uv(450.0)
    bare = g.mul(g.tex(PACK_DIRT + "2_D", u_bare), g.const3(1.05, 0.88, 0.62), "RGB")
    bare = g.mul(bare, macro(1700.0, 0.8, 1.15))
    # the skin track: a shallow groove, a shade darker
    ski = g.mul(base, g.const(0.84))

    col = g.lerp(base, ski, msk2, "", "", "B")
    col = g.lerp(col, track, msk, "", "", "G")
    col = g.lerp(col, bed, msk, "", "", "R")
    col = g.lerp(col, debris, msk, "", "", "B")
    col = g.lerp(col, bare, msk2, "", "", "R")
    col = g.lerp(col, rock_c, msk, "", "", "A")
    g.out(col, "MP_BaseColor")
    r = g.lerp(g.const(0.62), g.const(0.62), msk, "", "", "G")
    r = g.lerp(r, g.const(0.42), msk, "", "", "R")
    r = g.lerp(r, g.const(0.82), msk, "", "", "B")
    r = g.lerp(r, g.const(0.92), msk2, "", "", "R")
    r = g.lerp(r, g.const(0.85), msk, "", "", "A")
    g.out(r, "MP_Roughness")
    n = g.lerp(base_n, bed_n, msk, "RGB", "RGB", "R")
    n = g.lerp(n, deb_n, msk, "", "", "B")
    n = g.lerp(n, g.tex(PACK_DIRT + "2_N", u_bare, "SAMPLERTYPE_Normal"), msk2, "", "RGB", "R")
    n = g.lerp(n, rock_n, msk, "", "RGB", "A")
    g.out(n, "MP_Normal")
    m.tool(MT, "layout_expressions", {"material_or_function": g.mat})
    m.tool(MT, "recompile", {"material_or_function": g.mat})
    m.tool(A, "save_assets", {"asset_paths": [GROUND_MAT]})

    _open(m)
    for a in _named(m, "Landscape_"):
        setp(m, a, LandscapeMaterial=obj(GROUND_MAT)["refPath"])
    _save_settled(m)
    px = (float(lay["mask"]["extent_m"]) / mask.shape[0]) ** 2 / 1e4
    print(f"{LEVEL}: ground {GROUND_MAT.rsplit('/', 1)[-1]} set (slab bed {(mask[..., 2] >= 128).sum() * px:.1f} ha, "
          f"track {(mask[..., 1] >= 128).sum() * px:.1f} ha, debris {(mask[..., 0] >= 128).sum() * px:.1f} ha "
          f"({(mask2[..., 1] >= 128).sum() * px:.2f} ha dirty), rock {(mask[..., 3] >= 128).sum() * px:.1f} ha, "
          f"melted out {(mask2[..., 2] >= 128).sum() * px:.1f} ha, ski track {(mask2[..., 0] >= 128).sum() * px:.2f} ha), "
          "saved")


# -- clues ---------------------------------------------------------------------------------------------

CLUE_DIR = "/Game/VIVID2/Clues"
DECOY_MI = "/Game/VIVID2/Targets/Materials/MI_VIVID2_Decoy_"
CLUES = {  # blueprint -> (mesh, {slot: decoy colour})
    "VIVID2_Clue_Skis_Orange": ("Skis", {"Topsheet": "Orange", "Base": "Black", "Binding": "Steel"}),
    "VIVID2_Clue_Skis_Yellow": ("Skis", {"Topsheet": "Yellow", "Base": "Black", "Binding": "Black"}),
    "VIVID2_Clue_Snowboard_Blue": ("Snowboard", {"Topsheet": "Blue", "Base": "Black", "Binding": "Black"}),
    "VIVID2_Clue_Backpack_Red": ("Backpack", {"Pack": "Red", "Trim": "Black", "Straps": "Black"}),
    "VIVID2_Clue_Backpack_Orange": ("Backpack", {"Pack": "Orange", "Trim": "Navy", "Straps": "Black"}),
}
COLLISION = {"collisionEnabled": "QueryAndPhysics", "collisionProfileName": "BlockAll", "objectType": "ECC_WorldStatic"}
SM = "editor_toolset.toolsets.static_mesh.StaticMeshTools"
BP = "editor_toolset.toolsets.blueprint.BlueprintTools"


def step_clues(m: Mcp) -> None:
    for item in sorted({mesh for mesh, _ in CLUES.values()}):
        mesh = f"{CLUE_DIR}/SM_VIVID2_Clue_{item}"
        if m.tool(A, "exists", {"path": mesh}):
            m.tool(A, "delete", {"path": mesh})
        src = ROOT / f"unreal/VIVID2Env/SourceData/Clues/VIVID2_Clue_{item}.fbx"
        m.tool(SM, "import_file", {"folder_path": CLUE_DIR, "asset_name": f"SM_VIVID2_Clue_{item}",
                                   "source_file": str(src), "import_materials": False, "combine_meshes": True})
        m.tool(SM, "generate_convex_collisions", {"mesh": obj(mesh), "hull_count": 4, "max_hull_verts": 16})
        m.tool(A, "save_assets", {"asset_paths": [mesh]})
        b = m.tool(SM, "get_bounds", {"mesh": obj(mesh)})
        print(f"{mesh}: from {src.name}, slots {[s.get('slotName', s) if isinstance(s, dict) else s for s in m.tool(SM, 'get_material_slots', {'mesh': obj(mesh)})]}, bounds {b}")
    for name, (item, colours) in CLUES.items():
        path = f"{CLUE_DIR}/{name}"
        mesh = f"{CLUE_DIR}/SM_VIVID2_Clue_{item}"
        comp = {"refPath": f"{path}.{name}_C:Mesh_GEN_VARIABLE"}
        bp = obj(path)
        if not m.tool(A, "exists", {"path": path}):
            m.tool(BP, "create", {"folder_path": CLUE_DIR, "asset_name": name, "asset_type": {"refPath": "/Script/Engine.Actor"}})
            comp = m.tool(ACTOR, "add_component", {"owner": bp, "name": "Mesh",
                                                   "component_type": {"refPath": "/Script/Engine.StaticMeshComponent"}})
        slots = m.tool(SM, "get_material_slots", {"mesh": obj(mesh)})
        slot_names = [s.get("slotName") or s.get("name") if isinstance(s, dict) else s for s in slots]
        overrides = [obj(DECOY_MI + colours[sn])["refPath"] for sn in slot_names]
        setp(m, comp, OverrideMaterials=[])
        setp(m, comp, StaticMesh=obj(mesh)["refPath"])
        setp(m, comp, OverrideMaterials=overrides)
        bi = getp(m, comp, "bodyInstance")["bodyInstance"]
        bi.update(COLLISION)
        setp(m, comp, BodyInstance=bi)
        m.tool(BP, "compile_blueprint", {"blueprint": bp})
        m.tool(A, "save_assets", {"asset_paths": [path]})
        print(f"{path}: {item}, {', '.join(f'{k} {v}' for k, v in colours.items())}")


# -- debris, rocks and dwarf pine ----------------------------------------------------------------------

BLOCK_MAT = "/Game/VIVID2/Materials/M_VIVID2_SnowBlock"
ROCK_MAT = "/Game/VIVID2/Materials/M_VIVID2_SnowyRock"
PINE_MI = {"leaves": "/Game/VIVID2/Materials/Avalanche/MI_VIVID2_DwarfPine_Leaves",
           "bark": "/Game/VIVID2/Materials/Avalanche/MI_VIVID2_DwarfPine_Bark"}
# Meshes derived from the Forest Fire pack (its rocks wearing snow, its small birch as dwarf pine): pack
# content, so they live outside /Game/VIVID2 and are not committed; this tool remakes them.
BLOCK_DIR = "/Game/Megascans/AvalancheSnow"
ROCKS = {"L": ("SM_Rock_Large_01", 5.1), "M": ("SM_Rock_Medium_01", 1.95), "S": ("SM_Rock_Small_01", 1.1)}
BIRCH = ("SM_Birch_Small_01", 4.4, 10.0)  # crown width and height, m
LOG = ("SM_Stump_02", 11.0, 1.8)          # a fallen trunk 11 m long (local Y), 1.8 m thick
DEBRIS_TAG, LITTER_TAG = "VIVID2AvalancheDebris", "VIVID2AvalancheLitter"
ROCK_TAG, PINE_TAG, FEATURE_TAG = "VIVID2AvalancheRocks", "VIVID2AvalanchePine", "VIVID2AvalancheFeatures"
SCATTER_TAGS = (LITTER_TAG, ROCK_TAG, PINE_TAG, FEATURE_TAG)
FLOW = ((-10.0, 160.0), (40.0, -390.0))  # the avalanche's axis: crown -> toe (n, e)


def _mesh_path(m: Mcp, name: str) -> str:
    """The full path of a pack mesh by name (the pack's folders vary)."""
    found = m.tool(A, "find_assets", {"folder_path": "/Game", "name": name,
                                       "asset_type": {"refPath": "/Script/Engine.StaticMesh"}})
    exact = [f for f in found if f.rsplit("/", 1)[-1] == name]
    if not exact:
        raise RuntimeError(f"mesh {name} not in the project (the Forest Fire pack)")
    return exact[0]


def _slot_names(m: Mcp, mesh: str) -> list:
    return [s.get("slotName") or s.get("name") if isinstance(s, dict) else s
            for s in m.tool(SM, "get_material_slots", {"mesh": obj(mesh)})]


def _derived(m: Mcp, src: str, path: str, materials) -> str:
    """``path``: a copy of the pack mesh ``src`` with its slots set to ``materials`` (one, or one per slot)."""
    if not m.tool(A, "exists", {"path": path}):
        m.tool(A, "duplicate", {"path": _mesh_path(m, src), "new_path": path})
    for k, slot in enumerate(_slot_names(m, path)):
        mat = materials if isinstance(materials, str) else materials[min(k, len(materials) - 1)]
        m.tool(SM, "set_material", {"mesh": obj(path), "slot_name": slot, "material": obj(mat)})
    m.tool(A, "save_assets", {"asset_paths": [path]})
    return path


def scene_materials(m: Mcp) -> None:
    """M_VIVID2_SnowBlock (broken snow), M_VIVID2_SnowyRock (rock with snow lying on its upper faces) and the
    dwarf pine's material instances (the pack's birch, tinted dark green and dark brown)."""
    if not m.tool(A, "exists", {"path": BLOCK_MAT}):
        folder, asset = BLOCK_MAT.rsplit("/", 1)
        m.tool(MT, "create_material", {"folder_path": folder, "asset_name": asset})
        g = _Mat(m, BLOCK_MAT)
        uv = g.add("MaterialExpressionTextureCoordinate", UTiling=1.5, VTiling=1.5)
        g.out(g.mul(g.tex(snow_tex("WindsweptSnow", "BaseColor"), uv), g.const3(0.86, 0.87, 0.9), "RGB"), "MP_BaseColor")
        g.out(g.tex(snow_tex("TrampledSnow", "Normal"), uv, "SAMPLERTYPE_Normal"), "MP_Normal", "RGB")
        g.out(g.const(0.75), "MP_Roughness")
        m.tool(MT, "layout_expressions", {"material_or_function": g.mat})
        m.tool(MT, "recompile", {"material_or_function": g.mat})
        m.tool(A, "save_assets", {"asset_paths": [BLOCK_MAT]})
    if not m.tool(A, "exists", {"path": ROCK_MAT}):
        folder, asset = ROCK_MAT.rsplit("/", 1)
        m.tool(MT, "create_material", {"folder_path": folder, "asset_name": asset})
        g = _Mat(m, ROCK_MAT)
        uv = g.add("MaterialExpressionTextureCoordinate", UTiling=1.0, VTiling=1.0)
        uv_snow = g.add("MaterialExpressionTextureCoordinate", UTiling=2.5, VTiling=2.5)
        rock = g.mul(g.tex(PACK_ROCK + "_D", uv), g.const3(0.62, 0.61, 0.6), "RGB")
        snow_c = g.tex(snow_tex("WindsweptSnow", "BaseColor"), uv_snow)
        # snow where the face points up: world normal z from 0.5 (none) to 0.75 (all), broken up by the snow's texture
        up = g.add("MaterialExpressionComponentMask", R=False, G=False, B=True, A=False)
        g.link(g.add("MaterialExpressionVertexNormalWS"), "", up, "")
        lumpy = g.op("MaterialExpressionAdd", up, g.tex(snow_tex("TrampledSnow", "BaseColor"), uv_snow), "", "R")
        cover = g.add("MaterialExpressionSaturate")
        g.link(g.mul(g.op("MaterialExpressionSubtract", lumpy, g.const(1.55)), g.const(3.0)), "", cover, "")
        g.out(g.lerp(rock, snow_c, cover, "", "RGB"), "MP_BaseColor")
        g.out(g.lerp(g.tex(PACK_ROCK + "_N", uv, "SAMPLERTYPE_Normal"),
                     g.tex(snow_tex("WindsweptSnow", "Normal"), uv_snow, "SAMPLERTYPE_Normal"), cover, "RGB", "RGB"),
              "MP_Normal")
        g.out(g.lerp(g.const(0.85), g.const(0.7), cover), "MP_Roughness")
        m.tool(MT, "layout_expressions", {"material_or_function": g.mat})
        m.tool(MT, "recompile", {"material_or_function": g.mat})
        m.tool(A, "save_assets", {"asset_paths": [ROCK_MAT]})
    MI = "editor_toolset.toolsets.material_instance.MaterialInstanceTools"
    pack = "/Game/ForestFire/Materials/MatInstances"
    for part, (parent, tint, desat) in {"leaves": ("MI_Birch_Small_01", (0.16, 0.22, 0.12), 0.35),
                                        "bark": ("MI_Bark_Birch_Small_01", (0.28, 0.22, 0.17), 0.2)}.items():
        path = PINE_MI[part]
        folder, name = path.rsplit("/", 1)
        inst = obj(path) if m.tool(A, "exists", {"path": path}) else \
            m.tool(MI, "create", {"folder_path": folder, "asset_name": name, "parent": obj(f"{pack}/{parent}")})
        m.tool(MI, "set_vector_parameter", {"instance": inst, "name": "Base Tint",
                                            "value": {"r": tint[0], "g": tint[1], "b": tint[2], "a": 1}})
        m.tool(MI, "set_scalar_parameter", {"instance": inst, "name": "Base Desaturation", "value": desat})
        m.tool(A, "save_assets", {"asset_paths": [path]})


def scene_meshes(m: Mcp) -> dict:
    scene_materials(m)
    out = {f"block{k}": _derived(m, rock, f"{BLOCK_DIR}/SM_SnowBlock_{k}", BLOCK_MAT) for k, (rock, _) in ROCKS.items()}
    out.update({f"rock{k}": _derived(m, rock, f"{BLOCK_DIR}/SM_SnowyRock_{k}", ROCK_MAT) for k, (rock, _) in ROCKS.items()})
    # the birch's slots: bark, then leaves
    out["pine"] = _derived(m, BIRCH[0], f"{BLOCK_DIR}/SM_DwarfPine", [PINE_MI["bark"], PINE_MI["leaves"]])
    out["log"] = _mesh_path(m, LOG[0])
    return out


def ground_mask2() -> np.ndarray:
    """A second mask on the same grid, as BGRA for cv2: R bare ground where the snow has melted out (wind-scoured
    knolls and ridges off the avalanche path), G dirty patches in the debris, B the party's ski track."""
    import cv2
    from vivid2.damage import regions as rg
    from vivid2.terrain import TerrainModel

    lay = layout()
    size, extent = int(lay["mask"]["size_px"]), float(lay["mask"]["extent_m"])
    px_m = extent / size
    f = rg.fields(lay)
    t = TerrainModel.load(ROOT / f"sim_config/maps/{NAME}/terrain_dtm.npz")
    c = (np.arange(size) + 0.5) * px_m - extent / 2.0
    E, N = np.meshgrid(c, c, indexing="ij")
    z = -t.down_at(N.ravel(), E.ravel()).reshape(E.shape).astype(np.float32)
    zs = cv2.GaussianBlur(z, (0, 0), 1.5 / px_m)
    ge, gn = np.gradient(zs, px_m)
    slope = np.degrees(np.arctan(np.hypot(ge, gn)))
    convex = zs - cv2.GaussianBlur(zs, (0, 0), 12.0 / px_m)
    path = np.max([f["release"], f["track"], f["deposit"]], axis=0)
    k = int(round(12.0 / px_m)) * 2 + 1
    off_path = 1.0 - cv2.dilate((path > 0.05).astype(np.uint8), np.ones((k, k), np.uint8)).astype(np.float32)
    rng = np.random.default_rng(int(lay["scatter"]["seed"]) + 1)

    def noise(sigma_m):
        n = cv2.GaussianBlur(rng.standard_normal((size, size)).astype(np.float32), (0, 0), sigma_m / px_m)
        return n / n.std()

    score = (convex / 0.6 + 0.8 * noise(9.0) + 0.35 * noise(2.0) - np.clip((slope - 30.0) / 8.0, 0, None)
             - np.clip((6.0 - slope) / 6.0, 0, None))  # knolls and ridges, ragged at the edges
    rim = np.clip((extent / 2 - np.maximum(np.abs(E), np.abs(N)) - 30.0) / 10.0, 0, 1)  # the blur's edge artefacts
    bare = np.clip((score - 1.6) / 0.1, 0, 1) * off_path * rim
    dirt = np.clip((noise(0.7) - 1.4) / 0.4, 0, 1) * np.clip((f["deposit"] - 0.5) * 4, 0, 1)
    tr = lay["tracks"]
    ski = rg.region_field({"polylines_ne": tr["polylines_ne"], "width_m": float(tr["width_m"]), "edge": {"soft_m": 0.5}},
                          size, extent)
    k = int(round(3.0 / px_m)) * 2 + 1
    ski = ski * (1.0 - cv2.dilate((path > 0.05).astype(np.uint8), np.ones((k, k), np.uint8)).astype(np.float32))
    to8 = lambda a: np.clip(np.round(a * 255.0), 0, 255).astype(np.uint8)
    return np.dstack([to8(ski), to8(dirt), to8(bare), np.zeros((size, size), np.uint8)])


def plan_actors(m: Mcp) -> list:
    """Everything the debris and scatter steps place, from one seeded stream (the snow blocks first, so they
    stay where they are when the rest changes): snow blocks over the debris; rock fragments and broken
    branches in it; boulder fields and outcrops; dwarf pine mats on melted-out ground; and the layout's
    features beside the scenarios' spots."""
    import cv2
    from vivid2.damage import regions as rg
    from vivid2.terrain import TerrainModel

    lay = layout()
    sc = lay["scatter"]
    rng = np.random.default_rng(int(sc["seed"]))
    ext = float(lay["mask"]["extent_m"])
    fields = rg.fields(lay)
    gm, gm2 = ground_mask(), ground_mask2()  # BGRA, [row = east, col = north]
    t = TerrainModel.load(ROOT / f"sim_config/maps/{NAME}/terrain_dtm.npz")
    keep = np.asarray(sc["keep_clear_ne"], float)
    mesh = scene_meshes(m)
    a, b = np.asarray(FLOW[0]), np.asarray(FLOW[1])

    def field(name, n, e):
        return float(rg.sample(fields[name], np.array([n]), np.array([e]), ext)[0])

    def mask(img, ch, n, e):
        size = img.shape[0]
        i = int(np.clip((e + ext / 2) / ext * size, 0, size - 1))  # row = east
        j = int(np.clip((n + ext / 2) / ext * size, 0, size - 1))  # col = north
        return img[i, j, ch] / 255.0

    def along(n, e):  # 0 at the crown .. 1 at the toe
        d = b - a
        return float(np.clip(np.dot(np.array([n, e]) - a, d) / np.dot(d, d), 0, 1))

    def ground(n, e):
        return -float(t.down_at(n, e))

    def slope_at(n, e):
        return float(np.degrees(np.arctan(np.hypot(ground(n + 2, e) - ground(n - 2, e), ground(n, e + 2) - ground(n, e - 2)) / 4)))

    placed = []

    def free(n, e, r, keep_m=float(sc["keep_clear_m"])):
        if np.any(np.hypot(keep[:, 0] - n, keep[:, 1] - e) < keep_m + r):
            return False
        return all(np.hypot(pn - n, pe - e) > 0.9 * (pr + r) for pn, pe, pr in placed[-400:])

    out = []

    def add(name, kind, tag, n, e, z, rot, scale):
        out.append({"name": name, "mesh": mesh[kind], "tag": tag, "loc": [n * 100, e * 100, z * 100], "rot": rot,
                    "scale": scale})

    def rock_like(name, kind, tag, n, e, size, embed, native):
        s = size / native
        add(name, kind, tag, n, e, ground(n, e) + native * s * 0.7 * (0.5 - embed),
            [rng.uniform(-20, 20), rng.uniform(0, 360), rng.uniform(-20, 20)],
            [s * rng.uniform(0.8, 1.2), s * rng.uniform(0.8, 1.2), s * rng.uniform(0.55, 0.85)])

    # 1. snow blocks over the debris (the first draws of the stream: unchanged since they were placed)
    def block(kind, size, embed, weight, k):
        for _ in range(3000):
            n, e = rng.uniform(-75, 145), rng.uniform(-402, -150)
            f = field("deposit", n, e)
            if f < 0.6 or rng.uniform() > weight(f, along(n, e)) or not free(n, e, size / 2):
                continue
            placed.append((n, e, size / 2))
            native = ROCKS[kind][1]
            s = size / native
            h = native * s * 0.7
            out.append({"name": f"AV_Debris_Block{kind}_{k:04d}", "mesh": mesh[f"block{kind}"], "tag": DEBRIS_TAG,
                        "loc": [n * 100, e * 100, (ground(n, e) + h * (0.5 - embed)) * 100],
                        "rot": [rng.uniform(-25, 25), rng.uniform(0, 360), rng.uniform(-25, 25)],
                        "scale": [s * rng.uniform(0.8, 1.2), s * rng.uniform(0.8, 1.2), s * rng.uniform(0.55, 0.85)]})
            return

    edge = lambda f: 1.0 if f < 0.9 else 0.55          # more at the edges (levees)
    for k in range(140):  # big blocks pile at the toe and along the edges
        block("M", rng.uniform(1.2, 2.6), rng.uniform(0.25, 0.4), lambda f, u: (0.2 + 0.8 * u) * edge(f), k)
    for k in range(650):
        block("S", rng.uniform(0.55, 1.2), rng.uniform(0.2, 0.35), lambda f, u: (0.5 + 0.5 * u) * edge(f), k)
    for k in range(1100):
        block("S", rng.uniform(0.25, 0.55), rng.uniform(0.15, 0.3), lambda f, u: 0.6 + 0.4 * u, k + 650)

    # 2. litter the avalanche carried down: rock fragments and broken dwarf pine branches in the debris
    def in_debris(r, tries=3000):
        for _ in range(tries):
            n, e = rng.uniform(-75, 145), rng.uniform(-402, -150)
            if field("deposit", n, e) >= 0.6 and free(n, e, r):
                placed.append((n, e, r))
                return n, e
        return None

    for k in range(160):
        size = rng.uniform(0.3, 1.0)
        spot = in_debris(size / 2)
        if spot:
            rock_like(f"AV_Litter_Rock_{k:03d}", "rockS", LITTER_TAG, *spot, size, rng.uniform(0.25, 0.45), ROCKS["S"][1])
    for k in range(45):
        length, dia = rng.uniform(1.2, 3.0), rng.uniform(0.08, 0.18)
        spot = in_debris(length / 3)
        if spot:
            n, e = spot
            add(f"AV_Litter_Branch_{k:03d}", "log", LITTER_TAG, n, e, ground(n, e) + dia * 0.3,
                [0.0, rng.uniform(0, 360), rng.uniform(-8, 8)], [dia / LOG[2], length / LOG[1], dia / LOG[2]])

    # 3. rock: outcrops where the slope is too steep for snow, rock bands on steep convex ground (ridges, spurs),
    #    and boulder fields melting out of the snow on gentler ground, all off the avalanche path
    def off_path(n, e, margin=0.02):
        return max(mask(gm, 0, n, e), mask(gm, 1, n, e), mask(gm, 2, n, e)) <= margin

    k = 0
    for _ in range(30000):
        if k >= 110:
            break
        n, e = rng.uniform(-504, 504), rng.uniform(-504, 504)
        if mask(gm, 3, n, e) < 0.6:
            continue
        kind = "L" if rng.uniform() < 0.35 else "M"
        size = rng.uniform(2.5, 5.0) if kind == "L" else rng.uniform(1.0, 2.2)
        if not free(n, e, size / 2):
            continue
        placed.append((n, e, size / 2))
        rock_like(f"AV_Rock_Outcrop_{k:03d}", f"rock{kind}", ROCK_TAG, n, e, size, rng.uniform(0.3, 0.45), ROCKS[kind][1])
        k += 1
    k = 0
    for _ in range(60000):
        if k >= 140:
            break
        n, e = rng.uniform(-500, 500), rng.uniform(-500, 500)
        if not off_path(n, e) or slope_at(n, e) < 30:
            continue
        convex = ground(n, e) - np.mean([ground(n + dn, e + de) for dn, de in ((10, 0), (-10, 0), (0, 10), (0, -10))])
        if convex < 0.8 or rng.uniform() > min(1.0, convex / 3.0):
            continue
        kind = "L" if rng.uniform() < 0.4 else "M"
        size = rng.uniform(2.5, 5.0) if kind == "L" else rng.uniform(1.2, 2.5)
        if not free(n, e, size / 2):
            continue
        placed.append((n, e, size / 2))
        rock_like(f"AV_Rock_Band_{k:03d}", f"rock{kind}", ROCK_TAG, n, e, size, rng.uniform(0.35, 0.5), ROCKS[kind][1])
        k += 1
    k = 0
    for cluster in range(30):  # boulder fields
        for _ in range(500):
            cn, ce = rng.uniform(-480, 480), rng.uniform(-480, 480)
            if off_path(cn, ce) and slope_at(cn, ce) < 30:
                break
        else:
            continue
        radius = rng.uniform(10, 28)
        for _ in range(int(rng.integers(6, 15))):
            n, e = cn + rng.normal(0, radius / 2), ce + rng.normal(0, radius / 2)
            if not off_path(n, e) or not (-500 < n < 500 and -500 < e < 500):
                continue
            kind = "M" if rng.uniform() < 0.45 else "S"
            size = rng.uniform(1.5, 3.2) if kind == "M" else rng.uniform(0.8, 1.5)
            if not free(n, e, size / 2):
                continue
            placed.append((n, e, size / 2))
            rock_like(f"AV_Rock_Boulder_{k:03d}", f"rock{kind}", ROCK_TAG, n, e, size, rng.uniform(0.35, 0.55), ROCKS[kind][1])
            k += 1

    # 4. dwarf pine (haimatsu) mats on the melted-out ground and at its edges, a few on snow nearby
    k = 0
    for _ in range(120000):
        if k >= 600:
            break
        n, e = rng.uniform(-500, 500), rng.uniform(-500, 500)
        bare = mask(gm2, 2, n, e)
        if not off_path(n, e) or rng.uniform() > (0.9 * bare + 0.02):
            continue
        w = rng.uniform(3.0, 8.0)
        if not free(n, e, w / 2.5, keep_m=6.0):
            continue
        placed.append((n, e, w / 2.5))
        s = w / BIRCH[1]
        add(f"AV_Pine_{k:03d}", "pine", PINE_TAG, n, e, ground(n, e) - 0.15, [0.0, rng.uniform(0, 360), 0.0],
            [s, s * rng.uniform(0.8, 1.2), rng.uniform(0.5, 1.2) / BIRCH[2]])
        k += 1

    # 5. the layout's features beside the scenarios' spots
    for k, ft in enumerate(sc.get("features", [])):
        n, e = np.asarray(ft["near"], float) + np.asarray(ft["offset_m"], float)
        size = float(ft["size_m"])
        name = f"AV_Feature_{ft['kind'].capitalize()}_{k:02d}"
        if ft["kind"] in ("block", "rock"):
            kind = "M" if size > 1.1 else "S"
            rock_like(name, f"{ft['kind']}{kind}", FEATURE_TAG, n, e, size, 0.3, ROCKS[kind][1])
        elif ft["kind"] == "pine":
            s = size / BIRCH[1]
            add(name, "pine", FEATURE_TAG, n, e, ground(n, e) - 0.15, [0.0, rng.uniform(0, 360), 0.0], [s, s, 1.1 / BIRCH[2]])
        elif ft["kind"] == "branch":
            add(name, "log", FEATURE_TAG, n, e, ground(n, e) + 0.05, [0.0, rng.uniform(0, 360), 0.0],
                [0.15 / LOG[2], size / LOG[1], 0.15 / LOG[2]])
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
        execute_tool("editor_toolset.toolsets.actor.ActorTools.add_tag", json.dumps({"actor": act, "tag": a["tag"]}))
        made += 1
    return {"made": made}
"""


def _place(m: Mcp, tags) -> None:
    """Remove the actors with ``tags`` and place them again from plan_actors; save the level."""
    _open(m)
    old = 0
    for tag in tags:
        for act in m.tool(SCENE, "find_actors", {"name": "", "tag": tag, "collision_channels": []}):
            m.tool(SCENE, "remove_from_scene", {"actor": act})
            old += 1
    specs = [sp for sp in plan_actors(m) if sp["tag"] in tags]
    made = 0
    for i in range(0, len(specs), 80):
        script = f"SPECS = {json.dumps(json.dumps(specs[i:i + 80]))}\n" + _SPAWN
        r = m.tool("editor_toolset.toolsets.programmatic.ProgrammaticToolset", "execute_tool_script", {"script": script})
        made += (json.loads(r) if isinstance(r, str) else r)["made"]
    m.tool(A, "save_assets", {"asset_paths": [LEVEL]})
    if m.tool(A, "is_dirty", {"asset_path": LEVEL}):
        raise RuntimeError(f"{LEVEL} did not save")
    kinds = {}
    for sp in specs:
        k = "_".join(sp["name"].split("_")[1:3]).rstrip("0123456789LMS")
        kinds[k] = kinds.get(k, 0) + 1
    print(f"{LEVEL}: removed {old} old actors, placed {made}: {kinds}; saved")


def step_debris(m: Mcp) -> None:
    """The snow blocks over the debris (slow: about 1,900 actors)."""
    _place(m, (DEBRIS_TAG,))


def step_scatter(m: Mcp) -> None:
    """Everything else: litter in the debris, rock, dwarf pine and the features beside the scenarios' spots."""
    _place(m, SCATTER_TAGS)


STEPS = {"level": step_level, "check": step_check, "textures": step_textures, "ground": step_ground,
         "clues": step_clues, "debris": step_debris, "scatter": step_scatter}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--steps", default="level", help=f"comma-separated, from: {', '.join(STEPS)}")
    ap.add_argument("--url", default=URL)
    ap.add_argument("--fab-dir", default=str(ROOT / "unreal" / "fab"),
                    help="the Fab (Megascans) downloads, *_8k.zip (step textures; not committed)")
    args = ap.parse_args(argv)
    sys.path.insert(0, str(ROOT / "src"))
    m = Mcp(args.url)
    bad = 0
    for s in args.steps.split(","):
        s = s.strip()
        bad += (STEPS[s](m, Path(args.fab_dir)) if s == "textures" else STEPS[s](m)) or 0
    return int(bad > 0)


if __name__ == "__main__":
    sys.exit(main())
