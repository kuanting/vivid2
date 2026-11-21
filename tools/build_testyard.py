"""Build VIVID 2's test map, TestYard, in the open editor through its MCP server (UE 5.8).

TestYard is a flat yard at height 0 (Unreal Z = 0) for smoke tests and damage-mapper tests: two
roads, four box houses and a landslide's debris lobe. Everything is made of engine shapes
(/Engine/BasicShapes) with VIVID 2's own materials, and placed from
unreal/VIVID2Env/SourceData/TestYard/layout.json and its ground truth
(sim_config/maps/TestYard/damage_gt.json, from ``vivid2 damage build --map TestYard``), so what
the camera sees and what the ground truth says agree:

  * the debris lobe: brown ground (the layout's mask, read by the ground material), soil mounds,
    boulders and logs, and a mud cover where it crosses road R01 (R01-03 is blocked);
  * B01, B04: intact houses; B02 (partial): half its roof sags into the house, rubble against the
    south wall; B03 (collapsed): a rubble pile with roof fragments that spills across road R02
    (R02-03 is blocked).

Steps (run all, or some with --steps):
    level      create /Game/VIVID2/Maps/TestYard if it is missing: a copy of ForestSAR_Dense (its
               sun, sky and fog) without the landscape, forest or marker, with BP_VIVID2Map_TestYard
    mask       import masks/slide.png as T_VIVID2_TestYard_DebrisMask
    materials  M_VIVID2_TestYard (a colour with world-space noise) and its instances, and
               M_VIVID2_TestYard_Ground (grass, or debris where the mask says so)
    actors     remove the yard's actors (tag VIVID2TestYard), place them again, save the level

    1. vivid2 damage build --map TestYard
    2. open unreal/VIVID2Env in UE 5.8 with the MCP server on, stop Play-In-Editor
    3. python tools/build_testyard.py [--steps level,mask,materials,actors]
    4. after a run that created the level, restart the editor before pressing Play: the deleted
       landscape stays in the editor's memory, and Project AirSim crashes on it when a scene loads
       (UnrealHelpers::InitSegmentationID<ALandscapeProxy>). The saved level is clean.

Actor names start with TY_ and never contain a scenario id (Hiker1, Pack1...): the runner finds
spawned targets and clues by name.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from _mcp import URL, Mcp

ROOT = Path(__file__).resolve().parents[1]
LAYOUT = ROOT / "unreal/VIVID2Env/SourceData/TestYard/layout.json"
GT = ROOT / "sim_config/maps/TestYard/damage_gt.json"
MASK_PNG = ROOT / "unreal/VIVID2Env/SourceData/TestYard/masks/slide.png"

LEVEL = "/Game/VIVID2/Maps/TestYard"
LEVEL_SRC = "/Game/VIVID2/Maps/ForestSAR_Dense"
MARKER_SRC, MARKER = "/Game/VIVID2/Maps/BP_VIVID2Map_ForestSAR", "/Game/VIVID2/Maps/BP_VIVID2Map_TestYard"
FOLDER = "/Game/VIVID2/TestYard"
TEX = f"{FOLDER}/T_VIVID2_TestYard_DebrisMask"
M_BASE, M_GROUND = f"{FOLDER}/M_VIVID2_TestYard", f"{FOLDER}/M_VIVID2_TestYard_Ground"
TAG = "VIVID2TestYard"
CUBE, SPHERE, CYLINDER = "/Engine/BasicShapes/Cube", "/Engine/BasicShapes/Sphere", "/Engine/BasicShapes/Cylinder"
GROUND_SIZE_M = 600.0
KEEP_CLEAR = [(35.0, -40.0), (24.0, -24.0)]  # the scenarios' clues lie here: no boulder on top
CLEAR_M = 3.5
SEED = 20261121

# name: (colour, roughness)
LOOKS = {
    "Asphalt": ((0.045, 0.045, 0.05), 0.7), "Marking": ((0.75, 0.75, 0.7), 0.6),
    "Wall": ((0.55, 0.52, 0.47), 0.85), "Roof": ((0.33, 0.11, 0.07), 0.7), "RoofGrey": ((0.22, 0.23, 0.25), 0.6),
    "Rubble": ((0.42, 0.40, 0.37), 0.9), "Rock": ((0.30, 0.28, 0.26), 0.9), "Wood": ((0.24, 0.16, 0.09), 0.85),
    "Soil": ((0.27, 0.20, 0.13), 0.95), "Mud": ((0.23, 0.17, 0.11), 0.9), "Dark": ((0.03, 0.03, 0.03), 0.9),
}

A, OBJ, SCENE = "editor_toolset.toolsets.asset.AssetTools", "editor_toolset.toolsets.object.ObjectTools", \
    "editor_toolset.toolsets.scene.SceneTools"
MT, MI = "editor_toolset.toolsets.material.MaterialTools", "editor_toolset.toolsets.material_instance.MaterialInstanceTools"
PROG = "editor_toolset.toolsets.programmatic.ProgrammaticToolset"


def obj(path: str) -> Dict[str, str]:
    name = path.rsplit("/", 1)[-1]
    return {"refPath": path if "." in name else f"{path}.{name}"}


def mi_path(look: str) -> str:
    return f"{FOLDER}/MI_VIVID2_TestYard_{look}"


def setp(m: Mcp, target: Dict[str, str], **values) -> None:
    for k, v in values.items():
        if not m.tool(OBJ, "set_properties", {"instance": target, "values": json.dumps({k: v})}):
            raise RuntimeError(f"could not set {k} on {target['refPath']}")


# -- steps ---------------------------------------------------------------------------------------------

def step_level(m: Mcp) -> None:
    if not m.tool(A, "exists", {"path": MARKER}):
        m.tool(A, "duplicate", {"path": MARKER_SRC, "new_path": MARKER})
        m.tool(A, "save_assets", {"asset_paths": [MARKER]})
    if not m.tool(A, "exists", {"path": LEVEL}):
        m.tool(A, "duplicate", {"path": LEVEL_SRC, "new_path": LEVEL})
        m.tool(A, "save_assets", {"asset_paths": [LEVEL]})
    if m.tool(SCENE, "get_current_level", {}) != LEVEL:
        m.tool(SCENE, "load_level", {"level_path": LEVEL})
    keep = ("WorldSettings", "SunSky", "ExponentialHeightFog", "PlayerStart", "DefaultPhysicsVolume",
            "GameplayDebuggerPlayerManager", "ChaosDebugDrawActor", "AbstractNavData", "Brush", "TY_", "BP_VIVID2Map_TestYard")
    removed = []
    for a in m.tool(SCENE, "find_actors", {"name": "", "tag": "", "collision_channels": []}):
        name = a["refPath"].rsplit(".", 1)[-1]
        if not name.startswith(keep):
            m.tool(SCENE, "remove_from_scene", {"actor": a})
            removed.append(name)
    if not m.tool(SCENE, "find_actors", {"name": "BP_VIVID2Map_TestYard", "tag": "", "collision_channels": []}):
        m.tool(SCENE, "add_to_scene_from_asset", {"asset_path": MARKER, "name": "BP_VIVID2Map_TestYard",
                                                  "xform": {"location": {"x": 0, "y": 0, "z": 0}}})
    start = m.tool(SCENE, "find_actors", {"name": "PlayerStart", "tag": "", "collision_channels": []})
    for p in start:  # on the ground at the origin (docs/environments.md)
        m.tool("editor_toolset.toolsets.actor.ActorTools", "set_actor_transform",
               {"actor": p, "xform": {"location": {"x": 0, "y": 0, "z": 100}}})
    m.tool(A, "save_assets", {"asset_paths": [LEVEL]})
    print(f"{LEVEL}: removed {removed or 'nothing'}; marker and player start in place")


def step_mask(m: Mcp) -> None:
    if m.tool(A, "exists", {"path": TEX}):
        m.tool(A, "delete", {"path": TEX})
    folder, name = TEX.rsplit("/", 1)
    m.tool("editor_toolset.toolsets.texture.TextureTools", "import_file",
           {"folder_path": folder, "asset_name": name, "source_file": str(MASK_PNG)})
    setp(m, obj(TEX), SRGB=False, CompressionSettings="TC_Grayscale", MipGenSettings="TMGS_NoMipmaps",
         NeverStream=True, AddressX="TA_Clamp", AddressY="TA_Clamp")
    m.tool(A, "save_assets", {"asset_paths": [TEX]})
    print(f"{TEX}: imported from {MASK_PNG.name}")


class _Graph:
    def __init__(self, m: Mcp, mat: Dict[str, str]):
        self.m, self.mat = m, mat

    def add(self, cls: str, **props):
        e = self.m.tool(MT, "add_expression", {"material_or_function": self.mat,
                                                "expression_class": {"refPath": f"/Script/Engine.{cls}"}, "x": 0, "y": 0})
        if props:
            setp(self.m, e, **props)
        return e

    def link(self, a, a_out, b, b_in):
        self.m.tool(MT, "connect_expressions", {"from_expression": a, "from_output_name": a_out, "to_expression": b,
                                                "to_input_name": b_in})

    def out(self, e, prop):
        self.m.tool(MT, "connect_to_output", {"expression": e, "output_name": "", "material_property": prop})

    def noisy(self, colour_param: str, default, scale: float, lo: float, hi: float):
        """colour parameter x world-space noise in [lo, hi]."""
        noise = self.add("MaterialExpressionNoise", Scale=scale, Levels=4, OutputMin=lo, OutputMax=hi, Quality=2)
        self.link(self.add("MaterialExpressionWorldPosition"), "XYZ", noise, "World Position")
        mul = self.add("MaterialExpressionMultiply")
        self.link(self.add("MaterialExpressionVectorParameter", ParameterName=colour_param,
                           DefaultValue={"R": default[0], "G": default[1], "B": default[2], "A": 1.0}), "", mul, "A")
        self.link(noise, "", mul, "B")
        return mul

    def finish(self, path):
        self.m.tool(MT, "layout_expressions", {"material_or_function": self.mat})
        self.m.tool(MT, "recompile", {"material_or_function": self.mat})
        self.m.tool(A, "save_assets", {"asset_paths": [path]})


def _fresh_material(m: Mcp, path: str) -> Dict[str, str]:
    if m.tool(A, "exists", {"path": path}):
        return obj(path)
    folder, name = path.rsplit("/", 1)
    return m.tool(MT, "create_material", {"folder_path": folder, "asset_name": name})


def step_materials(m: Mcp) -> None:
    for path in (M_BASE, M_GROUND):  # rebuilt from scratch; the instances keep pointing at them
        mat = _fresh_material(m, path)
        for e in m.tool(MT, "get_expressions", {"material_or_function": mat}) or []:
            m.tool(MT, "delete_expression", {"material_or_function": mat, "expression": e})
    g = _Graph(m, obj(M_BASE))
    g.out(g.noisy("Color", (0.5, 0.5, 0.5), 0.004, 0.78, 1.18), "MP_BaseColor")
    g.out(g.add("MaterialExpressionScalarParameter", ParameterName="Roughness", DefaultValue=0.8), "MP_Roughness")
    g.finish(M_BASE)

    lay = json.loads(LAYOUT.read_text(encoding="utf-8"))
    half_cm = float(lay["mask"]["extent_m"]) * 50.0
    g = _Graph(m, obj(M_GROUND))
    xy = g.add("MaterialExpressionComponentMask", R=True, G=True, B=False, A=False)
    g.link(g.add("MaterialExpressionWorldPosition"), "XYZ", xy, "")
    plus = g.add("MaterialExpressionAdd", ConstB=half_cm)
    g.link(xy, "", plus, "A")
    uv = g.add("MaterialExpressionDivide", ConstB=2 * half_cm)
    g.link(plus, "", uv, "A")
    sample = g.add("MaterialExpressionTextureSample", Texture=obj(TEX)["refPath"], SamplerType="SAMPLERTYPE_LinearGrayscale")
    g.link(uv, "", sample, "UVs")
    two = g.add("MaterialExpressionMultiply", ConstB=2.0)  # the mask stores 0.5 + field / 2
    g.link(sample, "R", two, "A")
    less = g.add("MaterialExpressionSubtract", ConstB=1.0)
    g.link(two, "", less, "A")
    alpha = g.add("MaterialExpressionSaturate")
    g.link(less, "", alpha, "")
    lerp = g.add("MaterialExpressionLinearInterpolate")
    g.link(g.noisy("GrassColor", (0.15, 0.19, 0.08), 0.002, 0.7, 1.25), "", lerp, "A")
    g.link(g.noisy("DebrisColor", (0.27, 0.20, 0.13), 0.012, 0.55, 1.35), "", lerp, "B")
    g.link(alpha, "", lerp, "Alpha")
    g.out(lerp, "MP_BaseColor")
    g.out(g.add("MaterialExpressionConstant", R=0.92), "MP_Roughness")
    g.finish(M_GROUND)

    for look, (rgb, rough) in LOOKS.items():
        path = mi_path(look)
        folder, name = path.rsplit("/", 1)
        inst = obj(path) if m.tool(A, "exists", {"path": path}) else \
            m.tool(MI, "create", {"folder_path": folder, "asset_name": name, "parent": obj(M_BASE)})
        m.tool(MI, "set_vector_parameter", {"instance": inst, "name": "Color",
                                            "value": {"r": rgb[0], "g": rgb[1], "b": rgb[2], "a": 1.0}})
        m.tool(MI, "set_scalar_parameter", {"instance": inst, "name": "Roughness", "value": rough})
        m.tool(A, "save_assets", {"asset_paths": [path]})
    print(f"{FOLDER}: {M_BASE.rsplit('/', 1)[-1]}, {M_GROUND.rsplit('/', 1)[-1]} and {len(LOOKS)} instances")


# -- the yard's actors ----------------------------------------------------------------------------------

def _box(name, look, n, e, z, sn, se, sz, yaw=0.0, pitch=0.0, roll=0.0, mesh=CUBE) -> Dict[str, Any]:
    """An actor: centre (n, e, z up) in metres, size in metres (a 1 m engine shape scaled)."""
    return {"name": name, "mesh": mesh, "look": look, "loc": [n * 100, e * 100, z * 100],
            "rot": [pitch, yaw, roll], "scale": [sn, se, sz]}


def _roads(lay, field_at) -> List[Dict[str, Any]]:
    out = []
    for r in lay["roads"]:
        (n0, e0), (n1, e1) = r["polyline_ne"][0], r["polyline_ne"][-1]
        w, length = float(r["width_m"]), math.hypot(n1 - n0, e1 - e0)
        along_n = abs(n1 - n0) >= abs(e1 - e0)
        top = 0.04 if along_n else 0.05  # crossing roads don't fight over the same height
        cn, ce = (n0 + n1) / 2, (e0 + e1) / 2
        sn, se = (length, w) if along_n else (w, length)
        out.append(_box(f"TY_{r['id']}_Asphalt", "Asphalt", cn, ce, top / 2, sn, se, top))
        other = [x for x in lay["roads"] if x is not r]
        cross = [(x["polyline_ne"][0][0] if along_n else x["polyline_ne"][0][1]) for x in other]  # the crossing road's line
        t, k = 0.0, 0
        while t + 3.0 <= length:  # centre dashes: 3 m every 6 m, none in the crossing or under debris
            mid = (n0 if along_n else e0) + t + 1.5
            pn, pe = (mid, ce) if along_n else (cn, mid)
            if all(abs(mid - c) > w / 2 + 2 for c in cross) and field_at(pn, pe) < 0.3:
                out.append(_box(f"TY_{r['id']}_Dash_{k:03d}", "Marking", pn, pe, top + 0.005,
                                3.0 if along_n else 0.15, 0.15 if along_n else 3.0, 0.01))
                k += 1
            t += 6.0
    return out


def _house(b, rng) -> List[Dict[str, Any]]:
    fp = np.asarray(b["footprint_ne"], float)
    (n_lo, e_lo), (n_hi, e_hi) = fp.min(axis=0), fp.max(axis=0)
    dn, de, h = n_hi - n_lo, e_hi - e_lo, float(b["height_m"])
    cn, ce = (n_lo + n_hi) / 2, (e_lo + e_hi) / 2
    bid, roof = b["id"], "Roof" if de * dn > 60 else "RoofGrey"
    out = []
    if b["grade"] == "none":
        out.append(_box(f"TY_{bid}_Walls", "Wall", cn, ce, h / 2, dn, de, h))
        out.append(_box(f"TY_{bid}_Roof", roof, cn, ce, h + 0.15, dn + 0.6, de + 0.6, 0.3))
        out.append(_box(f"TY_{bid}_Vent", "RoofGrey", cn + dn * 0.2, ce - de * 0.25, h + 0.6, 1.2, 1.0, 0.6))
        return out
    if b["grade"] == "partial":
        t = 0.3
        out += [_box(f"TY_{bid}_WallN", "Wall", n_hi - t / 2, ce, h / 2, t, de, h),
                _box(f"TY_{bid}_WallE", "Wall", cn, e_hi - t / 2, h / 2, dn, t, h),
                _box(f"TY_{bid}_WallW", "Wall", cn, e_lo + t / 2, h / 2, dn, t, h),
                # the south wall is broken in the middle, its rubble outside
                _box(f"TY_{bid}_WallS1", "Wall", n_lo + t / 2, e_lo + de * 0.17, h / 2, t, de * 0.34, h),
                _box(f"TY_{bid}_WallS2", "Wall", n_lo + t / 2, e_hi - de * 0.17, h * 0.4, t, de * 0.34, h * 0.8),
                _box(f"TY_{bid}_Floor", "Dark", cn, ce, 0.05, dn - 2 * t, de - 2 * t, 0.1)]
        # the east 55 % of the roof stays; the west part has broken off and sags into the house
        keep = de * 0.55
        out.append(_box(f"TY_{bid}_Roof", roof, cn, e_hi - keep / 2 + 0.3, h + 0.15, dn + 0.6, keep + 0.6, 0.3))
        sag_w = de - keep
        drop = 2.6
        tilt = math.degrees(math.atan2(drop, sag_w))
        out.append(_box(f"TY_{bid}_RoofSag", roof, cn, e_lo + sag_w / 2, h - drop / 2, dn - 0.8, sag_w / math.cos(math.radians(tilt)),
                        0.3, roll=-tilt))
        for i in range(18):
            s = rng.uniform(0.4, 1.3)
            out.append(_box(f"TY_{bid}_Rubble_{i:02d}", "Rubble" if i % 4 else "Roof", n_lo - rng.uniform(0.3, 2.4),
                            ce + rng.uniform(-de * 0.22, de * 0.22), s * 0.35, s, s * rng.uniform(0.5, 1.2), s * 0.6,
                            yaw=rng.uniform(0, 360), pitch=rng.uniform(-25, 25), roll=rng.uniform(-25, 25)))
        return out
    # collapsed: a rubble pile over the footprint, with roof fragments, spilling over its debris polygon
    from vivid2.geo import points_in_polygon
    rub = float(b.get("post_height_m", 2.5))
    k = 0
    for i in range(80):
        u, v = rng.uniform(-0.5, 0.5, 2)
        n, e = cn + u * dn, ce + v * de
        prof = 1.0 - 1.6 * (u * u + v * v)
        s = rng.uniform(0.6, 2.2)
        look = "Rubble" if i % 5 else ("Wall" if i % 10 else "Roof")
        out.append(_box(f"TY_{bid}_Rubble_{k:03d}", look, n, e, max(s * 0.25, rub * prof * rng.uniform(0.3, 1.0)), s,
                        s * rng.uniform(0.5, 1.3), s * rng.uniform(0.3, 0.8),
                        yaw=rng.uniform(0, 360), pitch=rng.uniform(-35, 35), roll=rng.uniform(-35, 35)))
        k += 1
    for i in range(5):
        out.append(_box(f"TY_{bid}_RoofSlab_{i}", "Roof", cn + rng.uniform(-dn, dn) * 0.25, ce + rng.uniform(-de, de) * 0.25,
                        rub * rng.uniform(0.5, 0.9), rng.uniform(2.5, 4.0), rng.uniform(2.0, 3.5), 0.3,
                        yaw=rng.uniform(0, 360), pitch=rng.uniform(-30, 30), roll=rng.uniform(-30, 30)))
    poly = b.get("debris_ne")
    if poly:
        pn, pe = np.asarray(poly, float).min(axis=0), np.asarray(poly, float).max(axis=0)
        placed = 0
        while placed < 70:
            p = rng.uniform(pn, pe)
            if not points_in_polygon(p[None, :], poly)[0] or (n_lo <= p[0] <= n_hi and e_lo <= p[1] <= e_hi):
                continue
            s = rng.uniform(0.3, 1.3)
            out.append(_box(f"TY_{bid}_Spill_{placed:03d}", "Rubble" if placed % 6 else "Roof", p[0], p[1], s * 0.2,
                            s, s * rng.uniform(0.5, 1.3), s * 0.5,
                            yaw=rng.uniform(0, 360), pitch=rng.uniform(-30, 30), roll=rng.uniform(-30, 30)))
            placed += 1
    return out


def _debris(lay, field_at, rng) -> List[Dict[str, Any]]:
    pts = [(float(n), float(e)) for n in np.arange(-100, 100.5, 1.0) for e in np.arange(-100, 100.5, 1.0)
           if field_at(n, e) >= 0.6]

    def free(n, e, r):
        return all(math.hypot(n - a, e - b) > CLEAR_M + r for a, b in KEEP_CLEAR)

    def pick(r, min_field=0.6):
        for _ in range(200):
            n, e = pts[rng.integers(len(pts))]
            n, e = n + rng.uniform(-0.5, 0.5), e + rng.uniform(-0.5, 0.5)
            if field_at(n, e) >= min_field and free(n, e, r):
                return n, e
        raise RuntimeError("no free spot in the debris")

    out = []
    for i in range(10):  # low soil mounds: half spheres a metre or so high
        sx, sy, sz = rng.uniform(4, 8), rng.uniform(4, 8), rng.uniform(1.4, 2.6)
        n, e = pick(max(sx, sy) / 2, 0.8)
        out.append(_box(f"TY_Slide_Mound_{i:02d}", "Soil", n, e, 0.0, sx, sy, sz, yaw=rng.uniform(0, 180), mesh=SPHERE))
    for i in range(75):  # boulders
        s = rng.uniform(0.4, 1.8)
        n, e = pick(s / 2)
        sz = s * rng.uniform(0.5, 0.9)
        out.append(_box(f"TY_Slide_Boulder_{i:02d}", "Rock", n, e, sz * 0.35, s * rng.uniform(0.8, 1.3), s, sz,
                        yaw=rng.uniform(0, 360), pitch=rng.uniform(-15, 15), mesh=SPHERE))
    for i in range(22):  # broken trunks, lying
        d, length = rng.uniform(0.25, 0.45), rng.uniform(4, 9)
        n, e = pick(length / 2)
        out.append(_box(f"TY_Slide_Log_{i:02d}", "Wood", n, e, d / 2, d, d, length,
                        yaw=rng.uniform(0, 180), pitch=90.0, mesh=CYLINDER))
    for r in lay["roads"]:  # mud over the road where the debris crosses it
        (n0, e0), (n1, e1) = r["polyline_ne"][0], r["polyline_ne"][-1]
        along_n = abs(n1 - n0) >= abs(e1 - e0)
        w, steps = float(r["width_m"]), int(math.hypot(n1 - n0, e1 - e0))
        for k in range(steps + 1):
            t = k / max(steps, 1)
            n, e = n0 + t * (n1 - n0), e0 + t * (e1 - e0)
            if field_at(n, e) >= 0.5:
                out.append(_box(f"TY_{r['id']}_Mud_{k:03d}", "Mud", n, e, 0.08, 1.3 if along_n else w + 1.0,
                                w + 1.0 if along_n else 1.3, 0.12, yaw=rng.uniform(-6, 6)))
    return out


def yard_actors() -> List[Dict[str, Any]]:
    from vivid2.damage import regions as rg

    lay = json.loads(LAYOUT.read_text(encoding="utf-8"))
    gt = json.loads(GT.read_text(encoding="utf-8"))
    field = rg.fields(lay)["slide"]
    extent = float(lay["mask"]["extent_m"])

    def field_at(n, e):
        return float(rg.sample(field, np.array([n]), np.array([e]), extent)[0])

    rng = np.random.default_rng(SEED)
    out = [_box("TY_Ground", "Ground", 0.0, 0.0, -0.5, GROUND_SIZE_M, GROUND_SIZE_M, 1.0)]
    out += _roads(lay, field_at)
    for b in gt["buildings"]:
        out += _house(b, rng)
    out += _debris(lay, field_at, rng)
    return out


_SPAWN_SCRIPT = """
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
        comp = execute_tool("editor_toolset.toolsets.actor.ActorTools.get_root_component",
                            json.dumps({"actor": act}))["returnValue"]
        execute_tool("editor_toolset.toolsets.object.ObjectTools.set_properties",
                     json.dumps({"instance": comp, "values": json.dumps({"OverrideMaterials": [a["material"]]})}))
        made += 1
    return {"made": made}
"""


def step_actors(m: Mcp) -> None:
    if m.tool(SCENE, "get_current_level", {}) != LEVEL:
        m.tool(SCENE, "load_level", {"level_path": LEVEL})
    old = m.tool(SCENE, "find_actors", {"name": "", "tag": TAG, "collision_channels": []})
    for a in old:
        m.tool(SCENE, "remove_from_scene", {"actor": a})
    specs = yard_actors()
    for s in specs:
        path = M_GROUND if s["look"] == "Ground" else mi_path(s["look"])
        s["material"] = obj(path)["refPath"]
    made = 0
    for i in range(0, len(specs), 60):
        chunk = specs[i:i + 60]
        script = f"SPECS = {json.dumps(json.dumps(chunk))}\nTAG = {json.dumps(TAG)}\n" + _SPAWN_SCRIPT
        r = m.tool(PROG, "execute_tool_script", {"script": script})
        made += (json.loads(r) if isinstance(r, str) else r)["made"]
    m.tool(A, "save_assets", {"asset_paths": [LEVEL]})
    kinds: Dict[str, int] = {}
    for s in specs:
        k = s["name"].split("_")[2] if s["name"].count("_") >= 2 else s["name"]
        kinds[k] = kinds.get(k, 0) + 1
    print(f"{LEVEL}: removed {len(old)} old actors, placed {made}: {kinds}")


STEPS = {"level": step_level, "mask": step_mask, "materials": step_materials, "actors": step_actors}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--steps", default=",".join(STEPS), help=f"comma-separated, from {','.join(STEPS)}")
    args = ap.parse_args(argv)
    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    unknown = [s for s in steps if s not in STEPS]
    if unknown:
        ap.error(f"unknown steps {unknown}")
    if not GT.exists():
        print(f"{GT} is missing: run  vivid2 damage build --map TestYard  first")
        return 1
    try:
        m = Mcp()
    except OSError as e:
        print(f"no MCP server at {URL} ({e}): open unreal/VIVID2Env and run ModelContextProtocol.StartServer")
        return 1
    if m.tool("EditorToolset.EditorAppToolset", "IsPIERunning", {}):
        print("stop Play-In-Editor first")
        return 1
    for s in steps:
        STEPS[s](m)
    return 0


if __name__ == "__main__":
    sys.exit(main())
