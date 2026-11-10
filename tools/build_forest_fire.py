"""Build the ForestFire map's burnt forest in the open editor, through its MCP server (UE 5.8).

ForestFire is a copy of ForestSAR (same landscape, terrain model and forest) in which a fire has burnt
the east side. The burn is one mask, made from unreal/VIVID2Env/SourceData/ForestFire/burn_layout.json
by tools/burn_mask.py, and read by both:

  * the ground: M_VIVID2_Landscape_Fire, a copy of M_VIVID2_Landscape that blends to the Forest Fire
    pack's burnt-grass texture where the mask says so;
  * the forest: PCG_VIVID2_ForestFire, a copy of PCG_VIVID2_Forest that, after all of its sampling (so
    every tree stands where it stands in ForestSAR), splits each layer by the mask. Inside the burn,
    trees, stumps and rocks keep their meshes (and collision) but get charred / scorched material
    instances (/Game/VIVID2/Materials/Burnt); understory and grass are gone.

Steps (each rebuilds its assets from scratch; run all, or some with --steps):
    mask       import burn_mask.png as /Game/VIVID2/Textures/T_VIVID2_ForestFire_BurnMask
    materials  the burnt material instances
    ground     M_VIVID2_Landscape_Fire
    forest     PCG_VIVID2_ForestFire
    level      open /Game/VIVID2/Maps/ForestFire, set the ground material and forest graph,
               regenerate the forest and save the level

The level itself (a copy of ForestSAR with the marker BP_VIVID2Map_ForestFire) is committed; this
tool only re-derives the burnt parts. Two PCG behaviours shape the forest graph: the Sample Texture
node multiplies the point's density by the texture value and drops points where it is 0 (hence the
mask's 128..255 encoding and the density reset before sampling), and a node's cached result is
reused after its settings are edited in place (hence the graph is rebuilt, never patched).

    1. python tools/burn_mask.py
    2. open unreal/VIVID2Env in UE 5.8 with the MCP server on, stop Play-In-Editor
    3. python tools/build_forest_fire.py [--steps mask,materials,ground,forest,level]

Standard library only.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from _mcp import URL, Mcp

ROOT = Path(__file__).resolve().parents[1]
MASK_PNG = ROOT / "unreal/VIVID2Env/SourceData/ForestFire/burn_mask.png"
TEX = "/Game/VIVID2/Textures/T_VIVID2_ForestFire_BurnMask"
BURNT = "/Game/VIVID2/Materials/Burnt"
PACK_MI = "/Game/ForestFire/Materials/MatInstances"
GROUND_SRC, GROUND = "/Game/VIVID2/Materials/M_VIVID2_Landscape", "/Game/VIVID2/Materials/M_VIVID2_Landscape_Fire"
FOREST_SRC, FOREST = "/Game/VIVID2/PCG/PCG_VIVID2_Forest", "/Game/VIVID2/PCG/PCG_VIVID2_ForestFire"
LEVEL = "/Game/VIVID2/Maps/ForestFire"
HALF_EXTENT_CM = 50400.0  # the landscape spans -504..504 m; so does the mask
BURNT_FROM = 0.75  # mask density (0.5 unburnt .. 1 burnt) from which a point counts as burnt

A, OBJ = "editor_toolset.toolsets.asset.AssetTools", "editor_toolset.toolsets.object.ObjectTools"
MT, MI = "editor_toolset.toolsets.material.MaterialTools", "editor_toolset.toolsets.material_instance.MaterialInstanceTools"
PCG, SCENE = "PCGToolset.PCGToolset", "editor_toolset.toolsets.scene.SceneTools"

# Leaves: dead, blackened crowns. Bark, stumps: charred black. Rocks keep their materials.
LEAVES = {"tint": {"r": 0.07, "g": 0.05, "b": 0.035, "a": 0}, "desat": 0.6}
BARK = {"tint": {"r": 0.05, "g": 0.045, "b": 0.04, "a": 0}, "desat": 1.0}
BURNT_MIS = {"MI_Birch_Large_01": LEAVES, "MI_Birch_Large_02": LEAVES, "MI_Birch_Small_01": LEAVES,
             "MI_Bark_Birch_Large_01": BARK, "MI_Bark_Birch_Large_02": BARK, "MI_Bark_Birch_Small_01": BARK,
             "MI_Stump_01a": BARK, "MI_Stump_01b": BARK, "MI_Stump_02": BARK}


def obj(path: str) -> Dict[str, str]:
    name = path.rsplit("/", 1)[-1]
    return {"refPath": path if "." in name else f"{path}.{name}"}


def setp(m: Mcp, target: Dict[str, str], **values) -> None:
    for k, v in values.items():
        if not m.tool(OBJ, "set_properties", {"instance": target, "values": json.dumps({k: v})}):
            raise RuntimeError(f"could not set {k} on {target['refPath']}")


def recreate(m: Mcp, path: str) -> None:
    if m.tool(A, "exists", {"path": path}):
        m.tool(A, "delete", {"path": path})


# -- steps ---------------------------------------------------------------------------------------------

def step_mask(m: Mcp) -> None:
    recreate(m, TEX)
    folder, name = TEX.rsplit("/", 1)
    m.tool("editor_toolset.toolsets.texture.TextureTools", "import_file",
           {"folder_path": folder, "asset_name": name, "source_file": str(MASK_PNG)})
    setp(m, obj(TEX), SRGB=False, CompressionSettings="TC_Grayscale", MipGenSettings="TMGS_NoMipmaps",
         NeverStream=True, AddressX="TA_Clamp", AddressY="TA_Clamp")
    m.tool(A, "save_assets", {"asset_paths": [TEX]})
    print(f"{TEX}: imported from {MASK_PNG.name}")


def burnt_mi(src: str) -> str:
    return f"{BURNT}/MI_VIVID2_{src[len('MI_'):]}_Burnt"


def step_materials(m: Mcp) -> None:
    for src, look in BURNT_MIS.items():
        path = burnt_mi(src)
        folder, name = path.rsplit("/", 1)
        if m.tool(A, "exists", {"path": path}):  # update in place: the level's burnt trees reference them
            inst = obj(path)
        else:
            inst = m.tool(MI, "create", {"folder_path": folder, "asset_name": name, "parent": obj(f"{PACK_MI}/{src}")})
        m.tool(MI, "set_vector_parameter", {"instance": inst, "name": "Base Tint", "value": look["tint"]})
        m.tool(MI, "set_scalar_parameter", {"instance": inst, "name": "Base Desaturation", "value": look["desat"]})
        m.tool(A, "save_assets", {"asset_paths": [path]})
    print(f"{BURNT}: {len(BURNT_MIS)} burnt material instances")


def step_ground(m: Mcp) -> None:
    recreate(m, GROUND)
    m.tool(A, "duplicate", {"path": GROUND_SRC, "new_path": GROUND})
    mat = obj(GROUND)
    group = "VIVID2 Burn"

    def add(cls, **props):
        e = m.tool(MT, "add_expression", {"material_or_function": mat, "expression_class": {"refPath": f"/Script/Engine.{cls}"},
                                          "x": 0, "y": 0})
        setp(m, e, **props)
        return e

    def link(a, a_out, b, b_in):
        m.tool(MT, "connect_expressions", {"from_expression": a, "from_output_name": a_out, "to_expression": b,
                                           "to_input_name": b_in})

    base = m.tool(MT, "get_property_input", {"material": mat, "material_property": "MP_BaseColor"})
    rough = m.tool(MT, "get_property_input", {"material": mat, "material_property": "MP_Roughness"})
    xy = add("MaterialExpressionComponentMask", R=True, G=True, B=False, A=False)
    link(add("MaterialExpressionWorldPosition"), "", xy, "")
    # mask UV = (world XY + half extent) / extent; burn = saturate(2 R - 1) (the mask stores 0.5 + b/2)
    plus = add("MaterialExpressionAdd")
    link(xy, "", plus, "A")
    link(add("MaterialExpressionScalarParameter", ParameterName="BurnMaskOffset", DefaultValue=HALF_EXTENT_CM, Group=group),
         "", plus, "B")
    uv = add("MaterialExpressionDivide")
    link(plus, "", uv, "A")
    link(add("MaterialExpressionScalarParameter", ParameterName="BurnMaskExtent", DefaultValue=2 * HALF_EXTENT_CM,
             Group=group), "", uv, "B")
    sample = add("MaterialExpressionTextureSample", Texture=obj(TEX)["refPath"], SamplerType="SAMPLERTYPE_LinearGrayscale")
    link(uv, "", sample, "UVs")
    two = add("MaterialExpressionMultiply", ConstB=2.0)
    link(sample, "R", two, "A")
    less = add("MaterialExpressionSubtract", ConstB=1.0)
    link(two, "", less, "A")
    burn = add("MaterialExpressionSaturate")
    link(less, "", burn, "")
    alpha = add("MaterialExpressionMultiply")
    link(burn, "", alpha, "A")
    link(add("MaterialExpressionScalarParameter", ParameterName="BurnStrength", DefaultValue=1.0, Group=group), "", alpha, "B")
    # burnt ground: the pack's burnt grass, world-tiled and tinted
    buv = add("MaterialExpressionDivide")
    link(xy, "", buv, "A")
    link(add("MaterialExpressionScalarParameter", ParameterName="BurntTiling", DefaultValue=600.0, Group=group), "", buv, "B")
    btex = add("MaterialExpressionTextureSample",
               Texture="/Game/ForestFire/Textures/Ground/T_Ground_BurnedGrass_01_D.T_Ground_BurnedGrass_01_D")
    link(buv, "", btex, "UVs")
    bcol = add("MaterialExpressionMultiply")
    link(btex, "RGB", bcol, "A")
    link(add("MaterialExpressionVectorParameter", ParameterName="BurntTint", DefaultValue={"R": 0.28, "G": 0.26, "B": 0.25, "A": 1.0},
             Group=group), "", bcol, "B")
    lerp = add("MaterialExpressionLinearInterpolate")
    link(base["expression"], base["output_name"], lerp, "A")
    link(bcol, "", lerp, "B")
    link(alpha, "", lerp, "Alpha")
    m.tool(MT, "connect_to_output", {"expression": lerp, "output_name": "", "material_property": "MP_BaseColor"})
    rl = add("MaterialExpressionLinearInterpolate")
    link(rough["expression"], rough["output_name"], rl, "A")
    link(add("MaterialExpressionScalarParameter", ParameterName="BurntRoughness", DefaultValue=0.95, Group=group), "", rl, "B")
    link(alpha, "", rl, "Alpha")
    m.tool(MT, "connect_to_output", {"expression": rl, "output_name": "", "material_property": "MP_Roughness"})
    m.tool(MT, "layout_expressions", {"material_or_function": mat})
    m.tool(MT, "recompile", {"material_or_function": mat})
    m.tool(A, "save_assets", {"asset_paths": [GROUND]})
    print(f"{GROUND}: built")


def step_forest(m: Mcp) -> None:
    recreate(m, FOREST)
    m.tool(A, "duplicate", {"path": FOREST_SRC, "new_path": FOREST})
    g = obj(FOREST)
    gname = FOREST.rsplit("/", 1)[-1]

    def node(name):
        return {"refPath": f"{FOREST}.{gname}:{name}"}

    def add(kind, name, params, title, x, y):
        m.tool(PCG, "AddNode", {"graph": g, "nativeNodeType": kind, "nodeName": name, "jsonParams": json.dumps(params),
                                "nodeTitle": title, "nodeComment": "", "xPositionIdx": x, "yPositionIdx": y})

    def connect(a, ap, b, bp):
        m.tool(PCG, "ConnectNodePins", {"fromNode": node(a), "fromPinLabel": ap, "toNode": node(b), "toPinLabel": bp})

    def selector(name):
        info = m.tool(PCG, "GetNodeInfo", {"node": node(name)})
        info = json.loads(info) if isinstance(info, str) else info
        return info["paramOverrides"]["meshSelectorParameters"]

    add("Get Texture Data", "BurnMask", {
        "texture": obj(TEX), "bUseAbsoluteTransform": True, "colorChannel": "Red", "filter": "Bilinear",
        "transform": {"location": {"x": 0, "y": 0, "z": 0}, "rotation": {"pitch": 0, "yaw": 0, "roll": 0},
                      "scale": {"x": HALF_EXTENT_CM, "y": HALF_EXTENT_CM, "z": 1}}},
        "Burn Mask (SourceData/ForestFire)", 6, -2)
    layers = [("Tree", "TreePose", "TreeSpawner", True, 0), ("Under", "UnderPose", "UnderSpawner", False, 2),
              ("Grass", "GrassPose", "GrassSpawner", False, 4), ("Clutter", "ClutterPose", "ClutterSpawner", True, 6)]
    for key, pose, spawner, keep_burnt, row in layers:
        m.tool(PCG, "DisconnectNodePins", {"fromNode": node(pose), "fromPinLabel": "Out", "toNode": node(spawner), "toPinLabel": "In"})
        add("Attribute Maths Op", f"{key}DensityOne", {"operation": "Ceil", "inputSource1": "$Density", "outputTarget": "$Density"},
            "Density = 1 (the mask sample multiplies it)", 6, row)
        add("Sample Texture", f"{key}BurnSample", {"densityMergeFunction": "Set"}, "Sample Burn Mask (0.5 unburnt .. 1 burnt)", 7, row)
        add("Density Filter", f"{key}Unburnt", {"lowerBound": 0.0, "upperBound": BURNT_FROM - 0.001}, "Unburnt", 8, row)
        add("Density Filter", f"{key}Burnt", {"lowerBound": BURNT_FROM, "upperBound": 1.0}, "Burnt", 8, row + 1)
        connect(pose, "Out", f"{key}DensityOne", "In")
        connect(f"{key}DensityOne", "Out", f"{key}BurnSample", "Point")
        connect("BurnMask", "Out", f"{key}BurnSample", "BaseTexture")
        connect(f"{key}BurnSample", "Out", f"{key}Unburnt", "In")
        connect(f"{key}BurnSample", "Out", f"{key}Burnt", "In")
        connect(f"{key}Unburnt", "Out", spawner, "In")
        if not keep_burnt:
            continue  # burnt understory and grass: gone
        add("Static Mesh Spawner", f"{key}SpawnerBurnt", {}, f"Burnt {key}", 9, row + 1)
        connect(f"{key}Burnt", "Out", f"{key}SpawnerBurnt", "In")
        entries = json.loads(m.tool(OBJ, "get_properties", {"instance": selector(spawner), "properties": ["MeshEntries"]}))["MeshEntries"]
        for e in entries:  # same meshes and collision; burnt materials slot by slot
            slots = json.loads(m.tool(OBJ, "get_properties", {"instance": e["descriptor"]["staticMesh"],
                                                               "properties": ["StaticMaterials"]}))["StaticMaterials"]
            over: List[Any] = []
            for s in slots:
                mi = s.get("materialInterface")
                path = mi.get("refPath") if isinstance(mi, dict) else None
                short = path.rsplit(".", 1)[-1] if path else None
                over.append(obj(burnt_mi(short)) if short in BURNT_MIS else ({"refPath": path} if path else "None"))
            e["descriptor"]["overrideMaterials"] = over
        setp(m, selector(f"{key}SpawnerBurnt"), MeshEntries=entries)
    m.tool(A, "save_assets", {"asset_paths": [FOREST]})
    print(f"{FOREST}: built")


def step_level(m: Mcp) -> None:
    if m.tool(SCENE, "get_current_level", {}) != LEVEL:
        m.tool(SCENE, "load_level", {"level_path": LEVEL})
    land = [a for a in m.tool(SCENE, "find_actors", {"name": "Landscape", "tag": "", "collision_channels": []})
            if a["refPath"].rsplit(".", 1)[-1].startswith("Landscape")]
    for a in land:
        setp(m, a, LandscapeMaterial=obj(GROUND)["refPath"])
    vols = m.tool(PCG, "ListGraphInstances", {})
    if len(vols) != 1:
        raise RuntimeError(f"expected one PCG volume in {LEVEL}, found {len(vols)}")
    vol = vols[0]["actor"]
    setp(m, {"refPath": vol["refPath"] + ".PCG Component.PCGGraphInstance"}, Graph=obj(FOREST)["refPath"])
    for _ in range(12):  # the first call can collide with the refresh the graph change starts
        try:
            msgs = m.tool(PCG, "ExecuteGraphInstance", {"pCGVolume": vol})
            break
        except RuntimeError:
            time.sleep(5)
    else:
        raise RuntimeError("the forest did not regenerate")
    time.sleep(15)
    m.tool(A, "save_assets", {"asset_paths": [LEVEL]})
    print(f"{LEVEL}: ground and forest set, forest regenerated ({msgs or 'no messages'}), saved")


STEPS = {"mask": step_mask, "materials": step_materials, "ground": step_ground, "forest": step_forest, "level": step_level}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--steps", default=",".join(STEPS), help=f"comma-separated, from {','.join(STEPS)}")
    args = ap.parse_args(argv)
    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    unknown = [s for s in steps if s not in STEPS]
    if unknown:
        ap.error(f"unknown steps {unknown}")
    try:
        m = Mcp()
    except OSError as e:
        print(f"cannot reach the editor's MCP server at {URL} ({e}); start it with ModelContextProtocol.StartServer",
              file=sys.stderr)
        return 1
    if m.tool("EditorToolset.EditorAppToolset", "IsPIERunning", {}):
        print("stop Play-In-Editor first: assets can't be edited while it runs", file=sys.stderr)
        return 1
    for s in STEPS:  # always in this order
        if s in steps:
            STEPS[s](m)
    return 0


if __name__ == "__main__":
    sys.exit(main())
