"""Build VIVID 2's fire effects and fire blueprints in the open editor, through its MCP server (UE 5.8).

Scenarios spawn fires by blueprint name (``fires`` in a scenario; see src/vivid2/hazards/fire.py).
This tool (re)creates them:

  /Game/VIVID2/FX/NS_VIVID2_Fire_<Size>    Niagara systems: flames and a smoke plume (Smolder: smoke only)
  /Game/VIVID2/Fire/VIVID2_Fire_<Size>     the blueprints: the system, a light, and a ground-truth proxy

for <Size> in Small, Medium, Large and Smolder. The effects use the Forest Fire pack's flipbook
materials (Content/ForestFire, from Fab), so the pack must be installed; the pack's own Cascade
effects are not used: the engine reduces them with distance from the editor's player view, not
from the drone cameras, so a fire 450 m from the start vanishes from the payload camera.

The Niagara systems are built for a benchmark:
  * no effect type, so no scalability or distance culling; fixed bounds;
  * CPU simulation with a fixed random seed per system, and a warm-up so the plume is full when
    the fire spawns (flame and smoke motion still follow the game clock, which follows the wall
    clock, so frames differ a little between runs: see hazards/fire.py);
  * local space: the smoke drifts along the blueprint's +X axis, so the runner turns each fire to
    face downwind.

Each blueprint's ground-truth proxy is a cylinder the size of the flames that is not drawn in the
colour or depth image but is drawn into custom depth, so the segmentation camera sees the fire
(Project AirSim gives each fire a segmentation ID by name). It has query-only collision that
ignores every channel: preflight can measure it, drones fly through it, traces ignore it.

    1. open unreal/VIVID2Env in UE 5.8 with the MCP server on (ModelContextProtocol.StartServer)
    2. stop Play-In-Editor, then: python tools/build_fire_assets.py [--only Large] [--skip-blueprints]

Standard library only.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any, Dict, List, Optional

from _mcp import URL, Mcp

FX = "/Game/VIVID2/FX"
FIRE = "/Game/VIVID2/Fire"
PACK_FX = "/Game/ForestFire/Materials/MatInstances_FX"
SYS = "NiagaraToolsets.NiagaraToolset_System"
INFO = "NiagaraToolsets.NiagaraToolset_Info"
ASSETS = "editor_toolset.toolsets.asset.AssetTools"
TEMPLATE_SYSTEM = "/Niagara/DefaultAssets/Templates/Systems/MinimalLightweight.MinimalLightweight"
FOUNTAIN = "/Niagara/DefaultAssets/Templates/Emitters/Fountain.Fountain"
SUBUV = "/Niagara/Modules/Update/SubUV/V2/SubUVAnimation.SubUVAnimation"
FLOAT_FROM_CURVE = "/Niagara/DynamicInputs/ValueFromCurve/FloatFromCurve.FloatFromCurve"
SCALE_SPRITE = "/Niagara/Modules/Update/Size/ScaleSpriteSize.ScaleSpriteSize"
FLIPBOOK_GRID = (8, 8)  # the pack's fire and smoke flipbooks are 8 x 8 frames

# Per fire size (lengths in cm, Unreal units). radius: the burning patch; flame_h: flame height
# (also the proxy's); smoke: plume settings, or None.
SIZES: Dict[str, Dict[str, Any]] = {
    "Small": dict(seed=11, radius=150, flame_h=250, flame_rate=25, flame_size=(120, 220), flame_up=(80, 180),
                  flame_life=(0.7, 1.1), flame_mat="MI_FireFlipMedium_01",
                  smoke=dict(rate=1.5, size=(250, 400), grow=3.0, life=(8.0, 11.0), up=(120, 220), alpha=0.55,
                             colour=(0.30, 0.29, 0.28)),
                  light=dict(intensity=8000.0, radius=1500.0)),
    "Medium": dict(seed=12, radius=300, flame_h=400, flame_rate=45, flame_size=(200, 380), flame_up=(100, 220),
                   flame_life=(0.8, 1.3), flame_mat="MI_FlipFire_Large_01",
                   smoke=dict(rate=2.5, size=(400, 650), grow=3.0, life=(9.0, 12.0), up=(150, 260), alpha=0.6,
                              colour=(0.27, 0.26, 0.25)),
                   light=dict(intensity=20000.0, radius=2500.0)),
    "Large": dict(seed=13, radius=600, flame_h=700, flame_rate=80, flame_size=(350, 650), flame_up=(120, 260),
                  flame_life=(0.9, 1.5), flame_mat="MI_FlipFire_Large_01",
                  smoke=dict(rate=4.0, size=(600, 950), grow=3.0, life=(10.0, 13.0), up=(180, 300), alpha=0.65,
                             colour=(0.22, 0.21, 0.20)),
                  light=dict(intensity=50000.0, radius=4000.0)),
    # Smouldering ground: a thin pale plume and no flames.
    "Smolder": dict(seed=14, radius=200, flame_h=60, flame_rate=0, smoke=dict(rate=1.0, size=(250, 400), grow=2.5,
                    life=(8.0, 11.0), up=(60, 120), alpha=0.4, colour=(0.55, 0.54, 0.52)),
                    light=None),
}
# Smoke drifts downwind along local +X: a constant push against drag, so it approaches about
# SMOKE_DRIFT / SMOKE_DRAG (3 m/s) sideways and rises with buoyancy.
SMOKE_DRIFT = (75.0, 0.0, 45.0)  # cm/s^2 (x: downwind, z: buoyancy)
SMOKE_DRAG = 0.25
FLAME_BUOYANCY = 150.0  # cm/s^2 upwards
WARMUP_S = 14.0  # longer than the longest smoke lifetime, so the plume is full at spawn
# Fixed bounds of the whole system (cm, local): the plume drifts up to ~35 m downwind and rises ~40 m.
BOUNDS = {"min": {"x": -2500.0, "y": -2500.0, "z": -200.0}, "max": {"x": 6000.0, "y": 2500.0, "z": 6000.0}}


def ref(path: str) -> Dict[str, str]:
    """An object reference; a package path /Game/X/Name becomes /Game/X/Name.Name."""
    name = path.rsplit("/", 1)[-1]
    return {"refPath": path if "." in name else f"{path}.{name}"}


class NiagaraEdit:
    """Edits one Niagara system's stack through NiagaraToolset_System."""

    SCRIPTS = ("EmitterSpawnScript", "EmitterUpdateScript", "ParticleSpawnScript", "ParticleUpdateScript")

    def __init__(self, mcp: Mcp, system_path: str):
        self.mcp, self.system = mcp, ref(system_path)
        self._enums: Dict[str, Dict[str, str]] = {}

    def call(self, tool: str, args: Dict[str, Any]):
        return self.mcp.tool(SYS, tool, args)

    def item(self, emitter: str, script: str = "", module: str = "", inputs: Optional[List[str]] = None,
             renderer: int = -1) -> Dict[str, Any]:
        return {"system": self.system, "emitterName": emitter, "scriptName": script, "moduleName": module,
                "rendererIndex": renderer, "inputNameStack": list(inputs or [])}

    def set(self, emitter: str, script: str, module: str, inputs: List[str], struct: str, value: Dict[str, Any]):
        return self.call("SetStackInputData", {"stackInputRef": self.item(emitter, script, module, inputs),
                                               "inputData": {"struct": {"refPath": struct}, "value": value}})

    def f(self, e, s, m, name, v: float):
        return self.set(e, s, m, _names(name), "/Script/Niagara.NiagaraFloat", {"value": float(v)})

    def b(self, e, s, m, name, v: bool):
        return self.set(e, s, m, _names(name), "/Script/Niagara.NiagaraBool", {"value": -1 if v else 0})

    def i(self, e, s, m, name, v: int):
        return self.set(e, s, m, _names(name), "/Script/Niagara.NiagaraInt32", {"value": int(v)})

    def v3(self, e, s, m, name, v):
        return self.set(e, s, m, _names(name), "/Script/CoreUObject.Vector3f", {"x": v[0], "y": v[1], "z": v[2]})

    def v2(self, e, s, m, name, v):
        return self.set(e, s, m, _names(name), "/Script/CoreUObject.Vector2f", {"x": v[0], "y": v[1]})

    def colour(self, e, s, m, name, rgba):
        r, g, b_, a = rgba
        return self.set(e, s, m, _names(name), "/Script/CoreUObject.LinearColor", {"r": r, "g": g, "b": b_, "a": a})

    def enum(self, e, s, m, name, enum_path: str, display: str):
        if enum_path not in self._enums:
            info = self.mcp.tool(INFO, "UEnum_Info", {"enum": {"refPath": enum_path}})
            info = json.loads(info) if isinstance(info, str) else info
            self._enums[enum_path] = {v["DisplayName"]: v["Name"] for v in info["Values"]}
        short = enum_path.rsplit(".", 1)[-1]
        return self.set(e, s, m, _names(name), "/Script/NiagaraEditor.NiagaraExt_StackInputData_Enum",
                        {"enum": {"refPath": enum_path}, "enumName": f"{short}::{self._enums[enum_path][display]}",
                         "displayName": display})

    def dynamic(self, e, s, m, name, asset: str):
        return self.set(e, s, m, _names(name), "/Script/NiagaraEditor.NiagaraExt_StackInputData_DynamicInput",
                        {"dynamicInputAsset": {"refPath": asset}})

    def linked(self, e, s, m, name, variable: str, type_path: str):
        return self.set(e, s, m, _names(name), "/Script/NiagaraEditor.NiagaraExt_StackInputData_Linked",
                        {"linkedVariable": {"name": variable, "type": {"classStructOrEnum": {"refPath": type_path}}}})

    def curve(self, e, s, m, name, keys):
        """A float curve input (a curve data interface) through ``keys`` [(time, value), ...], linear."""
        k = [{"interpMode": "RCIM_Linear", "tangentMode": "RCTM_Auto", "tangentWeightMode": "RCTWM_WeightedNone",
              "time": float(t), "value": float(v), "arriveTangent": 0, "arriveTangentWeight": 0, "leaveTangent": 0,
              "leaveTangentWeight": 0} for t, v in keys]
        di = {"Curve": {"keys": k, "defaultValue": 3.4028234663852886e+38, "preInfinityExtrap": "RCCE_Constant",
                        "postInfinityExtrap": "RCCE_Constant"}, "CurveAsset": "None", "bUseLUT": True,
              "bExposeCurve": False, "bOptimizeLUT": True, "bOverrideOptimizeThreshold": False,
              "OptimizeThreshold": 0.01, "ExposedName": "Float Curve"}
        return self.set(e, s, m, _names(name), "/Script/NiagaraEditor.NiagaraExt_StackInputData_DataInterface",
                        {"propertyValues": json.dumps(di)})

    def add_module(self, emitter: str, script: str, asset: str):
        return self.call("AddModule", {"moduleLocationRef": self.item(emitter, script), "moduleAsset": {"refPath": asset}})

    def module_names(self, emitter: str, script: str) -> List[str]:
        topo = self.call("GetScriptStackTopology", {"scriptRef": self.item(emitter, script)})
        topo = json.loads(topo) if isinstance(topo, str) else topo
        return [m["moduleName"] for m in topo.get("modules", [])]

    def set_props(self, tool: str, key: str, target: Dict[str, Any], values: Dict[str, Any]):
        return self.call(tool, {key: target, ({"SetSystemData": "systemData", "SetEmitterData": "emitterData",
                                               "SetRendererData": "rendererData"}[tool]): {
                                     "propertyValues": json.dumps(values)}})


def _names(name) -> List[str]:
    return list(name) if isinstance(name, (list, tuple)) else [name]


PS, PU = "ParticleSpawnScript", "ParticleUpdateScript"
VEL_MODE = "/Niagara/Enums/Utility/ENiagara_VelocityMode.ENiagara_VelocityMode"
SHAPES = "/Niagara/Enums/Location/ENiagara_LocationShapes.ENiagara_LocationShapes"
SCALE_COLOR_MODE = "/Niagara/Enums/ENiagaraScaleColorMode.ENiagaraScaleColorMode"


def _emitter(nx: NiagaraEdit, name: str, seed: int) -> None:
    """A Fountain-based emitter turned into a local-space, deterministic, gravity-free base."""
    nx.call("AddEmitter", {"system": nx.system, "templateEmitter": {"refPath": FOUNTAIN}, "emitterName": name})
    nx.set_props("SetEmitterData", "emitter", nx.item(name), {"bLocalSpace": True, "bDeterminism": True,
                                                             "RandomSeed": seed})


def _flipbook(nx: NiagaraEdit, emitter: str, material: str) -> None:
    nx.add_module(emitter, PU, SUBUV)
    nx.set_props("SetRendererData", "renderer", nx.item(emitter, renderer=0), {
        "Material": ref(f"{PACK_FX}/{material}"),
        "SubImageSize": {"x": FLIPBOOK_GRID[0], "y": FLIPBOOK_GRID[1]},
        "bSubImageBlend": True, "SortMode": "ViewDistance"})


def _common(nx: NiagaraEdit, e: str, rate: float, life, size, up, radius: float, height: float) -> None:
    nx.f(e, "EmitterUpdateScript", "SpawnRate", "SpawnRate", rate)
    nx.f(e, PS, "InitializeParticle", "Lifetime Min", life[0])
    nx.f(e, PS, "InitializeParticle", "Lifetime Max", life[1])
    nx.f(e, PS, "InitializeParticle", "Uniform Sprite Size Min", size[0])
    nx.f(e, PS, "InitializeParticle", "Uniform Sprite Size Max", size[1])
    nx.enum(e, PS, "ShapeLocation", "Shape Primitive", SHAPES, "Cylinder")
    nx.f(e, PS, "ShapeLocation", "Cylinder Radius", radius)
    nx.f(e, PS, "ShapeLocation", "Cylinder Height", height)
    nx.f(e, PS, "AddVelocity", ["Velocity Speed", "Minimum"], up[0])
    nx.f(e, PS, "AddVelocity", ["Velocity Speed", "Maximum"], up[1])


def _fade(nx: NiagaraEdit, e: str, keys) -> None:
    """Scale the particles' alpha by a curve of normalised age (the template's ScaleColor module)."""
    nx.enum(e, PU, "ScaleColor", "Scale Mode", SCALE_COLOR_MODE, "RGB and Alpha Separately")
    nx.dynamic(e, PU, "ScaleColor", "Scale Alpha", FLOAT_FROM_CURVE)
    nx.curve(e, PU, "ScaleColor", ["Scale Alpha", "FloatCurve"], keys)


OBJ = "editor_toolset.toolsets.object.ObjectTools"


def ensure_niagara_usage(mcp: Mcp, instances: List[str]) -> None:
    """Flag the pack materials behind ``instances`` as used with Niagara sprites (they were made for
    Cascade); without it a cooked build draws the default material. Changes the pack's files locally."""
    done = set()
    for mi in instances:
        cur = ref(f"{PACK_FX}/{mi}")["refPath"]
        while cur and cur not in done:  # walk up the instance chain to the material
            done.add(cur)
            cls = mcp.tool(OBJ, "get_class", {"instance": {"refPath": cur}})["refPath"]
            parent = None
            if cls != "/Script/Engine.Material":
                parent = mcp.tool(OBJ, "get_properties", {"instance": {"refPath": cur}, "properties": ["Parent"]})
                parent = (json.loads(parent) if isinstance(parent, str) else parent).get("Parent")
            if not parent or not parent.get("refPath"):
                flags = mcp.tool(OBJ, "get_properties", {"instance": {"refPath": cur},
                                                         "properties": ["bUsedWithNiagaraSprites"]})
                flags = json.loads(flags) if isinstance(flags, str) else flags
                if not flags.get("bUsedWithNiagaraSprites"):
                    mcp.tool(OBJ, "set_properties", {"instance": {"refPath": cur},
                                                     "values": json.dumps({"bUsedWithNiagaraSprites": True})})
                    mcp.tool("editor_toolset.toolsets.material.MaterialTools", "recompile",
                             {"material_or_function": {"refPath": cur}})
                    mcp.tool(ASSETS, "save_assets", {"asset_paths": [cur.split(".")[0]]})
                    print(f"{cur.split('.')[0]}: now used with Niagara sprites (pack file changed locally)")
                break
            cur = parent["refPath"]


def build_system(mcp: Mcp, size: str, spec: Dict[str, Any]) -> str:
    path = f"{FX}/NS_VIVID2_Fire_{size}"
    if mcp.tool(ASSETS, "exists", {"path": path}):
        mcp.tool(ASSETS, "delete", {"path": path})
    mcp.tool(SYS, "CreateNiagaraSystem", {"assetName": path.rsplit("/", 1)[-1], "assetPath": FX,
                                          "templateSystem": {"refPath": TEMPLATE_SYSTEM}})
    nx = NiagaraEdit(mcp, path)
    summary = nx.call("GetSystemSummary", {"system": nx.system})
    summary = json.loads(summary) if isinstance(summary, str) else summary
    for em in summary["emitters"]:  # the template's lightweight emitter
        nx.call("RemoveEmitter", {"emitterToRemove": nx.item(em["emitterName"])})

    if spec["flame_rate"]:
        e = "Flames"
        _emitter(nx, e, spec["seed"])
        _common(nx, e, spec["flame_rate"], spec["flame_life"], spec["flame_size"], spec["flame_up"],
                spec["radius"], 50.0)
        nx.v3(e, PU, "GravityForce", "Gravity", (0.0, 0.0, FLAME_BUOYANCY))
        _fade(nx, e, [(0.0, 0.0), (0.1, 1.0), (0.7, 1.0), (1.0, 0.0)])
        _flipbook(nx, e, spec["flame_mat"])

    sm = spec["smoke"]
    e = "Smoke"
    _emitter(nx, e, spec["seed"] + 100)
    _common(nx, e, sm["rate"], sm["life"], sm["size"], sm["up"], spec["radius"] * 0.8, 150.0)
    r, g, b = sm["colour"]
    nx.colour(e, PS, "InitializeParticle", "Color", (r, g, b, sm["alpha"]))
    nx.v3(e, PU, "GravityForce", "Gravity", SMOKE_DRIFT)
    nx.f(e, PU, "Drag", "Drag", SMOKE_DRAG)
    # Puffs grow over their life (initial size x curve of normalised age) and fade in and out.
    nx.add_module(e, PU, SCALE_SPRITE)
    nx.curve(e, PU, "ScaleSpriteSize", "Uniform Curve Sprite Scale", [(0.0, 1.0), (1.0, sm["grow"])])
    _fade(nx, e, [(0.0, 0.0), (0.08, 1.0), (0.6, 0.8), (1.0, 0.0)])
    _flipbook(nx, e, "MI_SmokeLarge_FlipLit")

    nx.set_props("SetSystemData", "system", nx.system, {
        "EffectType": None, "bDeterminism": True, "RandomSeed": spec["seed"], "WarmupTime": WARMUP_S,
        "WarmupTickDelta": 1.0 / 15.0, "bFixedBounds": True, "FixedBounds": {**BOUNDS, "isValid": True}})
    for _ in range(240):  # wait for the system's compile to finish (it compiles after every edit)
        try:
            issues = nx.call("GetStackIssues", {"system": nx.system})
            break
        except RuntimeError as e:
            if "compile is in flight" not in str(e):
                raise
            time.sleep(0.5)
    else:
        raise RuntimeError(f"{path}: still compiling after 2 minutes")
    issues = json.loads(issues) if isinstance(issues, str) else issues
    mcp.tool(ASSETS, "save_assets", {"asset_paths": [path]})
    bad = [f"{i['severity']}: {i['shortDescription']} ({i['stackDisplayPath']})" for i in issues.get("issues", [])
           if i["severity"] in ("Error", "Warning") and i["shortDescription"] != "Object has warnings"]
    print(f"{path}: built" + ("".join(f"\n  {b}" for b in bad) if bad else ", no warnings"))
    return path


ACTOR = "editor_toolset.toolsets.actor.ActorTools"
# Every standard channel ignored: the proxy blocks and overlaps nothing, but it is query-enabled, so
# GetComponentsBoundingBox (preflight's size check) counts it.
PROXY_CHANNELS = ["WorldStatic", "WorldDynamic", "Pawn", "Visibility", "Camera", "PhysicsBody", "Vehicle", "Destructible"]
CYLINDER = "/Engine/BasicShapes/Cylinder.Cylinder"  # 1 m across and 1 m tall, pivot at its centre


def _set(mcp: Mcp, obj: Dict[str, str], values: Dict[str, Any]) -> None:
    for k, v in values.items():  # one at a time, so a failure names its property
        if not mcp.tool(OBJ, "set_properties", {"instance": obj, "values": json.dumps({k: v})}):
            raise RuntimeError(f"could not set {k} on {obj['refPath']}")


def build_blueprint(mcp: Mcp, size: str, spec: Dict[str, Any], system_path: str) -> str:
    path = f"{FIRE}/VIVID2_Fire_{size}"
    if mcp.tool(ASSETS, "exists", {"path": path}):
        mcp.tool(ASSETS, "delete", {"path": path})
    mcp.tool("NiagaraToolsets.NiagaraToolset_Blueprint", "ConstructNiagaraBPWrapperFromSystem", {
        "newAssetPath": path, "system": ref(system_path), "parentClass": {"refPath": "/Script/Engine.Actor"}})
    bp = ref(path)

    proxy = mcp.tool(ACTOR, "add_component", {"owner": bp, "component_type": {"refPath": "/Script/Engine.StaticMeshComponent"},
                                              "name": "GroundTruth"})
    d, h = 2.0 * spec["radius"] / 100.0, spec["flame_h"] / 100.0
    _set(mcp, proxy, {
        "StaticMesh": CYLINDER, "RelativeLocation": {"X": 0, "Y": 0, "Z": spec["flame_h"] / 2.0},
        "RelativeScale3D": {"X": d, "Y": d, "Z": h},
        # drawn into custom depth only (segmentation), never into the colour or depth image
        "bRenderInMainPass": False, "bRenderInDepthPass": False, "CastShadow": False, "bVisibleInRayTracing": False,
        "bVisibleInReflectionCaptures": False, "bVisibleInRealTimeSkyCaptures": False, "bReceivesDecals": False})
    body = mcp.tool(OBJ, "get_properties", {"instance": proxy, "properties": ["BodyInstance"]})
    body = (json.loads(body) if isinstance(body, str) else body)["BodyInstance"]
    body.update({"collisionEnabled": "QueryOnly", "collisionProfileName": "Custom", "objectType": "ECC_WorldDynamic",
                 "collisionResponses": {"responseArray": [{"channel": c, "response": "ECR_Ignore"} for c in PROXY_CHANNELS]}})
    _set(mcp, proxy, {"BodyInstance": body})

    if spec.get("light"):
        light = mcp.tool(ACTOR, "add_component", {"owner": bp, "component_type": {"refPath": "/Script/Engine.PointLightComponent"},
                                                  "name": "FireLight"})
        # constant (no flicker, so frames don't depend on when they were taken) and shadowless
        _set(mcp, light, {"IntensityUnits": "Unitless", "Intensity": spec["light"]["intensity"],
                          "AttenuationRadius": spec["light"]["radius"], "LightColor": {"R": 255, "G": 120, "B": 40, "A": 255},
                          "CastShadows": False, "RelativeLocation": {"X": 0, "Y": 0, "Z": spec["flame_h"] / 2.0}})
    mcp.tool("editor_toolset.toolsets.blueprint.BlueprintTools", "compile_blueprint", {"blueprint": bp})
    mcp.tool(ASSETS, "save_assets", {"asset_paths": [path]})
    print(f"{path}: built (proxy {d:g} x {d:g} x {h:g} m{', light' if spec.get('light') else ''})")
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", choices=sorted(SIZES), action="append", help="build only these sizes")
    ap.add_argument("--skip-blueprints", action="store_true")
    args = ap.parse_args(argv)
    try:
        mcp = Mcp()
    except OSError as e:
        print(f"cannot reach the editor's MCP server at {URL} ({e}); start it with ModelContextProtocol.StartServer",
              file=sys.stderr)
        return 1
    if mcp.tool("EditorToolset.EditorAppToolset", "IsPIERunning", {}):
        print("stop Play-In-Editor first: assets can't be edited while it runs", file=sys.stderr)
        return 1
    sizes = args.only or sorted(SIZES)
    ensure_niagara_usage(mcp, sorted({SIZES[s].get("flame_mat") for s in sizes if SIZES[s].get("flame_mat")}
                                     | {"MI_SmokeLarge_FlipLit"}))
    for size in sizes:
        system = build_system(mcp, size, SIZES[size])
        if not args.skip_blueprints:
            build_blueprint(mcp, size, SIZES[size], system)
    return 0


if __name__ == "__main__":
    sys.exit(main())
