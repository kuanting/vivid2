"""Build VIVID 2's hiker targets in the open editor, through its MCP server (UE 5.8).

Scenarios spawn hikers by blueprint name (``targets`` in a scenario). This tool (re)creates

  /Game/VIVID2/Targets/VIVID2_Hiker_<Outfit>_<Pose>     one blueprint per outfit and pose
  /Game/VIVID2/Targets/Materials/M_VIVID2_ClothTint      the clothing tint material
  /Game/VIVID2/Targets/Materials/Hiker/MI_VIVID2_Hiker_<Outfit>_<Jacket|Pants>
  /Game/VIVID2/Targets/Anims/VIVID2_Wave_<OneArm|TwoArms>[_Anim]   the waving clips (imported once)

from these Fab packs (installed locally, never committed; see docs/environments.md):

  Hiking_Collection (Content/Hiking_Collection)   a man and a woman in hiking clothes with a backpack,
                                                  one skeletal mesh each on a MetaHuman-style skeleton
  Lie Down Animation Pack (Content/Fab/...)        Manny clips: on the back, face down, on the side;
                                                  its Manny skeleton also carries the waving clips

the engine's Third Person template (Content/Characters: the crouch), and VIVID 2's own waving clips
(unreal/VIVID2Env/SourceData/Hikers/*.fbx, keyframed by tools/wave_clips_blender.py): imported onto
the Lie Down pack's skeleton when missing (SkeletalMeshTools.import_file, which makes a 1 cm proxy
mesh <name> and the clip <name>_Anim; it refuses to overwrite, so to update a clip delete both
assets, restart the editor, and run again). The Fab animation packs import from FBX into unsaved
assets: save them once after adding them to the project. A pose whose clip is missing is skipped.

How a hiker is made:
  * one skeletal mesh component holding one frame of one clip (single-node animation, not playing):
    spawned actors play no animation, and a moving pose would differ between runs;
  * the pose clips are on Manny skeletons; the hiker skeleton lists them as compatible skeletons
    and takes bone translations from them (local changes to the pack), so they play on it by bone
    name;
  * the jacket and trousers get M_VIVID2_ClothTint instances: the pack's fabric textures, with the
    colour set by a parameter (the colours of the old mannequin hikers and of the decoys);
  * the mesh's physics asset collides (BlockAll), so preflight's 3D box and drone collisions see it;
  * hair is mesh (cards), so segmentation covers the whole person (grooms would be missed);
  * the pivot is on the ground, under the body's centre.

It also caps the pack textures the hikers use at 2048 px (a local pack change; they ship at 8K).

    1. open unreal/VIVID2Env in UE 5.8 with the MCP server on (ModelContextProtocol.StartServer)
    2. stop Play-In-Editor, then: python tools/build_hikers.py [--only RedJacket_LyingBack] [--skip-textures]

Standard library only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from _mcp import URL, Mcp

TARGETS = "/Game/VIVID2/Targets"
MATS = f"{TARGETS}/Materials"
TINT = f"{MATS}/M_VIVID2_ClothTint"
HIKER_MATS = f"{MATS}/Hiker"  # the outfit instances
ANIMS = f"{TARGETS}/Anims"    # VIVID 2's own clips

PACK = "/Game/Hiking_Collection"
SKELETON = f"{PACK}/Humans/Meshes/SK_Human_Skeleton"
OUTFIT_TEX = f"{PACK}/Outfits/Textures"
LIE = "/Game/Fab/Lie_Down_Animation_Pack"
MANNY_SKELETON = f"{LIE}/Sprawling_Out_on_the_Floor_Skeleton"  # the skeleton the waving clips go on
WAVE_FBX = Path(__file__).resolve().parents[1] / "unreal" / "VIVID2Env" / "SourceData" / "Hikers"
WAVES = ["VIVID2_Wave_OneArm", "VIVID2_Wave_TwoArms"]  # tools/wave_clips_blender.py

ASSETS = "editor_toolset.toolsets.asset.AssetTools"
OBJ = "editor_toolset.toolsets.object.ObjectTools"
MAT = "editor_toolset.toolsets.material.MaterialTools"
MI = "editor_toolset.toolsets.material_instance.MaterialInstanceTools"
BP = "editor_toolset.toolsets.blueprint.BlueprintTools"
ACTOR = "editor_toolset.toolsets.actor.ActorTools"
SKM = "editor_toolset.toolsets.skeletal_mesh.SkeletalMeshTools"

# The two bodies: mesh, and the material slots of the jacket and the trousers.
BODIES: Dict[str, Dict[str, Any]] = {
    "man": dict(mesh=f"{PACK}/Outfits/Meshes/SKM_m_med_nrw_Hiking", slots=12, jacket=9, pants=10),
    "woman": dict(mesh=f"{PACK}/Outfits/Meshes/SKM_f_med_nrw_Hiking", slots=12, jacket=9, pants=10),
}

# Outfits: body, scale, linear colours of the jacket and trousers (as the old mannequins and the decoys).
OUTFITS: Dict[str, Dict[str, Any]] = {
    "RedJacket": dict(body="man", scale=1.0, jacket=(0.55, 0.03, 0.03), pants=(0.04, 0.04, 0.05)),
    "YellowJacket": dict(body="woman", scale=1.0, jacket=(0.75, 0.55, 0.03), pants=(0.12, 0.12, 0.13)),
    # a teenager; a darker blue than the mannequin's (0.02, 0.05, 0.18), which looked mid-blue on fabric
    "BlueHoodie": dict(body="man", scale=0.94, jacket=(0.015, 0.035, 0.13), pants=(0.08, 0.10, 0.16)),
    "GreenJacket": dict(body="man", scale=1.0, jacket=(0.08, 0.13, 0.05), pants=(0.20, 0.16, 0.10)),  # low contrast
    # ski tourers on snow (Avalanche): a black shell, and a light grey one about as bright as snow in shade
    "BlackJacket": dict(body="man", scale=1.0, jacket=(0.012, 0.012, 0.014), pants=(0.02, 0.02, 0.022)),
    "GreyJacket": dict(body="woman", scale=1.0, jacket=(0.30, 0.31, 0.33), pants=(0.05, 0.05, 0.06)),
}

# Mean linear luminance of the pack's base-colour textures (measured from their thumbnails): the tint
# divides by it, so the fabric keeps its relative light and dark detail around the outfit colour.
REF_LUM = {"Jacket": 0.0056, "Pants": 0.0244}
TEXTURES = {"Jacket": "T_Jacket_Camping", "Pants": "T_Pants_Camping"}

# Poses: clip (per body, or one for both), time in the clip (s), and per body the body component's
# offset (cm, in the hiker's frame) that puts the pivot under the body's centre: the negated centre
# of the posed body's footprint, measured with line traces in the editor (2026-11-18). The hiker
# faces +X (north at yaw 0). Frames were picked from contact sheets of each clip.
POSES: Dict[str, Dict[str, Any]] = {
    "Standing": dict(anim={"man": f"{PACK}/Demo/Animations/MM_Idle", "woman": f"{PACK}/Demo/Animations/MF_Idle"},
                     t=0.0, offset={"man": (0, 0), "woman": (0, 0)}),
    # the template's backward stagger, ending in a crouch (as the mannequin hikers); the clip moves
    # the body 2 m sideways
    "Crouching": dict(anim="/Game/Characters/Mannequins/Anims/Death/MM_Death_Back_01", t=1.1,
                      offset={"man": (15, -200), "woman": (15, -195)}),
    # spread-eagle on the back
    "LyingBack": dict(anim=f"{LIE}/Sprawling_Out_on_the_Floor_Anim", t=5.0, offset={"man": (35, -20), "woman": (35, -20)}),
    # face down, arms folded over the head, backpack up
    "LyingFront": dict(anim=f"{LIE}/Liying_Face_Dwon_on_the_Floor_Anim", t=5.0, offset={"man": (15, -5), "woman": (15, -5)}),
    # on the side, propped on one arm
    "LyingSide": dict(anim=f"{LIE}/Lying_on_the_Side_with_One_Arm_Anim", t=6.0, offset={"man": (25, -30), "woman": (25, -30)}),
    # waving for help: both arms overhead, at the open V of the wave (a still frame: a moving wave
    # would differ between runs)
    "Waving": dict(anim=f"{ANIMS}/VIVID2_Wave_TwoArms_Anim", t=0.0, offset={"man": (0, 0), "woman": (0, 0)}),
    # one arm (the right) raised and waving, at the outer end of the wave; the other arm hangs
    "WavingOneArm": dict(anim=f"{ANIMS}/VIVID2_Wave_OneArm_Anim", t=0.25, offset={"man": (0, 0), "woman": (0, 0)}),
}

# The body collides like any obstacle (drones hit hikers) and is query-enabled, so Project AirSim's
# 3D box (GetComponentsBoundingBox skips non-colliding components) has the hiker's size.
COLLISION = {"collisionEnabled": "QueryAndPhysics", "collisionProfileName": "BlockAll", "objectType": "ECC_WorldStatic"}
MAX_TEXTURE = 2048  # px; the payload camera never sees a hiker larger than ~1,000 px


def ref(path: str) -> Dict[str, str]:
    """An object reference for an asset path, with or without its object name."""
    leaf = path.rsplit("/", 1)[1]
    return {"refPath": path if "." in leaf else f"{path}.{leaf}"}


def _set(mcp: Mcp, obj: Dict[str, str], values: Dict[str, Any]) -> None:
    for k, v in values.items():  # one at a time, so a failure names its property
        if not mcp.tool(OBJ, "set_properties", {"instance": obj, "values": json.dumps({k: v})}):
            raise RuntimeError(f"could not set {k} on {obj['refPath']}")


def _get(mcp: Mcp, obj: Dict[str, str], name: str) -> Any:
    r = mcp.tool(OBJ, "get_properties", {"instance": obj, "properties": [name]})
    r = json.loads(r) if isinstance(r, str) else r
    return next(iter(r.values()))


# Assets are updated in place, never deleted and re-created: deleting an asset that something
# references force-deletes it and clears those references (instances lose their parent material,
# placed hikers their class).
def _exists(mcp: Mcp, path: str) -> bool:
    return bool(mcp.tool(ASSETS, "exists", {"path": path}))


# --------------------------------------------------------------------------------------- material
def _expr(mcp: Mcp, mat: Dict[str, str], cls: str, x: int, y: int, **props) -> Dict[str, str]:
    e = mcp.tool(MAT, "add_expression", {"material_or_function": mat,
                                         "expression_class": {"refPath": f"/Script/Engine.MaterialExpression{cls}"},
                                         "x": x, "y": y})
    if props:
        _set(mcp, e, props)
    return e


def _wire(mcp: Mcp, a: Dict[str, str], a_out: str, b: Dict[str, str], b_in: str) -> None:
    # returns nothing on success; Mcp.tool raises on failure
    mcp.tool(MAT, "connect_expressions", {"from_expression": a, "from_output_name": a_out,
                                          "to_expression": b, "to_input_name": b_in})


def _out(mcp: Mcp, e: Dict[str, str], name: str, prop: str) -> None:
    mcp.tool(MAT, "connect_to_output", {"expression": e, "output_name": name, "material_property": prop})


def build_tint_material(mcp: Mcp) -> str:
    """M_VIVID2_ClothTint: base colour = Color x clamp(luminance(BaseColor) / RefLum, 0.6, 1.6).

    The pack's clothes are near-black fabric whose folds and seams are in the normal map, so the
    tint keeps the pack's normal, roughness and metallic maps and only replaces the hue. Two-sided
    and opaque like the pack's master material; flagged for skeletal meshes (a cooked build draws
    the default material on a mesh type a material isn't flagged for).
    """
    mat = ref(TINT)
    if _exists(mcp, TINT):  # rebuild the graph inside the same asset
        for e in mcp.tool(MAT, "get_expressions", {"material_or_function": mat}) or []:
            mcp.tool(MAT, "delete_expression", {"material_or_function": mat, "expression": e})
    else:
        mcp.tool(MAT, "create_material", {"folder_path": MATS, "asset_name": TINT.rsplit("/", 1)[1]})
    tex = lambda name, kind, x, y, sampler: _expr(  # noqa: E731
        mcp, mat, "TextureSampleParameter2D", x, y, ParameterName=name, SamplerType=sampler,
        Texture=ref(f"{OUTFIT_TEX}/T_Jacket_Camping_{kind}")["refPath"])
    base = tex("BaseColor", "BaseColor", -900, -300, "SAMPLERTYPE_Color")
    normal = tex("Normal", "Normal", -900, 100, "SAMPLERTYPE_Normal")
    rough = tex("Roughness", "Roughness", -900, 400, "SAMPLERTYPE_Masks")
    metal = tex("Metallic", "Metallic", -900, 700, "SAMPLERTYPE_Masks")
    weights = _expr(mcp, mat, "Constant3Vector", -900, -500, Constant={"R": 0.2126, "G": 0.7152, "B": 0.0722, "A": 0})
    lum = _expr(mcp, mat, "DotProduct", -600, -400)
    reflum = _expr(mcp, mat, "ScalarParameter", -600, -250, ParameterName="RefLum", DefaultValue=REF_LUM["Jacket"])
    ratio = _expr(mcp, mat, "Divide", -400, -350)
    clamp = _expr(mcp, mat, "Clamp", -250, -350, MinDefault=0.6, MaxDefault=1.6)
    color = _expr(mcp, mat, "VectorParameter", -250, -550, ParameterName="Color",
                  DefaultValue={"R": 0.5, "G": 0.5, "B": 0.5, "A": 1})
    tinted = _expr(mcp, mat, "Multiply", -50, -450)
    _wire(mcp, base, "RGB", lum, "A")
    _wire(mcp, weights, "", lum, "B")
    _wire(mcp, lum, "", ratio, "A")
    _wire(mcp, reflum, "", ratio, "B")
    _wire(mcp, ratio, "", clamp, "None")
    _wire(mcp, color, "RGB", tinted, "A")
    _wire(mcp, clamp, "", tinted, "B")
    _out(mcp, tinted, "", "MP_BaseColor")
    _out(mcp, normal, "RGB", "MP_Normal")
    _out(mcp, rough, "R", "MP_Roughness")
    _out(mcp, metal, "R", "MP_Metallic")
    _set(mcp, mat, {"TwoSided": True, "bUsedWithSkeletalMesh": True})
    mcp.tool(MAT, "recompile", {"material_or_function": mat})
    mcp.tool(ASSETS, "save_assets", {"asset_paths": [TINT]})
    print(f"{TINT}: built")
    return TINT


def build_instances(mcp: Mcp, outfit: str) -> Dict[str, str]:
    spec, out = OUTFITS[outfit], {}
    for part in ("Jacket", "Pants"):
        name = f"MI_VIVID2_Hiker_{outfit}_{part}"
        path = f"{HIKER_MATS}/{name}"
        inst = ref(path)
        if _exists(mcp, path):
            mcp.tool(MI, "set_parent", {"instance": inst, "parent": ref(TINT)})
        else:
            mcp.tool(MI, "create", {"folder_path": HIKER_MATS, "asset_name": name, "parent": ref(TINT)})
        r, g, b = spec[part.lower()]
        mcp.tool(MI, "set_vector_parameter", {"instance": inst, "name": "Color", "value": {"r": r, "g": g, "b": b, "a": 1}})
        mcp.tool(MI, "set_scalar_parameter", {"instance": inst, "name": "RefLum", "value": REF_LUM[part]})
        for kind in ("BaseColor", "Normal", "Roughness", "Metallic"):
            mcp.tool(MI, "set_texture_parameter", {"instance": inst, "name": kind,
                                                   "value": ref(f"{OUTFIT_TEX}/{TEXTURES[part]}_{kind}")})
        mcp.tool(ASSETS, "save_assets", {"asset_paths": [path]})
        out[part] = path
    return out


# -------------------------------------------------------------------------------------- the pack
def import_wave_clips(mcp: Mcp) -> None:
    """Import VIVID 2's waving clips (SourceData/Hikers) onto the Lie Down pack's Manny skeleton, if missing."""
    for name in WAVES:
        if _exists(mcp, f"{ANIMS}/{name}_Anim"):
            continue
        src = WAVE_FBX / f"{name}.fbx"
        if not src.exists() or not _exists(mcp, MANNY_SKELETON):
            print(f"{name}: not imported ({src.name} or the Lie Down pack is missing)")
            continue
        mcp.tool(SKM, "import_file", {"folder_path": ANIMS, "asset_name": name, "source_file": src.as_posix(),
                                      "skeleton": ref(MANNY_SKELETON), "import_animations": True})
        mcp.tool(ASSETS, "save_assets", {"asset_paths": [f"{ANIMS}/{name}", f"{ANIMS}/{name}_Anim"]})
        print(f"{ANIMS}/{name}_Anim: imported from {src.name}")


def pose_skeletons(mcp: Mcp) -> List[str]:
    """Skeletons of every clip a pose uses (the Fab clips each come with their own Manny skeleton)."""
    skels = []
    for spec in POSES.values():
        anims = spec["anim"].values() if isinstance(spec["anim"], dict) else [spec["anim"]]
        for a in anims:
            s = _get(mcp, ref(a), "skeleton")["refPath"]
            if s != ref(SKELETON)["refPath"] and s not in skels:
                skels.append(s)
    return skels


def make_compatible(mcp: Mcp) -> None:
    """Let the hiker skeleton play the pose clips (a local pack change).

    The clips' skeletons become compatible skeletons, and bone translations follow the clips' own
    skeletons' retarget modes (Animation). With the hiker skeleton's modes (mostly
    AnimationRelative, set up for the pack's own retargeted clips) the right arm of a Manny clip
    comes apart at the elbow.
    """
    sk = ref(SKELETON)
    # soft references: {"refPath": ...} for a loaded skeleton, a bare path string for one that isn't
    have = [s["refPath"] if isinstance(s, dict) else s for s in _get(mcp, sk, "compatibleSkeletons")]
    want = sorted(set(have) | set(pose_skeletons(mcp)))
    if want != sorted(have) or not _get(mcp, sk, "bUseRetargetModesFromCompatibleSkeleton"):
        _set(mcp, sk, {"CompatibleSkeletons": want, "bUseRetargetModesFromCompatibleSkeleton": True})
        mcp.tool(ASSETS, "save_assets", {"asset_paths": [SKELETON]})
    print(f"{SKELETON}: {len(want)} compatible skeletons, their retarget modes")


def cap_textures(mcp: Mcp) -> None:
    """Cap every texture the hiker meshes use at MAX_TEXTURE px (local pack change; they are 8K)."""
    seen, todo, textures = set(), [b["mesh"] for b in BODIES.values()], []
    while todo:
        p = todo.pop()
        if p in seen:
            continue
        seen.add(p)
        if not mcp.tool(ASSETS, "exists", {"path": p}):
            # the pack's skin materials reference a texture it doesn't ship (an opacity mask for
            # hiding body parts under clothes); the engine uses its default texture instead
            print(f"  (missing pack dependency, skipped: {p})")
            continue
        cls = mcp.tool(ASSETS, "get_asset_class", {"asset_path": p})
        if cls.startswith("Texture"):
            textures.append(p)
        elif cls.startswith(("SkeletalMesh", "Material")):
            todo += [d for d in (mcp.tool(ASSETS, "get_dependencies", {"asset_path": p}) or []) if d.startswith("/Game/")]
    changed = []
    for t in sorted(textures):
        if _get(mcp, ref(t), "maxTextureSize") != MAX_TEXTURE:
            _set(mcp, ref(t), {"MaxTextureSize": MAX_TEXTURE})
            changed.append(t)
    if changed:
        mcp.tool(ASSETS, "save_assets", {"asset_paths": changed})
    print(f"textures: {len(textures)} used by the hikers, {len(changed)} capped at {MAX_TEXTURE} px")


# ----------------------------------------------------------------------------------- blueprints
def build_blueprint(mcp: Mcp, outfit: str, pose: str, mats: Dict[str, str], folder: str = TARGETS,
                    name: Optional[str] = None, t: Optional[float] = None, anim: Optional[str] = None) -> str:
    o, p = OUTFITS[outfit], POSES[pose]
    body = BODIES[o["body"]]
    name = name or f"VIVID2_Hiker_{outfit}_{pose}"
    path = f"{folder}/{name}"
    bp = ref(path)
    comp = {"refPath": f"{path}.{name}_C:Body_GEN_VARIABLE"}  # the component template of an existing blueprint
    if not _exists(mcp, path):
        mcp.tool(BP, "create", {"folder_path": folder, "asset_name": name, "asset_type": {"refPath": "/Script/Engine.Actor"}})
        comp = mcp.tool(ACTOR, "add_component", {"owner": bp, "name": "Body",
                                                 "component_type": {"refPath": "/Script/Engine.SkeletalMeshComponent"}})
    elif mcp.tool(OBJ, "get_class", {"instance": comp})["refPath"] != "/Script/Engine.SkeletalMeshComponent":
        raise RuntimeError(f"{path} has no SkeletalMeshComponent 'Body'; delete it and run again")
    clip = anim or (p["anim"][o["body"]] if isinstance(p["anim"], dict) else p["anim"])
    overrides: List[Optional[str]] = [None] * body["slots"]
    overrides[body["jacket"]] = ref(mats["Jacket"])["refPath"]
    overrides[body["pants"]] = ref(mats["Pants"])["refPath"]
    s = o["scale"]
    ox, oy = p["offset"][o["body"]]
    # clear first: resizing and changing an existing list in one go is refused as ambiguous
    _set(mcp, comp, {"OverrideMaterials": []})
    _set(mcp, comp, {
        "SkeletalMeshAsset": ref(body["mesh"])["refPath"],
        "OverrideMaterials": overrides,
        "AnimationMode": "AnimationSingleNode",
        "AnimationData": {"AnimToPlay": ref(clip)["refPath"], "bSavedLooping": False, "bSavedPlaying": False,
                          "SavedPosition": p["t"] if t is None else t, "SavedPlayRate": 0.0},
        # pose evaluated even when the component isn't rendered by the editor view (scene captures)
        "VisibilityBasedAnimTickOption": "AlwaysTickPoseAndRefreshBones",
        "bEnableUpdateRateOptimizations": False,
        "RelativeLocation": {"X": ox * s, "Y": oy * s, "Z": 0},
        "RelativeRotation": {"Pitch": 0, "Yaw": 0, "Roll": 0},  # the mannequin hikers lay rotated
        "RelativeScale3D": {"X": s, "Y": s, "Z": s},
    })
    bi = _get(mcp, comp, "bodyInstance")
    bi.update(COLLISION)
    _set(mcp, comp, {"BodyInstance": bi})
    mcp.tool(BP, "compile_blueprint", {"blueprint": bp})
    mcp.tool(ASSETS, "save_assets", {"asset_paths": [path]})
    return path


def available_poses(mcp: Mcp) -> List[str]:
    out = []
    for pose, spec in POSES.items():
        anims = spec["anim"].values() if isinstance(spec["anim"], dict) else [spec["anim"]]
        if all(mcp.tool(ASSETS, "exists", {"path": a}) for a in anims):
            out.append(pose)
        else:
            print(f"{pose}: skipped, its clip is not installed ({', '.join(anims)})")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", action="append", help="build only these <Outfit>_<Pose> hikers")
    ap.add_argument("--skip-textures", action="store_true", help="don't cap the pack's texture sizes")
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
    if not mcp.tool(ASSETS, "exists", {"path": SKELETON}):
        print(f"the Hiking_Collection pack is not installed ({SKELETON})", file=sys.stderr)
        return 1
    import_wave_clips(mcp)
    poses = available_poses(mcp)
    make_compatible(mcp)
    if not args.skip_textures:
        cap_textures(mcp)
    build_tint_material(mcp)
    wanted: List[Tuple[str, str]] = [(o, p) for o in OUTFITS for p in poses
                                     if not args.only or f"{o}_{p}" in args.only]
    mats = {o: build_instances(mcp, o) for o in sorted({o for o, _ in wanted})}
    for outfit, pose in wanted:
        print(f"{build_blueprint(mcp, outfit, pose, mats[outfit])}: built")
    return 0


if __name__ == "__main__":
    sys.exit(main())
