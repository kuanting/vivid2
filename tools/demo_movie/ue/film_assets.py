"""Editor side: the film's own materials under /Game/VIVID2/Demo/Materials (never cooked, git-ignored).

* M_FILM_Paint: opaque, parameters Color / Roughness / Metallic; instances MI_FILM_<name> per entry
  of ``paints`` in the args (the film drone's body, arms, ducts...);
* M_FILM_NoFly: translucent unlit red with a rim, for the fire no-fly volume;
* M_FILM_Glow: unlit emissive (Color, Strength), for small navigation-light beads.

Re-running updates the instances in place (assets are never deleted: see the MCP gotchas in memory).
Prints one JSON line.
"""
import json
import sys
from pathlib import Path

import unreal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ue_exec import script_args  # noqa: E402

ARGS = script_args()
DIR = "/Game/VIVID2/Demo/Materials"
tools = unreal.AssetToolsHelpers.get_asset_tools()
mel = unreal.MaterialEditingLibrary
eal = unreal.EditorAssetLibrary
made = []


def material(name, build):
    path = f"{DIR}/{name}"
    if eal.does_asset_exist(path):
        return unreal.load_asset(path)
    m = tools.create_asset(name, DIR, unreal.Material, unreal.MaterialFactoryNew())
    build(m)
    mel.recompile_material(m)
    eal.save_loaded_asset(m, False)
    made.append(path)
    return m


def vparam(m, name, default, x, y):
    e = mel.create_material_expression(m, unreal.MaterialExpressionVectorParameter, x, y)
    e.set_editor_property("parameter_name", name)
    e.set_editor_property("default_value", unreal.LinearColor(*default))
    return e


def sparam(m, name, default, x, y):
    e = mel.create_material_expression(m, unreal.MaterialExpressionScalarParameter, x, y)
    e.set_editor_property("parameter_name", name)
    e.set_editor_property("default_value", float(default))
    return e


def build_paint(m):
    c = vparam(m, "Color", (0.8, 0.8, 0.8, 1.0), -400, -100)
    r = sparam(m, "Roughness", 0.45, -400, 100)
    mt = sparam(m, "Metallic", 0.0, -400, 200)
    mel.connect_material_property(c, "", unreal.MaterialProperty.MP_BASE_COLOR)
    mel.connect_material_property(r, "", unreal.MaterialProperty.MP_ROUGHNESS)
    mel.connect_material_property(mt, "", unreal.MaterialProperty.MP_METALLIC)


def build_glow(m):
    m.set_editor_property("shading_model", unreal.MaterialShadingModel.MSM_UNLIT)
    c = vparam(m, "Color", (1.0, 0.1, 0.05, 1.0), -500, 0)
    s = sparam(m, "Strength", 20.0, -500, 150)
    mul = mel.create_material_expression(m, unreal.MaterialExpressionMultiply, -250, 50)
    mel.connect_material_expressions(c, "", mul, "A")
    mel.connect_material_expressions(s, "", mul, "B")
    mel.connect_material_property(mul, "", unreal.MaterialProperty.MP_EMISSIVE_COLOR)


def build_nofly(m):
    m.set_editor_property("blend_mode", unreal.BlendMode.BLEND_TRANSLUCENT)
    m.set_editor_property("shading_model", unreal.MaterialShadingModel.MSM_UNLIT)
    m.set_editor_property("two_sided", True)
    c = vparam(m, "Color", (1.0, 0.08, 0.03, 1.0), -600, -100)
    s = sparam(m, "Strength", 2.0, -600, 50)
    o = sparam(m, "Opacity", 0.18, -600, 200)
    fres = mel.create_material_expression(m, unreal.MaterialExpressionFresnel, -600, 300)
    rim = sparam(m, "RimWeight", 0.25, -600, 420)
    rimw = mel.create_material_expression(m, unreal.MaterialExpressionMultiply, -450, 330)
    mel.connect_material_expressions(fres, "", rimw, "A")
    mel.connect_material_expressions(rim, "", rimw, "B")
    add = mel.create_material_expression(m, unreal.MaterialExpressionAdd, -350, 250)
    mel.connect_material_expressions(o, "", add, "A")
    mel.connect_material_expressions(rimw, "", add, "B")
    sat = mel.create_material_expression(m, unreal.MaterialExpressionSaturate, -200, 250)
    mel.connect_material_expressions(add, "", sat, "")
    mul = mel.create_material_expression(m, unreal.MaterialExpressionMultiply, -350, 0)
    mel.connect_material_expressions(c, "", mul, "A")
    mel.connect_material_expressions(s, "", mul, "B")
    mel.connect_material_property(mul, "", unreal.MaterialProperty.MP_EMISSIVE_COLOR)
    mel.connect_material_property(sat, "", unreal.MaterialProperty.MP_OPACITY)


def instance(parent, name, vectors=None, scalars=None):
    path = f"{DIR}/{name}"
    if eal.does_asset_exist(path):
        mi = unreal.load_asset(path)
    else:
        mi = tools.create_asset(name, DIR, unreal.MaterialInstanceConstant, unreal.MaterialInstanceConstantFactoryNew())
        mi.set_editor_property("parent", parent)
        made.append(path)
    for k, v in (vectors or {}).items():
        mel.set_material_instance_vector_parameter_value(mi, k, unreal.LinearColor(*v))
    for k, v in (scalars or {}).items():
        mel.set_material_instance_scalar_parameter_value(mi, k, float(v))
    mel.update_material_instance(mi)
    eal.save_loaded_asset(mi, False)
    return mi


eal.make_directory(DIR)
paint = material("M_FILM_Paint", build_paint)
glow = material("M_FILM_Glow", build_glow)
nofly = material("M_FILM_NoFly2", build_nofly)  # (M_FILM_NoFly: first version, unweighted rim)
instance(nofly, "MI_FILM_NoFly", {"Color": (1.0, 0.12, 0.04, 1.0)}, {"Opacity": 0.035, "Strength": 0.06, "RimWeight": 0.06})
for name, p in (ARGS.get("paints") or {}).items():
    instance(paint, f"MI_FILM_{name}", {"Color": p["color"]}, {"Roughness": p.get("roughness", 0.45), "Metallic": p.get("metallic", 0.0)})
for name, p in (ARGS.get("glows") or {}).items():
    instance(glow, f"MI_FILM_{name}", {"Color": p["color"]}, {"Strength": p.get("strength", 20.0)})
# In-place copies of the forward-walk clips (root locked): the film moves the actor itself, at the
# clip's own stride speed, so feet don't slide and the mesh doesn't snap back at every loop.
ANIM_DIR = "/Game/VIVID2/Demo/Anims"
eal.make_directory(ANIM_DIR)
for src, name in (ARGS.get("inplace") or {}).items():
    dst = f"{ANIM_DIR}/{name}"
    if not eal.does_asset_exist(dst):
        if eal.duplicate_asset(src, dst) is None:
            raise RuntimeError(f"could not duplicate {src}")
        made.append(dst)
    anim = unreal.load_asset(dst)
    anim.set_editor_property("force_root_lock", True)
    anim.set_editor_property("enable_root_motion", False)
    eal.save_loaded_asset(anim, False)
print(json.dumps({"made": made, "dir": DIR}))
