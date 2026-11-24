"""Model VIVID 2's ski gear clues in Blender: a pair of skis, a snowboard and a touring backpack.

Run headless with Blender 4.2+ (tested 5.1):

    blender --background --factory-startup --python tools/ski_gear_blender.py -- \\
        --out unreal/VIVID2Env/SourceData/Clues [--preview DIR]

Writes one static mesh FBX per item: ``VIVID2_Clue_Skis.fbx``, ``VIVID2_Clue_Snowboard.fbx`` and
``VIVID2_Clue_Backpack.fbx``. Each lies as it would after an avalanche tore it from its owner: flat on
the snow, its lowest point at z = 0 (the pivot, which the runner puts on the ground), centred on the
pivot, its length along +X. Units are metres (Unreal gets centimetres). The meshes carry no colours,
only material slots by role, which ``tools/build_avalanche.py`` fills with the decoys' colour
materials (MI_VIVID2_Decoy_<Colour>), so one mesh makes several coloured clues:

    Skis       Topsheet, Base, Binding   two 1.75 m skis splayed in a V, toe and heel pieces
    Snowboard  Topsheet, Base, Binding   a 1.56 m board, two strap bindings with high-backs
    Backpack   Pack, Trim, Straps        a 35 l touring pack lying on its back panel, lid pocket,
                                         ski-carry and compression straps, hip-belt wings

Sizes are chosen to be seen from survey height: at 50 m above the ground the Payload camera has about
4.4 cm pixels, and a clue counts as visible from 80 pixels (docs/damage.md). One ski alone is about 80,
so the skis come as a pair; a pole or a glove would be a few pixels and is not made.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import bmesh
import bpy
from mathutils import Matrix, Vector


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for block in (bpy.data.meshes, bpy.data.materials):
        for item in list(block):
            block.remove(item)


def material(name: str, rgb) -> bpy.types.Material:
    """A slot material; the colour is only for the previews (Unreal assigns its own)."""
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.diffuse_color = (*rgb, 1.0)
    return mat


def new_object(name: str, mats) -> tuple:
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    for m in mats:
        mesh.materials.append(m)
    return obj, bmesh.new()


def finish(obj, bm) -> None:
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(obj.data)
    bm.free()
    for p in obj.data.polygons:
        p.use_smooth = False


def board(bm, length, half_width, thickness, lift, mat_top, mat_base, stations=64, across=4,
          matrix=Matrix.Identity(4)):
    """A ski or snowboard: a plank along X centred on 0, its outline ``half_width(t)``, its base
    raised ``lift(t)`` and ``thickness(t)`` thick (t = 0 at the tail .. 1 at the tip).
    The top faces get ``mat_top``, the base and the edges ``mat_base``."""
    rings = []
    for k in range(stations + 1):
        t = k / stations
        x = (t - 0.5) * length
        w, z0, th = half_width(t), lift(t), thickness(t)
        ring = []
        for a in range(across + 1):  # top, from -w to +w
            ring.append(bm.verts.new(matrix @ Vector((x, -w + 2 * w * a / across, z0 + th))))
        for a in range(across + 1):  # base, from +w to -w
            ring.append(bm.verts.new(matrix @ Vector((x, w - 2 * w * a / across, z0))))
        rings.append(ring)
    n = len(rings[0])
    for k in range(stations):
        a, b = rings[k], rings[k + 1]
        for i in range(n):
            j = (i + 1) % n
            f = bm.faces.new((a[i], a[j], b[j], b[i]))
            f.material_index = mat_top if i < across else mat_base
    for ring in (rings[0], rings[-1]):
        f = bm.faces.new(ring)
        f.material_index = mat_base


def box(bm, centre, size, mat, matrix=Matrix.Identity(4), yaw_deg=0.0, pitch_deg=0.0) -> None:
    rot = Matrix.Rotation(math.radians(yaw_deg), 4, "Z") @ Matrix.Rotation(math.radians(pitch_deg), 4, "Y")
    m = matrix @ Matrix.Translation(Vector(centre)) @ rot @ Matrix.Diagonal((*size, 1.0))
    res = bmesh.ops.create_cube(bm, size=1.0, matrix=m)
    for f in {f for v in res["verts"] for f in v.link_faces}:
        f.material_index = mat


# -- the items ------------------------------------------------------------------------------------

SKI_L = 1.75


def ski_outline(t: float) -> float:
    """Half width: tail 5.75 cm, waist 4.5 cm, shovel 6.5 cm, rounded ends."""
    if t < 0.5:
        w = 0.045 + (0.0575 - 0.045) * ((0.5 - t) / 0.45) ** 2
    else:
        w = 0.045 + (0.065 - 0.045) * min(1.0, ((t - 0.5) / 0.38)) ** 2
    end = min(t, 1 - t)
    if end < 0.035:  # round the tip and the tail
        w *= math.sqrt(max(0.0, 1 - ((0.035 - end) / 0.035) ** 2)) * 0.85 + 0.15
    return w


def ski_lift(t: float) -> float:
    if t > 0.86:   # tip rocker, 6 cm
        return 0.06 * ((t - 0.86) / 0.14) ** 2
    if t < 0.07:   # tail, 2.5 cm
        return 0.025 * ((0.07 - t) / 0.07) ** 2
    return 0.0


def ski_thickness(t: float) -> float:
    return 0.009 + 0.009 * math.sin(math.pi * min(max(t, 0.0), 1.0))


def binding(bm, matrix, mats) -> None:
    top = 0.018
    box(bm, (0.12, 0, top + 0.025), (0.09, 0.075, 0.05), mats, matrix)        # toe piece
    box(bm, (-0.19, 0, top + 0.03), (0.12, 0.075, 0.06), mats, matrix)        # heel piece
    box(bm, (-0.035, 0, top + 0.006), (0.30, 0.06, 0.012), mats, matrix)      # plate
    for s in (-1, 1):                                                        # brake arms
        box(bm, (-0.26, s * 0.052, top - 0.004), (0.10, 0.008, 0.012), mats, matrix, yaw_deg=0, pitch_deg=-15)


def make_skis():
    mats = [material("Topsheet", (0.9, 0.35, 0.05)), material("Base", (0.03, 0.03, 0.03)),
            material("Binding", (0.25, 0.25, 0.27))]
    obj, bm = new_object("VIVID2_Clue_Skis", mats)
    for yaw, dy, dx in ((6.0, 0.16, 0.0), (-6.0, -0.16, 0.06)):  # splayed in a V, as they came to rest
        mtx = Matrix.Translation((dx, dy, 0.0)) @ Matrix.Rotation(math.radians(yaw), 4, "Z")
        board(bm, SKI_L, ski_outline, ski_thickness, ski_lift, 0, 1, matrix=mtx)
        binding(bm, mtx, 2)
    finish(obj, bm)
    return obj


def make_snowboard():
    mats = [material("Topsheet", (0.1, 0.3, 0.85)), material("Base", (0.03, 0.03, 0.03)),
            material("Binding", (0.05, 0.05, 0.05))]
    obj, bm = new_object("VIVID2_Clue_Snowboard", mats)
    L = 1.56

    def outline(t):
        w = 0.1225 + 0.022 * (abs(t - 0.5) / 0.5) ** 2   # waist 24.5 cm, nose and tail 29 cm
        end = min(t, 1 - t)
        if end < 0.06:
            w *= math.sqrt(max(0.0, 1 - ((0.06 - end) / 0.06) ** 2)) * 0.75 + 0.25
        return w

    def lift(t):
        end = min(t, 1 - t)
        return 0.05 * ((0.12 - end) / 0.12) ** 2 if end < 0.12 else 0.0

    board(bm, L, outline, lambda t: 0.014, lift, 0, 1, stations=72, across=6)
    for x0, ang in ((0.27, 15.0), (-0.27, -6.0)):   # duck stance
        m = Matrix.Translation((x0, 0, 0.014)) @ Matrix.Rotation(math.radians(90 + ang), 4, "Z")
        box(bm, (0, 0, 0.008), (0.26, 0.11, 0.016), 2, m)          # base plate
        box(bm, (-0.075, 0, 0.09), (0.03, 0.12, 0.16), 2, m, pitch_deg=-18)   # high-back
        for dx in (0.03, 0.10):                                     # straps
            box(bm, (dx, 0, 0.07), (0.035, 0.13, 0.03), 2, m)
    finish(obj, bm)
    return obj


def make_backpack():
    mats = [material("Pack", (0.75, 0.05, 0.05)), material("Trim", (0.05, 0.05, 0.06)),
            material("Straps", (0.02, 0.02, 0.02))]
    obj, bm = new_object("VIVID2_Clue_Backpack", mats)
    # body: 56 cm long, 30 cm wide, 20 cm deep, lying on its back panel; lid pocket at the top end (+X)
    for centre, size, radius, segs, mat in (((0, 0, 0.10), (0.56, 0.30, 0.20), 0.075, 4, 0),
                                            ((0.22, 0, 0.17), (0.18, 0.28, 0.08), 0.03, 2, 1)):
        res = bmesh.ops.create_cube(bm, size=1.0, matrix=Matrix.Translation(centre) @ Matrix.Diagonal((*size, 1)))
        verts = set(res["verts"])
        for f in {f for v in verts for f in v.link_faces}:
            f.material_index = mat
        bmesh.ops.bevel(bm, geom=list(verts) + [e for e in bm.edges if set(e.verts) <= verts], offset=radius,
                        segments=segs, affect="EDGES", profile=0.5, material=mat)
    for v in bm.verts:  # narrower towards the bottom end (-X), as a pack is
        v.co.y *= 1.0 - 0.18 * max(0.0, -v.co.x) / 0.28
    # front strap panel, compression and ski-carry straps across the front (top) face
    box(bm, (-0.03, 0, 0.202), (0.30, 0.20, 0.008), 1)
    for x in (-0.16, 0.02):
        box(bm, (x, 0, 0.206), (0.025, 0.31, 0.01), 2)
    for y in (-0.13, 0.13):
        box(bm, (-0.05, y, 0.17), (0.40, 0.022, 0.012), 2)
    # hip-belt wings showing at the sides of the back panel
    for s in (-1, 1):
        box(bm, (-0.19, s * 0.155, 0.03), (0.11, 0.07, 0.035), 2, yaw_deg=s * 10)
    finish(obj, bm)
    return obj


ITEMS = {"VIVID2_Clue_Skis": make_skis, "VIVID2_Clue_Snowboard": make_snowboard,
         "VIVID2_Clue_Backpack": make_backpack}


def ground_and_centre(obj) -> None:
    """Pivot at the bottom centre of the item's footprint."""
    xs = [v.co.x for v in obj.data.vertices]
    ys = [v.co.y for v in obj.data.vertices]
    zs = [v.co.z for v in obj.data.vertices]
    shift = Vector(((max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2, min(zs)))
    for v in obj.data.vertices:
        v.co -= shift


def export(obj, out_dir: str) -> str:
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    path = os.path.join(out_dir, obj.name + ".fbx")
    bpy.ops.export_scene.fbx(
        filepath=path, use_selection=True, object_types={"MESH"}, apply_unit_scale=True,
        apply_scale_options="FBX_SCALE_NONE", global_scale=1.0, axis_forward="-Z", axis_up="Y",
        use_mesh_modifiers=False, mesh_smooth_type="FACE", use_tspace=False, add_leaf_bones=False,
        bake_anim=False)
    return path


def preview(objs, out_dir: str) -> None:
    """Workbench renders: all items from above (as the drone sees them) and at three-quarters."""
    sc = bpy.context.scene
    sc.render.engine = "BLENDER_WORKBENCH"
    sc.display.shading.color_type = "MATERIAL"
    sc.display.shading.light = "STUDIO"
    sc.render.resolution_x, sc.render.resolution_y = 1200, 700
    sc.world = sc.world or bpy.data.worlds.new("World")
    sc.world.color = (0.92, 0.93, 0.96)  # snow
    for k, o in enumerate(objs):
        o.location = (0, (k - 1) * 0.9, 0)
    cam_data = bpy.data.cameras.new("cam")
    cam = bpy.data.objects.new("cam", cam_data)
    bpy.context.collection.objects.link(cam)
    sc.camera = cam
    for name, loc, rot, ortho in (("top", (0, 0, 5), (0, 0, 0), 3.0),
                                  ("threequarter", (2.4, -2.6, 1.9), (math.radians(62), 0, math.radians(43)), None)):
        cam.location = loc
        cam.rotation_euler = rot
        if ortho:
            cam_data.type, cam_data.ortho_scale = "ORTHO", ortho
        else:
            cam_data.type, cam_data.lens = "PERSP", 35
        sc.render.filepath = os.path.join(out_dir, f"ski_gear_{name}.png")
        bpy.ops.render.render(write_still=True)


def main(argv) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--preview")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    clear_scene()
    bpy.context.scene.unit_settings.system = "METRIC"
    bpy.context.scene.unit_settings.scale_length = 1.0
    objs = []
    for name, make in ITEMS.items():
        obj = make()
        ground_and_centre(obj)
        dims = obj.dimensions
        print(f"{name}: {dims.x:.3f} x {dims.y:.3f} x {dims.z:.3f} m, {len(obj.data.polygons)} faces -> "
              f"{export(obj, args.out)}")
        objs.append(obj)
    if args.preview:
        os.makedirs(args.preview, exist_ok=True)
        preview(objs, args.preview)


if __name__ == "__main__":
    main(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
