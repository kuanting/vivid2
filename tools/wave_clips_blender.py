"""Author VIVID 2's waving clips in Blender: one arm, and both arms (waving for help).

Run headless with Blender 4.2+ (tested 5.1):

    blender --background --factory-startup --python tools/wave_clips_blender.py -- \\
        --skeleton <Ailive Manny FBX> --out unreal/VIVID2Env/SourceData/Hikers [--preview DIR]

``--skeleton`` is an FBX of the Lie Down Animation Pack (Ailive; it carries the UE5 Manny skeleton
the hiker skeleton plays the pose clips from), e.g. the launcher's Fab cache copy
``C:/ProgramData/Epic/EpicGamesLauncher/VaultCache/FabLibrary/Lie_Down_Animation_Pack-*/fbx/
lie_extracted/Sprawling_Out_on_the_Floor.FBX``. Only its bones are used: its mesh and its take are
dropped, and every key of the clips is made here (the body is the skeleton's reference pose, turned
to face +X as the pack's clips do; arms, clavicles, spine, neck, head and fingers are posed
procedurally), so the clips carry no pack motion.

Writes ``VIVID2_Wave_OneArm.fbx`` and ``VIVID2_Wave_TwoArms.fbx``: the skeleton, a 1 cm proxy mesh
skinned to the pelvis (Unreal imports animation through a skeletal mesh) and one looping take each.
``tools/build_hikers.py`` imports them onto the pack skeleton (SkeletalMeshTools.import_file).

  * OneArm: the right arm raised overhead, the hand swinging side to side (1 s per wave), palm
    forward; the left arm hangs, the head is tilted up; 4 s, 4 waves.
  * TwoArms: both arms overhead, swinging out to a V and crossing above the head (1.2 s per
    wave), palms forward, head up; 4.8 s, 4 waves.

Coordinates: Blender armature space of the imported FBX (cm; the armature object scales by 0.01).
The reference pose faces -Y; the clips turn the pelvis +90 deg about Z so the body faces +X, which
is Unreal's +X (the FBX axis conversion only flips Y), as in the pack's clips.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import bpy
from mathutils import Matrix, Vector

FPS = 30
F = Vector((1, 0, 0))   # forward (the body faces +X once the pelvis is turned)
U = Vector((0, 0, 1))   # up
L = Vector((0, 1, 0))   # the body's left
R = -L                  # the body's right


def rot(axis: Vector, deg: float) -> Matrix:
    return Matrix.Rotation(math.radians(deg), 4, axis.normalized())


def about(m: Matrix, pivot: Vector, delta: Matrix) -> Matrix:
    """Rotate an armature-space bone matrix by ``delta`` about ``pivot``."""
    return Matrix.Translation(pivot) @ delta @ Matrix.Translation(-pivot) @ m


def frame3(a: Vector, n: Vector) -> Matrix:
    """Orthonormal 3x3 frame from a primary axis and a (roughly) perpendicular second axis."""
    a = a.normalized()
    n = (n - a * n.dot(a)).normalized()
    return Matrix((a, n, a.cross(n))).transposed()


def signed_angle(v1: Vector, v2: Vector, axis: Vector) -> float:
    a = axis.normalized()
    p1 = (v1 - a * v1.dot(a)).normalized()
    p2 = (v2 - a * v2.dot(a)).normalized()
    return math.atan2(a.dot(p1.cross(p2)), max(-1.0, min(1.0, p1.dot(p2))))


class Rig:
    """The imported skeleton's reference pose, and forward kinematics with per-bone overrides."""

    def __init__(self, arm):
        self.arm = arm
        self.bones = arm.data.bones
        self.rest = {b.name: b.matrix_local.copy() for b in self.bones}
        self.order = []

        def walk(b):
            self.order.append(b.name)
            for c in b.children:
                walk(c)
        for b in self.bones:
            if b.parent is None:
                walk(b)
        self.parent = {b.name: (b.parent.name if b.parent else None) for b in self.bones}
        h = {b.name: b.head_local.copy() for b in self.bones}
        self.head = h
        # palm normals (rest): across the palm x along the hand, signed by the way the fingers curl
        self.palm = {}
        for s in ("l", "r"):
            hand = h[f"middle_01_{s}"] - h[f"hand_{s}"]
            across = h[f"index_01_{s}"] - h[f"pinky_01_{s}"]
            n = hand.cross(across).normalized()
            d1 = (h[f"index_02_{s}"] - h[f"index_01_{s}"]).normalized()
            d2 = (h[f"index_03_{s}"] - h[f"index_02_{s}"]).normalized()
            if n.dot(d2 - d1) < 0:
                n = -n
            self.palm[s] = n
        # rest bend of each finger joint (deg; towards the palm): the pose sets bends, not offsets
        self.bend = {}
        for name in self.rest:
            if is_finger_joint(name):
                parent = self.parent[name]
                self.bend[name] = math.degrees(self.ydir(parent).angle(self.ydir(name)))

    def ydir(self, name: str) -> Vector:
        return (self.rest[name].to_3x3() @ Vector((0, 1, 0))).normalized()  # Blender bones point along +Y

    def pose(self, controller) -> dict:
        """Armature-space pose matrices: each bone follows its parent unless ``controller`` overrides it."""
        pose = {}
        for name in self.order:
            p = self.parent[name]
            m = self.rest[name] if p is None else pose[p] @ self.rest[p].inverted() @ self.rest[name]
            pose[name] = controller(name, m, pose) or m
        return pose

    def key(self, pose: dict, frame: int) -> None:
        for name in self.order:
            p = self.parent[name]
            ref = self.rest[name] if p is None else pose[p] @ self.rest[p].inverted() @ self.rest[name]
            basis = ref.inverted() @ pose[name]
            pb = self.arm.pose.bones[name]
            pb.rotation_mode = "QUATERNION"
            loc, q, sc = basis.decompose()
            pb.location = loc
            pb.rotation_quaternion = q
            pb.scale = sc
            for path in ("location", "rotation_quaternion", "scale"):
                pb.keyframe_insert(path, frame=frame)


def is_finger_joint(name: str) -> bool:
    """index/middle/ring/pinky _01.._03 (the thumb keeps its reference pose)."""
    parts = name.split("_")
    return len(parts) == 3 and parts[0] in ("index", "middle", "ring", "pinky") and parts[1] in ("01", "02", "03")


def two_bone(s: Vector, w: Vector, l1: float, l2: float, pole: Vector) -> Vector:
    """Elbow position for shoulder ``s`` reaching wrist ``w``, bent towards ``pole``."""
    d = min((w - s).length, (l1 + l2) * 0.999)
    u = (w - s).normalized()
    cos_a = (l1 * l1 + d * d - l2 * l2) / (2 * l1 * d)
    sin_a = math.sqrt(max(0.0, 1 - cos_a * cos_a))
    v = (pole - u * pole.dot(u)).normalized()
    return s + l1 * (cos_a * u + sin_a * v)


def arm_dir(phi: float, theta: float, side: Vector) -> Vector:
    """Direction from the shoulder: ``phi`` deg from straight up towards ``side`` (negative: across
    the body), tilted ``theta`` deg forward."""
    p, t = math.radians(phi), math.radians(theta)
    v = math.cos(p) * U + math.sin(p) * side
    return (v + math.tan(t) * F).normalized()


class Pose:
    """One frame's pose targets."""

    def __init__(self):
        self.spine_pitch = 0.0       # deg per spine bone (03..05), + leans back
        self.spine_roll = 0.0        # deg per spine bone (03..05), + leans to the body's left
        self.head_pitch = 0.0        # deg in total over neck_01, neck_02, head; + looks up
        self.clavicle = {"l": 0.0, "r": 0.0}   # deg, + raises the shoulder
        self.reach = {}              # side -> (wrist direction, reach ratio, pole, palm direction, wrist flap deg)
        self.bend = {}               # side -> finger bends (knuckle, middle, tip), deg; unset: reference pose


def make_controller(rig: Rig, p: Pose):
    sides = {"l": L, "r": R}
    lengths = {}
    for s in ("l", "r"):
        h = rig.head
        a0 = h[f"lowerarm_{s}"] - h[f"upperarm_{s}"]
        b0 = h[f"hand_{s}"] - h[f"lowerarm_{s}"]
        lengths[s] = (a0, b0, a0.cross(b0).normalized())
    solved = {}

    def pitch_axis():
        return R  # rotating about the body's right by + tilts forward-facing things up

    def ctrl(name, m, pose):
        if name == "pelvis":
            return rot(U, 90.0) @ m
        if name in ("spine_03", "spine_04", "spine_05"):
            d = rot(pitch_axis(), p.spine_pitch) @ rot(F, -p.spine_roll)
            return about(m, m.translation, d)
        if name in ("neck_01", "neck_02", "head"):
            share = {"neck_01": 0.3, "neck_02": 0.3, "head": 0.4}[name]
            return about(m, m.translation, rot(pitch_axis(), p.head_pitch * share))
        for s in ("l", "r"):
            if name == f"clavicle_{s}":
                cdir = (pose_point(rig, pose, name, f"upperarm_{s}", m) - m.translation)
                axis = cdir.cross(U)
                return about(m, m.translation, rot(axis, p.clavicle[s]))
            if name == f"upperarm_{s}" and s in p.reach:
                wdir, ratio, pole, palm, flap = p.reach[s]
                a0, b0, n0 = lengths[s]
                l1, l2 = a0.length, b0.length
                sh = m.translation.copy()
                w = sh + wdir * (l1 + l2) * ratio
                e = two_bone(sh, w, l1, l2, pole)
                a1, b1 = e - sh, w - e
                n1 = a1.cross(b1).normalized()
                du = frame3(a1, n1) @ frame3(a0, n0).inverted()
                dl = frame3(b1, n1) @ frame3(b0, n0).inverted()
                # forearm twist that turns the palm to face ``palm``
                tw = signed_angle(dl @ rig.palm[s], palm, b1)
                dl = Matrix.Rotation(tw, 3, b1.normalized()) @ dl
                pn = dl @ rig.palm[s]
                dh = Matrix.Rotation(math.radians(flap), 3, pn) @ dl
                solved[s] = (du, dl, dh, e, w)
                return Matrix.Translation(sh) @ du.to_4x4() @ strip(m_rest(rig, name))
            if name == f"lowerarm_{s}" and s in solved:
                du, dl, dh, e, w = solved[s]
                return Matrix.Translation(m.translation) @ dl.to_4x4() @ strip(m_rest(rig, name))
            if name == f"hand_{s}" and s in solved:
                du, dl, dh, e, w = solved[s]
                return Matrix.Translation(m.translation) @ dh.to_4x4() @ strip(m_rest(rig, name))
            if name.endswith(f"_{s}") and is_finger_joint(name) and s in p.bend:
                target = p.bend[s][int(name.split("_")[1]) - 1]
                pn = rig.palm[s]
                parent_hand = pose[f"hand_{s}"]
                pn = (parent_hand.to_3x3() @ rig.rest[f"hand_{s}"].to_3x3().inverted() @ pn).normalized()
                d = (m.to_3x3() @ Vector((0, 1, 0))).normalized()  # Blender bones point along +Y
                # + turns the finger towards the palm
                return about(m, m.translation, rot(d.cross(pn), target - rig.bend[name]))
        return None

    return ctrl


def m_rest(rig: Rig, name: str) -> Matrix:
    return rig.rest[name]


def strip(m: Matrix) -> Matrix:
    """The rotation of a matrix as 4x4 (translation dropped)."""
    return m.to_3x3().to_4x4()


def pose_point(rig: Rig, pose: dict, parent: str, child: str, parent_m: Matrix) -> Vector:
    """Where ``child``'s head is when ``parent`` has matrix ``parent_m``."""
    return (parent_m @ rig.rest[parent].inverted() @ rig.rest[child]).translation


def one_arm(t: float) -> Pose:
    p = Pose()
    ph = 2 * math.pi * t / 1.0
    phi = 30.0 + 22.0 * math.sin(ph)
    p.reach["r"] = (arm_dir(phi, 12.0, R), 0.90, (R - 0.25 * F - 0.2 * U), F + 0.15 * R,
                    12.0 * math.sin(ph - 0.7))
    # left arm hangs by the side, slightly out and forward, palm to the thigh
    p.reach["l"] = ((-U + 0.10 * F + 0.16 * L).normalized(), 0.97, (-F + 0.3 * L), R, 0.0)
    p.bend = {"l": (32.0, 30.0, 14.0), "r": (8.0, 6.0, 4.0)}  # relaxed hand, open waving hand
    p.clavicle = {"l": 0.0, "r": 14.0 + 3.0 * math.sin(ph)}
    p.head_pitch = 18.0
    p.spine_pitch = 1.0
    p.spine_roll = 0.8 * math.sin(ph - 0.4)
    return p


def two_arms(t: float) -> Pose:
    p = Pose()
    ph = 2 * math.pi * t / 1.2
    phi = 16.0 + 26.0 * math.cos(ph)  # 42 deg out (a V) .. 10 deg across the head
    p.reach["r"] = (arm_dir(phi, 14.0, R), 0.95, (R - 0.3 * F - 0.1 * U), F, 10.0 * math.sin(ph))
    p.reach["l"] = (arm_dir(phi, 6.0, L), 0.95, (L - 0.3 * F - 0.1 * U), F, -10.0 * math.sin(ph))
    p.bend = {"l": (8.0, 6.0, 4.0), "r": (8.0, 6.0, 4.0)}
    cl = 18.0 + 4.0 * math.cos(ph)
    p.clavicle = {"l": cl, "r": cl}
    p.head_pitch = 24.0
    p.spine_pitch = 1.5
    return p


# name: (pose function, length s, wave period s)
CLIPS = {"VIVID2_Wave_OneArm": (one_arm, 4.0, 1.0), "VIVID2_Wave_TwoArms": (two_arms, 4.8, 1.2)}


def load_skeleton(fbx: str):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=fbx)
    arm = bpy.data.objects["root"]
    keep = None
    for o in list(bpy.data.objects):
        if o is arm:
            continue
        if o.type == "MESH" and o.name.endswith("LOD0"):
            keep = o  # for preview renders only; never exported
            continue
        bpy.data.objects.remove(o)
    arm.animation_data_clear()
    for a in list(bpy.data.actions):
        bpy.data.actions.remove(a)
    arm.location, arm.rotation_euler = (0, 0, 0), (0, 0, 0)
    return arm, keep


def add_proxy(arm, rig: Rig):
    """A 1 cm tetrahedron at the pelvis, skinned to it (Unreal imports animation through a mesh)."""
    c = rig.head["pelvis"]
    s = 0.5
    verts = [c + Vector(v) * s for v in ((1, 1, 1), (-1, -1, 1), (-1, 1, -1), (1, -1, -1))]
    me = bpy.data.meshes.new("VIVID2_WaveProxy")
    me.from_pydata([tuple(v) for v in verts], [], [(0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2)])
    ob = bpy.data.objects.new("VIVID2_WaveProxy", me)
    bpy.context.scene.collection.objects.link(ob)
    ob.parent = arm
    vg = ob.vertex_groups.new(name="pelvis")
    vg.add(list(range(4)), 1.0, "REPLACE")
    mod = ob.modifiers.new("Armature", "ARMATURE")
    mod.object = arm
    return ob


def bake(arm, rig: Rig, name: str, fn, seconds: float):
    act = bpy.data.actions.new(name)
    arm.animation_data_create()
    arm.animation_data.action = act
    sc = bpy.context.scene
    sc.render.fps, sc.render.fps_base = FPS, 1.0  # the source FBX sets 120
    n = int(round(seconds * FPS))
    for i in range(n + 1):  # frame i is at i / FPS s; the last frame equals the first: the take loops
        rig.key(rig.pose(make_controller(rig, fn(i / FPS))), i)
    sc.frame_start, sc.frame_end = 0, n
    return act


def export(arm, proxy, path: str):
    bpy.ops.object.select_all(action="DESELECT")
    arm.select_set(True)
    proxy.select_set(True)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.export_scene.fbx(
        filepath=path, use_selection=True, object_types={"ARMATURE", "MESH"},
        apply_unit_scale=True, apply_scale_options="FBX_SCALE_NONE", global_scale=1.0,
        axis_forward="-Z", axis_up="Y", primary_bone_axis="Y", secondary_bone_axis="X",
        add_leaf_bones=False, use_armature_deform_only=False, armature_nodetype="NULL",
        bake_anim=True, bake_anim_use_all_bones=True, bake_anim_use_nla_strips=False,
        bake_anim_use_all_actions=False, bake_anim_force_startend_keying=True, bake_anim_step=1.0,
        bake_anim_simplify_factor=0.0, use_mesh_modifiers=False, mesh_smooth_type="FACE")


def preview(arm, mesh, out_dir: str, name: str, frames):
    """Workbench renders (metres, world space): front, side, from above-front as a drone, hand close-up."""
    sc = bpy.context.scene
    try:
        sc.render.engine = "BLENDER_WORKBENCH"
    except TypeError:
        pass
    sc.display.shading.light = "STUDIO"
    sc.render.resolution_x, sc.render.resolution_y = 320, 420
    if sc.world is None:
        sc.world = bpy.data.worlds.new("w")
    cam = bpy.data.objects.get("cam")
    if cam is None:
        cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
        cam.data.type = "ORTHO"
        sc.collection.objects.link(cam)
    sc.camera = cam
    os.makedirs(out_dir, exist_ok=True)
    # (eye, target, ortho scale)
    views = {"front": ((5, 0, 1.1), (0, 0, 1.1), 2.4), "side": ((0, -5, 1.1), (0, 0, 1.1), 2.4),
             "drone": ((3.5, -1.5, 4.5), (0, 0, 1.0), 2.6), "hand": ((4, -1.2, 2.0), (0, -0.3, 1.9), 0.8)}
    for f in frames:
        sc.frame_set(f)
        for vn, (eye, target, scale) in views.items():
            cam.location = Vector(eye)
            cam.rotation_euler = (Vector(target) - Vector(eye)).to_track_quat("-Z", "Y").to_euler()
            cam.data.ortho_scale = scale
            sc.render.filepath = os.path.join(out_dir, f"{name}_{f:04d}_{vn}.png")
            bpy.ops.render.render(write_still=True)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--skeleton", required=True, help="an Ailive animation pack FBX (UE5 Manny skeleton)")
    ap.add_argument("--out", required=True, help="folder for the clip FBX files")
    ap.add_argument("--preview", help="also render preview frames into this folder")
    a = ap.parse_args(argv)
    arm, mesh = load_skeleton(a.skeleton)
    rig = Rig(arm)
    proxy = add_proxy(arm, rig)
    os.makedirs(a.out, exist_ok=True)
    for name, (fn, seconds, period) in CLIPS.items():
        bake(arm, rig, name, fn, seconds)
        path = os.path.join(a.out, f"{name}.fbx")
        export(arm, proxy, path)
        print(f"WROTE {path}")
        if a.preview:  # a quarter of a wave apart
            preview(arm, mesh, a.preview, name, [round(k * period * FPS / 4) for k in range(4)])


if __name__ == "__main__":
    main(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
