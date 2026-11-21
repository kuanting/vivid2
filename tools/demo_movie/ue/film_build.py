"""Editor side: build the film level and one Level Sequence per shot from a plan.json.

Runs inside the Unreal editor (stdlib + ``unreal``), via ``film.py ue-build`` (Python remote execution).
Everything it makes lives under ``/Game/VIVID2/Demo`` (never cooked, git-ignored):

* the film level: a persistent level with the benchmark map streamed in (always loaded), so the
  benchmark map itself is never modified or saved; its world settings use GameModeBase, so Project
  AirSim's game mode (which launches the simulation) is not used when filming;
* film actors (tag ``VIVID2Film``): hikers and fires (the benchmark blueprints), quadcopters made of
  the Project AirSim drone meshes with spinning propellers, cine cameras, helper meshes;
* ``LS_<shot>``: per-shot sequences with possessable bindings (transform keys, skeletal animation
  sections, camera lens keys, visibility), a camera cut, and the sun's date and solar time.

Plan units are Unreal's: centimetres, rotations as [roll, pitch, yaw] degrees, frames at plan["fps"].
Prints one JSON line (what was built) at the end.
"""
import json
import sys
from pathlib import Path

import unreal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ue_exec import script_args  # noqa: E402

TAG = "VIVID2Film"
ARGS = script_args()
PLAN = json.loads(Path(ARGS["plan"]).read_text(encoding="utf-8"))
ONLY = set(ARGS.get("shots") or [])
FPS = int(PLAN.get("fps", 30))

les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
ues = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem)
report = {"actors": [], "sequences": [], "warnings": []}

INTERP = {"auto": unreal.MovieSceneKeyInterpolation.AUTO, "linear": unreal.MovieSceneKeyInterpolation.LINEAR,
          "constant": unreal.MovieSceneKeyInterpolation.CONSTANT}


def warn(msg):
    report["warnings"].append(msg)
    unreal.log_warning(f"[film_build] {msg}")


# -- level -----------------------------------------------------------------------------------

def package_of(world):
    return world.get_outermost().get_name()


def ensure_film_level():
    film, base = PLAN["level"]["film"], PLAN["level"]["base"]
    world = ues.get_editor_world()
    if package_of(world) != film:
        if unreal.EditorAssetLibrary.does_asset_exist(film):
            if not les.load_level(film):
                raise RuntimeError(f"could not load {film} (is the current level unsaved?)")
        else:
            if not les.new_level(film):
                raise RuntimeError(f"could not create {film}")
            world = ues.get_editor_world()
            streamed = unreal.EditorLevelUtils.add_level_to_world(world, base, unreal.LevelStreamingAlwaysLoaded)
            if streamed is None:
                raise RuntimeError(f"could not stream {base} into {film}")
            ws = world.get_world_settings()
            ws.set_editor_property("default_game_mode", unreal.GameModeBase.static_class())
            save_film_level(world)
    world = ues.get_editor_world()
    if package_of(world) != film:
        raise RuntimeError(f"film level not open: {package_of(world)}")
    # new actors go into the persistent (film) level, never into the streamed benchmark map
    if not les.set_current_level_by_name(film.rsplit("/", 1)[-1]):
        raise RuntimeError("could not make the film level current")
    return world


def save_film_level(world):
    """Save the film map package only. The streamed benchmark map must never be saved (if anything
    marked it dirty, it is reported and left unsaved)."""
    pkg = world.get_outermost()
    if not unreal.EditorLoadingAndSavingUtils.save_packages([pkg], False):
        raise RuntimeError("could not save the film level")
    for p in unreal.EditorLoadingAndSavingUtils.get_dirty_map_packages():
        if p.get_name() != PLAN["level"]["film"]:
            warn(f"{p.get_name()} is dirty in the editor (not saved; do not save it)")


# -- actors ----------------------------------------------------------------------------------

def vec(v):
    return unreal.Vector(float(v[0]), float(v[1]), float(v[2]))


def rot(r):
    r = r or [0, 0, 0]
    return unreal.Rotator(roll=float(r[0]), pitch=float(r[1]), yaw=float(r[2]))


def tag(actor, label):
    actor.set_actor_label(label)
    actor.set_editor_property("tags", [unreal.Name(TAG)])
    actor.set_folder_path(unreal.Name("VIVID2Film"))


def clear_film_actors():
    for a in eas.get_all_level_actors():
        if a.actor_has_tag(TAG):
            a.destroy_actor()


def mesh_actor(label, mesh_path, loc, r, scale=None, material=None, collide=False):
    a = eas.spawn_actor_from_class(unreal.StaticMeshActor, vec(loc), rot(r))
    smc = a.static_mesh_component
    smc.set_mobility(unreal.ComponentMobility.MOVABLE)
    smc.set_static_mesh(unreal.load_asset(mesh_path))
    if scale:
        a.set_actor_scale3d(vec(scale))
    if material:
        smc.set_material(0, unreal.load_asset(material))
    if not collide:
        smc.set_collision_enabled(unreal.CollisionEnabled.NO_COLLISION)
    tag(a, label)
    return a


def attach(child, parent, rel_loc, rel_rot=None):
    child.attach_to_actor(parent, unreal.Name(""), unreal.AttachmentRule.KEEP_WORLD,
                          unreal.AttachmentRule.KEEP_WORLD, unreal.AttachmentRule.KEEP_WORLD, False)
    child.root_component.set_relative_location(vec(rel_loc), False, False)
    child.root_component.set_relative_rotation(rot(rel_rot), False, False)


def quad(spec):
    """A quadcopter from the Project AirSim drone meshes (robot_vivid_quad.jsonc): frame + 4 props,
    optional film paint (``frame_materials`` per slot, ``prop_materials`` front/rear) and navigation
    lights (small glowing beads, red left / green right, plus a weak light)."""
    name = spec["name"]
    body = mesh_actor(name, "/Drone/Quadrotor1", spec["location"], spec.get("rotation"))
    for i, mat in enumerate(spec.get("frame_materials") or []):
        if mat:
            body.static_mesh_component.set_material(i, unreal.load_asset(mat))
    parts = {"body": body}
    props_mat = spec.get("prop_materials") or {}
    for prop, (x, y), mesh, end in (("FL", (25.3, -25.3), "/Drone/PropellerRed", "front"),
                                    ("FR", (25.3, 25.3), "/Drone/PropellerRed", "front"),
                                    ("RL", (-25.3, -25.3), "/Drone/PropellerWhite", "rear"),
                                    ("RR", (-25.3, 25.3), "/Drone/PropellerWhite", "rear")):
        p = mesh_actor(f"{name}_Prop{prop}", mesh, spec["location"], spec.get("rotation"),
                       material=props_mat.get(end))
        attach(p, body, (x, y, 1.0))
        parts[prop] = p
    nav = spec.get("nav") or {}
    for side, y, glow, color in (("NavL", -1.0, nav.get("left"), (1.0, 0.05, 0.02)),
                                 ("NavR", 1.0, nav.get("right"), (0.05, 1.0, 0.1))):
        if not glow:
            continue
        r = float(nav.get("radius_cm", 46.0))
        bead = mesh_actor(f"{name}_{side}", "/Engine/BasicShapes/Sphere", spec["location"], None,
                          scale=[nav.get("bead_scale", 0.035)] * 3, material=glow)
        attach(bead, body, (0.0, y * r, 1.5))
        parts[side] = bead
        if nav.get("light_intensity", 0) > 0:
            lt = eas.spawn_actor_from_class(unreal.PointLight, vec(spec["location"]), rot(None))
            lc = lt.get_component_by_class(unreal.PointLightComponent)
            lc.set_mobility(unreal.ComponentMobility.MOVABLE)
            lc.set_light_color(unreal.LinearColor(color[0], color[1], color[2], 1.0))
            lc.set_intensity(float(nav["light_intensity"]))
            lc.set_attenuation_radius(float(nav.get("light_radius_cm", 80.0)))
            lc.set_cast_shadows(False)
            tag(lt, f"{name}_{side}Light")
            attach(lt, body, (0.0, y * r, 4.0))
    return parts


def camera(spec):
    a = eas.spawn_actor_from_class(unreal.CineCameraActor, vec(spec["location"]), rot(spec.get("rotation")))
    tag(a, spec["name"])
    cc = a.get_cine_camera_component()
    fb = cc.get_editor_property("filmback")
    fb.set_editor_property("sensor_width", float(spec.get("sensor_mm", [36.0, 20.25])[0]))
    fb.set_editor_property("sensor_height", float(spec.get("sensor_mm", [36.0, 20.25])[1]))
    cc.set_editor_property("filmback", fb)
    lens = cc.get_editor_property("lens_settings")
    lens.set_editor_property("min_focal_length", 4.0)
    lens.set_editor_property("max_focal_length", 1200.0)
    lens.set_editor_property("min_f_stop", 1.2)
    lens.set_editor_property("max_f_stop", 32.0)
    cc.set_editor_property("lens_settings", lens)
    cc.set_editor_property("current_focal_length", float(spec.get("focal_mm", 35.0)))
    cc.set_editor_property("current_aperture", float(spec.get("aperture", 8.0)))
    fs = cc.get_editor_property("focus_settings")
    if spec.get("focus_cm"):
        fs.set_editor_property("focus_method", unreal.CameraFocusMethod.MANUAL)
        fs.set_editor_property("manual_focus_distance", float(spec["focus_cm"]))
    else:
        fs.set_editor_property("focus_method", unreal.CameraFocusMethod.DISABLE)
    cc.set_editor_property("focus_settings", fs)
    exposure(cc, spec.get("exposure") or {})
    return {"body": a, "component": cc}


def exposure(cc, ex):
    """Camera exposure. Default: the same histogram auto exposure as the Project AirSim cameras."""
    pps = cc.get_editor_property("post_process_settings")
    method = ex.get("method", "histogram")
    pps.set_editor_property("override_auto_exposure_method", True)
    if method == "manual":
        pps.set_editor_property("auto_exposure_method", unreal.AutoExposureMethod.AEM_MANUAL)
        pps.set_editor_property("override_auto_exposure_apply_physical_camera_exposure", True)
        pps.set_editor_property("auto_exposure_apply_physical_camera_exposure", False)
    else:
        pps.set_editor_property("auto_exposure_method", unreal.AutoExposureMethod.AEM_HISTOGRAM)
        for k, v in (("auto_exposure_min_brightness", ex.get("min", 0.0005)), ("auto_exposure_max_brightness", ex.get("max", 2.0)),
                     ("histogram_log_min", ex.get("log_min", -14.0)), ("histogram_log_max", ex.get("log_max", 6.0)),
                     ("auto_exposure_speed_up", ex.get("speed", 10.0)), ("auto_exposure_speed_down", ex.get("speed", 10.0))):
            pps.set_editor_property("override_" + k, True)
            pps.set_editor_property(k, float(v))
    pps.set_editor_property("override_auto_exposure_bias", True)
    pps.set_editor_property("auto_exposure_bias", float(ex.get("bias", 0.0)))
    for k in ("motion_blur_amount", "motion_blur_max"):
        if k in ex:
            pps.set_editor_property("override_" + k, True)
            pps.set_editor_property(k, float(ex[k]))
    cc.set_editor_property("post_process_settings", pps)


def spawn_all(world):
    actors = {}
    for spec in PLAN["actors"]:
        kind = spec.get("kind", "bp")
        if kind == "bp":
            cls = unreal.EditorAssetLibrary.load_blueprint_class(spec["class"])
            if cls is None:
                raise RuntimeError(f"no blueprint {spec['class']}")
            a = eas.spawn_actor_from_class(cls, vec(spec["location"]), rot(spec.get("rotation")))
            tag(a, spec["name"])
            actors[spec["name"]] = {"body": a}
        elif kind == "mesh":
            a = mesh_actor(spec["name"], spec["mesh"], spec["location"], spec.get("rotation"), spec.get("scale"),
                           spec.get("material"))
            actors[spec["name"]] = {"body": a}
        elif kind == "quad":
            actors[spec["name"]] = quad(spec)
        elif kind == "camera":
            actors[spec["name"]] = camera(spec)
        else:
            raise RuntimeError(f"unknown actor kind {kind}")
        a = actors[spec["name"]]["body"]
        if spec.get("hidden"):
            a.set_actor_hidden_in_game(True)
        if a.get_level().get_outermost().get_name() != PLAN["level"]["film"]:
            raise RuntimeError(f"{spec['name']} was spawned outside the film level "
                               f"({a.get_level().get_outermost().get_name()})")
        report["actors"].append(spec["name"])
    return actors


# -- sequences -------------------------------------------------------------------------------

def frame(f):
    return unreal.FrameNumber(int(round(f)))


def key(ch, f, value, interp="auto"):
    ch.add_key(frame(f), float(value), 0.0, unreal.MovieSceneTimeUnit.DISPLAY_RATE, INTERP[interp])


# Sections reach past the shot on both sides: Movie Render Queue evaluates warm-up frames before frame 0
# and temporal sub-samples just outside the shot; without a camera cut there it renders the default
# pawn's view ("No camera actor found on Eval Tick"). Keys stay inside [0, n]; outside them every
# channel holds its first or last value.
PRE_ROLL, POST_ROLL = 200, 30


def section_for(track, n):
    s = track.add_section()
    s.set_range(-PRE_ROLL, n + POST_ROLL)
    return s


def channels(section):
    return {str(c.channel_name): c for c in section.get_all_channels()}


def key_transform(binding, tk, n):
    """tk: {"frames": [...], "loc": [[x, y, z]...], "rot": [[r, p, y]...], "interp": "auto"|"linear"}"""
    tr = binding.add_track(unreal.MovieScene3DTransformTrack)
    ch = channels(section_for(tr, n))
    interp = tk.get("interp", "auto")
    names = (("Location.X", "loc", 0), ("Location.Y", "loc", 1), ("Location.Z", "loc", 2),
             ("Rotation.X", "rot", 0), ("Rotation.Y", "rot", 1), ("Rotation.Z", "rot", 2))
    missing = [nm for nm, _, _ in names if nm not in ch]
    if missing:
        raise RuntimeError(f"transform channels missing {missing}; have {sorted(ch)}")
    for i, f in enumerate(tk["frames"]):
        for nm, field, j in names:
            if field in tk:
                key(ch[nm], f, tk[field][i][j], interp)


def key_float(binding, prop_name, prop_path, keys, n):
    t = binding.add_track(unreal.MovieSceneFloatTrack)
    t.set_property_name_and_path(prop_name, prop_path)
    ch = section_for(t, n).get_all_channels()[0]
    for f, v in keys:
        key(ch, f, v)


def key_double(binding, prop_name, keys, n):
    t = binding.add_track(unreal.MovieSceneDoubleTrack)
    t.set_property_name_and_path(prop_name, prop_name)
    ch = section_for(t, n).get_all_channels()[0]
    for f, v in keys:
        key(ch, f, v)


def key_int(binding, prop_name, value, n):
    t = binding.add_track(unreal.MovieSceneIntegerTrack)
    t.set_property_name_and_path(prop_name, prop_name)
    ch = section_for(t, n).get_all_channels()[0]
    ch.add_key(frame(0), int(value), 0.0, unreal.MovieSceneTimeUnit.DISPLAY_RATE)


def add_anims(seq, binding, anims, n):
    tick = seq.get_tick_resolution()
    tick_fps = tick.numerator / tick.denominator
    t = binding.add_track(unreal.MovieSceneSkeletalAnimationTrack)
    for a in anims:
        s = t.add_section()
        s.set_range(int(a["start"]), int(a["end"]))
        p = s.get_editor_property("params")
        clip = unreal.load_asset(a["clip"])
        if clip is None:
            raise RuntimeError(f"no animation {a['clip']}")
        p.set_editor_property("animation", clip)
        p.set_editor_property("start_frame_offset", frame(a.get("offset_s", 0.0) * tick_fps))
        if a.get("end_offset_s") is not None:
            p.set_editor_property("end_frame_offset", frame(a["end_offset_s"] * tick_fps))
        p.set_editor_property("force_custom_mode", bool(a.get("force_custom_mode", True)))
        s.set_editor_property("params", p)


def reset_sequence(seq):
    for b in list(seq.get_bindings()):
        b.remove()
    for t in list(seq.get_tracks()):
        seq.remove_track(t)


def sequence_for(shot):
    name = f"LS_{shot['id']}"
    folder = PLAN["seq_dir"]
    path = f"{folder}/{name}"
    if unreal.EditorAssetLibrary.does_asset_exist(path):
        seq = unreal.load_asset(path)
        reset_sequence(seq)
    else:
        seq = unreal.AssetToolsHelpers.get_asset_tools().create_asset(name, folder, unreal.LevelSequence,
                                                                     unreal.LevelSequenceFactoryNew())
    seq.set_display_rate(unreal.FrameRate(FPS, 1))
    seq.set_playback_start(0)
    seq.set_playback_end(int(shot["frames"]))
    seq.set_work_range_start(-PRE_ROLL / FPS)
    seq.set_work_range_end((int(shot["frames"]) + POST_ROLL) / FPS)
    return seq, path


def find_sunsky():
    for a in eas.get_all_level_actors():
        if a.get_actor_label() == "SunSky" or a.get_class().get_name().startswith("SunSky"):
            return a
    return None


def build_shot(shot, actors):
    seq, path = sequence_for(shot)
    n = int(shot["frames"])
    # camera
    cam = actors[shot["camera"]["actor"]]
    cb = seq.add_possessable(cam["body"])
    key_transform(cb, shot["camera"]["transform"], n)
    lens = shot["camera"].get("lens") or {}
    if lens:
        ccb = seq.add_possessable(cam["component"])
        if "focal_mm" in lens:
            key_float(ccb, "CurrentFocalLength", "CurrentFocalLength", lens["focal_mm"], n)
        if "aperture" in lens:
            key_float(ccb, "CurrentAperture", "CurrentAperture", lens["aperture"], n)
        if "focus_cm" in lens:
            key_float(ccb, "ManualFocusDistance", "FocusSettings.ManualFocusDistance", lens["focus_cm"], n)
    cut = seq.add_track(unreal.MovieSceneCameraCutTrack)
    cs = section_for(cut, n)
    try:
        bid = seq.get_portable_binding_id(seq, cb)
    except Exception:  # noqa: BLE001 - older API
        bid = seq.get_binding_id(cb)
    cs.set_camera_binding_id(bid)
    # sun
    sun = shot.get("sun")
    if sun:
        sky = find_sunsky()
        if sky is None:
            warn("no SunSky actor: sun not keyed")
        else:
            sb = seq.add_possessable(sky)
            key_int(sb, "Month", sun["month"], n)
            key_int(sb, "Day", sun["day"], n)
            key_double(sb, "SolarTime", [(0, sun["solar_time"])] + ([(n, sun["solar_time_end"])] if "solar_time_end" in sun else []), n)
    # actors
    for tr in shot.get("tracks", []):
        parts = actors[tr["actor"]]
        b = seq.add_possessable(parts["body"])
        if "transform" in tr:
            key_transform(b, tr["transform"], n)
        if tr.get("anims"):
            add_anims(seq, b, tr["anims"], n)
        if "visible" in tr:
            vt = b.add_track(unreal.MovieSceneVisibilityTrack)
            vt.set_property_name_and_path("bHiddenInGame", "bHiddenInGame")
            ch = section_for(vt, n).get_all_channels()[0]
            ch.add_key(frame(0), bool(tr["visible"]), 0.0, unreal.MovieSceneTimeUnit.DISPLAY_RATE)
        if tr.get("props_deg_per_frame"):
            rate = float(tr["props_deg_per_frame"])
            for prop, sign in (("FL", 1.0), ("FR", -1.0), ("RL", -1.0), ("RR", 1.0)):
                pa = parts.get(prop)
                if pa is None:
                    continue
                pb = seq.add_possessable(pa)
                rel = pa.root_component.get_editor_property("relative_location")
                key_transform(pb, {"frames": [0, n], "interp": "linear",
                                   "loc": [[rel.x, rel.y, rel.z]] * 2,
                                   "rot": [[0, 0, 0], [0, 0, sign * rate * n]]}, n)
    unreal.EditorAssetLibrary.save_loaded_asset(seq, False)
    report["sequences"].append(path)


def main():
    if les.is_in_play_in_editor():
        raise RuntimeError("stop Play-In-Editor first")
    world = ensure_film_level()
    if not ARGS.get("keep_actors"):
        clear_film_actors()
        actors = spawn_all(world)
    else:
        raise RuntimeError("keep_actors is not supported yet")
    unreal.EditorAssetLibrary.make_directory(PLAN["seq_dir"])
    # Every sequence is rebuilt: the actors were just respawned, and a sequence left from an earlier
    # build would still bind the deleted ones (Movie Render Queue then renders the default pawn's view).
    if ONLY:
        warn(f"--shots {sorted(ONLY)} ignored: all sequences are rebuilt with the new actors")
    for shot in PLAN["shots"]:
        build_shot(shot, actors)
    save_film_level(world)
    print(json.dumps(report))


main()
