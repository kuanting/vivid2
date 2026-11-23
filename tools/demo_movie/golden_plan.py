"""The plan of "The Golden Hour": film actors and the story shots (Level Sequences), from the real run.

Drones fly their logged paths (one key per film frame); hikers stand where the run placed them; the
fires, decoys and the fire's no-fly volume are the scenario's. Shot lengths are those of the edit
(``edit.py``); a shot that re-enacts part of the run maps run time [t0, t1] onto its frames.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from demo_movie import planlib as pl
from demo_movie import ue_coords as uc
from demo_movie.story import CHEN, LIN, Story

ANIM = {
    # root-locked copies of the forward walks (ue/film_assets.py): the actor moves at their stride speed
    "walk_m": f"{pl.ANIMS}/FILM_MM_Walk_InPlace",
    "walk_f": f"{pl.ANIMS}/FILM_MF_Walk_InPlace",
    "idle_f": "/Game/Hiking_Collection/Demo/Animations/MF_Idle",
    "lie_side": "/Game/Fab/Lie_Down_Animation_Pack/Lying_on_the_Side_with_One_Arm_Anim",
    "wave": "/Game/VIVID2/Targets/Anims/VIVID2_Wave_TwoArms_Anim",  # VIVID 2's own (tools/wave_clips_blender.py)
}
ZERO = {"roll": 0.0, "pitch": 0.0, "yaw": 0.0}
# the hiker meshes face the actor's +Y (the walk clips' root motion points along +Y): to face a
# heading h, an actor's yaw is h - 90
FACE = -90.0
# film frames of the story shots (must match edit.py)
FRAMES = {"S01": 180, "S02": 150, "S03a": 75, "S04": 150, "S06": 180, "S08": 180, "S10": 90,
          "S11": 180, "S12": 45, "S13": 180, "S14": 120, "S15": 60, "MAP": 2}
MAP_ALT_M = 1500.0
MAP_HFOV = 24.0


class Rig:
    """Camera helpers in NED (metres, down positive), converted to Unreal keys."""

    def __init__(self, ground: pl.Ground):
        self.g = ground

    def at(self, n: float, e: float, agl: float = 0.0) -> List[float]:
        return list(self.g.ue(n, e, agl))

    def look(self, frames: Sequence[float], eyes: Sequence[Sequence[float]], targets: Sequence[Sequence[float]],
             interp: str = "auto") -> Dict[str, Any]:
        for e in eyes:
            n, ee, z = uc.ue_cm_to_ned(e)
            agl = self.g.down(n, ee) - z
            if agl < 0.3:
                raise ValueError(f"camera below ground ({agl:.2f} m AGL) at NE {n:.1f}, {ee:.1f}")
        return pl.look_keys(frames, eyes, targets, interp)


def _lerp(a, b, t):
    return [x + (y - x) * t for x, y in zip(a, b)]


def drone_actor_keys(story: Story, drone: str, t0: float, t1: float, n: int) -> Dict[str, Any]:
    return pl.drone_keys(story.run.tracks[drone], t0, t1, n)


def golden_plan(story: Story) -> Dict[str, Any]:
    run, meta = story.run, story.run.meta
    ground = pl.Ground(meta)
    rig = Rig(ground)
    sun = pl.sun_of(meta["params"].get("environment.time_of_day")
                    or (meta.get("scenario_environment") or {}).get("time_of_day") or "2026-09-28 17:00:00")
    chen, lin = run.target(CHEN), run.target(LIN)
    cn, ce = chen["ned"][0], chen["ned"][1]
    ln, le = lin["ned"][0], lin["ned"][1]

    actors: List[Dict[str, Any]] = pl.scene_actors(meta)
    for f in meta.get("fleet") or []:
        actors.append(pl.quad_actor(f["id"], uc.ned_to_ue_cm(f["start_ned"]), [0, 0, f.get("yaw_deg", 0.0)]))
    # the fire no-fly volume (hidden except in its shot): a box from the ground to the ceiling
    zones = (meta.get("limits") or {}).get("no_fly") or []
    if zones:
        z = zones[0]
        poly = np.asarray(z["polygon_ne"], float)
        lo, hi = poly.min(0), poly.max(0)
        nn, ee = np.meshgrid(np.linspace(lo[0], hi[0], 9), np.linspace(lo[1], hi[1], 9))
        gmean = float(np.mean(ground.model.down_at(nn, ee)))
        ceil = float(z.get("ceiling_agl_m", 30.0))
        centre = uc.ned_to_ue_cm(((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, gmean - ceil / 2 + 5))
        actors.append({"name": "NoFly", "kind": "mesh", "mesh": "/Engine/BasicShapes/Cube", "location": list(centre),
                       "rotation": [0, 0, 0], "scale": [(hi[0] - lo[0]), (hi[1] - lo[1]), ceil + 10],
                       "material": f"{pl.MATS}/MI_FILM_NoFly", "hidden": True})
        # its edges: four corner posts from the ground and the ceiling outline, glowing red
        top = gmean - ceil
        corners = [(lo[0], lo[1]), (lo[0], hi[1]), (hi[0], hi[1]), (hi[0], lo[1])]
        th = 0.5
        for k, (cn_, ce_) in enumerate(corners):
            g = ground.down(cn_, ce_)
            mid = uc.ned_to_ue_cm((cn_, ce_, (g + top) / 2))
            actors.append({"name": f"NoFlyPost{k}", "kind": "mesh", "mesh": "/Engine/BasicShapes/Cube",
                           "location": list(mid), "rotation": [0, 0, 0], "scale": [th, th, g - top],
                           "material": f"{pl.MATS}/MI_FILM_NoFlyEdge", "hidden": True})
            (an_, ae_), (bn_, be_) = corners[k], corners[(k + 1) % 4]
            mid = uc.ned_to_ue_cm(((an_ + bn_) / 2, (ae_ + be_) / 2, top))
            length = math.hypot(bn_ - an_, be_ - ae_)
            yaw = math.degrees(math.atan2(be_ - ae_, bn_ - an_))
            actors.append({"name": f"NoFlyEdge{k}", "kind": "mesh", "mesh": "/Engine/BasicShapes/Cube",
                           "location": list(mid), "rotation": [0, 0, yaw], "scale": [length, th, th],
                           "material": f"{pl.MATS}/MI_FILM_NoFlyEdge", "hidden": True})

    shots: List[Dict[str, Any]] = []
    cams: List[Dict[str, Any]] = []

    def shot(sid, cam_keys, focal, tracks=(), aperture=5.6, focus=None, exposure=None, lens=None, sun_override=None, **extra):
        name = f"Cam_{sid}"
        cams.append(pl.camera_actor(name, focal_mm=focal, aperture=aperture, focus_cm=focus, exposure=exposure,
                                    location=cam_keys["loc"][0], rotation=cam_keys["rot"][0]))
        s = {"id": sid, "frames": FRAMES[sid], "sun": sun_override or sun,
             "camera": {"actor": name, "transform": cam_keys}, "tracks": list(tracks)}
        if lens:
            s["camera"]["lens"] = lens
        s.update(extra)
        shots.append(s)

    hold = lambda nm, loc, yaw: {"actor": nm, "transform": pl.static_keys(loc, {"roll": 0, "pitch": 0, "yaw": yaw})}

    # S01 establishing: crane from the west over the forest towards the burning slope
    n = FRAMES["S01"]
    eyes = [rig.at(-150, -70, 125), rig.at(-105, -5, 98)]
    tgts = [rig.at(-40, 95, 0), rig.at(-20, 120, 0)]
    shot("S01", rig.look([0, n], eyes, tgts), focal=26.0, aperture=11.0)

    # S02 the two hikers walk across Chen's clearing, backlit (sun in the west)
    n = FRAMES["S02"]
    # across Chen's clearing towards the north-west, at the walk clips' stride speed (feet don't slide)
    heading = math.radians(-53.0)
    u = (math.cos(heading), math.sin(heading))
    half = pl.WALK_SPEED_MPS * n / 30.0 / 2
    a0, a1 = (cn - half * u[0], ce - half * u[1]), (cn + half * u[0], ce + half * u[1])
    side = (-u[1] * 1.2, u[0] * 1.2)  # Lin walks at his right, half a step behind
    yaw = math.degrees(heading) + FACE
    walk_c = pl.transform_keys([0, n], [rig.at(*a0), rig.at(*a1)], [dict(ZERO, yaw=yaw)] * 2, "linear")
    walk_l = pl.transform_keys([0, n], [rig.at(a0[0] + side[0] - 0.5 * u[0], a0[1] + side[1] - 0.5 * u[1]),
                                        rig.at(a1[0] + side[0] - 0.5 * u[0], a1[1] + side[1] - 0.5 * u[1])],
                               [dict(ZERO, yaw=yaw)] * 2, "linear")
    # a tracking camera beside them (on her side), a little ahead, at chest height
    eyes = [rig.at(p[0] + side[0] * 5.0 + u[0] * 2.0, p[1] + side[1] * 5.0 + u[1] * 2.0, 1.3) for p in (a0, a1)]
    tgts = [rig.at(p[0] + side[0] * 0.5, p[1] + side[1] * 0.5, 1.1) for p in (a0, a1)]
    shot("S02", rig.look([0, n], eyes, tgts), focal=35.0, aperture=2.8, focus=650.0,
         tracks=[{"actor": CHEN, **{"transform": walk_c}, "anims": [{"clip": ANIM["walk_m"], "start": 0, "end": n}]},
                 {"actor": LIN, **{"transform": walk_l}, "anims": [{"clip": ANIM["walk_f"], "start": 0, "end": n}]}])

    # S03a Chen stumbles at the clearing's edge and collapses (the lie-down clip from standing)
    n = FRAMES["S03a"]
    yaw_c = float(chen.get("yaw_deg", 0.0))
    chen_loc = list(uc.ned_to_ue_cm(chen["ned"]))
    eye, tgt = rig.at(cn - 5.0, ce + 3.0, 1.6), rig.at(cn, ce, 0.9)
    eye2 = rig.at(cn - 4.0, ce + 2.4, 1.0)
    shot("S03a", rig.look([0, n], [eye, eye2], [tgt, rig.at(cn, ce, 0.3)]), focal=35.0, aperture=2.8, focus=550.0,
         tracks=[{"actor": CHEN, "transform": pl.static_keys(chen_loc, dict(ZERO, yaw=yaw_c)),
                  "anims": [{"clip": ANIM["lie_side"], "start": 0, "end": n, "offset_s": 1.6}]},
                 hold(LIN, rig.at(ln, le), 0.0)])

    # S04 Lin in the burnt area, the burning front behind her (west); she has turned east, away from it
    n = FRAMES["S04"]
    lin_loc = list(uc.ned_to_ue_cm(lin["ned"]))
    lin_east = {"actor": LIN, "transform": pl.static_keys(lin_loc, dict(ZERO, yaw=90.0 + FACE)),
                "anims": [{"clip": ANIM["idle_f"], "start": 0, "end": n}]}
    eyes = [rig.at(ln + 2.5, le + 10.0, 1.6), rig.at(ln + 1.5, le + 6.0, 1.5)]
    tgts = [rig.at(ln - 1.0, le - 8.0, 1.8), rig.at(ln - 0.5, le - 4.0, 1.3)]
    shot("S04", rig.look([0, n], eyes, tgts), focal=32.0, aperture=4.0, focus=650.0, tracks=[lin_east])

    # S06 launch: ground-level in the clearing, the four drones take off past the camera (real time)
    n = FRAMES["S06"]
    t0 = story.takeoff_rt - 0.5
    pads = np.asarray([f["start_ned"][:2] for f in meta["fleet"]], float).mean(0)
    eye = rig.at(pads[0] - 9.0, pads[1] - 2.5, 0.45)
    shot("S06", rig.look([0, n * 0.35, n], [eye] * 3,
                         [rig.at(pads[0], pads[1], 1.2), rig.at(pads[0], pads[1], 3.0), rig.at(pads[0], pads[1], 11.0)]),
         focal=18.0, aperture=5.6,
         tracks=[{"actor": d, "transform": drone_actor_keys(story, d, t0, t0 + n / 30.0, n),
                  "props_deg_per_frame": pl.PROP_DEG_PER_FRAME} for d in run.drones])

    # S08 chase behind a surveying drone (real time)
    n = FRAMES["S08"]
    d, t0 = story.chase["drone"], story.chase["start"]
    keys = drone_actor_keys(story, d, t0, t0 + n / 30.0, n)
    tr = run.tracks[d]
    eyes, tgts = [], []
    for k in range(0, n + 1, 6):
        p, q = tr.pose_at(t0 + k / 30.0)
        yawr = math.radians(uc.quat_ned_to_rotator(q)["yaw"])
        v = tr.pose_at(t0 + k / 30.0 + 0.5)[0] - tr.pose_at(t0 + k / 30.0 - 0.5)[0]
        h = math.atan2(v[1], v[0]) if np.linalg.norm(v[:2]) > 0.5 else yawr
        eyes.append(list(uc.ned_to_ue_cm((p[0] - 9 * math.cos(h), p[1] - 9 * math.sin(h), p[2] - 4.0))))
        tgts.append(list(uc.ned_to_ue_cm((p[0] + 5 * math.cos(h), p[1] + 5 * math.sin(h), p[2] + 7.0))))
    shot("S08", rig.look(list(range(0, n + 1, 6)), eyes, tgts, "auto"), focal=28.0, aperture=8.0,
         tracks=[{"actor": d, "transform": keys, "props_deg_per_frame": pl.PROP_DEG_PER_FRAME}])

    # S10 a dip: from the clearing's floor, looking up at the drone coming down between the crowns
    n = FRAMES["S10"]
    if story.dip and story.dip.get("bottom") is not None:
        dd = story.dip
        an, ae = dd["at"]
        t0, t1 = dd["start"] + 0.6 * (dd["bottom"] - dd["start"]), dd["bottom"]
        eye = rig.at(an - 22.0, ae - 14.0, 46.0)
        tr = run.tracks[dd["drone"]]
        p0, p1 = tr.pose_at(t0)[0], tr.pose_at(t1)[0]
        shot("S10", rig.look([0, n], [eye, eye], [list(uc.ned_to_ue_cm(p0)), list(uc.ned_to_ue_cm(p1))]),
             focal=32.0, aperture=5.6,
             tracks=[{"actor": dd["drone"], "transform": drone_actor_keys(story, dd["drone"], t0, t1, n),
                      "props_deg_per_frame": pl.PROP_DEG_PER_FRAME}], run_window=[t0, t1])

    # S11 telephoto: a drone tracking along the fire front above the no-fly ceiling
    n = FRAMES["S11"]
    if story.fire_pass:
        fp = story.fire_pass
        d = fp["drone"]
        tmid = (fp["start"] + fp["end"]) / 2
        t0, t1 = tmid - 3.0, tmid + 3.0
        tr = run.tracks[d]
        pm = tr.pose_at(tmid)[0]
        eye = list(uc.ned_to_ue_cm((pm[0] - 30.0, pm[1] - 95.0, pm[2] - 55.0)))
        tg = [list(uc.ned_to_ue_cm(tr.pose_at(t0 + (t1 - t0) * k / 4)[0] + np.array([0.0, 0.0, 4.0]))) for k in range(5)]
        tg = [list(np.asarray(t_) + np.array([0.0, 0.0, -600.0])) for t_ in tg]  # the zone's ceiling just below the drone
        shot("S11", rig.look([n * k / 4 for k in range(5)], [eye] * 5, tg), focal=85.0, aperture=8.0,
             tracks=[{"actor": d, "transform": drone_actor_keys(story, d, t0, t1, n),
                      "props_deg_per_frame": pl.PROP_DEG_PER_FRAME}]
                    + [{"actor": a_["name"], "visible": True} for a_ in actors if a_["name"].startswith("NoFly")],
             run_window=[t0, t1])

    # S12 match cut: the film camera exactly where the search camera was for the photo that started
    # the closer look at Chen (the edit dissolves from this render into the real photo)
    f_chen = story.finds.get(CHEN)
    if f_chen and f_chen.survey_frame:
        rec = run.frame(f_chen.survey_frame)
        cam = uc.camera_from_frame(rec)
        n = FRAMES["S12"]
        d = rec.get("drone", run.drones[0])
        shot("S12", pl.static_keys(cam["location"], cam["rotation"]), focal=cam["focal_mm"], aperture=16.0,
             tracks=[{"actor": dd_, "transform": drone_actor_keys(story, dd_, rec["rt"], rec["rt"] + n / 30.0, n),
                      "props_deg_per_frame": pl.PROP_DEG_PER_FRAME} for dd_ in run.drones if dd_ != d],
             match_frame=f_chen.survey_frame)

    # S13 Chen's closer look: low angle beside him, the drone comes down and hovers
    n = FRAMES["S13"]
    if f_chen:
        d = f_chen.drone
        if f_chen.verify and f_chen.verify.get("end"):
            t0, t1 = f_chen.verify["start"], f_chen.verify["end"]
        else:
            t0, t1 = f_chen.report_rt - 8.0, f_chen.report_rt + 2.0
        eye = rig.at(cn - 14.0, ce - 10.0, 40.0)
        tr = run.tracks[d]
        ks = [k / 4 for k in range(5)]
        tgts = [_lerp(list(uc.ned_to_ue_cm(tr.pose_at(t0 + (t1 - t0) * k)[0])), rig.at(cn, ce, 0.3), 0.5 * k) for k in ks]
        shot("S13", rig.look([n * k for k in ks], [eye] * 5, tgts),
             focal=22.0, aperture=5.6,
             tracks=[{"actor": d, "transform": drone_actor_keys(story, d, t0, t1, n),
                      "props_deg_per_frame": pl.PROP_DEG_PER_FRAME}], run_window=[t0, t1])

    # S14 Lin found: orbit around her waving, the drone hovering above, fire glow behind
    n = FRAMES["S14"]
    f_lin = story.finds.get(LIN)
    if f_lin:
        d = f_lin.drone
        if f_lin.verify and f_lin.verify.get("end"):
            t1 = f_lin.verify["end"]
        else:
            t1 = f_lin.report_rt + 1.0
        t0 = t1 - n / 30.0  # the end of the closer look, in real time: the drone already low over her
        frames, eyes, tgts = [], [], []
        lin_chest = np.array([ln, le, ground.down(ln, le) - 1.0])
        for k in range(9):
            t = t0 + (t1 - t0) * k / 8
            dp = run.tracks[d].pose_at(t)[0]
            away = np.array([dp[0] - ln, dp[1] - le, 0.0])
            away = away / max(1e-6, np.linalg.norm(away)) if np.linalg.norm(away) > 0.5 else np.array([1.0, 0.0, 0.0])
            # a follow camera above and behind the drone, looking past it down at her
            eye_ned = dp + away * 9.0 + np.array([0.0, 2.0, -1.5])
            frames.append(n * k / 8)
            eyes.append(list(uc.ned_to_ue_cm(eye_ned)))
            tgts.append(list(uc.ned_to_ue_cm(lin_chest)))
        shot("S14", rig.look(frames, eyes, tgts), focal=45.0, aperture=5.6,
             tracks=[{"actor": d, "transform": drone_actor_keys(story, d, t0, t1, n),
                      "props_deg_per_frame": pl.PROP_DEG_PER_FRAME},
                     dict(lin_east, anims=[{"clip": ANIM["idle_f"], "start": 0, "end": n}])], run_window=[t0, t1])

    # S15 landing: the last drone settles onto its pad (the others are already down)
    n = FRAMES["S15"]
    if story.landed_rt:
        last = max(story.landed_rt, key=story.landed_rt.get)
        t1 = story.landed_rt[last]
        t0 = t1 - n / 30.0
        eye = rig.at(pads[0] + 16.0, pads[1] + 12.0, 1.0)
        shot("S15", rig.look([0, n], [eye, eye], [rig.at(pads[0], pads[1], 2.0), rig.at(pads[0], pads[1], 1.0)]),
             focal=35.0, aperture=5.6,
             tracks=[{"actor": dd_, "transform": drone_actor_keys(story, dd_, t0, t1, n),
                      "props_deg_per_frame": pl.PROP_DEG_PER_FRAME if dd_ == last else 0.0} for dd_ in run.drones])

    # MAP plate: north-up nadir view of the search area for the data shots
    poly = np.asarray(meta["search_area"]["polygon_ne"], float)
    lo, hi = poly.min(0) - 30, poly.max(0) + 30
    c = (lo + hi) / 2
    gd = ground.down(c[0], c[1])
    cam_down = gd - MAP_ALT_M
    eye = list(uc.ned_to_ue_cm((c[0], c[1], cam_down)))
    cams_map = {"roll": 0.0, "pitch": -90.0, "yaw": 0.0}
    focal = uc.hfov_to_focal_mm(MAP_HFOV)
    shot("MAP", pl.static_keys(eye, cams_map), focal=focal, aperture=16.0,
         map_plate={"n0": float(c[0]), "e0": float(c[1]), "cam_down": float(cam_down),
                    "f_px": (1920 / 2) / math.tan(math.radians(MAP_HFOV) / 2)})

    return pl.plan_doc("golden_hour", actors + cams, shots, story_notes=story.notes)
