"""P0 probe plan: two short test shots on ForestFire.

* P1 registration: a film camera at the exact pose of one recorded payload frame (80 deg horizontal
  FOV) with the run's hikers, decoys and fires placed from its meta. Its render must line up with the
  recorded JPEG (central 16:9 band): this checks the NED -> Unreal transform the whole film relies on.
* P2 motion: in the launch clearing, a man walking (in-place clip), a woman walking (forward clip, to
  see root motion), the LyingSide hiker playing its lie-down clip from the start, the Waving hiker
  waving, and a hovering quadcopter with spinning propellers.
"""
from __future__ import annotations

from typing import Any, Dict

from demo_movie import planlib as pl
from demo_movie import ue_coords as uc
from demo_movie.run_data import FilmRun

ANIM = {
    "walk_m": "/Game/Hiking_Collection/Demo/Animations/MM_Walk_InPlace",
    "walk_f": "/Game/Hiking_Collection/Demo/Animations/MF_Walk_Fwd",
    "lie_side": "/Game/Fab/Lie_Down_Animation_Pack/Lying_on_the_Side_with_One_Arm_Anim",
    "wave": "/Game/VIVID2/Targets/Anims/VIVID2_Wave_TwoArms_Anim",  # VIVID 2's own (tools/wave_clips_blender.py)
}


def probe_plan(run: FilmRun, frame_id: int = 67) -> Dict[str, Any]:
    meta = run.meta
    ground = pl.Ground(meta)
    sun = pl.sun_of(meta["params"].get("environment.time_of_day", "2026-09-28 13:00:00"))
    actors = pl.scene_actors(meta)
    shots = []

    # P1: registration
    rec = run.frame(frame_id)
    cam = uc.camera_from_frame(rec)
    actors.append(pl.camera_actor("Cam_P1", focal_mm=cam["focal_mm"], aperture=16.0, location=cam["location"],
                                  rotation=[cam["rotation"][k] for k in ("roll", "pitch", "yaw")]))
    shots.append({"id": "P1", "frames": 3, "sun": sun,
                  "camera": {"actor": "Cam_P1", "transform": pl.static_keys(cam["location"], cam["rotation"])},
                  "registration": {"run_dir": str(run.run_dir), "frame_id": frame_id}})

    # P2: motion in the launch clearing (ground ~ flat, 30 x 30 m)
    n = 180
    def at(nn, ee, agl=0.0):
        return list(ground.ue(nn, ee, agl))

    actors += [
        {"name": "Walker_M", "kind": "bp", "class": f"{pl.TARGETS}/VIVID2_Hiker_RedJacket_Standing", "location": at(-4, -3), "rotation": [0, 0, 0]},
        {"name": "Walker_F", "kind": "bp", "class": f"{pl.TARGETS}/VIVID2_Hiker_YellowJacket_Standing", "location": at(-4, -1), "rotation": [0, 0, 0]},
        {"name": "Faller", "kind": "bp", "class": f"{pl.TARGETS}/VIVID2_Hiker_RedJacket_LyingSide", "location": at(1, 2), "rotation": [0, 0, 90]},
        {"name": "Waver", "kind": "bp", "class": f"{pl.TARGETS}/VIVID2_Hiker_YellowJacket_Waving", "location": at(1, 5), "rotation": [0, 0, 180]},
        pl.quad_actor("QuadT", at(3, 0.5, 2.5), frame_materials=[f"{pl.MATS}/MI_FILM_Slot{c}" for c in "RGB"]),
        pl.quad_actor("QuadP", at(3, 2.0, 2.5)),
    ]
    eye, tgt = at(-10, 1, 1.7), at(1, 1.5, 1.0)
    actors.append(pl.camera_actor("Cam_P2", focal_mm=24.0, aperture=8.0, location=eye,
                                  rotation=list(uc.look_at_rotator(eye, tgt).values())))
    walk = lambda e: pl.transform_keys([0, n], [at(-4, e), at(4, e)], [{"roll": 0, "pitch": 0, "yaw": 0}] * 2, "linear")
    quad_rot = [{"roll": 0, "pitch": 0, "yaw": 0}, {"roll": 0, "pitch": 0, "yaw": 90}]
    shots.append({"id": "P2", "frames": n, "sun": dict(sun, solar_time=17.0),
                  "camera": {"actor": "Cam_P2", "transform": pl.look_keys([0], [eye], [tgt])},
                  "tracks": [
                      {"actor": "Walker_M", "transform": walk(-3), "anims": [{"clip": ANIM["walk_m"], "start": 0, "end": n}]},
                      {"actor": "Walker_F", "transform": walk(-1), "anims": [{"clip": ANIM["walk_f"], "start": 0, "end": n}]},
                      {"actor": "Faller", "anims": [{"clip": ANIM["lie_side"], "start": 0, "end": n}]},
                      {"actor": "Waver", "anims": [{"clip": ANIM["wave"], "start": 0, "end": n}]},
                      {"actor": "QuadT", "props_deg_per_frame": pl.PROP_DEG_PER_FRAME,
                       "transform": pl.transform_keys([0, n], [at(3, 0.5, 2.5), at(3, 0.5, 3.5)], quad_rot, "auto")},
                  ]})

    # P3/P4: propeller close-up, and the sun: the same camera at 13:00 and 17:30 (shadows must turn)
    eye, tgt = at(0.2, -1.2, 3.4), at(3, 1.25, 2.5)
    actors.append(pl.camera_actor("Cam_P3", focal_mm=35.0, aperture=4.0, focus_cm=330.0, location=eye,
                                  rotation=list(uc.look_at_rotator(eye, tgt).values())))
    for sid, solar in (("P3", 17.5), ("P4", 13.0)):
        shots.append({"id": sid, "frames": 30 if sid == "P3" else 3, "sun": dict(sun, solar_time=solar),
                      "camera": {"actor": "Cam_P3", "transform": pl.look_keys([0], [eye], [tgt])},
                      "tracks": [{"actor": q, "props_deg_per_frame": pl.PROP_DEG_PER_FRAME,
                                  "transform": pl.static_keys(at(3, e, 2.5), {"roll": 0, "pitch": 0, "yaw": 20})}
                                 for q, e in (("QuadT", 0.5), ("QuadP", 2.0))]})
    return pl.plan_doc("probe", actors, shots)
