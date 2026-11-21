"""Building blocks of a film plan (plan.json, read by ue/film_build.py): actors placed where a run had
them, camera and drone keys, sun. All output in Unreal units (cm, [roll, pitch, yaw] degrees, frames).
"""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from demo_movie import ue_coords as uc
from vivid2 import terrain as terrain_mod

TARGETS = "/Game/VIVID2/Targets"
FIRES = "/Game/VIVID2/Fire"
FILM_LEVEL = "/Game/VIVID2/Demo/Maps/ForestFire_Film"
BASE_LEVEL = "/Game/VIVID2/Maps/ForestFire"
FPS = 30


def sun_of(time_of_day: str) -> Dict[str, Any]:
    """'2026-09-28 17:00:00' -> SunSky Month/Day/SolarTime (what Project AirSim's TimeofDay sets)."""
    date, clock = time_of_day.split(" ")
    _, month, day = (int(x) for x in date.split("-"))
    h, m, s = (int(x) for x in clock.split(":"))
    return {"month": month, "day": day, "solar_time": h + m / 60.0 + s / 3600.0}


def placed(meta_obj: Dict[str, Any], folder: str, name: Optional[str] = None, asset: Optional[str] = None) -> Dict[str, Any]:
    """A benchmark blueprint where the run spawned it (meta positions are terrain-snapped NED)."""
    return {"name": name or meta_obj["id"], "kind": "bp", "class": f"{folder}/{asset or meta_obj['asset']}",
            "location": list(uc.ned_to_ue_cm(meta_obj["ned"])), "rotation": [0.0, 0.0, float(meta_obj.get("yaw_deg", 0.0))]}


def scene_actors(meta: Dict[str, Any], hikers: bool = True, decoys: bool = True, fires: bool = True,
                 asset_map: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
    out = []
    asset_map = asset_map or {}
    if hikers:
        out += [placed(t, TARGETS, asset=asset_map.get(t["id"])) for t in meta.get("targets", [])]
    if decoys:
        out += [placed(d, TARGETS) for d in meta.get("distractors", [])]
    if fires:
        out += [placed(f, FIRES) for f in meta.get("fires", [])]
    return out


class Ground:
    """Terrain heights of the run's map, in NED down and Unreal Z."""

    def __init__(self, meta: Dict[str, Any]):
        self.model = terrain_mod.from_meta(meta)

    def down(self, n: float, e: float) -> float:
        return float(np.asarray(self.model.down_at(n, e)))

    def ue(self, n: float, e: float, agl_m: float = 0.0) -> Tuple[float, float, float]:
        return uc.ned_to_ue_cm((n, e, self.down(n, e) - agl_m))


def transform_keys(frames: Sequence[float], locs: Sequence[Sequence[float]], rots: Sequence[Dict[str, float]],
                   interp: str = "auto") -> Dict[str, Any]:
    rolls = uc.unwrap_deg(r["roll"] for r in rots)
    pitches = [r["pitch"] for r in rots]
    yaws = uc.unwrap_deg(r["yaw"] for r in rots)
    return {"frames": [float(f) for f in frames], "loc": [[float(v) for v in l] for l in locs],
            "rot": [[a, b, c] for a, b, c in zip(rolls, pitches, yaws)], "interp": interp}


def static_keys(loc: Sequence[float], rot: Dict[str, float]) -> Dict[str, Any]:
    return transform_keys([0], [loc], [rot])


def look_keys(frames: Sequence[float], eyes: Sequence[Sequence[float]], targets: Sequence[Sequence[float]],
              interp: str = "auto") -> Dict[str, Any]:
    """Camera keys: at each frame, the eye (UE cm) looks at the target (UE cm)."""
    rots = [uc.look_at_rotator(e, t) for e, t in zip(eyes, targets)]
    return transform_keys(frames, eyes, rots, interp)


def drone_keys(track, t0: float, t1: float, n_frames: int, frame0: int = 0) -> Dict[str, Any]:
    """A drone flying its logged path from run time t0 to t1 over n_frames film frames (one key per frame)."""
    ts = np.linspace(t0, t1, n_frames + 1)
    locs, rots = [], []
    for t in ts:
        p, q = track.pose_at(float(t))
        locs.append(uc.ned_to_ue_cm(p))
        rots.append(uc.quat_ned_to_rotator(q))
    return transform_keys([frame0 + i for i in range(n_frames + 1)], locs, rots, interp="linear")


def camera_actor(name: str, focal_mm: float = 35.0, aperture: float = 8.0, focus_cm: Optional[float] = None,
                 exposure: Optional[Dict[str, Any]] = None, location=(0, 0, 0), rotation=(0, 0, 0)) -> Dict[str, Any]:
    spec = {"name": name, "kind": "camera", "focal_mm": focal_mm, "aperture": aperture, "location": list(location),
            "rotation": list(rotation), "sensor_mm": list(uc.FILMBACK_16x9_MM)}
    if focus_cm:
        spec["focus_cm"] = focus_cm
    if exposure:
        spec["exposure"] = exposure
    return spec


MATS = "/Game/VIVID2/Demo/Materials"
ANIMS = "/Game/VIVID2/Demo/Anims"
# the forward-walk clips, root-locked copies (ue/film_assets.py); both stride at 2.40 m/s
# (MM: 6.558 m per 2.733 s cycle, MF: 4.478 m per 1.867 s, measured from the root bone)
INPLACE_CLIPS = {"/Game/Hiking_Collection/Demo/Animations/MM_Walk_Fwd": "FILM_MM_Walk_InPlace",
                 "/Game/Hiking_Collection/Demo/Animations/MF_Walk_Fwd": "FILM_MF_Walk_InPlace"}
WALK_SPEED_MPS = 2.40
# the film's paint and light materials (ue/film_assets.py makes MI_FILM_<name>)
PAINTS = {
    "DroneShell": {"color": (0.80, 0.81, 0.80, 1), "roughness": 0.32},
    "DroneDark": {"color": (0.025, 0.026, 0.03, 1), "roughness": 0.55},
    "DroneAccent": {"color": (0.95, 0.30, 0.02, 1), "roughness": 0.4},
    "PropDark": {"color": (0.04, 0.04, 0.045, 1), "roughness": 0.35},
    "SlotR": {"color": (1, 0, 0, 1)}, "SlotG": {"color": (0, 1, 0, 1)}, "SlotB": {"color": (0, 0, 1, 1)},
}
GLOWS = {"NavRed": {"color": (1.0, 0.04, 0.02, 1), "strength": 6.0},
         "NavGreen": {"color": (0.05, 1.0, 0.12, 1), "strength": 6.0},
         "NoFlyEdge": {"color": (1.0, 0.08, 0.02, 1), "strength": 4.0}}
# slots of /Drone/Quadrotor1 (found with a red/green/blue slot test): 0 = landing legs and the arms
# inside the ducts, 1 = small parts, 2 = the shell (ducts and body)
DRONE_PAINT = [f"{MATS}/MI_FILM_DroneDark", f"{MATS}/MI_FILM_DroneAccent", f"{MATS}/MI_FILM_DroneShell"]


def quad_actor(name: str, location, rotation=(0, 0, 0), nav_lights: bool = True,
               frame_materials: Optional[List[str]] = None) -> Dict[str, Any]:
    spec = {"name": name, "kind": "quad", "location": list(location), "rotation": list(rotation),
            "frame_materials": list(frame_materials if frame_materials is not None else DRONE_PAINT),
            "prop_materials": {"front": f"{MATS}/MI_FILM_PropDark", "rear": f"{MATS}/MI_FILM_PropDark"}}
    if nav_lights:
        spec["nav"] = {"left": f"{MATS}/MI_FILM_NavRed", "right": f"{MATS}/MI_FILM_NavGreen", "light_intensity": 0.0}
    return spec


# real propellers turn at ~100 rev/s; 45 rev/s = 540 deg per 30 fps frame still blurs to a disc with
# 8 temporal samples and doesn't strobe backwards
PROP_DEG_PER_FRAME = 540.0


def plan_doc(name: str, actors: List[Dict[str, Any]], shots: List[Dict[str, Any]], **extra) -> Dict[str, Any]:
    names = [a["name"] for a in actors]
    dup = {n for n in names if names.count(n) > 1}
    if dup:
        raise ValueError(f"duplicate actor names {sorted(dup)}")
    for s in shots:
        for tr in s.get("tracks", []):
            if tr["actor"] not in names:
                raise ValueError(f"shot {s['id']}: unknown actor {tr['actor']}")
        if s["camera"]["actor"] not in names:
            raise ValueError(f"shot {s['id']}: unknown camera {s['camera']['actor']}")
    return dict({"name": name, "fps": FPS, "level": {"film": FILM_LEVEL, "base": BASE_LEVEL},
                 "seq_dir": f"/Game/VIVID2/Demo/Sequences/{name}", "actors": actors, "shots": shots}, **extra)
