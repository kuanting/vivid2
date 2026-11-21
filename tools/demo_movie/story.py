"""The film's beats, chosen from the real run (nothing here is invented: every time, drone and frame
comes from the logs). Chen = Hiker1 (fell in the forest), Lin = Hiker2 (cut off by the fire).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from demo_movie.run_data import FilmRun
from vivid2.metrics.planning import points_in_polygon

CHEN, LIN = "Hiker1", "Hiker2"


@dataclass
class Find:
    target: str
    drone: str
    report_rt: float
    error_m: float
    frame_id: Optional[int]           # the photo the report cites
    verify: Optional[Dict[str, Any]]  # verify_start/end around it (closer look)
    survey_frame: Optional[int]       # the photo that triggered the closer look (or the cited one)
    closeup_frame: Optional[int]      # the closest look at the target during the verify


@dataclass
class Story:
    run: FilmRun
    finds: Dict[str, Find]
    takeoff_rt: float
    end_rt: float
    landed_rt: Dict[str, float]
    dip: Optional[Dict[str, Any]]
    fire_pass: Optional[Dict[str, Any]]
    chase: Dict[str, Any]
    notes: List[str] = field(default_factory=list)

    def target(self, tid: str) -> Dict[str, Any]:
        return self.run.target(tid)


def _near(a, b, r):
    return math.hypot(a[0] - b[0], a[1] - b[1]) <= r


def _visible(rec, tid) -> int:
    return max([g.get("visible_px", 0) for g in rec.get("gt", []) if g.get("object_id") == tid] or [0])


def build_story(run: FilmRun) -> Story:
    notes = []
    km = run.key_moments()
    takeoff = min(km["takeoff"].values()) if km["takeoff"] else 0.0
    finds: Dict[str, Find] = {}
    for tid, rep in run.matched_reports().items():
        t = run.target(tid)
        drone = rep["drone"]
        ver = None
        for v in km["verify"]:
            cand = v.get("candidate") or (v["event"].get("candidate") if v.get("event") else None)
            if v["drone"] == drone and v["start"] <= rep["rt"] + 0.5 and cand and _near(cand, t["ned"], 20):
                ver = v
        survey = (ver["event"].get("frame_id") if ver else None) or rep.get("frame_id")
        close = None
        if ver and ver.get("end") is not None:
            best = 0
            for f in run.frames("Payload", drone):
                if ver["start"] <= f["rt"] <= ver["end"] + 0.5 and _visible(f, tid) > best:
                    best, close = _visible(f, tid), f["frame_id"]
        finds[tid] = Find(tid, drone, rep["rt"], rep["error_m"], rep.get("frame_id"), ver, survey, close)
    for tid in (CHEN, LIN):
        if tid not in finds:
            notes.append(f"{tid} was not found in this run")

    dips = run.events("dip_start")
    dip = None
    if dips:
        d0 = dips[0]
        drone = d0.get("drone", run.drones[0])
        bottom = next((e for e in run.events("dip_bottom", drone) if e["rt"] >= d0["rt"]), None)
        end = next((e for e in run.events("dip_end", drone) if e["rt"] >= d0["rt"]), None)
        dip = {"drone": drone, "start": d0["rt"], "bottom": bottom["rt"] if bottom else None,
               "end": end["rt"] if end else None, "at": d0.get("at"), "radius_m": d0.get("radius_m")}
    else:
        notes.append("no dips in this run")

    # the drone that spends longest over the fire no-fly area (above its ceiling)
    fire_pass = None
    zones = (run.meta.get("limits") or {}).get("no_fly") or []
    if zones:
        poly = np.asarray(zones[0]["polygon_ne"], float)
        best = None
        for d, tr in run.tracks.items():
            inside = points_in_polygon(tr.pos[:, :2], poly)
            if inside.any():
                # longest contiguous stretch inside
                runs_, start = [], None
                for k, flag in enumerate(inside):
                    if flag and start is None:
                        start = k
                    if (not flag or k == len(inside) - 1) and start is not None:
                        runs_.append((start, k))
                        start = None
                s, e = max(runs_, key=lambda x: tr.t[x[1]] - tr.t[x[0]])
                dur = tr.t[e] - tr.t[s]
                if best is None or dur > best[1]:
                    best = (d, dur, tr.t[s], tr.t[e])
        if best:
            fire_pass = {"drone": best[0], "start": float(best[2]), "end": float(best[3])}

    # a calm survey stretch for the chase shot: the drone with the longest straight lane
    chase = {"drone": run.drones[min(1, len(run.drones) - 1)], "start": takeoff + 40.0}
    paths = run.events("path_start")
    if paths:
        p = max(paths, key=lambda e: max((np.linalg.norm(np.subtract(a[:2], b[:2])) for a, b in
                                          zip(e["waypoints"], e["waypoints"][1:])), default=0))
        chase = {"drone": p.get("drone", run.drones[0]), "start": p["rt"] + 8.0}

    return Story(run=run, finds=finds, takeoff_rt=takeoff, end_rt=run.duration, landed_rt=km["landed"],
                 dip=dip, fire_pass=fire_pass, chase=chase, notes=notes)
