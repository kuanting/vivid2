"""One recorded VIVID 2 run, prepared for filming: poses at any time, key moments, frames.

Times are run seconds (mission time from ``meta.t0_ns``). The trajectory log is not a steady 10 Hz
(samples land 0.003-0.34 s apart and some repeat), so each drone's track is de-duplicated and then
interpolated by time: position linearly, attitude by quaternion slerp.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from vivid2.recorder import RunLog

NS = 1e9


def _slerp(q0: np.ndarray, q1: np.ndarray, a: float) -> np.ndarray:
    d = float(np.dot(q0, q1))
    if d < 0.0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + a * (q1 - q0)
        return q / np.linalg.norm(q)
    th = math.acos(max(-1.0, min(1.0, d)))
    s = math.sin(th)
    return (math.sin((1 - a) * th) * q0 + math.sin(a * th) * q1) / s


@dataclass
class Track:
    """One drone's de-duplicated trajectory: times (run s), NED positions, wxyz quaternions."""
    t: np.ndarray
    pos: np.ndarray
    q: np.ndarray

    @classmethod
    def from_records(cls, recs: Sequence[Dict[str, Any]], t0_ns: int, min_dt_s: float = 0.02) -> "Track":
        recs = sorted(recs, key=lambda r: r["t"])
        t, pos, q = [], [], []
        for r in recs:
            tr = (r["t"] - t0_ns) / NS
            if t and tr - t[-1] < min_dt_s:  # near-duplicate sample: keep the later one
                t[-1], pos[-1], q[-1] = tr, r["pos"], r["q"]
                continue
            t.append(tr)
            pos.append(r["pos"])
            q.append(r["q"])
        qa = np.asarray(q, float)
        for i in range(1, len(qa)):  # keep consecutive quaternions in one hemisphere
            if float(np.dot(qa[i - 1], qa[i])) < 0.0:
                qa[i] = -qa[i]
        return cls(np.asarray(t, float), np.asarray(pos, float), qa)

    @property
    def start(self) -> float:
        return float(self.t[0])

    @property
    def end(self) -> float:
        return float(self.t[-1])

    def pose_at(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """(NED position, wxyz quaternion) at run time ``t`` (clamped to the track)."""
        if t <= self.t[0]:
            return self.pos[0].copy(), self.q[0].copy()
        if t >= self.t[-1]:
            return self.pos[-1].copy(), self.q[-1].copy()
        i = int(np.searchsorted(self.t, t)) - 1
        a = (t - self.t[i]) / (self.t[i + 1] - self.t[i])
        p = self.pos[i] + a * (self.pos[i + 1] - self.pos[i])
        return p, _slerp(self.q[i], self.q[i + 1], a)

    def sample(self, t0: float, t1: float, dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Poses every ``dt`` seconds over [t0, t1] (inclusive)."""
        n = max(2, int(round((t1 - t0) / dt)) + 1)
        ts = np.linspace(t0, t1, n)
        ps, qs = zip(*(self.pose_at(float(x)) for x in ts))
        return ts, np.asarray(ps), np.asarray(qs)

    def speed_at(self, t: float, window_s: float = 1.0) -> float:
        p0, _ = self.pose_at(t - window_s / 2)
        p1, _ = self.pose_at(t + window_s / 2)
        return float(np.linalg.norm(p1 - p0) / window_s)


@dataclass
class FilmRun:
    run_dir: Path
    log: RunLog
    t0_ns: int
    tracks: Dict[str, Track]
    result: Dict[str, Any] = field(default_factory=dict)

    @property
    def meta(self) -> Dict[str, Any]:
        return self.log.meta

    def rt(self, t_ns: Optional[float]) -> Optional[float]:
        """Run seconds of a recorded sim time (ns)."""
        return None if t_ns is None else (float(t_ns) - self.t0_ns) / NS

    @property
    def drones(self) -> List[str]:
        return sorted(self.tracks)

    @property
    def duration(self) -> float:
        return max(tr.end for tr in self.tracks.values())

    def events(self, kind: Optional[str] = None, drone: Optional[str] = None) -> List[Dict[str, Any]]:
        out = []
        for e in self.log["events"]:
            if kind is not None and e.get("kind") != kind:
                continue
            if drone is not None and e.get("drone", self.drones[0]) != drone:
                continue
            out.append(dict(e, rt=self.rt(e["t"])))
        return out

    def frames(self, camera: Optional[str] = "Payload", drone: Optional[str] = None) -> List[Dict[str, Any]]:
        out = []
        for f in self.log["frames"]:
            if camera is not None and f.get("camera", "Payload") != camera:
                continue
            if drone is not None and f.get("drone", self.drones[0]) != drone:
                continue
            out.append(dict(f, rt=self.rt(f["t"])))
        return out

    def frame(self, frame_id: int) -> Dict[str, Any]:
        for f in self.log["frames"]:
            if f["frame_id"] == frame_id:
                return dict(f, rt=self.rt(f["t"]))
        raise KeyError(frame_id)

    def reports(self) -> List[Dict[str, Any]]:
        return [dict(r, rt=self.rt(r.get("t_report", r["t"]))) for r in self.log["reports"]]

    def target(self, target_id: str) -> Dict[str, Any]:
        return next(t for t in self.meta["targets"] if t["id"] == target_id)

    def best_frame(self, target_id: str, camera: str = "Payload", drone: Optional[str] = None,
                   before_rt: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """The recorded frame in which a target shows the most visible pixels."""
        best, best_px = None, 0
        for f in self.frames(camera, drone):
            if before_rt is not None and f["rt"] > before_rt:
                continue
            for g in f.get("gt", []):
                if g.get("object_id") == target_id and g.get("visible_px", 0) > best_px:
                    best, best_px = f, g["visible_px"]
        return best

    def first_frame_seeing(self, target_id: str, min_px: int = 80, camera: str = "Payload") -> Optional[Dict[str, Any]]:
        for f in self.frames(camera):
            if any(g.get("object_id") == target_id and g.get("visible_px", 0) >= min_px for g in f.get("gt", [])):
                return f
        return None

    def key_moments(self) -> Dict[str, Any]:
        """Run times of the story beats: take-off and landing per drone, finds, verifies, end."""
        km: Dict[str, Any] = {"takeoff": {}, "rtl": {}, "landed": {}, "verify": [], "reports": [], "end": self.duration}
        for kind in ("takeoff", "rtl", "landed"):
            for e in self.events(kind):
                km[kind].setdefault(e.get("drone", self.drones[0]), e["rt"])
        starts = self.events("verify_start")
        ends = self.events("verify_end")
        for s in starts:
            d = s.get("drone", self.drones[0])
            e = next((x for x in ends if x.get("drone", self.drones[0]) == d and x["rt"] >= s["rt"]), None)
            km["verify"].append({"drone": d, "start": s["rt"], "end": e["rt"] if e else None,
                                 "candidate": s.get("candidate"), "event": s})
        for r in self.reports():
            km["reports"].append({"drone": r.get("drone", self.drones[0]), "rt": r["rt"], "ned": r["ned"],
                                  "frame_id": r.get("frame_id")})
        return km

    def matched_reports(self) -> Dict[str, Dict[str, Any]]:
        """{target id: first report within the success tolerance}, with the localisation error."""
        tol = float((self.meta.get("success") or {}).get("localization_tolerance_m", 10.0))
        out: Dict[str, Dict[str, Any]] = {}
        for r in sorted(self.reports(), key=lambda r: r["rt"]):
            for t in self.meta["targets"]:
                if t["id"] in out:
                    continue
                err = math.hypot(r["ned"][0] - t["ned"][0], r["ned"][1] - t["ned"][1])
                if err <= tol:
                    out[t["id"]] = dict(r, error_m=err)
                    break
        return out


def load_run(run_dir) -> FilmRun:
    run_dir = Path(run_dir)
    log = RunLog.load(run_dir)
    t0 = int(log.meta.get("t0_ns") or min(r["t"] for r in log["trajectory"]))
    by_drone: Dict[str, List[Dict[str, Any]]] = {}
    default = (log.meta.get("fleet") or [{"id": "Drone1"}])[0]["id"]
    for r in log["trajectory"]:
        by_drone.setdefault(r.get("drone", default), []).append(r)
    tracks = {d: Track.from_records(recs, t0) for d, recs in by_drone.items()}
    result_path = run_dir / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}
    return FilmRun(run_dir=run_dir, log=log, t0_ns=t0, tracks=tracks, result=result)
