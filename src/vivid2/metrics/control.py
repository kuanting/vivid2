"""Control metrics: path tracking, altitude hold, speed."""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from vivid2.recorder import RunLog


def point_to_polyline(p: np.ndarray, line: np.ndarray) -> float:
    best = float("inf")
    for a, b in zip(line[:-1], line[1:]):
        ab = b - a
        denom = float(ab @ ab)
        s = 0.0 if denom == 0 else float(np.clip((p - a) @ ab / denom, 0.0, 1.0))
        best = min(best, float(np.linalg.norm(p - (a + s * ab))))
    return best


def segment_speeds(t_s: np.ndarray, pos: np.ndarray, min_dt_s: float = 0.02) -> np.ndarray:
    """Speed per trajectory sample; samples closer than ``min_dt_s`` to the
    previous one (duplicate timestamps) get 0 instead of a huge spike."""
    speed = np.zeros(len(pos))
    if len(pos) > 1:
        dt = np.diff(t_s)
        dist = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        speed[1:] = np.where(dt >= min_dt_s, dist / np.maximum(dt, min_dt_s), 0.0)
    return speed


def control_metrics(log: RunLog) -> Optional[Dict[str, Any]]:
    traj = log["trajectory"]
    paths = [e for e in log["events"] if e.get("kind") == "path_start"]
    if not traj:
        return None
    t = np.asarray([r["t"] for r in traj], float) / 1e9
    pos = np.asarray([r["pos"] for r in traj], float)
    speed = segment_speeds(t, pos)

    out: Dict[str, Any] = {
        "max_speed_mps": float(speed.max()),
        "mean_speed_mps": float(speed[speed > 0.2].mean()) if (speed > 0.2).any() else 0.0,
    }
    if paths:
        # Tracking error while following each commanded path (until the next command).
        errs, alt_err = [], []
        cmd_times = [e["t"] / 1e9 for e in log["events"] if e.get("kind", "").startswith(("path_", "goto", "rtl", "hover"))]
        for ev in paths:
            wp = np.asarray(([ev["from"]] if ev.get("from") else []) + ev["waypoints"], float)
            if len(wp) < 2:
                continue
            t0 = ev["t"] / 1e9
            t1 = min([c for c in cmd_times if c > t0], default=t[-1])
            sel = (t >= t0 + 2.0) & (t <= t1)  # skip the initial transient
            errs.extend(point_to_polyline(p, wp) for p in pos[sel])
            alt_err.extend(pos[sel, 2] - wp[-1, 2])
        if errs:
            e = np.asarray(errs)
            out.update({
                "tracking_rmse_m": float(np.sqrt((e ** 2).mean())),
                "tracking_p95_m": float(np.percentile(e, 95)),
                "altitude_rmse_m": float(np.sqrt(np.mean(np.square(alt_err)))),
            })
    return out
