"""Control metrics: path tracking, altitude hold, speed."""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from vivid2.metrics.team import by_drone, fleet_of, is_team
from vivid2.recorder import RunLog


def point_to_polyline(p: np.ndarray, line: np.ndarray) -> float:
    best = float("inf")
    for a, b in zip(line[:-1], line[1:]):
        ab = b - a
        denom = float(ab @ ab)
        s = 0.0 if denom == 0 else float(np.clip((p - a) @ ab / denom, 0.0, 1.0))
        best = min(best, float(np.linalg.norm(p - (a + s * ab))))
    return best


def windowed_speeds(t_s: np.ndarray, pos: np.ndarray, window_s: float = 1.0) -> np.ndarray:
    """Average speed over the ``window_s`` following each trajectory sample.

    The sim's pose and clock update unevenly around camera captures, so speeds
    between consecutive 10 Hz samples spike to nearly twice the true value.
    Averaging over a second removes that jitter but still catches sustained
    speeding. Samples less than one window from the end get 0.
    """
    n = len(pos)
    speed = np.zeros(n)
    if n < 2:
        return speed
    j = np.searchsorted(t_s, t_s + window_s)
    i = np.nonzero(j < n)[0]
    jj = j[i]
    speed[i] = np.linalg.norm(pos[jj] - pos[i], axis=1) / (t_s[jj] - t_s[i])
    return speed


def _tracking(events, t: np.ndarray, pos: np.ndarray, errs: list, alt_err: list) -> None:
    """Tracking error while following each commanded path (until the next command)."""
    paths = [e for e in events if e.get("kind") == "path_start"]
    cmd_times = [e["t"] / 1e9 for e in events if e.get("kind", "").startswith(("path_", "goto", "rtl", "hover", "turn"))]
    for ev in paths:
        wp = np.asarray(([ev["from"]] if ev.get("from") else []) + ev["waypoints"], float)
        if len(wp) < 2:
            continue
        t0 = ev["t"] / 1e9
        t1 = min([c for c in cmd_times if c > t0], default=t[-1])
        sel = (t >= t0 + 2.0) & (t <= t1)  # skip the initial transient
        errs.extend(point_to_polyline(p, wp) for p in pos[sel])
        alt_err.extend(pos[sel, 2] - wp[-1, 2])


def control_metrics(log: RunLog) -> Optional[Dict[str, Any]]:
    traj = log["trajectory"]
    if not traj:
        return None
    if is_team(log.meta):  # each drone against its own commands; errors pooled over the team
        ids = [d["id"] for d in fleet_of(log.meta)]
        trajs, events = by_drone(traj, ids), by_drone(log["events"], ids)
        speeds, errs, alt_err = [], [], []
        for i, tr in trajs.items():
            if not tr:
                continue
            t = np.asarray([r["t"] for r in tr], float) / 1e9
            pos = np.asarray([r["pos"] for r in tr], float)
            speeds.append(windowed_speeds(t, pos))
            _tracking(events.get(i, []), t, pos, errs, alt_err)
        speed = np.concatenate(speeds)
    else:
        t = np.asarray([r["t"] for r in traj], float) / 1e9
        pos = np.asarray([r["pos"] for r in traj], float)
        speed = windowed_speeds(t, pos)
        errs, alt_err = [], []
        _tracking(log["events"], t, pos, errs, alt_err)

    out: Dict[str, Any] = {
        "max_speed_mps": float(speed.max()),
        "mean_speed_mps": float(speed[speed > 0.2].mean()) if (speed > 0.2).any() else 0.0,
    }
    if errs:
        e = np.asarray(errs)
        out.update({
            "tracking_rmse_m": float(np.sqrt((e ** 2).mean())),
            "tracking_p95_m": float(np.percentile(e, 95)),
            "altitude_rmse_m": float(np.sqrt(np.mean(np.square(alt_err)))),
        })
    return out
