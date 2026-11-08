"""Team runs: which drone a record belongs to, and how the team worked together.

A team run's ``meta.json`` has ``team`` ({mode, size}) and ``fleet`` (every drone's id and launch
pad), and its records carry a ``drone`` field. Single-drone runs have neither and are scored
exactly as before; the layer metrics only take their team branch when ``is_team(meta)``.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from vivid2.metrics.mission import match_reports
from vivid2.recorder import RunLog

CONTACT_GAP_S = 1.0  # collision records of one pair closer together than this are one contact


def is_team(meta: Dict[str, Any]) -> bool:
    return bool(meta.get("team"))


def fleet_of(meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Every drone's id and launch pad (one drone: Drone1 at ``start_ned``)."""
    if meta.get("fleet"):
        return [{"id": d["id"], "start_ned": d["start_ned"]} for d in meta["fleet"]]
    return [{"id": "Drone1", "start_ned": meta.get("start_ned")}]


def by_drone(records: List[Dict[str, Any]], ids: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """Records per drone, every drone present; records without a ``drone`` belong to the first."""
    out: Dict[str, List[Dict[str, Any]]] = {i: [] for i in ids}
    for r in records:
        out.setdefault(r.get("drone", ids[0]), []).append(r)
    return out


def contact_events(collisions: List[Dict[str, Any]], gap_s: float = CONTACT_GAP_S) -> List[Dict[str, Any]]:
    """One record per contact. The simulator reports a contact on every tick it lasts, and a
    drone-drone contact from both drones; those count once per pair."""
    out, last = [], {}
    for c in sorted(collisions, key=lambda c: c.get("t") or 0):
        other = c.get("other_drone")
        key = ("pair",) + tuple(sorted((c.get("drone", ""), other))) if other else (c.get("drone"), c.get("object_name"))
        t = (c.get("t") or 0) / 1e9
        if key in last and t - last[key] <= gap_s:
            last[key] = t
            continue
        last[key] = t
        out.append(c)
    return out


def separation(traj: List[Dict[str, Any]], ids: List[str], limit_m: float) -> Dict[str, Any]:
    """Closest approach between any two drones, and pair-seconds closer than ``limit_m`` (3D).

    The trajectory poller samples every drone at the same sim time, so samples pair up by ``t``.
    """
    at: Dict[int, Dict[str, np.ndarray]] = {}
    for r in traj:
        at.setdefault(int(r["t"]), {})[r.get("drone", ids[0])] = np.asarray(r["pos"], float)
    times = sorted(t for t, d in at.items() if len(d) == len(ids))
    if len(ids) < 2 or not times:
        return {"min_separation_m": None, "separation_violation_s": 0.0, "closest_pair": None}
    best, pair, violation = float("inf"), None, 0.0
    for k, t in enumerate(times):
        dt = (times[k + 1] - t) / 1e9 if k + 1 < len(times) else 0.0
        pos = at[t]
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                d = float(np.linalg.norm(pos[ids[a]] - pos[ids[b]]))
                if d < best:
                    best, pair = d, [ids[a], ids[b]]
                if d < limit_m:
                    violation += dt
    return {"min_separation_m": best, "separation_violation_s": violation, "closest_pair": pair}


def team_metrics(log: RunLog) -> Dict[str, Any]:
    """How the team shared the work (reported, not weighted into the overall score)."""
    meta = log.meta
    ids = [d["id"] for d in fleet_of(meta)]
    tol = meta.get("success", {}).get("localization_tolerance_m", 10.0)
    reports = log["reports"]
    m = match_reports(reports, meta["targets"], tol)
    t0 = meta["t0_ns"]
    timeout = float(meta["limits"]["timeout_s"])
    found_t = [(v["t"] - t0) / 1e9 for v in m["matched"].values()]
    all_found = len(m["matched"]) == len(meta["targets"])
    per_drone = {i: {"reports": 0, "frames": 0} for i in ids}
    for r in reports:
        per_drone.setdefault(r.get("drone", ids[0]), {"reports": 0, "frames": 0})["reports"] += 1
    for f in log["frames"]:
        per_drone.setdefault(f.get("drone", ids[0]), {"reports": 0, "frames": 0})["frames"] += 1
    flew = {r.get("drone", ids[0]) for r in log["events"] if r.get("kind") == "takeoff"}
    comms = log["comms"]
    return {
        **({"comms": comms_metrics(comms, ids)} if comms else {}),
        "mode": meta["team"].get("mode"),
        "size": int(meta["team"].get("size", len(ids))),
        "drones_used": len(flew) if flew else None,
        # time until every target was found; a target never found counts the whole timeout (censored)
        "time_to_find_all_s": max(found_t) if all_found and found_t else timeout,
        "all_found": all_found,
        "duplicate_reports": len(reports) - len(m["matched"]) - len(m["false_alarms"]),
        "per_drone": per_drone,
    }


def comms_metrics(comms: List[Dict[str, Any]], ids: List[str]) -> Dict[str, Any]:
    """What the drones said to each other: messages, bytes, and what the radio lost."""
    sent = {}
    for c in comms:
        sent.setdefault((c["from"], c["seq"]), c["bytes"])  # a broadcast is one message, many deliveries
    lost: Dict[str, int] = {}
    for c in comms:
        if not c["delivered"]:
            lost[c["reason"]] = lost.get(c["reason"], 0) + 1
    delays = [(c["t_deliver"] - c["t_send"]) / 1e9 for c in comms if c["delivered"]]
    return {
        "messages": len(sent),
        "bytes": int(sum(sent.values())),
        "deliveries": sum(1 for c in comms if c["delivered"]),
        "lost": lost,
        "mean_delay_s": float(np.mean(delays)) if delays else None,
        "per_drone": {i: {"messages": sum(1 for (f, _) in sent if f == i),
                          "bytes": int(sum(b for (f, _), b in sent.items() if f == i))} for i in ids},
    }
