"""Search rules checked on a run's records (see docs/rules.md).

Runs recorded with the rules on (``meta["rules"]``, SIM_VERSION 7 and later) are checked; older
runs and hand-made logs are scored as they always were.

- A report must cite a recorded photo of the drone's search camera, taken by the drone that
  reports (``frame_id``). An uncited report is a false alarm, whatever it found.
- A decision must be about a recorded photo.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from vivid2.metrics.cameras import frame_camera, search_cameras
from vivid2.recorder import RunLog


def rules_active(meta: Dict[str, Any]) -> bool:
    return bool(meta.get("rules"))


def eligible(meta: Dict[str, Any]) -> bool:
    """Whether the stack may be ranked: not the reference oracle (it reads ground truth)."""
    rules = meta.get("rules")
    if rules:
        return bool(rules.get("eligible", True))
    return (meta.get("agent") or {}).get("perception") != "gt_perception"  # runs from before the rules


def _violation(kind: str, rec: Dict[str, Any], t_key: str = "t") -> Dict[str, Any]:
    return {"kind": kind, "t": rec.get(t_key, rec.get("t")), "frame_id": rec.get("frame_id"),
            **({"drone": rec["drone"]} if "drone" in rec else {})}


def check(log: RunLog) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """The reports that may be scored, and the rule violations (none when the rules are off)."""
    meta = log.meta
    reports = list(log["reports"])
    if not rules_active(meta):
        return reports, []
    frames = {f["frame_id"]: f for f in log["frames"]}
    search = search_cameras(meta)
    team = bool(meta.get("team"))
    ok: List[Dict[str, Any]] = []
    violations: List[Dict[str, Any]] = []
    for r in reports:
        f = frames.get(r.get("frame_id"))
        if f is None:
            violations.append(_violation("uncited_report", r, "t_report"))
        elif frame_camera(f) not in search:
            violations.append(_violation("report_cites_nav_frame", r, "t_report"))
        elif team and f.get("drone") != r.get("drone"):
            violations.append(_violation("report_cites_other_drone", r, "t_report"))
        else:
            ok.append(r)
    for d in log["decisions"]:
        if d.get("frame_id") not in frames:
            violations.append(_violation("decision_on_unrecorded_frame", d))
    return ok, violations


def uncited_reports(violations: List[Dict[str, Any]]) -> int:
    return sum(v["kind"] != "decision_on_unrecorded_frame" for v in violations)
