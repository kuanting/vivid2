"""Mission-level metrics: were the targets found, how fast, how accurately."""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from vivid2.metrics import rules
from vivid2.recorder import RunLog


def match_reports(
    reports: List[Dict[str, Any]], targets: List[Dict[str, Any]], tolerance_m: float
) -> Dict[str, Any]:
    """Greedy chronological matching: each report claims the nearest unclaimed
    target within tolerance (horizontal distance); unmatched reports are false alarms."""
    claimed: Dict[str, Dict[str, Any]] = {}
    false_alarms = []
    for rep in sorted(reports, key=lambda r: r["t"]):
        p = np.asarray(rep["ned"][:2], float)
        best, best_d = None, float("inf")
        for tgt in targets:
            if tgt["id"] in claimed:
                continue
            d = float(np.linalg.norm(p - np.asarray(tgt["ned"][:2], float)))
            if d < best_d:
                best, best_d = tgt, d
        if best is not None and best_d <= tolerance_m:
            # timed when reported (the agent's thinking counts); older runs only have the frame time
            claimed[best["id"]] = {"t": rep.get("t_report", rep["t"]), "error_m": best_d,
                                   "confidence": rep.get("confidence")}
        else:
            # A repeat report of an already-found target is not a false alarm.
            dup = any(
                np.linalg.norm(p - np.asarray(t["ned"][:2], float)) <= tolerance_m
                for t in targets
                if t["id"] in claimed
            )
            if not dup:
                false_alarms.append(rep)
    return {"matched": claimed, "false_alarms": false_alarms}


def mission_metrics(log: RunLog) -> Dict[str, Any]:
    meta = log.meta
    targets = meta["targets"]
    tol = meta.get("success", {}).get("localization_tolerance_m", 10.0)
    require_all = meta.get("success", {}).get("require_all_targets", True)
    t0 = meta["t0_ns"]
    timeout_s = meta["limits"]["timeout_s"]

    cited, violations = rules.check(log)  # an uncited report is a false alarm, whatever it found
    m = match_reports(cited, targets, tol)
    uncited = rules.uncited_reports(violations)
    found = m["matched"]
    ttf = {tid: (v["t"] - t0) / 1e9 for tid, v in found.items()}
    n_found = len(found)
    success = (n_found == len(targets) if require_all else n_found > 0) and meta.get("end_reason") != "error"
    return {
        "success": bool(success),
        "targets_total": len(targets),
        "targets_found": n_found,
        "found_fraction": n_found / len(targets),
        "time_to_find_s": ttf,
        "mean_time_to_find_s": float(np.mean(list(ttf.values()))) if ttf else None,
        "localization_error_m": {tid: v["error_m"] for tid, v in found.items()},
        "mean_localization_error_m": float(np.mean([v["error_m"] for v in found.values()])) if found else None,
        "false_alarms": len(m["false_alarms"]) + uncited,
        **({"uncited_reports": uncited} if uncited else {}),
        "mission_time_s": (meta.get("end_ns", t0) - t0) / 1e9,
        "timeout_s": timeout_s,
        "end_reason": meta.get("end_reason"),
    }
