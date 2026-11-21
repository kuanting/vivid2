"""Damage-task metrics: how well a run mapped what the disaster did (docs/damage.md).

Only for runs with a damage task (``meta["damage"]``). Every report must cite one of the drone's
own recorded search photos; a road, building or clue report must also show its item in that
photo (``vivid2.damage.geometry.view_mask``). Reports breaking these rules are violations: invalid
road, building and clue reports count as false positives, invalid area reports are dropped. Valid
area reports are clipped to the ground the run's search photos observed: credit only where the
drone looked. All checks run on the area of interest's 2 m cells.

    areas      mean over classes of IoU(ground truth, reported area within the observed ground)
    roads      F1 of "blocked" over the segments in the area of interest
    buildings  1/2 F1(damaged: partial or collapsed) + 1/2 F1(collapsed)
    clues      F1, a report matching an unclaimed clue within clue_tolerance_m
    buried     share of buried victims within BURIED_BUFFER_M of a reported area cell, or in a
               building reported damaged

F1 conventions: nothing to find and nothing claimed scores 1; claims with nothing to find score 0.
The damage score is 100 x the weighted mean of the parts the task asks for; a stack that reports
nothing scores 0 (not "missing"), so skipping the task never pays.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from vivid2.damage import BURIED_BUFFER_M, CLUE_KINDS, GRADE_LEVEL, MAX_POLYGON_VERTICES
from vivid2.damage.geometry import distance_to_polygon, point_in_view, raster_polygons
from vivid2.damage.run import DamageTask
from vivid2.metrics.cameras import frame_camera, search_cameras
from vivid2.geo import points_in_polygon
from vivid2.recorder import RunLog


def f1(tp: int, claims: int, positives: int) -> float:
    if positives == 0:
        return 1.0 if claims == 0 else 0.0
    if claims == 0 or tp == 0:
        return 0.0
    p, r = tp / claims, tp / positives
    return 2 * p * r / (p + r)


def _violation(kind: str, r: Dict[str, Any], why: str = "") -> Dict[str, Any]:
    return {"kind": kind, "t": r.get("t_report", r.get("t")), "frame_id": r.get("frame_id"),
            "report": r.get("kind"), **({"drone": r["drone"]} if "drone" in r else {}), **({"why": why} if why else {})}


def check_reports(log: RunLog, task: DamageTask) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[int, np.ndarray]]:
    """Valid damage reports (in report order), the violations, and each cited photo's observed cells."""
    meta = log.meta
    frames = {f["frame_id"]: f for f in log["frames"]}
    search = search_cameras(meta)
    team = bool(meta.get("team"))
    seen_by: Dict[int, np.ndarray] = {}
    valid, violations = [], []
    roads = {r["id"] for r in task.roads}
    buildings = {b["id"] for b in task.buildings}
    for r in sorted(log["damage"], key=lambda r: (r.get("t_report", r.get("t", 0)), r.get("t", 0))):
        f = frames.get(r.get("frame_id"))
        if f is None:
            violations.append(_violation("uncited_damage_report", r))
            continue
        if frame_camera(f) not in search:
            violations.append(_violation("damage_cites_nav_frame", r))
            continue
        if team and f.get("drone") != r.get("drone"):
            violations.append(_violation("damage_cites_other_drone", r))
            continue
        kind = r.get("kind")
        why = ""
        if kind == "area":
            poly = r.get("polygon_ne") or []
            if r.get("area_class") not in task.area_classes:
                why = f"unknown area class {r.get('area_class')!r}"
            elif r.get("op", "add") not in ("add", "remove"):
                why = f"unknown op {r.get('op')!r}"
            elif not 3 <= len(poly) <= MAX_POLYGON_VERTICES:
                why = f"polygon with {len(poly)} vertices"
        elif kind == "road":
            if r.get("segment") not in roads:
                why = f"unknown road segment {r.get('segment')!r}"
        elif kind == "building":
            if r.get("building") not in buildings:
                why = f"unknown building {r.get('building')!r}"
            elif r.get("grade") not in GRADE_LEVEL:
                why = f"unknown grade {r.get('grade')!r}"
        elif kind == "clue":
            if "clues" not in task.products:
                why = "the task does not ask for clues"
            elif r.get("clue_kind") not in CLUE_KINDS:
                why = f"unknown clue kind {r.get('clue_kind')!r}"
        else:
            why = f"unknown report kind {kind!r}"
        if not why and kind in ("road", "building"):
            fid = r["frame_id"]
            if fid not in seen_by:
                seen_by[fid] = task.seen(f)
            ok = (task.road_in_view(seen_by[fid], r["segment"]) if kind == "road"
                  else task.building_in_view(seen_by[fid], r["building"]))
            if not ok:
                violations.append(_violation("damage_not_in_cited_frame", r,
                                             f"{kind} {r.get('segment') or r.get('building')} not in photo {fid}"))
                continue
        if not why and kind == "clue":
            ned = list(r["ned"][:2]) + [float(task.terrain.down_at(r["ned"][0], r["ned"][1]))]
            if not point_in_view(f, ned, task.terrain):
                violations.append(_violation("damage_not_in_cited_frame", r, f"clue at {r['ned'][:2]} not in photo"))
                continue
        if why:
            violations.append(_violation("invalid_damage_report", r, why))
            continue
        valid.append(r)
    return valid, violations, seen_by


def damage_metrics(log: RunLog) -> Optional[Dict[str, Any]]:
    meta = log.meta
    if not meta.get("damage"):
        return None
    from vivid2.terrain import from_meta

    task = DamageTask.from_meta(meta, from_meta(meta))
    valid, violations, _ = check_reports(log, task)
    invalid = {k: sum(v["report"] == k for v in violations) for k in ("road", "building", "clue")}
    frames = [f for f in log["frames"] if frame_camera(f) in search_cameras(meta)]
    observed = np.zeros(len(task.cells), bool)
    for f in frames:
        observed |= task.seen(f)
    out: Dict[str, Any] = {"products": task.products, "reports": len(log["damage"]), "valid_reports": len(valid),
                           "observed_pct": float(100.0 * observed.mean()) if len(observed) else 0.0}
    comps: Dict[str, float] = {}

    # areas
    reported_any = np.zeros(len(task.cells), bool)
    if task.area_classes:
        per = {}
        for k in task.area_classes:
            polys = [r for r in valid if r["kind"] == "area" and r["area_class"] == k]
            rep = raster_polygons(polys, task.cells) & observed
            reported_any |= rep
            g = task.gt_cells[k]
            inter, union = int((g & rep).sum()), int((g | rep).sum())
            go = int((g & observed).sum())
            per[k] = {"iou": inter / union if union else 1.0,
                      "precision": inter / int(rep.sum()) if rep.any() else None,
                      "recall": inter / int(g.sum()) if g.any() else None,
                      "observed_share": go / int(g.sum()) if g.any() else None,
                      "observed_recall": inter / go if go else None,
                      "gt_m2": float(g.sum() * task.cell_m ** 2), "reported_m2": float(rep.sum() * task.cell_m ** 2)}
        comps["areas"] = float(np.mean([v["iou"] for v in per.values()]))
        out["areas"] = per

    # roads
    final_road: Dict[str, bool] = {}
    for r in valid:
        if r["kind"] == "road":
            final_road[r["segment"]] = bool(r["blocked"])
    if task.roads:
        gt_blocked = {r["id"] for r in task.roads if r["blocked"]}
        claimed = {s for s, b in final_road.items() if b}
        tp = len(claimed & gt_blocked)
        n_claims = len(claimed) + invalid["road"]
        comps["roads"] = f1(tp, n_claims, len(gt_blocked))
        out["roads"] = {"segments": len(task.roads), "blocked": sorted(gt_blocked), "reported_blocked": sorted(claimed),
                        "tp": tp, "claims": n_claims, "invalid": invalid["road"],
                        "precision": tp / n_claims if n_claims else None,
                        "recall": tp / len(gt_blocked) if gt_blocked else None,
                        "first_blocked_report_s": _first_time(log, valid, "road", gt_blocked)}

    # buildings
    final_grade: Dict[str, str] = {}
    for r in valid:
        if r["kind"] == "building":
            final_grade[r["building"]] = r["grade"]
    if task.buildings:
        g = {b["id"]: GRADE_LEVEL[b["grade"]] for b in task.buildings}
        p = {b: GRADE_LEVEL[final_grade.get(b, "none")] for b in g}
        inv = invalid["building"]
        dmg_tp = sum(g[b] >= 1 and p[b] >= 1 for b in g)
        col_tp = sum(g[b] == 2 and p[b] == 2 for b in g)
        f_dmg = f1(dmg_tp, sum(v >= 1 for v in p.values()) + inv, sum(v >= 1 for v in g.values()))
        f_col = f1(col_tp, sum(v == 2 for v in p.values()) + inv, sum(v == 2 for v in g.values()))
        comps["buildings"] = 0.5 * f_dmg + 0.5 * f_col
        confusion = [[sum(g[b] == i and p[b] == j for b in g) for j in range(3)] for i in range(3)]
        out["buildings"] = {"buildings": len(g), "f1_damaged": f_dmg, "f1_collapsed": f_col, "invalid": inv,
                            "confusion_gt_by_reported": confusion, "reported": len(final_grade)}

    # clues
    if task.clues:
        matched, false_alarms = _match_clues([r for r in valid if r["kind"] == "clue"], task.clues, task.clue_tolerance_m)
        tp = len(matched)
        n_claims = tp + len(false_alarms) + invalid["clue"]
        comps["clues"] = f1(tp, n_claims, len(task.clues))
        out["clues"] = {"clues": len(task.clues), "found": tp, "false_alarms": len(false_alarms) + invalid["clue"],
                        "kind_accuracy": (sum(m["kind_ok"] for m in matched.values()) / tp) if tp else None,
                        "matched": {k: {"error_m": round(v["error_m"], 2), "kind_ok": v["kind_ok"]} for k, v in matched.items()}}

    # buried victims
    if task.buried:
        covered = {}
        rep_cells = task.cells[reported_any]
        for b in task.buried:
            ne = np.asarray(b["ned"][:2], float)
            near = bool(len(rep_cells)) and float(np.min(np.linalg.norm(rep_cells - ne, axis=1))) <= \
                BURIED_BUFFER_M + task.cell_m / 2
            inside = False
            for bld in task.gt.buildings:
                if GRADE_LEVEL[final_grade.get(bld["id"], "none")] < 1:
                    continue
                if b.get("building") == bld["id"] or points_in_polygon(ne[None, :], bld["footprint_ne"])[0] or \
                        distance_to_polygon(ne[None, :], bld["footprint_ne"])[0] <= float(bld.get("spill_m", 0.0)):
                    inside = True
                    break
            covered[b["id"]] = bool(near or inside)
        comps["buried"] = sum(covered.values()) / len(covered)
        out["buried"] = {"buried": len(covered), "covered": covered}

    w = task.weights()
    den = sum(w[k] for k in comps)
    out["components"] = comps
    out["weights"] = {k: w[k] for k in comps}
    out["score"] = float(100.0 * sum(w[k] * v for k, v in comps.items()) / den) if den else 0.0
    out["violations"] = violations
    mapper = [d for d in log["decisions"] if d.get("layer") == "damage_mapper"]
    if mapper:
        out["mapper"] = {"calls": len(mapper), "errors": sum(bool((d.get("output") or {}).get("error")) for d in mapper),
                         "cost_usd": float(sum(d.get("cost_usd", 0.0) for d in mapper)),
                         "input_tokens": int(sum(d.get("input_tokens", 0) for d in mapper)),
                         "output_tokens": int(sum(d.get("output_tokens", 0) for d in mapper))}
    return out


def _first_time(log: RunLog, valid: List[Dict[str, Any]], kind: str, positives) -> Optional[float]:
    t0 = log.meta.get("t0_ns", 0)
    ts = [r.get("t_report", r["t"]) for r in valid if r["kind"] == kind and r.get("blocked") and r["segment"] in positives]
    return (min(ts) - t0) / 1e9 if ts else None


def _match_clues(reports: List[Dict[str, Any]], clues: List[Dict[str, Any]], tol: float):
    """Greedy chronological matching (as targets are matched): nearest unclaimed clue within ``tol``;
    a repeat report of a found clue is not a false alarm."""
    claimed: Dict[str, Dict[str, Any]] = {}
    false_alarms = []
    for rep in reports:
        p = np.asarray(rep["ned"][:2], float)
        best, best_d = None, float("inf")
        for c in clues:
            if c["id"] in claimed:
                continue
            d = float(np.linalg.norm(p - np.asarray(c["ned"][:2], float)))
            if d < best_d:
                best, best_d = c, d
        if best is not None and best_d <= tol:
            claimed[best["id"]] = {"error_m": best_d, "kind_ok": rep.get("clue_kind") == best.get("kind")}
        elif not any(np.linalg.norm(p - np.asarray(c["ned"][:2], float)) <= tol for c in clues if c["id"] in claimed):
            false_alarms.append(rep)
    return claimed, false_alarms
