"""The cut of "The Golden Hour" (about 2 minutes at 30 fps), captions filled from the run.

Story shots come from the Movie Render Queue renders (placeholders until rendered); data shots are
drawn from the run's logs (data_shots.py). Numbers in captions are the run's own results.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from demo_movie import compose as cp
from demo_movie import data_shots as ds
from demo_movie import typo as ty
from demo_movie.story import CHEN, LIN, Story

FPS = 30


def _mmss(t: float) -> str:
    m, s = divmod(max(0, int(round(t))), 60)
    return f"{m}:{s:02d}"


MODEL_NAMES = {"claude-opus-5-5": "Claude Opus 5.5", "claude-sonnet-5-5": "Claude Sonnet 5.5"}


def _decisions(story: Story, claude: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """VLM assessments with words: Claude's offline answers (claude.jsonl) first, then the run's own."""
    return list(claude.values()) + [d for d in story.run.log["decisions"] if (d.get("output") or {}).get("description")
                                     and d.get("model") not in (None, "", "threshold")]


def _words(story: Story, frame_id: Optional[int], claude: Dict[int, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """A VLM's own assessment of a frame, if one was made: its words, confidence and point."""
    if frame_id is None:
        return None
    for d in _decisions(story, claude):
        if d.get("frame_id") == frame_id:
            out = d.get("output") or {}
            return {"description": out.get("description", ""), "confidence": out.get("confidence"),
                    "pixel": out.get("pixel"), "found": out.get("target_found"),
                    "model": MODEL_NAMES.get(d.get("model") or "", d.get("model") or "")}
    return None


def best_words_frame(story: Story, tid: str, claude: Dict[int, Dict[str, Any]]) -> Optional[int]:
    """The photo of a hiker whose VLM assessment says 'found' with the highest confidence."""
    best, best_c = None, -1.0
    for d in _decisions(story, claude):
        out = d.get("output") or {}
        if d.get("target", tid) != tid:
            continue
        rec = next((f for f in story.run.log["frames"] if f["frame_id"] == d.get("frame_id")), None)
        shows = rec is not None and any(g.get("object_id") == tid and g.get("visible_px", 0) > 0 for g in rec.get("gt", []))
        if shows and out.get("target_found") and (out.get("confidence") or 0) > best_c:
            best, best_c = d["frame_id"], out.get("confidence") or 0
    if best is None:
        f = story.finds.get(tid)
        best = (f.closeup_frame or f.survey_frame) if f else None
    return best


def agent_label(story: Story) -> str:
    cfg = story.run.result.get("agent_config") or {}
    model = cfg.get("reasoner_model") or ""
    team = story.run.meta.get("team") or {}
    if cfg.get("perception") == "gt_perception":
        who = "reference team (ground-truth eyes)"
    else:
        who = MODEL_NAMES.get(model, model or cfg.get("agent", story.run.meta.get("agent", "")))
    return f"{who} · {team.get('mode', 'single')} · {team.get('size', 1)} drones"


def _find_words(story: Story, tid: str, claude: Dict[int, Dict[str, Any]], found: bool) -> Optional[int]:
    """The photo of a hiker whose Claude assessment is found (or not found), the most confident first."""
    best, best_c = None, -1.0
    for fid, d in claude.items():
        out = d.get("output") or {}
        if d.get("target", tid) == tid and bool(out.get("target_found")) == found and (out.get("confidence") or 0) > best_c:
            best, best_c = fid, out.get("confidence") or 0
    return best


def timeline(story: Story, plan: Dict[str, Any], render_root: Path, quad_dir: Optional[Path] = None,
             claude: Optional[Dict[int, Dict[str, Any]]] = None, gimbal_dir: Optional[Path] = None,
             contrast_run: Optional[Any] = None) -> List[cp.Item]:
    """The cut. ``contrast_run``: another run whose false alarm is shown as a counter-example (optional)."""
    run = story.run
    claude = claude or {}
    shots = {s["id"]: s for s in plan["shots"]}
    seq = lambda sid, n, offset=0, label="": cp.Seq(render_root, sid, n, offset, label or sid)
    map_png = render_root / "MAP" / "MAP.000001.png"
    plate = ds.MapPlate(run, shots.get("MAP", {}).get("map_plate"), map_png if map_png.exists() else None)
    res = run.result.get("metrics", {})
    mission, safety = res.get("mission", {}), res.get("safety", {})
    poly = np.asarray(run.meta["search_area"]["polygon_ne"], float)
    span = poly.max(0) - poly.min(0)
    black = lambda n: cp.Fn(n, lambda i: np.zeros((ty.H, ty.W, 3), np.uint8))
    items: List[cp.Item] = []

    # Act I - the accident
    items.append(cp.Item("S01", seq("S01", 180, label="establishing: ridge, forest, burning slope"),
                         [cp.Caption("Lishan, Taiwan · 1,985 m", "terrain from real elevation data", 30)], "fade:24"))
    items.append(cp.Item("S02", seq("S02", 150, label="two hikers leave the trail"),
                         [cp.Caption("16:40 — two hikers leave the trail", "", 18)], "dissolve:12"))
    # he goes down; the cut to black comes before he settles (lying at rest doesn't read as injured)
    items.append(cp.Item("S03", cp.Concat([seq("S03a", 55, label="he stumbles"), black(8)]),
                         [cp.Caption("He falls. His leg is injured —", "he can't move.", 8, 55)]))
    items.append(cp.Item("S04", seq("S04", 150, label="Lin cut off by the fire"),
                         [cp.Caption("She went for help.", "The fire cut her off.", 24)], "dissolve:10"))
    items.append(cp.Item("S05", ds.MapShot(run, plate, 90, "call", lkp=(0.0, 0.0)),
                         [cp.Caption("17:02 — emergency call",
                                     f"search area {span[1]:.0f} × {span[0]:.0f} m · fire no-fly zone below 30 m", 8)],
                         "dissolve:8"))

    # Act II - the search: what the drones can do
    items.append(cp.Item("S06", seq("S06", 180, label="four drones take off"),
                         [cp.Caption("A team of 4 drones", "central or distributed control · own pads · height layers", 40)]))
    items.append(cp.Item("S07", ds.MapShot(run, plate, 150, "timelapse", t0=story.takeoff_rt, t1=story.end_rt),
                         [cp.Caption("Each drone sweeps its own strip", "every photo and radio packet is logged", 10)],
                         "dissolve:8"))
    # the picture-in-picture: a stretch when all four drones are taking search photos
    starts = [min((f["rt"] for f in run.frames("Payload", d)), default=story.end_rt) for d in run.drones]
    t8 = max(story.chase["start"], max(starts) + 4.0)
    feeds = ds.FeedsGrid(run, 180, t8, t8 + 6.0)
    items.append(cp.Item("S08", cp.PIP(seq("S08", 180, label="chase over the canopy"), lambda i: feeds.tile(i),
                                       (1236, 520, 1860, 988), 20, 178, "SEARCH CAMERAS · real photos of all 4 drones"),
                         [cp.Caption("Every drone photographs the ground", "1920 × 1440 search camera · a photo every second", 24)]))
    if gimbal_dir and (Path(gimbal_dir) / "photos.json").exists():
        g = ds.GimbalFeed(gimbal_dir, 270, label="Lin")
        pxs = [max([x["visible_px"] for x in p["gt"] if x["object_id"] == g.target] or [0]) for p in g.photos]
        first = next((v for v in pxs if v > 0), 0)
        items.append(cp.Item("G1", g, [cp.Caption("The search camera is on a gimbal",
                                                  f"tilt · pan · zoom 80° → 8°: the hiker grows from {first:,} to {max(pxs):,} pixels", 20)]))
    else:
        items.append(cp.Item("G1", cp.Fn(270, lambda i: ty.card([("gimbal demo", 50, "display"),
                                                                  ("capture_gimbal.py", 28, "light")])), []))
    quad_frame = shots.get("S12", {}).get("match_frame") or run.frames("Payload")[0]["frame_id"]
    quad = (ds.SensorQuad.from_capture(quad_dir, 150) if quad_dir and (Path(quad_dir) / "seg.png").exists()
            else ds.SensorQuad.from_run(run, quad_frame, 150))
    items.append(cp.Item("S09", quad, [cp.Caption("Pixel-exact ground truth", "depth · segmentation · visibility of every target", 14)]))
    if story.dip and "S10" in shots:
        w = shots["S10"].get("run_window") or [story.dip["start"], story.dip["bottom"]]
        tail = ds.FrontFeed(run, story.dip["drone"], 90, w[1], (story.dip.get("end") or w[1] + 6.0))
        s10 = cp.Concat([seq("S10", 90, label="a dip between the crowns"), tail])
    else:
        s10 = seq("S10", 180, label="(no dip in this run)")
    items.append(cp.Item("S10", s10, [cp.Caption("Flying low among the trees", "any contact is a collision", 14)]))
    items.append(cp.Item("S11", seq("S11", 180, label="the fire front from above"),
                         [cp.Caption("Fire = no-fly zone", f"time inside it this run: {safety.get('no_fly_s', 0.0):.0f} s", 30)]))

    # Act III - how the AI finds a hiker (Chen, step by step), then Lin
    fc, fl = story.finds.get(CHEN), story.finds.get(LIN)
    if fc:
        f_unsure = _find_words(story, CHEN, claude, False) or fc.survey_frame
        f_found = _find_words(story, CHEN, claude, True) or fc.closeup_frame or fc.frame_id
        w_unsure, w_found = _words(story, f_unsure, claude), _words(story, f_found, claude)
        who = (w_found or w_unsure or {}).get("model") or "the AI"
        s12 = cp.Concat([seq("S12", 45, label="the search camera's view"),
                         ds.DetectionZoom(run, f_unsure, CHEN, 180, words=w_unsure, zoom_to=3.0)], dissolve=15)
        items.append(cp.Item("S12", s12, [cp.Caption("How the AI finds a hiker", "", 8, 50),
                                          cp.Caption(f"1 · {who} checks every search photo",
                                                     "from 55 m: a speck of red — not sure", 56)], "dissolve:10"))
        items.append(cp.Item("S13", seq("S13", 180, label="the closer look at Chen"),
                             [cp.Caption("2 · Not sure? Fly closer",
                                         f"Drone {fc.drone[-1]} descends for a closer photo", 16)]))
        items.append(cp.Item("A4", ds.DetectionZoom(run, f_found, CHEN, 180, words=w_found, zoom_to=4.0),
                             [cp.Caption("3 · The closer photo: found", "the AI marks a point and says why", 14)]))
        if w_found and w_found.get("pixel"):
            geo = ds.GeoLocate(run, f_found, CHEN, w_found["pixel"], 180, who=who)
            items.append(cp.Item("A5", geo, [cp.Caption("4 · Point + depth → position",
                                                        f"{geo.err:.1f} m from where Chen lies" if geo.err is not None else "", 120)],
                                 "dissolve:8", grade=False))
        else:
            items.append(cp.Item("A5", cp.Fn(180, lambda i: ty.card([("(no point to locate)", 40, "regular")])), []))
        items.append(cp.Item("A6", ds.RadioShot(run, plate, 90, fc.drone, fc.report_rt, f"FOUND · Chen · {fc.drone}"),
                             [cp.Caption("5 · Tell the team", f"reported {fc.error_m:.1f} m from the truth · "
                                         f"T+{_mmss(fc.report_rt - story.takeoff_rt)}", 6)], "dissolve:8"))
    else:
        for sid, n in (("S12", 210), ("S13", 180), ("A4", 180), ("A5", 180), ("A6", 90)):
            items.append(cp.Item(sid, seq(sid, n, label="(Chen not found)"), []))
    if fl:
        wl = _find_words(story, LIN, claude, True) or fl.closeup_frame or fl.survey_frame
        s14 = cp.Concat([seq("S14", 120, label="Lin found"),
                         ds.DetectionZoom(run, wl, LIN, 105, words=_words(story, wl, claude), zoom_to=3.0)], dissolve=15)
    else:
        s14 = seq("S14", 210, label="(Lin not found)")
    found = mission.get("targets_found", 0)
    tt = max((mission.get("time_to_find_s") or {}).values() or [0])
    items.append(cp.Item("S14", s14, [cp.Caption("Lin is found the same way", "", 16, 110),
                                      cp.Caption(f"{found} of {mission.get('targets_total', 2)} found in {_mmss(tt)}",
                                                 f"{safety.get('collisions', 0)} collisions · {mission.get('false_alarms', 0)} "
                                                 f"false alarms · {safety.get('no_fly_s', 0.0):.0f} s in the no-fly zone", 120)]))
    if contrast_run is not None:
        fa = contrast_run.reports()[1] if len(contrast_run.reports()) > 1 else (contrast_run.reports() or [None])[0]
        if fa:
            dets = [x for d in contrast_run.log["detections"] if d["frame_id"] == fa["frame_id"] for x in d["dets"]]
            best = max(dets, key=lambda x: x.get("score", 0)) if dets else None
            c = ((best["bbox"][0] + best["bbox"][2]) / 2, (best["bbox"][1] + best["bbox"][3]) / 2) if best else None
            yz = ds.DetectionZoom(contrast_run, fa["frame_id"], None, 120, zoom_to=3.0, show_detector=True, center=c)
            items.append(cp.Item("Y1", yz, [cp.Caption("Not every AI gets it right",
                                                       "another stack: a detector took a spot fire for a person — scored as a false alarm", 14)],
                                 "dissolve:8"))
    items.append(cp.Item("S15", seq("S15", 60, label="landing at dusk"), [], "dissolve:8"))

    # Act IV - scored
    items.append(cp.Item("S16", ds.Scorecard(run, 180, agent_label(story)),
                         [cp.Caption("Same mission, any AI stack", "YOLO · Claude · yours", 100)], "dissolve:10", grade=False))
    end = ty.card([("VIVID 2", 120, "display"), ("Virtual Validation for AI Drones", 44, "light"),
                   ("open benchmark · MIT · github.com/kuanting/vivid2", 30, "regular")])
    ty.text(end, (ty.W // 2, ty.H - 92),
            f"Flights re-enacted from VIVID 2 run {run.meta.get('run_id', '')} ({agent_label(story)}); the search-camera demo is a live capture.",
            22, "light", color=(170, 170, 170), anchor="ma", shadow=False)
    ty.text(end, (ty.W // 2, ty.H - 58), "Claude's words are its real answers on the recorded photos. All on-screen data is real.",
            22, "light", color=(170, 170, 170), anchor="ma", shadow=False)
    items.append(cp.Item("S17", cp.Fn(120, lambda i: end), [], "dissolve:12", grade=False))
    return items
