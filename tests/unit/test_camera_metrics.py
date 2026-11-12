"""Scoring with several cameras: only the search camera's photos are scored; tilted photos cover only what they resolve."""
import json
import math

import numpy as np
import pytest

from vivid2.geo import rpy_to_quat
from vivid2.metrics.cameras import frame_camera, search_cameras, search_frames
from vivid2.metrics.perception import perception_metrics
from vivid2.metrics.planning import _cells, detectable_hits, planning_metrics
from vivid2.metrics.reasoning import reasoning_metrics
from vivid2.recorder import RunLog

NADIR = list(rpy_to_quat(0.0, -math.pi / 2, 0.0))
POLY = [[-500, -500], [-500, 500], [500, 500], [500, -500]]


def _write(run_dir, frames, dets=(), decisions=(), cameras=True):
    run_dir.mkdir(parents=True)
    meta = {"targets": [{"id": "Hiker1", "ned": [0, 0, 0]}], "search_area": {"polygon_ne": POLY, "ground_down_m": 0.0},
            "limits": {"timeout_s": 100}, "t0_ns": 0, "start_ned": [0, 0, 0]}
    if cameras:
        meta["cameras"] = {"Payload": {"role": "search"}, "Front": {"role": "nav"}}
    (run_dir / "meta.json").write_text(json.dumps(meta))
    for name, rows in (("frames", frames), ("detections", dets), ("decisions", decisions), ("trajectory", [])):
        (run_dir / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return RunLog.load(run_dir)


def _payload(fid, gt=True):
    return {"t": fid, "frame_id": fid, "camera": "Payload", "cam_pos": [0, 0, -60], "cam_q": NADIR, "width": 1920,
            "height": 1440, "fov_deg": 80.0,
            "gt": [{"object_id": "Hiker1", "bbox": [900, 700, 960, 760], "visible_px": 900, "contrast": 20.0}] if gt else []}


def _front(fid):
    return {"t": fid, "frame_id": fid, "camera": "Front", "cam_pos": [0, 0, -60],
            "cam_q": list(rpy_to_quat(0.0, math.radians(-10), 0.0)), "width": 640, "height": 480, "fov_deg": 90.0}


def _decision(fid, found):
    return {"t": fid, "frame_id": fid, "layer": "reasoner", "latency_s": 0.1,
            "output": {"target_found": found, "confidence": 0.9, "pixel": [930, 730] if found else None}}


def test_only_the_search_camera_s_photos_are_scored(tmp_path):
    log = _write(tmp_path / "r", [_payload(1), _front(2)],
                 dets=[{"t": 1, "frame_id": 1, "layer": "perception", "dets": [{"label": "person", "bbox": [900, 700, 960, 760], "score": 0.9}]},
                       {"t": 2, "frame_id": 2, "layer": "perception", "dets": [{"label": "person", "bbox": [10, 10, 50, 50], "score": 0.9}]}],
                 decisions=[_decision(1, True), _decision(2, True), _decision(99, True)])
    assert search_cameras(log.meta) == {"Payload"} and [f["frame_id"] for f in search_frames(log["frames"], log.meta)] == [1]
    p = perception_metrics(log)
    assert p["frames_evaluated"] == 1 and p["false_positives"] == 0 and p["precision"] == 1.0  # Front: not a false alarm
    r = reasoning_metrics(log)
    assert r["confusion"] == {"tp": 1, "fp": 0, "tn": 0, "fn": 0} and r["unscored_decisions"] == 2


def test_logs_without_cameras_are_all_search_photos(tmp_path):
    f = _payload(1)
    del f["camera"]
    log = _write(tmp_path / "r", [f], decisions=[_decision(1, True)], cameras=False)
    assert frame_camera(f) == "Payload" and search_cameras(log.meta) == {"Payload"}
    r = reasoning_metrics(log)
    assert r["confusion"]["tp"] == 1 and "unscored_decisions" not in r


def test_navigation_photos_add_no_coverage(tmp_path):
    one = planning_metrics(_write(tmp_path / "a", [_payload(1)]))
    both = planning_metrics(_write(tmp_path / "b", [_payload(1), _front(2)]))
    assert one["coverage_pct"] > 0 and one["covered_m2"] == both["covered_m2"]


def _tilted(tilt, fov, h=60.0):
    return {"cam_pos": [0, 0, -h], "cam_q": list(rpy_to_quat(0.0, math.radians(tilt), 0.0)), "width": 1920,
            "height": 1440, "fov_deg": fov, "gimbal": {"tilt_deg": tilt, "pan_deg": 0.0, "fov_deg": fov}}


def test_a_tilted_photo_covers_only_the_ground_where_a_person_would_show():
    cells = _cells([[-1000, -1000], [-1000, 1000], [1000, 1000], [1000, -1000]], 4.0)
    horizon = detectable_hits(_tilted(-5.0, 80.0), cells, 0.0)
    far = np.linalg.norm(cells[horizon], axis=1)
    # at 60 m a lying person keeps 80 px out to about 110 m, though the photo shows the horizon
    assert horizon.any() and far.max() < 120.0 and (cells[horizon][:, 0] > 0).all()
    zoomed = detectable_hits(_tilted(-5.0, 10.0), cells, 0.0)
    assert np.linalg.norm(cells[zoomed], axis=1).max() > 300.0  # zoom resolves people much farther away
    nadir = detectable_hits(_tilted(-90.0, 80.0), cells, 0.0)
    assert np.linalg.norm(cells[nadir], axis=1).max() == pytest.approx(math.hypot(50.0, 37.5), abs=8.0)
