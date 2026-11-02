"""Ground truth v2: hidden targets are not misses, partly visible ones are don't-care."""
from pathlib import Path

import numpy as np
import pytest

from vivid2.metrics.perception import perception_metrics
from vivid2.metrics.reasoning import reasoning_metrics
from vivid2.recorder import RunLog
from vivid2.sim.scene_builder import build_scene, captures
from vivid2.sim.segmentation import SegmentationError, assign_ids, visibility, visibility_state
from vivid2.stack.base import Frame, GTAnnotation
from vivid2.stack.reference.perception import GroundTruthPerception

COLOURS = np.stack([np.array([i, 255 - i, (7 * i) % 256], np.uint8) for i in range(256)])  # any 256 distinct colours


class SegWorld:
    def __init__(self, actors):
        self.actors = set(actors)
        self.calls = []

    def set_segmentation_id_by_name(self, name, sid, is_regex, use_owner):
        self.calls.append((name, sid, is_regex))
        return is_regex or name in self.actors


def test_ids_are_unique_and_unknown_actors_fail():
    world = SegWorld(["Hiker1", "Hiker2", "Backpack1"])
    ids = assign_ids(world, [{"id": "Hiker1"}, {"id": "Hiker2"}], [{"id": "Backpack1"}])
    assert ids == {"Hiker1": 1, "Hiker2": 2, "Backpack1": 101}
    assert world.calls[0] == (".*", 0, True)  # everything else becomes background first
    with pytest.raises(SegmentationError):
        assign_ids(SegWorld([]), [{"id": "Hiker1"}])


def test_visibility_counts_pixels_and_boxes_them():
    seg = np.zeros((20, 30, 3), np.uint8) + COLOURS[0]
    seg[5:9, 10:13] = COLOURS[1]  # 12 px of Hiker1
    vis = visibility(seg, {"Hiker1": 1, "Hiker2": 2}, COLOURS)
    assert vis["Hiker1"] == {"visible_px": 12, "bbox_vis": [10.0, 5.0, 13.0, 9.0]}
    assert vis["Hiker2"] == {"visible_px": 0, "bbox_vis": None}
    assert [visibility_state({"visible_px": n}) for n in (0, 5, 50)] == ["hidden", "partial", "visible"]
    assert visibility_state({"bbox": [0, 0, 1, 1]}) == "visible"  # ground truth v1


def _log(frames, decisions):
    return RunLog(Path("."), {"targets": [{"id": "Hiker1", "ned": [0, 0, 0]}]},
                  {"frames": frames, "decisions": decisions, "detections": []})


def _decision(fid, found, pixel=None):
    return {"frame_id": fid, "layer": "reasoner", "latency_s": 1.0,
            "output": {"target_found": found, "pixel": pixel}}


def test_reasoning_respects_visibility():
    box, vis_box = [100, 100, 140, 160], [110, 140, 130, 160]
    frames = [
        {"frame_id": 1, "gt": [{"object_id": "Hiker1", "bbox": box, "visible_px": 0, "bbox_vis": None}]},  # under canopy
        {"frame_id": 2, "gt": [{"object_id": "Hiker1", "bbox": box, "visible_px": 8, "bbox_vis": vis_box}]},  # a sliver
        {"frame_id": 3, "gt": [{"object_id": "Hiker1", "bbox": box, "visible_px": 300, "bbox_vis": vis_box}]},
        {"frame_id": 4, "gt": [], "distractors_visible": {"Backpack1": 90}},
    ]
    decisions = [_decision(1, False), _decision(2, True), _decision(3, True, [120, 150]), _decision(4, True)]
    r = reasoning_metrics(_log(frames, decisions))
    # "Not visible" under canopy is correct (TN, not FN); the sliver is not scored.
    assert r["confusion"] == {"tp": 1, "fp": 1, "tn": 1, "fn": 0} and r["dont_care"] == 1
    assert r["grounding_accuracy"] == 1.0 and r["fp_on_distractor"] == 1
    # Pointing at the occluded part of the sim's box is not grounded.
    r = reasoning_metrics(_log(frames, [_decision(3, True, [120, 110])]))
    assert r["grounding_accuracy"] == 0.0


def test_perception_ignores_partly_visible_targets():
    frames = [
        {"frame_id": 1, "t": 0, "cam_pos": [0, 0, -30],
         "gt": [{"object_id": "Hiker1", "bbox": [0, 0, 40, 40], "visible_px": 8, "bbox_vis": [0, 0, 4, 2]}]},
        {"frame_id": 2, "t": 1, "cam_pos": [0, 0, -30],
         "gt": [{"object_id": "Hiker1", "bbox": [0, 0, 40, 40], "visible_px": 400, "bbox_vis": [10, 10, 30, 30]}]},
    ]
    dets = [{"frame_id": 1, "layer": "perception", "dets": [{"bbox": [0, 0, 4, 2], "score": 0.9}]},
            {"frame_id": 2, "layer": "perception", "dets": [{"bbox": [10, 10, 30, 30], "score": 0.9}]}]
    log = RunLog(Path("."), {"targets": [{"id": "Hiker1", "ned": [0, 0, 0]}]},
                 {"frames": frames, "detections": dets})
    p = perception_metrics(log)
    assert p["gt_instances"] == 1 and p["false_positives"] == 0 and p["ap50"] == 1.0


def test_oracle_reports_only_visible_targets_with_tight_boxes():
    oracle = GroundTruthPerception([{"id": "Hiker1", "ned": [0, 0, 0]}, {"id": "Hiker2", "ned": [5, 0, 0]}])
    frame = Frame(0, 1, np.zeros((72, 96, 3), np.uint8), np.zeros(3), (1, 0, 0, 0), 80.0, gt=[
        GTAnnotation("Hiker1", (0, 0, 40, 40), 300, (10, 10, 30, 30)),
        GTAnnotation("Hiker2", (50, 0, 90, 40), 0, None),
    ])
    assert [d.bbox for d in oracle.detect(frame)] == [(10, 10, 30, 30)]


def test_distractors_are_spawned_and_segmentation_is_captured(root, blocks_run):
    spec = dict(blocks_run.spec, distractors=[{"id": "Backpack1", "asset": "1M_Cube", "ned": [5, 5, -0.2]}])
    run = type(blocks_run)(blocks_run.run_id, blocks_run.scenario_id, 0, {}, spec)
    scene = build_scene(run, str(root / "sim_config"), run.resolved_targets())
    names = [o["name"] for o in scene["spawn-objects"]["sim-packaged"]]
    assert names == ["Hiker1", "Backpack1"]
    assert captures(scene, "Payload", 3)


def test_visible_threshold_scales_with_image_size():
    from vivid2.sim.segmentation import min_visible_px, visibility_state
    assert min_visible_px() == 20 and min_visible_px(960, 720) == 20
    assert min_visible_px(1920, 1440) == 80          # the same patch of ground at twice the resolution
    assert min_visible_px(640, 480) == 20             # never below the base threshold
    g = {"visible_px": 50}
    assert visibility_state(g, frame_wh=(960, 720)) == "visible"
    assert visibility_state(g, frame_wh=(1920, 1440)) == "partial"
