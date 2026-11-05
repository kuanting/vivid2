"""Ground truth v3: fog washes a target out of the colour image, not out of segmentation.

A target counts as visible only if it also stands out from the ground around it
(CIELAB Delta E >= MIN_CONTRAST). And each run starts with the camera's exposure
settled to its own light, not the previous run's.
"""
import asyncio
from pathlib import Path

import numpy as np
import pytest

from vivid2 import runner
from vivid2.metrics.reasoning import reasoning_metrics
from vivid2.recorder import RunLog
from vivid2.sim.segmentation import MIN_CONTRAST, contrast, visibility, visibility_state
from vivid2.stack.base import Frame, GTAnnotation
from vivid2.stack.reference.perception import GroundTruthPerception

COLOURS = np.stack([np.array([i, 255 - i, (7 * i) % 256], np.uint8) for i in range(256)])
GROUND = np.array([60, 90, 120], np.uint8)       # brown (BGR)
JACKET = np.array([40, 40, 200], np.uint8)       # red
FOG = np.array([200, 205, 210], np.float32)      # pale grey


def _scene(fog: float = 0.0):
    """A 1920 x 1440 frame: brown ground, a red jacket of 20 x 40 px, and the matching segmentation."""
    rgb = np.zeros((1440, 1920, 3), np.uint8) + GROUND
    seg = np.zeros((1440, 1920, 3), np.uint8) + COLOURS[0]
    rgb[700:740, 900:920] = JACKET
    seg[700:740, 900:920] = COLOURS[1]
    if fog:  # fog veils everything towards the fog colour
        rgb = ((1 - fog) * rgb + fog * FOG).astype(np.uint8)
    return rgb, seg


def test_contrast_falls_as_fog_thickens():
    values = []
    for fog in (0.0, 0.5, 0.9, 0.98):
        rgb, seg = _scene(fog)
        values.append(visibility(seg, {"Hiker1": 1}, COLOURS, rgb)["Hiker1"]["contrast"])
    assert values == sorted(values, reverse=True)
    assert values[0] > 30 and values[-1] < MIN_CONTRAST


def test_without_a_colour_image_there_is_no_contrast_and_no_change():
    _, seg = _scene()
    assert "contrast" not in visibility(seg, {"Hiker1": 1}, COLOURS)["Hiker1"]


def test_the_ring_leaves_out_other_labelled_objects():
    rgb, seg = _scene()
    rgb[700:740, 920:940] = JACKET  # a red decoy right next to the hiker
    seg[700:740, 920:940] = COLOURS[101]
    vis = visibility(seg, {"Hiker1": 1, "Decoy1": 101}, COLOURS, rgb)
    assert vis["Hiker1"]["contrast"] > 30  # judged against the ground, not against the decoy


def test_contrast_is_none_when_nothing_surrounds_the_object():
    mask = np.ones((10, 10), bool)
    assert contrast(np.zeros((10, 10, 3), np.uint8), mask, [0, 0, 10, 10]) is None


def test_a_washed_out_target_is_dont_care():
    assert visibility_state({"visible_px": 800, "contrast": MIN_CONTRAST + 1}, frame_wh=(1920, 1440)) == "visible"
    assert visibility_state({"visible_px": 800, "contrast": MIN_CONTRAST - 1}, frame_wh=(1920, 1440)) == "partial"
    assert visibility_state({"visible_px": 800}, frame_wh=(1920, 1440)) == "visible"  # ground truth v2
    assert visibility_state({"visible_px": 0, "contrast": None}) == "hidden"


def test_saying_no_target_in_thick_fog_is_neither_right_nor_wrong():
    gt = {"object_id": "Hiker1", "bbox": [900, 700, 920, 740], "visible_px": 800, "bbox_vis": [900, 700, 920, 740]}
    frames = [{"frame_id": 1, "width": 1920, "height": 1440, "gt": [dict(gt, contrast=2.0)]},
              {"frame_id": 2, "width": 1920, "height": 1440, "gt": [dict(gt, contrast=40.0)]}]
    decisions = [{"frame_id": f, "layer": "reasoner", "latency_s": 1.0, "output": {"target_found": False}} for f in (1, 2)]
    r = reasoning_metrics(RunLog(Path("."), {"targets": [{"id": "Hiker1", "ned": [0, 0, 0]}]},
                                 {"frames": frames, "decisions": decisions, "detections": []}))
    assert r["dont_care"] == 1 and r["confusion"]["fn"] == 1


def test_the_oracle_does_not_see_through_fog():
    oracle = GroundTruthPerception([{"id": "Hiker1", "ned": [0, 0, 0]}])
    frame = Frame(0, 1, np.zeros((1440, 1920, 3), np.uint8), np.zeros(3), (1, 0, 0, 0), 80.0, gt=[
        GTAnnotation("Hiker1", (900, 700, 920, 740), 800, (900, 700, 920, 740), contrast=2.0)])
    assert oracle.detect(frame) == []
    frame.gt[0].contrast = 40.0
    assert len(oracle.detect(frame)) == 1


class _ExposureDrone:
    """A camera whose brightness starts at the previous run's level and adapts a step per capture."""

    def __init__(self, start=0.0, target=120.0, rate=0.5):
        self.level, self.target, self.rate, self.captures = start, target, rate, 0

    def get_images(self, camera, types):
        from projectairsim.types import ImageType
        self.captures += 1
        img = np.full((4, 4, 3), int(round(self.level)), np.uint8)
        self.level += self.rate * (self.target - self.level)
        return {ImageType.SCENE: {"data": img.tobytes(), "height": 4, "width": 4}}


def test_exposure_settles_before_the_mission(monkeypatch):
    pytest.importorskip("projectairsim")  # the Project AirSim client (not on PyPI, so not in CI)
    monkeypatch.setattr(runner, "EXPOSURE_PERIOD_S", 0.0)
    drone = _ExposureDrone(start=0.0)  # first capture black, as after a fog run
    out = asyncio.run(runner.settle_exposure(drone, ["Payload"]))
    p = out["Payload"]
    assert p["settled"] and 3 <= p["captures"] < runner.EXPOSURE_MAX_CAPTURES
    assert abs(p["mean_grey"][-1] - 120) < 3


def test_exposure_gives_up_after_the_capture_limit(monkeypatch):
    pytest.importorskip("projectairsim")  # the Project AirSim client (not on PyPI, so not in CI)
    monkeypatch.setattr(runner, "EXPOSURE_PERIOD_S", 0.0)
    drone = _ExposureDrone(start=255.0, rate=0.01)  # adapts too slowly to settle
    out = asyncio.run(runner.settle_exposure(drone, ["Payload", "Front"]))
    assert not out["Payload"]["settled"] and out["Payload"]["captures"] == runner.EXPOSURE_MAX_CAPTURES
    assert set(out) == {"Payload", "Front"}
