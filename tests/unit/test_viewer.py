"""Live view composite (vivid2 run --view): drawing only, no windows."""
import numpy as np
import pytest

from vivid2.stack.base import GTAnnotation
from vivid2.viewer import RunViewer

cv2 = pytest.importorskip("cv2")


def _viewer():
    v = object.__new__(RunViewer)  # no display thread / windows
    v.cv2, v._status = cv2, "run  agent"
    return v


def test_composite_has_rgb_depth_and_segmentation_panels():
    rgb = np.full((720, 960, 3), 40, np.uint8)
    seg = np.zeros_like(rgb)
    seg[300:320, 400:420] = (153, 108, 6)
    depth = np.full((720, 960), 75.0, np.float32)
    p = {"frame_id": 3, "image": rgb, "depth": depth, "seg": seg, "extra": "",
         "gt": [GTAnnotation("Hiker1", (400, 300, 420, 320), 400, (400, 300, 420, 320))],
         "dets": [{"bbox": [398, 298, 422, 322]}], "decision": {"target_found": True, "confidence": 0.8, "pixel": [410, 310]}}
    out = _viewer()._draw_payload(p)
    assert out.shape == (720, 1440, 3)
    assert tuple(out[300 - 6, 405]) == (60, 220, 60)               # green box: target well visible
    assert out[150, 1200].mean() == pytest.approx(75 / 150 * 255, abs=2)  # depth panel, half grey at 75 m
    assert (out[360 + 155, 960 + 205] == (153, 108, 6)).all()       # segmentation panel keeps palette colours


def test_missing_depth_and_segmentation_are_placeholders():
    p = {"frame_id": 1, "image": np.zeros((720, 960, 3), np.uint8), "depth": None, "seg": None,
         "gt": [GTAnnotation("Hiker1", (10, 10, 20, 20), 0, None)], "dets": [], "decision": None, "extra": ""}
    out = _viewer()._draw_payload(p)
    assert out.shape == (720, 1440, 3)
