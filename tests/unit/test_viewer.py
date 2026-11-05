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


def test_high_resolution_frames_are_scaled_for_display_but_insets_keep_detail():
    rgb = np.zeros((1440, 1920, 3), np.uint8)
    rgb[700:702, 1000:1040] = (0, 0, 255)  # a 2 px red line: visible only at full resolution
    p = {"frame_id": 5, "image": rgb, "depth": None, "seg": None, "extra": "", "dets": [], "decision": None,
         "gt": [GTAnnotation("Hiker1", (1000, 690, 1040, 712), 60, (1000, 690, 1040, 712))]}
    out = _viewer()._draw_payload(p)
    assert out.shape == (720, 1440, 3)                     # same window as a 960x720 camera
    inset = out[720 - 52 - 288:720 - 52, 8:8 + 384]
    red = (inset[..., 2] > 200) & (inset[..., 1] < 50)
    assert red.sum() > 200                                  # the line, magnified ~3x from camera pixels


def _live_viewer():
    v = _viewer()
    import threading
    v._lock = threading.Lock()
    v._payload, v._preview_msg, v._preview, v._has_preview, v._dirty = None, None, None, False, False
    return v


def test_preview_until_the_first_scored_frame():
    from vivid2.stack.base import Frame

    v = _live_viewer()
    v.start_run("run1  survey")
    assert v.live
    cb = v.preview_callback()
    cb("topic", {"msg": 1})
    assert v._preview_msg == {"msg": 1}
    v.set_progress(23.4, 9.0)
    assert v._progress == "height 23 m  t 9 s"
    rgb = np.zeros((720, 960, 3), np.uint8)
    v.show_frame(Frame(t_ns=1, frame_id=1, rgb=rgb, depth=None, cam_pos=np.zeros(3), cam_q=(1, 0, 0, 0), fov_deg=80))
    assert not v.live
    cb("topic", {"msg": 2})                  # late preview messages are dropped once the test runs
    assert v._preview_msg is None
    v.set_progress(55.0, 30.0)
    assert v._progress == "height 23 m  t 9 s"
    v.start_run("run2  survey")              # the next run starts with the preview again
    assert v.live and v._payload is None


def test_preview_panel_has_the_scored_layout_and_is_marked():
    v = _live_viewer()
    v.start_run("run1  survey")
    img = np.full((720, 960, 3), 90, np.uint8)
    out = v._draw_preview(img, "taking off", "height 5 m  t 3 s")
    assert out.shape == (720, 1440, 3)                      # same window as a scored frame
    banner = out[0:36, 0:400]
    assert ((banner[..., 1] > 200) & (banner[..., 2] > 200)).any()   # yellow "not scored" banner
    assert out[400, 400].mean() == pytest.approx(90, abs=1)          # the preview image itself
    assert v._draw_preview(None, "loading the scene", "").shape == (720, 1440, 3)   # camera not known yet
    blank = v._draw_preview(None, "warm-up", "", has_preview=False)
    assert blank.shape == (720, 1440, 3)
