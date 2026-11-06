"""The YOLO baseline: tiling, YOLOX decoding, weights, thresholds. No onnxruntime needed (fake session)."""
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from vivid2.registry import AgentFactory
from vivid2.stack.base import Detection, Frame
from vivid2.stack.reference import detectors
from vivid2.stack.reference.perception import OnnxYoloDetector, decode_yolox, detect_yolo_format, nms, tile_grid
from vivid2.stack.reference.reasoners import DetectorReasoner


def test_tiles_cover_the_frame_with_overlap():
    tiles = tile_grid(1920, 1440)
    assert len(tiles) == 12 and all((x1 - x0, y1 - y0) == (640, 640) for x0, y0, x1, y1 in tiles)
    covered = np.zeros((1440, 1920), bool)
    for x0, y0, x1, y1 in tiles:
        covered[y0:y1, x0:x1] = True
    assert covered.all()
    xs = sorted({t[0] for t in tiles})
    assert min(b - a for a, b in zip(xs, xs[1:])) <= 640 * 0.8  # neighbours overlap by >= 20%
    assert tile_grid(500, 400) == [(0, 0, 500, 400)]


def _raw_yolox(cells=((39, 39),), xy=1.0, wh=2.0, obj=0.9, cls=0.9):
    """A raw YOLOX-s head output (1, 8400, 85) with a person in stride-8 grid cells (gx, gy)."""
    out = np.zeros((1, 8400, 85), np.float32)
    out[0, :, 2:4] = -10.0
    for gx, gy in cells:
        i = gy * 80 + gx
        out[0, i, :4] = [xy, xy, np.log(wh), np.log(wh)]
        out[0, i, 4], out[0, i, 5] = obj, cls
    return out


def test_yolox_boxes_are_decoded_from_the_grid():
    p = decode_yolox(_raw_yolox()[0], 640)
    i = 39 * 80 + 39
    assert np.allclose(p[i, :4], [320, 320, 16, 16])  # (39 + 1) * 8, exp(log 2) * 8
    assert detect_yolo_format((1, 8400, 85)) == "yolox" and detect_yolo_format((1, 84, 8400)) == "yolov8"
    with pytest.raises(ValueError):
        detect_yolo_format((1, "n", 85))


def test_nms_keeps_the_best_of_overlapping_boxes():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], float)
    assert sorted(nms(boxes, np.array([0.5, 0.9, 0.3]), 0.45)) == [1, 2]


class _Session:
    """A fake YOLOX-s session: every input shows one person centred at input pixel (320, 320)."""

    def __init__(self):
        self.calls = 0

    def get_inputs(self):
        return [SimpleNamespace(name="images")]

    def get_outputs(self):
        return [SimpleNamespace(shape=[1, 8400, 85])]

    def run(self, names, feed):
        assert feed["images"].shape == (1, 3, 640, 640)
        self.calls += 1
        return [_raw_yolox()]


def test_tiled_detections_come_back_in_frame_pixels():
    det = OnnxYoloDetector("yolox_s.onnx", session=_Session())
    frame = Frame(0, 1, np.zeros((1440, 1920, 3), np.uint8), np.zeros(3), (1, 0, 0, 0), 80.0)
    dets = det.detect(frame)
    assert det.session.calls == 13  # the whole frame + 12 tiles
    centres = {tuple(round(c) for c in d.center) for d in dets}
    assert (320, 320) in centres and (832, 320) in centres  # tiles at x 0 and 512
    assert (960, 960) in centres  # whole frame, scaled by 3
    assert all(d.score == pytest.approx(0.81) for d in dets)


def test_named_weights_are_checked_by_sha256(tmp_path, monkeypatch):
    src = tmp_path / "w.onnx"
    src.write_bytes(b"weights")
    good = hashlib.sha256(b"weights").hexdigest()
    monkeypatch.setattr(detectors, "detector_cache", lambda: tmp_path / "cache")
    monkeypatch.setitem(detectors.KNOWN_DETECTORS, "tiny", {
        "url": src.as_uri(), "sha256": good, "file": "tiny.onnx", "format": "yolox", "license": "MIT",
        "threshold": 0.3})
    path, entry = detectors.resolve_detector("tiny", progress=lambda s: None)
    assert open(path, "rb").read() == b"weights" and entry["name"] == "tiny"
    monkeypatch.setitem(detectors.KNOWN_DETECTORS, "bad", dict(detectors.KNOWN_DETECTORS["tiny"], sha256="0" * 64,
                                                               file="bad.onnx"))
    with pytest.raises(ValueError):
        detectors.resolve_detector("bad", progress=lambda s: None)
    assert not (tmp_path / "cache" / "bad.onnx").exists()
    assert detectors.resolve_detector(str(src))[1]["sha256"] == good  # a plain file works too
    with pytest.raises(ValueError):
        detectors.resolve_detector("no-such-detector")


class _FakeDetector:
    name = "onnx_yolo"
    model_name = "fake"

    def __init__(self, score):
        self.score = score

    def detect(self, frame):
        return [Detection("person", (10, 10, 30, 30), self.score)]


def test_one_threshold_per_variant(blocks_run):
    plain = AgentFactory("coverage_yolo", {})
    plain._detector = _FakeDetector(0.5)
    a = plain(blocks_run)
    assert plain.name == "coverage_yolo-yolox-s" and plain.yolo_threshold == 0.3
    assert (a.reasoner.threshold, a.min_confidence) == (0.3, 0.3) and not a.verify

    look = AgentFactory("coverage_yolo", {"verify": "on", "dip": "on", "yolo_threshold": 0.4})
    look._detector = _FakeDetector(0.5)
    b = look(blocks_run)
    assert look.name == "coverage_yolo-yolox-s-verify-dip"
    assert (b.reasoner.threshold, b.candidate_confidence, b.min_confidence) == (0.2, 0.2, 0.4) and b.verify
    assert b.describe()["params"]["reasoner"]["threshold"] == 0.2


def test_detector_reasoner_runs_the_detector_on_saved_frames(blocks_run):
    from vivid2.runner import mission_spec

    frame = Frame(0, 1, np.zeros((100, 100, 3), np.uint8), np.zeros(3), (1, 0, 0, 0), 80.0)
    m = mission_spec(blocks_run)
    found, _ = DetectorReasoner(_FakeDetector(0.4), 0.3).assess(frame, [], m)
    missed, _ = DetectorReasoner(_FakeDetector(0.4), 0.5).assess(frame, [], m)
    assert found.target_found and found.pixel == (20.0, 20.0) and not missed.target_found
