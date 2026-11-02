"""High-resolution frames: what a VLM is sent (whole frame or tiles) and how tile answers merge."""
import threading

import numpy as np
import pytest

from vivid2.runner import mission_spec
from vivid2.stack.base import Assessment, Frame, UsageStats
from vivid2.stack.reference.reasoners import ASSESSMENT_SCHEMA, VLMReasoner, merge_tile_assessments
from vivid2.vlm.base import (FRAME_JPEG_MAX_SIDE, VLM_MAX_SIDE, VLMError, VLMProvider, decode_jpeg, encode_jpeg_bytes,
                             parse_tiles, vlm_views)


def _jpeg(w, h):
    img = np.zeros((h, w, 3), np.uint8)
    img[:, :, 1] = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
    return encode_jpeg_bytes(img, FRAME_JPEG_MAX_SIDE)


def test_small_frames_are_sent_unchanged():
    data = _jpeg(960, 720)
    [v] = vlm_views(data)
    assert v.jpeg == data and (v.width, v.height) == (960, 720) and v.to_frame(10, 20) == (10, 20)


def test_large_frames_are_downscaled_and_points_map_back():
    [v] = vlm_views(_jpeg(1920, 1440))
    assert max(v.width, v.height) == VLM_MAX_SIDE
    u, w = v.to_frame(v.width, v.height)
    assert u == pytest.approx(1920) and w == pytest.approx(1440)


def test_tiles_cover_the_frame_at_full_detail():
    views = vlm_views(_jpeg(1920, 1440), "2x2")
    assert [(v.width, v.height) for v in views] == [(960, 720)] * 4
    assert [(v.x0, v.y0) for v in views] == [(0, 0), (960, 0), (0, 720), (960, 720)]
    assert all(v.sx == v.sy == 1.0 for v in views)
    assert views[3].to_frame(100, 50) == (1060, 770)
    assert decode_jpeg(views[1].jpeg).shape == (720, 960, 3)
    assert vlm_views(_jpeg(1920, 1440), "2x2")[2].jpeg == views[2].jpeg  # deterministic (replay keys)
    with pytest.raises(ValueError):
        parse_tiles("2by2")


class TileVLM(VLMProvider):
    """Sees a person only in the tile whose top-left corner is (960, 720)."""

    model = "tile-vlm"

    def __init__(self, fail_tiles=()):
        self.prompts, self.fail_tiles, self.lock = [], set(fail_tiles), threading.Lock()

    def query(self, images, prompt, schema, system=None):
        assert schema is ASSESSMENT_SCHEMA
        img = decode_jpeg(images[0])
        with self.lock:
            self.prompts.append(prompt)
        idx = int(prompt.split("This image is tile ")[1].split(" ")[0]) - 1 if "This image is tile" in prompt else 0
        if idx in self.fail_tiles:
            raise VLMError("tile failed")
        found = idx == 3
        reply = {"target_found": found, "confidence": 0.9 if found else 0.1, "point": [500, 500] if found else [0, 0],
                 "description": f"tile {idx}", "hazards": ["trees"] if idx == 0 else [], "next_action": "continue"}
        assert img.shape == (720, 960, 3)
        return reply, UsageStats(latency_s=1.0 + idx, input_tokens=100, output_tokens=10, cost_usd=0.01, model=self.model)


def _frame(jpeg):
    return Frame(0, 1, decode_jpeg(jpeg), np.zeros(3), (1, 0, 0, 0), 80.0, jpeg=jpeg)


def test_tiled_reasoner_finds_the_target_in_its_tile_and_reports_frame_pixels(blocks_run):
    vlm = TileVLM()
    a, usage = VLMReasoner(vlm, tiles="2x2").assess(_frame(_jpeg(1920, 1440)), [], mission_spec(blocks_run))
    assert a.target_found and a.confidence == 0.9 and a.raw["tile"] == 3 and a.raw["tiles_found"] == [3]
    assert a.pixel == pytest.approx((960 + 480, 720 + 360))
    assert a.hazards == ["trees"]
    assert usage.input_tokens == 400 and usage.cost_usd == pytest.approx(0.04) and usage.latency_s == 4.0
    assert len(vlm.prompts) == 4 and all("of 4 of one camera frame" in p for p in vlm.prompts)


def test_tile_errors_are_ignored_unless_all_fail(blocks_run):
    m = mission_spec(blocks_run)
    a, _ = VLMReasoner(TileVLM(fail_tiles=[0, 1]), tiles="2x2").assess(_frame(_jpeg(1920, 1440)), [], m)
    assert a.target_found and a.raw["tiles_answered"] == 2
    a, _ = VLMReasoner(TileVLM(fail_tiles=[0, 1, 2, 3]), tiles="2x2").assess(_frame(_jpeg(1920, 1440)), [], m)
    assert not a.target_found and "all 4 tiles failed" in a.raw["error"]


def test_merge_prefers_the_most_confident_find():
    answers = [(Assessment(False, 0.3), UsageStats()), (Assessment(True, 0.6, (1, 1)), UsageStats()),
               (Assessment(True, 0.8, (2, 2)), UsageStats())]
    a, _ = merge_tile_assessments(answers)
    assert a.pixel == (2, 2) and a.raw["tiles_found"] == [1, 2]
