import numpy as np

from vivid2.runner import mission_spec
from vivid2.sim.scene_builder import build_scene, camera_settings
from vivid2.stack.base import Frame, UsageStats
from vivid2.stack.reference.planners import LawnmowerPlanner
from vivid2.stack.reference.reasoners import ASSESSMENT_SCHEMA, VLMReasoner
from vivid2.metrics.planning import points_in_polygon
from vivid2.vlm.base import VLMError, VLMProvider
from vivid2.vlm.replay import ReplayVLM


class FakeVLM(VLMProvider):
    model = "fake-vlm"

    def __init__(self, reply=None, fail=False):
        self.reply = reply or {"target_found": True, "confidence": 0.8, "point": [500, 250],
                               "description": "person", "hazards": [], "next_action": "continue"}
        self.fail = fail
        self.calls = 0

    def query(self, images, prompt, schema, system=None):
        self.calls += 1
        if self.fail:
            raise VLMError("boom")
        assert schema is ASSESSMENT_SCHEMA or schema == ASSESSMENT_SCHEMA
        return dict(self.reply), UsageStats(latency_s=1.0, input_tokens=100, output_tokens=20, model=self.model)


def _frame(w=800, h=600):
    return Frame(0, 1, np.zeros((h, w, 3), np.uint8), np.zeros(3), (1, 0, 0, 0), 80.0)


def test_lawnmower_stays_inside_and_spans_area(blocks_run):
    m = mission_spec(blocks_run)
    wps = LawnmowerPlanner().plan(m, 80.0, (960, 720))
    pts = np.asarray(wps)
    assert len(wps) >= 4 and len(wps) % 2 == 0
    assert points_in_polygon(pts[:, :2], m.search_polygon_ne).all()
    assert np.allclose(pts[:, 2], -m.altitude_m)
    assert pts[:, 1].max() - pts[:, 1].min() > 60


def test_scene_builder_injects_targets_and_annotations(root, blocks_run):
    targets = blocks_run.resolved_targets()
    scene = build_scene(blocks_run, str(root / "sim_config"), targets)
    actor = scene["actors"][0]
    assert isinstance(actor["robot-config"], dict)
    payload = next(s for s in actor["robot-config"]["sensors"] if s["id"] == "Payload")
    assert payload["annotation-settings"]["object-ids"] == ["Hiker1"]
    spawned = scene["spawn-objects"]["sim-packaged"]
    assert spawned[0]["name"] == "Hiker1" and spawned[0]["asset-path"] == "1M_Cube"
    assert camera_settings(scene, "Payload")["fov-degrees"] == 80


def test_vlm_reasoner_maps_normalised_point(blocks_run):
    r = VLMReasoner(FakeVLM())
    a, usage = r.assess(_frame(), [], mission_spec(blocks_run))
    assert a.target_found and a.pixel == (400.0, 150.0)
    assert usage.input_tokens == 100


def test_vlm_reasoner_survives_provider_error(blocks_run):
    a, _ = VLMReasoner(FakeVLM(fail=True)).assess(_frame(), [], mission_spec(blocks_run))
    assert not a.target_found and "error" in a.raw


def test_replay_record_then_replay(tmp_path):
    import pytest

    inner = FakeVLM()
    cache = tmp_path / "vlm.jsonl"
    rec = ReplayVLM(str(cache), inner, "record")
    img = np.ones((4, 4, 3), np.uint8)
    out1, u1 = rec.query([img], "p", ASSESSMENT_SCHEMA)
    replayer = FakeVLM()
    rep = ReplayVLM(str(cache), replayer, "replay")
    out2, u2 = rep.query([img], "p", ASSESSMENT_SCHEMA)
    assert out1 == out2 and inner.calls == 1 and replayer.calls == 0
    assert (u2.input_tokens, u2.latency_s, u2.model) == (u1.input_tokens, u1.latency_s, u1.model)  # recorded usage
    with pytest.raises(VLMError):
        rep.query([img * 2], "p", ASSESSMENT_SCHEMA)  # a different frame is a miss
    other = FakeVLM()
    other.model = "another-model"
    with pytest.raises(VLMError):
        ReplayVLM(str(cache), other, "replay").query([img], "p", ASSESSMENT_SCHEMA)  # so is another model


class _PromptVLM(FakeVLM):
    def query(self, images, prompt, schema, system=None):
        self.prompt = prompt
        return super().query(images, prompt, schema, system)


def test_the_prompt_for_an_ordinary_search_photo_is_unchanged_so_cached_answers_replay(blocks_run):
    import math

    from vivid2.geo import rpy_to_quat

    m = mission_spec(blocks_run)
    vlm = _PromptVLM()
    nadir = Frame(0, 1, np.zeros((600, 800, 3), np.uint8), np.zeros(3), rpy_to_quat(0, -math.pi / 2, 0), 80.0)
    VLMReasoner(vlm).assess(nadir, [], m)
    assert f"The attached image is from a downward-looking camera, {m.altitude_m:.0f} m above the ground." in vlm.prompt
    assert "This photo is from" not in vlm.prompt
    tilted = Frame(0, 2, np.zeros((600, 800, 3), np.uint8), np.zeros(3), rpy_to_quat(0, math.radians(-20), 0), 40.0,
                   gimbal={"tilt_deg": -20.0, "pan_deg": 0.0, "fov_deg": 40.0})
    VLMReasoner(vlm).assess(tilted, [], m)
    assert "tilted 20 degrees below the horizon" in vlm.prompt and "40 degree field of view" in vlm.prompt
