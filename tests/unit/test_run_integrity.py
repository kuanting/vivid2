"""Resume, rescoring, schema and end-of-run handling must not corrupt results."""
import asyncio
import copy
import json
from pathlib import Path

import jsonschema
import pytest

from conftest import FakeWorld, write_meta
from vivid2.metrics.mission import mission_metrics
from vivid2.metrics.safety import safety_metrics
from vivid2.recorder import Recorder, RunLog
from vivid2.registry import AGENTS, AgentFactory
from vivid2.runner import run_config_hash, score_run_dir
from vivid2.scenario import Scenario
from vivid2.stack.base import Assessment, Reasoner, UsageStats
from vivid2.stack.reference.agents import SearchAgent
from vivid2.stack.reference.planners import LawnmowerPlanner

ONLY_MISSION = {"mission": 1.0, "perception": 0.0, "reasoning": 0.0, "planning": 0.0, "control": 0.0, "safety": 0.0}


class NeverReasoner(Reasoner):
    name = "never"

    def assess(self, frame, detections, mission):
        return Assessment(False), UsageStats(model="never")


def test_rescore_uses_campaign_weights_from_meta(tmp_path, blocks_run):
    run_dir = tmp_path / "blind" / blocks_run.run_id
    rec = Recorder(run_dir)
    world = FakeWorld(blocks_run, rec)
    agent = SearchAgent("blind", LawnmowerPlanner(), NeverReasoner(), frame_period_s=0.0)
    asyncio.run(agent.run(world))
    write_meta(rec, blocks_run, agent.describe(), world.targets, world.t_ns)
    rec.close()
    assert score_run_dir(run_dir)["scores"]["overall"] > 0  # default weights: planning and safety still score

    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    meta["weights"] = ONLY_MISSION
    (run_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    assert score_run_dir(run_dir)["scores"]["overall"] == 0.0


def test_config_hash_tracks_agent_and_scenario_changes(blocks_run):
    base = run_config_hash(blocks_run, AgentFactory("coverage_gt")(blocks_run).describe())
    assert base == run_config_hash(blocks_run, AgentFactory("coverage_gt")(blocks_run).describe())
    wider = AgentFactory("coverage_gt", {"sidelap": 0.4})(blocks_run).describe()
    assert run_config_hash(blocks_run, wider) != base
    edited = copy.deepcopy(blocks_run)
    edited.spec["limits"]["timeout_s"] += 60
    assert run_config_hash(edited, AgentFactory("coverage_gt")(edited).describe()) != base


def test_recorder_ignores_writes_after_close(tmp_path):
    rec = Recorder(tmp_path / "r")
    rec.close()
    rec.log("collisions", {"t": 1})  # a late sim callback must not raise
    assert (tmp_path / "r" / "collisions.jsonl").read_text(encoding="utf-8") == ""


def test_error_run_is_never_a_success_and_hard_timeout_counts():
    meta = {"targets": [{"id": "A", "ned": [0, 0, 0]}], "t0_ns": 0, "limits": {"timeout_s": 100},
            "search_area": {"polygon_ne": [[-10, -10], [10, -10], [10, 10], [-10, 10]]}, "start_ned": [0, 0, 0]}
    streams = {"reports": [{"t": 1, "ned": [1, 0, 0]}],
               "trajectory": [{"t": 0, "pos": [0, 0, 0]}, {"t": 10**9, "pos": [0, 0, -5]}]}
    assert mission_metrics(RunLog(Path("."), dict(meta, end_reason="completed"), streams))["success"]
    assert not mission_metrics(RunLog(Path("."), dict(meta, end_reason="error"), streams))["success"]
    assert safety_metrics(RunLog(Path("."), dict(meta, end_reason="hard_timeout"), streams))["timed_out"]


def test_unknown_scenario_fields_are_rejected(root):
    spec = Scenario.load(root / "scenarios/forest_sar/blocks_hiker_01.jsonc").spec
    with pytest.raises(jsonschema.ValidationError):
        Scenario(dict(spec, target_jiter_m=5))  # typo
    with pytest.raises(jsonschema.ValidationError):
        Scenario(dict(spec, limits=dict(spec["limits"], timeout=60)))
    with pytest.raises(jsonschema.ValidationError):
        Scenario(dict(spec, environment={"weather": {"fgo": 0.3}}))


def test_swept_values_are_validated(root):
    spec = Scenario.load(root / "scenarios/forest_sar/blocks_hiker_01.jsonc").spec
    with pytest.raises(jsonschema.ValidationError, match="sweep"):
        Scenario(dict(spec, sweep={"environment.weather.fog": [0.2, 1.5]})).expand()
    with pytest.raises(jsonschema.ValidationError):
        Scenario(dict(spec, sweep={"limits.timout_s": [60]})).expand()


def test_sim_requests_are_serialized_across_threads():
    import threading
    import time

    from vivid2.runner import serialize_requests

    class OneSocketClient:
        """Like Project AirSim's client: overlapping requests break each other."""

        def __init__(self):
            self.in_flight = 0
            self.overlaps = 0

        def request(self, data):
            self.in_flight += 1
            if self.in_flight > 1:
                self.overlaps += 1
            time.sleep(0.01)
            self.in_flight -= 1
            return data

    client = OneSocketClient()
    serialize_requests(client)
    threads = [threading.Thread(target=lambda: [client.request(i) for i in range(5)]) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert client.overlaps == 0
    assert client.request({"x": 1}) == {"x": 1}


def test_depth_units_are_detected_from_an_airborne_frame():
    import numpy as np

    from vivid2.runner import DEPTH_NO_DATA, detect_depth_scale

    metres = np.full((10, 10), 33, np.uint16)  # what the UE 5.8 plugin build sends at 34 m
    millis = np.full((10, 10), 34_480, np.uint16)
    assert detect_depth_scale(metres, 34.5) == 1.0
    assert detect_depth_scale(millis, 34.5) == 0.001
    assert detect_depth_scale(metres, 1.0) is None  # on the ground: undecidable
    assert detect_depth_scale(np.full((10, 10), DEPTH_NO_DATA, np.uint16), 34.5) is None


def test_pad_contacts_are_not_collisions():
    import numpy as np

    from vivid2.metrics.safety import split_ground_contacts

    t = np.array([0.0, 10.0, 20.0, 30.0])
    pos = np.array([[0, 0, -0.4], [0, 0, -35], [40, 0, -35], [1, 1, -0.5]], float)
    hits = [{"t": 0.5e9, "object_name": "Ground"},    # resting on the pad at spawn
            {"t": 20e9, "object_name": "Block"},      # mid-flight: real
            {"t": 30e9, "object_name": "Ground"}]     # touchdown back home
    real, pad = split_ground_contacts(hits, t, pos, np.zeros(3), 0.0)
    assert [c["object_name"] for c in real] == ["Block"] and len(pad) == 2
    real, _ = split_ground_contacts(hits, np.array([]), np.zeros((0, 3)), np.zeros(3), 0.0)
    assert len(real) == 3  # without a trajectory every contact counts


def test_speed_ignores_sample_jitter_but_catches_speeding():
    import numpy as np

    from vivid2.metrics.control import windowed_speeds

    # 6 m/s flight sampled at ~10 Hz, but every 10th sample's position lags 0.5 m
    # behind (as around camera captures): consecutive-sample speeds spike to ~11 m/s.
    t = np.arange(0, 20, 0.1)
    pos = np.stack([6.0 * t, np.zeros_like(t), np.full_like(t, -35.0)], axis=1)
    pos[::10, 0] -= 0.5
    assert windowed_speeds(t, pos).max() < 7.5
    fast = np.stack([10.0 * t, np.zeros_like(t), np.zeros_like(t)], axis=1)
    assert windowed_speeds(t, fast).max() == pytest.approx(10.0, rel=0.05)


def test_cli_agent_choices_come_from_registry(capsys):
    from vivid2.cli import main

    with pytest.raises(SystemExit):
        main(["run", "--campaign", "x.jsonc", "--agent", "nope"])
    err = capsys.readouterr().err
    assert all(a in err for a in AGENTS)


def test_payload_camera_is_lifted_straight_up_for_the_exposure_warm_up():
    from vivid2.runner import lifted_pose
    from vivid2.scenario import load_jsonc
    from vivid2.sim.scene_builder import camera_config

    robot = load_jsonc(Path(__file__).parents[2] / "sim_config" / "robot_vivid_quad.jsonc")
    pose = lifted_pose(camera_config(robot, "Payload"), 55.0)
    assert pose["translation"] == pytest.approx({"x": 0.0, "y": 0.0, "z": 0.15 - 55.0})
    # rpy-deg "0 -90 0": still looking straight down (the recorded nadir frames have w = -y = 0.707)
    r = pose["rotation"]
    assert (r["w"], r["x"], r["y"], r["z"]) == pytest.approx((2 ** -0.5, 0.0, -(2 ** -0.5), 0.0))
