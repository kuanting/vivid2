"""End-to-end agent -> recorder -> metrics -> scoring, with the simulator faked."""
import asyncio
import json

import numpy as np

from conftest import FakeWorld, write_meta
from vivid2.recorder import Recorder
from vivid2.report import write_report
from vivid2.runner import score_run_dir
from vivid2.stack.base import Assessment, Reasoner, UsageStats
from vivid2.stack.reference.agents import SearchAgent
from vivid2.stack.reference.perception import GroundTruthPerception
from vivid2.stack.reference.planners import LawnmowerPlanner
from vivid2.stack.reference.reasoners import DetectionThresholdReasoner


class NeverReasoner(Reasoner):
    name = "never"

    def assess(self, frame, detections, mission):
        return Assessment(False), UsageStats(model="never")


class HallucinatingReasoner(Reasoner):
    """Always reports a target in the image centre."""

    name = "halluc"

    def assess(self, frame, detections, mission):
        return Assessment(True, 0.9, pixel=(frame.width / 2, frame.height / 2)), UsageStats(latency_s=0.5, cost_usd=0.01, model="fake")


def _fly(tmp_path, run, reasoner, perception=None, name="agent"):
    rec = Recorder(tmp_path / name / run.run_id)
    world = FakeWorld(run, rec)
    agent = SearchAgent(name, LawnmowerPlanner(), reasoner, perception, frame_period_s=0.0)
    asyncio.run(agent.run(world))
    write_meta(rec, run, agent.describe(), world.targets, world.t_ns,
               "timeout" if world.elapsed_s() >= world.mission.timeout_s else "completed")
    rec.close()
    return score_run_dir(tmp_path / name / run.run_id), world


def test_oracle_stack_finds_target(tmp_path, blocks_run):
    result, world = _fly(tmp_path, blocks_run, DetectionThresholdReasoner(0.5),
                         GroundTruthPerception(blocks_run.resolved_targets()), "coverage_gt")
    mi = result["metrics"]["mission"]
    assert mi["success"], mi
    assert mi["false_alarms"] == 0
    assert mi["mean_localization_error_m"] < 2.0
    assert result["metrics"]["perception"]["ap50"] == 1.0
    assert result["metrics"]["safety"]["collisions"] == 0
    assert result["metrics"]["safety"]["returned_home"]
    assert result["scores"]["overall"] > 60


def test_blind_stack_fails_but_covers_area(tmp_path, blocks_run):
    result, _ = _fly(tmp_path, blocks_run, NeverReasoner(), None, "blind")
    assert not result["metrics"]["mission"]["success"]
    assert result["metrics"]["planning"]["coverage_pct"] > 90
    assert result["scores"]["mission"] == 0
    assert result["scores"]["perception"] is None  # layer not exposed


def test_hallucinating_reasoner_penalised(tmp_path, blocks_run):
    result, _ = _fly(tmp_path, blocks_run, HallucinatingReasoner(), None, "halluc")
    r = result["metrics"]["reasoning"]
    assert r["hallucination_rate"] > 0.5
    assert result["metrics"]["mission"]["false_alarms"] >= 1
    assert r["cost_usd"] > 0


def test_report_renders(tmp_path, blocks_run):
    _fly(tmp_path, blocks_run, DetectionThresholdReasoner(0.5), GroundTruthPerception(blocks_run.resolved_targets()), "coverage_gt")
    _fly(tmp_path, blocks_run, NeverReasoner(), None, "blind")
    out = write_report(tmp_path)
    html = out.read_text(encoding="utf-8")
    assert "coverage_gt" in html and "blind" in html and "<table>" in html


def test_first_frame_is_taken_at_survey_height(tmp_path, blocks_run):
    """The mission spec carries the camera, so no frame is taken on the ground before takeoff."""
    _, world = _fly(tmp_path, blocks_run, NeverReasoner(), None, "first")
    frames = [json.loads(l) for l in open(tmp_path / "first" / blocks_run.run_id / "frames.jsonl")]
    m = world.mission
    survey_down = m.ground_down_m - m.altitude_m
    assert frames[0]["frame_id"] == 1
    assert abs(frames[0]["cam_pos"][2] - survey_down) < 1.0
