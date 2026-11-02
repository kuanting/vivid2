"""Closer look before reporting: fly over a candidate, descend, confirm or dismiss, resume the survey."""
import asyncio

import numpy as np
import pytest

from conftest import FakeWorld, write_meta
from vivid2.recorder import Recorder
from vivid2.runner import score_run_dir
from vivid2.stack.base import Assessment, Reasoner, UsageStats
from vivid2.stack.reference.agents import SearchAgent, remaining_path
from vivid2.stack.reference.planners import LawnmowerPlanner


class DistanceReasoner(Reasoner):
    """Unsure about a visible target from survey height, sure when close (like a person getting a better look)."""

    name = "distance"

    def __init__(self, sure_below_m: float):
        self.sure_below_m = sure_below_m

    def assess(self, frame, detections, mission):
        if not frame.gt:
            return Assessment(False, 0.05), UsageStats(model="fake")
        g = frame.gt[0]
        height = mission.ground_down_m - frame.cam_pos[2]
        conf = 0.9 if height < self.sure_below_m else 0.4
        return Assessment(True, conf, pixel=((g.bbox[0] + g.bbox[2]) / 2, (g.bbox[1] + g.bbox[3]) / 2)), UsageStats(model="fake")


class MirageReasoner(Reasoner):
    """Sees 'something' below the drone from survey height; nothing there when it looks closer."""

    name = "mirage"

    def assess(self, frame, detections, mission):
        height = mission.ground_down_m - frame.cam_pos[2]
        if height > 20:
            return Assessment(True, 0.5, pixel=(frame.width / 2, frame.height / 2)), UsageStats(model="fake")
        return Assessment(False, 0.1), UsageStats(model="fake")


def _fly(tmp_path, run, reasoner, **kw):
    rec = Recorder(tmp_path / reasoner.name / run.run_id)
    world = FakeWorld(run, rec)
    agent = SearchAgent(reasoner.name, LawnmowerPlanner(), reasoner, None, frame_period_s=0.0, verify=True, **kw)
    asyncio.run(agent.run(world))
    write_meta(rec, run, agent.describe(), world.targets, world.t_ns,
               "timeout" if world.elapsed_s() >= world.mission.timeout_s else "completed")
    rec.close()
    return score_run_dir(tmp_path / reasoner.name / run.run_id), world, rec


def _events(rec, kind):
    import json
    return [json.loads(l) for l in (rec.run_dir / "events.jsonl").read_text().splitlines()
            if json.loads(l).get("kind") == kind]


def test_unsure_find_is_confirmed_from_closer_and_reported(tmp_path, blocks_run):
    result, world, rec = _fly(tmp_path, blocks_run, DistanceReasoner(sure_below_m=20.0))
    mi = result["metrics"]["mission"]
    assert mi["success"] and mi["false_alarms"] == 0, mi
    start, = _events(rec, "verify_start")
    end, = _events(rec, "verify_end")
    assert end["confirmed"] and start["confidence"] == pytest.approx(0.4)
    m = world.mission
    assert start["height_m"] == pytest.approx(m.min_agl_m + 5.0)  # flat ground, no obstacles: min height + margin
    assert start["height_m"] < m.altitude_m
    assert result["metrics"]["planning"]["verifications"] == 1
    assert result["metrics"]["planning"]["verifications_confirmed"] == 1
    assert world.reports[0].confidence == pytest.approx(0.9)


def test_survey_only_agent_would_not_report_an_unsure_find(tmp_path, blocks_run):
    rec = Recorder(tmp_path / "plain" / blocks_run.run_id)
    world = FakeWorld(blocks_run, rec)
    agent = SearchAgent("plain", LawnmowerPlanner(), DistanceReasoner(sure_below_m=20.0), None, frame_period_s=0.0)
    asyncio.run(agent.run(world))
    assert world.reports == []


def test_mirages_are_dismissed_and_the_survey_resumes(tmp_path, blocks_run):
    result, world, rec = _fly(tmp_path, blocks_run, MirageReasoner(), max_verifications=4)
    assert world.reports == []
    assert len(_events(rec, "verify_start")) == 4                       # bounded detours
    assert not any(e["confirmed"] for e in _events(rec, "verify_end"))
    assert len(_events(rec, "path_start")) == 1 + 4                     # resumed after each look
    assert result["metrics"]["planning"]["coverage_pct"] > 80          # the lanes are not skipped
    assert result["metrics"]["planning"]["verify_time_s"] > 0
    assert result["metrics"]["mission"]["false_alarms"] == 0


def test_remaining_path_continues_after_the_nearest_segment():
    path = [[0, 0, -30], [100, 0, -30], [100, 50, -30], [0, 50, -30]]
    assert remaining_path(path, [40, 2, -30]) == [[100, 0, -30], [100, 50, -30], [0, 50, -30]]
    assert remaining_path(path, [101, 30, -30]) == [[100, 50, -30], [0, 50, -30]]
    assert remaining_path(path, [5, 49, -30]) == [[0, 50, -30]]


def test_close_up_frames_carry_the_candidate_as_focus(tmp_path, blocks_run):
    class FocusRecorder(DistanceReasoner):
        name = "focusrec"

        def __init__(self):
            super().__init__(sure_below_m=20.0)
            self.focus = []

        def assess(self, frame, detections, mission):
            self.focus.append(frame.focus)
            return super().assess(frame, detections, mission)

    r = FocusRecorder()
    _fly(tmp_path, blocks_run, r)
    close = [f for f in r.focus if f is not None]
    assert close, "no close-up frame had a focus"
    w, h = 960, 720  # FakeWorld camera; the drone hovers over the candidate, so it projects near the centre
    assert all(abs(u - w / 2) < 60 and abs(v - h / 2) < 60 for u, v in close)


class FlipFlopReasoner(Reasoner):
    """Unsure from survey height; close up, answers follow a fixed yes/no pattern (a noisy VLM)."""

    name = "flipflop"

    def __init__(self, pattern):
        self.pattern, self.i = list(pattern), 0

    def assess(self, frame, detections, mission):
        if not frame.gt:
            return Assessment(False, 0.05), UsageStats(model="fake")
        g = frame.gt[0]
        px = ((g.bbox[0] + g.bbox[2]) / 2, (g.bbox[1] + g.bbox[3]) / 2)
        if frame.focus is None:  # survey frame
            return Assessment(True, 0.5, pixel=px), UsageStats(model="fake")
        yes = self.pattern[self.i % len(self.pattern)]
        self.i += 1
        return (Assessment(True, 0.7, pixel=px) if yes else Assessment(False, 0.3)), UsageStats(model="fake")


@pytest.mark.parametrize("pattern,confirmed,looks", [
    ([False, True, False], False, 3),   # one noisy "yes" is not a confirmation
    ([True, False, True], True, 3),
    ([True, True, False], True, 2),     # majority reached: no third look
    ([False, False, True], False, 2),   # majority impossible: stop early
])
def test_confirmation_needs_a_majority_of_close_ups(tmp_path, blocks_run, pattern, confirmed, looks):
    _, world, rec = _fly(tmp_path, blocks_run, FlipFlopReasoner(pattern), max_verifications=1)
    end = _events(rec, "verify_end")[0]
    assert end["confirmed"] is confirmed and end["looks"] == looks
    assert bool(world.reports) is confirmed
