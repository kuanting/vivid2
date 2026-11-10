"""The reference stacks keep out of fire no-fly areas: closer looks stay above the ceiling, dips skip them."""
import asyncio
import copy
import json

import numpy as np
import pytest

from conftest import FakeWorld, write_meta
from test_lowflight import FoggyPerception, forest_with_clearing
from test_verify import DistanceReasoner
from vivid2.recorder import Recorder
from vivid2.runner import score_run_dir
from vivid2.scenario import Run
from vivid2.stack.reference.agents import NO_FLY_MARGIN_M, SearchAgent, no_fly_floor_down
from vivid2.stack.reference.lowflight import DipSettings
from vivid2.stack.reference.planners import LawnmowerPlanner
from vivid2.stack.reference.reasoners import DetectionThresholdReasoner


def _with_zone(run, ceiling, around, half=12.0):
    spec = copy.deepcopy(run.spec)
    n, e = around[0], around[1]
    spec["limits"]["no_fly"] = [{"polygon_ne": [[n - half, e - half], [n + half, e - half], [n + half, e + half],
                                                [n - half, e + half]], "ceiling_agl_m": ceiling}]
    start = spec["dut_start"]["ned"]
    assert not (abs(start[0] - n) < half and abs(start[1] - e) < half), "the start must stay outside the zone"
    return Run(run.run_id, run.scenario_id, run.seed, run.params, spec, run.source)


def _fly(tmp_path, run, agent, surface=None):
    rec = Recorder(tmp_path / agent.name / run.run_id)
    world = FakeWorld(run, rec, width=320, height=240, surface=surface) if surface else FakeWorld(run, rec)
    asyncio.run(agent.run(world))
    write_meta(rec, run, agent.describe(), world.targets, world.t_ns,
               "timeout" if world.elapsed_s() >= world.mission.timeout_s else "completed")
    rec.close()
    return score_run_dir(rec.run_dir), world, rec


def _events(rec, kind):
    rows = [json.loads(l) for l in (rec.run_dir / "events.jsonl").read_text().splitlines()]
    return [r for r in rows if r.get("kind") == kind]


def _lowest_in(rec, zone):
    poly = np.asarray(zone["polygon_ne"], float)
    lo, hi = poly.min(axis=0), poly.max(axis=0)
    traj = np.array([json.loads(l)["pos"] for l in (rec.run_dir / "trajectory.jsonl").read_text().splitlines()])
    inside = np.all((traj[:, :2] > lo) & (traj[:, :2] < hi), axis=1)
    return float((-traj[inside, 2]).min()) if inside.any() else None  # flat test ground at down 0


def test_the_floor_is_the_ceiling_plus_a_margin(blocks_run):
    t = blocks_run.resolved_targets()[0]["ned"]
    run = _with_zone(blocks_run, 15.0, t)
    from conftest import mission_spec

    m = mission_spec(run)
    assert no_fly_floor_down(m, t[0], t[1]) == pytest.approx(-(15.0 + NO_FLY_MARGIN_M))
    assert no_fly_floor_down(m, t[0] + 50, t[1]) is None


def test_a_closer_look_inside_a_no_fly_area_stays_above_its_ceiling(tmp_path, blocks_run):
    t = blocks_run.resolved_targets()[0]["ned"]
    run = _with_zone(blocks_run, 15.0, t)
    result, world, rec = _fly(tmp_path, run, SearchAgent("look", LawnmowerPlanner(), DistanceReasoner(25.0), None,
                                                         frame_period_s=0.0, verify=True))
    start = _events(rec, "verify_start")[0]
    assert start["no_fly_limited"] and start["height_m"] == pytest.approx(15.0 + NO_FLY_MARGIN_M, abs=0.5)
    assert _lowest_in(rec, run.spec["limits"]["no_fly"][0]) >= 15.0 + NO_FLY_MARGIN_M - 0.5
    assert result["metrics"]["mission"]["success"]  # still confirmed, from 20 m
    assert result["metrics"]["safety"]["no_fly_s"] == 0.0


def test_a_dip_into_a_clearing_inside_a_no_fly_area_is_skipped(tmp_path, blocks_run):
    t = blocks_run.resolved_targets()[0]["ned"]
    surface = forest_with_clearing(t[0], t[1])

    def agent():
        return SearchAgent("gt-dip", LawnmowerPlanner(), DetectionThresholdReasoner(0.5),
                           FoggyPerception(blocks_run.resolved_targets()), frame_period_s=0.0, dip=DipSettings())

    high = _with_zone(blocks_run, 25.0, t)  # a 12 m dip would go below 25 m
    result, world, rec = _fly(tmp_path / "high", high, agent(), surface)
    assert not _events(rec, "dip_start") and not world.reports
    assert result["metrics"]["safety"]["no_fly_s"] == 0.0

    low = _with_zone(blocks_run, 5.0, t)  # the 12 m dip stays above 5 m + the margin
    result, world, rec = _fly(tmp_path / "low", low, agent(), surface)
    assert _events(rec, "dip_start") and world.reports
    assert result["metrics"]["safety"]["no_fly_s"] == 0.0
