"""The reference team: strips, height layers, shared finds; central and distributed, flown offline."""
import json

import numpy as np
import pytest

from fleet_fakes import fly_team
from vivid2.registry import AgentFactory
from vivid2.scenario import Scenario
from vivid2.stack.base import DistributedTeam
from vivid2.stack.reference.planners import LawnmowerPlanner, StripPlanner
from vivid2.stack.reference.team import TeamOf, TeamSearchAgent
from vivid2.runner import mission_spec


def _run(root, size, comms=None, scenario="blocks_hiker_02"):
    sc = Scenario.load(root / f"scenarios/forest_sar/{scenario}.jsonc")
    run = sc.expand(seeds=[0], extra_sweep={"fleet.size": [size]})[0]
    if comms:
        run.spec["comms"] = comms
    return run


def _events(run_dir, kind):
    rows = [json.loads(l) for l in (run_dir / "events.jsonl").read_text().splitlines()]
    return [r for r in rows if r.get("kind") == kind]


def test_strips_split_the_lanes_of_one_lawnmower(root):
    run = _run(root, 1)
    m = mission_spec(run, camera={"width": 960, "height": 720, "fov_deg": 80.0})
    one = LawnmowerPlanner().plan(m, 80.0, (960, 720))
    lanes = [sorted({round(w[1], 6) for w in StripPlanner(i, 3).plan(m, 80.0, (960, 720))}) for i in range(3)]
    e_min, e_max = -80.0, 80.0
    edges = np.linspace(e_min, e_max, 4)
    for i, ls in enumerate(lanes):
        assert ls and edges[i] < min(ls) and max(ls) < edges[i + 1]  # each strip's lanes stay in the strip
    assert sum(len(l) for l in lanes) >= len({round(w[1], 6) for w in one})  # no fewer lanes than one drone
    assert StripPlanner(0, 1).plan(m, 80.0, (960, 720)) == one


def test_the_factory_builds_teams(root):
    run = _run(root, 3)
    central = AgentFactory("coverage_gt", {"team": "central"})
    team = central(run)
    assert isinstance(team, TeamOf) and central.name == "coverage_gt-central" and len(team.agents) == 3
    assert all(isinstance(a, TeamSearchAgent) for a in team.agents)
    assert [(a.index, a.planner.part, a.planner.parts) for a in team.agents] == [(0, 0, 3), (1, 1, 3), (2, 2, 3)]
    d = team.describe()
    assert d["agent"] == "coverage_gt-central" and d["team_mode"] == "central" and d["team"]["size"] == 3
    dist = AgentFactory("coverage_gt", {"team": "dist", "verify": "on"})
    assert isinstance(dist(run), DistributedTeam) and dist.name == "coverage_gt-verify-dist"
    with pytest.raises(ValueError):
        AgentFactory("survey", {"team": "central"})
    with pytest.raises(ValueError):
        AgentFactory("coverage_gt", {"team": "swarm"})


@pytest.mark.parametrize("mode,size", [("central", 2), ("dist", 4)])
def test_a_team_finds_everyone_safely_from_its_own_strips_and_layers(tmp_path, root, mode, size):
    run = _run(root, size, comms={"range_m": 1000, "latency_s": 0.2})
    agent = AgentFactory("coverage_gt", {"team": mode})(run)
    r = fly_team(tmp_path, run, agent)
    mi, s, t = r["metrics"]["mission"], r["metrics"]["safety"], r["metrics"]["team"]
    assert mi["success"] and mi["false_alarms"] == 0
    assert s["collisions"] == 0 and s["returned_home"] and s["separation_violation_s"] == 0.0
    assert s["min_separation_m"] >= 5.0
    assert t["size"] == size and t["drones_used"] == size and t["all_found"]
    run_dir = tmp_path / agent.name / run.run_id
    starts = {e["drone"]: e for e in _events(run_dir, "path_start")}
    assert len(starts) == size
    heights = sorted(-e["waypoints"][0][2] for e in starts.values())
    assert all(b - a >= 6.0 - 1e-6 for a, b in zip(heights, heights[1:]))  # one layer per drone
    easts = {d: (min(w[1] for w in e["waypoints"]), max(w[1] for w in e["waypoints"])) for d, e in starts.items()}
    spans = sorted(easts.values())
    assert all(a[1] < b[0] for a, b in zip(spans, spans[1:]))  # strips don't overlap
    if mode == "dist":
        assert t["comms"]["messages"] > 0 and t["comms"]["lost"] == {}


def test_a_team_is_faster_than_one_drone(tmp_path, root):
    one = fly_team(tmp_path / "one", _run(root, 1), AgentFactory("coverage_gt", {"team": "central"})(_run(root, 1)))
    two = fly_team(tmp_path / "two", _run(root, 2), AgentFactory("coverage_gt", {"team": "central"})(_run(root, 2)))
    assert one["metrics"]["mission"]["success"] and two["metrics"]["mission"]["success"]
    assert two["metrics"]["team"]["time_to_find_all_s"] < one["metrics"]["team"]["time_to_find_all_s"]


def test_out_of_radio_range_drones_cannot_share_but_raise_no_false_alarms(tmp_path, root):
    run = _run(root, 2, comms={"range_m": 5, "latency_s": 0.2})
    agent = AgentFactory("coverage_gt", {"team": "dist"})(run)
    r = fly_team(tmp_path, run, agent)
    assert r["metrics"]["mission"]["success"] and r["metrics"]["mission"]["false_alarms"] == 0
    assert r["metrics"]["team"]["comms"]["lost"].get("range", 0) > 0


def test_team_photos_fall_on_one_beat_so_they_share_batches(tmp_path, root):
    run = _run(root, 4)
    agent = AgentFactory("coverage_gt", {"team": "central"})(run)
    fly_team(tmp_path, run, agent)
    frames = [json.loads(l) for l in (tmp_path / agent.name / run.run_id / "frames.jsonl").read_text().splitlines()]
    by_t = {}
    for f in frames:
        by_t.setdefault(f["t"], set()).add(f["drone"])
    shared = sum(len(d) for d in by_t.values() if len(d) > 1)
    assert shared >= 0.8 * len(frames)  # most photos are taken together with teammates'
