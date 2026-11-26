"""The link to the control team (vivid2.control): the mission is over once everyone has been found."""
import asyncio
import json

import pytest

from conftest import FakeWorld, write_meta
from fleet_fakes import fly_team
from vivid2.control import ControlStation, control_link
from vivid2.recorder import Recorder
from vivid2.registry import AgentFactory
from vivid2.runner import mission_spec, score_run_dir
from vivid2.scenario import Scenario, validate_scenario
from vivid2.stack.base import MissionSpec
from vivid2.stack.reference.agents import SearchAgent
from vivid2.stack.reference.perception import GroundTruthPerception
from vivid2.stack.reference.planners import LawnmowerPlanner
from vivid2.stack.reference.reasoners import DetectionThresholdReasoner

TARGETS = [{"id": "A", "ned": [10.0, 0.0, 0.0]}, {"id": "B", "ned": [60.0, 40.0, 0.0]}]


def _station(link=True, damage=False):
    events = []
    st = ControlStation(TARGETS, 10.0, link, damage, lambda stream, rec: events.append((stream, rec)))
    return st, events


def _rep(t, ned, **kw):
    return {"t": t, "t_report": t + 5, "frame_id": 1, "ned": ned, **kw}


def test_a_link_unless_the_scenario_says_there_is_none():
    assert control_link({}) and control_link({"comms": {"range_m": 400}})
    assert not control_link({"comms": {"control_link": False}})


def test_the_mission_is_over_when_everyone_is_found():
    st, events = _station()
    st.report(_rep(1, [12.0, 3.0, 0.0]), cited=True)
    assert not st.mission_over() and list(st.found) == ["A"]
    st.report(_rep(2, [300.0, 0.0, 0.0]), cited=True)  # a false alarm changes nothing
    st.report(_rep(3, [55.0, 38.0, 0.0]), cited=True)
    assert st.mission_over() and st.end_reason() == "all_found"
    st.report(_rep(4, [10.0, 1.0, 0.0]), cited=True)  # a repeat after the end: still one event
    assert events == [("events", {"t": 8, "kind": "mission_over", "reason": "all_found", "found": ["A", "B"]})]


def test_an_uncited_or_distant_report_finds_nobody():
    st, _ = _station()
    st.report(_rep(1, [10.0, 0.0, 0.0]), cited=False)  # no photo of its own: a false alarm (docs/rules.md)
    st.report(_rep(2, [60.0, 52.0, 0.0]), cited=True)  # 12 m off, beyond the 10 m tolerance
    assert st.found == {} and not st.mission_over()


@pytest.mark.parametrize("link,damage", [(False, False), (True, True)])
def test_no_link_or_a_damage_task_and_finding_everyone_does_not_end_it(link, damage):
    st, events = _station(link, damage)
    for i, t in enumerate(TARGETS):
        st.report(_rep(i, t["ned"]), cited=True)
    assert len(st.found) == 2 and not st.mission_over() and st.end_reason() is None and events == []


def test_the_stack_is_told_whether_it_has_a_link(blocks_run):
    assert mission_spec(blocks_run).control_link is True
    blocks_run.spec["comms"] = {"control_link": False}
    validate_scenario(blocks_run.spec)
    m = mission_spec(blocks_run)
    assert m.control_link is False and m.record()["control_link"] is False
    assert MissionSpec.from_record(m.record()).control_link is False
    old = {k: v for k, v in m.record().items() if k != "control_link"}  # a record from before the link
    assert MissionSpec.from_record(old).control_link is None


def _oracle(tmp_path, root, link, name):
    run = Scenario.load(root / "scenarios/forest_sar/blocks_hiker_02.jsonc").expand(seeds=[0])[0]
    rec = Recorder(tmp_path / name / run.run_id)
    world = FakeWorld(run, rec, link=link)
    agent = SearchAgent(name, LawnmowerPlanner(), DetectionThresholdReasoner(0.5),
                        GroundTruthPerception(run.resolved_targets()), frame_period_s=0.0)
    asyncio.run(agent.run(world))
    end = "all_found" if link and world._control.mission_over() else "completed"
    write_meta(rec, run, agent.describe(), world.targets, world.t_ns, end, control_link=link)
    rec.close()
    events = [json.loads(l) for l in (tmp_path / name / run.run_id / "events.jsonl").read_text().splitlines()]
    return score_run_dir(tmp_path / name / run.run_id), world, [e["kind"] for e in events]


def test_with_a_link_the_mission_ends_at_the_last_find(tmp_path, root):
    linked, world, kinds = _oracle(tmp_path, root, True, "link")
    home, _, home_kinds = _oracle(tmp_path, root, False, "home")
    mi, s = linked["metrics"]["mission"], linked["metrics"]["safety"]
    assert mi["success"] and kinds.count("mission_over") == 1
    assert "rtl" not in kinds and "rtl" in home_kinds  # no flight home with a link
    last_find = max(mi["time_to_find_s"].values())
    assert mi["mission_time_s"] <= last_find + 1.0 < home["metrics"]["mission"]["mission_time_s"]
    assert "returned_home" not in s and s["end_distance_from_home_m"] > 5.0  # not owed with a link
    assert home["metrics"]["safety"]["returned_home"]  # without one the drone must come home
    for r in (linked, home):  # everyone found: the search is complete, whatever share of the area it took
        assert r["metrics"]["planning"]["search_complete"] and r["scores"]["planning"] == 100.0
    assert linked["scores"]["safety"] == home["scores"]["safety"]  # ending away from home costs nothing


def test_a_search_that_missed_someone_is_scored_on_its_coverage(tmp_path, blocks_run):
    rec = Recorder(tmp_path / "blind" / blocks_run.run_id)
    world = FakeWorld(blocks_run, rec, link=True)
    agent = SearchAgent("blind", LawnmowerPlanner(), DetectionThresholdReasoner(0.5), None, frame_period_s=0.0)
    asyncio.run(agent.run(world))
    write_meta(rec, blocks_run, agent.describe(), world.targets, world.t_ns, "completed", control_link=True)
    rec.close()
    r = score_run_dir(tmp_path / "blind" / blocks_run.run_id)
    assert not r["metrics"]["planning"]["search_complete"]
    assert r["scores"]["planning"] == pytest.approx(r["metrics"]["planning"]["coverage_pct"])


def test_a_team_with_a_link_ends_where_it_is(tmp_path, root):
    sc = Scenario.load(root / "scenarios/forest_sar/blocks_hiker_02.jsonc")
    run = sc.expand(seeds=[0], extra_sweep={"fleet.size": [2]})[0]
    r = fly_team(tmp_path, run, AgentFactory("coverage_gt", {"team": "central"})(run), link=True)
    s = r["metrics"]["safety"]
    assert r["metrics"]["mission"]["success"] and r["scores"]["planning"] == 100.0
    assert "returned_home" not in s and all("returned_home" not in p for p in s["per_drone"].values())
    assert s["collisions"] == 0 and max(p["end_distance_from_home_m"] for p in s["per_drone"].values()) > 5.0
