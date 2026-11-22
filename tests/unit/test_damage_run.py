"""The damage task end to end with the simulator faked: oracle mapper, reports, rules, metrics, scores."""
import asyncio
import json

import numpy as np
import pytest

from conftest import FakeWorld, write_meta
from vivid2.damage.run import DamageTask
from vivid2.metrics.damage import f1
from vivid2.recorder import Recorder
from vivid2.registry import AgentFactory
from vivid2.runner import score_run_dir
from vivid2.scenario import Scenario
from vivid2.terrain import TerrainModel


@pytest.fixture
def damage_run(root):
    return Scenario.load(root / "scenarios/landslide/testyard_damage_01.jsonc").expand()[0]


def _fly(tmp_path, run, sim_config, kind="coverage_gt"):
    terrain = TerrainModel.flat(0.0)
    task = DamageTask.for_run(run.spec, sim_config, terrain, run.resolved_clues(terrain),
                              run.resolved_buried(terrain))
    run_dir = tmp_path / kind / run.run_id
    rec = Recorder(run_dir)
    world = FakeWorld(run, rec, damage=task)
    agent = AgentFactory(kind)(run)
    asyncio.run(agent.run(world))
    write_meta(rec, run, agent.describe(), world.targets, world.t_ns,
               extra={"damage": task.meta(), "clues": task.clues, "mission_spec": world.mission.record()})
    rec.close()
    return run_dir, score_run_dir(run_dir), agent


def _rescore(run_dir, reports):
    (run_dir / "damage.jsonl").write_text("".join(json.dumps(r) + "\n" for r in reports), encoding="utf-8")
    return score_run_dir(run_dir)


def _frames(run_dir):
    return [json.loads(line) for line in (run_dir / "frames.jsonl").read_text(encoding="utf-8").splitlines()]


def _frame_over(run_dir, n, e):
    return min(_frames(run_dir), key=lambda f: np.hypot(f["cam_pos"][0] - n, f["cam_pos"][1] - e))


def _report(kind, frame, **kw):
    return {"t": frame["t"], "t_report": frame["t"], "frame_id": frame["frame_id"], "kind": kind,
            "confidence": 1.0, "note": "", **kw}


def test_f1_conventions():
    assert f1(0, 0, 0) == 1.0 and f1(0, 2, 0) == 0.0 and f1(0, 0, 3) == 0.0
    assert f1(1, 2, 1) == pytest.approx(2 / 3)


def test_the_oracle_maps_the_damage_it_photographed(tmp_path, damage_run, damage_sim_config, root):
    run_dir, res, agent = _fly(tmp_path, damage_run, damage_sim_config)
    d = res["metrics"]["damage"]
    assert d["components"] == {"areas": 1.0, "roads": 1.0, "buildings": 1.0, "clues": 1.0, "buried": 1.0}
    assert d["score"] == 100.0 and d["violations"] == [] and d["observed_pct"] > 99
    assert res["scores"]["damage"] == 100.0 and res["metrics"]["mission"]["success"]
    assert agent.describe()["damage_mapper"]["name"] == "gt_damage"
    spec = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))["mission_spec"]
    told = spec["damage"]
    assert {r["id"] for r in told["roads"]} == {"R01-02", "R01-03", "R02-02", "R02-03"}  # midpoints in the AOI
    assert "blocked" not in json.dumps(told) and "grade\"" not in json.dumps(told["buildings"])
    assert res["metrics"]["reasoning"].get("fp_on_clue") == 0


def test_reporting_nothing_scores_zero(tmp_path, damage_run, damage_sim_config, root):
    run_dir, _, _ = _fly(tmp_path, damage_run, damage_sim_config)
    res = _rescore(run_dir, [])
    d = res["metrics"]["damage"]
    assert d["score"] == 0.0 and set(d["components"].values()) == {0.0}
    assert res["scores"]["damage"] == 0.0


def test_report_rules_and_partial_credit(tmp_path, damage_run, damage_sim_config, root):
    run_dir, _, _ = _fly(tmp_path, damage_run, damage_sim_config)
    over_road = _frame_over(run_dir, 35, -40)   # sees the blocked segment R01-03 and the debris
    over_house = _frame_over(run_dir, -51, 20)  # sees the collapsed house B03
    far = _frame_over(run_dir, -50, 50)
    reports = [
        _report("road", over_road, segment="R01-03", blocked=True),
        _report("road", over_road, segment="R01-09", blocked=True),           # unknown segment: a false positive
        _report("road", far, segment="R01-02", blocked=True),                  # not in that photo: a violation
        {**_report("building", over_house, building="B03", grade="collapsed"), "frame_id": 99999},  # uncited
        _report("building", over_house, building="B03", grade="partial"),
        _report("area", over_road, area_class="debris",                        # the layout's outline of the lobe
                polygon_ne=[[16, -58], [44, -54], [50, -30], [46, -10], [28, -6], [18, -18], [12, -36]]),
        _report("area", over_road, area_class="debris",                        # outside the observed ground
                polygon_ne=[[200, 200], [210, 200], [210, 210]]),
        _report("clue", over_road, ned=[35.5, -40.2, 0], clue_kind="glove"),   # found, wrong kind
    ]
    d = _rescore(run_dir, reports)["metrics"]["damage"]
    kinds = sorted(v["kind"] for v in d["violations"])
    assert kinds == ["damage_not_in_cited_frame", "invalid_damage_report", "uncited_damage_report"]
    # every report that breaks a rule is a false positive: the unknown segment and the one not in its photo;
    # R02-03 (blocked by B03's rubble) is missed
    assert d["roads"]["tp"] == 1 and d["roads"]["claims"] == 3 and d["components"]["roads"] == pytest.approx(f1(1, 3, 2))
    # B03 collapsed, reported partial: right that it is damaged, wrong that it collapsed; B02 (partial)
    # missed; the uncited report is a false positive in both
    assert d["components"]["buildings"] == pytest.approx(0.5 * f1(1, 2, 2) + 0.5 * f1(0, 1, 1))
    assert 0.75 < d["areas"]["debris"]["iou"] < 1.0  # the outline without the ground truth's noisy edge
    assert d["clues"]["found"] == 1 and d["clues"]["kind_accuracy"] == 0.0
    assert d["buried"]["covered"] == {"Buried1": True, "Buried2": True}  # B03, reported damaged, holds Buried2
    assert 0 < d["score"] < 100


def test_runs_without_damage_get_no_new_keys(tmp_path, blocks_run):
    rec = Recorder(tmp_path / "coverage_gt" / blocks_run.run_id)
    world = FakeWorld(blocks_run, rec)
    agent = AgentFactory("coverage_gt")(blocks_run)
    asyncio.run(agent.run(world))
    write_meta(rec, blocks_run, agent.describe(), world.targets, world.t_ns)
    rec.close()
    res = score_run_dir(tmp_path / "coverage_gt" / blocks_run.run_id)
    assert "damage" not in res["metrics"] and "damage" not in res["scores"]
    assert "fp_on_clue" not in res["metrics"]["reasoning"]
    assert "damage_mapper" not in agent.describe() and agent.mapper is None
    assert "damage" not in world.mission.record()
    assert not (tmp_path / "coverage_gt" / blocks_run.run_id / "damage.jsonl").exists()
    assert not any("clues_visible" in f for f in _frames(tmp_path / "coverage_gt" / blocks_run.run_id))
    assert set(res["scores"]) == {"mission", "perception", "reasoning", "planning", "control", "safety", "overall"}


def test_the_scorecard_shows_damage_only_when_a_run_has_it(tmp_path, damage_run, blocks_run, damage_sim_config):
    from vivid2.report import render

    _, with_damage, _ = _fly(tmp_path, damage_run, damage_sim_config)
    html_with = render([with_damage], "t")
    rec = Recorder(tmp_path / "plain" / blocks_run.run_id)
    world = FakeWorld(blocks_run, rec)
    agent = AgentFactory("coverage_gt")(blocks_run)
    asyncio.run(agent.run(world))
    write_meta(rec, blocks_run, agent.describe(), world.targets, world.t_ns)
    rec.close()
    plain = score_run_dir(tmp_path / "plain" / blocks_run.run_id)
    assert "<th>damage</th>" in html_with and "<th>damage</th>" not in render([plain], "t")
    assert "<th>damage</th>" in render([plain, with_damage], "t")


def test_authoring_tools_run_on_a_damage_run(tmp_path, damage_run, damage_sim_config, root):
    import sys

    sys.path.insert(0, str(root / "tools"))
    import check_targets
    import damage_png

    run_dir, _, _ = _fly(tmp_path, damage_run, damage_sim_config)
    img = damage_png.render(run_dir, 0.5)
    assert img.ndim == 3 and img.shape[0] > 200 and img.shape[1] > 200
    rows = {r["id"]: r for r in check_targets.check(run_dir)}
    assert set(rows) == {"Hiker1", "Hiker2", "Pack1", "Jacket1"}
    assert rows["Pack1"]["max_px"] == 500 and not rows["Pack1"]["problems"]  # the fake world's clues show 500 px
    assert rows["Hiker1"]["problems"]  # the fake world measures no visible pixels for people


def test_a_team_reports_damage_through_the_drone_whose_photo_it_cites(tmp_path, damage_run, damage_sim_config):
    from test_team_runtime import _Drone, _World

    from vivid2.recorder import RunLog
    from vivid2.runner import SimMissionContext
    from vivid2.stack.base import RoadReport
    from vivid2.team_runtime import SimFleetContext, TeamRuntime

    terrain = TerrainModel.flat(0.0)
    task = DamageTask.for_run(damage_run.spec, damage_sim_config, terrain, damage_run.resolved_clues(terrain), [])
    world = _World(True)
    rec = Recorder(tmp_path / "team")
    ids = ["Drone1", "Drone2"]
    team = TeamRuntime(world, rec, ids)
    ctxs = {i: SimMissionContext(damage_run, world, _Drone(world, i), "Payload", rec, controller=None, fov_deg=80.0,
                                 drone_id=i, team=team, team_info=[{"id": j, "start_ned": [0, 0, 0]} for j in ids],
                                 team_mode="central", damage=task) for i in ids}
    assert ctxs["Drone1"].mission.damage["products"] == ["areas", "roads", "buildings", "clues"]
    team.frame_owner[7] = "Drone2"
    SimFleetContext(ctxs["Drone1"].mission, ctxs, team).report_damage(RoadReport("R01-02", True, 7, 123))
    rec.close()
    log = RunLog.load(tmp_path / "team")
    assert [(d["drone"], d["segment"], d["blocked"], d["frame_id"]) for d in log["damage"]] == [("Drone2", "R01-02", True, 7)]
    assert [e["rule"] for e in log["events"]] == ["uncited_damage_report"]  # frame 7 was never recorded


def test_check_targets_does_not_count_washed_out_photos(tmp_path, damage_run, damage_sim_config, root):
    """A clue (or person) fog or dusk washed out (contrast below the scoring's MIN_CONTRAST) is not visible."""
    import sys

    sys.path.insert(0, str(root / "tools"))
    import check_targets

    run_dir, _, _ = _fly(tmp_path, damage_run, damage_sim_config)
    rows = []
    for line in (run_dir / "frames.jsonl").read_text().splitlines():
        f = json.loads(line)
        if "Pack1" in (f.get("clues_visible") or {}):
            f["clues_visible"]["Pack1"]["contrast"] = 2.0
        rows.append(json.dumps(f))
    (run_dir / "frames.jsonl").write_text("\n".join(rows) + "\n")
    pack = {r["id"]: r for r in check_targets.check(run_dir)}["Pack1"]
    assert pack["max_px"] == 0 and "contrast below" in pack["problems"][0]
