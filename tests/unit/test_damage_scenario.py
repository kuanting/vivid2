"""Scenarios with clues and a damage task: schema, placement rules, resolution."""
import copy

import jsonschema
import numpy as np
import pytest

from vivid2.damage.check import check_damage_gt, check_damage_spec
from vivid2.damage.run import DamageTask
from vivid2.scenario import Run, Scenario, validate_scenario
from vivid2.terrain import TerrainModel


@pytest.fixture
def damage_spec(root):
    return Scenario.load(root / "scenarios/landslide/testyard_damage_01.jsonc").spec


def _run(spec, seed=0):
    return Run("r", spec["id"], seed, {}, spec)


def test_the_test_map_damage_scenario_is_valid(damage_spec):
    validate_scenario(damage_spec)
    assert check_damage_spec(damage_spec) == []


def test_damage_spec_rules(damage_spec):
    s = copy.deepcopy(damage_spec)
    s["mission"] = "forest_sar"
    assert any("disaster mission" in p for p in check_damage_spec(s))
    s = copy.deepcopy(damage_spec)
    s["damage"]["products"] = ["areas", "roads"]
    assert any("has clues but" in p for p in check_damage_spec(s))
    s = copy.deepcopy(damage_spec)
    s.pop("clues")
    assert any("has none" in p for p in check_damage_spec(s))
    s = copy.deepcopy(damage_spec)
    s["clues"][0]["id"] = "Hiker1"
    assert any("used twice" in p for p in check_damage_spec(s))
    s = copy.deepcopy(damage_spec)
    s["clues"][0]["id"] = "Hiker1Pack"
    assert any("contains" in p for p in check_damage_spec(s))
    s = copy.deepcopy(damage_spec)
    s["damage"]["aoi_ne"] = [[0, 0], [200, 0], [200, 50]]
    assert any("leaves the search area" in p for p in check_damage_spec(s))
    s = copy.deepcopy(damage_spec)
    s["damage"]["weights"] = {"speed": 1}
    with pytest.raises(jsonschema.ValidationError):
        validate_scenario(s)
    with pytest.raises(ValueError):
        Scenario(dict(damage_spec, mission="forest_sar"))


def test_scenarios_without_clues_or_damage_are_not_checked(root):
    for p in (root / "scenarios").rglob("*.jsonc"):
        spec = Scenario.load(p).spec
        if not spec.get("damage") and not spec.get("clues"):
            assert check_damage_spec(spec) == []


def test_a_target_without_jitter_does_not_move_the_others(damage_spec):
    s = copy.deepcopy(damage_spec)
    s["targets"] = [dict(s["targets"][0], id=f"P{i}", ned=[10.0 * i, 0, 0]) for i in range(3)]
    s["target_jitter_m"] = 5
    base = _run(s).resolved_targets()
    s2 = copy.deepcopy(s)
    s2["targets"].insert(1, dict(s["targets"][0], id="Fixed", ned=[0, 30, 0], jitter_m=0))
    with_fixed = _run(s2).resolved_targets()
    assert with_fixed[1]["ned"] == [0, 30, 0]
    assert [t["ned"] for t in with_fixed[:1] + with_fixed[2:]] == [t["ned"] for t in base]
    s3 = copy.deepcopy(s)
    s3["targets"][2]["jitter_m"] = 20
    far = _run(s3).resolved_targets()
    assert far[:2] == base[:2] and np.hypot(far[2]["ned"][0] - 20, far[2]["ned"][1]) <= 20


def test_clues_buried_and_surface_targets_resolve_onto_the_ground(damage_spec):
    terrain = TerrainModel(np.full((201, 201), 5.0), -100, -100, 1.0)  # ground 5 m below the origin
    run = _run(damage_spec)
    assert [c["ned"][2] for c in run.resolved_clues(terrain)] == pytest.approx([5.0, 5.0])  # authored on the ground
    assert run.resolved_buried(terrain)[0]["ned"][2] == pytest.approx(5.0)
    s = copy.deepcopy(damage_spec)
    s["targets"][0].update(snap="surface", ned=[-35, 25, s["search_area"]["ground_down_m"]])  # feet on the ground
    roof = TerrainModel(np.full((201, 201), -3.0), -100, -100, 1.0)  # a roof 8 m above that ground
    assert _run(s).resolved_targets(terrain, roof)[0]["ned"][2] == pytest.approx(-3.0)
    assert _run(s).resolved_targets(terrain)[0]["ned"][2] == pytest.approx(5.0)  # no surface model: the ground


def test_placements_are_checked_against_the_ground_truth(damage_spec, damage_sim_config):
    terrain = TerrainModel.flat(0.0)
    run = _run(damage_spec)
    task = DamageTask.for_run(damage_spec, damage_sim_config, terrain, run.resolved_clues(terrain),
                              run.resolved_buried(terrain))
    targets = run.resolved_targets(terrain)
    assert check_damage_gt(damage_spec, task, targets, 0.0, None, terrain) == []
    sunk_off = [dict(targets[0], ned=[targets[0]["ned"][0], targets[0]["ned"][1], 0.6])]
    assert any("partly buried" in p for p in check_damage_gt(damage_spec, task, sunk_off, 0.0, None, terrain))
    sunk_on = [dict(targets[0], ned=[40.0, -40.0, 0.6])]  # in the painted debris
    assert check_damage_gt(damage_spec, task, sunk_on, 0.0, None, terrain) == []
    lost = [{"id": "B9", "ned": [-50.0, 40.0, 0.0]}]
    assert any("not within" in p for p in check_damage_gt(damage_spec, task, targets, 0.0, None, terrain, lost))
    in_house = [{"id": "B9", "ned": [-51.0, 20.0, 0.0], "building": "B03"}, {"id": "B8", "ned": [13, 13, 0], "building": "B01"}]
    probs = check_damage_gt(damage_spec, task, targets, 0.0, None, terrain, in_house)
    assert len(probs) == 1 and "B01" in probs[0]  # B01 is intact
    roof = [dict(targets[0], snap="surface")]
    assert any("no surface model" in p for p in check_damage_gt(damage_spec, task, roof, 0.0, None, terrain))
