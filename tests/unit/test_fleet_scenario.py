"""Team scenarios: the fleet schema, launch pads, and one scene actor per drone."""
import copy

import jsonschema
import numpy as np
import pytest

from vivid2.scenario import Run, Scenario, fleet_size, validate_scenario
from vivid2.sim.scene_builder import build_scene, camera_config

ROOT_SCENARIO = "scenarios/forest_sar/blocks_hiker_01.jsonc"


def _scenario(root, **extra) -> Scenario:
    sc = Scenario.load(root / ROOT_SCENARIO)
    spec = copy.deepcopy(sc.spec)
    spec.update(extra)
    return Scenario(spec, source=sc.source)


def test_schema_takes_a_fleet_a_radio_and_a_separation_limit(root):
    spec = _scenario(root).spec
    ok = dict(spec, fleet={"size": 4, "spacing_m": 12}, comms={"range_m": 300, "latency_s": 0.2, "loss": 0.05},
              limits=dict(spec["limits"], min_separation_m=6))
    validate_scenario(ok)
    for bad in ({"size": 5}, {"size": 0}, {"size": 2, "spacing_m": 4}, {"size": 2, "colour": "red"}):
        with pytest.raises(jsonschema.ValidationError):
            validate_scenario(dict(spec, fleet=bad))
    with pytest.raises(jsonschema.ValidationError):
        validate_scenario(dict(spec, comms={"loss": 2}))


def test_team_size_sweeps_on_a_scenario_without_a_fleet(root):
    runs = _scenario(root).expand(seeds=[0], extra_sweep={"fleet.size": [1, 2, 4]})
    assert [r.fleet_size for r in runs] == [1, 2, 4]
    assert len({r.run_id for r in runs}) == 3
    assert fleet_size(_scenario(root).spec) == 1


def test_one_drone_sits_exactly_at_the_start(root):
    run = _scenario(root).expand(seeds=[0])[0]
    assert run.resolved_fleet() == [{"id": "Drone1", "ned": run.resolved_start(), "yaw_deg": 0.0}]
    sized = _scenario(root, fleet={"size": 1}).expand(seeds=[0])[0]
    assert sized.resolved_fleet()[0]["ned"] == run.resolved_start()


def test_team_pads_form_a_grid_around_the_start(root):
    run = _scenario(root, fleet={"size": 4}).expand(seeds=[0])[0]
    pads = run.resolved_fleet()
    start = np.asarray(run.spec["dut_start"]["ned"], float)
    assert [p["id"] for p in pads] == ["Drone1", "Drone2", "Drone3", "Drone4"]
    xy = np.array([p["ned"][:2] for p in pads]) - start[:2]
    assert sorted(map(tuple, xy)) == [(-5, -5), (-5, 5), (5, -5), (5, 5)]
    assert all(p["ned"][2] == start[2] for p in pads)
    two = _scenario(root, fleet={"size": 2, "spacing_m": 16}).expand(seeds=[0])[0].resolved_fleet()
    assert [tuple(np.subtract(p["ned"][:2], start[:2])) for p in two] == [(0, -8), (0, 8)]


def test_explicit_pads_and_their_spacing(root):
    pads = [{"ned": [0, 0, -1]}, {"ned": [0, 20, -1], "yaw_deg": 90}, {"ned": [20, 0, -1]}]
    run = _scenario(root, fleet={"size": 2, "drones": pads}).expand(seeds=[0])[0]
    assert [(p["ned"], p["yaw_deg"]) for p in run.resolved_fleet()] == [([0, 0, -1], 0.0), ([0, 20, -1], 90.0)]
    close = _scenario(root, fleet={"size": 2, "drones": [{"ned": [0, 0, -1]}, {"ned": [0, 6, -1]}]})
    with pytest.raises(ValueError, match="closer than"):
        close.expand(seeds=[0])[0].resolved_fleet()
    short = _scenario(root, fleet={"size": 3, "drones": pads[:2]})
    with pytest.raises(ValueError, match="only 2"):
        short.expand(seeds=[0])[0].resolved_fleet()


def _scene(run: Run, root):
    return build_scene(run, str(root / "sim_config"), run.resolved_targets(), run.resolved_distractors(),
                       fleet=run.resolved_fleet())


def test_a_one_drone_fleet_builds_the_same_scene(root):
    run = _scenario(root).expand(seeds=[0])[0]
    plain = build_scene(run, str(root / "sim_config"), run.resolved_targets(), run.resolved_distractors())
    assert _scene(run, root) == plain


def test_a_team_scene_has_one_annotated_actor_per_drone(root):
    run = _scenario(root, fleet={"size": 4}).expand(seeds=[0])[0]
    scene = _scene(run, root)
    robots = [a for a in scene["actors"] if a["type"] == "robot"]
    assert [a["name"] for a in robots] == ["Drone1", "Drone2", "Drone3", "Drone4"]
    assert len({a["origin"]["xyz"] for a in robots}) == 4
    for a in robots:
        assert isinstance(a["robot-config"], dict)  # inline: the simulator won't expand file names in a dict scene
        pay = camera_config(scene, "Payload", a["name"])
        assert pay["annotation-settings"]["enabled"] and "Hiker1" in pay["annotation-settings"]["object-ids"]
    assert robots[0]["robot-config"] == robots[3]["robot-config"] and robots[0] is not robots[3]


def test_teams_fly_simple_flight_only(root):
    run = _scenario(root, fleet={"size": 2}).expand(seeds=[0])[0]
    run.spec["scene"]["robot_config"] = "robot_vivid_quad_px4_sitl.jsonc"
    with pytest.raises(ValueError, match="simple_flight"):
        _scene(run, root)


def test_the_team_campaign_sweeps_team_sizes_and_the_single_drone_one_has_none(root):
    from vivid2.scenario import Campaign

    team = Campaign.load(root / "campaigns/forest_team_v1.jsonc").runs()
    assert sorted((r.fleet_size, r.seed) for r in team) == [(n, s) for n in (1, 2, 4) for s in range(3)]
    assert all(r.resolved_fleet()[0]["ned"] != r.resolved_start() for r in team if r.fleet_size > 1)
    assert not [r for r in Campaign.load(root / "campaigns/forest_sar_v1.jsonc").runs() if r.fleet_size > 1]


def test_the_cli_asks_for_a_team_mode(root, capsys):
    from vivid2.cli import main

    code = main(["run", "--campaign", str(root / "campaigns/forest_team_v1.jsonc"), "--agent", "coverage_gt"])
    assert code == 2 and "--team central" in capsys.readouterr().err
