"""Manual flight (vivid2 fly): the pilot drones and scenes, without a sim."""
import json

import pytest

from conftest import lfs_file
from vivid2 import pilot
from vivid2.cli import main
from vivid2.scenario import load_jsonc


@pytest.fixture
def forest_terrain(root):
    """ForestSAR's terrain (Git LFS): the pilot scenes put the drone and the hikers on it."""
    return lfs_file(root / "sim_config/maps/ForestSAR/terrain_dtm.npz")


def _settings(robot):
    return robot["controller"]["external-controller-api-settings"]


def test_air_density_ratio():
    assert pilot.air_density_ratio(0.0) == pytest.approx(1.0)
    assert pilot.air_density_ratio(2243.0) == pytest.approx(0.802, abs=0.002)  # ForestSAR's launch clearing
    assert pilot.air_density_ratio(-50.0) == pytest.approx(1.0)


def test_quad_is_the_benchmark_airframe_with_the_pilot_controller(root):
    bench = load_jsonc(root / "sim_config/robot_vivid_quad.jsonc")
    quad = pilot.quad_pilot_robot(bench, 0.8)
    # same body, rotors and sensor mounts as the benchmark quad
    assert quad["links"] == bench["links"] and quad["joints"] == bench["joints"]
    assert quad["actuators"] == bench["actuators"]
    for a, b in zip(quad["sensors"], bench["sensors"]):
        assert a["id"] == b["id"] and a.get("origin") == b.get("origin")
    s = _settings(quad)
    assert quad["controller"]["type"] == "external-controller-api"
    assert s["factory"] == "pilot-fc" and s["fc-profile"] == "vivid_quad" and s["modes"] == ["angle"]
    assert s["actuator-order"] == bench["controller"]["simple-flight-api-settings"]["actuator-order"]
    assert s["air-density-ratio"] == 0.8 and s["gimbal-cameras"] == ["Payload", "PayloadPreview"]
    # Tab cycles same-sized views; nothing else streams
    streaming = {x["id"]: (c["width"], c["height"]) for x in quad["sensors"] if x.get("type") == "camera"
                 for c in x["capture-settings"] if c.get("streaming-enabled")}
    assert streaming == {"Chase": (1280, 720), "PayloadPreview": (1280, 720), "Front": (1280, 720)}
    # the benchmark robot itself is untouched
    assert bench["controller"]["type"] == "simple-flight-api"


def test_fpv5_robot(root):
    robot = pilot.pilot_robot("fpv5", str(root / "sim_config"), 0.9)
    s = _settings(robot)
    assert s["factory"] == "pilot-fc" and "acro" in s["modes"] and s["air-density-ratio"] == 0.9
    text = (root / "sim_config/pilot/robot_fpv5.jsonc").read_text(encoding="utf-8")
    assert "skytwin" not in text.lower()  # public repo: names no private project
    with pytest.raises(ValueError):
        pilot.pilot_robot("strike", str(root / "sim_config"))


@pytest.mark.parametrize("drone", ["fpv5", "quad"])
def test_map_scene_lands_the_drone_at_the_clearing(root, forest_terrain, drone):
    scene, info = pilot.pilot_scene(pilot.map_run("ForestSAR"), str(root / "sim_config"), drone)
    actor = scene["actors"][0]
    assert actor["start-landed"] is True
    # ground at the origin is 257.665 m below NED zero (scene_forest_sar.jsonc)
    z = float(actor["origin"]["xyz"].split()[2])
    assert z == pytest.approx(-257.665 - pilot.DRONES[drone]["spawn_height_m"], abs=0.05)
    assert scene["clock"]["step-ns"] == 2_000_000 and scene["clock"]["real-time-update-rate"] == 500_000
    assert info["targets"] == [] and info["air_density_ratio"] == pytest.approx(0.802, abs=0.003)
    assert _settings(actor["robot-config"])["air-density-ratio"] == pytest.approx(0.802, abs=0.003)
    with pytest.raises(ValueError, match="unknown map"):
        pilot.map_run("Blocks")


def test_scenario_scene_has_its_hikers(root, forest_terrain):
    run = pilot.scenario_run(str(root / "scenarios/forest_sar/forest_clearing_01.jsonc"), seed=1)
    scene, info = pilot.pilot_scene(run, str(root / "sim_config"), "fpv5")
    names = [o["name"] for o in scene["spawn-objects"]["sim-packaged"]]
    assert info["targets"] == ["Hiker1", "Hiker2"] and {"Hiker1", "Hiker2", "Decoy1"} <= set(names)
    # the 5-inch has no Payload camera: nothing is annotated on it
    assert not any("annotation-settings" in x for x in scene["actors"][0]["robot-config"]["sensors"])
    bare, info = pilot.pilot_scene(run, str(root / "sim_config"), "quad", with_targets=False)
    assert info["targets"] == [] and bare["spawn-objects"]["sim-packaged"] == []
    on_fire = pilot.scenario_run(str(root / "scenarios/forest_sar/forest_clearing_01.jsonc"), map_name="ForestFire")
    assert on_fire.spec["scene"]["map"] == "ForestFire"


def test_team_scenarios_fly_their_one_drone_variant(root):
    from vivid2.scenario import Scenario

    path = root / "scenarios/forest_team/forest_team_01.jsonc"
    assert pilot.scenario_run(str(path)).fleet_size == 1  # the sweep's first combination
    team = Scenario.load(path).expand(seeds=[0], extra_sweep={"fleet.size": [4]})[0]
    with pytest.raises(ValueError, match="team"):
        pilot.pilot_scene(team, str(root / "sim_config"), "quad")


def test_fly_print_scene(root, forest_terrain, capsys):
    assert main(["fly", "--map", "ForestSAR_Dense", "--drone", "quad", "--print-scene"]) == 0
    out = capsys.readouterr().out
    scene = json.loads(out[out.index("\n{") + 1:])
    assert scene["id"] == "VIVID2Pilot"
    assert main(["fly", "--drone", "quad", "--print-scene"]) == 2  # needs --map or --scenario


def test_launch_needs_the_pilot_plugin(tmp_path):
    state = {"engine_dir": str(tmp_path), "uproject": "VIVID2Env.uproject", "pilot_plugin": None}
    with pytest.raises(RuntimeError, match="pilot plugin"):
        pilot.launch_command(state, "ForestSAR")
    state["pilot_plugin"] = "C:/x/FpvPilot"
    cmd = pilot.launch_command(state, "ForestSAR", ["-FpvPilotFlightTest"])
    assert cmd[2:4] == ["/Game/VIVID2/Maps/ForestSAR", "-game"] and cmd[-1] == "-FpvPilotFlightTest"
