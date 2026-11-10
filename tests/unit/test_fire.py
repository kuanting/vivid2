"""Static fires and no-fly areas: scenario rules, spawning, segmentation IDs, metrics and scoring."""
import copy
import json
import re

import numpy as np
import pytest

from vivid2.hazards.fire import check_fires, downwind_yaw_deg, in_no_fly, no_fly_ceiling_agl
from vivid2.metrics import compute_all
from vivid2.metrics.reasoning import reasoning_metrics
from vivid2.preflight import run_preflight
from vivid2.recorder import Recorder, RunLog
from vivid2.runner import mission_spec
from vivid2.scenario import Scenario
from vivid2.scoring import layer_scores
from vivid2.sim.scene_builder import build_scene
from vivid2.sim.segmentation import FIRE_ID_BASE, SegmentationError, assign_ids, split_seen
from vivid2.stack.base import MissionSpec

S = 1_000_000_000
ZONE = {"polygon_ne": [[20, 20], [60, 20], [60, 60], [20, 60]], "ceiling_agl_m": 25, "description": "fire front"}


@pytest.fixture
def fire_spec(root):
    """blocks_hiker_01 (hiker near -35, 25; jitter 15) with two fires to the east and wind blowing east."""
    spec = Scenario.load(root / "scenarios/forest_sar/blocks_hiker_01.jsonc").spec
    spec = copy.deepcopy(spec)
    spec["fires"] = [
        {"id": "Fire01", "asset": "VIVID2_Fire_Large", "ned": [40, 40, 0]},
        {"id": "Fire02", "asset": "VIVID2_Fire_Small", "ned": [30, 50, 0], "yaw_deg": 10},
    ]
    spec["environment"]["wind_ned"] = [0, 3, 0]  # towards the east
    spec["limits"]["no_fly"] = [copy.deepcopy(ZONE)]
    spec.pop("sweep")
    return spec


def test_a_scenario_with_fires_validates_and_resolves(fire_spec):
    run = Scenario(fire_spec).expand(seeds=[0])[0]
    fires = run.resolved_fires()
    assert [f["yaw_deg"] for f in fires] == [90.0, 10.0]  # downwind unless authored
    assert fires[0]["ned"] == [40, 40, 0] and "yaw_deg" not in fire_spec["fires"][0]  # the spec is not changed
    assert downwind_yaw_deg([0, 0, 0]) is None and downwind_yaw_deg([-2, 0, 0]) == pytest.approx(180.0)


@pytest.mark.parametrize("change, message", [
    (lambda s: s["fires"].append({"id": "Fire011", "asset": "A", "ned": [45, 45, 0]}), "contain one another"),
    (lambda s: s["fires"].append({"id": "Fire01", "asset": "A", "ned": [45, 45, 0]}), "used twice"),
    (lambda s: s["fires"].append({"id": "Fire03", "asset": "A", "ned": [-35, 45, 0]}), "within 15 m"),
    # 20 m upwind (west) of the hiker, so the hiker sits in its smoke
    (lambda s: s["fires"].append({"id": "Fire03", "asset": "A", "ned": [-35, -20, 0]}), "downwind smoke"),
    (lambda s: s["limits"]["no_fly"][0].update(ceiling_agl_m=35), "not below the survey height"),
    (lambda s: s["limits"]["no_fly"].append({"polygon_ne": [[-20, -5], [0, -5], [0, 5], [-20, 5]],
                                             "ceiling_agl_m": 10}), "start"),
])
def test_fire_rules_are_enforced(fire_spec, change, message):
    change(fire_spec)
    assert any(message in p for p in check_fires(fire_spec))
    with pytest.raises(ValueError, match=re.escape(message)):
        Scenario(fire_spec)


def test_a_wind_sweep_is_checked_per_run(fire_spec):
    fire_spec["fires"].append({"id": "Fire03", "asset": "A", "ned": [-35, -20, 0]})
    fire_spec["environment"]["wind_ned"] = [0, -3, 0]  # smoke drifts west, away from the hiker: fine
    sc = Scenario(fire_spec)
    sc.spec["sweep"] = {"environment.wind_ned": [[0, -3, 0], [0, 3, 0]]}
    with pytest.raises(ValueError, match="downwind smoke"):
        sc.expand(seeds=[0])


def test_scenarios_without_fires_have_no_fire_checks(root):
    for p in sorted((root / "scenarios").rglob("*.jsonc")):
        spec = Scenario.load(p).spec
        if not spec.get("fires") and not spec["limits"].get("no_fly"):
            assert check_fires(spec) == []


def test_fires_are_spawned_after_the_distractors(root, fire_spec):
    run = Scenario(fire_spec).expand(seeds=[0])[0]
    scene = build_scene(run, str(root / "sim_config"), run.resolved_targets(), [], fires=run.resolved_fires())
    spawned = scene["spawn-objects"]["sim-packaged"]
    assert [o["name"] for o in spawned] == ["Hiker1", "Fire01", "Fire02"]
    assert spawned[1]["origin"]["rpy-deg"] == "0 0 90.0" and spawned[1]["physics-enabled"] is False


class SegWorld:
    def __init__(self, actors):
        self.actors = set(actors)

    def set_segmentation_id_by_name(self, name, sid, is_regex, use_owner):
        return is_regex or name in self.actors


def test_fires_get_their_own_segmentation_ids_and_never_count_as_distractors():
    ids = assign_ids(SegWorld(["Hiker1", "Tarp1", "Fire01", "Fire02"]), [{"id": "Hiker1"}], [{"id": "Tarp1"}],
                     [{"id": "Fire01"}, {"id": "Fire02"}])
    assert ids == {"Hiker1": 1, "Tarp1": 101, "Fire01": FIRE_ID_BASE, "Fire02": FIRE_ID_BASE + 1}
    with pytest.raises(SegmentationError):
        assign_ids(SegWorld([]), [], [], [{"id": f"Fire{i:02d}"} for i in range(56)])
    vis = {"Hiker1": {"visible_px": 90, "contrast": 20.0}, "Tarp1": {"visible_px": 40, "contrast": 9.0},
           "Fire01": {"visible_px": 500, "contrast": 31.5}, "Fire02": {"visible_px": 0, "contrast": None}}
    distractors, fires = split_seen(vis, {"Hiker1"}, {"Fire01", "Fire02"})
    assert distractors == {"Tarp1": 40}
    assert fires == {"Fire01": {"visible_px": 500, "contrast": 31.5}}


def test_a_fire_shaped_box_passes_preflight(fire_spec):
    class World:
        def list_objects(self, pattern):
            names = ["TemplateCube_Rounded_1", "TemplateCube_Rounded_102", "Hiker1", "Fire01", "Fire02"]
            return [a for a in names if re.fullmatch(pattern, a)]

        def get_3d_bounding_box(self, name, alignment):
            x, y, z = (8.0, 8.0, 12.0) if name.startswith("Fire") else (0.5, 0.5, 1.7)
            return {"center": {"x": 0, "y": 0, "z": 0}, "size": {"x": x, "y": y, "z": z}}

    info = run_preflight(World(), fire_spec, fire_spec["targets"] + fire_spec["fires"])
    assert info["target_sizes_m"]["Fire01"] == [8.0, 8.0, 12.0]


def test_the_stack_is_told_the_no_fly_areas_only_when_there_are_some(fire_spec, blocks_run):
    run = Scenario(fire_spec).expand(seeds=[0])[0]
    ms = mission_spec(run)
    assert ms.no_fly_zones == [ZONE] and ms.record()["no_fly_zones"] == [ZONE]
    assert MissionSpec.from_record(ms.record()).no_fly_zones == [ZONE]
    plain = mission_spec(blocks_run)
    assert plain.no_fly_zones is None and "no_fly_zones" not in plain.record()


def test_no_fly_geometry():
    ne = np.array([[40.0, 40.0], [40.0, 40.0], [0.0, 0.0]])
    assert list(in_no_fly([ZONE], ne, np.array([10.0, 30.0, 5.0]))) == [True, False, False]
    assert no_fly_ceiling_agl([ZONE], 40, 40) == 25.0 and no_fly_ceiling_agl([ZONE], 0, 0) is None


def _meta(fires=None, no_fly=None):
    limits = {"timeout_s": 600, "max_speed_mps": 8, "min_agl_m": 0.5, "geofence_margin_m": 20, "energy_budget_m": 3000}
    if no_fly:
        limits["no_fly"] = no_fly
    return {
        "run_id": "fire-1", "scenario_id": "fire", "seed": 0, "params": {}, "mission": "wildfire_sar",
        "agent": {"agent": "coverage_gt"}, "targets": [{"id": "Hiker1", "ned": [-35.0, 25.0, 0.0]}],
        "search_area": {"polygon_ne": [[-60, -60], [60, -60], [60, 60], [-60, 60]], "altitude_m": 35, "ground_down_m": 0},
        "limits": limits, "success": {"localization_tolerance_m": 10, "require_all_targets": True},
        "start_ned": [-12.0, 0.0, -1.0], "t0_ns": 0, "end_ns": 100 * S, "end_reason": "completed",
        **({"fires": fires} if fires else {}),
    }


def _fly(tmp_path, meta, path, speed=5.0, dt=0.5):
    rec = Recorder(tmp_path / "run")
    rec.write_json("meta.json", meta)
    t = 0.0
    for a, b in zip(path[:-1], path[1:]):
        a, b = np.asarray(a, float), np.asarray(b, float)
        n = max(1, int(np.linalg.norm(b - a) / (speed * dt)))
        for k in range(n):
            rec.log("trajectory", {"t": int(t * S), "pos": list(a + (b - a) * k / n), "q": [1, 0, 0, 0]})
            t += dt
    rec.log("trajectory", {"t": int(t * S), "pos": list(map(float, path[-1])), "q": [1, 0, 0, 0]})
    rec.close()
    return tmp_path / "run"


# Out at 35 m, then inside the zone (ceiling 25 m) down to 10 m and back up: 15 m down (3 s at 5 m/s),
# 30 m of track (6 s) and 15 m up (3 s) below the ceiling; then home.
LOW_PASS = [[-12, 0, -1], [-12, 0, -35], [25, 40, -35], [25, 40, -10], [55, 40, -10], [55, 40, -35],
            [-12, 0, -35], [-12, 0, -1]]
FIRES = [{"id": "Fire01", "asset": "VIVID2_Fire_Large", "ned": [40.0, 40.0, 0.0], "yaw_deg": 90.0}]


def test_time_below_a_no_fly_ceiling_costs_safety_points(tmp_path):
    m = compute_all(_fly(tmp_path, _meta(FIRES, [ZONE]), LOW_PASS))
    s = m["safety"]
    assert s["no_fly_s"] == pytest.approx(12.0, abs=1.0)
    assert s["min_fire_clearance_m"] == pytest.approx(10.0, abs=0.6)  # right over the fire at 10 m
    assert layer_scores(m)["safety"] == pytest.approx(100.0 - min(40.0, 4.0 * s["no_fly_s"]))


def test_runs_without_fires_get_no_new_metric_keys(tmp_path):
    s = compute_all(_fly(tmp_path, _meta(), LOW_PASS))["safety"]
    assert "no_fly_s" not in s and "min_fire_clearance_m" not in s


def test_team_no_fly_time_is_per_drone_and_summed(tmp_path):
    meta = _meta(FIRES, [ZONE])
    pads = {"Drone1": [-12.0, -5.0, -1.0], "Drone2": [-12.0, 5.0, -1.0]}
    meta.update({"fleet": [{"id": i, "start_ned": p, "yaw_deg": 0} for i, p in pads.items()],
                 "team": {"mode": "central", "size": 2}, "start_ned": pads["Drone1"]})
    rec = Recorder(tmp_path / "run")
    rec.write_json("meta.json", meta)
    rows = []
    for drone, pad in pads.items():
        low = drone == "Drone2"
        path = [pad, [pad[0], pad[1], -35], [20, 40, -10 if low else -35], [60, 40, -10 if low else -35],
                [pad[0], pad[1], -35], pad]
        t = 0.0
        for a, b in zip(path[:-1], path[1:]):
            a, b = np.asarray(a, float), np.asarray(b, float)
            n = max(1, int(np.linalg.norm(b - a) / 2.5))
            for k in range(n):
                rows.append({"t": int(t * S), "pos": list(a + (b - a) * k / n), "q": [1, 0, 0, 0], "drone": drone})
                t += 0.5
        rows.append({"t": int(t * S), "pos": list(map(float, path[-1])), "q": [1, 0, 0, 0], "drone": drone})
    rec.close()
    rows.sort(key=lambda r: (r["t"], r["drone"]))
    (tmp_path / "run" / "trajectory.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    s = compute_all(tmp_path / "run")["safety"]
    assert s["per_drone"]["Drone1"]["no_fly_s"] == 0.0 and s["per_drone"]["Drone2"]["no_fly_s"] > 5.0
    assert s["no_fly_s"] == s["per_drone"]["Drone2"]["no_fly_s"]
    assert s["min_fire_clearance_m"] == pytest.approx(10.0, abs=0.6)


def test_a_report_with_only_a_fire_in_view_is_a_false_alarm_on_fire(tmp_path):
    frames = [{"frame_id": 1, "gt": [], "fires_visible": {"Fire01": {"visible_px": 900, "contrast": 30.0}}},
              {"frame_id": 2, "gt": [], "distractors_visible": {"Tarp1": 50}}]
    decisions = [{"frame_id": f, "layer": "reasoner", "latency_s": 1.0, "output": {"target_found": True}}
                 for f in (1, 2)]
    log = RunLog(tmp_path, _meta(FIRES), {"frames": frames, "decisions": decisions, "detections": []})
    r = reasoning_metrics(log)
    assert r["confusion"]["fp"] == 2 and r["fp_on_fire"] == 1 and r["fp_on_distractor"] == 1
    assert "fp_on_fire" not in reasoning_metrics(RunLog(tmp_path, _meta(), {"frames": frames, "decisions": decisions,
                                                                             "detections": []}))
