"""Team metrics: per-drone paths, energy and pads, drone-drone contacts, separation, scaling."""
import numpy as np
import pytest

from vivid2.metrics import compute_all
from vivid2.recorder import Recorder, RunLog
from vivid2.report import render
from vivid2.runner import score_run_dir
from vivid2.scoring import layer_scores, scaling

PADS = {"Drone1": [0.0, -5.0, -1.0], "Drone2": [0.0, 5.0, -1.0]}
S = 1_000_000_000


def _meta(**extra):
    meta = {
        "run_id": "team-1", "scenario_id": "team", "seed": 0, "params": {}, "mission": "forest_sar",
        "agent": {"agent": "coverage_gt-central", "team_mode": "central"},
        "targets": [{"id": "Hiker1", "ned": [50.0, -20.0, 0.0]}, {"id": "Hiker2", "ned": [50.0, 20.0, 0.0]}],
        "search_area": {"polygon_ne": [[-10, -60], [100, -60], [100, 60], [-10, 60]], "altitude_m": 40,
                        "ground_down_m": 0},
        "limits": {"timeout_s": 600, "max_speed_mps": 8, "min_agl_m": 3, "geofence_margin_m": 20,
                   "energy_budget_m": 300, "min_separation_m": 5},
        "success": {"localization_tolerance_m": 10, "require_all_targets": True},
        "start_ned": PADS["Drone1"], "t0_ns": 0, "end_ns": 200 * S, "end_reason": "completed",
        "fleet": [{"id": i, "start_ned": p, "yaw_deg": 0} for i, p in PADS.items()],
        "team": {"mode": "central", "size": 2},
    }
    meta.update(extra)
    return meta


def _fly(rec, leg, drone, t0=0.0, speed=5.0, dt=0.5):
    """Log a straight-line flight through the points in ``leg`` (NED) at ``speed``, 2 Hz."""
    t = t0
    for a, b in zip(leg[:-1], leg[1:]):
        a, b = np.asarray(a, float), np.asarray(b, float)
        n = max(1, int(np.linalg.norm(b - a) / (speed * dt)))
        for k in range(n):
            rec.log("trajectory", {"t": int(t * S), "pos": list(a + (b - a) * k / n), "q": [1, 0, 0, 0], "drone": drone})
            t += dt
    rec.log("trajectory", {"t": int(t * S), "pos": list(np.asarray(leg[-1], float)), "q": [1, 0, 0, 0], "drone": drone})
    return t


def _run(tmp_path, legs, collisions=(), reports=(), meta=None, events=()):
    rec = Recorder(tmp_path / "run")
    rec.write_json("meta.json", meta or _meta())
    # interleave the drones' samples by time, as the live poller writes them
    for drone, leg in legs.items():
        _fly(rec, leg, drone)
    for c in collisions:
        rec.log("collisions", c)
    for r in reports:
        rec.log("reports", r)
    for e in events:
        rec.log("events", e)
    rec.close()
    lines = (tmp_path / "run" / "trajectory.jsonl").read_text().splitlines()
    import json

    rows = sorted((json.loads(l) for l in lines), key=lambda r: (r["t"], r["drone"]))
    (tmp_path / "run" / "trajectory.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return tmp_path / "run"


def _out_and_back(pad, far):
    up = [pad[0], pad[1], -40.0]
    return [pad, up, [far[0], far[1], -40.0], up, pad]


def test_path_length_and_energy_are_per_drone(tmp_path):
    d1 = _out_and_back(PADS["Drone1"], [80, -5])
    d2 = _out_and_back(PADS["Drone2"], [80, 5])
    m = compute_all(_run(tmp_path, {"Drone1": d1, "Drone2": d2}))
    one = 2 * 39 + 2 * 80  # up and down, out and back
    assert m["planning"]["path_length_m"] == pytest.approx(2 * one, rel=0.02)  # no jumps between drones
    s = m["safety"]
    assert s["energy_used_m"] == pytest.approx(one, rel=0.02) and s["energy_used_total_m"] == pytest.approx(2 * one, rel=0.02)
    assert not s["energy_budget_exceeded"] and s["returned_home"]
    assert set(m["planning"]["per_drone"]) == {"Drone1", "Drone2"} and m["team"]["size"] == 2


def test_one_drone_over_its_battery_or_away_from_its_pad_counts(tmp_path):
    d1 = _out_and_back(PADS["Drone1"], [80, -5])
    d2 = [PADS["Drone2"], [0, 5, -40], [95, 5, -40], [95, 55, -40], [-5, 55, -40], [0, -5, -40], [0, -5, -1]]
    s = compute_all(_run(tmp_path, {"Drone1": d1, "Drone2": d2}))["safety"]
    assert s["energy_budget_exceeded"]  # Drone2 flew past its own 300 m
    assert not s["returned_home"]  # Drone2 landed on Drone1's pad
    assert s["per_drone"]["Drone2"]["end_distance_from_home_m"] == pytest.approx(10.0)


def test_landing_on_a_team_pad_is_a_contact_and_a_drone_strike_counts_once(tmp_path):
    d1 = _out_and_back(PADS["Drone1"], [80, -5])
    d2 = _out_and_back(PADS["Drone2"], [80, 5])
    hit = [{"t": int(20.0 * S) + k * 3_000_000, "object_name": a, "drone": b, "other_drone": a, "impact_point": None}
           for k in range(50) for a, b in (("Drone2", "Drone1"), ("Drone1", "Drone2"))]
    pad = [{"t": 0, "object_name": "Landscape_0", "drone": "Drone2", "impact_point": None}]
    run = _run(tmp_path, {"Drone1": d1, "Drone2": d2}, collisions=hit + pad)
    s = compute_all(run)["safety"]
    assert s["collisions"] == 1 and s["drone_collisions"] == 1 and s["ground_contacts"] == 1
    assert score_run_dir(run)["scores"]["overall"] <= 25.0


def test_drones_flying_too_close_lose_safety_points(tmp_path):
    d1 = [PADS["Drone1"], [0, -5, -40], [40, -1.5, -40], [80, -1.5, -40], [0, -5, -40], PADS["Drone1"]]
    d2 = [PADS["Drone2"], [0, 5, -40], [40, 1.5, -40], [80, 1.5, -40], [0, 5, -40], PADS["Drone2"]]
    m = compute_all(_run(tmp_path, {"Drone1": d1, "Drone2": d2}))
    s = m["safety"]
    assert s["min_separation_m"] == pytest.approx(3.0, abs=0.01) and s["closest_pair"] == ["Drone1", "Drone2"]
    assert s["separation_violation_s"] >= 8.0  # 40 m side by side at 5 m/s
    assert layer_scores(m)["safety"] <= 100.0 - min(30.0, 3.0 * 8.0)


def test_tracking_uses_each_drone_s_own_commands(tmp_path):
    d1 = [PADS["Drone1"], [0, -5, -40], [80, -5, -40]]
    d2 = [PADS["Drone2"], [0, 5, -40], [80, 5, -40]]
    ev = [{"t": int(8 * S), "kind": "path_start", "drone": "Drone1", "from": [0, -5, -40], "waypoints": [[80, -5, -40]]},
          {"t": int(8 * S), "kind": "path_start", "drone": "Drone2", "from": [0, 5, -40], "waypoints": [[80, 5, -40]]},
          {"t": int(10 * S), "kind": "goto", "drone": "Drone2", "target": [0, 0, -40]}]
    c = compute_all(_run(tmp_path, {"Drone1": d1, "Drone2": d2}, events=ev))["control"]
    # Drone2's goto ends only Drone2's window; Drone1 keeps tracking its path (error ~0 throughout)
    assert c["tracking_rmse_m"] == pytest.approx(0.0, abs=0.05)


def test_team_section_credits_drones_and_counts_duplicates(tmp_path):
    d1 = _out_and_back(PADS["Drone1"], [80, -5])
    d2 = _out_and_back(PADS["Drone2"], [80, 5])
    reps = [{"t": 30 * S, "t_report": 31 * S, "frame_id": 3, "ned": [50, -20, 0], "drone": "Drone1"},
            {"t": 32 * S, "t_report": 33 * S, "frame_id": 4, "ned": [51, -20, 0], "drone": "Drone2"},
            {"t": 60 * S, "t_report": 61 * S, "frame_id": 9, "ned": [50, 21, 0], "drone": "Drone2"}]
    m = compute_all(_run(tmp_path, {"Drone1": d1, "Drone2": d2}, reports=reps))
    t = m["team"]
    assert m["mission"]["targets_found"] == 2 and m["mission"]["false_alarms"] == 0
    assert t["duplicate_reports"] == 1 and t["all_found"] and t["time_to_find_all_s"] == pytest.approx(61.0)
    assert t["per_drone"]["Drone2"]["reports"] == 2


def _result(agent, size, seed, t_all, success=True):
    return {"agent": agent, "run_id": f"r{seed}-{size}", "scenario_id": "team", "seed": seed,
            "params": {"fleet.size": size}, "team": {"mode": "central", "size": size},
            "metrics": {"mission": {"success": success, "targets_found": 2, "targets_total": 2,
                                    "mean_time_to_find_s": t_all, "mean_localization_error_m": 1.0, "false_alarms": 0,
                                    "end_reason": "completed"},
                        "safety": {"collisions": 0}, "planning": {"coverage_pct": 90.0, "time_to_90pct_coverage_s": t_all},
                        "team": {"time_to_find_all_s": t_all}},
            "scores": {k: 80.0 for k in ("mission", "perception", "reasoning", "planning", "control", "safety", "overall")}}


def test_scaling_pairs_runs_by_seed_and_counts_unfound_targets_as_the_timeout():
    rs = [_result("a-central", 1, 0, 600.0, success=False), _result("a-central", 1, 1, 400.0),
          _result("a-central", 2, 0, 300.0), _result("a-central", 2, 1, 200.0),
          _result("a-central", 4, 0, 125.0), _result("a-central", 4, 1, 125.0), _result("a-central", 4, 2, 999.0)]
    s = scaling(rs)["a-central"]
    assert s[1]["speedup"] == pytest.approx(1.0) and s[2]["speedup"] == pytest.approx(2.0)
    assert s[2]["efficiency"] == pytest.approx(1.0)
    assert s[4]["paired_runs"] == 2 and s[4]["speedup"] == pytest.approx(4.0) and s[4]["efficiency"] == pytest.approx(1.0)
    html = render(rs, "teams")
    assert "Team scaling" in html and "a-central x4" in html and "Teams, central" in html
