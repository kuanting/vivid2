"""Search rules (docs/rules.md): no ground truth for stacks under test, reports cite their own recorded photos."""
import asyncio
import json
import math
import time
from types import SimpleNamespace

import numpy as np
import pytest

from vivid2.geo import rpy_to_quat
from vivid2.metrics.mission import mission_metrics
from vivid2.metrics.rules import check, eligible
from vivid2.recorder import Recorder, RunLog
from vivid2.runner import SimMissionContext, rules_record, uses_ground_truth
from vivid2.stack.base import Agent, DistributedTeam, TargetReport
from vivid2.stack.reference.agents import SearchAgent
from vivid2.stack.reference.perception import GroundTruthPerception
from vivid2.stack.reference.planners import LawnmowerPlanner
from vivid2.stack.reference.reasoners import DetectionThresholdReasoner
from vivid2.stack.reference.team import TeamOf

NADIR = list(rpy_to_quat(0.0, -math.pi / 2, 0.0))


def _ctx(run, tmp_path, gt_to_stack):
    world = SimpleNamespace(get_sim_time=lambda: 5_000_000_000, client=SimpleNamespace())
    drone = SimpleNamespace(sensors_topic="/Sim/s/robots/Drone1/sensors")
    rec = Recorder(tmp_path / f"ctx{int(gt_to_stack)}")
    ctx = SimMissionContext(run, world, drone, "Payload", rec, controller=None, fov_deg=80.0,
                            ground_truth_to_stack=gt_to_stack)
    return ctx, rec


def _shot():
    img = np.zeros((4, 4, 3), np.uint8)
    return {0: {"data": img.tobytes(), "height": 4, "width": 4, "time_stamp": 6_000_000_000, "pos_x": 0.0, "pos_y": 0.0,
                "pos_z": -50.0, "rot_w": NADIR[0], "rot_x": NADIR[1], "rot_y": NADIR[2], "rot_z": NADIR[3],
                "annotations": [{"object_id": "Hiker1", "bbox2d": {"center": {"x": 2, "y": 2}, "size": {"x": 2, "y": 2}}}]}}


@pytest.mark.parametrize("gt_to_stack", [False, True])
def test_frames_reach_a_stack_without_ground_truth_unless_it_is_the_oracle(blocks_run, tmp_path, airsim_types,
                                                                           gt_to_stack):
    ctx, rec = _ctx(blocks_run, tmp_path, gt_to_stack)
    imgs = _shot()
    frame = asyncio.run(ctx._record_frame(imgs, imgs[0], 5_000_000_000, 5_250_000_000, time.monotonic()))
    rec.close()
    assert [g.object_id for g in frame.gt] == (["Hiker1"] if gt_to_stack else [])
    recorded = RunLog.load(rec.run_dir)["frames"][0]
    assert [g["object_id"] for g in recorded["gt"]] == ["Hiker1"]  # scoring always has it


def test_a_report_on_a_photo_never_taken_is_flagged_live(blocks_run, tmp_path, airsim_types):
    ctx, rec = _ctx(blocks_run, tmp_path, False)
    imgs = _shot()
    f = asyncio.run(ctx._record_frame(imgs, imgs[0], 5_000_000_000, 5_250_000_000, time.monotonic()))
    ctx.report_target(TargetReport(ned=np.zeros(3), confidence=0.9, source="x", frame_id=f.frame_id, t_ns=f.t_ns))
    ctx.report_target(TargetReport(ned=np.zeros(3), confidence=0.9, source="x", frame_id=77, t_ns=f.t_ns))
    rec.close()
    ev = RunLog.load(rec.run_dir)["events"]
    assert [(e["kind"], e["frame_id"]) for e in ev] == [("rule_violation", 77)]


def test_only_the_oracle_is_marked_not_eligible():
    oracle = SearchAgent("coverage_gt", LawnmowerPlanner(), DetectionThresholdReasoner(0.5), GroundTruthPerception([]))
    plain = SearchAgent("survey_like", LawnmowerPlanner(), DetectionThresholdReasoner(0.5), None)
    assert uses_ground_truth(oracle) and not uses_ground_truth(plain)
    assert uses_ground_truth(TeamOf("t", [oracle, oracle])) and uses_ground_truth(DistributedTeam("d", [oracle]))
    assert not uses_ground_truth(TeamOf("t", [plain])) and not Agent.uses_ground_truth
    assert rules_record(oracle) == {"version": 1, "ground_truth_to_stack": True, "eligible": False,
                                    "citation": "own_recorded_search_frame"}
    assert rules_record(plain)["eligible"] and not rules_record(plain)["ground_truth_to_stack"]
    assert not eligible({"agent": {"perception": "gt_perception"}}) and eligible({"agent": {"perception": "yolo"}})


def _log(tmp_path, reports, decisions=(), rules=True, team=False):
    run_dir = tmp_path / ("team" if team else "one")
    run_dir.mkdir()
    meta = {"targets": [{"id": "Hiker1", "ned": [10, 0, 0]}], "t0_ns": 0, "end_ns": 100, "end_reason": "completed",
            "limits": {"timeout_s": 100}, "success": {"localization_tolerance_m": 10.0},
            "cameras": {"Payload": {"role": "search"}, "Front": {"role": "nav"}}}
    if rules:
        meta["rules"] = {"version": 1, "eligible": True}
    if team:
        meta["team"] = {"mode": "central", "size": 2}
    frames = [{"t": 1, "frame_id": 1, "camera": "Payload", **({"drone": "Drone1"} if team else {})},
              {"t": 2, "frame_id": 2, "camera": "Front", **({"drone": "Drone1"} if team else {})}]
    (run_dir / "meta.json").write_text(json.dumps(meta))
    for name, rows in (("frames", frames), ("reports", reports), ("decisions", decisions)):
        (run_dir / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return RunLog.load(run_dir)


def _report(fid, ned=(10, 0, 0), **kw):
    return {"t": fid, "t_report": fid, "frame_id": fid, "ned": list(ned), "confidence": 0.9, **kw}


def test_a_report_must_cite_a_recorded_search_photo(tmp_path):
    log = _log(tmp_path, [_report(99), _report(2), _report(1)], decisions=[{"t": 5, "frame_id": 42}])
    cited, violations = check(log)
    assert [r["frame_id"] for r in cited] == [1]
    assert [v["kind"] for v in violations] == ["uncited_report", "report_cites_nav_frame",
                                               "decision_on_unrecorded_frame"]
    m = mission_metrics(log)
    # found once, by the cited report; the two uncited ones are false alarms though they were right
    assert m["targets_found"] == 1 and m["false_alarms"] == 2 and m["uncited_reports"] == 2


def test_a_team_report_must_cite_its_own_drone_s_photo(tmp_path):
    log = _log(tmp_path, [_report(1, drone="Drone2")], team=True)
    assert [v["kind"] for v in check(log)[1]] == ["report_cites_other_drone"]


def test_runs_from_before_the_rules_are_scored_as_they_were(tmp_path):
    log = _log(tmp_path, [_report(99)], rules=False)
    assert check(log) == (log["reports"], [])
    m = mission_metrics(log)
    assert m["targets_found"] == 1 and m["false_alarms"] == 0 and "uncited_reports" not in m


def test_the_context_shows_a_stack_no_simulator_handles(blocks_run, tmp_path):
    ctx, rec = _ctx(blocks_run, tmp_path, False)
    rec.close()
    public = {k for k in vars(ctx) if not k.startswith("_")}
    assert public <= {"mission", "controller", "drone_id", "camera_id", "fov_deg", "reports", "radio"}
    from vivid2.stack.reference.controllers import AirSimController

    assert not hasattr(AirSimController(SimpleNamespace(), lambda: 0, lambda *a: None), "drone")


def test_offline_frames_carry_no_ground_truth(tmp_path):
    from vivid2.openloop import load_frame
    from vivid2.vlm.base import encode_jpeg_bytes, sha256_hex

    jpeg = encode_jpeg_bytes(np.zeros((8, 8, 3), np.uint8))
    (tmp_path / "frames").mkdir()
    (tmp_path / "frames" / "000001.jpg").write_bytes(jpeg)
    rec = {"t": 1, "frame_id": 1, "rgb": "frames/000001.jpg", "rgb_sha256": sha256_hex(jpeg), "cam_pos": [0, 0, -50],
           "cam_q": NADIR, "fov_deg": 80.0, "gt": [{"object_id": "Hiker1", "bbox": [0, 0, 4, 4]}]}
    f = load_frame(tmp_path, rec)
    assert f.gt == [] and f.camera == "Payload"


def test_the_scorecard_lists_the_oracle_last_and_not_eligible():
    from vivid2.report import _stack_rows

    def result(agent, overall, perception, eligible_):
        return {"run_id": f"r-{agent}", "scenario_id": "s", "agent": agent, "agent_config": {"agent": agent,
                "perception": perception}, "seed": 0, "params": {}, "eligible": eligible_,
                "metrics": {"mission": {"success": True}, "safety": {"collisions": 0}},
                "scores": {k: overall for k in ("mission", "perception", "reasoning", "planning", "control", "safety",
                                                "overall")}, "rule_violations": [{"kind": "uncited_report"}]}

    rows = _stack_rows([result("coverage_gt", 95.0, "gt_perception", False), result("mine", 60.0, "yolo", True)])
    assert rows.index(">mine<") < rows.index(">coverage_gt<") and "oracle, not eligible" in rows
