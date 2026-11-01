"""Resume, rescoring, schema and end-of-run handling must not corrupt results."""
import asyncio
import copy
import json
from pathlib import Path

import jsonschema
import pytest

from conftest import FakeWorld, write_meta
from vivid2.metrics.mission import mission_metrics
from vivid2.metrics.safety import safety_metrics
from vivid2.recorder import Recorder, RunLog
from vivid2.registry import AGENTS, AgentFactory
from vivid2.runner import run_config_hash, score_run_dir
from vivid2.scenario import Scenario
from vivid2.stack.base import Assessment, Reasoner, UsageStats
from vivid2.stack.reference.agents import SearchAgent
from vivid2.stack.reference.planners import LawnmowerPlanner

ONLY_MISSION = {"mission": 1.0, "perception": 0.0, "reasoning": 0.0, "planning": 0.0, "control": 0.0, "safety": 0.0}


class NeverReasoner(Reasoner):
    name = "never"

    def assess(self, frame, detections, mission):
        return Assessment(False), UsageStats(model="never")


def test_rescore_uses_campaign_weights_from_meta(tmp_path, blocks_run):
    run_dir = tmp_path / "blind" / blocks_run.run_id
    rec = Recorder(run_dir)
    world = FakeWorld(blocks_run, rec)
    agent = SearchAgent("blind", LawnmowerPlanner(), NeverReasoner(), frame_period_s=0.0)
    asyncio.run(agent.run(world))
    write_meta(rec, blocks_run, agent.describe(), world.targets, world.t_ns)
    rec.close()
    assert score_run_dir(run_dir)["scores"]["overall"] > 0  # default weights: planning and safety still score

    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    meta["weights"] = ONLY_MISSION
    (run_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    assert score_run_dir(run_dir)["scores"]["overall"] == 0.0


def test_config_hash_tracks_agent_and_scenario_changes(blocks_run):
    base = run_config_hash(blocks_run, AgentFactory("coverage_gt")(blocks_run).describe())
    assert base == run_config_hash(blocks_run, AgentFactory("coverage_gt")(blocks_run).describe())
    wider = AgentFactory("coverage_gt", {"sidelap": 0.4})(blocks_run).describe()
    assert run_config_hash(blocks_run, wider) != base
    edited = copy.deepcopy(blocks_run)
    edited.spec["limits"]["timeout_s"] += 60
    assert run_config_hash(edited, AgentFactory("coverage_gt")(edited).describe()) != base


def test_recorder_ignores_writes_after_close(tmp_path):
    rec = Recorder(tmp_path / "r")
    rec.close()
    rec.log("collisions", {"t": 1})  # a late sim callback must not raise
    assert (tmp_path / "r" / "collisions.jsonl").read_text(encoding="utf-8") == ""


def test_error_run_is_never_a_success_and_hard_timeout_counts():
    meta = {"targets": [{"id": "A", "ned": [0, 0, 0]}], "t0_ns": 0, "limits": {"timeout_s": 100},
            "search_area": {"polygon_ne": [[-10, -10], [10, -10], [10, 10], [-10, 10]]}, "start_ned": [0, 0, 0]}
    streams = {"reports": [{"t": 1, "ned": [1, 0, 0]}],
               "trajectory": [{"t": 0, "pos": [0, 0, 0]}, {"t": 10**9, "pos": [0, 0, -5]}]}
    assert mission_metrics(RunLog(Path("."), dict(meta, end_reason="completed"), streams))["success"]
    assert not mission_metrics(RunLog(Path("."), dict(meta, end_reason="error"), streams))["success"]
    assert safety_metrics(RunLog(Path("."), dict(meta, end_reason="hard_timeout"), streams))["timed_out"]


def test_unknown_scenario_fields_are_rejected(root):
    spec = Scenario.load(root / "scenarios/forest_sar/blocks_hiker_01.jsonc").spec
    with pytest.raises(jsonschema.ValidationError):
        Scenario(dict(spec, target_jiter_m=5))  # typo
    with pytest.raises(jsonschema.ValidationError):
        Scenario(dict(spec, limits=dict(spec["limits"], timeout=60)))
    with pytest.raises(jsonschema.ValidationError):
        Scenario(dict(spec, environment={"weather": {"fgo": 0.3}}))


def test_swept_values_are_validated(root):
    spec = Scenario.load(root / "scenarios/forest_sar/blocks_hiker_01.jsonc").spec
    with pytest.raises(jsonschema.ValidationError, match="sweep"):
        Scenario(dict(spec, sweep={"environment.weather.fog": [0.2, 1.5]})).expand()
    with pytest.raises(jsonschema.ValidationError):
        Scenario(dict(spec, sweep={"limits.timout_s": [60]})).expand()


def test_cli_agent_choices_come_from_registry(capsys):
    from vivid2.cli import main

    with pytest.raises(SystemExit):
        main(["run", "--campaign", "x.jsonc", "--agent", "nope"])
    err = capsys.readouterr().err
    assert all(a in err for a in AGENTS)
