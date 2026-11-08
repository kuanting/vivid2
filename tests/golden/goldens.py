"""Golden values recorded before the multi-drone work (commit e57b87b, SIM_VERSION 6).

Team support must leave single-drone runs untouched: the same run ids and config hashes (so
finished runs are not redone on resume), the same scene for the simulator, and the same metrics
and scores. ``tests/unit/test_golden.py`` recomputes these values and compares them with the
JSON files next to this module. Regenerate them only for a deliberate change:

    python tests/golden/goldens.py
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import numpy as np

TESTS = Path(__file__).resolve().parents[1]
ROOT = TESTS.parent
HERE = Path(__file__).resolve().parent
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

from vivid2 import registry  # noqa: E402
from vivid2.runner import SIM_VERSION, run_config_hash, score_run_dir  # noqa: E402
from vivid2.scenario import Campaign, Run, Scenario  # noqa: E402
from vivid2.sim.scene_builder import build_scene  # noqa: E402

# Every agent configuration the reference stacks are run with.
AGENT_CONFIGS: List[Tuple[str, Dict[str, Any]]] = [
    ("coverage_gt", {}), ("coverage_gt", {"verify": "on"}), ("coverage_gt", {"dip": "on"}),
    ("coverage_gt", {"verify": "on", "dip": "on"}),
    ("coverage_yolo", {}), ("coverage_yolo", {"verify": "on", "dip": "on"}),
    ("vlm", {}), ("vlm", {"dip": "on"}), ("vlm", {"verify": "off"}), ("vlm", {"vlm_tiles": "2x2"}),
    ("hybrid", {}), ("survey", {}), ("survey", {"dip": "on"}),
]


class _StubVLM:
    """Stands in for the Claude provider (no anthropic package or API key in CI)."""

    model = "claude-opus-5-5"
    mode = "record"

    def fingerprint(self) -> Dict[str, Any]:
        return {"provider": "anthropic", "model": self.model, "effort": "low", "max_tokens": 1024}


class _StubSession:
    """A YOLOX-shaped ONNX session, so the real OnnxYoloDetector settings feed describe()."""

    def get_inputs(self):
        return [SimpleNamespace(name="images")]

    def get_outputs(self):
        return [SimpleNamespace(shape=[1, 8400, 85])]


def _stub_detector(model: str):
    from vivid2.stack.reference.detectors import KNOWN_DETECTORS
    from vivid2.stack.reference.perception import OnnxYoloDetector

    entry = KNOWN_DETECTORS[model]
    return OnnxYoloDetector(entry["file"], fmt=entry["format"], model_name=model, model_sha8=entry["sha256"][:8],
                            session=_StubSession())


class _Stubs:
    """Patch the registry's VLM and detector builders for the duration of a block."""

    def __enter__(self):
        self._saved = registry.make_vlm, registry.make_detector
        registry.make_vlm = lambda opts, closed_loop=False: _StubVLM()
        registry.make_detector = _stub_detector
        return self

    def __exit__(self, *exc):
        registry.make_vlm, registry.make_detector = self._saved


def catalog_runs() -> List[Run]:
    """Every run of every campaign and scenario file, once each."""
    runs: Dict[Tuple[str, str], Run] = {}
    for path in sorted((ROOT / "campaigns").glob("*.jsonc")):
        for run in Campaign.load(path).runs():
            runs.setdefault((run.run_id, _spec_sha(run.spec)), run)
    for path in sorted((ROOT / "scenarios").rglob("*.jsonc")):
        for run in Scenario.load(path).expand():
            runs.setdefault((run.run_id, _spec_sha(run.spec)), run)
    return [runs[k] for k in sorted(runs)]


def _spec_sha(spec: Dict[str, Any]) -> str:
    return hashlib.sha1(json.dumps(spec, sort_keys=True, default=str).encode()).hexdigest()[:12]


def config_hashes() -> Dict[str, Any]:
    """{agent name: {run id/spec: config hash}} for every reference agent and catalog run."""
    out: Dict[str, Any] = {"sim_version": SIM_VERSION, "agents": {}}
    runs = catalog_runs()
    with _Stubs():
        for kind, opts in AGENT_CONFIGS:
            factory = registry.AgentFactory(kind, dict(opts))
            table = out["agents"].setdefault(factory.name, {})
            for run in runs:
                table[f"{run.run_id}/{_spec_sha(run.spec)}"] = run_config_hash(run, factory(run).describe())
    return out


def scenes() -> Dict[str, Any]:
    """The scene dict sent to the simulator for the first run of each scenario (targets unsnapped,
    so CI needs no terrain file)."""
    out = {}
    for path in sorted((ROOT / "scenarios").rglob("*.jsonc")):
        run = Scenario.load(path).expand(seeds=[0])[0]
        out[run.scenario_id] = build_scene(run, str(ROOT / "sim_config"), run.resolved_targets(),
                                           run.resolved_distractors())
    return json.loads(json.dumps(out, sort_keys=True))


def fake_runs() -> Dict[str, Any]:
    """Metrics and scores of reference agents flown in FakeWorld (the offline simulator)."""
    from conftest import FakeWorld, write_meta
    from vivid2.recorder import Recorder
    from vivid2.stack.base import Assessment, Reasoner, UsageStats
    from vivid2.stack.reference.agents import SearchAgent
    from vivid2.stack.reference.lowflight import DipSettings
    from vivid2.stack.reference.perception import GroundTruthPerception
    from vivid2.stack.reference.planners import LawnmowerPlanner
    from vivid2.stack.reference.reasoners import DetectionThresholdReasoner

    class Halluc(Reasoner):
        name = "halluc"

        def assess(self, frame, detections, mission):
            return (Assessment(True, 0.9, pixel=(frame.width / 2, frame.height / 2)),
                    UsageStats(latency_s=0.5, cost_usd=0.01, model="fake"))

    def run_of(scenario: str) -> Run:
        return Scenario.load(ROOT / f"scenarios/forest_sar/{scenario}.jsonc").expand(seeds=[0])[0]

    class Foggy(GroundTruthPerception):
        """Sees people only from below 20 m (fog), so finding the hiker takes a dip."""

        def detect(self, frame):
            return [] if -frame.cam_pos[2] > 20.0 else super().detect(frame)

    def clearing_at(target):  # 20 m trees everywhere except a 14 m clearing around the hiker
        n0, e0 = target["ned"][0], target["ned"][1]
        return lambda n, e: np.where(np.hypot(np.asarray(n) - n0, np.asarray(e) - e0) <= 14.0, 0.0, 20.0)

    cases = {
        "oracle": (run_of("blocks_hiker_01"), lambda r: dict(reasoner=DetectionThresholdReasoner(0.5),
                                                             perception=GroundTruthPerception(r.resolved_targets())), {}),
        "oracle_verify_2targets": (run_of("blocks_hiker_02"),
                                   lambda r: dict(reasoner=DetectionThresholdReasoner(0.5),
                                                  perception=GroundTruthPerception(r.resolved_targets()), verify=True),
                                   {}),
        "halluc": (run_of("blocks_hiker_01"), lambda r: dict(reasoner=Halluc(), perception=None), {}),
        "foggy_dip": (run_of("blocks_hiker_01"), lambda r: dict(reasoner=DetectionThresholdReasoner(0.5),
                                                                perception=Foggy(r.resolved_targets()),
                                                                dip=DipSettings()),
                      {"surface": clearing_at(run_of("blocks_hiker_01").resolved_targets()[0]),
                       "width": 320, "height": 240}),
    }
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        for name, (run, layers, world_kw) in cases.items():
            rec = Recorder(Path(tmp) / name / run.run_id)
            world = FakeWorld(run, rec, **world_kw)
            kw = layers(run)
            agent = SearchAgent(name, LawnmowerPlanner(), kw.pop("reasoner"), kw.pop("perception"),
                                frame_period_s=0.0, **kw)
            asyncio.run(agent.run(world))
            write_meta(rec, run, agent.describe(), world.targets, world.t_ns,
                       "timeout" if world.elapsed_s() >= world.mission.timeout_s else "completed")
            rec.close()
            result = score_run_dir(Path(tmp) / name / run.run_id)
            streams = {s: sum(1 for _ in open(Path(tmp) / name / run.run_id / f"{s}.jsonl", encoding="utf-8"))
                       for s in ("trajectory", "frames", "decisions", "reports", "events")}
            out[name] = {"metrics": result["metrics"], "scores": result["scores"], "records": streams}
    return json.loads(json.dumps(out, sort_keys=True, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))


GOLDENS = {"config_hashes.json": config_hashes, "scenes.json": scenes, "fake_runs.json": fake_runs}


def write_all() -> None:
    for name, fn in GOLDENS.items():
        (HERE / name).write_text(json.dumps(fn(), indent=1, sort_keys=True) + "\n", encoding="utf-8")
        print("wrote", HERE / name)


if __name__ == "__main__":
    write_all()
