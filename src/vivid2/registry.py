"""Named reference stacks, so campaigns can be run from the CLI.

    coverage_gt    lawnmower + ground-truth perception + threshold   (oracle upper bound)
    coverage_yolo  lawnmower + ONNX YOLO detector + threshold        (classic CV stack)
    vlm            lawnmower + VLM reasoner on raw frames            (VLM-only stack)
    hybrid         lawnmower + ONNX YOLO detector + VLM verification (detector proposes, VLM decides)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from vivid2.scenario import Run
from vivid2.stack.base import Agent
from vivid2.stack.reference.agents import SearchAgent
from vivid2.stack.reference.planners import LawnmowerPlanner
from vivid2.stack.reference.reasoners import DetectionThresholdReasoner, VLMReasoner

AGENTS = ("coverage_gt", "coverage_yolo", "vlm", "hybrid")


def make_vlm(opts: Dict[str, Any]):
    kind = opts.get("vlm", "anthropic")
    if kind == "anthropic":
        from vivid2.vlm.anthropic_provider import DEFAULT_MODEL, AnthropicVLM

        inner = AnthropicVLM(model=opts.get("vlm_model") or DEFAULT_MODEL, effort=opts.get("vlm_effort", "low"))
    elif kind == "local":
        from vivid2.vlm.openai_compat import OpenAICompatVLM

        if not opts.get("vlm_model"):
            raise ValueError("--vlm local needs --vlm-model (e.g. Qwen/Qwen2.5-VL-7B-Instruct)")
        inner = OpenAICompatVLM(model=opts["vlm_model"], base_url=opts.get("vlm_base_url") or "http://localhost:8001/v1")
    elif kind == "replay":
        from vivid2.vlm.replay import ReplayVLM

        if not opts.get("vlm_cache"):
            raise ValueError("--vlm replay needs --vlm-cache path/to/cache.jsonl (recorded by an earlier --vlm-cache run)")
        return ReplayVLM(opts["vlm_cache"], mode="replay",
                         miss={"target_found": False, "confidence": 0.0, "point": [-1, -1],
                               "description": "cache miss", "hazards": [], "next_action": "continue"})
    else:
        raise ValueError(f"unknown VLM backend {kind!r}")
    if opts.get("vlm_cache"):
        from vivid2.vlm.replay import ReplayVLM

        return ReplayVLM(opts["vlm_cache"], mode="record", inner=inner)
    return inner


@dataclass
class AgentFactory:
    """Builds a fresh agent per run (the GT oracle needs that run's target positions)."""

    kind: str
    opts: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.kind not in AGENTS:
            raise ValueError(f"unknown agent {self.kind!r}; choose from {AGENTS}")
        self._vlm = make_vlm(self.opts) if self.kind in ("vlm", "hybrid") else None
        self._detector = None

    @property
    def name(self) -> str:
        if self._vlm is not None:
            return f"{self.kind}-{self._vlm.model.replace('/', '_').replace(':', '_')}"
        return self.kind

    def _yolo(self):
        if self._detector is None:
            from vivid2.stack.reference.perception import OnnxYoloDetector

            path = self.opts.get("yolo_model")
            if not path:
                raise ValueError(f"agent {self.kind} needs --yolo-model path/to/model.onnx")
            self._detector = OnnxYoloDetector(path)
        return self._detector

    def __call__(self, run: Run) -> Agent:
        planner = LawnmowerPlanner(sidelap=float(self.opts.get("sidelap", 0.2)))
        period = float(self.opts.get("frame_period_s", 1.0 if self._vlm is None else 2.0))
        if self.kind == "coverage_gt":
            from vivid2.stack.reference.perception import GroundTruthPerception

            return SearchAgent(self.name, planner, DetectionThresholdReasoner(0.5),
                               GroundTruthPerception(run.resolved_targets()), frame_period_s=period)
        if self.kind == "coverage_yolo":
            return SearchAgent(self.name, planner, DetectionThresholdReasoner(0.5), self._yolo(), frame_period_s=period)
        if self.kind == "vlm":
            return SearchAgent(self.name, planner, VLMReasoner(self._vlm), None, frame_period_s=period)
        return SearchAgent(self.name, planner, VLMReasoner(self._vlm), self._yolo(), frame_period_s=period)
