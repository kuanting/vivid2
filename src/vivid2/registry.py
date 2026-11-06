"""Named reference stacks, so campaigns can be run from the CLI.

    coverage_gt    lawnmower + ground-truth perception + threshold   (oracle upper bound)
    coverage_yolo  lawnmower + ONNX YOLO detector + threshold        (classic CV stack)
    vlm            lawnmower + VLM reasoner on raw frames            (VLM-only stack)
    hybrid         lawnmower + ONNX YOLO detector + VLM verification (detector proposes, VLM decides)
    survey         lawnmower, records every frame, no decisions      (frame bank for offline VLM comparison)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict

from vivid2.scenario import Run
from vivid2.stack.base import Agent
from vivid2.stack.reference.agents import SearchAgent, SurveyAgent
from vivid2.stack.reference.lowflight import DipSettings
from vivid2.stack.reference.planners import LawnmowerPlanner
from vivid2.stack.reference.reasoners import DetectionThresholdReasoner, VLMReasoner

AGENTS = ("coverage_gt", "coverage_yolo", "vlm", "hybrid", "survey")


def make_vlm(opts: Dict[str, Any], closed_loop: bool = False):
    """Build the VLM provider from CLI options, wrapped in the record/replay cache if one is given.

    ``closed_loop``: cached answers take as long as the original call (thinking takes mission time).
    """
    kind = opts.get("vlm", "anthropic")
    if kind == "anthropic":
        from vivid2.vlm.anthropic_provider import DEFAULT_MODEL, AnthropicVLM

        inner = AnthropicVLM(model=opts.get("vlm_model") or DEFAULT_MODEL, effort=opts.get("vlm_effort", "low"))
    elif kind == "local":
        from vivid2.vlm.openai_compat import OpenAICompatVLM

        if not opts.get("vlm_model"):
            raise ValueError("--vlm local needs --vlm-model (e.g. Qwen/Qwen2.5-VL-7B-Instruct)")
        inner = OpenAICompatVLM(model=opts["vlm_model"], base_url=opts.get("vlm_base_url") or "http://localhost:8001/v1",
                                point_format=opts.get("vlm_point_format", "norm1000"))
    elif kind == "replay":
        raise ValueError("--vlm replay was replaced: rerun with the original --vlm/--vlm-model settings plus "
                         "--vlm-cache CACHE --vlm-cache-mode replay")
    else:
        raise ValueError(f"unknown VLM backend {kind!r}")
    mode = opts.get("vlm_cache_mode", "record")
    if opts.get("vlm_cache"):
        from vivid2.vlm.replay import ReplayVLM

        return ReplayVLM(opts["vlm_cache"], inner, mode=mode, hold_latency=closed_loop)
    if mode == "replay":
        raise ValueError("--vlm-cache-mode replay needs --vlm-cache path/to/cache.jsonl")
    return inner


def vlm_agent_name(kind: str, vlm, tiles: str = "1x1") -> str:
    """``vlm-claude-opus-5-5`` (``-t2x2`` when frames are sent as tiles); replayed answers get a
    ``-replay`` suffix so they never overwrite live runs."""
    name = f"{kind}-{re.sub(r'[^A-Za-z0-9._-]+', '_', vlm.model)}"
    if tiles != "1x1":
        name += f"-t{tiles}"
    return name + "-replay" if getattr(vlm, "mode", None) == "replay" else name


@dataclass
class AgentFactory:
    """Builds a fresh agent per run (the GT oracle needs that run's target positions)."""

    kind: str
    opts: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.kind not in AGENTS:
            raise ValueError(f"unknown agent {self.kind!r}; choose from {AGENTS}")
        self._vlm = make_vlm(self.opts, closed_loop=True) if self.kind in ("vlm", "hybrid") else None
        self._detector = None

    @property
    def name(self) -> str:
        name = vlm_agent_name(self.kind, self._vlm, self.tiles) if self._vlm is not None else self.kind
        if self.verify != self.default_verify:  # keep runs with the non-default behaviour apart
            name += "-verify" if self.verify else "-noverify"
        if self.dip:
            name += "-dip"
        return name

    @property
    def dip(self) -> bool:
        """Low flight: dip into the clearings mapped during the survey (off by default)."""
        return self.opts.get("dip") == "on"

    @property
    def default_verify(self) -> bool:
        """VLM stacks take a closer look before reporting, like a search crew; the GT/YOLO baselines don't."""
        return self.kind in ("vlm", "hybrid")

    @property
    def verify(self) -> bool:
        v = self.opts.get("verify")
        return self.default_verify if v is None else v == "on"

    @property
    def tiles(self) -> str:
        return self.opts.get("vlm_tiles") or "1x1"

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
        dip = DipSettings() if self.dip else None
        if self.kind == "survey":
            return SurveyAgent(self.name, planner, frame_period_s=period, dip=dip)
        kw = {"frame_period_s": period, "verify": self.verify, "dip": dip}
        if self.kind == "coverage_gt":
            from vivid2.stack.reference.perception import GroundTruthPerception

            return SearchAgent(self.name, planner, DetectionThresholdReasoner(0.5),
                               GroundTruthPerception(run.resolved_targets()), **kw)
        if self.kind == "coverage_yolo":
            return SearchAgent(self.name, planner, DetectionThresholdReasoner(0.5), self._yolo(), **kw)
        if self.kind == "vlm":
            return SearchAgent(self.name, planner, VLMReasoner(self._vlm, tiles=self.tiles), None, **kw)
        return SearchAgent(self.name, planner, VLMReasoner(self._vlm, tiles=self.tiles), self._yolo(), **kw)
