"""Named reference stacks, so campaigns can be run from the CLI.

    coverage_gt    lawnmower + ground-truth perception + threshold   (oracle upper bound)
    coverage_yolo  lawnmower + ONNX YOLO detector + threshold        (classic CV stack)
    vlm            lawnmower + VLM reasoner on raw frames            (VLM-only stack)
    hybrid         lawnmower + ONNX YOLO detector + VLM verification (detector proposes, VLM decides)
    survey         lawnmower, records every frame, no decisions      (frame bank for offline VLM comparison)

With ``--team central`` or ``--team dist`` (scenarios with ``fleet.size``), any of the search stacks
flies as a team: each drone searches its own strip at its own height and shares its finds
(``vivid2.stack.reference.team``), as one central agent or one agent per drone over the radio.
"""
from __future__ import annotations

import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict

from vivid2.scenario import Run
from vivid2.stack.base import Agent, DistributedTeam
from vivid2.stack.reference.agents import SearchAgent, SurveyAgent
from vivid2.stack.reference.lowflight import DipSettings
from vivid2.stack.reference.planners import LawnmowerPlanner, StripPlanner
from vivid2.stack.reference.reasoners import DetectionThresholdReasoner, VLMReasoner

AGENTS = ("coverage_gt", "coverage_yolo", "vlm", "hybrid", "survey")
TEAM_MODES = {"central": "central", "dist": "distributed"}  # --team value -> team_mode


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


def make_detector(model: str):
    """The tiled ONNX person detector for a KNOWN_DETECTORS name or a file path (downloaded if needed)."""
    from vivid2.stack.reference.detectors import resolve_detector
    from vivid2.stack.reference.perception import OnnxYoloDetector

    path, entry = resolve_detector(model)
    return OnnxYoloDetector(path, fmt=entry.get("format"), model_name=entry["name"], model_sha8=entry["sha256"][:8])


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
        if self.team is not None and self.team not in TEAM_MODES:
            raise ValueError(f"unknown team mode {self.team!r}; choose from {sorted(TEAM_MODES)}")
        if self.team is not None and self.kind == "survey":
            raise ValueError("the survey agent flies alone; teams search with coverage_gt, coverage_yolo, vlm or hybrid")
        self._vlm = make_vlm(self.opts, closed_loop=True) if self.kind in ("vlm", "hybrid") else None
        self._detector = None

    @property
    def name(self) -> str:
        name = vlm_agent_name(self.kind, self._vlm, self.tiles) if self._vlm is not None else self.kind
        if self.kind == "coverage_yolo":
            name += "-" + re.sub(r"[^A-Za-z0-9._-]+", "_", Path(self.yolo_model).stem if Path(self.yolo_model).suffix
                                 else self.yolo_model)
        if self.verify != self.default_verify:  # keep runs with the non-default behaviour apart
            name += "-verify" if self.verify else "-noverify"
        if self.dip:
            name += "-dip"
        if self.team is not None:  # central and distributed runs of a team never share a folder
            name += f"-{self.team}"
        return name

    @property
    def team(self):
        """None (one drone), "central" or "dist"."""
        return self.opts.get("team")

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

    @property
    def yolo_model(self) -> str:
        """A detector name from KNOWN_DETECTORS (default yolox-s) or a path to an ONNX file."""
        from vivid2.stack.reference.detectors import DEFAULT_DETECTOR

        return self.opts.get("yolo_model") or DEFAULT_DETECTOR

    @property
    def yolo_threshold(self) -> float:
        """Detection score that counts as a find: --yolo-threshold, else the named detector's calibrated one."""
        from vivid2.stack.reference.detectors import KNOWN_DETECTORS

        t = self.opts.get("yolo_threshold")
        return float(t) if t is not None else float(KNOWN_DETECTORS.get(self.yolo_model, {}).get("threshold", 0.3))

    def _yolo(self):
        if self._detector is None:
            self._detector = make_detector(self.yolo_model)
        return self._detector

    def __call__(self, run: Run):
        """One Agent, or for ``--team`` a TeamOf (central) or DistributedTeam of one agent per drone."""
        if self.team is None:
            return self._agent(run, LawnmowerPlanner(sidelap=self.sidelap), SearchAgent)
        from vivid2.stack.reference.team import TeamOf, TeamSearchAgent

        n = run.fleet_size
        agents = [self._agent(run, StripPlanner(i, n, sidelap=self.sidelap), TeamSearchAgent, index=i, size=n)
                  for i in range(n)]
        return TeamOf(self.name, agents) if TEAM_MODES[self.team] == "central" else DistributedTeam(self.name, agents)

    @property
    def sidelap(self) -> float:
        return float(self.opts.get("sidelap", 0.2))

    def _agent(self, run: Run, planner, cls, **team) -> Agent:
        period = float(self.opts.get("frame_period_s", 1.0 if self._vlm is None else 2.0))
        dip = DipSettings() if self.dip else None
        if self.kind == "survey":
            return SurveyAgent(self.name, planner, frame_period_s=period, dip=dip)
        kw = {"frame_period_s": period, "verify": self.verify, "dip": dip, **team}
        if self.kind == "coverage_gt":
            from vivid2.stack.reference.perception import GroundTruthPerception

            return cls(self.name, planner, DetectionThresholdReasoner(0.5),
                       GroundTruthPerception(run.resolved_targets()), **kw)
        if self.kind == "coverage_yolo":
            # One threshold T. With closer looks, half of T is enough to go and look; a report needs
            # the majority of the close-ups (or dip looks) at T.
            t = self.yolo_threshold
            look = t / 2 if self.verify else t
            return cls(self.name, planner, DetectionThresholdReasoner(look), self._yolo(),
                       min_confidence=t, candidate_confidence=look, **kw)
        if self.kind == "vlm":
            return cls(self.name, planner, VLMReasoner(self._vlm, tiles=self.tiles), None, **kw)
        return cls(self.name, planner, VLMReasoner(self._vlm, tiles=self.tiles), self._yolo(), **kw)
