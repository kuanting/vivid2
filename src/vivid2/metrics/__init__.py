"""Offline metrics over recorded run logs, one module per evaluated layer."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union

from vivid2.metrics.control import control_metrics
from vivid2.metrics.damage import damage_metrics
from vivid2.metrics.mission import mission_metrics
from vivid2.metrics.perception import perception_metrics
from vivid2.metrics.planning import planning_metrics
from vivid2.metrics.reasoning import reasoning_metrics
from vivid2.metrics.safety import safety_metrics
from vivid2.metrics.team import is_team, team_metrics
from vivid2.recorder import RunLog


def compute_all(log_or_dir: Union[RunLog, str, Path]) -> Dict[str, Any]:
    log = log_or_dir if isinstance(log_or_dir, RunLog) else RunLog.load(Path(log_or_dir))
    out = {
        "mission": mission_metrics(log),
        "perception": perception_metrics(log),
        "reasoning": reasoning_metrics(log),
        "planning": planning_metrics(log),
        "control": control_metrics(log),
        "safety": safety_metrics(log),
    }
    if "control_link" in log.meta:  # runs since the control link (vivid2.control); older runs as they were
        # Everyone found: the search is complete, however much of the area it took.
        mi = out["mission"]
        out["planning"]["search_complete"] = bool(mi["targets_total"] and mi["targets_found"] == mi["targets_total"])
    if is_team(log.meta):
        out["team"] = team_metrics(log)
    if log.meta.get("damage"):  # runs with a damage task only
        out["damage"] = damage_metrics(log)
    return out


__all__ = ["compute_all", "reasoning_metrics"]
