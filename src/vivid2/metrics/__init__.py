"""Offline metrics over recorded run logs, one module per evaluated layer."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union

from vivid2.metrics.control import control_metrics
from vivid2.metrics.mission import mission_metrics
from vivid2.metrics.perception import perception_metrics
from vivid2.metrics.planning import planning_metrics
from vivid2.metrics.reasoning import reasoning_metrics
from vivid2.metrics.safety import safety_metrics
from vivid2.recorder import RunLog


def compute_all(log_or_dir: Union[RunLog, str, Path]) -> Dict[str, Any]:
    log = log_or_dir if isinstance(log_or_dir, RunLog) else RunLog.load(Path(log_or_dir))
    return {
        "mission": mission_metrics(log),
        "perception": perception_metrics(log),
        "reasoning": reasoning_metrics(log),
        "planning": planning_metrics(log),
        "control": control_metrics(log),
        "safety": safety_metrics(log),
    }


__all__ = ["compute_all"]
