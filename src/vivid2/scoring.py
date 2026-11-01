"""Normalise raw metrics into 0-100 layer scores and aggregate across runs.

Scores are deliberately simple and documented so vendors can reproduce them.
A layer the stack does not expose (e.g. no separate perception output) scores
``None`` and is excluded from the weighted overall score, not zeroed.
Any collision caps the overall score (safety gate).
"""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

DEFAULT_WEIGHTS = {
    "mission": 0.35,
    "perception": 0.15,
    "reasoning": 0.15,
    "planning": 0.10,
    "control": 0.05,
    "safety": 0.20,
}
COLLISION_CAP = 25.0


def _clamp(x: float) -> float:
    return float(max(0.0, min(100.0, x)))


def layer_scores(m: Dict[str, Any]) -> Dict[str, Optional[float]]:
    mi = m["mission"]
    timeout = mi["timeout_s"]
    per_target = [1.0 - 0.5 * min(t / timeout, 1.0) for t in mi["time_to_find_s"].values()]
    mission = 100.0 * sum(per_target) / mi["targets_total"] - 10.0 * mi["false_alarms"]

    perception = None
    if m.get("perception") and not math.isnan(m["perception"]["ap50"]):
        perception = 100.0 * m["perception"]["ap50"]

    reasoning = None
    r = m.get("reasoning")
    if r:
        f1 = r["f1"] if r["f1"] is not None else (1.0 if r["confusion"]["fp"] == 0 else 0.0)
        halluc = r["hallucination_rate"] or 0.0
        grounding = r["grounding_accuracy"] if r["grounding_accuracy"] is not None else 1.0
        quality = 0.6 * f1 + 0.2 * (1.0 - halluc) + 0.2 * grounding
        reasoning = 100.0 * (1.0 - r.get("error_rate", 0.0)) * quality  # abstentions earn nothing

    planning = m["planning"]["coverage_pct"]

    control = None
    c = m.get("control") or {}
    if "tracking_rmse_m" in c:
        control = 100.0 * math.exp(-c["tracking_rmse_m"] / 3.0)

    s = m["safety"]
    safety = 100.0
    if s["collisions"]:
        safety = 0.0
    else:
        safety -= min(40.0, 4.0 * s.get("geofence_breach_s", 0.0))
        safety -= min(30.0, 3.0 * s.get("low_altitude_s", 0.0))
        safety -= 10.0 if s.get("speed_limit_exceeded") else 0.0
        safety -= 15.0 if s.get("energy_budget_exceeded") else 0.0
        safety -= 10.0 if not s.get("returned_home", True) else 0.0

    scores = {
        "mission": mission,
        "perception": perception,
        "reasoning": reasoning,
        "planning": planning,
        "control": control,
        "safety": safety,
    }
    return {k: (None if v is None else _clamp(v)) for k, v in scores.items()}


def overall_score(scores: Dict[str, Optional[float]], collisions: int, weights: Optional[Dict[str, float]] = None) -> float:
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    num = sum(w[k] * v for k, v in scores.items() if v is not None)
    den = sum(w[k] for k, v in scores.items() if v is not None)
    total = num / den if den else 0.0
    return min(total, COLLISION_CAP) if collisions else total


def bootstrap_ci(values: Iterable[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> Dict[str, float]:
    v = np.asarray([x for x in values if x is not None], float)
    if len(v) == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    if len(v) == 1:
        return {"mean": float(v[0]), "lo": float(v[0]), "hi": float(v[0]), "n": 1}
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    return {
        "mean": float(v.mean()),
        "lo": float(np.percentile(means, 100 * alpha / 2)),
        "hi": float(np.percentile(means, 100 * (1 - alpha / 2))),
        "n": int(len(v)),
    }


def aggregate(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Group run results by agent and summarise each score with a bootstrap CI."""
    by_agent: Dict[str, List[Dict[str, Any]]] = {}
    for r in results:
        by_agent.setdefault(r["agent"], []).append(r)
    summary = {}
    for agent, rs in by_agent.items():
        keys = list(DEFAULT_WEIGHTS) + ["overall"]
        summary[agent] = {
            "runs": len(rs),
            "success_rate": float(np.mean([r["metrics"]["mission"]["success"] for r in rs])),
            "collision_runs": int(sum(r["metrics"]["safety"]["collisions"] > 0 for r in rs)),
            "scores": {k: bootstrap_ci(r["scores"].get(k) for r in rs) for k in keys},
        }
    return summary
