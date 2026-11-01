"""Reasoning (VLM) layer metrics: decision quality, grounding, cost, latency."""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from vivid2.recorder import RunLog


def reasoning_metrics(log: RunLog, layer: str = "reasoner") -> Optional[Dict[str, Any]]:
    recs = [d for d in log["decisions"] if d.get("layer") == layer]
    if not recs:
        return None
    frames = {f["frame_id"]: f for f in log["frames"]}
    tp = fp = tn = fn = 0
    errors = 0
    grounded = grounding_trials = 0
    for rec in recs:
        # A failed call or unusable reply is an abstention, not a "no target" answer:
        # counting it as a negative would reward a model that errors often.
        if rec["output"].get("error"):
            errors += 1
            continue
        gts = frames.get(rec["frame_id"], {}).get("gt", [])
        found = bool(rec["output"].get("target_found"))
        visible = len(gts) > 0
        if found and visible:
            tp += 1
            px = rec["output"].get("pixel")
            if px is not None:
                grounding_trials += 1
                grounded += any(g["bbox"][0] <= px[0] <= g["bbox"][2] and g["bbox"][1] <= px[1] <= g["bbox"][3] for g in gts)
        elif found:
            fp += 1
        elif visible:
            fn += 1
        else:
            tn += 1

    lat = np.asarray([r.get("latency_s", 0.0) for r in recs])
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    if precision is None or recall is None:
        f1 = None
    else:
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    scored = len(recs) - errors
    return {
        "decisions": len(recs),
        "errors": errors,
        "error_rate": errors / len(recs),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "accuracy": (tp + tn) / scored if scored else None,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "hallucination_rate": fp / (fp + tn) if fp + tn else None,
        "grounding_accuracy": grounded / grounding_trials if grounding_trials else None,
        "latency_mean_s": float(lat.mean()),
        "latency_p95_s": float(np.percentile(lat, 95)),
        "input_tokens": int(sum(r.get("input_tokens", 0) for r in recs)),
        "output_tokens": int(sum(r.get("output_tokens", 0) for r in recs)),
        "cost_usd": float(sum(r.get("cost_usd", 0.0) for r in recs)),
        "models": sorted({r.get("model", "") for r in recs}),
        "fallback_decisions": int(sum(bool(r.get("fallback")) for r in recs)),
    }
