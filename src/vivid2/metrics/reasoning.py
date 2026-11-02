"""Reasoning (VLM) layer metrics: decision quality, grounding, cost, latency.

A frame counts as showing a target only if the target is actually visible.
With ground truth v2 (segmentation pixel counts), a target needs at least
``MIN_VISIBLE_PX`` visible pixels; a frame where targets are only partly
visible is "don't care" and not scored, and a target fully hidden by canopy
counts as absent. Ground truth v1 logs (no pixel counts) fall back to "any
annotation box in frame means visible".
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from vivid2.recorder import RunLog
from vivid2.sim.segmentation import visibility_state


def _in_box(px, box) -> bool:
    return box is not None and box[0] <= px[0] <= box[2] and box[1] <= px[1] <= box[3]


def reasoning_metrics(log: RunLog, layer: str = "reasoner") -> Optional[Dict[str, Any]]:
    recs = [d for d in log["decisions"] if d.get("layer") == layer]
    if not recs:
        return None
    frames = {f["frame_id"]: f for f in log["frames"]}
    tp = fp = tn = fn = 0
    errors = dont_care = fp_on_distractor = 0
    grounded = grounding_trials = 0
    for rec in recs:
        # A failed call or unusable reply is an abstention, not a "no target" answer:
        # counting it as a negative would reward a model that errors often.
        if rec["output"].get("error"):
            errors += 1
            continue
        frame = frames.get(rec["frame_id"], {})
        wh = (frame.get("width"), frame.get("height"))
        states = [(g, visibility_state(g, frame_wh=wh)) for g in frame.get("gt", [])]
        visible = [g for g, st in states if st == "visible"]
        if not visible and any(st == "partial" for _, st in states):
            dont_care += 1
            continue
        found = bool(rec["output"].get("target_found"))
        if found and visible:
            tp += 1
            px = rec["output"].get("pixel")
            if px is not None:
                grounding_trials += 1
                grounded += any(_in_box(px, g.get("bbox_vis") or g["bbox"]) for g in visible)
        elif found:
            fp += 1
            fp_on_distractor += bool(frame.get("distractors_visible"))
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
    scored = tp + fp + tn + fn
    return {
        "decisions": len(recs),
        "errors": errors,
        "error_rate": errors / len(recs),
        "dont_care": dont_care,
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "accuracy": (tp + tn) / scored if scored else None,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "hallucination_rate": fp / (fp + tn) if fp + tn else None,
        "fp_on_distractor": fp_on_distractor,
        "grounding_accuracy": grounded / grounding_trials if grounding_trials else None,
        "latency_mean_s": float(lat.mean()),
        "latency_p95_s": float(np.percentile(lat, 95)),
        "input_tokens": int(sum(r.get("input_tokens", 0) for r in recs)),
        "output_tokens": int(sum(r.get("output_tokens", 0) for r in recs)),
        "cost_usd": float(sum(r.get("cost_usd", 0.0) for r in recs)),
        "models": sorted({r.get("model", "") for r in recs}),
        "fallback_decisions": int(sum(bool(r.get("fallback")) for r in recs)),
    }
