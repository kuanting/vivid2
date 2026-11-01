"""Perception metrics against simulator ground truth.

With ground truth v2, boxes are the tight box around each target's visible
pixels (the sim's own box covers the whole object, occluded or not, which is
too loose for IoU against a detector's box).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from vivid2.recorder import RunLog
from vivid2.sim.segmentation import visibility_state


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def average_precision(scores: List[float], is_tp: List[bool], n_gt: int) -> float:
    """VOC-style all-point interpolated AP."""
    if n_gt == 0:
        return float("nan")
    if not scores:
        return 0.0
    order = np.argsort(-np.asarray(scores))
    tp = np.asarray(is_tp, float)[order]
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    recall = ctp / n_gt
    precision = ctp / np.maximum(ctp + cfp, 1e-9)
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[0.0], precision, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def perception_metrics(
    log: RunLog, layer: str = "perception", iou_thr: float = 0.5, score_thr: float = 0.5
) -> Optional[Dict[str, Any]]:
    det_recs = [d for d in log["detections"] if d.get("layer") == layer]
    if not det_recs:
        return None
    frames = {f["frame_id"]: f for f in log["frames"]}
    dets_by_frame = defaultdict(list)
    for rec in det_recs:
        dets_by_frame[rec["frame_id"]].extend(rec["dets"])

    targets = {t["id"]: np.asarray(t["ned"], float) for t in log.meta.get("targets", [])}
    scores, is_tp = [], []
    n_gt = 0
    tp_at_thr = fp_at_thr = 0
    first_detect: Dict[str, Dict[str, float]] = {}

    evaluated = [fid for fid in frames if fid in dets_by_frame]
    for fid in evaluated:
        # Visible targets are matched against the box around their visible pixels; partly
        # visible or hidden ones are ignore regions: a detection there is neither a hit nor a false alarm.
        gts, ignore = [], []
        for g in frames[fid].get("gt", []):
            box = g.get("bbox_vis") or g["bbox"]
            (gts if visibility_state(g) == "visible" else ignore).append({**g, "box": box})
        n_gt += len(gts)
        used = set()
        for det in sorted(dets_by_frame[fid], key=lambda d: -d["score"]):
            best_j, best_iou = -1, iou_thr
            for j, gt in enumerate(gts):
                if j in used:
                    continue
                v = iou(det["bbox"], gt["box"])
                if v >= best_iou:
                    best_j, best_iou = j, v
            hit = best_j >= 0
            if not hit and any(iou(det["bbox"], g["box"]) >= iou_thr for g in ignore):
                continue
            if hit:
                used.add(best_j)
                oid = gts[best_j]["object_id"]
                if oid not in first_detect and oid in targets and det["score"] >= score_thr:
                    rng = float(np.linalg.norm(np.asarray(frames[fid]["cam_pos"]) - targets[oid]))
                    first_detect[oid] = {"t": frames[fid]["t"], "range_m": rng}
            scores.append(det["score"])
            is_tp.append(hit)
            if det["score"] >= score_thr:
                tp_at_thr += int(hit)
                fp_at_thr += int(not hit)

    precision = tp_at_thr / (tp_at_thr + fp_at_thr) if (tp_at_thr + fp_at_thr) else None
    recall = tp_at_thr / n_gt if n_gt else None
    return {
        "frames_evaluated": len(evaluated),
        "gt_instances": n_gt,
        "ap50": average_precision(scores, is_tp, n_gt),
        "precision": precision,
        "recall": recall,
        "false_positives": fp_at_thr,
        "first_detection": first_detect,
        "mean_first_detection_range_m": (
            float(np.mean([v["range_m"] for v in first_detect.values()])) if first_detect else None
        ),
    }
