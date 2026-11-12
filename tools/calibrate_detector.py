"""Choose the YOLO baseline's detection threshold on frames the benchmark does not use.

First record every frame's best detection score with a zero threshold:

    vivid2 eval-frames runs/forest_canopy/survey-dip --agent coverage_yolo --yolo-threshold 0 --out runs/_yolo_calib/canopy

then sweep the threshold over those open-loop runs:

    python tools/calibrate_detector.py runs/_yolo_calib/canopy/coverage_yolo-yolox-s/forest_hiker_01-1acc3afa ...

Each threshold is scored exactly as the benchmark scores the reasoning layer
(``reasoning_metrics``: a frame with a visible target is a hit if the agent
says "found"; partly visible targets are don't-care), separately for survey
frames (40 m or more above the ground) and low frames (close looks and dips).
The F1-best threshold over all frames is printed last.
"""
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

from vivid2.metrics.reasoning import reasoning_metrics
from vivid2.recorder import RunLog

HIGH_M = 40.0


def load(run_dir: Path):
    log = RunLog.load(run_dir)
    ground = log.meta["mission_spec"]["ground_down_m"]
    # straight-down payload photos only (navigation and tilted photos are binned by nothing here)
    height = {f["frame_id"]: ground - f["cam_pos"][2] for f in log["frames"]
              if f.get("camera", "Payload") == "Payload" and not f.get("gimbal")}
    return log, height


def at_threshold(log: RunLog, keep, t: float):
    """The run's decisions as they would be at threshold t, for the frames ``keep`` selects."""
    out = copy.copy(log)
    decisions = []
    for d in log["decisions"]:
        if not keep(d["frame_id"]):
            continue
        d = copy.deepcopy(d)
        d["output"]["target_found"] = bool(d["output"].get("target_found")) and d["output"]["confidence"] >= t
        decisions.append(d)
    out.streams = dict(log.streams, decisions=decisions)
    return out


def sweep(runs, keep_fn, thresholds):
    rows = []
    for t in thresholds:
        tp = fp = fn = tn = 0
        for log, height in runs:
            m = reasoning_metrics(at_threshold(log, lambda fid: fid in height and keep_fn(height[fid]), t))
            if m:
                c = m["confusion"]
                tp, fp, fn, tn = tp + c["tp"], fp + c["fp"], fn + c["fn"], tn + c["tn"]
        p = tp / (tp + fp) if tp + fp else None
        r = tp / (tp + fn) if tp + fn else None
        f1 = 2 * p * r / (p + r) if p and r else 0.0
        rows.append((t, tp, fp, fn, tn, p, r, f1))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="open-loop run directories (eval-frames output, threshold 0)")
    ap.add_argument("--thresholds", default="0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.5,0.6,0.7")
    args = ap.parse_args()
    runs = [load(Path(r)) for r in args.runs]
    thresholds = [float(t) for t in args.thresholds.split(",")]
    best = None
    for label, keep in (("survey (>= 40 m)", lambda h: h >= HIGH_M), ("low (< 40 m)", lambda h: h < HIGH_M),
                        ("all frames", lambda h: True)):
        print(f"\n{label}")
        print("  thr    tp   fp   fn    tn  precision  recall    F1")
        rows = sweep(runs, keep, thresholds)
        for t, tp, fp, fn, tn, p, r, f1 in rows:
            fmt = lambda v: f"{v:9.2f}" if v is not None else "      n/a"
            print(f"  {t:4.2f} {tp:5d} {fp:4d} {fn:4d} {tn:5d} {fmt(p)} {fmt(r)} {f1:5.2f}")
        if label == "all frames":
            best = max(rows, key=lambda row: (row[7], row[0]))
    if best is None or best[7] == 0.0:
        print("\nno threshold finds anything: keep the conventional 0.3")
        return 1
    print(f"\nF1-best threshold over all frames: {best[0]:.2f} (precision {best[5]:.2f}, recall {best[6]:.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
