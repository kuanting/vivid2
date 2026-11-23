"""Evaluate the VLM damage mapper offline, on a saved run's photos (docs/damage.md).

    python tools/eval_damage_mapper.py RUN_DIR [--answers truth|anthropic] [--out DIR] [--save-sent]

RUN_DIR is a closed-loop run with a damage task that recorded its frame bank with depth
(``--frame-bank rgbd``), usually a ``coverage_gt`` oracle run. The mapper picks photos from the
run's survey photos as it would in flight, asks about each, and its reports are scored with the
damage metrics against the run's ground truth:

    --answers truth      answers from the ground truth, as a perfect model would give them (free): the
                         best this mapper can do with these photos, and a check of the whole pipeline
    --answers anthropic  Claude (--vlm-model, --vlm-effort). Costs money: prints an estimate and asks
                         first unless --yes. Answers are cached in OUT/vlm_cache.jsonl, so a rerun is free
    --dry-run            pick the photos and write what would be sent (OUT/<run>/sent/), ask nothing

The scored copy of the run is written to OUT/<run_id>/ (default: RUN_DIR/../damage_mapper_<answers>).
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from vivid2.damage.gt import DamageGT, load_gt
from vivid2.metrics.cameras import frame_camera, search_cameras
from vivid2.metrics.damage import damage_metrics
from vivid2.openloop import load_frame
from vivid2.recorder import STREAMS, RunLog
from vivid2.runner import DUT_DEM_CELL_M
from vivid2.stack.base import Frame, MissionSpec, UsageStats, damage_record
from vivid2.stack.reference.damage_vlm import MAX_CALLS, MapperPhoto, VLMDamageMapper, cell_label
from vivid2.terrain import TerrainModel
from vivid2.vlm.base import VLMError, VLMProvider

TOKENS_PER_IMAGE = 2500   # a 1568 x 1176 JPEG (width x height / 750)
PROMPT_TOKENS = 1000
OUTPUT_TOKENS = 1500      # the reply and the model's thinking at low effort


class _NoModel(VLMProvider):
    model = "ground_truth"

    def query(self, images, prompt, schema, system=None):
        raise VLMError("no model: answers come from the ground truth")


class TruthAnsweringMapper(VLMDamageMapper):
    """The VLM mapper with every question answered from the damage ground truth: the cells where a
    class covers at least half of the ground, the true road states and grades, the clues in the photo."""

    def __init__(self, gt: DamageGT, clues: List[Dict[str, Any]], **kw):
        super().__init__(_NoModel(), **kw)
        self.gt, self.truth_clues = gt, clues

    def _ask(self, photo: MapperPhoto) -> Tuple[Dict[str, Any], UsageStats]:
        g = photo.ground
        cols, rows = self.grid
        out: Dict[str, Any] = {"summary": "the ground truth"}
        ok = np.isfinite(g.ne[:, 0])
        if "areas" in self.products:
            total = np.bincount(g.box[ok], minlength=cols * rows)
            areas = []
            for k in self.area_classes:
                inside = np.zeros(len(g.ne), bool)
                inside[ok] = self.gt.class_at(k, g.ne[ok, 0], g.ne[ok, 1])
                pos = np.bincount(g.box[inside], minlength=cols * rows)
                areas.append({"area_class": k, "cells": [cell_label(b % cols, b // cols) for b in range(cols * rows)
                                                          if total[b] and pos[b] >= 0.5 * total[b]]})
            out["areas"] = areas
        if "roads" in self.products:
            truth = {r["id"]: r["blocked"] for r in self.gt.roads}
            out["roads"] = [{"segment": s, "status": "blocked" if truth[s] else "open"} for s in photo.roads]
        if "buildings" in self.products:
            truth = {b["id"]: b["grade"] for b in self.gt.buildings}
            out["buildings"] = [{"building": b, "grade": truth[b]} for b in photo.buildings]
        if "clues" in self.products:
            clues = []
            for c in self.truth_clues:
                u, v, inside = g.project(np.asarray([c["ned"]], float))
                if inside[0]:
                    clues.append({"kind": c.get("kind", "other"), "description": c["id"],
                                  "point": [float(u[0] / g.W * 1000), float(v[0] / g.H * 1000)]})
            out["clues"] = clues
        return out, UsageStats(model="ground_truth")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] \
        if path.exists() else []


def load_depth(run_dir: Path, rec: Dict[str, Any]) -> Optional[np.ndarray]:
    """A banked depth image (uint16 mm, half resolution, 0 = no data) at the photo's full resolution, metres."""
    import cv2

    if not rec.get("depth"):
        return None
    raw = cv2.imread(str(Path(run_dir) / rec["depth"]), cv2.IMREAD_UNCHANGED)
    if raw is None:
        return None
    d = raw.astype(np.float32) / 1000.0
    full = np.repeat(np.repeat(d, 2, axis=0), 2, axis=1)
    return full[:rec["height"], :rec["width"]]


def stack_mission(meta: Dict[str, Any]) -> MissionSpec:
    """The MissionSpec the stack flew with, its coarse DEM rebuilt as the runner builds it."""
    mission = MissionSpec.from_record(meta["mission_spec"])
    src = (meta.get("dut_terrain") or {}).get("file") or (meta.get("terrain") or {}).get("file")
    mission.terrain = (TerrainModel.load(src).downsample(DUT_DEM_CELL_M) if src
                       else TerrainModel.flat(mission.ground_down_m))
    return mission


def _pose_frame(rec: Dict[str, Any]) -> Frame:
    """A frame with the photo's pose only (enough for the mapper to decide whether it wants it)."""
    return Frame(t_ns=rec["t"], frame_id=rec["frame_id"], rgb=np.zeros((rec["height"], rec["width"], 3), np.uint8),
                 cam_pos=np.asarray(rec["cam_pos"], float), cam_q=tuple(rec["cam_q"]), fov_deg=rec["fov_deg"],
                 camera=rec.get("camera", "Payload"), gimbal=rec.get("gimbal"))


def evaluate(run_dir: Path, mapper: VLMDamageMapper, out_root: Path, concurrency: int = 1, dry_run: bool = False,
             save_sent: bool = False, log=print) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    if not meta.get("damage"):
        raise SystemExit(f"{run_dir} has no damage task")
    if (meta.get("frame_bank") or {}).get("mode") != "rgbd":
        log("  note: no depth in this run's frame bank: photos are placed on the DEM only")
    mission = stack_mission(meta)
    search = search_cameras(meta)
    frames = [f for f in _read_jsonl(run_dir / "frames.jsonl") if frame_camera(f) in search and f.get("rgb")]
    picked = [f for f in frames if mapper.wants(_pose_frame(f), mission)]
    log(f"  {meta['run_id']}: {len(frames)} survey photos, {len(picked)} picked")
    out_dir = Path(out_root) / meta["run_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    sent_dir = out_dir / "sent"

    def save(photo: MapperPhoto) -> None:
        sent_dir.mkdir(exist_ok=True)
        fid = photo.frame.frame_id
        for name, img in zip(("photo", "grid"), photo.images):
            (sent_dir / f"{fid:06d}_{name}.jpg").write_bytes(img)
        (sent_dir / f"{fid:06d}_prompt.txt").write_text(photo.prompt, encoding="utf-8")

    if save_sent:
        mapper.on_photo = save

    def look(rec: Dict[str, Any]):
        frame = load_frame(run_dir, rec)
        frame.depth = load_depth(run_dir, rec)
        if dry_run:
            save(mapper.prepare(frame, mission))
            return rec, [], UsageStats(), {}
        reports, usage, out = mapper.observe(frame, mission)
        return rec, reports, usage, out

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        looked = list(pool.map(look, picked))
    if dry_run:
        log(f"  wrote what would be sent for {len(picked)} photos to {sent_dir}")
        return {}
    damage, decisions = [], []
    for rec, reports, usage, out in sorted(looked, key=lambda x: x[0]["frame_id"]):
        decisions.append({"t": rec["t"], "frame_id": rec["frame_id"], "layer": "damage_mapper", "output": out,
                          "latency_s": usage.latency_s, "input_tokens": usage.input_tokens,
                          "output_tokens": usage.output_tokens, "cost_usd": usage.cost_usd, "model": usage.model})
        damage += [{"t": rec["t"], "t_report": rec["t"], **damage_record(r)} for r in reports]
    end_t = max((f["t"] for f in frames), default=0)
    damage += [{"t": r.t_ns, "t_report": end_t, **damage_record(r)} for r in mapper.finish(mission)]
    out_meta = {**meta, "mode": "open_loop", "source_run": str(run_dir),
                "agent": {"agent": f"damage_mapper:{mapper.provider.model}", "damage_mapper": mapper.describe()}}
    (out_dir / "meta.json").write_text(json.dumps(out_meta, indent=2), encoding="utf-8")
    (out_dir / "frames.jsonl").write_text("".join(json.dumps(f) + "\n" for f in frames), encoding="utf-8")
    (out_dir / "damage.jsonl").write_text("".join(json.dumps(r) + "\n" for r in damage), encoding="utf-8")
    (out_dir / "decisions.jsonl").write_text("".join(json.dumps(d, default=_jsonable) + "\n" for d in decisions),
                                             encoding="utf-8")
    for s in STREAMS:
        (out_dir / f"{s}.jsonl").touch()
    metrics = damage_metrics(RunLog.load(out_dir))
    (out_dir / "damage_metrics.json").write_text(json.dumps(metrics, indent=2, default=_jsonable), encoding="utf-8")
    return metrics


def _jsonable(o):
    return o.tolist() if hasattr(o, "tolist") else sorted(o) if isinstance(o, set) else str(o)


def summary(m: Dict[str, Any]) -> str:
    lines = [f"  damage score {m['score']:.1f}  components " +
             ", ".join(f"{k} {v:.2f}" for k, v in m["components"].items())]
    for k, a in (m.get("areas") or {}).items():
        lines.append(f"    {k}: IoU {a['iou']:.2f}, precision {a['precision'] or 0:.2f}, recall {a['recall'] or 0:.2f}"
                     f" ({a['reported_m2']:.0f} of {a['gt_m2']:.0f} m2 reported)")
    if m.get("roads"):
        r = m["roads"]
        lines.append(f"    roads: blocked {r['blocked']}, reported blocked {r['reported_blocked']}")
    if m.get("buildings"):
        lines.append(f"    buildings: confusion (truth x reported) {m['buildings']['confusion_gt_by_reported']}")
    if m.get("clues"):
        c = m["clues"]
        lines.append(f"    clues: {c['found']} of {c['clues']} found, {c['false_alarms']} false alarms, matched {c['matched']}")
    if m.get("buried"):
        lines.append(f"    buried: {m['buried']['covered']}")
    if m.get("violations"):
        lines.append(f"    violations: {m['violations']}")
    if m.get("mapper"):
        mp = m["mapper"]
        lines.append(f"    model: {mp['calls']} calls, {mp['errors']} errors, ${mp['cost_usd']:.2f}, "
                     f"{mp['input_tokens']} in / {mp['output_tokens']} out tokens")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", type=Path, help="run directories (with a damage task and an rgbd frame bank)")
    ap.add_argument("--answers", choices=["truth", "anthropic"], default="truth")
    ap.add_argument("--vlm-model", default=None)
    ap.add_argument("--vlm-effort", default="low")
    ap.add_argument("--max-calls", type=int, default=MAX_CALLS)
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--save-sent", action="store_true", help="also write the images and prompt sent for each photo")
    ap.add_argument("--yes", action="store_true", help="don't ask before spending on model calls")
    args = ap.parse_args(argv)

    results = {}
    for run_dir in args.runs:
        meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
        out_root = args.out or run_dir.parent.parent / f"damage_mapper_{args.answers}"
        if args.answers == "truth":
            gt = load_gt(meta["damage"]["gt"]["file"], meta["damage"]["gt"].get("sha256"))
            mapper: VLMDamageMapper = TruthAnsweringMapper(gt, meta.get("clues") or [], max_calls=args.max_calls)
        else:
            from vivid2.registry import make_vlm
            from vivid2.vlm.anthropic_provider import DEFAULT_MODEL, PRICES

            model = args.vlm_model or DEFAULT_MODEL
            provider = make_vlm({"vlm": "anthropic", "vlm_model": model, "vlm_effort": args.vlm_effort,
                                 "vlm_cache": str(out_root / "vlm_cache.jsonl")})
            mapper = VLMDamageMapper(provider, max_calls=args.max_calls)
            if not args.dry_run:
                probe = VLMDamageMapper(provider, max_calls=args.max_calls)
                mission = stack_mission(meta)
                frames = [f for f in _read_jsonl(run_dir / "frames.jsonl")
                          if frame_camera(f) in search_cameras(meta) and f.get("rgb")]
                n = sum(probe.wants(_pose_frame(f), mission) for f in frames)
                pin, pout = PRICES.get(model, (0.0, 0.0))
                est = n * ((2 * TOKENS_PER_IMAGE + PROMPT_TOKENS) * pin + OUTPUT_TOKENS * pout) / 1e6
                print(f"{run_dir.name}: {n} model calls to {model}, about ${est:.2f} (cached answers are free)")
                if not args.yes and input("Proceed? [y/N] ").strip().lower() != "y":
                    return 1
        print(f"{run_dir} ({args.answers})")
        m = evaluate(run_dir, mapper, out_root, args.concurrency if args.answers == "anthropic" else 1,
                     args.dry_run, args.save_sent)
        if m:
            print(summary(m))
            results[str(run_dir)] = m["score"]
    if len(results) > 1:
        print(f"mean damage score {np.mean(list(results.values())):.1f} over {len(results)} runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
