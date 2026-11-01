"""Open-loop evaluation: replay a run's saved frames through a reasoner.

A closed-loop run flies the drone and the reasoner sees whatever frames its
own timing produces. To compare VLMs fairly (and to run a local model with
Unreal closed), record a frame bank once (``--frame-bank rgb``, usually with
the ``survey`` agent), then evaluate each model on exactly those frames:

    vivid2 eval-frames runs/blocks_v1/survey --vlm anthropic --vlm-model claude-sonnet-5-5
    vivid2 eval-frames runs/blocks_v1/survey --vlm local --vlm-model qwen2.5vl:7b --vlm-base-url http://localhost:11434/v1

Results land in ``<campaign>/openloop/<agent>/<run_id>/`` with the same
``decisions`` stream as a closed-loop run and are scored on the reasoning
layer only. Evaluation resumes per frame, so an interrupted local-model run
picks up where it stopped.
"""
from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np

from vivid2.recorder import STREAMS
from vivid2.stack.base import Frame, GTAnnotation, MissionSpec, Reasoner
from vivid2.stack.reference.agents import decision_record
from vivid2.vlm.base import decode_jpeg, sha256_hex


def find_banked_runs(paths: Iterable[Path]) -> List[Path]:
    """Closed-loop run directories under ``paths`` that recorded a frame bank."""
    found = []
    for root in paths:
        for meta_path in sorted(Path(root).rglob("meta.json")):
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("mode") != "open_loop" and (meta.get("frame_bank") or {}).get("mode", "off") != "off":
                found.append(meta_path.parent)
    return found


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_frame(run_dir: Path, rec: Dict[str, Any]) -> Frame:
    """Rebuild a Frame from a ``frames`` record and its banked JPEG (verified by hash)."""
    jpeg = (Path(run_dir) / rec["rgb"]).read_bytes()
    if rec.get("rgb_sha256") and sha256_hex(jpeg) != rec["rgb_sha256"]:
        raise ValueError(f"{run_dir}/{rec['rgb']} does not match its recorded hash")
    return Frame(
        t_ns=rec["t"], frame_id=rec["frame_id"], rgb=decode_jpeg(jpeg),
        cam_pos=np.asarray(rec["cam_pos"], float), cam_q=tuple(rec["cam_q"]), fov_deg=rec["fov_deg"],
        gt=[GTAnnotation(g["object_id"], tuple(g["bbox"])) for g in rec.get("gt", [])], jpeg=jpeg,
    )


def evaluate_run(
    run_dir: Path,
    reasoner: Reasoner,
    out_root: Path,
    agent_name: str,
    agent_desc: Dict[str, Any],
    concurrency: int = 4,
    log: Callable[[str], None] = print,
) -> Path:
    """Assess every banked frame of one run; returns the open-loop run directory."""
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    if "mission_spec" not in meta:
        raise ValueError(f"{run_dir} has no mission_spec in meta.json (recorded before frame banks existed)")
    mission = MissionSpec(**meta["mission_spec"])
    frames = [f for f in _read_jsonl(run_dir / "frames.jsonl") if f.get("rgb")]

    out_dir = Path(out_root) / agent_name / meta["run_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    decisions_path = out_dir / "decisions.jsonl"
    done = {d["frame_id"] for d in _read_jsonl(decisions_path) if not d["output"].get("error")}
    if done:  # keep the good decisions; redo errored ones
        kept = [d for d in _read_jsonl(decisions_path) if d["frame_id"] in done]
        decisions_path.write_text("".join(json.dumps(d) + "\n" for d in kept), encoding="utf-8")
    elif decisions_path.exists():
        decisions_path.unlink()
    todo = [f for f in frames if f["frame_id"] not in done]
    log(f"  {meta['run_id']}: {len(frames)} frames, {len(done)} already assessed, {len(todo)} to go")

    lock = threading.Lock()

    def assess(rec: Dict[str, Any]) -> None:
        frame = load_frame(run_dir, rec)
        assessment, usage = reasoner.assess(frame, [], mission)
        line = json.dumps(decision_record(frame, assessment, usage), default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
        with lock, open(decisions_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        list(pool.map(assess, todo))

    # Sort decisions by frame so files are identical however the threads interleaved.
    decisions = sorted(_read_jsonl(decisions_path), key=lambda d: d["frame_id"])
    decisions_path.write_text("".join(json.dumps(d) + "\n" for d in decisions), encoding="utf-8")
    (out_dir / "frames.jsonl").write_text("".join(json.dumps(f) + "\n" for f in frames), encoding="utf-8")
    for s in STREAMS:
        (out_dir / f"{s}.jsonl").touch()
    open_meta = {k: meta[k] for k in ("run_id", "scenario_id", "seed", "params", "mission", "targets", "map")
                 if k in meta}
    open_meta.update({"mode": "open_loop", "source_run": str(run_dir), "agent": agent_desc,
                      "mission_spec": meta["mission_spec"]})
    (out_dir / "meta.json").write_text(json.dumps(open_meta, indent=2), encoding="utf-8")
    return out_dir


def evaluate_runs(
    run_dirs: List[Path],
    reasoner: Reasoner,
    agent_name: str,
    agent_desc: Dict[str, Any],
    out_root: Optional[Path] = None,
    concurrency: int = 4,
    log: Callable[[str], None] = print,
) -> List[Dict[str, Any]]:
    """Evaluate and score each run; ``out_root`` defaults to ``<campaign>/openloop`` next to the source runs."""
    from vivid2.runner import score_run_dir

    results = []
    for run_dir in run_dirs:
        root = Path(out_root) if out_root else Path(run_dir).parent.parent / "openloop"
        out_dir = evaluate_run(run_dir, reasoner, root, agent_name, agent_desc, concurrency, log)
        results.append(score_run_dir(out_dir))
    return results
