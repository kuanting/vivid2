"""Claude's own assessment of a few recorded hiker photos, for the film's "AI" shots.

The photos go through the benchmark's VLM reasoner unchanged (``VLMReasoner``: same mission prompt,
same JSON schema, same image bytes as a live run). A VLM answers with a verdict, a confidence, ONE
point (where it sees the person) and a description, not a bounding box; the film draws that point
and the words. Answers are cached (``claude_cache.jsonl``), so re-composing costs nothing.

Output: <film dir>/claude.jsonl, one ``agents.decision_record`` per photo plus the target id.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from demo_movie.run_data import FilmRun
from demo_movie.story import CHEN, LIN, Story

EST_COST_PER_CALL = 0.03  # USD, generous for one 1568 px image + JSON answer at low effort


def pick_frames(story: Story) -> List[Tuple[str, int]]:
    """(target, frame id): for each hiker, the search photo it shows best in (and the closer look's best)."""
    out: List[Tuple[str, int]] = []
    for tid in (CHEN, LIN):
        f = story.finds.get(tid)
        ids = []
        if f:
            ids = [x for x in (f.survey_frame, f.closeup_frame) if x is not None]
        if not ids:
            best = story.run.best_frame(tid)
            if best:
                ids = [best["frame_id"]]
        for fid in dict.fromkeys(ids):
            out.append((tid, fid))
    return out


def assess(run: FilmRun, picks: List[Tuple[str, int]], out_dir: Path, model: Optional[str] = None,
           max_cost: float = 0.25, effort: str = "low") -> List[Dict[str, Any]]:
    from vivid2.openloop import load_frame
    from vivid2.registry import make_vlm
    from vivid2.stack.base import MissionSpec
    from vivid2.stack.reference.agents import decision_record
    from vivid2.stack.reference.reasoners import VLMReasoner

    if len(picks) * EST_COST_PER_CALL > max_cost:
        raise RuntimeError(f"{len(picks)} photos could cost about ${len(picks) * EST_COST_PER_CALL:.2f} > --max-cost {max_cost}")
    cache = out_dir / "claude_cache.jsonl"
    vlm = make_vlm({"vlm": "anthropic", "vlm_model": model, "vlm_effort": effort, "vlm_cache": str(cache)})
    reasoner = VLMReasoner(vlm)
    mission = MissionSpec.from_record(run.meta["mission_spec"])
    recs, spent = [], 0.0
    for tid, fid in picks:
        rec = run.frame(fid)
        frame = load_frame(run.run_dir, rec)
        a, usage = reasoner.assess(frame, [], mission)
        d = decision_record(frame, a, usage)
        d.update(target=tid, source="claude_frames (offline, same prompt and image bytes as a live run)")
        recs.append(d)
        spent += usage.cost_usd or 0.0
        print(f"  {tid} frame {fid}: found={a.target_found} conf={a.confidence:.2f} point={a.pixel} "
              f"${usage.cost_usd or 0:.3f}  {a.description[:90]}")
        if spent > max_cost:
            print(f"  stopping: spent ${spent:.2f} > ${max_cost}")
            break
    path = out_dir / "claude.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    print(f"{path}: {len(recs)} assessments, ${spent:.3f} (cached answers cost nothing)")
    return recs


def load(out_dir: Path) -> Dict[int, Dict[str, Any]]:
    """{frame id: decision record} from claude.jsonl (empty if not made)."""
    path = Path(out_dir) / "claude.jsonl"
    if not path.exists():
        return {}
    return {r["frame_id"]: r for r in (json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip())}
