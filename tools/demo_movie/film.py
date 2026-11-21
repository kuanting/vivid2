"""The demo movie "The Golden Hour": one command per production step (docs/demo_movie.md).

    python tools/demo_movie/film.py editor                      # start the editor for filming
    python tools/demo_movie/film.py probe                       # P0: what the editor side has
    python tools/demo_movie/film.py plan    --run RUN_DIR       # plan.json from the real run
    python tools/demo_movie/film.py claude  --run RUN_DIR       # Claude on the hiker frames
    python tools/demo_movie/film.py ue-build                    # film level + sequences
    python tools/demo_movie/film.py ue-render --preset preview  # Movie Render Queue
    python tools/demo_movie/film.py compose --run RUN_DIR       # data shots + edit + encode

Everything is written under runs/_film/<name>/ (git-ignored).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from demo_movie import ue_exec  # noqa: E402

PROJECT = ROOT / "unreal" / "VIVID2Env" / "VIVID2Env.uproject"
UE_DIR = Path(__file__).resolve().parent / "ue"
FILM_DIR = ROOT / "runs" / "_film"

HIKER_BPS = ["/Game/VIVID2/Targets/VIVID2_Hiker_RedJacket_LyingSide", "/Game/VIVID2/Targets/VIVID2_Hiker_YellowJacket_Waving"]
DRONE_MESHES = ["/Drone/Quadrotor1", "/Drone/PropellerRed", "/Drone/PropellerWhite"]
FIRE_BPS = ["/Game/VIVID2/Fire/VIVID2_Fire_Large", "/Game/VIVID2/Fire/VIVID2_Fire_Smolder"]
CLIPS = ["MM_Walk_Fwd", "MF_Walk_Fwd", "MM_Walk_InPlace", "MF_Walk_InPlace", "MM_Fall_Loop", "MM_Death_Front_01",
         "Lying_on_the_Side_with_One_Arm_Anim", "Lying_on_the_Side_with_One_Arm", "Both_Arm_Wave_Anim", "Both_Arm_Wave",
         "MM_Idle", "MF_Idle"]


def cmd_editor(a) -> None:
    cmd = ue_exec.editor_command(PROJECT, a.level)
    print(" ".join(cmd))
    subprocess.Popen(" ".join(f'"{c}"' if " " in c and not c.startswith("-") else c for c in cmd))


def cmd_probe(a) -> None:
    with ue_exec.EditorPython() as ed:
        out = ed.run_file(UE_DIR / "probe_env.py", {"assets": HIKER_BPS + DRONE_MESHES + FIRE_BPS,
                                                     "anims": CLIPS, "hikers": HIKER_BPS})
    print(json.dumps(ue_exec.last_json(out), indent=1))


def film_dir(name: str) -> Path:
    d = FILM_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_plan(plan) -> Path:
    path = film_dir(plan["name"]) / "plan.json"
    path.write_text(json.dumps(plan, indent=1), encoding="utf-8")
    print(f"plan: {path} ({len(plan['actors'])} actors, {len(plan['shots'])} shots, "
          f"{sum(s['frames'] for s in plan['shots'])} frames)")
    return path


def cmd_probe_plan(a) -> None:
    from demo_movie.probe_plan import probe_plan
    from demo_movie.run_data import load_run
    write_plan(probe_plan(load_run(a.run), a.frame))


def _story(run_dir):
    from demo_movie.run_data import load_run
    from demo_movie.story import build_story
    return build_story(load_run(run_dir))


def cmd_plan(a) -> None:
    from demo_movie.golden_plan import golden_plan
    story = _story(a.run)
    for note in story.notes:
        print(f"  note: {note}")
    for tid, f in story.finds.items():
        print(f"  {tid}: found by {f.drone} at {f.report_rt:.1f} s, {f.error_m:.1f} m off, frame {f.frame_id}, "
              f"closer look {'yes' if f.verify else 'no'}")
    plan = golden_plan(story)
    plan["run_dir"] = str(Path(a.run).resolve())
    write_plan(plan)


def cmd_compose(a) -> None:
    import time
    from demo_movie import compose as cp
    from demo_movie.edit import timeline
    from demo_movie.encode import VideoOut
    plan = json.loads((film_dir(a.name) / "plan.json").read_text(encoding="utf-8"))
    story = _story(a.run or plan["run_dir"])
    root = film_dir(a.name) / "ue" / a.preset
    quad, gimbal = film_dir(a.name) / "quad", film_dir(a.name) / "gimbal"
    from demo_movie import claude_frames
    from demo_movie.run_data import load_run
    contrast = load_run(a.contrast_run) if a.contrast_run else None
    items = timeline(story, plan, root, quad if quad.exists() else None, claude_frames.load(film_dir(a.name)),
                     gimbal if gimbal.exists() else None, contrast)
    print(f"cut: {len(items)} shots, {sum(it.n for it in items)} frames ({sum(it.n for it in items) / 30:.1f} s)")
    out = Path(a.out) if a.out else film_dir(a.name) / f"{a.name}{'' if a.preset == 'final' else '_' + a.preset}.mp4"
    w, h = (1920, 1080) if not a.half else (960, 540)
    vo = VideoOut(out, w, h, 30, Path(a.music) if a.music else None, duration_s=sum(it.n for it in items) / 30.0)
    t0 = time.time()
    frames = range(a.start * 30, a.end * 30) if a.end else None
    n = cp.render(items, vo, lambda g, tot: print(f"  {g}/{tot} frames  {time.time() - t0:5.0f} s", flush=True)
                  if g % 300 == 0 else None, frames)
    vo.close()
    print(f"{out}: {n} frames, {vo.backend}, {time.time() - t0:.0f} s")


def cmd_claude(a) -> None:
    import os
    from demo_movie import claude_frames
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set")
    plan = json.loads((film_dir(a.name) / "plan.json").read_text(encoding="utf-8"))
    story = _story(a.run or plan["run_dir"])
    picks = claude_frames.pick_frames(story)
    print(f"{len(picks)} photos: {picks}")
    claude_frames.assess(story.run, picks, film_dir(a.name), model=a.model, max_cost=a.max_cost)


def cmd_ue_assets(a) -> None:
    from demo_movie import planlib as pl
    with ue_exec.EditorPython() as ed:
        out = ed.run_file(UE_DIR / "film_assets.py", {"paints": pl.PAINTS, "glows": pl.GLOWS, "inplace": pl.INPLACE_CLIPS})
    print(json.dumps(ue_exec.last_json(out), indent=1))


def cmd_ue_build(a) -> None:
    plan_path = film_dir(a.name) / "plan.json"
    with ue_exec.EditorPython() as ed:
        out = ed.run_file(UE_DIR / "film_build.py", {"plan": str(plan_path), "shots": a.shots})
    res = ue_exec.last_json(out)
    print(json.dumps(res, indent=1))


def cmd_ue_render(a) -> None:
    import time
    plan = json.loads((film_dir(a.name) / "plan.json").read_text(encoding="utf-8"))
    shots = [{"id": s["id"], "sequence": f"{plan['seq_dir']}/LS_{s['id']}"} for s in plan["shots"]
             if not a.shots or s["id"] in a.shots]
    out_dir = film_dir(a.name) / "ue" / a.preset
    args = {"out": str(out_dir), "preset": a.preset, "shots": shots, "map": plan["level"]["film"]}
    if a.res:
        args["res"] = a.res
    with ue_exec.EditorPython() as ed:
        print(ue_exec.last_json(ed.run_file(UE_DIR / "film_render.py", args)))
    status = out_dir / "render_status.json"
    t0 = time.time()
    while a.wait:
        time.sleep(5)
        st = json.loads(status.read_text()) if status.exists() else {}
        n = sum(len(list((out_dir / s["id"]).glob("*.png"))) for s in shots)
        print(f"  {time.time() - t0:6.0f} s  {n} frames  {st.get('state')}  done={[d.get('id') for d in st.get('done', [])]}",
              flush=True)
        if st.get("state") == "finished":
            print(json.dumps(st))
            break


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("editor", help="start the Unreal editor with MCP + Python remote execution")
    s.add_argument("--level", default="/Game/VIVID2/Maps/ForestFire")
    s.set_defaults(fn=cmd_editor)
    s = sub.add_parser("probe", help="P0: check the editor has what the film needs")
    s.set_defaults(fn=cmd_probe)
    s = sub.add_parser("probe-plan", help="P0: test shots (registration, motion) from a ForestFire run")
    s.add_argument("--run", required=True)
    s.add_argument("--frame", type=int, default=67)
    s.set_defaults(fn=cmd_probe_plan)
    s = sub.add_parser("plan", help="plan the story shots from the real run (plan.json)")
    s.add_argument("--run", required=True)
    s.set_defaults(fn=cmd_plan)
    s = sub.add_parser("compose", help="data shots + edit + encode (placeholders for unrendered shots)")
    s.add_argument("--name", default="golden_hour")
    s.add_argument("--run", help="run directory (default: the plan's)")
    s.add_argument("--preset", choices=["preview", "final"], default="preview")
    s.add_argument("--music", help="music file (needs imageio-ffmpeg)")
    s.add_argument("--contrast-run", help="another run whose false alarm is shown as a counter-example")
    s.add_argument("--out")
    s.add_argument("--half", action="store_true", help="960 x 540 output (quick look)")
    s.add_argument("--start", type=int, default=0, help="first second")
    s.add_argument("--end", type=int, default=0, help="last second (0 = all)")
    s.set_defaults(fn=cmd_compose)
    s = sub.add_parser("claude", help="Claude's assessment of the hiker photos (real answers, cached)")
    s.add_argument("--name", default="golden_hour")
    s.add_argument("--run", help="run directory (default: the plan's)")
    s.add_argument("--model", default=None)
    s.add_argument("--max-cost", type=float, default=0.25)
    s.set_defaults(fn=cmd_claude)
    s = sub.add_parser("ue-assets", help="create/update the film's materials in the editor")
    s.set_defaults(fn=cmd_ue_assets)
    s = sub.add_parser("ue-build", help="build the film level and shot sequences in the editor")
    s.add_argument("--name", default="golden_hour")
    s.add_argument("--shots", nargs="*")
    s.set_defaults(fn=cmd_ue_build)
    s = sub.add_parser("ue-render", help="render shot sequences with Movie Render Queue")
    s.add_argument("--name", default="golden_hour")
    s.add_argument("--preset", choices=["preview", "final"], default="preview")
    s.add_argument("--shots", nargs="*")
    s.add_argument("--res", type=int, nargs=2)
    s.add_argument("--no-wait", dest="wait", action="store_false")
    s.set_defaults(fn=cmd_ue_render)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
