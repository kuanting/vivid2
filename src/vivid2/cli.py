"""vivid2 command line.

    vivid2 list-scenarios campaigns/smoke.jsonc
    vivid2 run --campaign campaigns/smoke.jsonc --agent coverage_gt
    vivid2 run --campaign campaigns/smoke.jsonc --agent vlm --vlm anthropic
    vivid2 run --campaign campaigns/smoke.jsonc --agent vlm --vlm local --vlm-model Qwen/Qwen2.5-VL-7B-Instruct
    vivid2 rescore runs/smoke
    vivid2 report runs/smoke
    vivid2 unreal setup --engine 5.8 --download
    vivid2 unreal setup --engine 5.7 --airsim-root ../ProjectAirSim
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path


def cmd_list(args) -> int:
    from vivid2.scenario import Campaign

    c = Campaign.load(args.campaign)
    runs = c.runs()
    print(f"campaign {c.name}: {len(c.scenarios)} scenarios, {len(runs)} runs (sim_config_dir={c.sim_config_dir})")
    for r in runs:
        print(f"  {r.run_id:40s} seed={r.seed} params={json.dumps(r.params)}")
    return 0


def cmd_run(args) -> int:
    from vivid2.registry import AgentFactory
    from vivid2.report import write_report
    from vivid2.runner import execute_campaign
    from vivid2.scenario import Campaign

    campaign = Campaign.load(args.campaign)
    if args.seeds is not None:
        campaign.seeds = args.seeds
    opts = {k: v for k, v in vars(args).items() if k.startswith(("vlm", "yolo", "sidelap", "frame_period")) and v is not None}
    factory = AgentFactory(args.agent, opts)
    results = asyncio.run(execute_campaign(campaign, factory, Path(args.out), address=args.address,
                                           resume=not args.no_resume, airsim_root=args.airsim_root))
    ok = sum(r["metrics"]["mission"]["success"] for r in results)
    print(f"{factory.name}: {ok}/{len(results)} runs successful")
    print(f"scorecard: {write_report(Path(args.out) / campaign.name)}")
    return 0


def cmd_rescore(args) -> int:
    from vivid2.runner import score_run_dir

    n = 0
    for meta in sorted(Path(args.runs).rglob("meta.json")):
        score_run_dir(meta.parent)
        n += 1
    print(f"rescored {n} runs")
    return 0


def cmd_report(args) -> int:
    from vivid2.report import write_report

    print(write_report(Path(args.runs), Path(args.out) if args.out else None, args.title))
    return 0


def cmd_unreal_setup(args) -> int:
    from vivid2 import unreal_setup as us

    try:
        state = us.setup(
            args.engine,
            plugin_dir=Path(args.plugin_dir) if args.plugin_dir else None,
            airsim_root=Path(args.airsim_root) if args.airsim_root else None,
            plugin_zip=Path(args.plugin_zip) if args.plugin_zip else None,
            do_download=args.download,
            apply_patch=args.apply_patch,
            force=args.force,
        )
    except us.SetupError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"VIVID2Env set up for UE {state['engine']}: {state['uproject']}")
    print(f"  plugins: {state['plugins_dir']} (BuildId {state['plugin_build_id']}, engine {state['engine_build_id']})")
    return 0


def cmd_unreal_status(args) -> int:
    from vivid2 import unreal_setup as us

    state = us.status()
    if state is None:
        print("VIVID2Env is not set up; run: vivid2 unreal setup --engine 5.8 ...")
        return 1
    print(json.dumps(state, indent=2))
    return 0


def cmd_unreal_pack(args) -> int:
    from vivid2 import unreal_setup as us

    info = us.pack_plugin(Path(args.plugin_dir), args.engine, Path(args.out), args.strip_pdb, progress=print)
    print(f"wrote {info['zip']} ({info['files']} files, {info['size'] / 1e9:.2f} GB)")
    print("after uploading, put this in unreal/engines.json -> engines -> "
          f"\"{args.engine}\" -> plugin_zip (set url and provider):")
    print(json.dumps({"url": "<share link>", "provider": "gdrive", "sha256": info["sha256"],
                      "size": info["size"], "file_name": info["file_name"]}, indent=2))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="vivid2", description="VIVID 2 - Virtual Validation for AI Drones")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("list-scenarios", help="expand a campaign into runs")
    s.add_argument("campaign")
    s.set_defaults(fn=cmd_list)

    s = sub.add_parser("run", help="run a campaign against a live Project AirSim server")
    s.add_argument("--campaign", required=True)
    s.add_argument("--agent", required=True, choices=["coverage_gt", "coverage_yolo", "vlm", "hybrid"])
    s.add_argument("--out", default="runs")
    s.add_argument("--address", default="127.0.0.1")
    s.add_argument("--seeds", type=int, help="override seed count")
    s.add_argument("--no-resume", action="store_true")
    s.add_argument("--airsim-root", default=str(Path(__file__).resolve().parents[3] / "ProjectAirSim"),
                   help="ProjectAirSim checkout, used to stamp the build commit into run metadata")
    s.add_argument("--vlm", choices=["anthropic", "local", "replay"])
    s.add_argument("--vlm-model")
    s.add_argument("--vlm-base-url")
    s.add_argument("--vlm-effort", choices=["low", "medium", "high", "xhigh", "max"])
    s.add_argument("--vlm-cache", help="JSONL record/replay cache for VLM calls")
    s.add_argument("--yolo-model", help="ONNX YOLO person detector for coverage_yolo / hybrid")
    s.add_argument("--sidelap", type=float)
    s.add_argument("--frame-period-s", dest="frame_period_s", type=float)
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("rescore", help="recompute metrics/scores from recorded logs")
    s.add_argument("runs")
    s.set_defaults(fn=cmd_rescore)

    s = sub.add_parser("report", help="write the HTML scorecard")
    s.add_argument("runs")
    s.add_argument("--out")
    s.add_argument("--title")
    s.set_defaults(fn=cmd_report)

    s = sub.add_parser("unreal", help="set up unreal/VIVID2Env for UE 5.7 or 5.8")
    usub = s.add_subparsers(dest="unreal_cmd", required=True)
    u = usub.add_parser("setup", help="link the Project AirSim plugin and render VIVID2Env.uproject")
    u.add_argument("--engine", required=True, choices=["5.7", "5.8"])
    src = u.add_mutually_exclusive_group(required=True)
    src.add_argument("--plugin-dir", help="already-built Plugins folder containing ProjectAirSim/ and Drone/")
    src.add_argument("--airsim-root", help="Project AirSim checkout (uses unreal/Blocks/Plugins)")
    src.add_argument("--plugin-zip", help="prebuilt plugin zip downloaded manually")
    src.add_argument("--download", action="store_true", help="download the prebuilt plugin zip from unreal/engines.json")
    u.add_argument("--apply-patch", action="store_true", help="with --airsim-root: apply the UE 5.8 patch if needed")
    u.add_argument("--force", action="store_true", help="replace real (non-link) plugin folders in VIVID2Env")
    u.set_defaults(fn=cmd_unreal_setup)
    u = usub.add_parser("status", help="show the current VIVID2Env engine/plugin setup")
    u.set_defaults(fn=cmd_unreal_status)
    u = usub.add_parser("pack-plugin", help="zip a built plugin for distribution (maintainers)")
    u.add_argument("--plugin-dir", required=True)
    u.add_argument("--engine", required=True, choices=["5.7", "5.8"])
    u.add_argument("--out", default="dist")
    u.add_argument("--strip-pdb", action="store_true", help="omit debug symbols (*.pdb) to shrink the zip")
    u.set_defaults(fn=cmd_unreal_pack)

    args = p.parse_args(argv)
    if getattr(args, "vlm", None) is None and getattr(args, "agent", None) in ("vlm", "hybrid"):
        args.vlm = "anthropic"
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
