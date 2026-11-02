"""vivid2 command line.

    vivid2 list-scenarios campaigns/smoke.jsonc
    vivid2 run --campaign campaigns/smoke.jsonc --agent coverage_gt
    vivid2 run --campaign campaigns/smoke.jsonc --agent vlm --vlm anthropic
    vivid2 run --campaign campaigns/smoke.jsonc --agent vlm --vlm local --vlm-model Qwen/Qwen2.5-VL-7B-Instruct
    vivid2 run --campaign campaigns/blocks_v1.jsonc --agent survey --frame-bank rgb
    vivid2 run --campaign campaigns/forest_smoke.jsonc --agent coverage_gt --view
    vivid2 eval-frames runs/blocks_v1/survey --vlm anthropic --vlm-model claude-sonnet-5-5
    vivid2 rescore runs/smoke
    vivid2 report runs/smoke
    vivid2 unreal setup --engine 5.8 --download
    vivid2 unreal setup --engine 5.7 --airsim-root ../ProjectAirSim
    vivid2 terrain build --map ForestSAR --lat 24.2436 --lon 121.2467
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path


def cmd_list(args) -> int:
    from vivid2.scenario import Campaign

    c = Campaign.load(args.campaign)
    runs = c.runs()
    print(f"campaign {c.name}: {len(c.scenarios)} scenarios, {len(runs)} runs (sim_config_dir={c.sim_config_dir})")
    for r in runs:
        print(f"  {r.run_id:40s} map={r.spec['scene'].get('map', '?'):12s} seed={r.seed} params={json.dumps(r.params)}")
    return 0


def cmd_run(args) -> int:
    from vivid2.registry import AgentFactory
    from vivid2.report import write_report
    from vivid2.runner import execute_campaign
    from vivid2.scenario import Campaign

    campaign = Campaign.load(args.campaign)
    if args.seeds is not None:
        campaign.seeds = args.seeds
    opts = {k: v for k, v in vars(args).items()
            if k.startswith(("vlm", "yolo", "sidelap", "frame_period", "verify")) and v is not None}
    try:
        factory = AgentFactory(args.agent, opts)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    results = asyncio.run(execute_campaign(campaign, factory, Path(args.out), address=args.address,
                                           resume=not args.no_resume, airsim_root=args.airsim_root,
                                           frame_bank=args.frame_bank, view=args.view))
    ok = sum(r["metrics"]["mission"]["success"] for r in results)
    print(f"{factory.name}: {ok}/{len(results)} runs successful")
    if not results:
        # every run was skipped (e.g. preflight stopped the map); the reasons were printed above
        print("no runs finished; no scorecard written", file=sys.stderr)
        return 1
    print(f"scorecard: {write_report(Path(args.out) / campaign.name)}")
    return 0


def _add_vlm_args(s) -> None:
    s.add_argument("--vlm", choices=["anthropic", "local"])
    s.add_argument("--vlm-model")
    s.add_argument("--vlm-base-url")
    s.add_argument("--vlm-effort", choices=["low", "medium", "high", "xhigh", "max"])
    s.add_argument("--vlm-point-format", choices=["norm1000", "pixels"],
                   help="local VLMs: how the model writes the target point (Qwen-VL tends to use pixels)")
    s.add_argument("--vlm-cache", help="JSONL cache of VLM answers")
    s.add_argument("--vlm-tiles", default=None,
                   help="send each frame as a grid of full-detail tiles, e.g. 2x2 (default 1x1: the whole frame, "
                        "at most 1568 px per side); tiles cost proportionally more per frame")
    s.add_argument("--vlm-cache-mode", choices=["record", "replay"],
                   help="record (default): call the model and save answers; replay: answer from the cache only")


def cmd_eval_frames(args) -> int:
    from vivid2.openloop import evaluate_runs, find_banked_runs
    from vivid2.registry import make_vlm, vlm_agent_name
    from vivid2.stack.reference.reasoners import VLMReasoner

    runs = find_banked_runs([Path(p) for p in args.runs])
    if not runs:
        print("error: no runs with a frame bank under " + ", ".join(args.runs)
              + " (record one with: vivid2 run ... --agent survey --frame-bank rgb)", file=sys.stderr)
        return 1
    opts = {k: v for k, v in vars(args).items() if k.startswith("vlm") and v is not None}
    try:
        vlm = make_vlm(opts)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    tiles = args.vlm_tiles or "1x1"
    reasoner = VLMReasoner(vlm, tiles=tiles)
    name = vlm_agent_name("vlm", vlm, tiles)
    desc = {"agent": name, "mode": "open_loop", "reasoner": reasoner.name, "reasoner_model": vlm.model,
            "provider": vlm.fingerprint(), "tiles": tiles}
    print(f"{name}: evaluating {len(runs)} runs")
    results = evaluate_runs(runs, reasoner, name, desc, Path(args.out) if args.out else None, args.concurrency)
    decisions = sum(r["metrics"]["reasoning"]["decisions"] for r in results if r["metrics"]["reasoning"])
    errors = sum(r["metrics"]["reasoning"]["errors"] for r in results if r["metrics"]["reasoning"])
    cost = sum(r["metrics"]["reasoning"]["cost_usd"] for r in results if r["metrics"]["reasoning"])
    print(f"{name}: {decisions} frames assessed, {errors} errors, ${cost:.2f}")
    print("scorecard: vivid2 report <campaign dir> (open-loop results are listed in their own table)")
    return 0


def cmd_rescore(args) -> int:
    from vivid2.runner import score_run_dir

    n = 0
    for meta in sorted(Path(args.runs).rglob("meta.json")):
        score_run_dir(meta.parent)  # uses the campaign weights recorded in meta.json
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


def cmd_terrain_build(args) -> int:
    from vivid2 import dem as dm

    repo = Path(__file__).resolve().parents[2]
    if args.dem == "copernicus":
        path = dm.fetch_copernicus(args.lat, args.lon, Path(args.cache_dir).expanduser())
        info = {"name": "Copernicus GLO-30 DSM", "file": path.name, "url": dm.COPERNICUS_URL.format(name=path.stem),
                "attribution": dm.COPERNICUS_ATTRIBUTION}
    else:
        path = Path(args.dem)
        info = {"name": args.dem_name or path.name, "file": path.name, "attribution": args.attribution or ""}
    info["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        build = dm.build_terrain(dm.read_geotiff(path, info["name"]), args.lat, args.lon,
                                 vertices=args.vertices, cell_m=args.cell_m, margin_m=args.margin_m)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    out = dm.write_outputs(build, args.map, Path(args.sim_config_dir or repo / "sim_config"),
                           Path(args.unreal_dir or repo / "unreal" / "VIVID2Env" / "SourceData"), info)
    s = build.settings
    print(f"terrain for {args.map}: {out['terrain']}")
    print(f"heightmap: {out['heightmap']}")
    print(f"ground: Unreal Z {s['ground_ue_z_m']['min']:.1f}..{s['ground_ue_z_m']['max']:.1f} m "
          f"(origin {s['ground_ue_z_m']['origin']:.1f} m); step {s['height_step_cm']:.2f} cm")
    print("Landscape mode -> Manage -> Import from File:")
    print(f"  resolution {s['resolution'][0]}x{s['resolution'][1]}, section size {s['section_size_quads']}x"
          f"{s['section_size_quads']} quads, {s['sections_per_component']}x{s['sections_per_component']} sections/"
          f"component, {s['components'][0]}x{s['components'][1]} components")
    print(f"  location (centre) {s['landscape_centre_cm']} cm, scale {s['scale']}")
    print(f"scene home-geo-point: {json.dumps(s['home_geo_point'])}")
    return 0


def main(argv=None) -> int:
    from vivid2.registry import AGENTS

    p = argparse.ArgumentParser(prog="vivid2", description="VIVID 2 - Virtual Validation for AI Drones")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("list-scenarios", help="expand a campaign into runs")
    s.add_argument("campaign")
    s.set_defaults(fn=cmd_list)

    s = sub.add_parser("run", help="run a campaign against a live Project AirSim server")
    s.add_argument("--campaign", required=True)
    s.add_argument("--agent", required=True, choices=list(AGENTS))
    s.add_argument("--out", default="runs")
    s.add_argument("--address", default="127.0.0.1")
    s.add_argument("--seeds", type=int, help="override seed count")
    s.add_argument("--no-resume", action="store_true")
    s.add_argument("--airsim-root", default=str(Path(__file__).resolve().parents[3] / "ProjectAirSim"),
                   help="ProjectAirSim checkout, used to stamp the build commit into run metadata")
    _add_vlm_args(s)
    s.add_argument("--yolo-model", help="ONNX YOLO person detector for coverage_yolo / hybrid")
    s.add_argument("--sidelap", type=float)
    s.add_argument("--frame-period-s", dest="frame_period_s", type=float)
    s.add_argument("--frame-bank", choices=["off", "rgb", "rgbd"], default="rgb",
                   help="save each frame the agent fetches (JPEG at camera resolution; rgbd adds depth)")
    s.add_argument("--verify", choices=["on", "off"],
                   help="closer look before reporting: fly over a candidate, descend, confirm (default: on for "
                        "vlm/hybrid, off for the coverage baselines)")
    s.add_argument("--view", action="store_true",
                   help="live windows: the payload frames with ground truth and the agent's decisions, and the chase camera")
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("eval-frames", help="evaluate a VLM offline on the saved frames of recorded runs")
    s.add_argument("runs", nargs="+", help="run, agent or campaign directories containing frame banks")
    _add_vlm_args(s)
    s.add_argument("--concurrency", type=int, default=4, help="parallel VLM calls (use 1 for a local model)")
    s.add_argument("--out", help="default: <campaign>/openloop next to the source runs")
    s.set_defaults(fn=cmd_eval_frames)

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

    s = sub.add_parser("terrain", help="build a map's terrain from a real elevation model")
    tsub = s.add_subparsers(dest="terrain_cmd", required=True)
    t = tsub.add_parser("build", help="DEM -> Unreal Landscape heightmap + VIVID 2 terrain model")
    t.add_argument("--map", required=True, help="map name, e.g. ForestSAR")
    t.add_argument("--lat", type=float, required=True, help="home point latitude (map centre, Unreal origin)")
    t.add_argument("--lon", type=float, required=True, help="home point longitude")
    t.add_argument("--dem", default="copernicus",
                   help="GeoTIFF in EPSG:4326, or 'copernicus' to download the GLO-30 tile (default)")
    t.add_argument("--dem-name", help="name recorded for a local DEM")
    t.add_argument("--attribution", help="attribution text recorded for a local DEM")
    t.add_argument("--vertices", type=int, default=1009, help="landscape vertices per side (k*126+1)")
    t.add_argument("--cell-m", type=float, default=1.0, help="landscape grid spacing, m")
    t.add_argument("--margin-m", type=float, default=10.0, help="height of the lowest ground above Unreal Z = 0")
    t.add_argument("--cache-dir", default="~/.cache/vivid2/dem")
    t.add_argument("--sim-config-dir", help="default: sim_config")
    t.add_argument("--unreal-dir", help="default: unreal/VIVID2Env/SourceData")
    t.set_defaults(fn=cmd_terrain_build)

    args = p.parse_args(argv)
    if getattr(args, "vlm", None) is None and (getattr(args, "agent", None) in ("vlm", "hybrid") or args.cmd == "eval-frames"):
        args.vlm = "anthropic"
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
