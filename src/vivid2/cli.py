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
    vivid2 damage build --map Landslide
    vivid2 fly --map ForestSAR --drone fpv5
    vivid2 fly --scenario scenarios/forest_sar/forest_clearing_01.jsonc --drone quad --launch
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shlex
import subprocess
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
            if k.startswith(("vlm", "yolo", "sidelap", "frame_period", "verify", "dip", "team", "look_around"))
            and v is not None}
    try:
        factory = AgentFactory(args.agent, opts)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    teams = sorted({r.scenario_id for r in campaign.runs() if r.fleet_size > 1})
    if teams and factory.team is None:
        print(f"error: {', '.join(teams)} fly teams of drones (fleet.size > 1): add --team central (one agent "
              "commands every drone) or --team dist (one agent per drone, radio only)", file=sys.stderr)
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


def _add_yolo_args(s) -> None:
    s.add_argument("--yolo-model", help="person detector for coverage_yolo / hybrid: a name (default yolox-s, "
                                        "downloaded on first use) or a path to an ONNX file (YOLOX or YOLOv8 format)")
    s.add_argument("--yolo-threshold", type=float,
                   help="detection score that counts as a find (default: the named detector's calibrated value)")


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
    if args.agent == "coverage_yolo":
        from vivid2.registry import AgentFactory
        from vivid2.stack.reference.reasoners import DetectorReasoner

        opts = {k: v for k, v in vars(args).items() if k.startswith("yolo") and v is not None}
        try:
            factory = AgentFactory("coverage_yolo", opts)
            detector = factory._yolo()
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        reasoner = DetectorReasoner(detector, factory.yolo_threshold)
        name = factory.name
        desc = {"agent": name, "mode": "open_loop", "reasoner": reasoner.name, "threshold": factory.yolo_threshold,
                "perception": {k: v for k, v in vars(detector).items()
                               if not k.startswith("_") and isinstance(v, (bool, int, float, str))}}
    else:
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
            copy_simlibs_from=Path(args.copy_simlibs_from) if args.copy_simlibs_from else None,
            pilot_plugin=Path(args.pilot_plugin) if args.pilot_plugin else None,
            no_pilot_plugin=args.no_pilot_plugin,
        )
    except us.SetupError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"VIVID2Env set up for UE {state['engine']}: {state['uproject']}")
    print(f"  plugins: {state['plugins_dir']} (BuildId {state['plugin_build_id']}, engine {state['engine_build_id']})")
    print(f"  manual flight: pilot hooks {'yes' if state['pilot_hooks'] else 'no'}, "
          f"pilot plugin {state['pilot_plugin'] or 'not linked'}")
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


def cmd_fly(args) -> int:
    from vivid2 import pilot
    from vivid2 import unreal_setup as us

    repo = Path(__file__).resolve().parents[2]
    sim_config_dir = str(Path(args.sim_config_dir) if args.sim_config_dir else repo / "sim_config")
    try:
        if args.scenario:
            run = pilot.scenario_run(args.scenario, args.seed, args.map)
        elif args.map:
            run = pilot.map_run(args.map)
        else:
            print("error: give --map or --scenario", file=sys.stderr)
            return 2
        scene, info = pilot.pilot_scene(run, sim_config_dir, args.drone, with_targets=not args.no_targets)
    except (ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    map_name = info["map"]
    print(f"{info['drone']} on {map_name} ({info['scenario']}{', ' + json.dumps(info['params']) if info['params'] else ''}): "
          f"start {info['start_ned']} NED, {info['altitude_m']} m above sea level, air density x{info['air_density_ratio']}")
    if info["targets"] or info["distractors"] or info["fires"]:
        print(f"  targets {info['targets']}, decoys {info['distractors']}, fires {info['fires']}")
    if args.print_scene:
        print(json.dumps(scene, indent=2))
        return 0

    proc = None
    if args.launch:
        state = us.status()
        try:
            if state is None:
                raise RuntimeError("VIVID2Env is not set up; run vivid2 unreal setup ...")
            cmd = pilot.launch_command(state, map_name)
        except RuntimeError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        if os.name == "nt":
            # Unreal's own quoting (-ExecCmds="a b", Windows paths): pass the extra arguments verbatim.
            cmd = subprocess.list2cmdline(cmd) + (" " + args.unreal_args if args.unreal_args else "")
            print("launching:", cmd)
        else:
            cmd += shlex.split(args.unreal_args or "")
            print("launching:", " ".join(cmd))
        # MSYS_NO_PATHCONV keeps a Git Bash parent from rewriting /Game/... into a file path.
        proc = subprocess.Popen(cmd, env={**os.environ, "MSYS_NO_PATHCONV": "1"})
        try:
            pilot.wait_for_sim(args.address, args.timeout, proc)
        except (RuntimeError, TimeoutError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
    try:
        pilot.load_into_sim(scene, map_name, run.spec.get("environment", {}), sim_config_dir, args.address)
    except Exception as e:  # noqa: BLE001 - connection, preflight and sim errors all end the command
        print(f"error: {e}", file=sys.stderr)
        print("  Is the map open and playing (editor Play, or --launch), with the pilot plugin set up "
              "(vivid2 unreal status)?", file=sys.stderr)
        return 1
    print(pilot.KEYS)
    if args.drone == "quad":
        print(pilot.KEYS_GIMBAL)
    if proc is not None and args.wait:
        return proc.wait()
    return 0


def cmd_terrain_build(args) -> int:
    from vivid2 import dem as dm

    repo = Path(__file__).resolve().parents[2]
    sim_cfg = Path(args.sim_config_dir or repo / "sim_config")
    unreal_dir = Path(args.unreal_dir or repo / "unreal" / "VIVID2Env" / "SourceData")
    layout = json.loads(Path(args.layout).read_text(encoding="utf-8")) if args.layout else None
    like = args.like or ((layout or {}).get("terrain") or {}).get("like")
    pin = None
    if like:
        pin = dm.pin_of(json.loads((sim_cfg / "maps" / like / "terrain.json").read_text(encoding="utf-8")))
        if args.lat is None and args.lon is None:
            args.lat, args.lon = pin["lat"], pin["lon"]
        if (args.lat, args.lon) != (pin["lat"], pin["lon"]) or [args.vertices] * 2 != pin["resolution"] \
                or args.cell_m != pin["cell_m"]:
            print(f"error: --like {like} needs its home point {pin['lat']}, {pin['lon']} and grid "
                  f"{pin['resolution'][0]} x {pin['cell_m']:g} m", file=sys.stderr)
            return 1
    if args.lat is None or args.lon is None:
        print("error: give --lat and --lon (or --like MAP)", file=sys.stderr)
        return 1
    if args.stage and layout is None:
        print("error: --stage needs --layout", file=sys.stderr)
        return 1
    half_m = (args.vertices - 1) / 2 * args.cell_m
    if args.dem == "copernicus":
        tiles = dm.fetch_copernicus_area(args.lat, args.lon, half_m, Path(args.cache_dir).expanduser())
        present = [(t, pth) for t, pth in tiles if pth is not None]
        if len(tiles) == 1 and present:  # one tile, as before
            path = present[0][1]
            info = {"name": "Copernicus GLO-30 DSM", "file": path.name, "url": dm.COPERNICUS_URL.format(name=path.stem),
                    "attribution": dm.COPERNICUS_ATTRIBUTION, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            source = dm.read_geotiff(path, info["name"])
        else:
            info = {"name": "Copernicus GLO-30 DSM", "attribution": dm.COPERNICUS_ATTRIBUTION,
                    "tiles": [{"tile": list(t), "file": pth.name if pth else None,
                               "sha256": hashlib.sha256(pth.read_bytes()).hexdigest() if pth else None} for t, pth in tiles]}
            try:
                source = dm.mosaic([(t, dm.read_geotiff(pth, pth.name) if pth else None) for t, pth in tiles])
            except ValueError as e:
                print(f"error: {e}", file=sys.stderr)
                return 1
    else:
        path = Path(args.dem)
        info = {"name": args.dem_name or path.name, "file": path.name, "attribution": args.attribution or "",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        source = dm.read_geotiff(path, info["name"])
    sea = None
    if args.sea == "fill":
        source, sea = dm.fill_sea(source)
        info["sea"] = "filled: no data or ground <= 0.5 m joined to it or the edge -> shelf 5 % to -30 m"

    def edit_for(stage):
        if not stage:
            return None
        from vivid2.terrain_edit import apply_edits

        return lambda n, e, z: apply_edits(layout, stage, n, e, z)

    try:
        build = dm.build_terrain(source, args.lat, args.lon, vertices=args.vertices, cell_m=args.cell_m,
                                 margin_m=args.margin_m, pin=pin, edit=edit_for(args.stage), sea=sea)
        pre = None
        if args.stage == "post":  # the same map before the event: what the stack's DEM shows
            pre = dm.build_terrain(source, args.lat, args.lon, vertices=args.vertices, cell_m=args.cell_m,
                                   margin_m=args.margin_m, pin=pin or dm.pin_of({"landscape": build.settings}),
                                   edit=edit_for("pre"))
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    if like:
        info["like"] = like
    if args.stage:
        info["layout"] = {"file": str(Path(args.layout)).replace("\\", "/"), "stage": args.stage,
                          "sha256": hashlib.sha256(Path(args.layout).read_bytes()).hexdigest(), "edits": build.edits}
    out = dm.write_outputs(build, args.map, sim_cfg, unreal_dir, info)
    if build.land is not None:
        import numpy as np

        np.savez_compressed(out["terrain"].parent / "landsea.npz", land=build.land)
        print(f"land/sea mask: {out['terrain'].parent / 'landsea.npz'} ({100.0 * build.land.mean():.0f} % land)")
    if pre is not None:
        pre.terrain.save(out["terrain"].parent / "pre_event_dtm.npz")
        print(f"pre-event terrain: {out['terrain'].parent / 'pre_event_dtm.npz'}")
    s = build.settings
    print(f"terrain for {args.map}: {out['terrain']}")
    print(f"heightmap: {out['heightmap']}")
    print(f"ground: Unreal Z {s['ground_ue_z_m']['min']:.1f}..{s['ground_ue_z_m']['max']:.1f} m "
          f"(origin {s['ground_ue_z_m']['origin']:.1f} m); step {s['height_step_cm']:.2f} cm")
    if "headroom_m" in s:
        print(f"pinned to {like}: headroom {s['headroom_m']['below']:.1f} m below, {s['headroom_m']['above']:.1f} m above")
    for ed in build.edits:
        print(f"edit {ed['op']}: {ed['changed_vertices']} vertices, cut {ed['cut_m3']:.0f} m3, fill {ed['fill_m3']:.0f} m3"
              + (f", within {ed['bbox_ne']}" if "bbox_ne" in ed else ""))
    print("Landscape mode -> Manage -> Import from File:")
    print(f"  resolution {s['resolution'][0]}x{s['resolution'][1]}, section size {s['section_size_quads']}x"
          f"{s['section_size_quads']} quads, {s['sections_per_component']}x{s['sections_per_component']} sections/"
          f"component, {s['components'][0]}x{s['components'][1]} components")
    print(f"  location (centre) {s['landscape_centre_cm']} cm, scale {s['scale']}")
    print(f"scene home-geo-point: {json.dumps(s['home_geo_point'])}")
    return 0


def cmd_damage_build(args) -> int:
    """A map's layout -> its damage ground truth, surface model, Unreal masks and placements."""
    from vivid2.damage.gt import gt_paths, sha256_file
    from vivid2.damage.layout import build_damage, write_build
    from vivid2.terrain import TerrainModel, terrain_file

    repo = Path(__file__).resolve().parents[2]
    sim_cfg = Path(args.sim_config_dir or repo / "sim_config")
    layout_path = Path(args.layout or repo / "unreal" / "VIVID2Env" / "SourceData" / args.map / "layout.json")
    if not layout_path.exists():
        print(f"error: no layout {layout_path}", file=sys.stderr)
        return 1
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    paths = gt_paths(sim_cfg, args.map)
    if paths["json"].exists() and not args.force:
        print(f"error: {paths['json']} exists; runs may have been scored against it. Rebuild with --force only if "
              "no run uses it (rescoring checks its checksum).", file=sys.stderr)
        return 1
    tfile = terrain_file(sim_cfg, args.map)
    post = TerrainModel.load(tfile) if tfile else None
    pre = TerrainModel.load(paths["pre_event"]) if paths["pre_event"].exists() else None
    info = paths["dir"] / "terrain.json"
    home_alt = None
    if info.exists():
        home_alt = (json.loads(info.read_text(encoding="utf-8")).get("landscape") or {}).get("home_geo_point", {}).get("altitude")
    landsea_p = paths["dir"] / "landsea.npz"
    if landsea_p.exists():
        import numpy as np

        landsea = np.load(landsea_p)["land"]
    else:
        landsea = None
    cat_p = layout_path.parent / "asset_catalog.json"
    catalog = json.loads(cat_p.read_text(encoding="utf-8")) if cat_p.exists() else {}
    try:
        build = build_damage(layout, post, pre, catalog, home_alt, landsea)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    if build.problems:
        for prob in build.problems:
            print(f"error: {prob}", file=sys.stderr)
        return 1
    shas = {**({"post_sha256": sha256_file(tfile)} if tfile else {}),
            **({"pre_sha256": sha256_file(paths["pre_event"])} if pre is not None else {})}
    out = write_build(build, layout_path, paths["dir"], shas)
    cell = build.grid[2]
    for name, ac in build.doc["area_classes"].items():
        n = int(((build.classes >> ac["bit"]) & 1).sum())
        print(f"area {name}: {n * cell * cell:.0f} m2")
    roads = build.doc["roads"]
    print(f"roads: {len(roads)} segments, blocked {[r['id'] for r in roads if r['blocked']]}")
    grades = {}
    for b in build.doc["buildings"]:
        grades[b["grade"]] = grades.get(b["grade"], 0) + 1
    print(f"buildings: {grades or 'none'}")
    for k, v in out.items():
        print(f"wrote {k}: {v}")
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
    _add_yolo_args(s)
    s.add_argument("--sidelap", type=float)
    s.add_argument("--frame-period-s", dest="frame_period_s", type=float)
    s.add_argument("--frame-bank", choices=["off", "rgb", "rgbd"], default="rgb",
                   help="save each frame the agent fetches (JPEG at camera resolution; rgbd adds depth)")
    s.add_argument("--verify", choices=["on", "off"],
                   help="closer look before reporting: fly over a candidate, descend, confirm (default: on for "
                        "vlm/hybrid, off for the coverage baselines)")
    s.add_argument("--dip", choices=["on", "off"],
                   help="low flight: after the survey, descend into the clearings mapped from the depth image "
                        "and look from about 12 m above the ground (default: off)")
    s.add_argument("--look-around", dest="look_around", choices=["on", "off"],
                   help="with --dip on: at the bottom of each dip, tilt the search camera 20 degrees below the "
                        "horizon and pan it all round in 8 steps (default: off)")
    s.add_argument("--team", choices=["central", "dist"],
                   help="fly scenarios with fleet.size > 1 as a team: central (one agent commands every drone) or "
                        "dist (one agent per drone; they share only what they send over the scenario's radio)")
    s.add_argument("--view", action="store_true",
                   help="live windows: a preview of the payload camera from takeoff (not scored), then the payload "
                        "frames the agent fetches, with ground truth and its decisions; and the chase camera")
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("eval-frames", help="evaluate a VLM (or the YOLO detector) offline on the saved frames of recorded runs")
    s.add_argument("runs", nargs="+", help="run, agent or campaign directories containing frame banks")
    s.add_argument("--agent", choices=["vlm", "coverage_yolo"], default="vlm",
                   help="vlm (default): a VLM looks at each frame; coverage_yolo: the person detector and its threshold")
    _add_vlm_args(s)
    _add_yolo_args(s)
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
    u.add_argument("--apply-patch", action="store_true",
                   help="with --airsim-root: apply the UE 5.8, team weather and pilot hooks patches if needed")
    u.add_argument("--force", action="store_true", help="replace real (non-link) plugin folders in VIVID2Env")
    u.add_argument("--copy-simlibs-from", help="with --airsim-root: copy this prebuilt SimLibs folder into the "
                                               "checkout's plugin if it has none")
    pil = u.add_mutually_exclusive_group()
    pil.add_argument("--pilot-plugin", help="link and enable the pilot plugin (folder with FpvPilot.uplugin) for "
                                            "manual flight (vivid2 fly); kept on later setups")
    pil.add_argument("--no-pilot-plugin", action="store_true", help="unlink the pilot plugin")
    u.set_defaults(fn=cmd_unreal_setup)
    u = usub.add_parser("status", help="show the current VIVID2Env engine/plugin setup")
    u.set_defaults(fn=cmd_unreal_status)
    u = usub.add_parser("pack-plugin", help="zip a built plugin for distribution (maintainers)")
    u.add_argument("--plugin-dir", required=True)
    u.add_argument("--engine", required=True, choices=["5.7", "5.8"])
    u.add_argument("--out", default="dist")
    u.add_argument("--strip-pdb", action="store_true", help="omit debug symbols (*.pdb) to shrink the zip")
    u.set_defaults(fn=cmd_unreal_pack)

    s = sub.add_parser("fly", help="fly a map by hand with a radio or gamepad (needs the pilot plugin; "
                                   "docs/manual_flight.md)")
    s.add_argument("--map", help="map to fly (with --scenario: fly the scenario on this map instead)")
    s.add_argument("--scenario", help="scenario file: its start, hikers, decoys, fires and weather")
    s.add_argument("--seed", type=int, default=0, help="the scenario's seed (target jitter)")
    s.add_argument("--drone", choices=["fpv5", "quad"], default="fpv5",
                   help="fpv5: 5-inch FPV quad (acro); quad: VIVID 2's benchmark quad (angle mode, tilting "
                        "search camera)")
    s.add_argument("--no-targets", action="store_true", help="leave the scenario's hikers and decoys out")
    s.add_argument("--launch", action="store_true", help="start the map in a game window first (else load into "
                                                         "the editor's Play session or a running game)")
    s.add_argument("--wait", action="store_true", help="with --launch: wait for the game to exit, return its code")
    s.add_argument("--unreal-args", help="extra Unreal command-line arguments for --launch (write it with '=': "
                                         "--unreal-args=\"-FpvPilotFlightTest -FpvPilotQuitAfterTest\")")
    s.add_argument("--timeout", type=float, default=900.0, help="seconds to wait for a launched game's sim")
    s.add_argument("--address", default="127.0.0.1")
    s.add_argument("--sim-config-dir", help="default: sim_config")
    s.add_argument("--print-scene", action="store_true", help="print the scene JSON and exit (no sim)")
    s.set_defaults(fn=cmd_fly)

    s = sub.add_parser("terrain", help="build a map's terrain from a real elevation model")
    tsub = s.add_subparsers(dest="terrain_cmd", required=True)
    t = tsub.add_parser("build", help="DEM -> Unreal Landscape heightmap + VIVID 2 terrain model")
    t.add_argument("--map", required=True, help="map name, e.g. ForestSAR")
    t.add_argument("--lat", type=float, help="home point latitude (map centre, Unreal origin)")
    t.add_argument("--lon", type=float, help="home point longitude")
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
    t.add_argument("--like", help="pin the height datum and scale to this map (a variant of it: same home point)")
    t.add_argument("--layout", help="a map layout.json whose terrain operations to apply (vivid2.terrain_edit)")
    t.add_argument("--stage", choices=["pre", "post"],
                   help="with --layout: the map before the event (pre) or after it (post; also writes pre_event_dtm.npz)")
    t.add_argument("--sea", choices=["error", "fill"], default="error",
                   help="no data / sea in the DEM: an error (default), or a synthetic seabed plus landsea.npz")
    t.set_defaults(fn=cmd_terrain_build)

    s = sub.add_parser("damage", help="build a map's damage ground truth from its layout (docs/damage.md)")
    dsub = s.add_subparsers(dest="damage_cmd", required=True)
    d = dsub.add_parser("build", help="layout.json -> damage ground truth, surface model, masks, placements")
    d.add_argument("--map", required=True, help="map name, e.g. Landslide")
    d.add_argument("--layout", help="default: unreal/VIVID2Env/SourceData/<map>/layout.json")
    d.add_argument("--sim-config-dir", help="default: sim_config")
    d.add_argument("--force", action="store_true", help="overwrite an existing ground truth (only if no run uses it)")
    d.set_defaults(fn=cmd_damage_build)

    args = p.parse_args(argv)
    if getattr(args, "vlm", None) is None and (getattr(args, "agent", None) in ("vlm", "hybrid") or args.cmd == "eval-frames"):
        args.vlm = "anthropic"
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
