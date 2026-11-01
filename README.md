# VIVID 2: Virtual Validation for AI Drones

VIVID 2 is an open benchmark and validation framework for drone AI, built on
[Project AirSim](https://github.com/iamaisim/ProjectAirSim) (Unreal Engine 5.8).
It flies a drone AI stack, the device under test (DUT), through standardised,
reproducible emergency-response missions. It then scores each layer of the
stack separately:

| Layer | What is scored |
|---|---|
| **Perception** | AP@0.5, precision/recall against simulator ground-truth boxes, first-detection range |
| **Reasoning (VLM/VLA)** | Decision accuracy, hallucination rate, grounding accuracy, latency, tokens and cost |
| **Planning** | Search-area coverage, coverage per km flown, revisits |
| **Control** | Path-tracking RMSE, altitude hold, speed |
| **Safety** | Collisions, terrain clearance, geofence/no-fly breaches, energy budget, return home |
| **Mission** | Targets found, time-to-find, localisation error, false alarms |

VIVID 2 succeeds VIVID (2018, built on
Microsoft AirSim, ACM MM Best Open Source Software Award).

## Missions

| Mission | Status |
|---|---|
| Forest S&R: find missing hikers under canopy | **Phase 0**: runs on Blocks (proxy targets); ForestSAR map in progress |
| Wildfire S&R: find trapped people near a dynamic fire front, keep out of the fire TFR | Phase 2 |
| Landslide detection: map debris, report blocked roads | Phase 2 |

## Layout

```
src/vivid2/          Python package
  scenario.py        scenario/campaign spec, sweeps, deterministic run ids
  sim/               scene building for Project AirSim
  stack/             DUT layer interfaces (Perception, Reasoner, Planner, Controller, VLAPolicy, Agent)
  stack/reference/   baseline layers and the reference SearchAgent
  vlm/               VLM providers: Claude API, OpenAI-compatible local servers, record/replay cache
  metrics/           offline per-layer metrics over recorded logs
  scoring.py         0-100 layer scores, safety gate, bootstrap CIs
  report.py          HTML scorecard
  runner.py, cli.py  execute campaigns against a live sim
scenarios/           mission catalog (JSONC)
campaigns/           sets of scenarios to run together
sim_config/          Project AirSim scene/robot configs used by the scenarios
unreal/VIVID2Env/    environment project (maps authored in UE 5.8; UE 5.7 users run packaged builds)
unreal/engines.json  per-engine profiles (UE 5.7 / 5.8) and the prebuilt plugin download
patches/             Project AirSim UE 5.8 compatibility patch
tools/               packaging script
```

## Quick start

```powershell
# 1. Python (Python 3.9+). Install the Project AirSim client from your fork, then VIVID 2.
pip install -e ..\ProjectAirSim\client\python\projectairsim
pip install -e ".[dev,claude]"          # add local-vlm for vLLM/Ollama
pytest                                   # unit tests, no simulator needed

# 2. Simulator. Either run a packaged VIVID2Env build (any user, no Unreal needed),
#    or set up the editor project for your engine (see docs/unreal_versions.md):
pip install -e ".[setup]"                                   # gdown, for the Google Drive download
vivid2 unreal setup --engine 5.8 --download                # prebuilt plugin for UE 5.8 (once published; until then use --plugin-dir)
vivid2 unreal setup --engine 5.8 --airsim-root ..\ProjectAirSim --apply-patch   # or: upgrade your own checkout
vivid2 unreal setup --engine 5.7 --airsim-root ..\ProjectAirSim                 # UE 5.7 (upstream as-is)
#    then open unreal\VIVID2Env\VIVID2Env.uproject and press Play.

# 3. Run a campaign and open the scorecard
vivid2 list-scenarios campaigns\smoke.jsonc
vivid2 run --campaign campaigns\smoke.jsonc --agent coverage_gt          # oracle upper bound
vivid2 run --campaign campaigns\smoke.jsonc --agent vlm --vlm anthropic  # Claude as the reasoner
vivid2 run --campaign campaigns\smoke.jsonc --agent vlm --vlm local --vlm-model Qwen/Qwen2.5-VL-7B-Instruct --vlm-base-url http://localhost:8001/v1
vivid2 report runs\smoke                                                 # -> runs\smoke\scorecard.html
```

Add `--vlm-cache runs\vlm_cache.jsonl` to record VLM answers. A later run with
`--vlm replay --vlm-cache ...` answers from the cache at no cost. Replay only hits
on byte-identical frames, and live re-renders usually differ, so exact offline
reproduction arrives with saved-frame evaluation in Phase 1.

## Reference stacks

| `--agent` | Perception | Reasoner | Purpose |
|---|---|---|---|
| `coverage_gt` | Simulator ground truth (occlusion-checked) | Threshold | Oracle upper bound |
| `coverage_yolo` | ONNX YOLO (`--yolo-model`) | Threshold | Classic CV baseline |
| `vlm` | none | VLM on raw frames | VLM-only baseline |
| `hybrid` | ONNX YOLO | VLM verifies candidates | Detector + VLM |

To plug in your own stack, see [docs/adding_a_dut.md](docs/adding_a_dut.md).

## Roadmap

0. **SIL framework + Forest S&R.** Current phase.
1. ForestSAR map (DEM terrain, Fab forest, PCG hiker placement); 20-scenario benchmark; local VLM vs Claude.
2. Wildfire map with fire-spread model; landslide map with before/after change ground truth.
3. Vendor container interface (ROS 2 / gRPC), parallel headless runners, hidden holdout set, leaderboard.
4. Processor-in-the-loop: PX4 HITL + companion computer running the vendor AI.
5. Rig-in-the-loop: instrumented 3-DOF gimbal rig, rig-coupled physics, then a screen-based sensor-stimulation (VR) rig.

See [docs/architecture.md](docs/architecture.md) for the design.

## License

MIT
