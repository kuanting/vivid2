# VIVID 2 architecture

```
 campaign.jsonc ──► Scenario.expand() ──► Run (params, seed, run_id)
                                              │
                                              ▼
 sim_config/*.jsonc ──► scene_builder ──► World.load_scene(dict)   (Project AirSim)
                                              │   targets spawned, payload camera annotated
                                              ▼
                     ┌──────────── SimMissionContext ────────────┐
                     │ get_frame(): RGB + planar depth + camera   │
                     │              pose + GT boxes (eval only)   │
                     │ report_target(), log(), should_stop()      │
                     └───────┬──────────────────────────▲─────────┘
                             │                          │
                 Agent (DUT) │  Perception → Reasoner → Planner → Controller
                             ▼                          │
                         Recorder ──► runs/<campaign>/<agent>/<run_id>/*.jsonl + meta.json
                                              │
                          metrics (offline) ──► scoring ──► result.json ──► report.html
```

## Design rules

- **Ground truth never reaches the DUT.** `Frame.gt` exists only so the
  recorder can log it. Reference stacks read it only in the explicitly named
  oracle (`GroundTruthPerception`). `MissionSpec` carries only what a real crew
  would give the drone: the brief, the search polygon, altitude and limits.
- **Every score can be recomputed offline.** A run writes raw JSONL streams.
  `vivid2 rescore` regenerates every `result.json` after a metric changes.
- **Reproducible runs.** A run is identified by (scenario, swept params, seed).
  Target jitter comes from the seed. The ProjectAirSim commit hash is stored in
  `meta.json`. VLM answers can be recorded and replayed.
- **Same frames for every model.** The runner JPEG-encodes each camera frame
  once and sends those exact bytes to the VLM. With a frame bank
  (`--frame-bank rgb`), the bytes are saved under `frames/` with their SHA-256,
  and `vivid2 eval-frames` replays them through any model offline (open loop).
  Open-loop results are scored on the reasoning layer only and listed in their
  own scorecard table.
- **Results don't depend on the machine's speed.** The simulator runs in real
  time, and three things take time during a run:
  - **Taking a photo** uses lockstep. The runner pauses the simulator and asks
    for the images. It then lets the simulator run exactly one capture step:
    the camera's `capture-interval` plus 0.05 s, so 0.25 s for the payload
    camera. The photo is taken at the pose where the simulator paused. Unreal
    needs a second or more to read back and send three 1920×1440 images (1.2 s
    on an RTX 5090 laptop), and that happens while the simulator stays paused.
    So every photo costs the same 0.25 s of mission time on any machine.
    `frames.jsonl` records it as `capture_s`.
  - **VIVID 2's own work on each frame** (ground truth, JPEG encoding,
    saving) also happens with the simulator paused, so the drone doesn't fly
    further on a slower machine. It is recorded as `record_wall_s`.
  - The cost is wall time. A run takes longer than its mission time: about
    1.7 s more per frame on that laptop (1.2 s sending the images, 0.5 s
    recording them).
  - If a lockstep frame fails, the run fetches frames with the simulator
    running from then on. That cost then depends on the machine. The run logs
    a `lockstep_unavailable` event, and `meta.json` records it under
    `frame_timing`.
  - **The agent's thinking takes mission time**, as on a real drone: the drone
    flies on, or hovers, while the model answers. Decisions record it as
    `response_s`. A find is timed when the agent reports it, not when its frame
    was taken. Each run's `meta.json` records the machine (`host`) and the
    simulator's address, since a model's speed depends on its hardware or API.
    In closed-loop runs, replayed VLM answers take as long as the original call.

  Agents pace themselves in mission time with `ctx.sleep()` and
  `ctx.elapsed_s()`. The reference agents take a frame every `frame_period_s`
  of mission time. The hard time limit (twice the timeout plus 180 s) is
  mission time as well.
- **Missing layers are not zero.** If a stack doesn't expose separate
  detections, its perception score is `n/a` and the layer is left out of the
  weighted overall score.
- **Safety gate.** Any collision caps the overall score at 25.

## Teams of drones

A scenario's `fleet.size` (up to 4) puts one drone per launch pad in the scene (`Drone1` to
`DroneN`, the same robot config). The run then goes through `runner.execute_team_run` and
`vivid2/team_runtime.py`:

- **One clock, photo batches.** Project AirSim's clock is world-wide: pausing it stops every
  drone. Drones that ask for a photo at the same moment share one lockstep step: one pause, one
  GetImages per drone, one capture step, all photos recorded paused, one resume. A photo still
  costs 0.25 s of mission time.
  - All of a batch's photos are stamped within its step (probe: 0–6 ms after the request).
  - One batch of 4 photos took 1.9 s of wall time, against 1.2 s for one photo
    (`tools/probe_fleet.py`, 2026-11-08).
- **Navigation photos** (fetched running) wait for a batch to finish and hold the next one off,
  so a pause never holds one past Project AirSim's 5 s limit.
- **Fair thinking.** A pause ledger records when the clock stood still; `ctx.think` charges the
  pause that overlapped a model call back in mission time.
- **Every record is tagged** with its `drone`. Frame ids stay unique across the team.
- **Weather.** Project AirSim gives each drone its own weather actor, whose particle fog
  follows the drone. Its plugin must hide each drone's weather from the other drones'
  cameras (`patches/projectairsim-team-weather.patch`, in VIVID2Env builds from
  2026-11-09). Without it every camera saw every drone's fog: with four drones a hiker's
  contrast fell to about half, and the oracle team found fewer people than one drone.
- **Single-drone runs are untouched:** a scenario without `fleet` takes the old path. The
  golden tests (`tests/golden/`) pin its run ids, config hashes, scene and scores, and a team
  run's config hash includes `team_runtime.FLEET_VERSION`.

See [teams.md](teams.md) for the agent API (central `FleetAgent` and distributed
`DistributedTeam` with the radio), scoring and the reference team.

## Coordinate conventions

- World is NED in metres, with the origin at the Unreal world origin (0, 0, 0),
  not at the PlayerStart.
- The camera optical frame is x forward, y right, z down, which is the Project
  AirSim convention. Image messages carry the camera's world pose.
- Depth comes as 16-bit integers labelled millimetres (`16UC1`). The current
  UE 5.8 plugin build actually packs whole metres, so the runner detects the
  unit from the first frame taken above 5 m and records it in `meta.json`
  (`depth_metres_per_unit`). Raw 0 and 65504 mean no data. Where depth is
  missing, geolocation falls back to a flat-ground ray intersection.
- Contacts on the home pad (within 5 m of the start, below 2 m AGL) are
  takeoff/landing contacts (`ground_contacts`), not collisions.
- Terrain: flat maps use `search_area.ground_down_m`. A map with relief ships a
  terrain model (`sim_config/maps/<map>/terrain_dtm.npz`, see `vivid2/terrain.py`).
  Targets and the drone start keep their authored height above ground but sit
  on the real terrain, and safety measures height above it. `meta.json` records
  the file and its hash for rescoring.
- The DUT's `MissionSpec.ground_down_m` is the terrain height under the start,
  not the authored plane. `search_area.altitude_m` is therefore height above
  the start's ground: the reference agents fly at
  `ground_down_m - altitude_m` NED. On ForestSAR the start is 257.7 m up, so the
  authored plane at down 0 would put "55 m up" inside the hill. Coverage is
  scored on the same plane. A terrain-following planner (M3.4) will replace
  this fixed altitude.

## YOLO baseline

`coverage_yolo` flies the same zig-zag as the other reference agents and looks
with an off-the-shelf COCO person detector. The default is YOLOX-s (Apache-2.0),
downloaded to the VIVID 2 cache on first use and checked by SHA-256.

- **Tiled at full resolution.** From 55 m a person is about 15×20 px of a
  1920×1440 frame. Shrunk to the detector's 640 px input, that would be 5×7 px.
  So the frame is cut into 12 overlapping 640 px tiles, plus one pass over the
  whole frame for people seen up close. The boxes are merged with NMS.
- **One threshold, 0.3,** YOLOX's conventional value. Calibrating it on frames
  the benchmark does not use gave no usable operating point, so it is not tuned
  on the benchmark (`tools/calibrate_detector.py`, notes in
  `stack/reference/detectors.py`).
  - Plain: a detection at 0.3 or more is reported.
  - With `--verify on`: 0.15 is enough to go and look, and most close-ups (or
    dip looks) must reach 0.3.
- **Two variants in the benchmark:**
  - `coverage_yolo`, the plain zig-zag;
  - `coverage_yolo --verify on --dip on`, which flies like `vlm --dip`. The
    only difference from Claude is the eyes.
- **Detector time is thinking time** and takes mission time. The execution
  provider (DirectML, CUDA or CPU) is part of the agent's settings. On an RTX
  5090 laptop with DirectML, a frame takes about 120 ms.
- **Offline:** `vivid2 eval-frames <runs> --agent coverage_yolo` scores the
  detector on saved frames, like a VLM.
- **What it sees** (offline, earlier ForestSAR frames, at 0.3):
  - no hiker from 55 m (0 of 104 frames with a visible hiker);
  - 33% of low frames with a visible hiker (18 of 54), precision 0.47.

  Claude found about 90% on comparable low frames. A COCO detector rarely
  recognises people seen from straight above.

## Closer look before reporting (reference agents)

Search crews survey from a safe height, then go and look closer at anything
suspicious before calling it in. With `verify` on (the default for the `vlm`
and `hybrid` agents; `vivid2 run --verify on|off` overrides it), the reference
`SearchAgent` does the same:

1. A *candidate* is a find at confidence 0.3 or more. The agent leaves the
   lawnmower path and flies over it.
2. It descends to 10 m above the highest obstacle around the candidate (from the
   depth image), never below its own floor of 8 m above the ground
   (`LOOK_MIN_AGL_M`; stacks under test may fly down to the scenario's
   `min_agl_m`, 0.5 m), and never above the survey altitude.
3. It takes up to three close-ups. For each one the VLM gets a full-resolution
   960 x 720 crop around the candidate (a digital zoom). It reports only if a
   majority confirm the target at 0.6 or more, located as the mean of those
   close-ups. One noisy "yes" is not enough: in the second Claude run on
   ForestSAR (2026-11-03), a single confirming close-up turned the red-tarp
   decoy into a false alarm.
4. It climbs back and flies the rest of the path. Dismissed spots are not checked
   again, and after 8 detours it falls back to reporting finds at 0.6 directly.

The cost of looking is scored by the existing layers: time to find, path length
and energy. Planning metrics also count `verifications`,
`verifications_confirmed` and `verify_time_s`. Decisions taken during a closer
look carry `"phase": "verify"`.

Agent names separate the behaviours: `vlm-<model>` verifies, while
`vlm-<model>-noverify` and `coverage_gt-verify` are the non-default variants.

## Controllers

`AirSimController` drives the Project AirSim Drone API. It works for both
`simple-flight-api` and `px4-api` robot configs, because Project AirSim
forwards move commands to PX4 over MAVLink. To fly PX4 SITL, use
`robot_vivid_quad_px4_sitl.jsonc` in the scenario's `scene.robot_config` and
start PX4 as described in `ProjectAirSim/docs/controllers/px4/px4_sitl.md`.

## HITL path (Phases 4–5)

1. **Processor-in-the-loop:** PX4 on a real flight controller (`px4-api` HITL
   config) and the vendor AI on a companion computer. VIVID 2 streams camera
   frames to the companion computer and reads its reports.
2. **Rig-in-the-loop:** a real drone on an instrumented 3-DOF gimbal. Rig
   encoders and a load cell feed a rig-coupled physics mode, in which real
   attitude and thrust drive the simulated translation. The simulated world is
   then shown to the drone, either by injecting sensor data or by rendering it
   on a screen in front of the real camera.
