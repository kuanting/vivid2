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
- **Missing layers are not zero.** If a stack doesn't expose separate
  detections, its perception score is `n/a` and the layer is left out of the
  weighted overall score.
- **Safety gate.** Any collision caps the overall score at 25.

## Coordinate conventions

- World is NED in metres, with the origin at the scene's player start.
- The camera optical frame is x forward, y right, z down, which is the Project
  AirSim convention. Image messages carry the camera's world pose.
- Depth comes as 16-bit millimetres, so it saturates at 65.5 m. Geolocation
  falls back to a flat-ground ray intersection beyond that.

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
