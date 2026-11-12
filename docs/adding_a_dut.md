# Adding your drone AI stack (DUT)

Implement any subset of the layer interfaces in `vivid2/stack/base.py`, then
compose them. The less you expose, the fewer layers VIVID 2 can score
separately. Mission and safety are always scored.

## Replace one layer

```python
from vivid2.stack.base import Perception, Detection
from vivid2.stack.reference.agents import SearchAgent
from vivid2.stack.reference.planners import LawnmowerPlanner
from vivid2.stack.reference.reasoners import DetectionThresholdReasoner

class MyDetector(Perception):
    name = "acme_detector_v3"
    def detect(self, frame):
        boxes = acme.infer(frame.rgb)            # frame.rgb is HxWx3 BGR uint8
        return [Detection("person", b.xyxy, b.score) for b in boxes]

agent = SearchAgent("acme", LawnmowerPlanner(), DetectionThresholdReasoner(0.5), MyDetector())
```

## Plug in a VLM

Any model behind an OpenAI-compatible endpoint works without writing code:

```
vllm serve Qwen/Qwen2.5-VL-7B-Instruct --port 8001   # port 8000 is taken by the UE 5.8 editor's MCP server
vivid2 run --campaign campaigns/smoke.jsonc --agent vlm --vlm local --vlm-model Qwen/Qwen2.5-VL-7B-Instruct
```

To compare models on identical images, record frames once and evaluate each
model offline; see "Comparing VLMs on identical frames" in the README. For
Ollama, use `--vlm-base-url http://localhost:11434/v1`, and add
`--vlm-point-format pixels` if the model answers with pixel coordinates.

For anything else, subclass `vivid2.vlm.base.VLMProvider` and implement
`query(images, prompt, schema, system) -> (dict, UsageStats)`. Images arrive as
JPEG bytes (or BGR arrays); send them as they are. Override `fingerprint()` to
add every setting that changes the answers, because it keys the record/replay
cache.

## Full custom agent

Subclass `vivid2.stack.base.Agent` and implement `async run(ctx)`. Use:

- `ctx.mission`: brief, search polygon, altitude, start position, the number
  of people to find, and the limits you are scored on: speed, timeout, minimum
  height above ground, geofence margin and energy budget (no ground truth).
- `ctx.mission.terrain`: a coarse bare-ground model, like the public DEM a
  real crew carries: 20 m cells on maps with relief, flat ground otherwise.
  `terrain.down_at(n, e)` gives the NED down of the ground. It has no trees,
  and it is never the 1 m model VIVID 2 scores height above ground with.
- `ctx.mission.camera`: the payload camera's `width`, `height` (pixels) and
  `fov_deg`, for planning the survey before the first frame.
- `ctx.mission.cameras`: every camera you may use, by id.
  - **"Payload"** (`"role": "search"`): 1920 × 1440, 80° field of view, on a gimbal, the only
    camera with ground truth and scored.
  - **"Front"** (`"role": "nav"`): 640 × 480, 90°, looking forward from the drone's body,
    pitched 10° down, for steering among trees.
  - Each entry gives the size, field of view, mount and capture step; the search camera's adds
    its gimbal limits.
- `await ctx.get_frame()`: the next search-camera photo with its pose and
  depth, or `None` if no image is ready yet. Each costs 0.25 s of mission time
  (one capture step), whatever the machine.
  - `await ctx.get_frame("Front")` takes a forward photo (0.15 s);
    `await ctx.get_frames(["Payload", "Front"])` takes both in one step (0.25 s).
  - Every photo is recorded, so take them where you search, not on the ground.
    Perception and reasoning are scored only on search-camera photos you log
    detections or decisions for.
  - `get_nav_frame("Front")` is the same as `get_frame("Front")`.
- `await ctx.point_camera(tilt_deg, pan_deg, fov_deg)`: move the search camera's
  gimbal. Tilt runs from -90 (straight down, the default) to +10, pan to ±180 degrees around the
  drone, and the field of view (zoom) from 80 down to 8 degrees. A move takes mission time:
  60 degrees per second, plus 0.5 s per e-fold of zoom, plus 0.2 s. A photo records where the
  gimbal pointed (`frame.gimbal`).
- `ctx.controller`: `takeoff`, `fly_path` (returns immediately; poll
  `path_done()`), `goto`, `turn_to`, `hover`, `return_and_land`.
  - `fly_path(..., heading="travel")` faces the direction of travel, so the forward camera looks
    where you fly; `heading=<degrees>` holds a heading.
  - `goto(..., yaw_deg=...)` turns on the way, and `turn_to(yaw_deg)` turns on the spot.
  - Headings are degrees: 0 north, 90 east. The default holds the drone's heading.
- `ctx.pose()`, `ctx.elapsed_s()`, `ctx.should_stop()`: drone pose, mission
  time, and whether the time limit is up.
- `await ctx.think(fn, *args)`: run your model (a function, in a worker thread, or a
  coroutine function) and return its answer. Thinking takes mission time; in a team, `think`
  also charges the time the clock stood still for other drones' photos.
- `await ctx.sleep(seconds)`: wait in mission time. Pace your agent with this
  and `ctx.elapsed_s()`, not `asyncio.sleep` or the wall clock. VIVID 2 pauses
  the simulator while it records each frame, so wall time runs ahead of mission
  time by an amount that depends on the machine. Your agent's own thinking
  takes mission time, as on a real drone: the drone keeps flying while your
  model answers.
- `ctx.report_target(TargetReport(...))`: a found person, as a NED position.
- `ctx.log("detections" | "decisions", {...})`: lets VIVID 2 score your
  perception and reasoning layers. The record formats are listed in
  `vivid2/recorder.py`.

To run it from the CLI, add its name to `AGENTS` in `vivid2/registry.py` and
build it in `AgentFactory.__call__`; the CLI's `--agent` choices come from that
list. From Python, pass any callable that takes a `Run` and returns an `Agent`
to `vivid2.runner.execute_campaign`. Put every setting that changes behaviour in
`Agent.describe()`: resume compares it to decide whether a run must be redone.

For a team of drones (a `FleetAgent` commanding every drone, or one agent per drone over a
simulated radio), see [teams.md](teams.md). Run your models through `await ctx.think(fn, ...)`:
in a team it charges the time the clock stood still for other drones' photos.

Rules: see [rules.md](rules.md).
- **Ground truth:** frames reach your stack without it (`frame.gt` is empty).
- **Simulator handles:** don't use the context's private (`_`) attributes, or the simulator
  itself.
- **Citations:** a report must cite one of your drone's recorded search-camera photos
  (`frame_id`); otherwise it counts as a false alarm.
- **Holdout:** published results will use hidden holdout scenarios (Phase 3).
