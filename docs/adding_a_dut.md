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
- `await ctx.get_frame()`: the latest payload-camera frame with its pose and
  depth, or `None` if no image is ready yet.
- `ctx.controller`: `takeoff`, `fly_path` (returns immediately; poll
  `path_done()`), `goto`, `hover`, `return_and_land`.
- `ctx.pose()`, `ctx.elapsed_s()`, `ctx.should_stop()`: drone pose, mission
  time, and whether the time limit is up.
- `ctx.report_target(TargetReport(...))`: a found person, as a NED position.
- `ctx.log("detections" | "decisions", {...})`: lets VIVID 2 score your
  perception and reasoning layers. The record formats are listed in
  `vivid2/recorder.py`.

To run it from the CLI, add its name to `AGENTS` in `vivid2/registry.py` and
build it in `AgentFactory.__call__`; the CLI's `--agent` choices come from that
list. From Python, pass any callable that takes a `Run` and returns an `Agent`
to `vivid2.runner.execute_campaign`. Put every setting that changes behaviour in
`Agent.describe()`: resume compares it to decide whether a run must be redone.

Rules: do not read `frame.gt`, and do not query simulator ground truth APIs
(`get_ground_truth_*`, object poses). Published results will use hidden
holdout scenarios (Phase 3).
