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

For anything else, subclass `vivid2.vlm.base.VLMProvider` and implement
`query(images, prompt, schema, system) -> (dict, UsageStats)`.

## Full custom agent

Subclass `vivid2.stack.base.Agent` and implement `async run(ctx)`. Use:

- `ctx.mission`: brief, search polygon, altitude, limits (no ground truth).
- `await ctx.get_frame()`: the latest payload-camera frame with its pose and depth.
- `ctx.controller`: takeoff, fly_path, goto, hover, return_and_land.
- `ctx.report_target(TargetReport(...))`: a found person, as a NED position.
- `ctx.log("detections" | "decisions", {...})`: lets VIVID 2 score your
  perception and reasoning layers. The record formats are listed in
  `vivid2/recorder.py`.

Register it in `vivid2/registry.py` (or build an `AgentFactory`-like callable)
to run it from the CLI.

Rules: do not read `frame.gt`, and do not query simulator ground truth APIs
(`get_ground_truth_*`, object poses). Hidden holdout scenarios will be used
for published results.
