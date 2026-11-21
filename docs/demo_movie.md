# Demo movie: "The Golden Hour"

A two-minute film (1920 × 1080, 30 fps, silent: add music when editing) that shows what VIVID 2 does by
telling one rescue story: two hikers on the ForestFire map have accidents, a team of four drones
searches and finds them. It has four acts:

1. the accident;
2. the search: team, coverage, the gimballed search camera, ground truth, low flight, the fire no-fly zone;
3. how the AI finds a hiker, step by step on real photos:
   - the AI checks a search photo;
   - not sure, so the drone flies closer;
   - found, with a point and reasons;
   - point + depth → position;
   - the team is told;
   - a detector's false alarm as a counter-example;
4. the scores.
The tools are in `tools/demo_movie/`; everything they produce goes to `runs/_film/<name>/`
(git-ignored), and the Unreal side lives in `/Game/VIVID2/Demo` (git-ignored, never cooked).

## What is real

The drones in the film fly **the logged paths of one real VIVID 2 run** (`scenarios/demo/golden_hour_01.jsonc`,
`campaigns/demo_golden_hour.jsonc`), re-enacted in an Unreal Level Sequence. Every number, box, track,
photo and AI sentence on screen comes from that run or from real model answers:

* map shots: the run's trajectories, photo footprints (coverage), radio packets and reports;
* camera shots: the run's recorded search-camera photos. The ground-truth shot (`capture_quad.py`;
  runs don't keep segmentation images) and the gimbal shot (`capture_gimbal.py`: tilt, pan, zoom
  80° → 8° on a hiker) are live captures with the same scenario and camera;
* the position step: the AI's point + the recorded depth at that pixel + the camera pose, through
  `vivid2.geo.pixel_to_world`, compared with the hiker's true position;
* the counter-example: a false alarm of another stack (`coverage_yolo`) in another run of the same
  scenario (`--contrast-run`);
* green boxes: the simulator's ground truth;
* the AI's words and point: Claude's real answer to a recorded photo, through the benchmark's own
  VLM reasoner (same prompt, schema and image bytes as a live run). A VLM answers with a verdict, a
  confidence, **one point** and its reasons, not a bounding box;
* captions with numbers (time to find, localisation error, collisions, no-fly time) and the
  scorecard: the run's `result.json`.

What is staged: the hikers' walk, fall and wave (animated copies of the benchmark hikers, ending
in their benchmark poses), the story's clock times and names, the cameras, and the drones' paint.
The end card says which stack flew the run.

## Steps

```powershell
# 0. the editor with MCP + Python remote execution (no tracked config is changed)
python tools/demo_movie/film.py editor
python tools/demo_movie/film.py probe              # Movie Render Queue, assets, clips present?

# 1. the real run (Play-In-Editor on ForestFire, with the v2 hikers)
vivid2 run --campaign campaigns\demo_golden_hour.jsonc --agent coverage_gt --team central --verify on --dip on --frame-bank rgbd --out runs\demo
python tools/demo_movie/capture_quad.py            # RGB / depth / segmentation / ground truth, same instant
python tools/demo_movie/capture_gimbal.py          # the search camera tilting, panning, zooming onto a hiker

# 2. plan the story shots from the run, then Claude's assessment of the hiker photos (~$0.10, cached)
python tools/demo_movie/film.py plan --run runs\demo\demo_golden_hour\<agent>\<run>
python tools/demo_movie/film.py claude

# 3. build and render in the editor (stop PIE first)
python tools/demo_movie/film.py ue-assets          # the film's materials
python tools/demo_movie/film.py ue-build           # film level + one Level Sequence per shot
python tools/demo_movie/film.py ue-render --preset preview     # or final (8 temporal samples)

# 4. the cut: data shots, captions, transitions, encode
pip install -e ".[demo]"                           # bundled ffmpeg: H.264 + AAC music
python tools/demo_movie/film.py compose --preset final --contrast-run runs\demo\demo_golden_hour\coverage_yolo-yolox-s-verify-dip-central\<run>
```

`compose` works at any time: shots that are not rendered yet show a labelled placeholder, so the
cut can be watched as an animatic first (`--half` for 960 × 540, `--start/--end` seconds for a part).
Without `imageio-ffmpeg` the MP4 is written by OpenCV (H.264 via Media Foundation) without music.

## How it fits together

| Module | Does |
|---|---|
| `ue_coords.py` | NED ↔ Unreal (`(100N, 100E, −100D)` cm, origin = world origin; ZYX Euler → `FRotator`), lens maths |
| `run_data.py` | a run, de-duplicated and interpolated per drone (`pose_at`), its frames, events, reports |
| `story.py` | the beats chosen from the run: finds, closer looks, a dip, the fire pass, take-off, landing |
| `golden_plan.py`, `planlib.py` | `plan.json`: film actors and per-shot cameras, drone keys (one per film frame), hiker animation |
| `ue/film_build.py` | editor side: film level (benchmark map streamed in, never saved), actors, Level Sequences |
| `ue/film_render.py` | editor side: Movie Render Queue (PIE executor, GameModeBase: Project AirSim does not launch) |
| `ue/film_assets.py` | the film's materials (drone paint, nav-light glow, translucent no-fly volume) |
| `data_shots.py` | map, time-lapse, camera feeds, gimbal feed, sensor quad, detection zoom with the AI's words and point, geolocation, radio, scorecard |
| `edit.py`, `compose.py`, `typo.py`, `encode.py` | the cut (about 2 minutes), captions, transitions, MP4 |
| `ue_exec.py` | runs a Python file in the editor (Python Remote Execution; the MCP sandbox can't `import unreal`) |

The registration check (`film.py probe-plan`): a film camera placed at a recorded photo's pose
renders the same trees, rocks and shadows as the photo, pixel for pixel. That is what makes the
match cuts (render → real photo) in the AI steps work.

The first reference-team run missed Lin when she stood under 9 m burnt snags:
- its closer look had to stay at obstacle + 10 m = 36 m;
- it lost her at the frame's leading edge, because the closer look doesn't use the gimbal;
- it then dismissed the spot, so later clear sightings of her were ignored.

For the film she stands in an open burnt patch. The reference agent's closer look should learn to
point the gimbal at the candidate.

Movie Render Queue is enabled for the editor only (`unreal/engines.json`, `TargetAllowList: Editor`),
so packaged builds are unchanged.
