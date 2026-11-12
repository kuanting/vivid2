# Search rules

Every stack in a VIVID 2 competition follows these rules. They keep results comparable: every
stack sees the same sensors, pays the same mission time for them, and is scored on what it can
show it found.

Some rules are checked in code today. Full isolation (the stack in its own process or on its own
hardware, talking to VIVID 2 only through the API) comes with the hardware-in-the-loop tests.
Until then, published results are also reviewed.

## What a stack may use

Everything below comes from the mission context (`ctx`, a `vivid2.stack.base.MissionContext`), or
for a central team from `fleet` and each drone's context.

- **`ctx.mission`:** the brief, the search polygon, the suggested survey height, the start, the
  limits, a coarse terrain model (no trees), and the drone's cameras (`ctx.mission.cameras`).
  Teams also get the team list and pads.
- **Photos:**
  - `await ctx.get_frame(camera)` and `await ctx.get_frames([...])` take photos.
  - The search camera ("Payload") has a gimbal and zoom (`await ctx.point_camera(...)`).
  - A navigation camera ("Front") looks forward from the drone's body.
- **`ctx.pose()`:** the drone's position and attitude, like a navigation solution.
- **`ctx.controller`:** take off, fly a path, go to a point, turn (`turn_to`, a heading for a
  path or a goto), hover, return and land.
- **Time and thinking:** `ctx.elapsed_s()`, `ctx.sleep()`, `ctx.should_stop()`, and
  `await ctx.think(fn, ...)` for every model call.
- **Teams:** `ctx.radio` in a distributed team.
- **`ctx.log(...)`** for your own records, and **`ctx.report_target(...)`** to report a find.

## What a stack may not use

- **Ground truth:** `frame.gt`, which is empty for every stack except the reference oracle.
- **Simulator handles:** any attribute starting with `_` on the context, the controller or the
  fleet. Those are the simulator, the recorder and the runner's own state. Nor may a stack
  open its own simulator connection or call the Project AirSim API itself.
- **Run files:** the run's own logs (`frames.jsonl`, `meta.json`, ...), the scenario files, and
  the target or decoy positions in them.
- **Other drones' data, in a distributed team,** except what arrives over the radio.
- **Any other channel** to information about the scene, such as reading the map's assets.

## Sensing costs mission time, the same on every machine

- **Photos:**
  - Every photo is taken in lockstep. The simulator pauses, captures one camera step, and
    stays paused while VIVID 2 records the photo.
  - The camera step costs mission time: Payload 0.25 s, Front 0.15 s. Photos asked for
    together share one step.
  - Every photo is recorded, from any camera. Photos taken only to steer cost nothing in the
    score; perception and reasoning are scored only on photos the stack logs detections or
    decisions for.
- **The gimbal** (`point_camera(tilt_deg, pan_deg, fov_deg)`):
  - Tilt runs from -90 (straight down) to +10 degrees, pan to ±180 degrees around the drone,
    and the field of view from 80 down to 8 degrees.
  - A move takes `max(|tilt change|, |pan change|) / 60` seconds, plus 0.5 s per e-fold of zoom,
    plus 0.2 s to settle. The drone flies on meanwhile.
- **Turning the drone** takes the time the flight controller takes.
- **Thinking** takes mission time: run models through `ctx.think`. In a team, it also charges
  the time the clock stood still for other drones' photos.

## Reports must cite a recorded photo

A report's `frame_id` must be a photo from the drone's own search camera ("Payload"), recorded in
this run. Otherwise the report is counted as a false alarm, even if it is right, and listed under
`rule_violations` in `result.json`. The violations are:

| Violation | When |
|---|---|
| `uncited_report` | the frame was never recorded |
| `report_cites_nav_frame` | it is a navigation-camera photo, which has no ground truth |
| `report_cites_other_drone` | in a team, another drone's photo |
| `decision_on_unrecorded_frame` | a decision is logged for a frame that was never recorded |

A stack may still use navigation photos to steer, or to decide where to look closer. It then
cites the search-camera photo that confirmed the find.

## Scoring with several cameras

- **Perception and reasoning:** scored on search-camera photos only, the ones with ground truth.
  Decisions on navigation photos are counted as `unscored_decisions`, not as answers.
- **Coverage:**
  - A photo with the gimbal at rest (straight down, full field of view) covers its ground
    footprint.
  - A tilted or zoomed photo covers only the ground where a lying person would show at least the
    visibility threshold of pixels: 80 px at 1920×1440. So looking at the horizon doesn't count
    the far ground, and zooming in counts more distant ground.
- **The mission score** credits every target found by a cited report.

## Compute

- **Today:** a stack's thinking time is measured on the machine that runs it and counted as
  mission time. `meta.json` records the host's CPU and GPU.
- **Later:** per-drone compute budgets (a drone's own processor, and the time it takes to process
  two cameras) come with the hardware-in-the-loop tests.

## The reference oracle

`coverage_gt` reads ground truth: it shows what a perfect detector would achieve with the
reference flight. Its runs are marked `"eligible": false` in `result.json` and "oracle, not
eligible" on the scorecard, listed after the ranked stacks.

## Rules and how they are enforced

| Rule | Enforced |
|---|---|
| No ground truth | In code: frames reach a stack with empty `gt`, except the oracle. Offline evaluation (`eval-frames`) strips it too. |
| Photos cost fixed mission time | In code: lockstep capture for every camera, gimbal moves charged in mission time. |
| Every photo recorded | In code: there is no unrecorded way to take a photo. |
| Reports cite own search photos | In code: uncited reports are false alarms and are listed as violations; a live `rule_violation` event too. |
| No simulator handles | Convention: they are private (`_`). Checked by review now, and by process isolation with hardware-in-the-loop. |
| Radio only, in distributed teams | Review now; isolation later. |
| No scenario files or run logs | Review now; isolation later. |
