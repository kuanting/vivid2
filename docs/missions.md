# Mission catalog

## Forest S&R: missing hiker (`mission: forest_sar`)

The drone gets a text brief and a search polygon. It must find every missing
person and report each one's position, then return home.

- **Success:** every target reported within `success.localization_tolerance_m`.
- **Mission score:** for each target, 1 − 0.5 × (time-to-find / timeout) if
  found, otherwise 0. The mean over targets is multiplied by 100, then 10
  points are subtracted per false alarm.
- **Variation axes:** target jitter (seeded), fog, time of day, wind, canopy
  density (`ForestSAR_Open` / `ForestSAR` / `ForestSAR_Dense`) and decoys.

| Scenario | Map | Targets | Sweep |
|---|---|---|---|
| `blocks_hiker_01` | Blocks | 1 proxy | 3 seeds |
| `blocks_hiker_02` | Blocks | 2 proxies | fog × time × 2 seeds |
| `forest_hiker_01` | ForestSAR | 1 hiker, red jacket, lying; 4 decoys | 5 seeds |
| `forest_hiker_02` | ForestSAR | 2 hikers at dusk; 5 decoys | fog 0 / 0.02 / 0.05 × 3 seeds |
| `forest_fog_01` | ForestSAR | 1 hiker in dense fog at midday; 3 decoys | fog 0.2 / 0.4 × 3 seeds |

### Challenge: fly low in the woods

Dense fog is not a defect of the benchmark; it is the case it is built for.
Real searches happen in fog, and a drone surveying from above the canopy
cannot see the ground. Finding people then means flying low, among the trees.
VIVID 2 is set up to reward stacks that can do that autonomously:

- **The fog is real.** At dusk the fog lies on the ground. With fog 0.02 or 0.05
  only the treetops show from 60 m, while from 20 m a hiker is visible and from
  10 m clear. At midday fog 0.4 (the densest the simulator makes) leaves the
  ground barely visible from 60 m.
- **The trees are solid.** Every tree has collision for its trunk and crown (24
  convex hulls per tree mesh). Below the crowns, about 4–6 m up, and between
  the trunks there is open space. Rocks, stumps and fallen logs collide too;
  grass, ferns and shrubs do not. Crowns reach about 30 m.
- **Flying low is allowed.** Forest scenarios set `min_agl_m` to 2–3 m, a floor
  against hitting the ground, not a survey height. `search_area.altitude_m` is
  only a suggestion: the reference agents fly it and mostly fail in dense fog.
- **Hitting things is expensive.** Every tree, rock or log strike counts as a
  collision in the safety score, which carries 20 % of the overall weight.
- **The sensors are there.** Besides the nadir payload camera, the drone has a
  forward `Front` camera (RGB + depth, 640 × 480, 90° FOV, pitched 10° down;
  `await ctx.get_nav_frame()`), an IMU, GPS, a barometer and a magnetometer.
  The stack gets a coarse terrain model, never the tree positions.

Things a capable stack might do: plan a terrain-following route through
clearings, avoid trunks and crowns with the forward depth camera, drop below
the canopy to inspect a spot, and weigh time and energy against coverage.

**The reference low-flight baseline** (`--dip on`, any reference agent) does the
simplest safe version of this, to show the fog scenarios can be solved:

1. During the high survey it maps the highest surface in every 2 m cell from
   the nadir depth image. Depth is not affected by fog. It stores this as
   height above the coarse terrain model (`ctx.mission.terrain`).
2. After the survey it flies to the clearings it found, nearest first, while
   time and energy allow, skipping the launch clearing.
3. It descends vertically into each to about 12 m above the ground. It goes
   down in 10 m steps, re-checking the clearing with each closer depth image.
4. It looks three times from the bottom, and reports only what a majority of
   those looks agree on. Then it climbs back.
5. A guard climbs when the mapped canopy or ground ahead comes within 10 m of
   the flight level.

It never flies under the canopy or between trunks. People under the crowns,
and spots too narrow to descend into vertically, are left for capable stacks.
That is the headroom the benchmark measures.

**The oracle sees through fog.** `coverage_gt` uses segmentation ground truth,
which is geometry only, so in fog it is an upper bound on planning, not on
seeing. Compare stacks in fog against `vlm`, `coverage_yolo` and their `--dip`
variants.

## Wildfire S&R (`mission: wildfire_sar`), Phase 2

Find people trapped ahead of a moving fire front while staying outside the
fire's temporary flight restriction (TFR) and above the smoke plume.
Additional metrics: fire-front mapping IoU, minimum clearance to the fire, and
TFR breach time.

## Landslide detection (`mission: landslide`), Phase 2

Survey a mountain road after a slide. Report the debris extent, whether the
road is blocked, and damaged structures. Additional metrics: change-mask IoU,
road-blockage recall, and a rubric score for the VLM's situation report.

## Writing a scenario

See `src/vivid2/schema/scenario.schema.json`. Any field can be swept with a
dotted path, for example `"environment.weather.fog": [0, 0.3]`. `seeds` sets the
number of seeded repetitions.

Set `scene.map` to the Unreal map the scenario needs. Campaigns run grouped by
map, and before each run a preflight check confirms that map is the one open
(by its `BP_VIVID2Map_<map>` marker actor, or the actors listed in
`scene.map_check.actors`) and that every target spawned under its own name.
If not, the runs for that map are skipped with a message saying what to fix.
