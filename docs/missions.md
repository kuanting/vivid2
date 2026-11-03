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
| `forest_clearing_01` | ForestSAR | 2 hikers in small clearings at dusk; 3 decoys | fog 0 / 0.02 / 0.05 × 3 seeds |

### Challenge: fly low in the woods

Dense fog is not a defect of the benchmark; it is the case it is built for.
Real searches happen in fog, and a drone surveying from above the canopy
cannot see the ground. Finding people then means flying low, among the trees.
VIVID 2 is set up to reward stacks that can do that autonomously:

- **The fog is real.** At dusk the fog lies on the ground. With fog 0.05 the
  ground is washed out from 55 m and a hiker barely shows, while from 12 m the
  ground is clear. Fog 0.02 is in between. At midday the fog is thinner near
  the ground: even fog 0.4 (the densest the simulator makes) only hazes it from
  55 m, and a red jacket still shows.
- **People blend in.** At dusk a yellow jacket among yellow birch crowns, or a
  blue hoodie beside a fallen log, is hard to pick out even in clear air. From
  55 m a person is only 100–250 pixels of a 1920 × 1440 image.
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

It never flies under the canopy or between trunks. From 12 m its camera sees
about 20 × 15 m of ground, so it only finds people in or right next to a
clearing at least 6 m in radius. People under the crowns, and spots too narrow
to descend into vertically, are left for capable stacks. That is the headroom
the benchmark measures.

On ForestSAR about 37 % of the ground is open, but a dip can see only about
15 % of a search area. On `ForestSAR_Open` it is 73 % and 62 %; on
`ForestSAR_Dense` 4 % of the ground is open and nothing outside the launch
clearing is wide enough to descend into, so the baseline does not dip at all.
`forest_clearing_01` puts its hikers inside clearings a dip reaches; the other
forest scenarios put them under crown edges, out of its reach. To place targets either way, record a survey with `--dip on
--frame-bank rgbd` and run `python tools/dip_reach.py <run dir>`.

**Visible means visible in the image, fog included.** Segmentation ground
truth is geometry only: fog does not touch it. So the runner also measures each
target's colour contrast against the ground around it in the rendered frame
(CIELAB ΔE, recorded as `contrast` in `frames.jsonl`). A target washed out by
fog, below `MIN_CONTRAST` (ΔE 6, in `src/vivid2/sim/segmentation.py`), is
don't-care in the perception and reasoning scores: missing it is not an error,
and reporting it earns nothing. Measured on 2026-11-03 (`forest_clearing_01`,
`forest_hiker_02`, `forest_fog_01`; medians per hiker, then the range):

| Condition | Camera | Hiker contrast (ΔE) | Don't-care at ΔE 6 |
|---|---|---|---|
| Dusk, clear air (including camouflaged hikers) | 55 m | 22–32 (12–41) | 0 % |
| Dusk, fog 0.02 | 55 m | 12–32 (8–38) | 0 % |
| Noon, fog 0.2 | 55 m | 10 (9–11) | 0 % |
| Dusk, fog 0.05 | 55 m | 7–16 (5–30) | 17 % |
| Dusk, fog 0.05 | 12 m (a dip) | 20–26 (13–27) | 0 % |
| Noon, fog 0.4 | 55 m | 5 (4–10) | 57 % |

Camouflage is not fog: a yellow jacket among yellow birch crowns stands well
above the threshold, so missing it still counts. The oracle `coverage_gt` uses
the same rule, so it does not see through fog either. It is perfect perception
of the rendered frames, not of the scene. The contrast of every target is kept
in `frames.jsonl`, so a different threshold can be applied to finished runs
with `vivid2 rescore`.

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
