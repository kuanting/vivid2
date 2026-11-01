# Mission catalog

## Forest S&R: missing hiker (`mission: forest_sar`)

The drone gets a text brief and a search polygon. It must find every missing
person and report each one's position, then return home.

- **Success:** every target reported within `success.localization_tolerance_m`.
- **Mission score:** for each target, 1 − 0.5 × (time-to-find / timeout) if
  found, otherwise 0. The mean over targets is multiplied by 100, then 10
  points are subtracted per false alarm.
- **Variation axes:** target jitter (seeded), fog, time of day and wind. Canopy
  density and distractors arrive with the ForestSAR map in Phase 1.

| Scenario | Map | Targets | Sweep |
|---|---|---|---|
| `blocks_hiker_01` | Blocks | 1 proxy | 3 seeds |
| `blocks_hiker_02` | Blocks | 2 proxies | fog × time × 2 seeds |
| `forest_hiker_01` | ForestSAR | 1 hiker, red jacket, lying | 5 seeds |
| `forest_hiker_02` | ForestSAR | 2 hikers at dusk | fog × 3 seeds |

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
