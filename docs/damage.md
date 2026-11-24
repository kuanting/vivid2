# The damage task

In the disaster scenes (landslide, avalanche, earthquake, tsunami) a drone has two jobs:

1. **Find people**, scored exactly as in the forest and wildfire missions.
2. **Map the damage**: what the disaster did, for the crews who follow. This document is about
   the second job.

Why both: people buried under debris, snow or rubble can't be seen by a drone's colour camera,
and not by a thermal camera either. A few centimetres of snow or rubble hide body heat. Real
crews find buried people with dogs, probes, avalanche beacons and listening devices. The
drone's job is to tell them *where to look*: the debris field, the blocked roads, the collapsed
buildings, the clues on the surface. So:

- **Scored people** in these scenes are people a camera can see: on the surface, on roofs, in
  water or vehicles, or partly buried with something showing.
- **Fully buried people** are hidden ground-truth points (`damage.buried`). They are never
  targets. They score how well the damage map would send rescuers to them.

## Products

A scenario turns the task on with a `damage` block and asks for some of four products:

| Product | What the stack reports | Report type |
|---|---|---|
| `areas` | polygons of ground of an area class (`debris`, `flood`, ...), added or removed in order | `AreaReport(area_class, polygon_ne, frame_id, t_ns, op="add"\|"remove")` |
| `roads` | whether a known road segment is blocked (the last report counts) | `RoadReport(segment, blocked, frame_id, t_ns)` |
| `buildings` | a known building's damage grade: `none`, `partial` or `collapsed` (the last report counts) | `BuildingReport(building, grade, frame_id, t_ns)` |
| `clues` | where a surface clue lies (a ski, a backpack...) and its kind | `ClueReport(ned, kind, frame_id, t_ns)` |

Reports go through `ctx.report_damage(report)` (all four types are in `vivid2.stack.base`).

```jsonc
"clues": [{"id": "Pack1", "asset": "...", "ned": [42, -48, 0], "kind": "backpack"}],
"damage": {
  "products": ["areas", "roads", "buildings", "clues"],
  "area_classes": ["debris"],             // default: every class in the map's ground truth
  "aoi_ne": [[...], ...],                 // default: the search area (must lie inside it)
  "buried": [{"id": "Buried1", "ned": [35, -50, 0], "building": "B07"}],
  "weights": {"buried": 0.5},             // per component, see Scoring
  "clue_tolerance_m": 5
}
```

The mission must be a disaster mission (`landslide_sar`, `avalanche_sar`, `earthquake_sar`,
`tsunami_sar`); each of these also finds people.

## What the stack is told

`ctx.mission.damage` is what responders would have before the event, never the damage:

- `products`, `aoi_ne`;
- `area_classes` (`{name: description}`), `building_grades`, `clue_kinds`;
- `roads`: every segment in the area of interest, `{id, road, polyline_ne, width_m, kind}`;
- `buildings`: every building in it, `{id, footprint_ne, height_m, roof, use}`;
- `terrain`: `"pre_event"` when `ctx.mission.terrain` is the DEM from before the event. A map
  whose ground changed (a landslide deposit) ships its pre-event terrain, and the stack's coarse
  DEM is made from that.

## Rules

Each report must cite one of the drone's own recorded search-camera photos (as target reports
do, [rules.md](rules.md)). A road, building or clue report must also show its item in that
photo: at least one cell of the road's corridor, a quarter of the building's footprint, or the
clue's position must be in the photo's observed ground (below). A report that breaks a rule is a
violation in `result.json`'s `rule_violations`:

| Violation | When |
|---|---|
| `uncited_damage_report` | the photo was never recorded |
| `damage_cites_nav_frame` | it is a navigation-camera photo |
| `damage_cites_other_drone` | in a team, another drone's photo |
| `damage_not_in_cited_frame` | the road, building or clue is not in that photo |
| `invalid_damage_report` | an unknown segment, building, area class, grade or clue kind, or a polygon with fewer than 3 or more than 2000 vertices |

A road, building or clue report that breaks a rule counts as a false positive. An area report
that breaks one is dropped.

## What a photo observes

One definition, `vivid2.damage.geometry.view_mask`, used by the metrics, the rules and the
reference oracle. A ground cell (2 m) is observed by a photo when:

- it is in the image;
- it is at most 400 m away;
- the photo resolves it to 0.30 m per pixel or better (ground sampling distance
  √(x³ / (f² r cos i)) for planar depth x, range r, incidence i, focal length f in pixels);
- the terrain does not rise above the line of sight to it.

Trees and buildings never hide a cell: a cell under a crown counts as observed. The ground the
run observed is the union over all its search photos (a team's, for a team).

## Scoring

All on the area of interest's 2 m cells:

| Component | Score |
|---|---|
| `areas` | per class, IoU of the ground truth and the reported area clipped to the observed ground; the mean over classes |
| `roads` | F1 of "blocked" over the segments whose midpoint is in the area of interest |
| `buildings` | ½ F1(damaged: partial or collapsed) + ½ F1(collapsed), over the buildings whose centre is in it |
| `clues` | F1; a report matches the nearest unclaimed clue within `clue_tolerance_m` (a repeat report of a found clue is not a false alarm) |
| `buried` | the share of buried victims within 3 m of a reported area cell, or in a building reported partial or collapsed |

- F1 when there is nothing to find: 1 if nothing was claimed, 0 otherwise.
- The **damage score** is 100 × the weighted mean of the components the task asks for and the
  ground truth has, default weights 1 each and 0.5 for `buried`.
- A stack that reports nothing scores **0**, not "n/a": skipping the task never pays.
- The damage score is an extra weighted layer, `damage` (weight 0.25 next to the other six
  layers' 1.0 total, so 20 % of a damage run's overall score). Runs without a damage task have no
  `damage` layer and score as before.
- `metrics.damage` also holds per-class precision and recall, the share of each class the run
  observed, a 3 × 3 building confusion matrix, clue kind accuracy, which buried victims were
  covered, and the damage mapper's model calls and cost.

## Reference stacks

Every reference search agent can carry a damage layer (`vivid2.stack.base.DamageMapper`). It
looks at the survey photos it picks (`wants`), after the person search or in the background, and
its reports go out as it goes (roads, buildings, clues) or at the end of the survey (area
outlines). With a damage layer the agent flies the whole survey even after everyone is found.

| Stack | Damage layer |
|---|---|
| `coverage_gt` | `GroundTruthDamageMapper`: reports the ground truth its own photos observed. The upper bound for its flight; not eligible for ranking. |
| `vlm`, `hybrid` | `VLMDamageMapper`: asks the stack's VLM about labelled survey photos (below) |
| `coverage_yolo`, `survey` | none yet: they score 0 on the damage task |

### The VLM damage mapper

`vivid2.stack.reference.damage_vlm.VLMDamageMapper` uses the stack's own VLM (the same model, effort
and record/replay cache as its person search) with its own prompt, `vlm/prompts/<mission>_damage.md`
(an intro and one `@@` section per product; a mission needs sections only for the products its
scenarios ask for: the avalanche's prompt has areas and clues, no roads or buildings).

- **Which photos.** A straight-down search photo from at least 60 % of the survey height whose
  footprint is at least half new ground (ground no earlier mapped photo showed), up to 80 per run.
  On the landslide that is about 21 photos, so each patch of ground is seen once or twice.
- **What is sent.** Two images of the photo, each at most 1568 px: the photo as taken, and the
  photo with a 12 × 9 grid of labelled cells (A1 at the top left), the task's road segments drawn
  along their centrelines and its buildings outlined, each with its id. One structured reply
  answers every product asked for: the cells each area class covers at least half of, each
  labelled segment open, blocked or not visible, each labelled building's grade or not visible,
  and clues with their kind and image point.
- **How answers become reports.** The photo's depth places a lattice of its pixels on the ground
  (where there is no depth, the pixel's ray meets the DEM), so each answer is a vote on the 2 m
  cells under the grid cells it names. A cell is in a class when at least half of the looks at it
  put it there; the outlines go out at the end of the survey. A segment is blocked when its
  "blocked" answers reach half of its "open" ones, each "open" counted by the share of the segment
  that photo showed (the blockage may be in the part it did not show). A building takes the grade
  most answers gave (a tie: the more severe). Roads and buildings are reported whenever that
  changes, citing the photo that changed it. A clue placed by depth is reported at once; one placed
  on the DEM waits for a better look and goes out at the end if none comes.
- **Cost and timing.** One call per mapped photo, logged in `decisions` with layer `damage_mapper`
  (the person-search reasoning metrics leave them out; `metrics.damage.mapper` counts calls, errors,
  tokens and cost). Calls run in the background, three at once, so the person search goes on; the
  drone waits for the last ones before it reports the outlines and flies home.

Evaluate the mapper offline on a run's saved photos (a run recorded with `--frame-bank rgbd`):

```
python tools/eval_damage_mapper.py runs\landslide_v1\coverage_gt\landslide_sar_01-*                    # answers from the ground truth
python tools/eval_damage_mapper.py runs\landslide_v1\coverage_gt\landslide_sar_01-* --answers anthropic  # Claude: asks before spending
```

`--answers truth` answers every question from the ground truth: the best this mapper can do with
those photos (on the landslide oracle runs: damage about 91, areas IoU 0.86, the grid's resolution).
`--save-sent` writes the images and prompt of every call; `--dry-run` writes them without asking.
Saved depth is millimetres and saturates at 65.5 m, which on the landslide's steep slope leaves
far parts of the photos to the DEM; live depth has no such limit, so offline scores are a little
pessimistic.

## The ground truth

Each disaster map has one source, `unreal/VIVID2Env/SourceData/<map>/layout.json`, and
`vivid2 damage build --map <map>` turns it into:

- `sim_config/maps/<map>/damage_gt.json` + `damage_gt.npz`: area classes (a bit field on the
  terrain model's 1 m grid), ground raised or lowered by the event, water depth, road segments
  with their blocked flags and causes, buildings with footprints and grades
  (`vivid2.damage.gt`);
- `surface_dsm.npz`: ground, roofs and water as one height model, for targets with
  `"snap": "surface"` (people on roofs or on water);
- the region masks (`masks/<region>.png`) the landscape material and the PCG graphs read, and
  `placement.json`, the actors the map build places.

The rules it applies (`vivid2.damage.layout`):

- An area class covers a grid node where its regions' field is at least 0.5, the same threshold
  the materials and PCG use. A `water` class covers land under the water level.
- A road segment (at most 100 m long) is **blocked** when some 4 m stretch has at least 60 %
  of its corridor covered by: a road-blocking area class, water at least 0.3 m deep, a collapsed
  building's rubble (its debris polygon, or its footprint plus 30 % of its height), or a
  blocker polygon. One lane free leaves a road open; debris across its whole width for about
  2.5 m or more blocks it.
- Building grades come from EMS-98 grades in the layout (0-1 none, 2-3 partial, 4-5 collapsed)
  or, with `grade_from`, from how much of the footprint an area class covers (half or more:
  collapsed; 15 % or more: partial). Grade a building partial only if the damage shows from
  the air.
- `expect` in the layout lists the blocked segments and grades the author intends; the build
  fails if they differ.

The ground truth is committed and never overwritten once runs exist (`vivid2 damage build`
refuses without `--force`): rescoring checks its checksums, as it does the terrain's.

## Authoring a disaster scenario

- Every id (targets, decoys, fires, clues, buried) is unique and none contains another.
- Clues lie inside the area of interest; buried victims within 2 m of a damage area or in a
  damaged building; partly buried people (authored more than 0.2 m below the ground) on debris,
  flood or rubble; roof people (`"snap": "surface"`) on a flat roof or water, jitter included.
  `vivid2 run` checks all of these before a run starts.
- Fly the oracle with `--frame-bank rgbd` and run `python tools/check_targets.py <run>`: every
  scored person must reach twice the visibility threshold in some photo from survey height
  (move it, or make it a buried victim), every clue the threshold, and nobody may float.
  Only photos where it stands out from the ground around it count (contrast ΔE 6, the scoring's
  rule): at dusk, fog of 0.05 already washes the ground out from survey height, so dusk
  scenarios use haze (0.02) at most. On a terrain-following survey "survey height" is above
  the ground under the drone. People on roofs are checked against the surface model, and
  partly buried people (checked on debris before the run) are a note, not a failure.
  Project AirSim's depth comes in whole metres for now, so the float check uses the terrain
  model: check targets standing on meshes (rubble, boulders) by eye. For the legacy
  Blocks proxies add `--centre-pivot` (the cubes' positions are their centres).
- On steep ground (a landslide, an avalanche path) use `"altitude_ref": "terrain"` in the
  search area, so the survey keeps its height above the ground ([missions.md](missions.md#writing-a-scenario)).
  The reference stacks geolocate against the pre-event DEM, which lies under a deposit: allow
  10 m localization tolerance, as on the other maps with relief.
- `python tools/damage_png.py <run>` draws the ground truth over the run's photos: check that
  the debris, roads and buildings in the ground truth are what the photos show.

## Terrain that changed

`vivid2 terrain build --like <map> --layout layout.json --stage pre|post` builds a map's terrain
with the layout's `terrain` operations (`vivid2.terrain_edit`): a road bench cut into a slope, a
slide scar lowered, a deposit raised, a town's plain flattened. `--like` pins the height datum
and heightmap scale to an existing map, so ground the edits leave alone gets exactly the same
heightmap values. `--stage post` also writes `pre_event_dtm.npz`, the terrain before the event
that the stack's DEM is made from. Landscape heightmaps can't be imported by script in UE 5.8:
import each one in Landscape mode, as for ForestSAR ([environments.md](environments.md)).
