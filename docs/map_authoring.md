# Map authoring rules

These rules come from how Project AirSim and VIVID 2 actually behave (checked in
the plugin source and on the Blocks map). A map that breaks them still opens and
flies, but its scores are wrong. Read this before building ForestSAR, Wildfire or
Landslide.

## Coordinates and height

- **NED is measured from the Unreal world origin**, not from the PlayerStart.
  Put the PlayerStart at X = Y = 0 anyway, on the ground (on terrain, the
  ground height at the origin plus its 92 cm capsule half-height), so the editor
  camera starts where the drone does.
- **Keep the whole play area between UE Z = 0 and Z = +1000 m.** Ground-height
  queries (`GetZAtPoint`) trace down from Z = +1000 m to Z = 0 and return 1000
  when they hit nothing. Offset the landscape vertically if the real terrain
  is higher or lower.
- **Set `home-geo-point.altitude`** in the scene config to the real-world
  elevation of UE Z = 0, so GPS altitudes stay right.
- **Build terrain with `vivid2 terrain build`** (see
  [environments.md](environments.md#forestsar-phase-1)). It writes the heightmap
  and the terrain model from the same numbers and follows the two rules above.
- **Start the drone above clear ground.** `dut_start.ned` must not be inside or
  touching any mesh, or the physics pins the drone to the contact point. On
  real terrain, take its height from the terrain model. Keep vegetation off
  the start: ForestSAR's PCG graph subtracts a 30 × 30 m launch clearing
  (`LaunchClearing`) around the origin from every layer.

## Everything must exist when the scene loads

The runner assigns segmentation IDs right after spawning the targets. Only
objects present at that moment are drawn as occluders in the segmentation
image. An occluder that appears later is see-through, and targets under it
count as visible when they aren't.

- **Bake PCG output** (foliage, rocks, clutter) into the level. Don't generate
  at runtime.
- **No World Partition streaming** for the search area. Load the whole area
  (about 1 km²) at all times, or disable World Partition.
- Avoid HLOD swaps over the search area, for the same reason.
- **Turn off occlusion culling** (Project Settings > Engine > Rendering >
  Occlusion Culling, `r.AllowOcclusionQueries=False`). With it on, the payload
  camera reuses stale visibility, and landscape tiles go missing (white voids
  under floating trees). In ForestSAR this hit 13 of 130 survey frames; with it
  off, none. Check a new map with `tools/count_holes.py` on a survey recorded
  with `--frame-bank rgb`.

## Map identity

- **Add a marker actor** named `BP_VIVID2Map_<map>` (any empty blueprint), for
  example `BP_VIVID2Map_ForestSAR`. Preflight checks for it, so a run never flies
  on the wrong level. Each canopy variant is its own map with its own marker
  (`ForestSAR_Open`, `ForestSAR_Dense`).
- Scenarios name the map in `scene.map`.

## Obstacles must collide

Benchmarks reward flying low among trees, so everything a drone could hit needs
collision, and collision that matches the shape. Check both levels:

- **The mesh:** asset packs often give trees a token trunk capsule. Use a convex
  decomposition that follows trunk and crown (`tools/apply_tree_collision.py`
  for the Forest Fire pack), not one hull around the whole tree, which would
  close the space under the canopy.
- **The placement:** PCG's Static Mesh Spawner defaults to *no collision*. Set
  the descriptor's collision profile to `BlockAll` for obstacles.
- **Verify in Play,** not with editor line traces: the editor's trace tool skips
  instanced meshes. `world.get_surface_elevation_at_point` over a tree must
  return the crown top, and a drone flown into a tree must report a collision.

## Targets and distractors

- **Asset names must be unique across all content**, Fab packs included.
  Project AirSim looks assets up by short name, and the last one found wins.
  Prefix ours: `VIVID2_Hiker_RedJacket_Lying`.
- **Target ids must not be contained in any existing actor's name.** Spawning
  `Hiker1` renames it if an actor such as `Hiker1_Statue` exists. Preflight
  catches this, but it's better avoided.
- **Pose hikers in the mesh.** Spawned skeletal meshes play no animation and
  stand in the reference pose. Make standing, sitting and lying variants as
  static meshes, or as blueprints with a fixed pose.
- **Put the pivot at ground level** (feet, or the lowest point when lying), so
  a target's `ned.z` is the ground height.
- **Cook everything spawned by name.** Assets referenced only by name are not
  packaged. Add the folder to `DirectoriesToAlwaysCook` in
  `Config/DefaultGame.ini`, for example `+DirectoriesToAlwaysCook=(Path="/Game/VIVID2/Targets")`.

## Lighting and weather

- Use **SunPosition** with the scene's home latitude and longitude, so
  `environment.time_of_day` sets the sun correctly.
- Add a volumetric fog volume, so `environment.weather.fog` has a visible effect.
- After building, run a scenario with each fog and time-of-day value and check
  the camera images actually change.

## Third-party content

Fab and Megascans source assets may not be redistributed. Keep them under
their own `Content/` folders (the repo ignores everything except
`Content/VIVID2/`). List the packs a map needs in `docs/environments.md`, and
ship the map to users as a packaged build.
