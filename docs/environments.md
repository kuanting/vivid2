# Building the VIVID 2 environments

All maps live in `unreal/VIVID2Env` and use `ProjectAirSimGameMode`. Maps are
**authored in UE 5.8**: 5.8 assets can't be opened in 5.7, so 5.7 users run
packaged builds. See [unreal_versions.md](unreal_versions.md).

## Setup

1. Get a UE 5.8 Project AirSim plugin and render the 5.8 project, using either:
   `vivid2 unreal setup --engine 5.8 --download` (prebuilt), or
   `vivid2 unreal setup --engine 5.8 --plugin-dir ..\PASBlocks\Plugins` (a local build).
2. Open `unreal\VIVID2Env\VIVID2Env.uproject` with UE 5.8 and let it compile the
   module. For 5.8, the project enables the MCP server and toolsets
   (ModelContextProtocol, EditorToolset, PCGToolset, MetaHumanGenerator,
   NiagaraToolsets, ConfigSettingsToolset). Start the server from the editor
   console with `ModelContextProtocol.StartServer`; it then serves MCP at
   `http://127.0.0.1:8000/mcp`. Only one editor can use the port, so close other
   UE projects first. The repo's `.mcp.json` registers it as `unreal-mcp` for
   Claude Code sessions started in the VIVID2 folder.
3. Create maps under `/Game/VIVID2/Maps/`: `ForestSAR` (and its variants), `ForestFire`, `Landslide`.
   In each map, set World Settings → GameMode Override to `ProjectAirSimGameMode`.
4. Place a marker actor named `BP_VIVID2Map_<map>` (for example
   `BP_VIVID2Map_ForestSAR`, any empty blueprint). The runner's preflight check
   uses it to confirm the right map is open.
5. Put a PlayerStart at X = Y = 0, on the ground. Project AirSim measures NED
   coordinates from the Unreal world origin, not from the PlayerStart.

## Third-party content and distribution

The repo never contains third-party source assets. Only VIVID 2's own content
under `Content/VIVID2/` is committed. Maps there may reference pack assets, but
the packs themselves are ignored by git.

- **Users** get packaged `VIVID2Env` builds (cooked, in `.pak` files), made with
  `tools\package_env.ps1` and published as release downloads, one per group of
  maps that share content ([unreal_versions.md](unreal_versions.md#packaged-builds)).
  They need no Unreal or Fab account.
- **Map authors** who want to edit the maps in the editor must get each pack
  below from Fab themselves and install it at the listed folder.

| Pack | Folder | Used for | Notes |
|---|---|---|---|
| Forest Fire (Fab) | `Content/ForestFire/` | ForestSAR vegetation (birch trees, shrubs, ferns, grass, stumps, logs, rocks); the flipbook textures and materials of VIVID 2's fire effects; ForestFire's burnt ground texture and the parents of its burnt material instances | Made for UE4; see [the pack in UE 5.8](#the-forest-fire-pack-in-ue-58). |
| Hiking Collection (Fab) | `Content/Hiking_Collection/` | The hikers: a man and a woman in hiking clothes with a backpack | `tools/build_hikers.py` changes it locally: compatible skeletons, textures capped at 2048 px |
| Lie Down Animation Pack (Fab, Ailive) | `Content/Fab/Lie_Down_Animation_Pack/` | Hiker poses: on the back, face down, on the side; its Manny skeleton also carries VIVID 2's waving clips | Fab imports it from FBX; save it after adding it to the project |
| Third Person template (engine) | `Content/Characters/` | Hiker pose: crouching (`MM_Death_Back_01`) | See [Targets](#targets-hikers) |
| Velocity Cars Pack (Fab) | `Content/Velocity_Cars_Pack/` | The cars in LandslideBefore and Landslide | Only its static blueprints (`Blueprints/Static_Blueprints/BP_Static_Velocity_*`) are placed, so builds carry their meshes and materials, not the drivable cars' skeletal meshes |

## Targets (hikers)

Scenarios spawn targets by **asset name** through `spawn-objects`. Project
AirSim looks names up in the asset registry: static meshes, skeletal meshes and
blueprints.

**Built (v3, 2026-11-23):** 28 blueprints in `/Game/VIVID2/Targets`, named
`VIVID2_Hiker_<Outfit>_<Pose>`: realistic people in hiking clothes (jacket,
trousers, boots, a dark backpack with a harness), with faces and hair. In v2
(2026-11-18, `SIM_VERSION` 8) they replaced the v1 hikers, which were UE5
mannequins in flat colours, under the same names, so scenarios didn't change.
v3 (`SIM_VERSION` 9) gave `Waving` VIVID 2's own two-arm wave and added
`WavingOneArm`. `tools/build_hikers.py` builds them through the editor's MCP
server. Rebuilding a blueprint doesn't change copies already saved in a map
with the old values; scenarios spawn hikers from the blueprint, so they always
get the current one.
- **Outfits:** the jacket and trouser colours are those of the v1 hikers and of
  the decoys, so the decoys still match.

  | Outfit | Person | Jacket | Trousers |
  |---|---|---|---|
  | `RedJacket` | man | red | dark |
  | `YellowJacket` | woman | yellow | grey |
  | `BlueHoodie` | teenager (the man at 94 %) | dark blue | blue |
  | `GreenJacket` | man | olive (low contrast) | khaki |

- **Poses:** each is one frame of a clip: motion capture from the packs, or
  VIVID 2's own keyframed waving clips.

  | Pose | Clip (pack) | Looks like |
  |---|---|---|
  | `Standing` | `MM_Idle` / `MF_Idle` (Hiking Collection) | standing, arms down |
  | `Crouching` | `MM_Death_Back_01` at 1.1 s (Third Person template) | crouched, hunched forward |
  | `LyingBack` | `Sprawling_Out_on_the_Floor` at 5 s (Lie Down) | on the back, arms and legs spread |
  | `LyingFront` | `Liying_Face_Dwon_on_the_Floor` at 5 s (Lie Down) | face down, arms folded over the head, backpack up |
  | `LyingSide` | `Lying_on_the_Side_with_One_Arm` at 6 s (Lie Down) | on the side, propped on one arm |
  | `Waving` | `VIVID2_Wave_TwoArms` at 0 s (VIVID 2) | waving for help: both arms overhead, open in a V, head up |
  | `WavingOneArm` | `VIVID2_Wave_OneArm` at 0.25 s (VIVID 2) | right arm raised and waving out to the side, left arm down, head up |

  `LyingSide` and `Waving` are new in v2, `WavingOneArm` in v3. The hiker faces
  +X (north at `yaw_deg` 0).
- **Waving clips:** `tools/wave_clips_blender.py` keyframes them in Blender
  (headless) on the Lie Down pack's Manny skeleton: the skeleton's reference
  pose turned to face +X, a two-bone arm solve, forearm twist for a forward
  palm, raised clavicles, open fingers and the head tilted up. They loop (one
  arm: 4 s, a wave a second; both arms: 4.8 s, between the V and crossed over
  the head every 1.2 s), so the demo movie can play them. Its FBX output is
  committed in `unreal/VIVID2Env/SourceData/Hikers/`; `tools/build_hikers.py`
  imports it into `/Game/VIVID2/Targets/Anims/`. They replace the Body Wave
  Animation Pack's `Both_Arm_Wave`, a dance move that looked like dancing.
- **Construction:**
  - **Mesh:** each blueprint is one skeletal mesh component, the pack's
    combined man or woman (body, head, card hair and outfit in one mesh). It
    holds one frame of a clip: single-node animation, not playing. A moving pose
    would differ between runs, because the game clock follows the wall clock.
  - **Colours:** the jacket and trousers use instances of
    `M_VIVID2_ClothTint`. It keeps the pack's fabric (normal, roughness and
    metallic maps) and sets the colour: base colour = colour × the texture's
    relative brightness, clamped to 0.6–1.6.
  - **Collision:** the mesh's physics asset collides (BlockAll), so preflight's
    3D box and drone collisions see the hiker.
- **The clips are Manny clips on a MetaHuman-style skeleton.** The pack's
  skeleton (`SK_Human_Skeleton`) lists the clips' skeletons as compatible, and
  takes bone translations from them (`bUseRetargetModesFromCompatibleSkeleton`).
  With the pack's own retarget modes, a Manny clip pulls the right forearm off
  at the elbow. These are local changes to the pack, made by the builder.
- **Pivot:** on the ground, under the body's centre. Some clips move the body
  away from the root (the crouch by 2 m). The builder offsets the mesh by the
  measured centre of each pose's footprint, from line traces against the
  posed body. Measured centres are within 20 cm of the pivot.
- **Size:** a standing man is 1.78 m, the woman 1.68 m.
- **Textures:** the pack ships 8K textures. The builder caps the 64 the hikers
  use at 2048 px (a local pack change): the payload camera never sees a hiker
  larger than about 1,000 pixels, and the packaged build stays small.
- **Cooking:** `DirectoriesToAlwaysCook` includes `/Game/VIVID2/Targets`,
  because hikers are spawned by name and no map references them. The pack
  assets they use are cooked as references.
- **Rebuild:** add the packs (table above; save the Fab animation packs after
  adding them), open the editor with the MCP server on, then run
  `python tools/build_hikers.py`. It updates existing assets in place. Deleting
  and re-creating an asset that something references breaks the references.
- **Third Person template content:** the crouch clip comes from the engine's
  Third Person template (`Content/Characters`, not committed). Add it with
  Add → Add Feature or Content Pack → Third Person, or copy
  `<UE>/Templates/TemplateResources/High/Characters/Content` to
  `Content/Characters`.
- **Not done yet:** sitting on the ground (from the Homeless NPC Animation
  Pack, not installed yet), more outfits, and realistic decoys. Next to real
  people, the basic-shape decoys are easier to rule out.

Rules for any target asset:
- The origin must be at ground level (the feet, or the lowest point when
  lying), because scenario `ned.z` is the ground height.
- Make sure the asset gets a segmentation ID. The runner sets one per target
  by name right after spawning.

On Blocks, `1M_Cube` scaled to 0.5 × 0.5 × 1.7 stands in for a person. This
is only good for pipeline bring-up. It is meaningless for evaluating a VLM.
The legacy `blocks_*` scenarios run in Project AirSim's own Blocks project
(BlocksMap's floor is at NED −1.0 m, so their proxies stand 1 m deep in it;
they stay that way to keep their saved results). VIVID2Env has its own test
map instead.

## TestYard (test map)

`/Game/VIVID2/Maps/TestYard` is VIVID 2's own test map: a flat yard with its
ground at height 0, two crossing roads, four box houses and a landslide's
debris lobe, all made of engine shapes and VIVID 2's materials. The smoke
campaigns run on it (`campaigns/smoke.jsonc`: one hiker;
`campaigns/smoke_damage.jsonc`: the damage task). Open it in VIVID2Env and
press Play, or start any packaged build without a map.
- The damage is real geometry, so a camera-based damage mapper can be tested
  here: debris (brown ground, soil mounds, boulders, logs) across road R01;
  house B02 partly damaged (half its roof sagging in, rubble at its south
  wall); house B03 collapsed into a rubble pile across road R02; B01 and B04
  intact.
- One source, `SourceData/TestYard/layout.json`, gives both the ground truth
  (`vivid2 damage build --map TestYard`) and the geometry
  (`python tools/build_testyard.py`, which places every actor from the layout
  and its ground truth). It is a small rehearsal of how the disaster maps are
  built.
- The level is a copy of ForestSAR_Dense without its landscape and forest, so
  it has the same sun, sky and fog.

## ForestSAR (Phase 1)

- **Site:** a forest-zone bench near Lishan, Taichung, Taiwan, centred on
  24.2436 N, 121.2467 E (2242 m mean elevation). The central 400 m search area
  varies 27 m in height; a steep valley to the east gives 278 m of relief across
  the 1 × 1 km map. The Hehuan Main Peak point first planned (24.1426 N,
  121.2712 E) was rejected: it is alpine grassland with 397 m relief and 41°
  slopes inside 1 km.
- **Terrain:** built from the Copernicus GLO-30 DEM (30 m; a surface model, so
  it includes some canopy height) by

  ```
  vivid2 terrain build --map ForestSAR --lat 24.2436 --lon 121.2467
  ```

  This writes `unreal/VIVID2Env/SourceData/ForestSAR/heightmap.png`
  (1009 × 1009 at 1 m), the terrain model `sim_config/maps/ForestSAR/terrain_dtm.npz`
  and `terrain.json`, which holds the Landscape import settings, the home geo point
  and the DEM attribution. Import it in Landscape mode → Manage → New → Import from
  File with the printed settings (location `[0, 0, 14892]` cm, scale
  `[100, 100, 55]`, 63 × 63 quad sections, 2 × 2 sections per component,
  8 × 8 components). UE 5.8 has no scripting call for creating a landscape, so
  this one step is manual. Then check it: tracing the landscape at random points
  must give the terrain model's heights. On 2026-11-02, 400 points matched to
  within 2.6 mm.
- **Lighting:** the SunPosition plugin's `SunSky` actor. It contains the sun,
  sky light, atmosphere and clouds; the template's separate ones were removed.
  Project AirSim finds it by name for `set_sun_position_from_date_time`. Its
  settings:
  - latitude 24.2436, longitude 121.2467, time zone +8, no daylight saving;
  - sun intensity 6, the template's value, because the project uses fixed
    exposure;
  - north offset 0, which makes Unreal +X north, matching NED.

  Noon on 28 Sep puts the sun at 63.5° elevation, azimuth 188°.
- **Backdrop (optional):** the landscape ends at the 1 km edge. The payload
  camera looks straight down (80° FOV, about 85 m wide at 50 m), so the edge
  never shows in evaluated frames. Only the chase camera and screenshots see
  the empty space past it. A coarser surrounding landscape from the same DEM
  would fix that.
- **Ground material:** `/Game/VIVID2/Materials/M_VIVID2_Landscape`. It tiles in
  world space and needs no painted layers:
  - leaf-litter dirt, varied with pebbly dirt and a larger copy of itself to
    hide tiling;
  - green moss/herb patches from noise;
  - rock on slopes steeper than about 31°.

  Its parameters (group "VIVID2 Ground") set the tiling, the slope threshold,
  the patch amount and the tint.
- **Vegetation (built):** `/Game/VIVID2/PCG/PCG_VIVID2_Forest`, run by the
  `PCG_Forest` volume over the whole landscape. It has four layers:
  - **Trees:** birches, 19–28 m meshes scaled 0.7–1.1.
  - **Understory:** ferns, ground cover and young birches.
  - **Grass:** in the clearings.
  - **Clutter:** rocks, stumps and 11 m fallen logs.

  One canopy noise (about 65 m features) splits the map into forest patches and
  clearings. Graph parameters set the densities and the clearing fraction:
  `TreeDensity` 0.012, `ClearingFraction` 0.45, `UnderstoryDensity` 0.04,
  `GrassDensity` 0.4, `ClutterDensity` 0.002. With these the base map has about
  8,600 trees, 36,000 understory plants, 2,000 clutter objects and about
  120,000 grass clumps. Trees, young birches, rocks, stumps and fallen logs
  collide (BlockAll); ferns, shrubs and grass do not. The graph's mesh spawners
  default to **no collision**, so each obstacle entry asks for it explicitly.
  The pack's birches ship with one small capsule on the lower trunk;
  `tools/apply_tree_collision.py` replaces it with a 24-hull convex
  decomposition (trunk, then crown layers from about 4–6 m up). Run it once
  after installing the pack. Checked in the sim on 2026-11-03: surface queries
  at ten tree positions all hit the crown 8–19 m above the ground, and a drone
  flown into a crown at 8 m reported a collision. The volume is
  set to **Generate On Demand**, so it never regenerates at runtime: generate it
  in the editor and save the level, which then stores every instance.
  Not done yet: thinning trees on steep slopes, and keeping a trail
  spline clear.
- **Canopy variants (built):** `ForestSAR_Open` and `ForestSAR_Dense` are copies
  of the level, each with its own marker (`BP_VIVID2Map_ForestSAR_Open` /
  `_Dense`) and these forest volume parameters:

  | Map | TreeDensity | ClearingFraction | UnderstoryDensity | GrassDensity | ClutterDensity |
  |---|---|---|---|---|---|
  | `ForestSAR_Open` | 0.005 | 0.6 | 0.03 | 0.5 | 0.002 |
  | `ForestSAR` | 0.012 | 0.45 | 0.04 | 0.4 | 0.002 |
  | `ForestSAR_Dense` | 0.025 | 0.2 | 0.06 | 0.4 | 0.003 |

  They share ForestSAR's terrain: their `sim_config/maps/<map>/terrain.json`
  says `"terrain_of": "ForestSAR"`. A scenario picks one with `scene.map`, and a
  campaign can sweep it for all its scenarios: `campaigns/forest_canopy.jsonc`
  runs `forest_hiker_01` on all three. Preflight only runs the runs for the
  map that is open; open the next map and run the campaign again, since
  finished runs are skipped.
- **Clutter:** the PCG rocks, stumps and fallen logs are natural look-alikes
  but not scored objects.
- **Decoys (built):** `VIVID2_Decoy_*` blueprints in `/Game/VIVID2/Targets`, made
  from engine basic shapes in the hikers' clothing colours:

  | Decoy | Size | Looks like |
  |---|---|---|
  | `Backpack_Red` | 0.5 m | a red jacket, small |
  | `Tarp_Red` | 2.0 x 1.4 m, flat | the red jacket |
  | `Tent_Orange`, `Tent_Green` | 2.2 m domes | a bright / low-contrast blob |
  | `SleepingBag_Blue` | 2.1 m, person-sized | a lying person in a blue hoodie |
  | `Jacket_Yellow` | 0.9 m, arms spread | the yellow jacket |
  | `Barrel_Blue` | 0.6 m drum | blue clothing |

  Scenarios place them under `distractors`. They get segmentation IDs from 101,
  each frame records how much of each is visible (`distractors_visible`), and a
  report near one is a false alarm (`fp_on_distractor` in the reasoning
  metrics). In a 2026-11-03 survey of `forest_hiker_01`, all four of its decoys
  were seen from survey height, with up to 100–1,800 visible pixels.
- **Fog:** the scenario's `weather.fog` goes to Project AirSim's weather
  effects. The level's ExponentialHeightFog is only the base haze.
- **Terrain height:** the runner snaps targets and the drone start onto
  `terrain_dtm.npz`, and the safety metrics measure height above it. The
  reference planner flies at a fixed altitude above the start unless the
  scenario asks for a terrain-following survey (`search_area.altitude_ref`,
  [missions.md](missions.md#writing-a-scenario)), as the landslide scenarios do.

## ForestFire (Phase 2)

ForestSAR after a forest fire: the same landscape, terrain model, trees and
lighting, with the east side burnt. It is the map of the Wildfire S&R mission
([missions.md](missions.md#wildfire-sr-mission-wildfire_sar)).

- **The burn.** A fire came up the steep eastern valley and burnt everything
  east of an irregular front 70-130 m east of the start: 30 % of the map. The
  burn polygon and suggested fire positions are in
  `unreal/VIVID2Env/SourceData/ForestFire/burn_layout.json`;
  `tools/burn_mask.py` turns them into `burn_mask.png` (edges roughened by
  fixed noise), imported as `/Game/VIVID2/Textures/T_VIVID2_ForestFire_BurnMask`.
- **Ground:** `M_VIVID2_Landscape_Fire`, a copy of `M_VIVID2_Landscape` that
  blends to the pack's burnt-grass texture where the mask says so (parameters
  in group "VIVID2 Burn").
- **Forest:** `PCG_VIVID2_ForestFire`, a copy of `PCG_VIVID2_Forest` with the
  same parameters and seed. After all of its sampling it splits each layer by
  the mask, so every tree stands where it stands in ForestSAR. Inside the burn,
  the 2,419 trees, the stumps and the rocks keep their meshes and collision but
  get charred bark and scorched brown crowns (material instances in
  `/Game/VIVID2/Materials/Burnt`); understory and grass are gone. Outside it,
  6,174 trees, as in ForestSAR.
- **Rebuild:** `tools/build_forest_fire.py` (through the editor's MCP server)
  re-derives all of the above in the committed level; the level itself is a
  copy of ForestSAR with its own marker `BP_VIVID2Map_ForestFire`. Its terrain is
  ForestSAR's (`sim_config/maps/ForestFire/terrain.json`). The lighting and base
  haze are left as in ForestSAR, so the two maps differ only in the burn and the
  fires; scenarios add smoky haze with `weather.fog`.
- **Fires** are not part of the map: scenarios list them (`fires`). The
  blueprints `VIVID2_Fire_{Small,Medium,Large,Smolder}` in `/Game/VIVID2/Fire`
  wrap Niagara systems `NS_VIVID2_Fire_*` in `/Game/VIVID2/FX`, built by
  `tools/build_fire_assets.py` from the pack's flipbook materials. Each has
  flames (none for `Smolder`), a smoke plume that drifts along its +X axis
  (about 3 m/s sideways, rising to some 40 m), a constant shadowless light, and
  a ground-truth proxy for segmentation. The rules are in
  [map_authoring.md](map_authoring.md#fires-and-smoke).
- **Checked live** (2026-11-10, `wildfire_hiker_01`, oracle, noon and dusk): all three hikers found, no
  false alarms, no collisions, no time in the no-fly area, 0 landscape holes. Check holes on this map
  with a survey recorded with `--frame-bank rgbd`: in colour alone, `tools/count_holes.py` takes the pale
  ash for holes. The same run twice: same frames, camera 0.7 m apart (median), no hiker switched
  between visible and don't-care, overall 96.61 vs 96.65.
- **Not done yet:** a spreading fire front (`vivid2.hazards`), Megascans burnt
  snags and logs (free on Fab: Burnt Tree Trunk, Burnt Log, Burnt Tree Stump,
  Military Trenches Tree Burnt; Coal Ash for the ground), and fire crackle
  audio (the pack has none).

### The Forest Fire pack in UE 5.8

The pack was made for UE 4.27 (most assets saved in 4.10). It loads in UE 5.8
without errors; checked on 2026-11-10:

| What | Effect | VIVID 2 |
|---|---|---|
| 16 fire and smoke effects are Cascade (`PS_*`); 9 use GPU particles and vector fields | Rendered, but reduced with distance from the player view, not the drone cameras: gone 450 m from the start. `PS_FireLarge_01` also has an inverted fixed bounding box and both LOD distances at 0 | Not used. VIVID 2 builds its own Niagara effects from the pack's flipbooks |
| Fire and smoke materials were made for Cascade | Without "used with Niagara sprites" a cooked build draws the default material | `tools/build_fire_assets.py` sets the flag (local change to the pack) |
| 4 base materials use UE4 tessellation (`M_BurningGround_01`, `M_Basic_01`, `M_3MatVertBlend_01`, `M_Foliage_Trunk_01`) | The displacement is dropped: rocks, stumps and trunks are a little flatter | Accepted |
| 28 empty 534-byte packages (e.g. `PS_SmokeLarge_Liot`, most `MI_*` in `BaseMaterials*`) | Nothing references them | Never referenced |
| 13 meshes log "recomputing physics on load" | The cook is not deterministic until they are resaved | Resave them once (local) |
| `T_Flip_Fire_02_D` is 32 × 32 px | Useless as a flipbook | Not used |
| No burnt tree meshes, no sound | — | Burnt material instances on the same meshes |
| The demo maps use the deprecated AtmosphericFog and the UE4 sky | Only the demo maps | Not used |

## Disaster maps (Phase 2)

Each disaster map is described by one layout, `SourceData/<map>/layout.json`: its regions (a
slide scar, a deposit, a flood zone), roads, buildings and terrain edits. Two commands turn it
into everything else ([damage.md](damage.md)):

- `vivid2 terrain build --like <base map> --layout ... --stage pre|post`: the heightmap to import
  in Landscape mode, the terrain model, and for `post` the pre-event terrain the stack's DEM is
  made from. `--like` keeps every vertex the edits leave alone identical to the base map's.
- `vivid2 damage build --map <map>`: the damage ground truth, the surface model (roofs, water),
  the region masks the landscape material and PCG graphs read, and the actor placements.

A map before the event (e.g. LandslideBefore: ForestSAR's slope with the road bench) and after
it (Landslide) are separate levels with separate landscapes, since a landscape can't be swapped
at run time. The test map, TestYard, is built the same way at small scale
(`scenarios/landslide/testyard_damage_01.jsonc`).

Each new map goes into a build group in `unreal/VIVID2Env/package_groups.json` (the landslide
maps join `forest`, whose trees they share) and into `MapsToCook`; `tests/unit/test_packaging.py`
checks the two agree.

### Landslide and LandslideBefore

ForestSAR's site (Lishan) with a two-lane mountain road, R01, cut along the 2188 m contour
(Unreal Z 203 m) across the south-east slope, four houses and an orchard shed beside it, and the
forest cleared from the road, its cut and fill faces and the house yards. In **Landslide** a slab
of the steep slope above the road failed: the scar is stripped to raw soil and rock, and the
debris (soil, boulders, broken trunks) buried about 70 m of road (segments R01-04 and R01-05 blocked),
crushed house B03 and the shed B05, piled against farmhouse B02 (its slope end caved in) and ran
220 m down the slope. **LandslideBefore** is the same slope before the slide: the pre-event twin
and a null control (nothing damaged).

- **Source:** `SourceData/Landslide/layout.json` (regions, road, buildings, terrain edits, the
  spots kept clear for people and clues, the cars). Terrains:
  `vivid2 terrain build --map LandslideBefore|Landslide --like ForestSAR --layout ... --stage pre|post`;
  the ground truth: `vivid2 damage build --map Landslide`.
- **Levels:** copies of ForestSAR with their own markers and landscapes (heightmaps imported by
  hand), built in the editor by `tools/build_landslide.py` in steps: `level`, `check` (400
  landscape traces against the terrain model), `masks`, `forest` (a copy of ForestSAR's PCG graph
  that drops everything on cleared ground), `ground` (asphalt, bare soil, scar and debris in the
  landscape material), `houses` (box houses, placeholders until a building pack; damaged as
  graded), `debris` (Forest Fire pack boulders, rocks, logs and stumps over the debris) and `cars`.
- **Cars** (Velocity Cars Pack, no drivers or passengers, lights off): four parked by the houses
  and one on the road before the slide; after it, B03's car lies on its side down the debris,
  one is sunk in the debris on the road, and two have stopped either side of the blockage, the
  east one with its driver's door open.
- **Scenarios:** `scenarios/landslide/landslide_sar_01.jsonc` (noon) and `_02` (dusk, light
  fog), campaign `campaigns/landslide_v1.jsonc`. The slide falls 185 m from the scar to its toe,
  so they fly a terrain-following survey (`search_area.altitude_ref: "terrain"`, see
  [missions.md](missions.md)).
