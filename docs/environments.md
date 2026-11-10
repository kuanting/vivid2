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

- **Users** get a packaged `VIVID2Env` build (cooked, in `.pak` files), made with
  `tools\package_env.ps1` and published as a release download. They need no
  Unreal or Fab account.
- **Map authors** who want to edit the maps in the editor must get each pack
  below from Fab themselves and install it at the listed folder.

| Pack | Folder | Used for | Notes |
|---|---|---|---|
| Forest Fire (Fab) | `Content/ForestFire/` | ForestSAR vegetation (birch trees, shrubs, ferns, grass, stumps, logs, rocks); the flipbook textures and materials of VIVID 2's fire effects; ForestFire's burnt ground texture and the parents of its burnt material instances | Made for UE4; see [the pack in UE 5.8](#the-forest-fire-pack-in-ue-58). |

## Targets (hikers)

Scenarios spawn targets by **asset name** through `spawn-objects`. Project
AirSim looks names up in the asset registry: static meshes, skeletal meshes and
blueprints.

**Built (v1):** 16 blueprints in `/Game/VIVID2/Targets`, named
`VIVID2_Hiker_<Outfit>_<Pose>`.
- **Outfits:**

  | Outfit | Body | Jacket | Trousers |
  |---|---|---|---|
  | `RedJacket` | Manny | red | dark |
  | `YellowJacket` | Manny | yellow | grey |
  | `BlueHoodie` | Quinn | dark blue | blue |
  | `GreenJacket` | Manny | olive (low contrast) | khaki |

- **Poses:** `Standing`, `Crouching`, `LyingBack`, `LyingFront`.
- **Construction:** each blueprint is one skeletal mesh (the UE5 mannequin) held
  in a fixed animation pose. The jacket and trousers use flat matte colours
  (`M_VIVID2_Clothing`); the mannequin's own material leaves white panels.
- **Lying poses:** the template has no lying animation (its "death" clips are
  short staggers ending in a crouch), so the lying poses are the idle pose
  rotated flat and lifted 15 cm.
- **Pivot:** the pivot is at the feet, so scenario `ned.z` = 0 puts the hiker
  on the terrain.
- **Size from the air:** from 45–60 m a lying hiker is about 20–25 payload
  pixels long; a standing one shows only head and shoulders.
- **Cooking:** `DirectoriesToAlwaysCook` includes `/Game/VIVID2/Targets`,
  because hikers are spawned by name and no map references them.
- **Mannequin content:** the mannequins come from the engine's Third Person
  template content (`Content/Characters`, not committed). Add it with
  Add → Add Feature or Content Pack → Third Person, or copy
  `<UE>/Templates/TemplateResources/High/Characters/Content` to
  `Content/Characters`.
- **Upgrade path:** realistic clothed people (a Fab human pack, or MetaHumans
  once the MetaHuman optional content is installed) can replace the mannequins
  under the same asset names.

Rules for any target asset:
- The origin must be at ground level (the feet, or the lowest point when
  lying), because scenario `ned.z` is the ground height.
- Make sure the asset gets a segmentation ID. The runner sets one per target
  by name right after spawning.

On Blocks, `1M_Cube` scaled to 0.5 × 0.5 × 1.7 stands in for a person. This
is only good for pipeline bring-up. It is meaningless for evaluating a VLM.

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
  reference planner still flies at a fixed altitude; terrain following is M3.4.

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

## Landslide (Phase 2)

There are two terrain states, before and after (debris meshes, a displaced
road section, tilted trees). The landslide hazard swaps them in, and exports
the change mask as ground truth for change-detection IoU.
