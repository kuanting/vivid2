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
3. Create maps under `/Game/VIVID2/Maps/`: `ForestSAR`, `Wildfire`, `Landslide`.
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
| Forest Fire (Fab) | `Content/ForestFire/` | ForestSAR vegetation (birch trees, shrubs, ferns, grass, stumps, logs, rocks); Wildfire fire and smoke | Made for UE4. Checked in UE 5.8 (2026-11-02): all three maps load with 0 errors. The meshes log "recomputing physics on load" until they are resaved. Fire and smoke are Cascade particle systems (`PS_*`), which UE 5.8 still renders. |

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
  120,000 grass clumps. Understory and grass have no collision. The volume is
  set to **Generate On Demand**, so it never regenerates at runtime: generate it
  in the editor and save the level, which then stores every instance. The
  Dense/Open variants are copies of the level with different graph parameters.
  Not done yet: thinning trees on steep slopes, and keeping a trail
  spline clear.
- **Clutter and distractors:** logs, rocks, stumps and red/yellow objects such
  as backpacks, tents and fallen signs. These are what make false alarms
  measurable.
- **Fog:** the scenario's `weather.fog` goes to Project AirSim's weather
  effects. The level's ExponentialHeightFog is only the base haze.
- **Terrain height:** the runner snaps targets and the drone start onto
  `terrain_dtm.npz`, and the safety metrics measure height above it. The
  reference planner still flies at a fixed altitude; terrain following is M3.4.

## Wildfire (Phase 2)

Niagara fire and smoke emitters packaged as blueprints (`BP_FireCell`,
`BP_SmokeColumn`) are spawned and moved by `vivid2.hazards.fire_spread`, a
cellular-automaton spread model driven by wind and slope. The fire front
becomes a moving no-fly polygon in the safety metrics.

## Landslide (Phase 2)

There are two terrain states, before and after (debris meshes, a displaced
road section, tilted trees). The landslide hazard swaps them in, and exports
the change mask as ground truth for change-detection IoU.
