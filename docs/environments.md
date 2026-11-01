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
4. Put a PlayerStart at the world origin (0, 0, 0). Project AirSim measures NED
   coordinates from the Unreal world origin, not from the PlayerStart.

## Targets (hikers)

Scenarios spawn targets by **asset name** through `spawn-objects`. Project
AirSim looks names up in the asset registry: static meshes, skeletal meshes and
blueprints.

- Import a skeletal human (for example UE5 Manny/Quinn, or MetaHumans baked to
  skeletal meshes) and name it `SKM_Manny`. Add clothing-colour variants such as
  `SKM_Hiker_RedJacket` and `SKM_Hiker_BlueHoodie`. Then use those names in the
  scenario `asset` fields.
- Poses such as lying, sitting or waving can be separate skeletal mesh assets or
  blueprints with a fixed animation pose.
- The origin of each asset must be at ground level (the feet), because scenario
  `ned.z` is the ground height.
- Make sure the asset gets a segmentation ID. The scene's `segmentation` block
  initialises IDs by owner name.

On Blocks, `1M_Cube` scaled to 0.5 × 0.5 × 1.7 stands in for a person. This
is only good for pipeline bring-up. It is meaningless for evaluating a VLM.

## ForestSAR (Phase 1)

- **Terrain:** import a real DEM (e.g. Taiwan's 20 m DTM or Copernicus GLO-30) as
  a Landscape heightmap, or use Cesium for Unreal with Google Photorealistic 3D
  Tiles for context. Keep the playable area about 1 × 1 km at around 1 m
  resolution.
- **Vegetation:** a Fab conifer/broadleaf forest pack scattered with a **PCG
  graph**, driven by slope and height and excluded from a trail spline. Expose
  density, species mix and seed as PCG parameters, so scenario variants can
  change canopy density.
- **Clutter and distractors:** logs, rocks, stumps and red/yellow objects such
  as backpacks, tents and fallen signs. These are what make false alarms
  measurable.
- **Lighting:** SunPosition plugin (the scenario's `time_of_day`) and volumetric
  fog (the scenario's `weather.fog`).
- **Terrain height:** scenarios currently assume flat ground at
  `search_area.ground_down_m`. For real relief, Phase 1 adds a heightmap
  export so metrics and planners can use true AGL.

## Wildfire (Phase 2)

Niagara fire and smoke emitters packaged as blueprints (`BP_FireCell`,
`BP_SmokeColumn`) are spawned and moved by `vivid2.hazards.fire_spread`, a
cellular-automaton spread model driven by wind and slope. The fire front
becomes a moving no-fly polygon in the safety metrics.

## Landslide (Phase 2)

There are two terrain states, before and after (debris meshes, a displaced
road section, tilted trees). The landslide hazard swaps them in, and exports
the change mask as ground truth for change-detection IoU.
