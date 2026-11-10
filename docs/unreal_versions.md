# Unreal Engine versions (5.7 and 5.8)

VIVID 2 itself (the Python framework, scenarios and scoring) doesn't depend
on the engine version. The engine only matters for the Unreal side: the Project
AirSim plugin and the `unreal/VIVID2Env` project.

| Engine | Open/author VIVID 2 maps in the editor | Run evaluations | How to get the Project AirSim plugin |
|---|---|---|---|
| **5.8** (recommended) | Yes, and the MCP toolsets are enabled | Editor (Play) or packaged build | Prebuilt zip, patch your checkout, or a local build |
| **5.7** | No: maps are saved in 5.8. Blocks-based bring-up only | Packaged build, or a 5.7 Blocks project | Upstream Project AirSim, as-is |

Maps are authored in 5.8 because the 5.8 editor has the MCP toolsets. Unreal
can't open assets saved by a newer engine, so 5.7 users run **packaged
VIVID2Env builds**. Those need no Unreal install at all.

## Setting up `unreal/VIVID2Env`

`vivid2 unreal setup` does three things:
- links the plugin into `unreal/VIVID2Env/Plugins` (directory junctions)
- renders `VIVID2Env.uproject` for your engine from
  `VIVID2Env.uproject.template.json` plus that engine's profile in
  `unreal/engines.json`. The 5.8-only MCP plugins are listed only for 5.8,
  because 5.7 would refuse to open a project that lists them.
- records what it did in `unreal/VIVID2Env/.vivid2-setup.json`. Each run's
  `meta.json` includes this record.

Pick one plugin source:

```powershell
# A. Download the prebuilt UE 5.8 plugin (about 3 GB, Google Drive / OneDrive)
pip install -e ".[setup]"
vivid2 unreal setup --engine 5.8 --download
#    If the scripted download is blocked (Drive quota or virus-scan page), download it in a
#    browser and pass the file. The SHA-256 is checked against unreal/engines.json:
vivid2 unreal setup --engine 5.8 --plugin-zip "%USERPROFILE%\Downloads\ProjectAirSim-Plugins-UE5.8-Win64.zip"

# B. Upgrade your own Project AirSim checkout (build its SimLibs with build.cmd first)
vivid2 unreal setup --engine 5.8 --airsim-root ..\ProjectAirSim --apply-patch
#    The patch is patches/projectairsim-ue5.8.patch. It touches 6 files and only adds
#    version macros, so the same source still builds for UE 5.7.
#    UE compiles the plugin on first open; that needs Visual Studio 2022.

#    --apply-patch also applies patches/projectairsim-team-weather.patch (any engine), which
#    teams of drones need: without it each drone's camera also sees the other drones' fog.

# C. Use any plugin folder you already built (e.g. another 5.8 project)
vivid2 unreal setup --engine 5.8 --plugin-dir ..\PASBlocks\Plugins

# UE 5.7
vivid2 unreal setup --engine 5.7 --airsim-root ..\ProjectAirSim

vivid2 unreal status
```

Setup compares the plugin binaries' BuildId with the installed engine's. If
they differ, Unreal offers to rebuild the plugin when the project opens.

Downloads are cached in `%LOCALAPPDATA%\VIVID2\plugins\ue<version>\<sha8>\`.
Set `VIVID2_CACHE` to put them somewhere else.

## Publishing the prebuilt plugin (maintainers)

```powershell
vivid2 unreal pack-plugin --plugin-dir ..\PASBlocks\Plugins --engine 5.8 --out dist --strip-pdb
```

- This zips `ProjectAirSim/` and `Drone/`. It leaves out `Intermediate/` (about
  0.8 GB, rebuilt locally), every `.git/`, and with `--strip-pdb` the debug symbols.
- It writes `dist/<zip>.sha256` and prints the `plugin_zip` block for
  `unreal/engines.json`.
- Upload the zip and share it as "anyone with the link".
  - **Google Drive:** use `provider: "gdrive"` with the share link. Downloads go
    through `gdown`. Drive may throttle popular large files, so keep the browser
    + `--plugin-zip` fallback documented.
  - **OneDrive:** use `provider: "onedrive"` with the share link. `download=1`
    is added automatically.
- Commit the updated `unreal/engines.json`.

## Packaged builds

A packaged build runs the benchmark without Unreal: the cooked maps, the
simulator and nothing else. Use it on every machine that only runs campaigns.

### Making one

```powershell
vivid2 unreal setup --engine 5.8 --download      # once
powershell -ExecutionPolicy Bypass -File tools\package_env.ps1 -Engine 5.8 -Zip
# -> build\VIVID2Env-ue5.8-Win64\Windows\VIVID2Env.exe       (2.2 GB)
# -> dist\VIVID2Env-ue5.8-Win64-<commit>.zip  + .sha256      (to copy to other machines)
```

- About 6 minutes when the editor has already compiled the shaders. The
  editor may stay open: only the game is compiled.
- Cooked: `ForestSAR`, `ForestSAR_Open`, `ForestSAR_Dense` and `ForestFire`
  (`MapsToCook` in `Config/DefaultGame.ini`). Also the content spawned at
  runtime: hikers, decoys and fires, the drone, and Project AirSim's weather
  actor.
- `vivid2_build.json` next to the exe records the VIVID 2 commit and any
  uncommitted changes to the project's content or config.
- The build contains third-party content (Fab packs) in cooked form. Share
  the zip as a binary download; never commit it.

### Running one

On any 64-bit Windows 10/11 machine with a DirectX 12 GPU:

1. Unzip. On a fresh machine, run
   `Engine\Extras\Redist\en-us\vc_redist.x64.exe` once.
2. **Smart App Control must be off.** It blocks the unsigned game exe: the
   launcher shows "Couldn't start ... CreateProcess() returned 11c7". Check it
   with
   `(Get-ItemProperty HKLM:\SYSTEM\CurrentControlSet\Control\CI\Policy).VerifiedAndReputablePolicyState`
   (0 = off). Turn it off in Windows Security → App & browser control →
   Smart App Control settings. Windows may not let you switch it back on
   without a reset. A code-signed build would avoid this; none exists yet.
3. Start one map. ForestSAR is the default, and it is ready in about 10 s:
   ```powershell
   VIVID2Env.exe /Game/VIVID2/Maps/ForestSAR_Open -windowed -ResX=1280 -ResY=720
   ```
   Use `-RenderOffScreen` for no window. Runs whose map is not the open one are
   skipped by the preflight check: restart the build with the next map and run
   the campaign again (finished runs are skipped).
4. The simulator listens on every network interface: port 8989 for topics and
   8990 for services. Use `-topicsport=` and `-servicesport=` to run two on one
   machine. On first start, Windows Firewall asks to allow `VIVID2Env`. Allow
   it if the runner is on another machine, then use
   `vivid2 run ... --address <that machine's IP>`.

Checked on 2026-11-04 against the editor, flying the same runs in both.
Five surveys covered dusk with fog 0, 0.02 and 0.05, and noon with fog 0.2 and
0.4. One dip run used dusk fog 0.05, and the oracle flew `ForestSAR_Open` and
`ForestSAR_Dense`.
- **Identical:** depth, hiker pixels (packaged/editor ratio 0.99–1.00) and
  hiker contrast (within 1.4 ΔE).
- **Dip run:** both made 12 dips with 0 collisions, and their lowest height
  above ground was 11.84 m and 11.87 m. Both saw the hikers in the same 6 and
  7 low frames.

Settings that keep the images the same:
- **Render quality** is fixed at Epic in `Config/DefaultGameUserSettings.ini`,
  which is what the editor uses. A machine therefore can't drop to lower
  quality on its own.
- **Shader model** is SM5 in both the editor and the build.
- **Frame rate** is capped at 30 fps (`t.MaxFPS` in `Config/DefaultEngine.ini`).
  Camera auto-exposure adapts by speed × frame time each time a camera
  captures. Uncapped, a packaged build runs at hundreds of fps, and its first
  frames after the climb came out too dark or too bright for about 10 s.
  Much faster adaptation is no fix: the exposure then oscillates. The editor
  renders only a few frames per second in the background, so it adapts almost
  at once. Benchmark results should therefore come from packaged builds; use
  the editor for authoring.
- **Exposure warm-up at survey height.** Before the mission clock starts, the
  runner captures from each camera until its exposure is steady. On the
  ground, the payload camera sits centimetres above the terrain and renders
  flat grey, so it is lifted to the survey height for its warm-up and then put
  back. Without this, the first three survey frames at dusk were too dark
  (brightness 23, 68 and 113 against 125 for the rest of the run). With it,
  they match the rest (126, 125 and 127). `meta.json` records each warm-up
  under `exposure`.
- **Lumen's stored lighting range** is moved down 6 stops
  (`r.EyeAdaptation.CachedLightingPreExposure=-2`, range −14 to +6). At dusk
  the cameras expose near −11, below the default range of −8 to +12. Unreal
  then warns "Cached lighting in Lumen and real-time sky capture lighting is
  going to be clipped", and bounce light and sky light lose precision. The
  editor reads this setting when it starts.

## Upstream status

Upstream Project AirSim `main` supports UE 5.8 itself since `e055190`, so a
current checkout needs no 5.8 patch; `patches/projectairsim-ue5.8.patch` is for
older checkouts.

The team weather fix (`patches/projectairsim-team-weather.patch`, one file:
`WeatherLib.cpp`) is submitted upstream as
[#243](https://github.com/iamaisim/ProjectAirSim/pull/243); until it is merged,
`--apply-patch` adds it. Single-drone runs don't need it; their images are the same
with or without it. See `patches/UPSTREAM_PR.md`.
