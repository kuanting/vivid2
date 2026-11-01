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

```powershell
vivid2 unreal setup --engine 5.8 --download
powershell -ExecutionPolicy Bypass -File tools\package_env.ps1 -Engine 5.8
# -> build\VIVID2Env-ue5.8-Win64\Windows\VIVID2Env.exe
```

## Upstream status

The UE 5.8 change is on branch `ue5.8-support` of the Project AirSim fork.
Once it's merged upstream, option B won't need `--apply-patch`.
