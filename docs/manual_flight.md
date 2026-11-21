# Manual flight

`vivid2 fly` puts a drone on a VIVID 2 map that **you** fly with an RC radio, a gamepad or the
keyboard. Use it to look around a map, check a scenario's hikers and decoys from the air, or fly a
search by hand to compare with an agent. Manual flights are not benchmark runs: nothing is recorded
or scored, and the benchmark's robot config, scenes and maps are not changed.

## What it needs

Python only builds the scene and loads it into the sim. The drone is flown inside Unreal by a
**pilot plugin**, `FpvPilot`. It is not part of VIVID 2 and has to be obtained separately. It
provides:

- the flight controller, which runs in Project AirSim's physics loop;
- the radio or gamepad input and its calibration;
- an on-screen display and a setup menu.

It plugs into two hooks in the Project AirSim plugin, both in
`patches/projectairsim-pilot-hooks.patch`:

- an **external flight controller**: a robot whose controller is `external-controller-api` is flown
  by a controller the host registers;
- an **RC input mailbox**, plus real-time pacing of the sim clock.

The hooks do nothing unless a robot uses `external-controller-api`. Benchmark robots use
`simple-flight-api`, so benchmark runs behave the same with or without them.

Setup needs a Project AirSim checkout, so the patch can be applied, plus the plugin folder:

```powershell
git -C ..\ProjectAirSim worktree add ..\ProjectAirSim-vivid2 -b vivid2-hooks ue5.8-support   # or any checkout
vivid2 unreal setup --engine 5.8 --airsim-root ..\ProjectAirSim-vivid2 --apply-patch `
    --copy-simlibs-from ..\PASBlocks\Plugins\ProjectAirSim\SimLibs `
    --pilot-plugin <folder with FpvPilot.uplugin>
vivid2 unreal status            # "pilot_hooks": true, "pilot_plugin": "<folder>"
```

What each option does:

- `--apply-patch` applies the UE 5.8, team weather and pilot hooks patches where they are missing.
- `--copy-simlibs-from` copies prebuilt SimLibs into a fresh checkout, so SimLibs don't have to be
  built.
- `--pilot-plugin` links the plugin into `unreal/VIVID2Env/Plugins` (gitignored) and enables it in
  `VIVID2Env.uproject`. Later setups keep it; `--no-pilot-plugin` removes it.

After setting up, close the editor and build it once (the Project AirSim plugin is compiled from
the checkout):

```powershell
& "C:\Program Files\Epic Games\UE_5.8\Engine\Build\BatchFiles\Build.bat" VIVID2EnvEditor Win64 Development `
    -Project="$PWD\unreal\VIVID2Env\VIVID2Env.uproject" -WaitMutex
```

Packaged builds: `tools\package_env.ps1` refuses to package while the pilot plugin is enabled,
because it is not part of VIVID 2. Run `vivid2 unreal setup ... --no-pilot-plugin` to package a
build to share.

## Flying

There are two ways to start, in the editor or in a game window.

**In the editor:**

1. Open the map (for example ForestSAR) and press Play.
2. Load the drone:

```powershell
vivid2 fly --map ForestSAR                     # the 5-inch FPV quad at the launch clearing
vivid2 fly --scenario scenarios/forest_sar/forest_clearing_01.jsonc --drone quad
```

**In a game window:** the same commands with `--launch`, which starts the map in a game window
first and waits for it:

```powershell
vivid2 fly --map ForestFire --drone quad --launch
```

| Option | |
|---|---|
| `--map` | `ForestSAR`, `ForestSAR_Open`, `ForestSAR_Dense` or `ForestFire`. With `--scenario`: fly the scenario on this map instead of its own. |
| `--scenario` | A scenario file. The drone starts at its `dut_start`, with its hikers, decoys, fires and weather (time of day, fog, wind), the first sweep combination and `--seed` (target jitter). |
| `--drone fpv5` | A 5-inch freestyle FPV quad (0.65 kg). It flies in acro, horizon and angle modes, with an FPV camera (120°, 30° uptilt) and a chase camera. Its robot config is `sim_config/pilot/robot_fpv5.jsonc`. |
| `--drone quad` | VIVID 2's benchmark quad (`robot_vivid_quad.jsonc`) with the pilot's flight controller instead of simple_flight. It flies in angle mode only. Tab cycles the chase, search (Payload) and forward cameras, and the search camera tilts. |
| `--no-targets` | Leave the scenario's hikers and decoys out. |
| `--launch`, `--wait`, `--unreal-args` | Start a game window first. `--wait` waits for the game to exit; `--unreal-args` passes extra Unreal arguments. |
| `--print-scene` | Print the scene JSON and stop (no sim needed). |

The drone starts landed on the ground at its start point.

The scene runs the sim at a 2 ms step, and the plugin keeps the sim clock on wall-clock time. The
air at the forest maps' launch clearing (about 2,240 m) is 20 % thinner than at sea level, and the
flight controller is told so: a spring-centred throttle stick rests near the hover there. The
benchmark quad has a thrust-to-weight ratio of only about 1.4 at that height, so it climbs gently.

Loading a benchmark run afterwards (`vivid2 run`) in the same Play session replaces the scene. The
plugin then steps aside and puts its frame-rate and clock settings back.

## Controls

| Key / button | |
|---|---|
| Space / RB | arm toggle (throttle fully down first) |
| M / LB | flight mode: acro, horizon, angle (the 5-inch only) |
| R / Y | back to the start point, disarmed |
| Tab | next camera |
| F / X | search camera (quad): straight down, 45°, horizon |
| PgUp / PgDn | tilt the search camera 5° (quad) |
| F2 / Start | setup menu: receiver, mapping, calibration, rates, modes, PID tuning (Esc in a game window; in the editor Esc stops Play) |
| K | calibrate the connected radio or gamepad |
| J | input source: auto, joystick, Xbox gamepad, keyboard |
| H | help |
| Keyboard | W/S throttle, X cut, A/D yaw, arrows pitch and roll |

The plugin keeps its controller mappings and your flight-controller changes in
`unreal/VIVID2Env/Saved/FpvPilot/`.

## Checks

The following flies a scripted test with the drone and exits:

```
vivid2 fly ... --launch --wait --unreal-args="-FpvPilotFlightTest -FpvPilotQuitAfterTest -abslog=<log>"
```

- **5-inch:** arm, climb, acro roll and pitch steps, land, hop and reset.
- **Quad:** half-stick angle steps and a yaw step instead of the acro steps.

The log line `FpvPilotFlightTest: RESULT PASS` (or `FAIL`) reports the result; Unreal's exit code
doesn't. The trace is in `Saved/Logs/FpvPilotFlightTest.csv` and a screenshot of the view in
`Saved/Screenshots/FpvPilotFlightTest_fpv.png`. Add `-FpvPilotMenuShots` instead to save a
screenshot of each setup page.

`--launch` runs the editor binary in game mode. It compiles textures synchronously
(`-DPCVars=Editor.AsyncTextureCompilation=0`), because UE 5.8's asynchronous texture compiler
crashed (TextureCompiler.cpp:355) while loading a drone there. The first launch takes longer.
