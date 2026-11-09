# Project AirSim changes VIVID 2 needs, and their upstream status

| Patch | What it does | Upstream (iamaisim/ProjectAirSim) |
|---|---|---|
| `projectairsim-ue5.8.patch` | Builds the plugin on UE 5.8 (`UE_IS_5_7_OR_LATER` macros, 6 files) | Not needed with current `main`: upstream added UE 5.8 support itself (`e055190`). Older checkouts still need it. |
| `projectairsim-team-weather.patch` | Each robot's cameras see only its own weather actor, so a team's fog looks the same to every drone as to a lone one (`WeatherLib.cpp`) | Submitted as [#243](https://github.com/iamaisim/ProjectAirSim/pull/243), open (2026-11-09). |

`vivid2 unreal setup --airsim-root ... --apply-patch` applies whichever of the two a checkout
lacks.

## #243: hide other robots' weather actors from each robot's cameras

`UWeatherLib::initWeather` spawns one `WeatherActor` per robot and attaches it 5 m above the
robot. Its effects follow the robot (e.g. the particle fog, `P_Weather_FogVolumetric_Main`).
Every robot's scene captures rendered every weather actor, so in a scene with N robots each
camera looked through N robots' fog: images got hazier with every robot added.

The fix adds each other robot's weather actor to the `HiddenActors` of every
`USceneCaptureComponent2D` owned by a robot. Single-robot scenes are unchanged.

Tested on a UE 5.8 packaged build at dusk with fog 0.02: Drone1's payload image, hovering 10 m
from three other drones, matches the one-drone scene (grey standard deviation 16.41 vs 16.33,
Laplacian variance 666 vs 661). Before the fix, a hiker's contrast in four-drone runs was about
half that in one-drone runs at the same pose. Not built on UE 5.2 or 5.7.
