# Draft PR for iamaisim/ProjectAirSim

**Title:** Support Unreal Engine 5.8 (UE_IS_5_7_OR_LATER)

**Branch:** `ue5.8-support` (fork) → `main`

## Summary
The UE 5.7 code paths are guarded with `#if UE_IS_5_7`, so the plugin doesn't
compile on UE 5.8. This PR:

- adds `UE_IS_5_8` and `UE_IS_5_7_OR_LATER` to `Private/UnrealCompatibility.h`, and
  includes 5.8 in `UE_IS_SUPPORTED`
- replaces `UE_IS_5_7` with `UE_IS_5_7_OR_LATER` in `GPULidar.cpp`,
  `LidarIntensitySceneViewExtension.cpp`, `LidarIntensityShader.h`,
  `UnrealCamera.cpp` and `UnrealLogger.h`

No behaviour changes on UE 5.7 or 5.2: the macro is true on 5.7 exactly as before.

## Testing
- UE 5.8: Blocks builds and runs (`hello_drone.py`, camera and lidar sensors).
- UE 5.7: unchanged code paths.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
