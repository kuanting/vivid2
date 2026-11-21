"""Editor-side probe (P0): what the film tools need, present or not. Prints one JSON line.

Runs inside the Unreal editor (stdlib + ``unreal``) via ``film.py probe``.
"""
import json
import sys
from pathlib import Path

import unreal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ue_exec import script_args  # noqa: E402

args = script_args()
out = {"engine": unreal.SystemLibrary.get_engine_version()}

# Movie Render Queue + Sequencer scripting
out["mrq"] = all(hasattr(unreal, n) for n in (
    "MoviePipelineQueueSubsystem", "MoviePipelinePIEExecutor", "MoviePipelineImageSequenceOutput_PNG",
    "MoviePipelineAntiAliasingSetting", "MoviePipelineGameOverrideSetting", "MoviePipelineDeferredPassBase"))
out["sequencer"] = all(hasattr(unreal, n) for n in (
    "LevelSequence", "MovieSceneSkeletalAnimationTrack", "MovieScene3DTransformTrack", "CineCameraActor",
    "MovieSceneCinematicShotTrack", "MovieSceneCameraCutTrack"))

reg = unreal.AssetRegistryHelpers.get_asset_registry()


def exists(path):
    return unreal.EditorAssetLibrary.does_asset_exist(path)


out["assets"] = {p: exists(p) for p in args.get("assets", [])}

# animation clips by name anywhere under /Game
want = set(args.get("anims", []))
found = {}
if want:
    flt = unreal.ARFilter(class_paths=[unreal.TopLevelAssetPath("/Script/Engine", "AnimSequence")],
                          package_paths=["/Game"], recursive_paths=True)
    for a in reg.get_assets(flt):
        name = str(a.asset_name)
        if name in want:
            found.setdefault(name, []).append(str(a.package_name))
out["anims"] = found

# the hiker BPs: one SkeletalMeshComponent? its animation mode
hikers = {}
for bp in args.get("hikers", []):
    cls = unreal.EditorAssetLibrary.load_blueprint_class(bp)
    if cls is None:
        hikers[bp] = None
        continue
    cdo = unreal.get_default_object(cls)
    info = {"class": cls.get_name()}
    try:
        comps = unreal.SubobjectDataSubsystem  # noqa: F841 (presence only)
    except Exception:
        pass
    hikers[bp] = info
out["hikers"] = hikers

# the open level and its SunSky
world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
out["level"] = world.get_path_name() if world else None
actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors()
out["sunsky"] = [a.get_actor_label() for a in actors if "SunSky" in a.get_actor_label() or "SunSky" in a.get_class().get_name()]
out["n_actors"] = len(actors)
print(json.dumps(out))
