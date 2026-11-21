"""Editor side: open a level and start/stop Play-In-Editor (for the real VIVID 2 run). One JSON line.

args: {"action": "open"|"play"|"stop"|"status", "level": "/Game/..."}
"open" refuses if any map package is dirty (it never saves anything).
"""
import json
import sys
from pathlib import Path

import unreal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ue_exec import script_args  # noqa: E402

A = script_args()
les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
ues = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem)
act = A.get("action", "status")
out = {"action": act}
if act == "open":
    if les.is_in_play_in_editor():
        raise RuntimeError("stop PIE first")
    dirty = [p.get_name() for p in unreal.EditorLoadingAndSavingUtils.get_dirty_map_packages()]
    if dirty:
        raise RuntimeError(f"unsaved map packages: {dirty} (not saving them)")
    out["ok"] = bool(les.load_level(A["level"]))
elif act == "play":
    if not les.is_in_play_in_editor():
        les.editor_request_begin_play()
    out["ok"] = True
elif act == "stop":
    if les.is_in_play_in_editor():
        les.editor_request_end_play()
    out["ok"] = True
out["pie"] = bool(les.is_in_play_in_editor())
w = ues.get_editor_world()
out["level"] = w.get_outermost().get_name() if w else None
print(json.dumps(out))
