"""Editor-side: material slots and bounds of static meshes (developer aid). Prints one JSON line."""
import json
import sys
from pathlib import Path

import unreal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ue_exec import script_args  # noqa: E402

out = {}
for path in script_args().get("meshes", []):
    m = unreal.load_asset(path)
    if m is None:
        out[path] = None
        continue
    slots = []
    for i, sm in enumerate(m.get_editor_property("static_materials")):
        mat = sm.get_editor_property("material_interface")
        slots.append({"i": i, "slot": str(sm.get_editor_property("material_slot_name")),
                      "material": mat.get_path_name() if mat else None})
    b = m.get_bounds()
    out[path] = {"slots": slots, "extent_cm": [b.box_extent.x, b.box_extent.y, b.box_extent.z],
                 "origin_cm": [b.origin.x, b.origin.y, b.origin.z]}
print(json.dumps(out))
