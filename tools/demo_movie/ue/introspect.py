"""Editor-side: list the Python API of some ``unreal`` classes (developer aid). Prints one JSON line."""
import json
import sys
from pathlib import Path

import unreal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ue_exec import script_args  # noqa: E402

args = script_args()
out = {}
for name in args.get("classes", []):
    obj = getattr(unreal, name, None)
    if obj is None:
        out[name] = None
        continue
    pat = args.get("filter", "")
    out[name] = [m for m in dir(obj) if not m.startswith("_") and pat in m.lower()]
for expr in args.get("doc", []):
    try:
        out[expr] = str(eval(expr, {"unreal": unreal}).__doc__)[:1500]
    except Exception as e:  # noqa: BLE001
        out[expr] = f"ERR {e}"
print(json.dumps(out))
