"""Named detector weights for the YOLO baseline, downloaded on first use.

Only permissively licensed weights are listed, so results with them can be published with
VIVID 2. Any other ONNX detector can be passed as a file path instead of a name.

``threshold`` is the detection score that counts as a find. Calibrating it on frames the
benchmark does not use (tools/calibrate_detector.py on the ForestSAR_Open/_Dense survey-dip
frames, 2026-11-07) gave no usable operating point. YOLOX-s found the lying red-jacket hiker
in 0 of 12 survey frames and at most 3 of 19 low frames. So yolox-s keeps the conventional
0.3, untuned on the benchmark.

For information only, on the benchmark scenarios' own frames at 0.3:
- survey frames (55 m): recall 0/104;
- low frames (close looks and dips): recall 18/54, precision 0.47.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

from vivid2.unreal_setup import cache_root, sha256_file

KNOWN_DETECTORS: Dict[str, Dict[str, Any]] = {
    "yolox-s": {
        "url": "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx",
        "sha256": "c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063",
        "file": "yolox_s.onnx",
        "format": "yolox",
        "license": "Apache-2.0",
        "source": "YOLOX-s (Megvii), COCO-trained",
        "threshold": 0.3,
    },
}
DEFAULT_DETECTOR = "yolox-s"


def detector_cache() -> Path:
    return cache_root().parent / "detectors"


def resolve_detector(name_or_path: str, progress: Callable[[str], None] = print) -> Tuple[str, Dict[str, Any]]:
    """The weights' local path and their registry entry (``{"name": ...}`` only for a plain file).

    A known name is downloaded into the VIVID 2 cache (``VIVID2_CACHE`` or %LOCALAPPDATA%\\VIVID2)
    on first use and checked against its SHA-256 every time.
    """
    entry = KNOWN_DETECTORS.get(name_or_path)
    if entry is None:
        path = Path(name_or_path)
        if not path.is_file():
            raise ValueError(f"detector {name_or_path!r} is neither a file nor one of {sorted(KNOWN_DETECTORS)}")
        return str(path), {"name": path.stem, "sha256": sha256_file(path)}
    dest = detector_cache() / entry["file"]
    if not dest.is_file() or sha256_file(dest) != entry["sha256"]:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".part")
        progress(f"downloading {name_or_path} ({entry['license']}) from {entry['url']}")
        urllib.request.urlretrieve(entry["url"], tmp)
        got = sha256_file(tmp)
        if got != entry["sha256"]:
            tmp.unlink()
            raise ValueError(f"{name_or_path}: downloaded file has SHA-256 {got}, expected {entry['sha256']}")
        tmp.replace(dest)
    return str(dest), {"name": name_or_path, **entry}
