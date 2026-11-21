"""Write a map layout's region masks (Unreal textures) without building its ground truth.

    python tools/region_mask.py unreal/VIVID2Env/SourceData/<map>/layout.json [--out DIR]

For each region in the layout (vivid2.damage.regions): an 8-bit greyscale PNG over the whole map,
128 + 127 b (never 0: PCG's Sample Texture drops points that read 0), pixel [row, col] =
[east, north]. ``vivid2 damage build`` writes the same masks along with the ground truth; this is
for previewing a layout while it is being drawn. Import as sRGB off, no mipmaps, Grayscale.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from vivid2.damage import REGION_THRESHOLD
from vivid2.damage import regions as rg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("layout")
    ap.add_argument("--out", help="default: masks/ next to the layout")
    args = ap.parse_args(argv)
    path = Path(args.layout)
    layout = json.loads(path.read_text(encoding="utf-8"))
    out = Path(args.out) if args.out else path.parent / "masks"
    out.mkdir(parents=True, exist_ok=True)
    for name, field in rg.fields(layout).items():
        p = out / f"{name}.png"
        cv2.imwrite(str(p), rg.to_png(field))
        print(f"{p}: {100.0 * (field >= REGION_THRESHOLD).mean():.2f} % of the map inside")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
