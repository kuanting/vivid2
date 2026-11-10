"""Make the ForestFire map's burn mask from its burn layout.

    python tools/burn_mask.py [unreal/VIVID2Env/SourceData/ForestFire/burn_layout.json]

Writes burn_mask.png next to the layout: an 8-bit greyscale image over the whole landscape. The
burn amount b (1 burnt, 0 not, blending over ``edge.soft_m`` metres at the burn's edge, which is
roughened by ``edge.noise_m`` of smooth random noise with a fixed ``edge.seed``) is stored as
128 + 127 b, so the image is never 0: PCG's Sample Texture node drops points where a texture reads
0, and point density is clamped to [0, 1]. The forest graph therefore reads density 0.5 (unburnt)
to 1 (burnt), and the landscape material uses b = saturate(2 R - 1).

Pixel [row, col] = [east, north]: both sample it at U = (world X + extent/2) / extent (north),
V = (world Y + extent/2) / extent (east), with world X north and Y east in Unreal units. It is
imported as /Game/VIVID2/Textures/T_VIVID2_ForestFire_BurnMask (sRGB off, no mipmaps, Grayscale).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

DEFAULT = Path(__file__).resolve().parents[1] / "unreal/VIVID2Env/SourceData/ForestFire/burn_layout.json"


def burn_mask(layout: dict) -> np.ndarray:
    m = layout["mask"]
    size, extent = int(m["size_px"]), float(m["extent_m"])
    px_m = extent / size  # metres per pixel
    half = extent / 2.0
    poly = np.asarray(layout["burn_polygon_ne"], float)
    # (n, e) -> pixel (x = col = north, y = row = east)
    pts = np.round((poly[:, [0, 1]] + half) / px_m).astype(np.int32)
    inside = np.zeros((size, size), np.uint8)
    cv2.fillPoly(inside, [pts.reshape(-1, 1, 2)], 1)
    # signed distance (m): positive inside the polygon
    d_in = cv2.distanceTransform(inside, cv2.DIST_L2, 5) * px_m
    d_out = cv2.distanceTransform(1 - inside, cv2.DIST_L2, 5) * px_m
    sd = d_in - d_out
    e = layout["edge"]
    rng = np.random.default_rng(int(e["seed"]))
    sigma = float(e["noise_scale_m"]) / px_m
    noise = cv2.GaussianBlur(rng.standard_normal((size, size)).astype(np.float32), (0, 0), sigma)
    noise *= float(e["noise_m"]) / max(float(noise.std()), 1e-6)
    sd = sd + noise
    soft = float(e["soft_m"])
    mask = np.clip(0.5 + sd / soft, 0.0, 1.0)
    mask = mask * mask * (3 - 2 * mask)  # smoothstep
    return (128.0 + mask * 127.0 + 0.5).astype(np.uint8)  # fillPoly drew x = north (col), y = east (row)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("layout", nargs="?", default=str(DEFAULT))
    args = ap.parse_args(argv)
    path = Path(args.layout)
    layout = json.loads(path.read_text(encoding="utf-8"))
    mask = burn_mask(layout)
    out = path.parent / layout["mask"]["file"]
    cv2.imwrite(str(out), mask)
    print(f"{out}: {mask.shape[1]} x {mask.shape[0]}, burnt {100.0 * (mask > 191).mean():.1f} % of the map")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
