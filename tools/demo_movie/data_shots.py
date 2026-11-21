"""Data shots: frames drawn from a real run's logs (every number, box and track on screen is recorded).

Each shot is an object with ``n`` frames and ``frame(i) -> BGR 1920x1080``:

* ``MapShot``      top-down map: search area, no-fly zone, fires, pads; optional time-lapse of the
                   team (tracks, photo footprints = coverage, radio packets, reports) and pin drops;
* ``FeedsGrid``    2x2 grid of the drones' real search-camera photos at a run time (picture-in-picture);
* ``SensorQuad``   one frame as RGB / depth / segmentation / ground-truth boxes;
* ``FrontFeed``    the forward (navigation) camera frames of a stretch of flight;
* ``DetectionZoom`` a real frame zooming onto a target, detector boxes, and the AI's words typed out;
* ``Scorecard``    the six layer scores and the overall score.

The map background is the top-down Movie Render Queue plate (``MapPlate``) when it exists, otherwise a
hillshade of the run's terrain model.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from demo_movie import typo as ty
from demo_movie.run_data import FilmRun
from vivid2.geo import pixel_ray_to_ground, world_to_pixel

W, H = ty.W, ty.H


# -- map ---------------------------------------------------------------------------------------

@dataclass
class MapView:
    """North-up top-down view: a nadir pinhole camera ``alt_m`` above (n0, e0) (image right = east).

    With ``alt_m`` very large this is an orthographic map of ``m_per_px``."""
    n0: float
    e0: float
    cam_down: float     # camera NED down
    f_px: float         # focal length in pixels
    w: int = W
    h: int = H

    @classmethod
    def ortho(cls, n0: float, e0: float, m_per_px: float, w: int = W, h: int = H) -> "MapView":
        alt = 1e6
        return cls(n0, e0, -alt, alt / m_per_px, w, h)

    def px(self, ned: Sequence[float], ground_down: Optional[float] = None) -> Tuple[float, float]:
        d = ned[2] if len(ned) > 2 else ground_down
        depth = (d if d is not None else 0.0) - self.cam_down
        return (self.w / 2 + self.f_px * (ned[1] - self.e0) / depth, self.h / 2 - self.f_px * (ned[0] - self.n0) / depth)

    def pts(self, ne: Sequence[Sequence[float]], down: float) -> np.ndarray:
        return np.array([self.px((p[0], p[1], down)) for p in ne], np.float32)

    def m_per_px(self, down: float) -> float:
        return (down - self.cam_down) / self.f_px


def hillshade(run: FilmRun, view: MapView, ground: Any) -> np.ndarray:
    """A shaded-relief stand-in for the map plate (pixels -> NE -> terrain)."""
    ys, xs = np.mgrid[0:view.h:4, 0:view.w:4].astype(np.float32)
    dmean = float(np.mean(ground.down_at(view.n0, view.e0)))
    mpp = view.m_per_px(dmean)
    e = view.e0 + (xs - view.w / 2) * mpp
    n = view.n0 - (ys - view.h / 2) * mpp
    z = -ground.down_at(n, e)
    gy, gx = np.gradient(z, 4 * mpp)
    az, alt = math.radians(300), math.radians(40)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    shade = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    shade = np.clip(shade, 0, 1)
    zn = (z - z.min()) / max(1e-6, z.max() - z.min())
    base = np.stack([60 + 40 * zn, 85 + 50 * zn, 70 + 30 * zn], -1) * (0.55 + 0.6 * shade[..., None])
    img = cv2.resize(np.clip(base, 0, 255).astype(np.uint8), (view.w, view.h), interpolation=cv2.INTER_CUBIC)
    return img


class MapPlate:
    """The map background and its projection: an MRQ render from a nadir camera, or a hillshade."""

    def __init__(self, run: FilmRun, plate: Optional[Dict[str, Any]] = None, png: Optional[Path] = None):
        from vivid2 import terrain as terrain_mod
        self.run = run
        self.ground = terrain_mod.from_meta(run.meta)
        poly = np.asarray(run.meta["search_area"]["polygon_ne"], float)
        pads = np.asarray([f["start_ned"][:2] for f in run.meta.get("fleet") or []] or [[0, 0]], float)
        pts = np.vstack([poly, pads])
        lo, hi = pts.min(0) - 25, pts.max(0) + 25
        if plate:  # the camera the plate was rendered with
            self.view = MapView(plate["n0"], plate["e0"], plate["cam_down"], plate["f_px"])
        else:
            span_n, span_e = hi[0] - lo[0], hi[1] - lo[1]
            mpp = max(span_e / W, span_n / H) * 1.05
            self.view = MapView.ortho((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, mpp)
        if png is not None and Path(png).exists():
            self.base = cv2.resize(cv2.imread(str(png)), (W, H))
        else:
            self.base = hillshade(run, self.view, self.ground)

    def gd(self, n: float, e: float) -> float:
        return float(self.ground.down_at(n, e))

    def p(self, n: float, e: float, down: Optional[float] = None) -> Tuple[int, int]:
        u, v = self.view.px((n, e, self.gd(n, e) if down is None else down))
        return int(round(u)), int(round(v))

    def poly_px(self, ne: Sequence[Sequence[float]]) -> np.ndarray:
        return np.array([self.p(a, b) for a, b in ne], np.int32)


def _alpha_fill(img, poly, color, alpha):
    over = img.copy()
    cv2.fillPoly(over, [poly], color, cv2.LINE_AA)
    cv2.addWeighted(over, alpha, img, 1 - alpha, 0, img)


def _partial_polyline(poly: np.ndarray, frac: float) -> np.ndarray:
    """The first ``frac`` of a closed polygon's perimeter."""
    pts = np.vstack([poly, poly[:1]]).astype(np.float32)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    total, target = seg.sum(), seg.sum() * min(1.0, max(0.0, frac))
    out, acc = [pts[0]], 0.0
    for a, b, L in zip(pts[:-1], pts[1:], seg):
        if acc + L >= target:
            out.append(a + (b - a) * ((target - acc) / max(L, 1e-6)))
            break
        out.append(b)
        acc += L
    return np.array(out, np.int32)


def fire_icon(img, xy, r=7, alpha=1.0):
    x, y = xy
    over = img.copy()
    cv2.circle(over, (x, y), r + 6, (0, 90, 255), -1, cv2.LINE_AA)
    cv2.circle(over, (x, y), r, (40, 200, 255), -1, cv2.LINE_AA)
    cv2.addWeighted(over, 0.85 * alpha, img, 1 - 0.85 * alpha, 0, img)


class MapShot:
    """Top-down map. ``mode``: 'call' (search area draws itself, no-fly fades in), 'timelapse'
    (run time t0..t1 over n frames: tracks, coverage, radio, reports), or 'pins' (all tracks, pin drop)."""

    def __init__(self, run: FilmRun, plate: MapPlate, n: int, mode: str = "call", t0: float = 0.0,
                 t1: Optional[float] = None, pin: Optional[str] = None, lkp: Optional[Sequence[float]] = None):
        self.run, self.plate, self.n, self.mode = run, plate, n, mode
        self.t0, self.t1 = t0, (t1 if t1 is not None else run.duration)
        self.pin, self.lkp = pin, lkp
        self.colors = {d: ty.DRONE_COLORS[i % 4] for i, d in enumerate(run.drones)}
        m = run.meta
        self.search = plate.poly_px(m["search_area"]["polygon_ne"])
        lim = m.get("limits") or {}
        self.nofly = [plate.poly_px(z["polygon_ne"]) for z in lim.get("no_fly", [])]
        self.fires = [plate.p(f["ned"][0], f["ned"][1], f["ned"][2]) for f in m.get("fires") or []]
        self.pads = [(f["id"], plate.p(f["start_ned"][0], f["start_ned"][1], f["start_ned"][2])) for f in m.get("fleet") or []]
        self._cov = None
        if mode in ("timelapse", "pins"):
            self._prepare_tracks()

    def _prepare_tracks(self):
        self.tracks = {}
        for d, tr in self.run.tracks.items():
            pts = [self.plate.view.px(p) for p in tr.pos]
            self.tracks[d] = (tr.t, np.array(pts, np.float32))
        self.footprints = []
        for f in self.run.frames("Payload"):
            gd = self.plate.gd(f["cam_pos"][0], f["cam_pos"][1])
            corners = [pixel_ray_to_ground(u, v, f["cam_pos"], f["cam_q"], f["width"], f["height"], f["fov_deg"], gd)
                       for u, v in ((0, 0), (f["width"], 0), (f["width"], f["height"]), (0, f["height"]))]
            if any(c is None for c in corners):
                continue
            poly = np.array([self.plate.view.px(c) for c in corners], np.int32)
            self.footprints.append((f["rt"], f.get("drone", self.run.drones[0]), poly))
        self.footprints.sort(key=lambda x: x[0])
        self.comms = [dict(c, rt=self.run.rt(c["t_send"])) for c in self.run.log["comms"] if c.get("delivered")]
        self.reports = self.run.reports()
        self._cov = np.zeros((H, W, 3), np.uint8)
        self._cov_mask = np.zeros((H, W), np.uint8)
        self._cov_i = 0

    def _coverage_until(self, t):
        while self._cov_i < len(self.footprints) and self.footprints[self._cov_i][0] <= t:
            _, d, poly = self.footprints[self._cov_i]
            cv2.fillPoly(self._cov, [poly], self.colors[d], cv2.LINE_AA)
            cv2.fillPoly(self._cov_mask, [poly], 255, cv2.LINE_AA)
            self._cov_i += 1

    def _static(self, img, i):
        a_area = ty.ease(i / (0.6 * self.n)) if self.mode == "call" else 1.0
        if self.mode == "call":
            cv2.polylines(img, [_partial_polyline(self.search, a_area)], False, (255, 255, 255), 3, cv2.LINE_AA)
        else:
            cv2.polylines(img, [self.search], True, (255, 255, 255), 2, cv2.LINE_AA)
        a_nf = ty.ease((i - 0.45 * self.n) / (0.3 * self.n)) if self.mode == "call" else 1.0
        for z in self.nofly:
            _alpha_fill(img, z, (30, 30, 230), 0.28 * a_nf)
            if a_nf > 0:
                cv2.polylines(img, [z], True, (40, 40, 255), 2, cv2.LINE_AA)
        for xy in self.fires:
            fire_icon(img, xy, alpha=a_nf if self.mode == "call" else 1.0)
        for name, (x, y) in self.pads:
            cv2.circle(img, (x, y), 5, (255, 255, 255), 1, cv2.LINE_AA)
        if self.nofly and a_nf > 0.5:
            z = self.nofly[0]
            ty.text(img, (int(z[:, 0].mean()), int(z[:, 1].min()) - 34), "NO-FLY < 30 m", 24, "semibold",
                    color=(80, 80, 255), alpha=a_nf, anchor="ma")

    def frame(self, i: int) -> np.ndarray:
        img = self.plate.base.copy()
        img = (img.astype(np.float32) * 0.82).astype(np.uint8)
        t = self.t0 + (self.t1 - self.t0) * (i / max(1, self.n - 1))
        if self.mode in ("timelapse", "pins"):
            if self.mode == "pins":
                t = self.t1
            self._coverage_until(t)
            m = (self._cov_mask.astype(np.float32) / 255.0 * 0.32)[..., None]
            img = (img * (1 - m) + self._cov * m).astype(np.uint8)
        self._static(img, i)
        if self.mode == "call" and self.lkp is not None:
            x, y = self.plate.p(*self.lkp)
            ph = (i % 30) / 30.0
            cv2.circle(img, (x, y), int(8 + 40 * ph), (255, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(img, (x, y), 7, ty.ACCENT, -1, cv2.LINE_AA)
            ty.text(img, (x + 16, y - 12), "trailhead", 24, "semibold")
        if self.mode in ("timelapse", "pins"):
            self._draw_team(img, t)
        return img

    def _draw_team(self, img, t):
        for d, (ts, pts) in self.tracks.items():
            k = int(np.searchsorted(ts, t))
            if k >= 2:
                cv2.polylines(img, [pts[:k].astype(np.int32)], False, self.colors[d], 2, cv2.LINE_AA)
            if k >= 1:
                x, y = pts[min(k, len(pts)) - 1]
                cv2.circle(img, (int(x), int(y)), 9, self.colors[d], -1, cv2.LINE_AA)
                cv2.circle(img, (int(x), int(y)), 9, (20, 20, 20), 2, cv2.LINE_AA)
                ty.text(img, (int(x) + 14, int(y) - 14), d.replace("Drone", "D"), 22, "semibold", color=self.colors[d])
        # radio: each delivered packet draws a fading line sender -> receiver for 0.6 s
        pos_at = {d: self._pos_px(d, t) for d in self.tracks}
        for c in self.comms:
            age = t - c["rt"]
            if 0 <= age <= 0.6 * max(1.0, (self.t1 - self.t0) / 150):
                a, b = pos_at.get(c["from"]), pos_at.get(c["to"])
                if a is not None and b is not None:
                    over = img.copy()
                    cv2.line(over, a, b, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.addWeighted(over, 0.5, img, 0.5, 0, img)
        for r in self.reports:
            if r["rt"] <= t:
                x, y = self.plate.p(r["ned"][0], r["ned"][1])
                cv2.circle(img, (x, y), 16, (255, 255, 255), 3, cv2.LINE_AA)
                cv2.drawMarker(img, (x, y), ty.ACCENT, cv2.MARKER_CROSS, 18, 3, cv2.LINE_AA)
        mm, ss = divmod(max(0, int(t)), 60)
        ty.panel(img, W - 330, 40, W - 40, 120, alpha=0.55)
        ty.text(img, (W - 315, 52), f"T+{mm:02d}:{ss:02d}", 48, "display")
        ty.text(img, (W - 120, 66), "mission time", 18, "regular", color=(200, 200, 200), anchor="ma")

    def _pos_px(self, d, t):
        p, _ = self.run.tracks[d].pose_at(t)
        u, v = self.plate.view.px(p)
        return int(u), int(v)


# -- camera feeds ------------------------------------------------------------------------------

class FrameCache:
    def __init__(self, run: FilmRun, size: int = 64):
        self.run, self.size, self.cache = run, size, {}

    def get(self, rec: Dict[str, Any]) -> np.ndarray:
        key = rec["rgb"]
        img = self.cache.get(key)
        if img is None:
            img = cv2.imread(str(self.run.run_dir / rec["rgb"]))
            if len(self.cache) > self.size:
                self.cache.pop(next(iter(self.cache)))
            self.cache[key] = img
        return img


def latest_frame(frames: List[Dict[str, Any]], t: float) -> Optional[Dict[str, Any]]:
    best = None
    for f in frames:
        if f["rt"] <= t:
            best = f
        else:
            break
    return best


class FeedsGrid:
    """2x2 grid of each drone's newest real search-camera photo at run times t0..t1."""

    def __init__(self, run: FilmRun, n: int, t0: float, t1: float, size: Tuple[int, int] = (640, 480)):
        self.run, self.n, self.t0, self.t1 = run, n, t0, t1
        self.w, self.h = size
        self.frames = {d: run.frames("Payload", d) for d in run.drones}
        self.cache = FrameCache(run)

    def tile(self, i: int, w: Optional[int] = None, h: Optional[int] = None) -> np.ndarray:
        t = self.t0 + (self.t1 - self.t0) * (i / max(1, self.n - 1))
        w, h = w or self.w, h or self.h
        tw, th = w // 2, h // 2
        out = np.zeros((h, w, 3), np.uint8)
        for k, d in enumerate(self.run.drones[:4]):
            f = latest_frame(self.frames[d], t)
            x0, y0 = (k % 2) * tw, (k // 2) * th
            if f is None:
                continue
            out[y0:y0 + th, x0:x0 + tw] = ty.cover(self.cache.get(f), tw, th)
            agl = -(f["cam_pos"][2] - self._gd(f))
            g = f.get("gimbal") or {}
            tilt = g.get("tilt_deg", -90)
            fs = max(16, th // 14)
            ty.text(out, (x0 + 12, y0 + 8), f"{d}  {agl:.0f} m AGL  tilt {tilt:.0f}°", fs, "semibold",
                    color=ty.DRONE_COLORS[k])
            cv2.rectangle(out, (x0, y0), (x0 + tw - 1, y0 + th - 1), (30, 30, 30), 2)
        return out

    def _gd(self, f):
        from vivid2 import terrain as terrain_mod
        if not hasattr(self, "_ground"):
            self._ground = terrain_mod.from_meta(self.run.meta)
        return float(self._ground.down_at(f["cam_pos"][0], f["cam_pos"][1]))

    def frame(self, i: int) -> np.ndarray:
        return self.tile(i, W, H)


class FrontFeed:
    """A drone's real forward-camera frames between run times t0 and t1, full screen with a HUD
    (its search-camera photos if it took no forward-camera photos then)."""

    def __init__(self, run: FilmRun, drone: str, n: int, t0: float, t1: float):
        self.run, self.n, self.t0, self.t1, self.drone = run, n, t0, t1, drone
        self.camera = "FORWARD CAMERA"
        self.frames = [f for f in run.frames("Front", drone) if t0 - 2 <= f["rt"] <= t1 + 2]
        if not self.frames:
            self.frames = [f for f in run.frames("Payload", drone) if t0 - 2 <= f["rt"] <= t1 + 2]
            self.camera = "SEARCH CAMERA"
        self.cache = FrameCache(run)
        from vivid2 import terrain as terrain_mod
        self.ground = terrain_mod.from_meta(run.meta)

    def frame(self, i: int) -> np.ndarray:
        t = self.t0 + (self.t1 - self.t0) * (i / max(1, self.n - 1))
        f = latest_frame(self.frames, t) or (self.frames[0] if self.frames else None)
        if f is None:
            return ty.card([("no forward-camera frames", 40, "regular")])
        img = ty.cover(self.cache.get(f))
        p, _ = self.run.tracks[self.drone].pose_at(t)
        agl = float(self.ground.down_at(p[0], p[1]) - p[2])
        ty.panel(img, 40, 40, 520, 150, alpha=0.5)
        ty.text(img, (60, 52), f"{self.drone} · {self.camera}", 26, "semibold")
        ty.text(img, (60, 92), f"{agl:4.1f} m above ground   {self.run.tracks[self.drone].speed_at(t):3.1f} m/s", 28, "mono")
        return img


MASK_COLORS = [(60, 220, 255), (255, 120, 90), (120, 255, 140), (255, 90, 220), (90, 160, 255)]


def target_mask(seg: np.ndarray) -> np.ndarray:
    """VIVID 2 labels targets, decoys and fires; everything else shares one background label. Show the
    background dark and each labelled object in its own bright colour, slightly grown so small ones read."""
    flat = seg.reshape(-1, 3)
    colors, counts = np.unique(flat, axis=0, return_counts=True)
    bg = colors[np.argmax(counts)]
    out = np.full_like(seg, (26, 24, 22))
    k = 0
    for c in colors[np.argsort(-counts)]:
        if np.array_equal(c, bg):
            continue
        m = np.all(seg == c, axis=-1).astype(np.uint8)
        m = cv2.dilate(m, np.ones((9, 9), np.uint8))
        out[m > 0] = MASK_COLORS[k % len(MASK_COLORS)]
        k += 1
    return out


class SensorQuad:
    """One search-camera view as RGB / depth / segmentation / ground-truth boxes (2x2): from a live
    capture (capture_quad.py: all four from the same instant), or from a run's frame bank (no
    segmentation: runs don't keep it)."""

    def __init__(self, rgb: np.ndarray, depth_m: Optional[np.ndarray], seg: Optional[np.ndarray],
                 gts: List[Dict[str, Any]], n: int):
        self.n, self.rgb, self.seg = n, rgb, seg
        self.center = None  # the tiles zoom together towards the first target
        if gts:
            x0, y0, x1, y1 = gts[0]["bbox"]
            self.center = ((x0 + x1) / 2, (y0 + y1) / 2)
        if depth_m is not None:
            d = depth_m.astype(np.float32)
            valid = d > 0
            lo, hi = (np.percentile(d[valid], 2), np.percentile(d[valid], 98)) if valid.any() else (0, 1)
            dn = np.clip((d - lo) / max(1e-3, hi - lo), 0, 1)
            col = cv2.applyColorMap((255 * (1 - dn)).astype(np.uint8), cv2.COLORMAP_TURBO)
            col[~valid] = 0
            self.depth = cv2.resize(col, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            self.depth = np.zeros_like(rgb)
        self.gt = rgb.copy()
        for g in gts:
            x0, y0, x1, y1 = g["bbox"]
            pad = 14
            ty.label_box(self.gt, (x0 - pad, y0 - pad, x1 + pad, y1 + pad), f"{g['object_id']} · {g['visible_px']} px",
                         color=(80, 255, 120), thick=4)

    @classmethod
    def from_capture(cls, folder: Path, n: int) -> "SensorQuad":
        folder = Path(folder)
        rgb = cv2.imread(str(folder / "rgb.png"))
        dmm = cv2.imread(str(folder / "depth_mm.png"), cv2.IMREAD_UNCHANGED)
        seg = cv2.imread(str(folder / "seg.png"))
        if seg is not None:
            seg = target_mask(seg)
        gts = json.loads((folder / "gt.json").read_text(encoding="utf-8"))["gt"]
        return cls(rgb, None if dmm is None else dmm.astype(np.float32) / 1000.0, seg, gts, n)

    @classmethod
    def from_run(cls, run: FilmRun, frame_id: int, n: int) -> "SensorQuad":
        rec = run.frame(frame_id)
        rgb = cv2.imread(str(run.run_dir / rec["rgb"]))
        dpath = rec.get("depth")
        dmm = cv2.imread(str(run.run_dir / dpath), cv2.IMREAD_UNCHANGED) if dpath else None
        return cls(rgb, None if dmm is None else dmm.astype(np.float32) / 1000.0, None, rec.get("gt", []), n)

    def frame(self, i: int) -> np.ndarray:
        tw, th = W // 2, H // 2
        z = (1.15 + 0.75 * ty.ease(i / max(1, self.n - 1))) if self.center else 1.0 + 0.05 * ty.ease(i / max(1, self.n - 1))
        tiles = [("RGB", self.rgb), ("DEPTH", self.depth),
                 ("SEGMENTATION · targets, decoys, fires", self.seg if self.seg is not None else None), ("GROUND TRUTH", self.gt)]
        out = np.zeros((H, W, 3), np.uint8)
        for k, (name, img) in enumerate(tiles):
            x0, y0 = (k % 2) * tw, (k // 2) * th
            if img is None:
                tile = ty.card([("segmentation", 30, "regular"), ("(capture it live: capture_quad.py)", 22, "light")], w=tw, h=th)
            else:
                tile = ty.cover(img, tw, th, center=self.center, zoom=z)
            out[y0:y0 + th, x0:x0 + tw] = tile
            ty.panel(out, x0 + 18, y0 + 18, x0 + 40 + ty.text_size(name, 24, "semibold")[0], y0 + 62, alpha=0.6, radius=8)
            ty.text(out, (x0 + 30, y0 + 24), name, 24, "semibold")
        cv2.line(out, (tw, 0), (tw, H), (10, 10, 10), 3)
        cv2.line(out, (0, th), (W, th), (10, 10, 10), 3)
        return out


class DetectionZoom:
    """A real search-camera frame: the detector's boxes pop up, the view zooms onto the target, and
    the AI's assessment types out in a side panel."""

    def __init__(self, run: FilmRun, frame_id: int, target: Optional[str], n: int, words: Optional[Dict[str, Any]] = None,
                 zoom_to: float = 5.0, show_detector: bool = False, center: Optional[Tuple[float, float]] = None):
        self.run, self.n, self.target, self.words, self.zoom_to = run, n, target, words, zoom_to
        self.show_detector = show_detector
        self.rec = run.frame(frame_id)
        self.img = cv2.imread(str(run.run_dir / self.rec["rgb"]))
        g = next((g for g in self.rec.get("gt", []) if g["object_id"] == target), None)
        if g is not None:
            x0, y0, x1, y1 = g["bbox"]
            self.center = ((x0 + x1) / 2, (y0 + y1) / 2)
        else:
            self.center = (self.img.shape[1] / 2, self.img.shape[0] / 2)
        if center is not None:
            self.center = center
        self.gt = g
        self.dets = []
        for d in run.log["detections"]:
            if d.get("frame_id") == frame_id:
                self.dets += d.get("dets", [])

    def frame(self, i: int) -> np.ndarray:
        n = self.n
        zt = ty.ease((i - 0.15 * n) / (0.45 * n))
        z = 1.0 + (self.zoom_to - 1.0) * zt
        ih, iw = self.img.shape[:2]
        s0 = max(W / iw, H / ih)
        # view centre moves from the image centre to the target
        cx = iw / 2 + (self.center[0] - iw / 2) * zt
        cy = ih / 2 + (self.center[1] - ih / 2) * zt
        out = ty.cover(self.img, W, H, (cx, cy), z)
        s = s0 * z
        cw, ch = W / s, H / s
        x0 = min(max(0.0, cx - cw / 2), iw - cw)
        y0 = min(max(0.0, cy - ch / 2), ih - ch)
        to_px = lambda u, v: ((u - x0) * s, (v - y0) * s)
        a_box = ty.ease((i - 0.05 * n) / (0.08 * n))
        pad = 10
        if self.gt is not None:  # the simulator's ground truth for this target
            gx0, gy0, gx1, gy1 = self.gt["bbox"]
            p0, p1 = to_px(gx0, gy0), to_px(gx1, gy1)
            ty.label_box(out, (p0[0] - pad, p0[1] - pad, p1[0] + pad, p1[1] + pad),
                         f"ground truth · {self.gt.get('visible_px', 0)} px", color=(90, 230, 110), alpha=a_box)
        if self.show_detector:
            for d in self.dets:
                bx0, by0, bx1, by1 = d["bbox"]
                p0, p1 = to_px(bx0, by0), to_px(bx1, by1)
                ty.label_box(out, (p0[0] - pad, p0[1] - pad, p1[0] + pad, p1[1] + pad),
                             f"{d.get('label', 'person')} {d.get('score', 0):.2f}", alpha=a_box)
        px = (self.words or {}).get("pixel")
        a_pt = ty.ease((i - 0.62 * n) / (0.08 * n))
        if px and a_pt > 0:  # the VLM's answer is a point, not a box
            u, v = to_px(px[0], px[1])
            u, v = int(round(u)), int(round(v))
            over = out.copy()
            r = 26
            cv2.circle(over, (u, v), r, ty.ACCENT, 3, cv2.LINE_AA)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                cv2.line(over, (u + dx * (r - 10), v + dy * (r - 10)), (u + dx * (r + 14), v + dy * (r + 14)),
                         ty.ACCENT, 3, cv2.LINE_AA)
            cv2.addWeighted(over, a_pt, out, 1 - a_pt, 0, out)
            ty.text(out, (u + r + 18, v + 8), (self.words.get("model") or "VLM") + "'s point", 24, "semibold",
                    color=ty.ACCENT, alpha=a_pt)
        if self.words:
            tx, _ = to_px(*self.center)
            self._words(out, i, left=tx > W / 2)
        return out

    def _words(self, img, i, left=False):
        n = self.n
        a = ty.ease((i - 0.55 * n) / (0.08 * n))
        if a <= 0:
            return
        desc = self.words.get("description", "")
        conf = self.words.get("confidence")
        model = self.words.get("model", "")
        head = "AI ASSESSMENT" + (f" · {model}" if model else "")
        chars = int(len(desc) * min(1.0, max(0.0, (i - 0.6 * n) / (0.3 * n))))
        body = _wrap(desc[:chars], 34)
        x0, y0 = (60 if left else W - 760), 120
        ty.panel(img, x0, y0, x0 + 700, y0 + 120 + 44 * max(3, body.count("\n") + 1) + 80, alpha=0.62 * a)
        ty.text(img, (x0 + 32, y0 + 26), head, 24, "semibold", color=ty.ACCENT, alpha=a)
        ty.text(img, (x0 + 32, y0 + 76), body, 34, "regular", alpha=a, spacing=10)
        if conf is not None and chars >= len(desc):
            yb = y0 + 120 + 44 * max(3, body.count("\n") + 1)
            found = self.words.get("found", True)
            verdict = (f"target found · confidence {conf:.2f}" if found else f"not sure · confidence {conf:.2f}")
            ty.text(img, (x0 + 32, yb), verdict, 30, "semibold",
                    color=(120, 255, 140) if found else (60, 200, 255), alpha=a)


def _wrap(s: str, width: int) -> str:
    words, lines, cur = s.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    return "\n".join(lines)


# -- scorecard ---------------------------------------------------------------------------------

LAYERS = ["perception", "reasoning", "planning", "control", "safety", "mission"]


class Scorecard:
    def __init__(self, run: FilmRun, n: int, agent_label: str = ""):
        self.run, self.n, self.agent_label = run, n, agent_label
        self.scores = run.result.get("scores", {})

    def frame(self, i: int) -> np.ndarray:
        img = np.full((H, W, 3), (16, 16, 18), np.uint8)
        n = self.n
        ty.text(img, (160, 110), "SCORED, LAYER BY LAYER", 54, "display", alpha=ty.fade(i, n, 12, 0))
        if self.agent_label:
            ty.text(img, (160, 180), self.agent_label, 28, "regular", color=(190, 190, 190), alpha=ty.fade(i, n, 12, 0))
        x0, x1 = 520, 1640
        for k, layer in enumerate(LAYERS):
            y = 270 + k * 92
            v = float(self.scores.get(layer) or 0.0)
            grow = ty.ease((i - 8 - 6 * k) / 24.0)
            ty.text(img, (160, y), layer.capitalize(), 40, "semibold", alpha=min(1.0, grow * 3))
            cv2.rectangle(img, (x0, y + 6), (x1, y + 46), (45, 45, 50), -1)
            cv2.rectangle(img, (x0, y + 6), (x0 + int((x1 - x0) * v / 100.0 * grow), y + 46), ty.ACCENT, -1)
            ty.text(img, (x1 + 30, y), f"{v * grow:5.1f}", 40, "mono", alpha=min(1.0, grow * 3))
        ov = float(self.scores.get("overall") or 0.0)
        g = ty.ease((i - 60) / 30.0)
        ty.text(img, (160, 860), "Overall", 52, "display", alpha=g)
        ty.text(img, (x0, 845), f"{ov * g:.1f}", 96, "display", color=ty.ACCENT, alpha=g)
        return img


# -- the search camera's gimbal, seen through it --------------------------------------------------

def _lerp(a, b, t):
    return a + (b - a) * t


class GimbalFeed:
    """Photos of a live gimbal demonstration (capture_gimbal.py) as a camera feed with a HUD: tilt, pan,
    field of view, and the hiker's size in pixels (ground truth) as the camera zooms in."""

    def __init__(self, folder: Path, n: int, label: str = "", fade: int = 5):
        folder = Path(folder)
        meta = json.loads((folder / "photos.json").read_text(encoding="utf-8"))
        self.target, self.photos = meta["target"], meta["photos"]
        self.folder, self.n, self.label, self.fade = folder, n, label or meta["target"], fade
        self.hold = n / len(self.photos)
        self.cache: Dict[str, np.ndarray] = {}

    def _img(self, k: int) -> np.ndarray:
        p = self.photos[min(k, len(self.photos) - 1)]
        img = self.cache.get(p["file"])
        if img is None:
            img = ty.cover(cv2.imread(str(self.folder / p["file"])))
            self.cache[p["file"]] = img
        return img

    def _box(self, k: int):
        p = self.photos[min(k, len(self.photos) - 1)]
        g = next((g for g in p["gt"] if g["object_id"] == self.target and g["visible_px"] > 0), None)
        if g is None:
            return None, 0
        # photo pixels (1920 x 1440) -> the 16:9 cover crop
        s = W / 1920.0
        x0, y0, x1, y1 = g["bbox"]
        return (x0 * s, (y0 - 180) * s, x1 * s, (y1 - 180) * s), g["visible_px"]

    def frame(self, i: int) -> np.ndarray:
        k = int(i // self.hold)
        a = (i - k * self.hold) / self.hold
        img = self._img(k)
        nxt = k + 1 < len(self.photos)
        into = max(0.0, (i - (k + 1) * self.hold + self.fade)) / self.fade if nxt else 0.0
        if into > 0:
            img = ty.blend(img, self._img(k + 1), into)
        else:
            img = img.copy()
        g0 = self.photos[min(k, len(self.photos) - 1)]["gimbal"]
        g1 = self.photos[min(k + 1, len(self.photos) - 1)]["gimbal"]
        tilt, pan, fov = (_lerp(g0[x], g1[x], ty.ease(a)) for x in ("tilt_deg", "pan_deg", "fov_deg"))
        box, px = self._box(k)
        if box is not None:
            pad = 16
            ty.label_box(img, (box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad),
                         f"{self.label} · {px:,} px", color=(90, 230, 110), thick=3)
        # HUD: three readouts and a zoom bar
        x0, y0 = 60, 60
        ty.panel(img, x0, y0, x0 + 430, y0 + 250, alpha=0.55)
        ty.text(img, (x0 + 24, y0 + 18), "SEARCH CAMERA · GIMBAL", 24, "semibold", color=ty.ACCENT)
        for r, (name, val) in enumerate((("TILT", f"{tilt:6.1f}°"), ("PAN", f"{pan:6.1f}°"), ("FOV", f"{fov:6.1f}°"))):
            ty.text(img, (x0 + 24, y0 + 64 + r * 46), name, 28, "semibold", color=(200, 200, 200))
            ty.text(img, (x0 + 150, y0 + 60 + r * 46), val, 34, "mono")
        mag = math.tan(math.radians(40.0)) / math.tan(math.radians(fov / 2.0))
        bx0, by = x0 + 24, y0 + 212
        cv2.rectangle(img, (bx0, by), (bx0 + 300, by + 10), (60, 60, 60), -1)
        cv2.rectangle(img, (bx0, by), (bx0 + int(300 * min(1.0, (mag - 1) / 9.0)), by + 10), ty.ACCENT, -1)
        ty.text(img, (bx0 + 316, by - 12), f"{mag:4.1f}×", 26, "semibold")
        # crosshair at the optical centre
        c = (W // 2, H // 2)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            cv2.line(img, (c[0] + dx * 18, c[1] + dy * 18), (c[0] + dx * 42, c[1] + dy * 42), (240, 240, 240), 2, cv2.LINE_AA)
        return img


# -- from a point in a photo to a position on the map -------------------------------------------

class GeoLocate:
    """How a find becomes a position: the AI's point in the photo, the depth at that pixel and the
    camera's pose give a point on the ground; compared with where the hiker really is."""

    def __init__(self, run: FilmRun, frame_id: int, target: str, point: Sequence[float], n: int, who: str = "AI"):
        from vivid2.geo import pixel_to_world
        self.run, self.n, self.who = run, n, who
        self.rec = run.frame(frame_id)
        self.img = cv2.imread(str(run.run_dir / self.rec["rgb"]))
        dm = cv2.imread(str(run.run_dir / self.rec["depth"]), cv2.IMREAD_UNCHANGED) if self.rec.get("depth") else None
        self.u, self.v = float(point[0]), float(point[1])
        h, w = self.img.shape[:2]
        if dm is not None:
            sy, sx = dm.shape[0] / h, dm.shape[1] / w
            win = dm[max(0, int(self.v * sy) - 3):int(self.v * sy) + 4, max(0, int(self.u * sx) - 3):int(self.u * sx) + 4]
            vals = win[win > 0]
            self.depth_m = float(np.median(vals)) / 1000.0 if vals.size else None
            d = dm.astype(np.float32) / 1000.0
            valid = d > 0
            lo, hi = np.percentile(d[valid], 2), np.percentile(d[valid], 98)
            col = cv2.applyColorMap((255 * (1 - np.clip((d - lo) / max(1e-3, hi - lo), 0, 1))).astype(np.uint8),
                                    cv2.COLORMAP_TURBO)
            col[~valid] = 0
            self.depth_img = cv2.resize(col, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            self.depth_m, self.depth_img = None, np.zeros_like(self.img)
        t = run.target(target)
        self.truth = t["ned"]
        if self.depth_m:
            self.pos = pixel_to_world(self.u, self.v, self.depth_m, self.rec["cam_pos"], self.rec["cam_q"], w, h,
                                      self.rec["fov_deg"])
            self.err = float(math.hypot(self.pos[0] - self.truth[0], self.pos[1] - self.truth[1]))
        else:
            self.pos, self.err = None, None

    def frame(self, i: int) -> np.ndarray:
        n = self.n
        out = np.full((H, W, 3), (16, 16, 18), np.uint8)
        zw = 900
        crop = lambda img: ty.cover(img, zw, 506, center=(self.u, self.v), zoom=2.4)
        a1 = ty.ease(i / (0.12 * n))
        a2 = ty.ease((i - 0.25 * n) / (0.12 * n))
        out[90:596, 60:60 + zw] = (crop(self.img).astype(np.float32) * a1).astype(np.uint8)
        out[90:596, 960:960 + zw] = (crop(self.depth_img).astype(np.float32) * a2).astype(np.uint8)
        ih, iw = self.img.shape[:2]
        pu, pv = ty.cover_map(iw, ih, zw, 506, (self.u, self.v), 2.4)(self.u, self.v)
        for x0, a, name in ((60, a1, f"PHOTO · {self.who}'s point"), (960, a2, "DEPTH at that pixel")):
            if a > 0:
                c = (int(x0 + pu), int(90 + pv))
                cv2.circle(out, c, 24, ty.ACCENT, 3, cv2.LINE_AA)
                cv2.drawMarker(out, c, ty.ACCENT, cv2.MARKER_CROSS, 16, 2, cv2.LINE_AA)
                ty.text(out, (x0, 40), name, 28, "semibold", alpha=a)
        if self.depth_m and a2 > 0.5:
            ty.text(out, (int(960 + pu + 40), int(90 + pv - 18)), f"{self.depth_m:.1f} m", 34, "mono", alpha=a2)
        lines = [f"pixel ({self.u:.0f}, {self.v:.0f})",
                 f"+ depth {self.depth_m:.1f} m" if self.depth_m else "+ depth",
                 "+ camera pose (position, attitude, gimbal)",
                 (f"= N {self.pos[0]:.1f} m   E {self.pos[1]:.1f} m" if self.pos is not None else "= position"),
                 (f"hiker's true position: N {self.truth[0]:.1f}   E {self.truth[1]:.1f}   →   {self.err:.1f} m off"
                  if self.err is not None else "")]
        for k, s in enumerate(lines):
            a = ty.ease((i - (0.4 + 0.1 * k) * n) / (0.06 * n))
            ty.text(out, (120, 624 + k * 52), s, 36 if k < 4 else 32, "mono" if k < 4 else "semibold",
                    color=(120, 255, 140) if k == 4 else ty.WHITE, alpha=a)
        return out


class RadioShot(MapShot):
    """The team on the map at the moment of a find: the finder's 'FOUND' goes out over the radio."""

    def __init__(self, run: FilmRun, plate: MapPlate, n: int, finder: str, t: float, label: str):
        super().__init__(run, plate, n, "pins", t1=t)
        self.finder, self.t_find, self.label = finder, t, label

    def frame(self, i: int) -> np.ndarray:
        img = super().frame(i)
        src = self._pos_px(self.finder, self.t_find)
        ph = (i % 36) / 36.0
        for k in range(3):
            r = int(30 + 260 * ((ph + k / 3) % 1.0))
            over = img.copy()
            cv2.circle(over, src, r, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.addWeighted(over, 0.6 * (1 - r / 290), img, 1 - 0.6 * (1 - r / 290), 0, img)
        a = ty.ease((i - 15) / 15.0)
        for d in self.run.drones:
            if d == self.finder:
                continue
            dst = self._pos_px(d, self.t_find)
            over = img.copy()
            cv2.line(over, src, dst, ty.ACCENT, 3, cv2.LINE_AA)
            cv2.addWeighted(over, 0.8 * a, img, 1 - 0.8 * a, 0, img)
        ty.panel(img, src[0] + 24, src[1] - 74, src[0] + 34 + ty.text_size(self.label, 30, "semibold")[0] + 20, src[1] - 22,
                 color=ty.ACCENT, alpha=0.9 * a, radius=8)
        ty.text(img, (src[0] + 40, src[1] - 70), self.label, 30, "semibold", color=ty.INK, alpha=a, shadow=False)
        return img
