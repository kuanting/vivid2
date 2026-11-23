"""The VLM damage mapper: the damage layer of the ``vlm`` and ``hybrid`` stacks (docs/damage.md).

For a straight-down survey photo that shows enough new ground, the model gets two images of it:
the photo as taken, and the same photo with a lettered grid (columns A, B, ... left to right, rows
1, 2, ... top to bottom), the known road segments drawn along their centrelines and the known
buildings outlined, each with its label (from the pre-event map the stack is told). One reply
(``damage_schema``) answers every product the task asks for:

    areas      for each area class, the grid cells it covers at least half of
    roads      for each labelled segment: open, blocked or not visible
    buildings  for each labelled building: none, partial, collapsed or not visible
    clues      surface clues: kind and image point

The photo's depth puts a lattice of its pixels on the ground (pixels without depth: where the ray
meets the DEM), so the answers become votes on the task's 2 m cells. A cell is in a class when at
least half of all the looks at it, over every mapped photo, put it there. A road segment is blocked
when its "blocked" answers reach half of its "open" ones, each "open" counted by the share of the
segment that photo showed (a blockage may lie in the part it did not show); a building takes the
grade most answers gave (a tie: the more severe). Both are reported whenever that changes. A clue
placed by depth is reported at once and once only; one placed on the DEM (no depth there) waits for
a better look and is reported at the end if none comes. Area outlines go out at the end of the survey.

Model calls run in the background (``max_in_flight`` at once), so mapping does not slow the person
search; the agent waits for them before reporting the area outlines.
"""
from __future__ import annotations

import re
import string
import threading
from importlib import resources
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from vivid2.damage import CLUE_TOLERANCE_M, DAMAGE_CELL_M, GRADES, GRADE_LEVEL, VIEW_MAX_RANGE_M
from vivid2.damage.geometry import corridor_cells, grid_cells, mask_polygons
from vivid2.geo import focal_px, points_in_polygon, quat_to_rot
from vivid2.stack.base import (AreaReport, BuildingReport, ClueReport, DamageMapper, Frame, MissionSpec, RoadReport,
                               UsageStats)
from vivid2.stack.reference.agents import is_default_view
from vivid2.terrain import TerrainModel
from vivid2.vlm.base import VLMError, VLMProvider, decode_jpeg, encode_jpeg_bytes, vlm_views

GRID = (12, 9)               # columns x rows of the labelled grid (square cells on a 4:3 photo)
LATTICE_PX = 8               # one ground sample every this many photo pixels, each way
NEW_GROUND_SHARE = 0.5       # map a photo when at least this share of the ground it shows is new ...
MIN_HEIGHT_SHARE = 0.6       # ... and it was taken at least this share of the survey height up
MAX_CALLS = 80               # model calls per run, at most
MAX_IN_FLIGHT = 3            # model calls at once
ROAD_IN_PHOTO_SHARE = 0.2    # label a road segment when the photo shows this share of its corridor
BUILDING_IN_PHOTO_SHARE = 0.6  # ... and a building when it shows this share of its footprint
AREA_VOTE = 0.5              # share of the looks at a cell that must put it in a class
ROAD_OPEN_WEIGHT = 0.5       # a segment is blocked when its "blocked" answers reach this share of its "open" ones
ROAD_STATUS = ("open", "blocked", "not_visible")
BUILDING_ANSWERS = GRADES + ("not_visible",)

_ROAD_BGR = [(0, 230, 255), (0, 140, 255), (255, 0, 255), (60, 255, 60), (180, 105, 255), (255, 160, 0),
             (0, 80, 255), (200, 255, 0)]
_BUILDING_BGR = (255, 255, 0)
MM_SATURATION_M = 65.535     # millimetre depth (a frame bank) saturates here: beyond it, no depth


def usable_depth(d: np.ndarray) -> np.ndarray:
    """Depth values that place a point: not missing, not saturated millimetres, within viewing range.

    Live photos carry depth in whole metres, with no 65 m limit; banked photos are millimetres and
    saturate at 65.535 m, so only that value (not everything past 65 m) means "too far to tell".
    """
    return (d > 0.1) & (d < VIEW_MAX_RANGE_M) & (np.abs(d - MM_SATURATION_M) > 0.01)


def damage_schema(products: Sequence[str], area_classes: Sequence[str], clue_kinds: Sequence[str]) -> Dict[str, Any]:
    """The reply's JSON schema: one key per product the task asks for, and a summary."""
    def items(props: Dict[str, Any]) -> Dict[str, Any]:
        return {"type": "array", "items": {"type": "object", "properties": props, "required": list(props),
                                           "additionalProperties": False}}

    props: Dict[str, Any] = {}
    if "areas" in products:
        props["areas"] = items({"area_class": {"type": "string", "enum": list(area_classes)},
                                "cells": {"type": "array", "items": {"type": "string"}}})
    if "roads" in products:
        props["roads"] = items({"segment": {"type": "string"}, "status": {"type": "string", "enum": list(ROAD_STATUS)}})
    if "buildings" in products:
        props["buildings"] = items({"building": {"type": "string"},
                                    "grade": {"type": "string", "enum": list(BUILDING_ANSWERS)}})
    if "clues" in products:
        props["clues"] = items({"kind": {"type": "string", "enum": list(clue_kinds)},
                                "point": {"type": "array", "items": {"type": "number"}},
                                "description": {"type": "string"}})
    props["summary"] = {"type": "string"}
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def load_damage_prompt(mission: str) -> Tuple[str, Dict[str, str]]:
    """``<mission>_damage.md``: the text before the first ``@@ name`` line, and each named section."""
    text = resources.files("vivid2.vlm.prompts").joinpath(f"{mission}_damage.md").read_text(encoding="utf-8")
    parts = re.split(r"^@@ *(\w+) *\n", text, flags=re.M)
    return parts[0].strip(), {name: body.strip() for name, body in zip(parts[1::2], parts[2::2])}


def cell_label(col: int, row: int) -> str:
    return f"{string.ascii_uppercase[col]}{row + 1}"


def parse_cell(label: Any, grid: Tuple[int, int] = GRID) -> Optional[Tuple[int, int]]:
    """'C4' -> (column 2, row 3); None if it is not a cell of the grid."""
    m = re.fullmatch(r"\s*([A-Za-z])\s*(\d{1,2})\s*", str(label))
    if not m:
        return None
    col, row = string.ascii_uppercase.index(m.group(1).upper()), int(m.group(2)) - 1
    return (col, row) if 0 <= col < grid[0] and 0 <= row < grid[1] else None


def rays_to_terrain(cam: np.ndarray, dirs: np.ndarray, terrain: TerrainModel, iterations: int = 4) -> np.ndarray:
    """Where rays from ``cam`` along ``dirs`` (N, 3) meet the terrain: a few flat-ground steps; NaN if never."""
    out = np.full((len(dirs), 3), np.nan)
    if not len(dirs):
        return out
    ok = dirs[:, 2] > 1e-6
    down = np.full(len(dirs), float(terrain.down_at(cam[0], cam[1])))
    p = np.tile(cam, (len(dirs), 1))
    for _ in range(iterations):
        s = (down - cam[2]) / np.where(ok, dirs[:, 2], 1.0)
        ok &= s > 0
        p = cam + np.where(ok, s, 0.0)[:, None] * dirs
        down = np.asarray(terrain.down_at(p[:, 0], p[:, 1]), float)
    out[ok] = p[ok]
    return out


class CellGrid:
    """The task's 2 m cells over the area of interest (``grid_cells``, the scoring's grid), with a
    lookup from ground positions to cells."""

    def __init__(self, aoi_ne: Sequence[Sequence[float]], cell_m: float = DAMAGE_CELL_M):
        aoi = np.asarray(aoi_ne, float)
        self.cell_m = cell_m
        self.cells = grid_cells(aoi, cell_m)
        self.lo = aoi.min(axis=0)
        hi = aoi.max(axis=0)
        self.shape = (len(np.arange(self.lo[0] + cell_m / 2, hi[0], cell_m)),
                      len(np.arange(self.lo[1] + cell_m / 2, hi[1], cell_m)))
        self.index = np.full(self.shape, -1, int)
        if len(self.cells):
            ij = np.floor((self.cells - self.lo) / cell_m).astype(int)
            self.index[ij[:, 0], ij[:, 1]] = np.arange(len(self.cells))

    def lookup(self, ne: np.ndarray) -> np.ndarray:
        """Cell index of each ground point (N, 2); -1 outside the area of interest."""
        ne = np.asarray(ne, float).reshape(-1, 2)
        out = np.full(len(ne), -1, int)
        fin = np.isfinite(ne).all(axis=1)
        ij = np.zeros((len(ne), 2), int)
        ij[fin] = np.floor((ne[fin] - self.lo) / self.cell_m).astype(int)
        ok = fin & (ij >= 0).all(axis=1) & (ij[:, 0] < self.shape[0]) & (ij[:, 1] < self.shape[1])
        out[ok] = self.index[ij[ok, 0], ij[ok, 1]]
        return out

    def polygons(self, selected: np.ndarray) -> List[Dict[str, Any]]:
        """Outlines of the selected cells (``mask_polygons``): they rasterise back to exactly these cells."""
        mask = np.zeros(self.shape, bool)
        has = self.index >= 0
        mask[has] = selected[self.index[has]]
        return mask_polygons(mask, self.lo[0] + self.cell_m / 2, self.lo[1] + self.cell_m / 2, self.cell_m)


class PhotoGround:
    """Where a photo's pixels lie on the ground: a lattice of pixels (``step_px`` apart), each placed
    by the photo's planar depth or, without depth, where its ray meets the terrain."""

    def __init__(self, frame: Frame, terrain: TerrainModel, grid: CellGrid, step_px: int = LATTICE_PX,
                 boxes: Tuple[int, int] = GRID):
        self.frame = frame
        W, H = frame.width, frame.height
        self.W, self.H, self.boxes = W, H, boxes
        self.cam = np.asarray(frame.cam_pos, float)
        self.R = quat_to_rot(frame.cam_q)
        self.f = focal_px(W, frame.fov_deg)
        uu, vv = np.meshgrid(np.arange(step_px / 2.0, W, step_px), np.arange(step_px / 2.0, H, step_px))
        self.u, self.v = uu.ravel(), vv.ravel()
        dirs = (self.R @ np.stack([np.ones_like(self.u), (self.u - W / 2.0) / self.f, (self.v - H / 2.0) / self.f])).T
        pts = np.full((len(self.u), 3), np.nan)
        if frame.depth is not None:
            dh, dw = frame.depth.shape[:2]
            d = frame.depth[np.clip((self.v * dh / H).astype(int), 0, dh - 1),
                            np.clip((self.u * dw / W).astype(int), 0, dw - 1)].astype(float)
            ok = usable_depth(d)
            pts[ok] = self.cam + d[ok, None] * dirs[ok]
        miss = np.isnan(pts[:, 0])
        if miss.any():
            pts[miss] = rays_to_terrain(self.cam, dirs[miss], terrain)
        self.ne, self.down = pts[:, :2], pts[:, 2]
        self.cell = grid.lookup(self.ne)
        self.box = (np.minimum((self.v / H * boxes[1]).astype(int), boxes[1] - 1) * boxes[0] +
                    np.minimum((self.u / W * boxes[0]).astype(int), boxes[0] - 1))
        on = self.cell >= 0
        n = len(grid.cells)
        self.hits = np.bincount(self.cell[on], minlength=n).astype(float)
        down_sum = np.bincount(self.cell[on], weights=self.down[on], minlength=n)
        with np.errstate(invalid="ignore", divide="ignore"):
            self.cell_down = np.where(self.hits > 0, down_sum / np.maximum(self.hits, 1), np.nan)
        self._terrain, self._grid = terrain, grid

    def hits_in(self, boxes: Sequence[int]) -> np.ndarray:
        """Per cell, the lattice points that fall in the given grid boxes (``row * columns + column``)."""
        sel = (self.cell >= 0) & np.isin(self.box, list(boxes))
        return np.bincount(self.cell[sel], minlength=len(self.hits)).astype(float)

    def surface_down(self, ne: np.ndarray) -> np.ndarray:
        """The photo's ground (NED down) at points: the mean depth point of their cell, else the terrain."""
        ne = np.asarray(ne, float).reshape(-1, 2)
        idx = self._grid.lookup(ne)
        out = np.asarray(self._terrain.down_at(ne[:, 0], ne[:, 1]), float).reshape(-1)
        have = idx >= 0
        cd = np.full(len(ne), np.nan)
        cd[have] = self.cell_down[idx[have]]
        return np.where(np.isfinite(cd), cd, out)

    def project(self, ned: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Photo pixels (u, v) of NED points, and whether each is in front of the camera and in the photo."""
        rel = np.asarray(ned, float).reshape(-1, 3) - self.cam
        pc = rel @ self.R
        front = pc[:, 0] > 1e-6
        x = np.where(front, pc[:, 0], 1.0)
        u, v = self.W / 2.0 + self.f * pc[:, 1] / x, self.H / 2.0 + self.f * pc[:, 2] / x
        return u, v, front & (u >= 0) & (u < self.W) & (v >= 0) & (v < self.H)

    def locate(self, u: float, v: float) -> Tuple[Optional[np.ndarray], bool]:
        """NED of the surface at a photo pixel, and whether depth placed it: the median depth around the
        pixel, else (no depth, or beyond the depth camera's range) the ray to the terrain."""
        dirs = (self.R @ np.array([1.0, (u - self.W / 2.0) / self.f, (v - self.H / 2.0) / self.f]))[None, :]
        depth = self.frame.depth
        if depth is not None:
            dh, dw = depth.shape[:2]
            x, y = int(u * dw / self.W), int(v * dh / self.H)
            patch = depth[max(y - 2, 0):y + 3, max(x - 2, 0):x + 3]
            valid = patch[usable_depth(patch)]
            if valid.size:
                return self.cam + float(np.median(valid)) * dirs[0], True
        p = rays_to_terrain(self.cam, dirs, self._terrain)[0]
        return (None if np.isnan(p[0]) else p), False


class MapperPhoto:
    """One photo the mapper asks about: where it lies, what is labelled on it, and the images sent."""

    def __init__(self, frame: Frame, ground: PhotoGround, roads: List[str], buildings: List[str]):
        self.frame, self.ground, self.roads, self.buildings = frame, ground, roads, buildings
        self.road_share: Dict[str, float] = {}  # share of each labelled segment's corridor the photo shows
        self.images: List[bytes] = []
        self.prompt = ""


class VLMDamageMapper(DamageMapper):
    """Maps the damage from a VLM's answers about labelled survey photos (module docstring)."""

    name = "vlm_damage"

    def __init__(self, provider: VLMProvider, grid: Tuple[int, int] = GRID, new_ground_share: float = NEW_GROUND_SHARE,
                 max_calls: int = MAX_CALLS, max_in_flight: int = MAX_IN_FLIGHT, lattice_px: int = LATTICE_PX):
        if not (1 <= grid[0] <= 26 and 1 <= grid[1] <= 26):
            raise ValueError(f"grid {grid} out of range (1..26 each way)")
        self.provider = provider
        self.grid = tuple(grid)
        self.new_ground_share = new_ground_share
        self.max_calls = max_calls
        self.max_in_flight = max_in_flight  # the agent runs observe() in the background, this many at once
        self.lattice_px = lattice_px
        self.on_photo = None  # called with each MapperPhoto before it is sent (offline evaluation saves them)
        self._lock = threading.Lock()
        self._ready = False
        self.calls = 0

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "model": getattr(self.provider, "model", None), "grid": f"{self.grid[0]}x{self.grid[1]}",
                "new_ground_share": self.new_ground_share, "max_calls": self.max_calls,
                "max_in_flight": self.max_in_flight, "areas": "grid_cell_votes", "roads": "majority",
                "buildings": "majority"}

    # -- set-up from what the stack is told ------------------------------------------------------
    def _setup(self, mission: MissionSpec) -> None:
        d = mission.damage
        self.products = list(d["products"])
        self.cells = CellGrid(d["aoi_ne"])
        n = len(self.cells.cells)
        self.area_classes: Dict[str, str] = dict(d.get("area_classes") or {}) if "areas" in self.products else {}
        self.clue_kinds: List[str] = list(d.get("clue_kinds") or [])
        self.roads = {r["id"]: r for r in (d.get("roads") or [])} if "roads" in self.products else {}
        self.buildings = {b["id"]: b for b in (d.get("buildings") or [])} if "buildings" in self.products else {}
        self.road_cells = {k: corridor_cells(r["polyline_ne"], r["width_m"], self.cells.cells, self.cells.cell_m)
                           for k, r in self.roads.items()}
        self.building_cells = {}
        for k, b in self.buildings.items():
            idx = np.flatnonzero(points_in_polygon(self.cells.cells, b["footprint_ne"])) if n else np.zeros(0, int)
            if not len(idx) and n:
                c = np.mean(b["footprint_ne"], axis=0)
                idx = np.array([int(np.argmin(np.linalg.norm(self.cells.cells - c, axis=1)))])
            self.building_cells[k] = idx
        self.terrain = mission.terrain if mission.terrain is not None else TerrainModel.flat(mission.ground_down_m)
        self.schema = damage_schema(self.products, list(self.area_classes), self.clue_kinds)
        self.intro, self.sections = load_damage_prompt(mission.mission)
        # votes
        self.selected = np.zeros(n, bool)       # ground shown by the photos picked so far
        self.looks = np.zeros(n)                # lattice points that fell in each cell, over the mapped photos
        self.class_looks = {k: np.zeros(n) for k in self.area_classes}  # ... that the model put in each class
        self.class_photo: Dict[str, Tuple[int, int]] = {}  # class -> (frame_id, t_ns) of a photo that showed it
        self.road_votes = {k: {"open": 0.0, "blocked": 0.0} for k in self.roads}
        self.building_votes = {k: {g: 0 for g in GRADES} for k in self.buildings}
        self.road_state: Dict[str, bool] = {}
        self.building_state: Dict[str, str] = {}
        self.clue_seen: List[Dict[str, Any]] = []  # {"best": sighting, "reported": bool}, one per clue seen
        self.new_share: Dict[int, float] = {}
        self._ready = True

    # -- which photos ------------------------------------------------------------------------------
    def wants(self, frame: Frame, mission: MissionSpec) -> bool:
        """A straight-down search photo from near survey height that shows enough new ground."""
        if not self._ready:
            self._setup(mission)
        if self.calls >= self.max_calls or not len(self.cells.cells) or not is_default_view(frame):
            return False
        cam = np.asarray(frame.cam_pos, float)
        if float(self.terrain.down_at(cam[0], cam[1])) - cam[2] < MIN_HEIGHT_SHARE * float(mission.altitude_m):
            return False
        W, H, f = frame.width, frame.height, focal_px(frame.width, frame.fov_deg)
        corners = np.array([[0, 0], [W, 0], [W, H], [0, H]], float)
        dirs = (quat_to_rot(frame.cam_q) @ np.stack([np.ones(4), (corners[:, 0] - W / 2) / f,
                                                      (corners[:, 1] - H / 2) / f])).T
        quad = rays_to_terrain(cam, dirs, self.terrain)
        if np.isnan(quad).any():
            return False
        inside = points_in_polygon(self.cells.cells, quad[:, :2])
        if not inside.any():
            return False
        share = float((~self.selected[inside]).mean())
        if share < self.new_ground_share:
            return False
        self.selected[inside] = True
        self.new_share[frame.frame_id] = round(share, 3)
        self.calls += 1
        return True

    # -- one photo ----------------------------------------------------------------------------------
    def prepare(self, frame: Frame, mission: MissionSpec) -> MapperPhoto:
        """Where the photo lies, what is labelled on it, and the images and prompt to send."""
        if not self._ready:
            self._setup(mission)
        ground = PhotoGround(frame, self.terrain, self.cells, self.lattice_px, self.grid)
        road_share = {k: float((ground.hits[idx] > 0).mean()) for k, idx in self.road_cells.items() if len(idx)}
        roads = [k for k, share in road_share.items() if share >= ROAD_IN_PHOTO_SHARE]
        buildings = [k for k, idx in self.building_cells.items()
                     if len(idx) and float((ground.hits[idx] > 0).mean()) >= BUILDING_IN_PHOTO_SHARE]
        photo = MapperPhoto(frame, ground, roads, buildings)
        photo.road_share = {k: road_share[k] for k in roads}
        clean = vlm_views(frame.jpeg if frame.jpeg is not None else frame.rgb, "1x1")[0]
        photo.images = [clean.jpeg, self._overlay(clean.jpeg, photo)]
        photo.prompt = self._prompt(mission, photo)
        return photo

    def observe(self, frame: Frame, mission: MissionSpec):
        photo = self.prepare(frame, mission)
        if self.on_photo is not None:
            self.on_photo(photo)
        rec: Dict[str, Any] = {"mapper": self.name, "new_ground": self.new_share.get(frame.frame_id),
                               "labelled": {"roads": photo.roads, "buildings": photo.buildings}}
        try:
            out, usage = self._ask(photo)
        except VLMError as e:
            return [], UsageStats(model=getattr(self.provider, "model", "")), {**rec, "error": str(e)}
        try:
            answer = self._parse(out, photo)
        except ValueError as e:
            return [], usage, {**rec, "error": f"invalid reply: {e}", "reply": out}
        with self._lock:
            reports = self._apply(answer, photo)
        rec.update({k: v for k, v in answer.items() if k != "boxes"})
        return reports, usage, rec

    def _ask(self, photo: MapperPhoto):
        """The model's reply for a photo (a test or an offline evaluation may answer otherwise)."""
        return self.provider.query(photo.images, photo.prompt, self.schema)

    # -- what is sent --------------------------------------------------------------------------------
    def _prompt(self, mission: MissionSpec, photo: MapperPhoto) -> str:
        cols, rows = self.grid
        fill = {
            "altitude_m": float(mission.altitude_m), "cols": cols, "rows": rows,
            "first_col": "A", "last_col": string.ascii_uppercase[cols - 1],
            "area_classes": "\n".join(f"- {k}: {v}" if v else f"- {k}" for k, v in self.area_classes.items()),
            "roads": ", ".join(photo.roads), "buildings": ", ".join(photo.buildings),
            "clue_kinds": ", ".join(self.clue_kinds),
        }
        parts = [self.intro.format(**fill)]
        for product in ("areas", "roads", "buildings", "clues"):
            if product not in self.products:
                continue
            key = product
            if product == "roads" and not photo.roads:
                key = "roads_none"
            elif product == "buildings" and not photo.buildings:
                key = "buildings_none"
            if key in self.sections:
                parts.append(self.sections[key].format(**fill))
        if "end" in self.sections:
            parts.append(self.sections["end"].format(**fill))
        return "\n\n".join(parts)

    def _overlay(self, jpeg: bytes, photo: MapperPhoto) -> bytes:
        """The photo with the grid, the labelled road segments and the outlined buildings drawn on it."""
        import cv2

        img = decode_jpeg(jpeg)
        h, w = img.shape[:2]
        g = photo.ground
        sx, sy = w / g.W, h / g.H
        cols, rows = self.grid
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.4, w / 3000.0)
        thick = max(1, round(w / 1000.0))

        def label(text, x, y, bg, fg=(0, 0, 0)):
            (tw, th), base = cv2.getTextSize(text, font, scale * 1.2, thick)
            x = int(np.clip(x - tw / 2, 0, w - tw - 4))
            y = int(np.clip(y, th + 4, h - base - 2))
            cv2.rectangle(img, (x - 2, y - th - 3), (x + tw + 2, y + base + 1), bg, -1)
            cv2.putText(img, text, (x, y), font, scale * 1.2, fg, thick, cv2.LINE_AA)

        for k, bid in enumerate(photo.buildings):
            fp = np.asarray(self.buildings[bid]["footprint_ne"], float)
            u, v, _ = g.project(np.column_stack([fp, g.surface_down(fp)]))
            pts = np.column_stack([u * sx, v * sy]).round().astype(np.int32)
            cv2.polylines(img, [pts], True, (0, 0, 0), thick + 2, cv2.LINE_AA)
            cv2.polylines(img, [pts], True, _BUILDING_BGR, thick, cv2.LINE_AA)
            label(bid, float(pts[:, 0].mean()), float(pts[:, 1].mean()), _BUILDING_BGR)
        for k, sid in enumerate(photo.roads):
            line = np.asarray(self.roads[sid]["polyline_ne"], float)
            seg = np.diff(line, axis=0)
            steps = [np.linspace(0, 1, max(2, int(np.ceil(np.linalg.norm(s))) + 1)) for s in seg]
            dense = np.concatenate([line[i] + t[:, None] * seg[i] for i, t in enumerate(steps)])
            u, v, inside = g.project(np.column_stack([dense, g.surface_down(dense)]))
            colour = _ROAD_BGR[k % len(_ROAD_BGR)]
            pts = np.column_stack([u * sx, v * sy])
            runs, cur = [], []
            for p, ok in zip(pts, inside):
                if ok:
                    cur.append(p)
                elif cur:
                    runs.append(cur)
                    cur = []
            if cur:
                runs.append(cur)
            for r in runs:
                arr = np.asarray(r).round().astype(np.int32)
                cv2.polylines(img, [arr], False, (0, 0, 0), thick + 4, cv2.LINE_AA)
                cv2.polylines(img, [arr], False, colour, thick + 2, cv2.LINE_AA)
            for end in (0, len(dense) - 1):  # where the segment starts and ends
                if inside[end]:
                    cv2.circle(img, (int(pts[end, 0]), int(pts[end, 1])), 3 * thick + 3, colour, -1, cv2.LINE_AA)
            if runs:
                longest = max(runs, key=len)
                mid = longest[len(longest) // 2]
                label(sid, float(mid[0]), float(mid[1]) - 6 * thick, colour)
        for c in range(1, cols):
            x = round(c * w / cols)
            cv2.line(img, (x, 0), (x, h), (0, 0, 0), thick + 2)
            cv2.line(img, (x, 0), (x, h), (255, 255, 255), thick)
        for r in range(1, rows):
            y = round(r * h / rows)
            cv2.line(img, (0, y), (w, y), (0, 0, 0), thick + 2)
            cv2.line(img, (0, y), (w, y), (255, 255, 255), thick)
        for r in range(rows):
            for c in range(cols):
                text = cell_label(c, r)
                (tw, th), base = cv2.getTextSize(text, font, scale, thick)
                x0, y0 = round(c * w / cols) + 3, round(r * h / rows) + 3
                cv2.rectangle(img, (x0, y0), (x0 + tw + 4, y0 + th + base + 3), (0, 0, 0), -1)
                cv2.putText(img, text, (x0 + 2, y0 + th + 2), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
        return encode_jpeg_bytes(img, max(w, h))

    # -- what comes back -------------------------------------------------------------------------------
    def _parse(self, out: Any, photo: MapperPhoto) -> Dict[str, Any]:
        """The reply, cleaned: unknown classes, cells, ids, grades and points are dropped (and listed)."""
        if not isinstance(out, dict):
            raise ValueError("reply is not an object")
        missing = [k for k in self.schema["required"] if k != "summary" and k not in out]
        if missing:
            raise ValueError(f"reply has no {', '.join(missing)}")
        cols, rows = self.grid
        ans: Dict[str, Any] = {"summary": str(out.get("summary") or "")}
        ignored: List[str] = []
        if "areas" in self.products:
            cells: Dict[str, List[str]] = {}
            boxes: Dict[str, set] = {}
            for a in out.get("areas") or []:
                k = a.get("area_class") if isinstance(a, dict) else None
                if k not in self.area_classes:
                    ignored.append(f"area class {k!r}")
                    continue
                for lab in a.get("cells") or []:
                    cr = parse_cell(lab, self.grid)
                    if cr is None:
                        ignored.append(f"cell {lab!r}")
                        continue
                    if cr[1] * cols + cr[0] not in boxes.setdefault(k, set()):
                        boxes[k].add(cr[1] * cols + cr[0])
                        cells.setdefault(k, []).append(cell_label(*cr))
            ans["areas"], ans["boxes"] = cells, boxes
        if "roads" in self.products:
            roads = {}
            for r in out.get("roads") or []:
                sid, status = (r.get("segment"), r.get("status")) if isinstance(r, dict) else (None, None)
                if sid not in photo.roads or status not in ROAD_STATUS:
                    ignored.append(f"road {sid!r} {status!r}")
                    continue
                roads[sid] = status
            ans["roads"] = roads
        if "buildings" in self.products:
            blds = {}
            for b in out.get("buildings") or []:
                bid, grade = (b.get("building"), b.get("grade")) if isinstance(b, dict) else (None, None)
                if bid not in photo.buildings or grade not in BUILDING_ANSWERS:
                    ignored.append(f"building {bid!r} {grade!r}")
                    continue
                blds[bid] = grade
            ans["buildings"] = blds
        if "clues" in self.products:
            clues = []
            for c in out.get("clues") or []:
                if not isinstance(c, dict) or c.get("kind") not in self.clue_kinds:
                    ignored.append(f"clue {c!r}"[:80])
                    continue
                pt = c.get("point")
                try:
                    x, y = float(pt[0]), float(pt[1])
                except (TypeError, ValueError, IndexError):
                    ignored.append(f"clue point {pt!r}")
                    continue
                if not (0 <= x <= 1000 and 0 <= y <= 1000):
                    ignored.append(f"clue point {pt!r}")
                    continue
                clues.append({"kind": c["kind"], "point": [round(x, 1), round(y, 1)],
                              "description": str(c.get("description") or "")})
            ans["clues"] = clues
        if ignored:
            ans["ignored"] = ignored
        return ans

    def _apply(self, ans: Dict[str, Any], photo: MapperPhoto) -> List[Any]:
        """Add one photo's answers to the votes; the road, building and clue reports that changes."""
        g, f = photo.ground, photo.frame
        out: List[Any] = []
        self.looks += g.hits
        for k, boxes in (ans.get("boxes") or {}).items():
            if boxes:
                self.class_looks[k] += g.hits_in(boxes)
                self.class_photo[k] = (f.frame_id, f.t_ns)
        for sid, status in (ans.get("roads") or {}).items():
            if status == "not_visible":
                continue
            # A photo that saw a blockage counts in full; one that saw the road open, by the share of
            # the segment it showed (the blockage may be in the part it did not show).
            votes = self.road_votes[sid]
            votes[status] += 1.0 if status == "blocked" else photo.road_share.get(sid, 1.0)
            blocked = votes["blocked"] > 0 and votes["blocked"] >= ROAD_OPEN_WEIGHT * votes["open"]
            if self.road_state.get(sid) != blocked:
                self.road_state[sid] = blocked
                out.append(RoadReport(sid, blocked, f.frame_id, f.t_ns,
                                      note=f"{votes['blocked']:.0f} blocked / {votes['open']:.2f} open"))
        for bid, grade in (ans.get("buildings") or {}).items():
            if grade == "not_visible":
                continue
            votes = self.building_votes[bid]
            votes[grade] += 1
            top = max(votes.values())
            best = max((gr for gr in GRADES if votes[gr] == top), key=GRADE_LEVEL.get)  # a tie: the worse grade
            if self.building_state.get(bid) != best:
                self.building_state[bid] = best
                out.append(BuildingReport(bid, best, f.frame_id, f.t_ns,
                                          note=" / ".join(f"{votes[gr]} {gr}" for gr in GRADES)))
        for c in ans.get("clues") or []:
            u, v = c["point"][0] / 1000.0 * g.W, c["point"][1] / 1000.0 * g.H
            p, by_depth = g.locate(u, v)
            if p is None:
                continue
            c["ned"] = [round(float(x), 2) for x in p]
            sighting = {"ned": [float(x) for x in p], "kind": c["kind"], "by_depth": by_depth,
                        "off_centre": float(np.hypot(u / g.W - 0.5, v / g.H - 0.5)), "frame_id": f.frame_id,
                        "t_ns": f.t_ns, "note": c["description"][:120]}
            seen = self._clue_near(p, by_depth)
            if seen is None:
                seen = {"best": sighting, "reported": False}
                self.clue_seen.append(seen)
            elif seen["reported"]:
                c["repeat"] = True
                continue
            elif (by_depth, -sighting["off_centre"]) > (seen["best"]["by_depth"], -seen["best"]["off_centre"]):
                seen["best"] = sighting
            if seen["best"]["by_depth"]:  # placed by depth: report at once
                seen["reported"] = True
                out.append(self._clue_report(seen["best"]))
            else:
                c["held"] = True  # placed on the DEM only: wait for a better look, else report it at the end
        return out

    def _clue_near(self, p: np.ndarray, by_depth: bool) -> Optional[Dict[str, Any]]:
        """The clue already seen that a sighting at ``p`` is (within the clue tolerance, or twice that
        when either was placed on the DEM only), or None."""
        best, best_d = None, float("inf")
        for s in self.clue_seen:
            q = np.asarray(s["best"]["ned"][:2])
            d = float(np.linalg.norm(p[:2] - q))
            reach = CLUE_TOLERANCE_M * (1.0 if by_depth and s["best"]["by_depth"] else 2.0)
            if d <= reach and d < best_d:
                best, best_d = s, d
        return best

    @staticmethod
    def _clue_report(s: Dict[str, Any]) -> ClueReport:
        return ClueReport(s["ned"], s["kind"], s["frame_id"], s["t_ns"], note=s["note"])

    # -- the end of the survey -------------------------------------------------------------------------
    def finish(self, mission: MissionSpec) -> List[Any]:
        """Clues held for a better look, and area outlines: the cells where at least ``AREA_VOTE`` of
        the looks put them in the class."""
        if not self._ready:
            return []
        out: List[Any] = []
        with self._lock:
            for s in self.clue_seen:
                if not s["reported"]:
                    s["reported"] = True
                    out.append(self._clue_report(s["best"]))
            for k in self.area_classes:
                if k not in self.class_photo:
                    continue
                sel = (self.looks > 0) & (self.class_looks[k] >= AREA_VOTE * self.looks)
                fid, t = self.class_photo[k]
                for p in self.cells.polygons(sel):
                    out.append(AreaReport(k, p["polygon_ne"], fid, t, op=p["op"], note="vlm grid votes"))
        return out
