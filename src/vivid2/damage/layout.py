"""A map's layout -> its damage ground truth, surface model, Unreal masks and placements.

``unreal/VIVID2Env/SourceData/<map>/layout.json`` is the one source for what a disaster did to a
map; ``vivid2 damage build`` turns it into everything else, so the pictures and the ground truth
come from the same numbers::

    map, about
    mask {size_px, extent_m}                  region masks for Unreal (vivid2.damage.regions)
    grid {n0, e0, size [nN, nE], cell_m, ground_down_m}   maps without a terrain model only
    regions {name: {polygon_ne | polylines_ne + width_m, edge {noise_m, noise_scale_m, soft_m, seed}}}
    area_classes {name: {regions: [..] | water: true, about, blocks_roads (default true)}}
    roads [{id, polyline_ne, width_m, kind, segment_m (<= 100)}]
    buildings [{id, asset, ne, yaw_deg, size_m [l, w, h] | footprint_ne + height_m, roof, use,
                ems98 | (grade from grade_from), post_height_m, rubble_m, spill_m, debris_ne, variant}]
    grade_from "<area class>"                  buildings without ems98: graded by how much of
                                               their footprint that class covers
    blockers [{polygon_ne, cause}]
    water {level_down_m | level_m_msl, polygon_ne}
    props [{name, asset, ne, yaw_deg, z_offset_m}]   placement only
    expect {blocked: [segment ids], grades: {building: grade}}   the build fails if these differ
    terrain {...}                              terrain edits (vivid2.terrain_edit), not used here

Ground truth rules (docs/damage.md): an area class covers a node where its regions' field is at
least REGION_THRESHOLD (or, for ``water``, where the ground lies under the water level). A road
segment is blocked when some BLOCK_WINDOW_M stretch has BLOCK_COVER of its corridor covered by a
road-blocking class, water BLOCK_MIN_WATER_M deep, a collapsed building's debris, or a blocker.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from vivid2.damage import (BLOCK_COVER, BLOCK_MIN_WATER_M, BLOCK_WINDOW_M, CORRIDOR_STEP_M, EMS98_TO_GRADE, GRADES,
                           REGION_THRESHOLD)
from vivid2.damage import regions as rg
from vivid2.damage.geometry import corridor, distance_to_polygon, in_any_polygon, rect_polygon, split_polyline
from vivid2.geo import points_in_polygon
from vivid2.terrain import TerrainModel

DEFAULT_SEGMENT_M = 100.0
DEFAULT_RUBBLE_M = 2.0
COLLAPSE_SPILL_SHARE = 0.3  # a collapsed building's debris reaches this share of its height beyond its walls
GRADE_FROM_COLLAPSED = 0.5  # share of a footprint under grade_from's class: collapsed (EMS-98 5) ...
GRADE_FROM_PARTIAL = 0.15   # ... partial (EMS-98 3)


@dataclass
class DamageBuild:
    doc: Dict[str, Any]                       # damage_gt.json without file pointers
    classes: np.ndarray
    debris_m: np.ndarray
    water_m: Optional[np.ndarray]
    grid: Tuple[float, float, float]          # n0, e0, cell_m
    surface: TerrainModel
    placement: Dict[str, Any]
    masks: Dict[str, np.ndarray] = field(default_factory=dict)  # region name -> 8-bit png
    problems: List[str] = field(default_factory=list)


def _grid(layout: Dict[str, Any], post: Optional[TerrainModel]) -> TerrainModel:
    """The ground truth grid: the terrain model's, or (flat maps) the layout's own."""
    if post is not None and not post.is_flat:
        return post
    g = layout.get("grid")
    if not g:
        raise ValueError("a map without a terrain model needs layout.grid {n0, e0, size, cell_m, ground_down_m}")
    nn, ne = (int(v) for v in g["size"])
    down = float(g.get("ground_down_m", 0.0 if post is None else float(post.down)))
    return TerrainModel(np.full((nn, ne), down), float(g["n0"]), float(g["e0"]), float(g["cell_m"]), "flat grid")


def _nodes(t: TerrainModel) -> Tuple[np.ndarray, np.ndarray]:
    nn, ne = t.down.shape
    n = t.n0 + np.arange(nn) * t.cell_m
    e = t.e0 + np.arange(ne) * t.cell_m
    return np.meshgrid(n, e, indexing="ij")


def _poly_nodes(grid: TerrainModel, poly) -> Tuple[np.ndarray, np.ndarray]:
    """Indices (i, j) of the grid nodes inside a polygon (searched within its bounding box only)."""
    p = np.asarray(poly, float)
    nn, ne = grid.down.shape
    i0 = int(np.clip(np.floor((p[:, 0].min() - grid.n0) / grid.cell_m), 0, nn - 1))
    i1 = int(np.clip(np.ceil((p[:, 0].max() - grid.n0) / grid.cell_m), 0, nn - 1))
    j0 = int(np.clip(np.floor((p[:, 1].min() - grid.e0) / grid.cell_m), 0, ne - 1))
    j1 = int(np.clip(np.ceil((p[:, 1].max() - grid.e0) / grid.cell_m), 0, ne - 1))
    ii, jj = np.meshgrid(np.arange(i0, i1 + 1), np.arange(j0, j1 + 1), indexing="ij")
    pts = np.column_stack([grid.n0 + ii.ravel() * grid.cell_m, grid.e0 + jj.ravel() * grid.cell_m])
    sel = points_in_polygon(pts, p)
    return ii.ravel()[sel], jj.ravel()[sel]


def _footprint(b: Dict[str, Any], catalog: Dict[str, Any]) -> Tuple[List[List[float]], float]:
    if b.get("footprint_ne"):
        return [list(map(float, p)) for p in b["footprint_ne"]], float(b["height_m"])
    size = b.get("size_m") or (catalog.get(b.get("asset", ""), {}) or {}).get("size_m")
    if not size:
        raise ValueError(f"building {b['id']}: give size_m [length, width, height] or footprint_ne + height_m, "
                         f"or measure its asset ({b.get('asset')}) into asset_catalog.json")
    return rect_polygon(b["ne"], float(size[0]), float(size[1]), float(b.get("yaw_deg", 0.0))), float(b.get("height_m", size[2]))


def build_damage(layout: Dict[str, Any], post: Optional[TerrainModel] = None, pre: Optional[TerrainModel] = None,
                 catalog: Optional[Dict[str, Any]] = None, home_alt: Optional[float] = None,
                 landsea: Optional[np.ndarray] = None) -> DamageBuild:
    """Ground truth from a layout on the post-event terrain (``pre``: the terrain before, same grid)."""
    catalog = catalog or {}
    grid = _grid(layout, post)
    N, E = _nodes(grid)
    ground = grid.down
    problems: List[str] = []

    # -- regions and area classes ------------------------------------------------------------
    m = layout.get("mask") or {}
    flds = rg.fields(layout) if layout.get("regions") else {}
    extent = float(m.get("extent_m", 0.0))
    at_nodes = {k: rg.sample(f, N, E, extent) for k, f in flds.items()}
    debris_m = np.zeros(ground.shape, np.float32)
    if pre is not None and not pre.is_flat:
        if pre.down.shape != ground.shape or (pre.n0, pre.e0, pre.cell_m) != (grid.n0, grid.e0, grid.cell_m):
            raise ValueError("the pre-event terrain must be on the post-event terrain's grid")
        debris_m = (pre.down - ground).astype(np.float32)  # NED down: raised ground has a smaller down

    water = layout.get("water")
    water_m = None
    level_down = None
    if water:
        if "level_down_m" in water:
            level_down = float(water["level_down_m"])
        else:
            if home_alt is None:
                raise ValueError("water.level_m_msl needs the map's home altitude (terrain.json)")
            level_down = float(home_alt) - float(water["level_m_msl"])
        ii, jj = _poly_nodes(grid, water["polygon_ne"])
        water_m = np.zeros(ground.shape, np.float32)
        water_m[ii, jj] = np.maximum(ground[ii, jj] - level_down, 0.0)

    classes = np.zeros(ground.shape, np.uint8)
    area_doc: Dict[str, Dict[str, Any]] = {}
    for bit, (name, ac) in enumerate((layout.get("area_classes") or {}).items()):
        if bit > 7:
            raise ValueError("at most 8 area classes")
        if ac.get("water"):
            if water_m is None:
                raise ValueError(f"area class {name}: water, but the layout has no water")
            on = water_m > 0.0
            if landsea is not None:
                on &= landsea.astype(bool)  # the permanent sea is not flooded land
        else:
            missing = [r for r in ac.get("regions", []) if r not in at_nodes]
            if missing or not ac.get("regions"):
                raise ValueError(f"area class {name}: unknown or no regions {missing}")
            on = np.max([at_nodes[r] for r in ac["regions"]], axis=0) >= REGION_THRESHOLD
        classes |= (on.astype(np.uint8) << bit)
        area_doc[name] = {"bit": bit, "about": ac.get("about", ""), "blocks_roads": bool(ac.get("blocks_roads", True))}

    def class_at(name: str, pts: np.ndarray) -> np.ndarray:
        i = np.clip(np.rint((pts[:, 0] - grid.n0) / grid.cell_m), 0, ground.shape[0] - 1).astype(int)
        j = np.clip(np.rint((pts[:, 1] - grid.e0) / grid.cell_m), 0, ground.shape[1] - 1).astype(int)
        return (classes[i, j] >> area_doc[name]["bit"] & 1).astype(bool)

    def water_at(pts: np.ndarray) -> np.ndarray:
        if water_m is None:
            return np.zeros(len(pts))
        inside = points_in_polygon(pts, water["polygon_ne"])
        return np.where(inside, np.maximum(grid.down_at(pts[:, 0], pts[:, 1]) - level_down, 0.0), 0.0)

    # -- buildings ----------------------------------------------------------------------------
    buildings = []
    grade_from = layout.get("grade_from")
    for b in layout.get("buildings", []):
        fp, height = _footprint(b, catalog)
        ii, jj = _poly_nodes(grid, fp)
        under = np.column_stack([N[ii, jj], E[ii, jj]])
        base = float(np.mean(ground[ii, jj])) if len(ii) else float(grid.down_at(*np.asarray(fp).mean(axis=0)))
        ems = b.get("ems98")
        cause = b.get("cause", "")
        if ems is None and grade_from:
            share = float(class_at(grade_from, under).mean()) if len(under) else 0.0
            ems = 5 if share >= GRADE_FROM_COLLAPSED else 3 if share >= GRADE_FROM_PARTIAL else 0
            cause = cause or (grade_from if ems else "")
        ems = int(ems or 0)
        grade = EMS98_TO_GRADE[ems]
        rec = {"id": b["id"], "footprint_ne": [[round(x, 3) for x in p] for p in fp], "height_m": round(height, 3),
               "base_down_m": round(base, 3), "roof": b.get("roof", "flat"), "use": b.get("use", "building"),
               "ems98": ems, "grade": grade, "cause": cause}
        if grade == "collapsed":
            rec["spill_m"] = float(b.get("spill_m", round(COLLAPSE_SPILL_SHARE * height, 2)))
            if b.get("debris_ne"):
                rec["debris_ne"] = b["debris_ne"]
        rec["post_height_m"] = float(b.get("post_height_m", height if grade != "collapsed"
                                           else b.get("rubble_m", DEFAULT_RUBBLE_M)))
        buildings.append(rec)

    def collapsed_debris(pts: np.ndarray) -> Dict[str, np.ndarray]:
        out = {}
        for b in buildings:
            if b["grade"] != "collapsed":
                continue
            if b.get("debris_ne"):
                out[b["id"]] = points_in_polygon(pts, b["debris_ne"])
            else:
                out[b["id"]] = distance_to_polygon(pts, b["footprint_ne"]) <= b["spill_m"]
        return out

    # -- roads -------------------------------------------------------------------------------
    roads = []
    blockers = layout.get("blockers", [])
    for r in layout.get("roads", []):
        pieces = split_polyline(r["polyline_ne"], float(r.get("segment_m", DEFAULT_SEGMENT_M)))
        for k, piece in enumerate(pieces, 1):
            sid = f"{r['id']}-{k:02d}"
            along, pts = corridor(piece, float(r["width_m"]))
            covered = np.zeros(len(pts), bool)
            causes = set()
            for name, ac in area_doc.items():
                if ac["blocks_roads"]:
                    hit = class_at(name, pts)
                    covered |= hit
                    if hit.any():
                        causes.add(name)
            deep = water_at(pts) >= BLOCK_MIN_WATER_M
            if deep.any():
                causes.add("water")
            covered |= deep
            for bid, hit in collapsed_debris(pts).items():
                if hit.any():
                    causes.add(f"building:{bid}")
                covered |= hit
            if blockers:
                hit = in_any_polygon(pts, [bl["polygon_ne"] for bl in blockers])
                if hit.any():
                    causes.add("blocker")
                covered |= hit
            blocked, at = _blocked(along, pts, covered)
            roads.append({"id": sid, "road": r["id"], "polyline_ne": piece, "width_m": float(r["width_m"]),
                          "kind": r.get("kind", "road"), "blocked": blocked,
                          "cause": sorted(causes) if blocked else [],
                          **({"blocked_at_ne": at} if blocked else {})})

    # -- surface model: ground, then roofs (or rubble), then water ----------------------------
    surface = ground.copy()
    for b in buildings:
        ii, jj = _poly_nodes(grid, b["footprint_ne"])
        surface[ii, jj] = np.minimum(surface[ii, jj], b["base_down_m"] - b["post_height_m"])
    if water_m is not None:
        surface = np.where(water_m > 0, np.minimum(surface, level_down), surface)

    # -- checks against the layout's expectations ---------------------------------------------
    exp = layout.get("expect") or {}
    if "blocked" in exp:
        got = sorted(r["id"] for r in roads if r["blocked"])
        if got != sorted(exp["blocked"]):
            problems.append(f"blocked segments {got} != expected {sorted(exp['blocked'])}")
    for bid, g in (exp.get("grades") or {}).items():
        b = next((x for x in buildings if x["id"] == bid), None)
        if b is None or b["grade"] != g:
            problems.append(f"building {bid}: grade {b and b['grade']} != expected {g}")

    doc = {"version": 1, "map": layout.get("map", ""), "area_classes": area_doc, "roads": roads,
           "buildings": buildings, "blockers": blockers,
           "water": {"level_down_m": round(level_down, 3), "polygon_ne": water["polygon_ne"]} if water else None,
           "grades": list(GRADES)}
    placement = _placement(layout, buildings)
    masks = {k: rg.to_png(f) for k, f in flds.items()}
    return DamageBuild(doc=doc, classes=classes, debris_m=debris_m, water_m=water_m,
                       grid=(grid.n0, grid.e0, grid.cell_m),
                       surface=TerrainModel(surface, grid.n0, grid.e0, grid.cell_m, "ground + roofs + water"),
                       placement=placement, masks=masks, problems=problems)


def _blocked(along: np.ndarray, pts: np.ndarray, covered: np.ndarray) -> Tuple[bool, List[List[float]]]:
    """Whether some BLOCK_WINDOW_M stretch has BLOCK_COVER of its corridor covered; where (window centres)."""
    stations = np.unique(along)
    share = np.array([covered[along == s].mean() for s in stations])
    k = max(1, int(round(BLOCK_WINDOW_M / CORRIDOR_STEP_M)) + 1)
    if len(share) <= k:
        windows = [(0, len(share))]
    else:
        windows = [(i, i + k) for i in range(len(share) - k + 1)]
    hits = [(a, b) for a, b in windows if share[a:b].mean() >= BLOCK_COVER]
    if not hits:
        return False, []
    at = []
    for a, b in (hits[0], hits[-1]):
        mid = stations[(a + b - 1) // 2]
        p = pts[along == mid].mean(axis=0)
        at.append([round(float(p[0]), 2), round(float(p[1]), 2)])
    return True, at if at[0] != at[1] else at[:1]


def _placement(layout: Dict[str, Any], buildings: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Actors for Unreal (tools/_derive.py place_actors): location in cm, Unreal Z = -NED down."""
    src = {b["id"]: b for b in layout.get("buildings", [])}
    actors = []
    for b in buildings:
        s = src[b["id"]]
        if not s.get("asset") and not s.get("variant"):
            continue
        ne = s.get("ne") or np.asarray(b["footprint_ne"]).mean(axis=0).tolist()
        actors.append({"name": b["id"], "asset": s.get("variant") or s["asset"], "kind": "building",
                       "grade": b["grade"], "location_cm": [round(ne[0] * 100, 1), round(ne[1] * 100, 1),
                                                            round(-b["base_down_m"] * 100, 1)],
                       "yaw_deg": float(s.get("yaw_deg", 0.0))})
    for p in layout.get("props", []):
        actors.append({"name": p["name"], "asset": p["asset"], "kind": p.get("kind", "prop"),
                       "ne": p["ne"], "z_offset_m": float(p.get("z_offset_m", 0.0)),
                       "yaw_deg": float(p.get("yaw_deg", 0.0))})
    return {"map": layout.get("map", ""), "actors": actors}


def write_build(build: DamageBuild, layout_path: Path, sim_dir: Path, terrain_shas: Dict[str, str],
                source_dir: Optional[Path] = None) -> Dict[str, Path]:
    """Write damage_gt.json/.npz and surface_dsm.npz to ``sim_dir``, and placement.json and the region
    masks (masks/<region>.png) to ``source_dir`` (default: the layout's folder)."""
    from vivid2.damage.gt import RASTER_FILE, SURFACE_FILE, GT_FILE, sha256_file, write_raster

    sim_dir.mkdir(parents=True, exist_ok=True)
    raster = sim_dir / RASTER_FILE
    n0, e0, cell = build.grid
    write_raster(raster, build.classes, n0, e0, cell, build.debris_m, build.water_m)
    surface = sim_dir / SURFACE_FILE
    build.surface.save(surface)
    out = {"raster": raster, "surface": surface}
    doc = dict(build.doc)
    try:
        rel = layout_path.resolve().relative_to(Path(__file__).resolve().parents[3])
    except ValueError:
        rel = layout_path
    doc["layout"] = {"file": str(rel).replace("\\", "/"), "sha256": sha256_file(layout_path)}
    doc["terrain"] = terrain_shas
    doc["raster"] = {"file": RASTER_FILE, "sha256": sha256_file(raster)}
    doc["surface"] = {"file": SURFACE_FILE, "sha256": sha256_file(surface)}
    gt = sim_dir / GT_FILE
    gt.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    out["gt"] = gt
    src = Path(source_dir) if source_dir is not None else layout_path.parent
    src.mkdir(parents=True, exist_ok=True)
    pl = src / "placement.json"
    pl.write_text(json.dumps(build.placement, indent=1) + "\n", encoding="utf-8")
    out["placement"] = pl
    if build.masks:
        import cv2

        mdir = src / "masks"
        mdir.mkdir(parents=True, exist_ok=True)
        for name, img in build.masks.items():
            p = mdir / f"{name}.png"
            cv2.imwrite(str(p), img)
            out[f"mask:{name}"] = p
    return out
