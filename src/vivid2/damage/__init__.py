"""The damage task: map what a disaster did, as well as find people (docs/damage.md).

A scenario with a ``damage`` block asks the stack for some of four products:

    areas      polygons of changed ground by class (debris, flood, ...)
    roads      which known road segments are blocked
    buildings  a damage grade for each known building (none, partial, collapsed)
    clues      where surface clues lie (a ski, a backpack...), by kind

The stack is told what responders would have before the event (road segments, building
footprints, a pre-event DEM), never the damage. It reports through ``ctx.report_damage``; every
report cites one of the drone's own recorded search photos. Ground truth comes from the map's
layout (``vivid2 damage build``); hidden buried victims (scenario ``damage.buried``) score how
well the map would send rescuers to them.

Modules: ``regions`` (noisy-edged region fields shared by Unreal masks and ground truth),
``geometry`` (cells, polygons, road corridors, what a photo saw), ``layout`` (layout -> ground
truth), ``gt`` (loading it, with checksums), ``run`` (one run's damage task), ``check``
(placement rules).
"""
from __future__ import annotations

from typing import Dict

PRODUCTS = ("areas", "roads", "buildings", "clues")
COMPONENTS = PRODUCTS + ("buried",)
GRADES = ("none", "partial", "collapsed")
GRADE_LEVEL = {g: i for i, g in enumerate(GRADES)}
# EMS-98 damage grades 0-5 (0 = no damage, 5 = destruction): five grades can't be told apart
# from 55 m, so the task uses three.
EMS98_TO_GRADE = {0: "none", 1: "none", 2: "partial", 3: "partial", 4: "collapsed", 5: "collapsed"}
CLUE_KINDS = ("ski", "ski_pole", "glove", "backpack", "helmet", "clothing", "vehicle", "other")
# Missions that may carry a damage task (each also finds people).
DAMAGE_MISSIONS = ("landslide_sar", "avalanche_sar", "earthquake_sar", "tsunami_sar")

DAMAGE_CELL_M = 2.0        # scoring grid
GSD_MAX_M = 0.30           # a photo observes a cell only this sharply (ground sampling distance) or better
VIEW_MAX_RANGE_M = 400.0   # ... and no farther away
CLUE_TOLERANCE_M = 5.0     # a clue report within this of a clue matches it
BURIED_BUFFER_M = 3.0      # a buried victim within this of a reported area cell is covered
BLOCK_MIN_WATER_M = 0.3    # water this deep blocks a road (area classes with blocks_roads block it anyway)
BLOCK_WINDOW_M = 4.0       # a segment is blocked if some stretch this long ...
BLOCK_COVER = 0.6          # ... has this share of its corridor covered
CORRIDOR_STEP_M = 0.5      # corridor sampling, along and across the road
IN_PHOTO_BUILDING_SHARE = 0.25  # a building report's photo must show this share of its footprint
MAX_POLYGON_VERTICES = 2000
REGION_THRESHOLD = 0.5     # region field value that counts as inside (ground truth, PCG split, material)

DEFAULT_COMPONENT_WEIGHTS: Dict[str, float] = {"areas": 1.0, "roads": 1.0, "buildings": 1.0, "clues": 1.0,
                                               "buried": 0.5}
