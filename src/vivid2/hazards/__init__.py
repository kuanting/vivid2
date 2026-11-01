"""Dynamic hazards (Phase 2): wildfire spread and landslide change ground truth.

Phase 0 defines the interface only; see docs/missions.md.
"""
from __future__ import annotations

import abc
from typing import Any, Dict, Optional

import numpy as np


class Hazard(abc.ABC):
    """A time-varying hazard driven from the client during a run."""

    @abc.abstractmethod
    def setup(self, world, rng: np.random.Generator) -> None:
        """Spawn initial hazard objects (fire emitters, debris...)."""

    @abc.abstractmethod
    def step(self, world, sim_time_s: float) -> None:
        """Advance the hazard model and update the scene."""

    @abc.abstractmethod
    def ground_truth(self, sim_time_s: float) -> Dict[str, Any]:
        """GT for scoring, e.g. fire-front mask or landslide change mask."""

    def no_fly_polygon(self, sim_time_s: float) -> Optional[np.ndarray]:
        """Airspace the DUT must avoid (e.g. fire TFR); None if none."""
        return None
