"""A crash ends the mission (docs/architecture.md).

A drone has crashed when it hit something away from its launch pad, or when it has been tipped
over (past UPRIGHT_DEG) for TIPPED_FOR_S of mission time. Collisions are told apart as the safety
metrics do (``vivid2.metrics.safety.split_ground_contacts``): a contact within 5 m of a launch pad
and below 2 m above the ground is the takeoff or the landing; anything else is a crash.

The runner feeds every recorded pose and every collision report to a ``CrashWatch``, stops the
agents as soon as it sees a crash, and ends the run as ``crashed``. A crash before the drone has
finished its takeoff climb ends the run as an ``error`` instead: the takeoff is the runner's own
controller, not the stack's, so it is the simulator's failure, and running the campaign again
re-runs it.
"""
from __future__ import annotations

import threading
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from vivid2.geo import tilt_deg_of
from vivid2.terrain import TerrainModel

UPRIGHT_DEG = 60.0  # a flying multirotor stays well inside this
TIPPED_FOR_S = 1.0  # tipped past UPRIGHT_DEG this long (mission time) is a crash


class CrashWatch:
    """Watches a run's drones for the first crash; ``crash`` holds it (None until then).

    ``pads``: every drone's launch pad (NED); ``log(stream, record)`` records the crash event;
    ``took_off(drone)`` tells whether that drone's takeoff climb is done (drone is None in a
    single-drone run, the drone's id in a team).
    """

    def __init__(self, pads: Sequence[Sequence[float]], terrain: TerrainModel,
                 log: Callable[[str, Dict[str, Any]], None], took_off: Callable[[Optional[str]], bool]):
        self._pads = np.atleast_2d(np.asarray(pads, float))
        self._terrain = terrain
        self._log = log
        self._took_off = took_off
        self._lock = threading.Lock()
        self._contacts: List[Dict[str, Any]] = []
        self._tipped_since: Dict[Optional[str], int] = {}
        self.crash: Optional[Dict[str, Any]] = None

    def crashed(self) -> bool:
        return self.crash is not None

    def end_reason(self) -> Optional[str]:
        """``crashed``, ``error`` (a crash during the takeoff climb), or None (no crash)."""
        if self.crash is None:
            return None
        return "error" if self.crash.get("during_takeoff") else "crashed"

    def collision(self, record: Dict[str, Any]) -> None:
        """A collision report, as recorded (called on the client's message thread): judged at the drone's next pose."""
        if self.crash is None:
            with self._lock:
                self._contacts.append(record)

    def pose(self, t_ns: int, pos: Sequence[float], q_wxyz: Sequence[float], drone: Optional[str] = None) -> None:
        """A recorded pose of ``drone``: judges the collisions reported since, and the drone's tilt."""
        if self.crash is not None:
            return
        from vivid2.metrics.safety import split_ground_contacts

        with self._lock:
            mine = [c for c in self._contacts if c.get("drone") == drone]
            if mine:
                self._contacts = [c for c in self._contacts if c.get("drone") != drone]
        p = np.asarray(pos, float)
        if mine:  # judged where the drone is now (the poses come 10 times a second)
            real, _ = split_ground_contacts([{**c, "t": t_ns} for c in mine], np.array([t_ns / 1e9]), p[None, :],
                                            self._pads, self._terrain)
            if real:
                return self._crashed(t_ns, p, drone, cause="collision", object_name=real[0].get("object_name"))
        tilt = tilt_deg_of(q_wxyz)
        if tilt <= UPRIGHT_DEG:
            self._tipped_since.pop(drone, None)
            return
        since = self._tipped_since.setdefault(drone, t_ns)
        if t_ns - since >= TIPPED_FOR_S * 1e9:
            self._crashed(t_ns, p, drone, cause="tipped over", tilt_deg=round(tilt, 1))

    def _crashed(self, t_ns: int, p: np.ndarray, drone: Optional[str], **why) -> None:
        self.crash = {"t": int(t_ns), "kind": "crash", **why, "pos": [round(float(v), 2) for v in p],
                      **({"drone": drone} if drone is not None else {}),
                      **({"during_takeoff": True} if not self._took_off(drone) else {})}
        self._log("events", self.crash)
