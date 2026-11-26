"""The control team at the other end of the drones' link (scenario ``comms.control_link``, default on).

In a real search the drones report to a control team over a link: a temporary network the
rescuers set up, or satellite. A find counts the moment it is reported, and once every missing
person has been found the mission is over; nobody needs the drone to fly home first. Without a
link (``control_link: false``) the drone has to bring itself home, as in VIVID 2 before.

``ControlStation`` stands in for that team. The runner hands it every report as it is made; it
judges them with the scoring's own rules (a report must cite one of the reporting drone's
recorded search photos, and is matched to a person within ``success.localization_tolerance_m``,
``vivid2.metrics.mission.match_reports``), so a person found here is a person found in the score.
When everyone has been found it logs a ``mission_over`` event and the runner ends the run
(end reason ``all_found``). On a run with a damage task finding everyone does not end the mission:
the stack ends it when its damage map is done.

It sends nothing back to the stack: confirming or rejecting a report would leak ground truth.
"""
from __future__ import annotations

import threading
from typing import Any, Callable, Dict, List, Optional


def control_link(spec: Dict[str, Any]) -> bool:
    """Whether a scenario's drones have a link to the control team (``comms.control_link``, default True)."""
    return bool((spec.get("comms") or {}).get("control_link", True))


class ControlStation:
    def __init__(self, targets: List[Dict[str, Any]], tolerance_m: float, link: bool, damage_task: bool,
                 log: Callable[[str, Dict[str, Any]], None]):
        self.targets = targets
        self.tolerance_m = float(tolerance_m)
        self.link = bool(link)
        self.damage_task = bool(damage_task)
        self._log = log
        self._lock = threading.Lock()
        self._reports: List[Dict[str, Any]] = []
        self.found: Dict[str, Any] = {}
        self._over = False

    @classmethod
    def for_run(cls, spec: Dict[str, Any], targets: List[Dict[str, Any]], damage_task: bool,
                log: Callable[[str, Dict[str, Any]], None]) -> "ControlStation":
        tol = (spec.get("success") or {}).get("localization_tolerance_m", 10.0)
        return cls(targets, tol, control_link(spec), damage_task, log)

    def report(self, record: Dict[str, Any], cited: bool) -> None:
        """A report as logged in the ``reports`` stream; ``cited``: it cites one of this drone's search photos."""
        from vivid2.metrics.mission import match_reports

        if not cited:
            return  # a false alarm whatever it found (docs/rules.md)
        with self._lock:
            self._reports.append(record)
            self.found = match_reports(self._reports, self.targets, self.tolerance_m)["matched"]
            if (self.link and not self.damage_task and not self._over and self.targets
                    and len(self.found) == len(self.targets)):
                self._over = True
                self._log("events", {"t": record.get("t_report", record.get("t")), "kind": "mission_over",
                                     "reason": "all_found", "found": sorted(self.found)})

    def mission_over(self) -> bool:
        """Every missing person has been found (link on, no damage task): the run ends."""
        return self._over

    def end_reason(self) -> Optional[str]:
        return "all_found" if self._over else None
