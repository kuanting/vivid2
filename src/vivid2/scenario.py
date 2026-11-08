"""Scenario and campaign specs: loading, validation and sweep expansion.

A *scenario* is a logical test (e.g. "find one hiker in the north valley").
A *run* is one concrete, reproducible instance of it: swept parameters fixed,
a seed fixed, and a deterministic ``run_id`` derived from both.
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import commentjson
import jsonschema
import numpy as np


def _load_schema(name: str) -> Dict:
    text = resources.files("vivid2.schema").joinpath(name).read_text(encoding="utf-8")
    return json.loads(text)


def load_jsonc(path: Union[str, Path]) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return commentjson.load(f)


def get_path(d: Dict, dotted: str) -> Any:
    cur = d
    for key in dotted.split("."):
        cur = cur[key]
    return cur


def set_path(d: Dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    cur = d
    for key in keys[:-1]:
        cur = cur.setdefault(key, {})
    cur[keys[-1]] = value


@dataclass
class Run:
    """One concrete run of a scenario."""

    run_id: str
    scenario_id: str
    seed: int
    params: Dict[str, Any]
    spec: Dict[str, Any]
    source: Optional[str] = None

    @property
    def rng(self) -> np.random.Generator:
        return np.random.default_rng(self.seed)

    @property
    def authored_ground_down_m(self) -> float:
        return float(self.spec["search_area"].get("ground_down_m", 0.0))

    def resolved_targets(self, terrain=None) -> List[Dict]:
        """Targets with seeded horizontal jitter, then (given a terrain model) set on the ground there."""
        jitter = float(self.spec.get("target_jitter_m", 0.0))
        rng = self.rng
        out = []
        for tgt in self.spec["targets"]:
            t = copy.deepcopy(tgt)
            if jitter > 0:
                r = jitter * np.sqrt(rng.uniform())
                a = rng.uniform(0, 2 * np.pi)
                t["ned"] = [t["ned"][0] + r * np.cos(a), t["ned"][1] + r * np.sin(a), t["ned"][2]]
            if terrain is not None:
                t["ned"] = _snap(t["ned"], terrain, self.authored_ground_down_m)
            out.append(t)
        return out

    def resolved_distractors(self, terrain=None) -> List[Dict]:
        out = copy.deepcopy(self.spec.get("distractors", []))
        if terrain is not None:
            for d in out:
                d["ned"] = _snap(d["ned"], terrain, self.authored_ground_down_m)
        return out

    def resolved_start(self, terrain=None) -> List[float]:
        ned = list(self.spec["dut_start"]["ned"])
        return _snap(ned, terrain, self.authored_ground_down_m) if terrain is not None else ned

    @property
    def fleet_size(self) -> int:
        return fleet_size(self.spec)

    def resolved_fleet(self, terrain=None) -> List[Dict[str, Any]]:
        """Every drone's id and launch pad, ``[{"id", "ned", "yaw_deg"}]``, set on the terrain like the start.

        One drone (no ``fleet``, or size 1) sits exactly at ``dut_start``. A team gets explicit
        ``fleet.drones`` pads or, by default, a grid centred on ``dut_start``, ``spacing_m`` apart.
        """
        n = self.fleet_size
        start = self.spec["dut_start"]
        if n == 1:
            name = self.spec["scene"].get("drone_name", "Drone1")
            return [{"id": name, "ned": self.resolved_start(terrain), "yaw_deg": float(start.get("yaw_deg", 0.0))}]
        fleet = self.spec["fleet"]
        if fleet.get("drones"):
            if len(fleet["drones"]) < n:
                raise ValueError(f"scenario {self.scenario_id}: fleet.size {n} but only {len(fleet['drones'])} drones listed")
            pads = [(list(d["ned"]), float(d.get("yaw_deg", start.get("yaw_deg", 0.0)))) for d in fleet["drones"][:n]]
        else:
            h = float(fleet.get("spacing_m", DEFAULT_PAD_SPACING_M)) / 2.0
            n0, e0, d0 = start["ned"]
            pads = [([n0 + dn * h, e0 + de * h, d0], float(start.get("yaw_deg", 0.0))) for dn, de in GRID_LAYOUT[n]]
        for i, (a, _) in enumerate(pads):
            for b, _ in pads[:i]:
                if np.hypot(a[0] - b[0], a[1] - b[1]) < MIN_PAD_SPACING_M - 1e-9:
                    raise ValueError(f"scenario {self.scenario_id}: launch pads {b[:2]} and {a[:2]} are closer than "
                                     f"{MIN_PAD_SPACING_M} m (each pad's 5 m radius is its own)")
        return [{"id": f"Drone{i + 1}", "ned": _snap(ned, terrain, self.authored_ground_down_m) if terrain is not None
                 else ned, "yaw_deg": yaw} for i, (ned, yaw) in enumerate(pads)]


# A team's automatic launch layout around dut_start, in half-spacings (north, east): a row of two,
# then a 2 x 2 square. Pads at least 10 m apart keep each drone's 5 m home-pad radius its own.
GRID_LAYOUT = {2: [(0, -1), (0, 1)], 3: [(-1, -1), (-1, 1), (1, -1)], 4: [(-1, -1), (-1, 1), (1, -1), (1, 1)]}
DEFAULT_PAD_SPACING_M = 10.0
MIN_PAD_SPACING_M = 10.0


def fleet_size(spec: Dict[str, Any]) -> int:
    """Number of drones a scenario flies (1 unless ``fleet.size`` says otherwise)."""
    return int((spec.get("fleet") or {}).get("size", 1))


def _snap(ned, terrain, authored_ground_down_m: float) -> List[float]:
    from vivid2.terrain import snap_to_ground

    return snap_to_ground(ned, terrain, authored_ground_down_m)


def validate_scenario(spec: Dict, where: str = "") -> None:
    """Raise jsonschema.ValidationError if ``spec`` isn't a valid scenario."""
    try:
        jsonschema.validate(spec, _load_schema("scenario.schema.json"))
    except jsonschema.ValidationError as e:
        if where:
            e.message = f"{where}: {e.message}"
        raise


def make_run_id(scenario_id: str, params: Dict[str, Any], seed: int) -> str:
    blob = json.dumps({"s": scenario_id, "p": params, "seed": seed}, sort_keys=True)
    return f"{scenario_id}-{hashlib.sha1(blob.encode()).hexdigest()[:8]}"


class Scenario:
    def __init__(self, spec: Dict, source: Optional[str] = None):
        validate_scenario(spec)
        self.spec = spec
        self.source = source

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Scenario":
        return cls(load_jsonc(path), source=str(path))

    @property
    def id(self) -> str:
        return self.spec["id"]

    def expand(self, seeds: Optional[Union[int, List[int]]] = None,
               extra_sweep: Optional[Dict[str, List[Any]]] = None) -> List[Run]:
        """Grid-expand ``sweep`` and seeds into concrete runs.

        ``seeds`` overrides ``sweep.seeds``; an int N means seeds 0..N-1.
        ``extra_sweep`` (a campaign's sweep) adds or replaces dotted-path axes.
        """
        sweep = dict(self.spec.get("sweep", {}))
        sweep.update(extra_sweep or {})
        seed_spec = sweep.pop("seeds", 1) if seeds is None else seeds
        sweep.pop("seeds", None)
        seed_list = list(range(seed_spec)) if isinstance(seed_spec, int) else list(seed_spec)

        keys = sorted(sweep)
        combos = list(itertools.product(*(sweep[k] for k in keys))) if keys else [()]
        runs = []
        for combo in combos:
            params = dict(zip(keys, combo))
            base = copy.deepcopy(self.spec)
            base.pop("sweep", None)
            for k, v in params.items():
                set_path(base, k, v)
            # Sweeps write by dotted path, so a typo'd path or out-of-range value
            # only shows up in the expanded spec.
            validate_scenario(base, f"scenario {self.id} with sweep {params}")
            for seed in seed_list:
                runs.append(
                    Run(
                        run_id=make_run_id(self.id, params, seed),
                        scenario_id=self.id,
                        seed=seed,
                        params=params,
                        spec=base,
                        source=self.source,
                    )
                )
        return runs


@dataclass
class Campaign:
    name: str
    scenarios: List[Scenario]
    seeds: Optional[Union[int, List[int]]] = None
    weights: Dict[str, float] = field(default_factory=dict)
    sim_config_dir: Optional[str] = None
    sweep: Dict[str, List[Any]] = field(default_factory=dict)  # applied to every scenario, e.g. {"scene.map": [...]}

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Campaign":
        path = Path(path)
        spec = load_jsonc(path)
        scenario_paths: List[Path] = []
        for entry in spec["scenarios"]:
            p = path.parent / entry
            if "*" in p.name:
                scenario_paths.extend(sorted(p.parent.resolve().glob(p.name)))
            else:
                scenario_paths.append(p.resolve())
        sim_cfg = spec.get("sim_config_dir")
        return cls(
            name=spec.get("name", path.stem),
            scenarios=[Scenario.load(p) for p in scenario_paths],
            seeds=spec.get("seeds"),
            weights=spec.get("weights", {}),
            sim_config_dir=str((path.parent / sim_cfg).resolve()) if sim_cfg else None,
            sweep=spec.get("sweep", {}),
        )

    def runs(self) -> List[Run]:
        out: List[Run] = []
        for sc in self.scenarios:
            out.extend(sc.expand(self.seeds, self.sweep))
        return out
