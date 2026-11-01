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

    def resolved_targets(self) -> List[Dict]:
        """Targets with seeded position jitter applied (horizontal only)."""
        jitter = float(self.spec.get("target_jitter_m", 0.0))
        rng = self.rng
        out = []
        for tgt in self.spec["targets"]:
            t = copy.deepcopy(tgt)
            if jitter > 0:
                r = jitter * np.sqrt(rng.uniform())
                a = rng.uniform(0, 2 * np.pi)
                t["ned"] = [t["ned"][0] + r * np.cos(a), t["ned"][1] + r * np.sin(a), t["ned"][2]]
            out.append(t)
        return out


def make_run_id(scenario_id: str, params: Dict[str, Any], seed: int) -> str:
    blob = json.dumps({"s": scenario_id, "p": params, "seed": seed}, sort_keys=True)
    return f"{scenario_id}-{hashlib.sha1(blob.encode()).hexdigest()[:8]}"


class Scenario:
    def __init__(self, spec: Dict, source: Optional[str] = None):
        jsonschema.validate(spec, _load_schema("scenario.schema.json"))
        self.spec = spec
        self.source = source

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Scenario":
        return cls(load_jsonc(path), source=str(path))

    @property
    def id(self) -> str:
        return self.spec["id"]

    def expand(self, seeds: Optional[Union[int, List[int]]] = None) -> List[Run]:
        """Grid-expand ``sweep`` and seeds into concrete runs.

        ``seeds`` overrides ``sweep.seeds``; an int N means seeds 0..N-1.
        """
        sweep = dict(self.spec.get("sweep", {}))
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
        )

    def runs(self) -> List[Run]:
        out: List[Run] = []
        for sc in self.scenarios:
            out.extend(sc.expand(self.seeds))
        return out
