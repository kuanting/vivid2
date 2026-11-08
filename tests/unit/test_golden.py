"""Single-drone runs must not change: run ids and config hashes, scenes, metrics and scores.

The golden files were recorded before team support (see tests/golden/goldens.py). A failure here
means existing results would be redone on resume or scored differently.
"""
import json
import math
from pathlib import Path

import pytest

from golden import goldens

HERE = Path(goldens.__file__).resolve().parent


def _golden(name):
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def _assert_close(got, want, path="$"):
    if isinstance(want, dict):
        assert isinstance(got, dict) and sorted(got) == sorted(want), f"{path}: keys {sorted(got)} != {sorted(want)}"
        for k in want:
            _assert_close(got[k], want[k], f"{path}.{k}")
    elif isinstance(want, list):
        assert isinstance(got, list) and len(got) == len(want), f"{path}: length {len(got)} != {len(want)}"
        for i, (g, w) in enumerate(zip(got, want)):
            _assert_close(g, w, f"{path}[{i}]")
    elif isinstance(want, float) and not isinstance(want, bool):
        assert isinstance(got, (int, float)), f"{path}: {got!r} != {want!r}"
        assert (math.isnan(want) and math.isnan(got)) or got == pytest.approx(want, rel=1e-7, abs=1e-7), \
            f"{path}: {got!r} != {want!r}"
    else:
        assert got == want, f"{path}: {got!r} != {want!r}"


def test_config_hashes_of_single_drone_runs_are_unchanged():
    want, got = _golden("config_hashes.json"), goldens.config_hashes()
    assert got["sim_version"] == want["sim_version"]
    for agent, table in want["agents"].items():
        changed = [k for k, h in table.items() if got["agents"].get(agent, {}).get(k) != h]
        assert not changed, f"{agent}: {len(changed)} runs would be redone, e.g. {changed[:3]}"


def test_single_drone_scenes_are_unchanged():
    want = _golden("scenes.json")
    got = goldens.scenes()
    _assert_close({k: got.get(k) for k in want}, want)  # scenarios added since are not in the golden file


def test_single_drone_metrics_and_scores_are_unchanged():
    _assert_close(goldens.fake_runs(), _golden("fake_runs.json"))
