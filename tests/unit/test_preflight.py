"""Preflight must stop a run on the wrong map or with a renamed target, with a useful message."""
import re

import pytest

from vivid2.preflight import PreflightError, map_check_patterns, run_preflight
from vivid2.runner import order_runs
from vivid2.scenario import Campaign, Scenario


class FakeSimWorld:
    """Answers the World queries preflight uses, with full-match regexes like the plugin."""

    def __init__(self, actors, sizes=None):
        self.actors = list(actors)
        self.sizes = sizes or {}

    def list_objects(self, pattern):
        return [a for a in self.actors if re.fullmatch(pattern, a)]

    def get_3d_bounding_box(self, name, alignment):
        x, y, z = self.sizes.get(name, (0.5, 0.5, 1.7))
        return {"center": {"x": 0, "y": 0, "z": 0}, "size": {"x": x, "y": y, "z": z}}


BLOCKS = ["TemplateCube_Rounded_1", "TemplateCube_Rounded_102", "Ground"]


@pytest.fixture
def blocks_spec(root):
    return Scenario.load(root / "scenarios/forest_sar/blocks_hiker_01.jsonc").spec


def test_blocks_scenario_passes_on_blocks(blocks_spec):
    info = run_preflight(FakeSimWorld(BLOCKS + ["Hiker1"]), blocks_spec, blocks_spec["targets"])
    assert info["map"] == "BlocksMap" and info["target_sizes_m"]["Hiker1"] == [0.5, 0.5, 1.7]


def test_wrong_map_is_reported_with_what_is_open(blocks_spec):
    world = FakeSimWorld(["BP_VIVID2Map_ForestSAR_C_0", "Hiker1"])
    with pytest.raises(PreflightError, match="not map 'BlocksMap'.*BP_VIVID2Map_ForestSAR"):
        run_preflight(world, blocks_spec, blocks_spec["targets"])


def test_vivid2_maps_are_recognised_by_their_marker():
    [pat] = map_check_patterns({"map": "ForestSAR"})
    for name in ["BP_VIVID2Map_ForestSAR", "BP_VIVID2Map_ForestSAR_C_0", "BP_VIVID2Map_ForestSAR_C_12",
                 "BP_VIVID2Map_ForestSAR_C_UAID_A4BB6D3F01A2B3C4_1234567890"]:
        assert re.fullmatch(pat, name), name
    # a canopy variant's marker must not pass for the base map
    for name in ["BP_VIVID2Map_ForestSAR_Dense_C_0", "BP_VIVID2Map_ForestSAR_Open", "BP_VIVID2Map_ForestSARX_C_0"]:
        assert not re.fullmatch(pat, name), name
    [dense] = map_check_patterns({"map": "ForestSAR_Dense"})
    assert re.fullmatch(dense, "BP_VIVID2Map_ForestSAR_Dense_C_0")
    assert map_check_patterns({}) == []
    spec = {"scene": {"map": "ForestSAR"}}
    run_preflight(FakeSimWorld(["BP_VIVID2Map_ForestSAR_C_1", "Hiker1"]), spec, [{"id": "Hiker1", "asset": "x"}])


def test_renamed_or_odd_sized_targets_fail(blocks_spec):
    renamed = FakeSimWorld(BLOCKS + ["Hiker1_2"])  # the sim renamed it: an actor already contained "Hiker1"
    with pytest.raises(PreflightError, match="not spawned under its own name.*Hiker1_2"):
        run_preflight(renamed, blocks_spec, blocks_spec["targets"])
    tiny = FakeSimWorld(BLOCKS + ["Hiker1"], sizes={"Hiker1": (0.01, 0.01, 0.01)})
    with pytest.raises(PreflightError, match="implausible size"):
        run_preflight(tiny, blocks_spec, blocks_spec["targets"])
    empty = FakeSimWorld(BLOCKS + ["Hiker1"], sizes={"Hiker1": (0.0, 0.0, 0.0)})  # e.g. no colliding component
    with pytest.raises(PreflightError, match="implausible size"):
        run_preflight(empty, blocks_spec, blocks_spec["targets"])
    flat = FakeSimWorld(BLOCKS + ["Hiker1"], sizes={"Hiker1": (2.0, 1.4, 0.02)})  # a tarp is thin but real
    run_preflight(flat, blocks_spec, blocks_spec["targets"])


def test_runs_are_grouped_by_map(root):
    blocks = Campaign.load(root / "campaigns/blocks_v1.jsonc").runs()
    forest = Campaign.load(root / "campaigns/forest_sar_v1.jsonc").runs()
    ordered = order_runs(forest[:2] + blocks[:2] + forest[2:4] + blocks[2:4])
    maps = [r.spec["scene"]["map"] for r in ordered]
    assert maps == sorted(maps) and maps.count("BlocksMap") == 4


def test_campaign_sweep_runs_a_scenario_on_every_canopy_map(root):
    c = Campaign.load(root / "campaigns/forest_canopy.jsonc")
    runs = c.runs()
    assert sorted(r.spec["scene"]["map"] for r in runs) == ["ForestSAR", "ForestSAR_Dense", "ForestSAR_Open"]
    assert len({r.run_id for r in runs}) == 3
    assert [r.spec["scene"]["map"] for r in order_runs(runs)] == ["ForestSAR", "ForestSAR_Dense", "ForestSAR_Open"]
