"""The Avalanche map: its layout, ground truth and scenarios agree (the terrain is in Git LFS: skipped without it)."""
import json

import pytest

from conftest import lfs_file
from vivid2.damage.layout import build_damage
from vivid2.runner import damage_setup
from vivid2.scenario import Scenario
from vivid2.terrain import TerrainModel


@pytest.fixture
def avalanche(root):
    lay = json.loads((root / "unreal/VIVID2Env/SourceData/Avalanche/layout.json").read_text(encoding="utf-8"))
    post = TerrainModel.load(lfs_file(root / "sim_config/maps/Avalanche/terrain_dtm.npz"))
    return lay, post


def test_the_avalanche_layout_builds_its_debris(root, avalanche):
    """One area class (the debris), no roads or buildings; the committed ground truth is the layout's."""
    lay, post = avalanche
    built = build_damage(lay, post)
    assert set(built.doc["area_classes"]) == {"debris"}
    assert not built.doc["roads"] and not built.doc["buildings"]
    gt = json.loads((root / "sim_config/maps/Avalanche/damage_gt.json").read_text(encoding="utf-8"))
    assert gt["area_classes"] == built.doc["area_classes"]


def test_the_deposit_raises_the_ground_and_the_slab_lowers_it(root, avalanche):
    """The post-event terrain carries the edits: debris on top of the snow, the slab and track removed."""
    from vivid2.damage import regions as rg
    import numpy as np

    lay, post = avalanche
    pre = TerrainModel.load(lfs_file(root / "sim_config/maps/Avalanche/pre_event_dtm.npz"))
    f = rg.fields(lay)
    ax = np.arange(-500.0, 501.0, 4.0)
    N, E = np.meshgrid(ax, ax, indexing="ij")
    raised = (pre.down_at(N.ravel(), E.ravel()) - post.down_at(N.ravel(), E.ravel())).reshape(N.shape)
    inside = {k: rg.sample(f[k], N, E, float(lay["mask"]["extent_m"])) >= 0.9 for k in ("release", "track", "deposit")}
    assert np.median(raised[inside["deposit"]]) > 1.0
    assert np.median(raised[inside["release"]]) < -1.2
    assert np.median(raised[inside["track"]]) < -0.3
    assert np.abs(raised[~(inside["release"] | inside["track"] | inside["deposit"])]).max() < 2.5


def test_the_avalanche_scenarios_pass_their_placement_checks(root, avalanche):
    """People and clues on the debris, buried victims inside it: damage_setup raises on any problem."""
    lfs_file(root / "sim_config/maps/Avalanche/damage_gt.npz")
    _, post = avalanche
    runs = [run for p in sorted((root / "scenarios/avalanche").glob("*.jsonc")) for run in Scenario.load(p).expand()]
    assert len(runs) == 4
    for run in runs:
        dmg = damage_setup(run, str(root / "sim_config"), post)
        assert dmg["task"] is not None and dmg["clues"]
