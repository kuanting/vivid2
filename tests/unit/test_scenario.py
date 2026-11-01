import jsonschema
import numpy as np
import pytest

from vivid2.scenario import Campaign, Scenario


def test_catalog_validates(root):
    paths = sorted((root / "scenarios").rglob("*.jsonc"))
    assert paths
    for p in paths:
        Scenario.load(p)


def test_sweep_expansion_and_determinism(root):
    sc = Scenario.load(root / "scenarios/forest_sar/blocks_hiker_02.jsonc")
    runs = sc.expand()
    assert len(runs) == 2 * 2 * 2  # fog x time x seeds
    assert len({r.run_id for r in runs}) == len(runs)
    again = sc.expand()
    assert [r.run_id for r in runs] == [r.run_id for r in again]
    fogs = {r.spec["environment"]["weather"]["fog"] for r in runs}
    assert fogs == {0.0, 0.4}
    assert "sweep" not in runs[0].spec


def test_seed_override(root):
    sc = Scenario.load(root / "scenarios/forest_sar/blocks_hiker_01.jsonc")
    assert len(sc.expand(seeds=5)) == 5
    assert [r.seed for r in sc.expand(seeds=[7, 9])] == [7, 9]


def test_target_jitter_is_seeded_and_bounded(root):
    sc = Scenario.load(root / "scenarios/forest_sar/blocks_hiker_01.jsonc")
    a, b = sc.expand(seeds=[1, 2])
    base = np.asarray(sc.spec["targets"][0]["ned"][:2])
    ta, tb = a.resolved_targets()[0], b.resolved_targets()[0]
    assert ta["ned"] == a.resolved_targets()[0]["ned"]
    assert ta["ned"] != tb["ned"]
    assert np.linalg.norm(np.asarray(ta["ned"][:2]) - base) <= sc.spec["target_jitter_m"] + 1e-9
    assert ta["ned"][2] == sc.spec["targets"][0]["ned"][2]


def test_invalid_scenario_rejected(root):
    sc = Scenario.load(root / "scenarios/forest_sar/blocks_hiker_01.jsonc").spec
    bad = dict(sc, mission="moon_landing")
    with pytest.raises(jsonschema.ValidationError):
        Scenario(bad)


def test_campaign_glob(root):
    c = Campaign.load(root / "campaigns/blocks_v1.jsonc")
    assert {s.id for s in c.scenarios} == {"blocks_hiker_01", "blocks_hiker_02"}
    assert c.sim_config_dir.endswith("sim_config")
    assert len(c.runs()) == 3 + 8
