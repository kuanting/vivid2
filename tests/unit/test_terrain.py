"""Terrain: flat maps behave as before; on relief, objects sit on the ground and AGL is real."""
import hashlib
from pathlib import Path

import numpy as np
import pytest

from vivid2.metrics.safety import safety_metrics
from vivid2.recorder import RunLog
from vivid2.runner import mission_spec
from vivid2.terrain import TerrainModel, from_meta, load_for_run, snap_to_ground


def slope(cell_m=10.0):
    """Ground rising 1 m per 10 m northward (NED down decreases), over a 100 m x 100 m grid."""
    n = np.arange(11) * cell_m
    down = np.tile(-(n / 10.0)[:, None], (1, 11))
    return TerrainModel(down, n0=0.0, e0=0.0, cell_m=cell_m, source="test slope")


def test_flat_terrain_matches_the_old_constant_ground():
    flat = TerrainModel.flat(0.0)
    assert flat.is_flat and float(flat.down_at(12.3, -4.0)) == 0.0
    assert flat.agl(np.array([[0, 0, -35.0]]))[0] == 35.0
    assert snap_to_ground([35, 20, -0.85], flat, 0.0) == [35.0, 20.0, -0.85]


def test_bilinear_heights_and_snapping_keep_the_authored_offset():
    t = slope()
    assert float(t.down_at(55.0, 30.0)) == pytest.approx(-5.5)
    assert float(t.down_at(500.0, 0.0)) == pytest.approx(-10.0)  # clamped at the grid edge
    # A centre-pivot box authored 0.85 m above flat ground ends up 0.85 m above the slope.
    assert snap_to_ground([55, 30, -0.85], t, 0.0) == pytest.approx([55, 30, -6.35])


def test_terrain_file_round_trip_and_tamper_check(tmp_path):
    t = slope()
    path = tmp_path / "maps" / "Hill" / "terrain_dtm.npz"
    path.parent.mkdir(parents=True)
    t.save(path)
    spec = {"scene": {"map": "Hill"}, "search_area": {"ground_down_m": 0.0}}
    model, rec = load_for_run(tmp_path, spec)
    assert rec["kind"] == "dtm" and rec["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert float(model.down_at(55, 0)) == pytest.approx(-5.5)
    assert float(from_meta({"terrain": rec}).down_at(55, 0)) == pytest.approx(-5.5)
    TerrainModel(np.zeros((2, 2)), source="edited").save(path)
    with pytest.raises(ValueError, match="changed"):
        from_meta({"terrain": rec})
    assert load_for_run(tmp_path, {"scene": {"map": "NoSuchMap"}, "search_area": {"ground_down_m": -2.0}})[1] == \
        {"kind": "flat", "ground_down_m": -2.0}


def test_safety_agl_uses_the_terrain(tmp_path):
    path = tmp_path / "terrain_dtm.npz"
    slope().save(path)
    rec = {"kind": "dtm", "file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    meta = {"limits": {"timeout_s": 100, "min_agl_m": 10}, "start_ned": [0, 0, -1], "end_reason": "completed",
            "search_area": {"polygon_ne": [[0, 0], [100, 0], [100, 100], [0, 100]], "ground_down_m": 0.0}, "terrain": rec}
    # Level flight at 12 m above the start: fine over flat ground, too low where the ground has risen 8 m.
    traj = [{"t": int(i * 1e9), "pos": [float(n), 50.0, -12.0]} for i, n in enumerate(range(30, 100, 10))]
    s = safety_metrics(RunLog(Path("."), meta, {"trajectory": traj, "collisions": []}))
    assert s["min_agl_m"] == pytest.approx(12.0 - 9.0) and s["low_altitude_s"] > 0
    flat = safety_metrics(RunLog(Path("."), dict(meta, terrain=None), {"trajectory": traj, "collisions": []}))
    assert flat["min_agl_m"] == pytest.approx(12.0) and flat["low_altitude_s"] == 0


def test_targets_and_start_snap_and_the_dut_sees_its_limits(blocks_run):
    t = slope()
    assert blocks_run.resolved_targets()[0]["ned"][2] == -0.85  # no terrain: unchanged
    snapped = blocks_run.resolved_targets(t)[0]["ned"]
    assert snapped[2] == pytest.approx(float(t.down_at(snapped[0], snapped[1])) - 0.85)
    m = mission_spec(blocks_run, blocks_run.resolved_start(t))
    assert m.min_agl_m == blocks_run.spec["limits"]["min_agl_m"]
    assert m.energy_budget_m == blocks_run.spec["limits"]["energy_budget_m"]


def test_dut_ground_is_the_terrain_under_the_start(blocks_run):
    """On a raised map the authored ground (down 0) can be far below the real ground; the DUT
    must get the terrain height at its start, or "fly 35 m up" means "fly into the hillside"."""
    hill = TerrainModel(np.full((3, 3), -257.7), n0=-100.0, e0=-100.0, cell_m=100.0, source="raised plateau")
    start = blocks_run.resolved_start(hill)
    m = mission_spec(blocks_run, start, hill)
    assert m.ground_down_m == pytest.approx(-257.7)
    takeoff_down = m.ground_down_m - m.altitude_m          # what the reference agents fly to
    assert takeoff_down < start[2]                          # above the start, not below the ground
    assert hill.agl(np.array([[start[0], start[1], takeoff_down]]))[0] == pytest.approx(m.altitude_m)
    # flat maps keep the authored ground
    assert mission_spec(blocks_run, start).ground_down_m == blocks_run.spec["search_area"].get("ground_down_m", 0.0)
    assert mission_spec(blocks_run, start, TerrainModel.flat(0.0)).ground_down_m == 0.0


def test_variant_maps_share_the_base_terrain(tmp_path):
    base = tmp_path / "maps" / "Hill"
    base.mkdir(parents=True)
    slope().save(base / "terrain_dtm.npz")
    (tmp_path / "maps" / "Hill_Dense").mkdir()
    (tmp_path / "maps" / "Hill_Dense" / "terrain.json").write_text('{"terrain_of": "Hill"}')
    model, rec = load_for_run(tmp_path, {"scene": {"map": "Hill_Dense"}, "search_area": {"ground_down_m": 0.0}})
    assert rec["kind"] == "dtm" and rec["file"].endswith("terrain_dtm.npz") and "Hill_Dense" not in rec["file"]
    assert float(model.down_at(55, 0)) == pytest.approx(-5.5)
