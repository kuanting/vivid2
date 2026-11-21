"""The demo movie's offline pieces (tools/demo_movie): coordinates, run loading, timeline."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

from demo_movie import ue_coords as uc  # noqa: E402
from demo_movie.run_data import Track, load_run  # noqa: E402
from vivid2.geo import quat_to_rot, rpy_to_quat  # noqa: E402


def test_ned_to_ue_axes_and_units():
    # Project AirSim: NED origin = UE origin, north -> +X, east -> +Y, down -> -Z, m -> cm
    assert uc.ned_to_ue_cm([1, 2, 3]) == (100.0, 200.0, -300.0)
    assert uc.ue_cm_to_ned(uc.ned_to_ue_cm([12.5, -7, -258.06])) == pytest.approx((12.5, -7, -258.06))


@pytest.mark.parametrize("rpy", [(0, 0, 0), (0, 0, 90), (0, 0, -135), (10, 0, 0), (0, 20, 0), (5, -15, 170),
                                 (0, -89, 30)])
def test_quaternion_to_rotator_round_trip(rpy):
    q = rpy_to_quat(*(math.radians(a) for a in rpy))
    r = uc.quat_ned_to_rotator(q)
    assert (r["roll"], r["pitch"], r["yaw"]) == pytest.approx(rpy, abs=1e-6)


def _ue_rotator_matrix(roll, pitch, yaw):
    """FRotationMatrix(FRotator(pitch, yaw, roll)) axes in UE world coordinates (rows X, Y, Z axes)."""
    sp, cp = math.sin(math.radians(pitch)), math.cos(math.radians(pitch))
    sy, cy = math.sin(math.radians(yaw)), math.cos(math.radians(yaw))
    sr, cr = math.sin(math.radians(roll)), math.cos(math.radians(roll))
    x = np.array([cp * cy, cp * sy, sp])
    y = np.array([sr * sp * cy - cr * sy, sr * sp * sy + cr * cy, -sr * cp])
    z = np.array([-(cr * sp * cy + sr * sy), cy * sr - cr * sp * sy, cr * cp])
    return x, y, z


@pytest.mark.parametrize("rpy", [(0, 0, 45), (12, -30, 100), (-20, 10, -60), (3, -88, 10)])
def test_rotator_points_the_same_way_as_the_ned_attitude(rpy):
    """The UE rotator's forward and right axes equal the NED body axes after NED -> UE conversion
    (forward/right keep their N, E components, the down component flips sign)."""
    q = rpy_to_quat(*(math.radians(a) for a in rpy))
    R = quat_to_rot(q)  # body -> NED
    fwd_ned, right_ned = R[:, 0], R[:, 1]
    r = uc.quat_ned_to_rotator(q)
    x, y, _ = _ue_rotator_matrix(r["roll"], r["pitch"], r["yaw"])
    flip = np.array([1.0, 1.0, -1.0])
    assert x == pytest.approx(fwd_ned * flip, abs=1e-9)
    assert y == pytest.approx(right_ned * flip, abs=1e-9)


def test_look_at_and_unwrap():
    r = uc.look_at_rotator((0, 0, 0), (100, 100, -141.4213562))
    assert r["yaw"] == pytest.approx(45) and r["pitch"] == pytest.approx(-45, abs=1e-6)
    assert uc.unwrap_deg([170, -175, -170, 179]) == pytest.approx([170, 185, 190, 179])


def test_focal_length_matches_payload_fov():
    assert uc.hfov_to_focal_mm(80) == pytest.approx(21.4516, abs=1e-3)
    assert uc.focal_mm_to_hfov(uc.hfov_to_focal_mm(33.0)) == pytest.approx(33.0)
    assert uc.payload_crop_16x9(1920, 1440) == (0, 180, 1920, 1260)


def test_track_dedups_and_interpolates():
    t0 = 1_000_000_000
    recs = [{"t": t0 + int(s * 1e9), "pos": p, "q": [1, 0, 0, 0]} for s, p in
            [(0.0, [0, 0, 0]), (0.003, [0, 0, 0.001]), (0.1, [1, 0, 0]), (0.44, [4.4, 0, 0])]]
    tr = Track.from_records(recs, t0)
    assert len(tr.t) == 3  # the 3 ms near-duplicate replaced its predecessor
    p, q = tr.pose_at(0.27)
    assert p[0] == pytest.approx(2.7) and q == pytest.approx([1, 0, 0, 0])
    assert tr.pose_at(-5)[0][0] == pytest.approx(0) and tr.pose_at(99)[0][0] == pytest.approx(4.4)
    # slerp half-way between yaw 0 and yaw 90 is yaw 45
    q90 = rpy_to_quat(0, 0, math.pi / 2)
    tr = Track(np.array([0.0, 1.0]), np.zeros((2, 3)), np.array([[1, 0, 0, 0], q90], float))
    assert uc.quat_ned_to_rotator(tr.pose_at(0.5)[1])["yaw"] == pytest.approx(45)


def test_load_run_from_logs(tmp_path):
    t0 = 5_000_000_000
    meta = {"t0_ns": t0, "fleet": [{"id": "Drone1"}, {"id": "Drone2"}],
            "targets": [{"id": "Hiker1", "ned": [10, 0, -250]}], "success": {"localization_tolerance_m": 10}}
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    traj = [{"t": t0 + i * 100_000_000, "pos": [i, 0, -260], "q": [1, 0, 0, 0], "drone": d}
            for i in range(20) for d in ("Drone1", "Drone2")]
    (tmp_path / "trajectory.jsonl").write_text("\n".join(json.dumps(r) for r in traj))
    events = [{"t": t0, "kind": "takeoff", "drone": "Drone1"}, {"t": t0 + 10**9, "kind": "verify_start", "drone": "Drone2"},
              {"t": t0 + 2 * 10**9, "kind": "verify_end", "drone": "Drone2"}]
    (tmp_path / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events))
    (tmp_path / "reports.jsonl").write_text(json.dumps(
        {"t": t0 + 15 * 10**8, "t_report": t0 + 16 * 10**8, "drone": "Drone2", "ned": [12, 1, -250], "frame_id": 3}))
    run = load_run(tmp_path)
    assert run.drones == ["Drone1", "Drone2"] and run.duration == pytest.approx(1.9)
    km = run.key_moments()
    assert km["takeoff"] == {"Drone1": 0.0}
    assert km["verify"][0]["start"] == pytest.approx(1.0) and km["verify"][0]["end"] == pytest.approx(2.0)
    hit = run.matched_reports()["Hiker1"]
    assert hit["rt"] == pytest.approx(1.6) and hit["error_m"] == pytest.approx(math.hypot(2, 1))


# -- the edit -------------------------------------------------------------------------------------

from demo_movie import compose as cp  # noqa: E402
from demo_movie import typo as ty  # noqa: E402
from demo_movie.data_shots import MapView  # noqa: E402


class _Solid:
    def __init__(self, n, value):
        self.n, self.value = n, value

    def frame(self, i):
        return np.full((ty.H, ty.W, 3), self.value, np.uint8)


class _Sink:
    def __init__(self):
        self.frames = []

    def write(self, f):
        self.frames.append(int(f[540, 960, 0]))


def test_concat_dissolve_overlaps_parts():
    c = cp.Concat([_Solid(10, 0), _Solid(10, 200)], dissolve=4)
    assert c.n == 16
    assert c.frame(0)[0, 0, 0] == 0 and c.frame(15)[0, 0, 0] == 200
    mids = [int(c.frame(i)[0, 0, 0]) for i in range(6, 10)]
    assert mids == sorted(mids) and 0 < mids[0] < mids[-1] < 200


def test_render_writes_every_frame_once_and_ranges():
    items = [cp.Item("A", _Solid(5, 10), grade=False), cp.Item("B", _Solid(7, 90), transition="fade:3", grade=False)]
    sink = _Sink()
    assert cp.render(items, sink) == 12 and len(sink.frames) == 12
    assert sink.frames[:5] == [10] * 5 and sink.frames[5] < sink.frames[7] < sink.frames[8] == 90
    sink = _Sink()
    assert cp.render(items, sink, frames=range(4, 8)) == 4


def test_unrendered_shot_is_a_placeholder(tmp_path):
    s = cp.Seq(tmp_path, "S99", 30, label="test")
    img = s.frame(3)
    assert img.shape == (ty.H, ty.W, 3) and img.mean() > 10


def test_map_view_is_north_up():
    v = MapView.ortho(0.0, 0.0, 0.5)
    assert v.px((0, 0, 0)) == pytest.approx((960, 540))
    assert v.px((10, 0, 0)) == pytest.approx((960, 520), abs=1e-3)   # north is up
    assert v.px((0, 10, 0)) == pytest.approx((980, 540), abs=1e-3)   # east is right


def test_golden_plan_shot_lengths_match_the_edit():
    from demo_movie.golden_plan import FRAMES
    # the edit uses: S03a 55 (+8 black), S10 90, S12 45, S14 120 of the rendered frames
    assert FRAMES["S03a"] >= 55 and FRAMES["S10"] >= 90
    assert FRAMES["S12"] >= 45 and FRAMES["S14"] >= 120
