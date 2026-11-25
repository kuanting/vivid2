"""Turning the drone: headings for paths and gotos, and turns on the spot (AirSimController and the fakes)."""
import asyncio
import math

import numpy as np
import pytest

from conftest import FakeWorld
from vivid2.geo import yaw_deg_of
from vivid2.metrics.control import _tracking
from vivid2.recorder import Recorder, RunLog
from vivid2.stack.reference.controllers import (TURN_LIMIT_DEG, AirSimController, TakeoffError, tilt_deg,
                                                travel_legs)

pytestmark = pytest.mark.usefixtures("airsim_types")


class _Drone:
    """Records every client call; each move finishes at once."""

    def __init__(self):
        self.calls = []
        self.yaw = 0.0

    async def _done(self):
        return None

    async def _task(self, *call):
        self.calls.append(call)
        return asyncio.ensure_future(self._done())

    async def move_on_path_async(self, path, velocity, **kw):
        return await self._task("path", [list(map(float, w)) for w in path], kw)

    async def move_to_position_async(self, n, e, d, velocity, **kw):
        return await self._task("goto", [n, e, d], kw)

    async def rotate_to_yaw_async(self, yaw, timeout_sec, margin, yaw_rate):
        self.yaw = yaw
        return await self._task("rotate", yaw, timeout_sec, margin, yaw_rate)

    def get_ground_truth_pose(self):
        c, s = math.cos(self.yaw / 2), math.sin(self.yaw / 2)
        return {"translation": {"x": 0.0, "y": 0.0, "z": -50.0}, "rotation": {"w": c, "x": 0.0, "y": 0.0, "z": s}}

    def cancel_last_task(self):
        self.calls.append(("cancel",))


def _controller():
    events = []
    drone = _Drone()
    return AirSimController(drone, lambda: 7, lambda stream, rec: events.append(rec)), drone, events


def test_holding_the_heading_sends_the_same_commands_as_before():
    c, d, ev = _controller()
    asyncio.run(c.fly_path([[10.0, 0.0, -50.0]], 5.0))
    asyncio.run(c.goto([0.0, 0.0, -40.0], 3.0))
    assert [call[2] for call in d.calls if call[0] in ("path", "goto")] == [{}, {}]  # no yaw arguments
    assert "heading" not in ev[0] and "yaw_deg" not in ev[1]


def test_headings_become_absolute_yaws_kept_short_of_due_south():
    from projectairsim.drone import YawControlMode

    c, d, ev = _controller()
    asyncio.run(c.fly_path([[0.0, 0.0, -50.0]], 5.0, heading=270.0))
    asyncio.run(c.goto([0.0, 0.0, -50.0], 3.0, yaw_deg=45.0))
    asyncio.run(c.goto([0.0, 0.0, -50.0], 3.0, yaw_deg=180.0))
    kws = [call[2] for call in d.calls if call[0] in ("path", "goto")]
    assert kws[0]["yaw_control_mode"] == YawControlMode.MaxDegreeOfFreedom and kws[0]["yaw"] == pytest.approx(-math.pi / 2)
    assert kws[1]["yaw"] == pytest.approx(math.pi / 4) and not kws[1]["yaw_is_rate"]
    assert kws[2]["yaw"] == pytest.approx(math.radians(TURN_LIMIT_DEG))
    assert ev[0]["heading"] == 270.0 and ev[1]["yaw_deg"] == 45.0 and ev[2]["yaw_deg"] == TURN_LIMIT_DEG
    with pytest.raises(ValueError):
        asyncio.run(c.fly_path([[0.0, 0.0, -50.0]], 5.0, heading="north"))


def test_facing_the_travel_direction_flies_each_leg_along_its_bearing():
    # north, east, south (kept short of 180), then a climb (holds the heading)
    path = [[20.0, 0.0, -50.0], [20.0, 30.0, -50.0], [0.0, 30.0, -50.0], [0.0, 30.0, -70.0]]
    assert [b for _, b in travel_legs([0.0, 0.0, -50.0], path)] == [0.0, 90.0, 180.0, None]
    c, d, ev = _controller()

    async def fly():
        await c.fly_path(path, 5.0, heading="travel")
        while not c.path_done():
            await asyncio.sleep(0)

    asyncio.run(fly())
    yaws = [round(math.degrees(call[2]["yaw"]), 6) if call[2] else None for call in d.calls]
    assert [call[0] for call in d.calls] == ["goto"] * 4 and yaws == [0.0, 90.0, TURN_LIMIT_DEG, None]
    assert ev[0]["heading"] == "travel"


class _SlowDrone(_Drone):
    """Moves last until cancelled."""

    async def _done(self):
        await self.release.wait()

    def cancel_last_task(self):
        super().cancel_last_task()
        self.release.set()


def test_a_new_command_stops_a_path_flown_in_pieces():
    d = _SlowDrone()
    c = AirSimController(d, lambda: 7, lambda stream, rec: None)

    async def go():
        d.release = asyncio.Event()
        await c.fly_path([[0.0, 0.0, -30.0], [20.0, 0.0, -30.0]], 5.0, heading="travel")
        for _ in range(3):
            await asyncio.sleep(0)  # the first piece (the climb) is under way
        await c.goto([0.0, 0.0, -50.0], 3.0)  # cancels it; the second piece never starts

    asyncio.run(go())
    assert [call[0] for call in d.calls] == ["goto", "cancel", "goto"]


class _TakeoffDrone(_Drone):
    """Takes off to 2 m; after the climb it is at ``end_z`` with rotation ``end_q`` (w, x, y, z)."""

    def __init__(self, end_z=-50.0, end_q=(1.0, 0.0, 0.0, 0.0)):
        super().__init__()
        self.end = {"translation": {"x": 0.0, "y": 0.0, "z": end_z}, "rotation": dict(zip("wxyz", end_q))}
        self.climbed = False

    def enable_api_control(self):
        self.calls.append(("api",))

    def arm(self):
        self.calls.append(("arm",))

    async def takeoff_async(self):
        return await self._task("takeoff")

    async def move_to_position_async(self, n, e, d, velocity, **kw):
        self.climbed = True
        return await self._task("goto", [n, e, d], velocity, kw)

    def get_ground_truth_pose(self):
        if self.climbed:
            return self.end
        return {"translation": {"x": 3.0, "y": 4.0, "z": -2.0}, "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}}


def test_the_climb_after_takeoff_has_a_time_limit():
    d = _TakeoffDrone()
    events = []
    asyncio.run(AirSimController(d, lambda: 7, lambda s, r: events.append(r)).takeoff(-50.0))
    assert [call[0] for call in d.calls] == ["api", "arm", "takeoff", "goto"]
    _, ned, speed, kw = d.calls[-1]
    assert ned == [3.0, 4.0, -50.0] and speed == 3.0
    assert kw == {"timeout_sec": pytest.approx(2 * 48.0 / 3.0 + 30.0)}  # twice the 16 s climb, plus 30 s
    assert [e["kind"] for e in events] == ["takeoff"]


@pytest.mark.parametrize("end_z, end_q", [
    (-1.0, (math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0)),  # rolled over onto its side on the ground
    (-30.0, (1.0, 0.0, 0.0, 0.0)),  # upright, but the climb ran out of time 20 m short
])
def test_a_crash_at_takeoff_ends_the_run_at_once(end_z, end_q):
    events = []
    c = AirSimController(_TakeoffDrone(end_z, end_q), lambda: 7, lambda s, r: events.append(r))
    with pytest.raises(TakeoffError, match="takeoff failed"):
        asyncio.run(c.takeoff(-50.0))
    assert events[-1]["kind"] == "takeoff_failed" and events[-1]["down"] == end_z and events[-1]["target_down"] == -50.0


def test_tilt_from_a_rotation():
    s = math.sin(math.pi / 4)
    assert tilt_deg({"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}) == pytest.approx(0.0)
    assert tilt_deg({"w": s, "x": 0.0, "y": 0.0, "z": s}) == pytest.approx(0.0)  # a turn is not a tilt
    assert tilt_deg({"w": s, "x": 0.0, "y": s, "z": 0.0}) == pytest.approx(90.0)
    assert tilt_deg({"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0}) == pytest.approx(180.0)


def test_turns_near_south_stop_short_of_the_wrap():
    c, d, ev = _controller()
    asyncio.run(c.turn_to(180.0, rate_dps=30.0))
    _, yaw, timeout, margin, rate = d.calls[-1]
    assert yaw == pytest.approx(math.radians(TURN_LIMIT_DEG)) and margin == pytest.approx(math.radians(5.0))
    assert rate == pytest.approx(math.radians(30.0)) and timeout == 20.0
    assert ev[0] == {"t": 7, "kind": "turn", "yaw_deg": TURN_LIMIT_DEG, "commanded_deg": 180.0, "rate_dps": 30.0}
    assert ev[1]["kind"] == "turn_end" and ev[1]["yaw_deg"] == pytest.approx(TURN_LIMIT_DEG, abs=0.1)
    asyncio.run(c.turn_to(-90.0))
    assert d.calls[-1][1] == pytest.approx(-math.pi / 2) and d.calls[-1][4] == 0.0


def test_a_turn_ends_a_path_s_tracking_window():
    t = np.arange(0.0, 20.0, 0.5)
    pos = np.column_stack([t, np.zeros_like(t), np.full_like(t, -50.0)])
    pos[t > 10.0, 1] = 30.0  # wandering off after the turn at 10 s
    path = {"t": 0, "kind": "path_start", "from": [0.0, 0.0, -50.0], "waypoints": [[100.0, 0.0, -50.0]]}
    errs, alt = [], []
    _tracking([path], t, pos, errs, alt)
    assert max(errs) == pytest.approx(30.0)
    errs, alt = [], []
    _tracking([path, {"t": int(10e9), "kind": "turn", "yaw_deg": 90.0}], t, pos, errs, alt)
    assert max(errs) == pytest.approx(0.0)


def test_the_fake_drone_turns_and_its_camera_turns_with_it(blocks_run, tmp_path):
    rec = Recorder(tmp_path / "w")
    world = FakeWorld(blocks_run, rec, width=64, height=48)
    asyncio.run(world.controller.turn_to(90.0))
    assert world.elapsed_s() == pytest.approx(1.0) and yaw_deg_of(world.pose()[1]) == pytest.approx(90.0)
    asyncio.run(world.point_camera(tilt_deg=0.0))
    f = asyncio.run(world.get_frame())
    from vivid2.geo import quat_to_rot

    assert quat_to_rot(f.cam_q)[:, 0] == pytest.approx([0.0, 1.0, 0.0], abs=1e-9)  # looking east, level
    asyncio.run(world.controller.fly_path([list(world.pos + [0.0, -20.0, 0.0])], 5.0, heading="travel"))
    world.advance(1.0)
    rec.close()
    assert yaw_deg_of(world.pose()[1]) == pytest.approx(-90.0)  # facing west, where it flies
    assert RunLog.load(tmp_path / "w")["events"][0]["kind"] == "turn"
