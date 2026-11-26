"""A crash ends the mission: a drone tipped over, or one that hit something away from its pad (vivid2.crash)."""
import asyncio
import math

from vivid2.crash import CrashWatch
from vivid2.geo import tilt_deg_of
from vivid2.runner import run_agent
from vivid2.stack.base import Agent
from vivid2.team_runtime import run_tasks
from vivid2.terrain import TerrainModel

S = 1_000_000_000
LEVEL = (1.0, 0.0, 0.0, 0.0)
ON_ITS_SIDE = (math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0)  # rolled 90 degrees
PADS = [[0.0, 0.0, -0.2], [0.0, 20.0, -0.2]]


def _watch(took_off=True):
    events = []
    w = CrashWatch(PADS, TerrainModel.flat(0.0), lambda stream, rec: events.append((stream, rec)),
                   took_off=lambda drone: took_off)
    return w, events


def test_tilt_of_an_attitude():
    s = math.sin(math.pi / 4)
    assert tilt_deg_of(LEVEL) == 0.0 and tilt_deg_of((s, 0.0, 0.0, s)) == 0.0  # a turn is not a tilt
    assert math.isclose(tilt_deg_of(ON_ITS_SIDE), 90.0) and math.isclose(tilt_deg_of((0.0, 0.0, 1.0, 0.0)), 180.0)


def test_a_drone_tipped_over_for_a_second_has_crashed():
    w, events = _watch()
    for t in (0.0, 0.5):
        w.pose(int(t * S), [30.0, 0.0, -0.3], ON_ITS_SIDE)
    assert not w.crashed()
    w.pose(1 * S, [30.0, 0.0, -0.3], ON_ITS_SIDE)
    w.pose(2 * S, [30.0, 0.0, -0.3], ON_ITS_SIDE)  # reported once
    assert w.end_reason() == "crashed"
    assert events == [("events", {"t": 1 * S, "kind": "crash", "cause": "tipped over", "tilt_deg": 90.0,
                                  "pos": [30.0, 0.0, -0.3]})]


def test_a_brief_tilt_is_not_a_crash():
    w, _ = _watch()
    for t, q in ((0.0, ON_ITS_SIDE), (0.5, LEVEL), (1.0, ON_ITS_SIDE), (1.5, ON_ITS_SIDE)):
        w.pose(int(t * S), [30.0, 0.0, -40.0], q)
    assert not w.crashed() and w.end_reason() is None


def test_takeoff_and_landing_contacts_are_not_crashes_but_hitting_a_tree_is():
    w, events = _watch()
    w.collision({"t": None, "object_name": "Landscape_0"})
    w.pose(1 * S, [1.0, 1.0, -0.5], LEVEL)  # on its pad, below 2 m
    assert not w.crashed()
    w.collision({"t": 5 * S, "object_name": "Tree_12"})
    w.pose(5 * S, [80.0, 40.0, -12.0], LEVEL)
    assert w.end_reason() == "crashed"
    assert events[-1][1] == {"t": 5 * S, "kind": "crash", "cause": "collision", "object_name": "Tree_12",
                             "pos": [80.0, 40.0, -12.0]}


def test_a_crash_during_the_takeoff_climb_is_an_error():
    w, events = _watch(took_off=False)
    w.pose(0, [0.0, 0.0, -3.0], ON_ITS_SIDE)
    w.pose(1 * S, [0.0, 0.0, -0.3], ON_ITS_SIDE)
    assert w.end_reason() == "error" and events[-1][1]["during_takeoff"] is True


def test_a_team_drone_s_crash_names_the_drone():
    w, events = _watch()
    w.collision({"t": 3 * S, "object_name": "Rock_7", "drone": "Drone2"})
    w.pose(3 * S, [60.0, 0.0, -1.0], LEVEL, drone="Drone1")  # another drone's pose judges nothing
    assert not w.crashed()
    w.pose(3 * S, [60.0, 20.0, -1.0], LEVEL, drone="Drone2")
    assert w.crash["drone"] == "Drone2" and w.crash["object_name"] == "Rock_7"


class _Stuck(Agent):
    """Waits forever on a move that never ends (a crashed drone's)."""

    async def run(self, ctx):
        await asyncio.Event().wait()


class _Ctx:
    def elapsed_s(self):
        return 0.0


def test_a_crash_stops_the_agent_and_every_team_drone():
    flag = {"crashed": False}

    async def later():
        await asyncio.sleep(0.2)
        flag["crashed"] = True

    async def stuck():
        await asyncio.Event().wait()

    async def single():
        asyncio.ensure_future(later())
        return await run_agent(_Stuck(), _Ctx(), hard_s=3600.0, stop_when=lambda: flag["crashed"])

    assert asyncio.run(single()) is True
    flag["crashed"] = False

    async def team():
        asyncio.ensure_future(later())
        return await run_tasks({"Drone1": stuck(), "Drone2": stuck()}, lambda: 0.0, 3600.0, 3.0,
                               stop_when=lambda: flag["crashed"])

    assert asyncio.run(team()) is True
