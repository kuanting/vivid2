"""Team runs on one simulator clock: shared photo batches, the navigation gate, fair thinking time."""
import asyncio
import enum
import sys
import threading
import time
import types
from types import SimpleNamespace

import numpy as np
import pytest

from vivid2.recorder import Recorder, RunLog
from vivid2.scenario import Scenario
from vivid2.stack.base import MissionSpec
from vivid2.team_runtime import PauseLedger, TeamAgentError, TeamRuntime, run_tasks


@pytest.fixture(autouse=True)
def image_types(monkeypatch):
    """Project AirSim's ImageType, or a stand-in where the client is not installed (CI)."""
    try:
        import projectairsim.types  # noqa: F401
    except ImportError:
        mod = types.ModuleType("projectairsim.types")
        mod.ImageType = enum.IntEnum("ImageType", {"SCENE": 0, "DEPTH_PLANAR": 1, "DEPTH_PERSPECTIVE": 2,
                                                   "SEGMENTATION": 3})
        pkg = types.ModuleType("projectairsim")
        pkg.types = mod
        monkeypatch.setitem(sys.modules, "projectairsim", pkg)
        monkeypatch.setitem(sys.modules, "projectairsim.types", mod)


def _images(t_ns):
    img = np.zeros((4, 4, 3), np.uint8)
    return {0: {"data": img.tobytes(), "height": 4, "width": 4, "time_stamp": t_ns, "pos_x": 0.0, "pos_y": 0.0,
                "pos_z": -50.0, "rot_w": 1.0, "rot_x": 0.0, "rot_y": 0.0, "rot_z": 0.0, "annotations": []}}


class _Socket:
    def __init__(self, world):
        self.world = world

    async def asend(self, data):
        self.world.calls.append("request")
        self.stamp = self.world.t

    async def arecv_msg(self):
        while self.world.steps_done == 0 or self.world.calls[-1] == "request":
            await asyncio.sleep(0.001)
        return {str(k): v for k, v in _images(self.stamp).items()}

    def close(self):
        pass


class _World:
    """A fake simulator recording the order of calls. Time moves only in capture steps."""

    def __init__(self, lockstep=True):
        self.t = 5_000_000_000
        self.calls = []
        self.steps_done = 0
        self.stepped = threading.Event()
        client = SimpleNamespace(unsubscribe=lambda topic: None, preprocess_request=lambda r: r,
                                 postprocess_response=lambda m: m)
        if lockstep:
            client.socket_services = SimpleNamespace(new_context=lambda: _Socket(self))
        self.client = client

    def get_sim_time(self):
        return self.t

    def pause(self):
        self.calls.append("pause")

    def resume(self):
        self.calls.append("resume")

    def continue_for_sim_time(self, ns, wait):
        self.t += ns
        self.calls.append("step")
        self.steps_done += 1
        self.stepped.set()
        return self.t


class _Drone:
    def __init__(self, world, name):
        self.world = world
        self.name = name
        self.sensors_topic = f"/Sim/Scene/robots/{name}/sensors"

    def get_images(self, camera, types_):
        self.world.calls.append(f"capture {self.name} {camera}")
        return _images(self.world.t)


@pytest.fixture
def run(root):
    return Scenario.load(root / "scenarios/forest_sar/blocks_hiker_01.jsonc").expand(seeds=[0])[0]


def _team(run, tmp_path, n=2, lockstep=True):
    from vivid2.runner import SimMissionContext

    world = _World(lockstep)
    rec = Recorder(tmp_path / "team")
    ids = [f"Drone{i + 1}" for i in range(n)]
    team = TeamRuntime(world, rec, ids)
    ctxs = [SimMissionContext(run, world, _Drone(world, i), "Payload", rec, controller=None, fov_deg=80.0,
                              drone_id=i, team=team, team_info=[{"id": j, "start_ned": [0, 0, 0]} for j in ids],
                              team_mode="central") for i in ids]
    return world, rec, team, ctxs


def test_drones_asking_together_share_one_capture_step(run, tmp_path):
    world, rec, team, (a, b) = _team(run, tmp_path)

    async def both():
        return await asyncio.gather(b.get_frame(), a.get_frame())  # Drone2 asks first

    fb, fa = asyncio.run(both())
    rec.close()
    assert world.calls == ["pause", "request", "request", "step", "resume"]
    assert fa.t_ns == fb.t_ns and (fa.frame_id, fb.frame_id) == (1, 2)  # ids in drone order
    frames = RunLog.load(tmp_path / "team")["frames"]
    assert [(f["frame_id"], f["drone"], f["batch"]) for f in frames] == [(1, "Drone1", 1), (2, "Drone2", 1)]
    assert team.timing()["batches"] == 1 and team.timing()["max_batch"] == 2
    assert team.frame_owner == {1: "Drone1", 2: "Drone2"}


def test_a_drone_asking_during_a_batch_gets_the_next_step_before_the_clock_runs(run, tmp_path):
    world, rec, team, (a, b) = _team(run, tmp_path)

    async def late():
        await asyncio.to_thread(world.stepped.wait)  # Drone1's step has started
        return await b.get_frame()

    async def both():
        return await asyncio.gather(a.get_frame(), late())

    fa, fb = asyncio.run(both())
    rec.close()
    assert world.calls == ["pause", "request", "step", "request", "step", "resume"]
    assert fb.t_ns == fa.t_ns + team.capture_step_ns


def test_a_drone_due_to_wake_joins_the_batch(run, tmp_path):
    world, rec, team, (a, b) = _team(run, tmp_path)

    async def sleeper():  # due 30 ms after Drone1's photo: woken to share its batch
        await b.sleep(0.03)
        return await b.get_frame()

    async def both():
        task = asyncio.ensure_future(sleeper())
        await asyncio.sleep(0.01)
        fa = await a.get_frame()
        return fa, await task

    fa, fb = asyncio.run(both())
    rec.close()
    assert world.calls == ["pause", "request", "request", "step", "resume"]
    assert fa.t_ns == fb.t_ns


def test_navigation_photos_wait_for_the_batch_and_hold_the_next_one_off(run, tmp_path):
    world, rec, team, (a, b) = _team(run, tmp_path)

    async def nav_after_batch_started():
        await asyncio.to_thread(world.stepped.wait)
        return await b.get_nav_frame("Front")

    async def both():
        return await asyncio.gather(a.get_frame(), nav_after_batch_started())

    asyncio.run(both())
    rec.close()
    assert world.calls == ["pause", "request", "step", "resume", "capture Drone2 Front"]


def test_after_a_failed_batch_photos_are_fetched_running_one_at_a_time(run, tmp_path):
    world, rec, team, (a, b) = _team(run, tmp_path, lockstep=False)

    async def both():
        return await asyncio.gather(a.get_frame(), b.get_frame())

    fa, fb = asyncio.run(both())
    rec.close()
    assert not team.lockstep and {fa.frame_id, fb.frame_id} == {1, 2}
    assert world.calls[:2] == ["pause", "resume"]  # the failed batch
    rest = world.calls[2:]
    assert sorted(rest[i] for i in (0, 3)) == ["capture Drone1 Payload", "capture Drone2 Payload"]
    assert rest[1:3] == ["pause", "resume"] and rest[4:] == ["pause", "resume"]
    events = RunLog.load(tmp_path / "team")["events"]
    assert [e["kind"] for e in events] == ["lockstep_unavailable"]


def test_thinking_while_another_drone_s_photo_pauses_the_clock_is_charged(run, tmp_path):
    world, rec, team, (a, b) = _team(run, tmp_path)
    slept = []

    async def fake_sleep(s):
        slept.append(s)

    b.sleep = fake_sleep

    def model():  # 0.2 s of thinking; the clock is paused for about 0.1 s of it
        team.ledger.paused()
        time.sleep(0.1)
        team.ledger.resumed()
        time.sleep(0.1)
        return "answer"

    assert asyncio.run(b.think(model)) == "answer"
    assert len(slept) == 1 and slept[0] == pytest.approx(0.1, abs=0.04)
    assert asyncio.run(b.think(lambda: 7)) == 7 and len(slept) == 1  # no pause, nothing to charge
    st = team.think_stats["Drone2"]
    assert st["calls"] == 2 and st["paused_s"] == pytest.approx(slept[0])
    rec.close()


def test_team_records_carry_the_drone(run, tmp_path):
    from vivid2.stack.base import TargetReport

    world, rec, team, (a, b) = _team(run, tmp_path)
    b.log("decisions", {"t": 1, "frame_id": 2})
    b.report_target(TargetReport(ned=[1.0, 2.0, 0.0], confidence=0.9, source="x", frame_id=2, t_ns=1))
    rec.close()
    log = RunLog.load(tmp_path / "team")
    assert log["decisions"][0]["drone"] == "Drone2" and log["reports"][0]["drone"] == "Drone2"


def test_pause_ledger_overlap():
    led = PauseLedger()
    led.intervals = [(1.0, 2.0), (3.0, 4.0)]
    assert led.overlap(0.0, 5.0) == 2.0 and led.overlap(1.5, 3.5) == 1.0 and led.overlap(4.5, 9.0) == 0.0


def test_one_drone_s_error_stops_the_team_and_names_it():
    stopped = []

    async def fine():
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            stopped.append("Drone1")
            raise

    async def broken():
        await asyncio.sleep(0.01)
        raise ValueError("bad plan")

    with pytest.raises(TeamAgentError) as e:
        asyncio.run(run_tasks({"Drone1": fine(), "Drone2": broken()}, lambda: 0.0, 100.0, 3.0))
    assert e.value.who == "Drone2" and isinstance(e.value.error, ValueError) and stopped == ["Drone1"]


def test_the_hard_time_limit_stops_every_drone():
    t = {"now": 0.0}

    async def forever():
        while True:
            t["now"] += 1.0
            await asyncio.sleep(0.3)

    assert asyncio.run(run_tasks({"Drone1": forever(), "Drone2": forever()}, lambda: t["now"], 2.0, 3.0))


def test_team_fields_stay_out_of_single_drone_records(run):
    from vivid2.runner import mission_spec

    single = mission_spec(run).record()
    assert not {"drone_id", "team", "team_mode", "min_separation_m"} & set(single)
    team = mission_spec(run, start_ned=[5, 5, -1], datum_ned=[0, 0, -1], drone_id="Drone4",
                        team=[{"id": "Drone4", "start_ned": [5, 5, -1]}], team_mode="distributed").record()
    assert team["drone_id"] == "Drone4" and team["min_separation_m"] == 5.0 and team["start_ned"] == [5, 5, -1]
    assert MissionSpec.from_record(team).team_mode == "distributed"


def test_team_runs_hash_the_team_runtime_version(root):
    from vivid2.runner import run_config_hash

    sc = Scenario.load(root / "scenarios/forest_sar/blocks_hiker_01.jsonc")
    one, = sc.expand(seeds=[0])
    one_sized, two = sc.expand(seeds=[0], extra_sweep={"fleet.size": [1, 2]})
    desc = {"agent": "x"}
    assert run_config_hash(one, desc) != run_config_hash(one_sized, desc)  # the spec gained fleet.size
    import vivid2.runner as runner

    h = run_config_hash(two, desc)
    runner.FLEET_VERSION, saved = runner.FLEET_VERSION + 1, runner.FLEET_VERSION
    try:
        assert run_config_hash(two, desc) != h and run_config_hash(one, desc) == run_config_hash(one, desc)
    finally:
        runner.FLEET_VERSION = saved


def test_an_error_recording_a_photo_reaches_the_agent_and_keeps_lockstep(run, tmp_path):
    world, rec, team, (a, b) = _team(run, tmp_path)

    async def broken(*args, **kwargs):
        raise ValueError("bad image")

    b._record_frame = broken

    async def both():
        return await asyncio.gather(a.get_frame(), b.get_frame(), return_exceptions=True)

    fa, fb = asyncio.run(both())
    rec.close()
    assert fa.frame_id == 1 and isinstance(fb, ValueError)  # Drone1 still gets its photo
    assert team.lockstep and world.calls == ["pause", "request", "request", "step", "resume"]
