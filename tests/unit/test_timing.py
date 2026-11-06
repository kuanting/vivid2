"""Results must not depend on machine speed: frames are paced by mission (sim) time, VIVID 2's own
per-frame work happens with the simulator paused, and the agent's thinking takes mission time."""
import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import FakeWorld
from vivid2.metrics.mission import match_reports
from vivid2.metrics.reasoning import reasoning_metrics
from vivid2.recorder import Recorder, RunLog
from vivid2.runner import run_agent
from vivid2.stack.base import Agent, Assessment, Reasoner, UsageStats
from vivid2.stack.reference.agents import SearchAgent, SurveyAgent
from vivid2.stack.reference.planners import LawnmowerPlanner
from vivid2.vlm.replay import ReplayVLM


def _frame_times(rec_dir):
    return [r["t"] / 1e9 for r in RunLog.load(rec_dir)["frames"]]


def _survey(blocks_run, tmp_path, name, wall_delay_s=0.0, period=3.0):
    rec = Recorder(tmp_path / name)
    world = FakeWorld(blocks_run, rec)
    if wall_delay_s:
        fast = world.get_frame

        async def slow_get_frame():  # a slower machine: recording a frame takes wall time
            await asyncio.sleep(wall_delay_s)
            return await fast()

        world.get_frame = slow_get_frame
    asyncio.run(SurveyAgent("survey", LawnmowerPlanner(), frame_period_s=period).run(world))
    rec.close()
    return _frame_times(tmp_path / name)


def test_frames_are_paced_by_mission_time_not_the_wall_clock(blocks_run, tmp_path):
    fast = _survey(blocks_run, tmp_path, "fast")
    slow = _survey(blocks_run, tmp_path, "slow", wall_delay_s=0.01)
    assert len(fast) > 10
    assert slow == fast  # same frames, at the same mission times, however slow the machine
    assert np.allclose(np.diff(fast), 3.0)  # one frame every frame_period_s of mission time


class SlowThinker(Reasoner):
    """Takes ``think_s`` of mission time per frame: the simulator runs on while the agent thinks."""

    name = "slow"

    def __init__(self, world, think_s):
        self.world, self.think_s = world, think_s

    def assess(self, frame, detections, mission):
        self.world.advance(self.think_s)
        return Assessment(False), UsageStats(model="slow")


def test_thinking_takes_mission_time_and_is_recorded(blocks_run, tmp_path):
    rec = Recorder(tmp_path / "think")
    world = FakeWorld(blocks_run, rec)
    asyncio.run(SearchAgent("slow", LawnmowerPlanner(), SlowThinker(world, 2.5), frame_period_s=1.0).run(world))
    rec.close()
    log = RunLog.load(tmp_path / "think")
    assert np.allclose(np.diff(_frame_times(tmp_path / "think")), 3.5)  # capture 1 s + thinking 2.5 s > the period
    assert all(d["response_s"] == pytest.approx(2.5) for d in log["decisions"])
    assert reasoning_metrics(log)["response_mean_s"] == pytest.approx(2.5)


def test_a_find_is_timed_when_it_is_reported():
    reports = [{"t": 10e9, "t_report": 14e9, "ned": [0, 0, 0]}]
    assert match_reports(reports, [{"id": "H", "ned": [1, 0, 0]}], 10.0)["matched"]["H"]["t"] == 14e9
    old = [{"t": 10e9, "ned": [0, 0, 0]}]  # runs from before the report time was recorded
    assert match_reports(old, [{"id": "H", "ned": [1, 0, 0]}], 10.0)["matched"]["H"]["t"] == 10e9


class Overrun(Agent):
    async def run(self, ctx):
        while True:  # ignores should_stop()
            await ctx.sleep(10.0)
            await asyncio.sleep(0)


class Crash(Agent):
    async def run(self, ctx):
        raise ValueError("agent bug")


def test_hard_timeout_is_in_mission_time(blocks_run, tmp_path):
    world = FakeWorld(blocks_run, Recorder(tmp_path / "hard"))
    assert asyncio.run(run_agent(Overrun(), world, hard_s=60.0)) is True
    assert world.elapsed_s() > 60.0
    with pytest.raises(ValueError):
        asyncio.run(run_agent(Crash(), FakeWorld(blocks_run, Recorder(tmp_path / "crash")), hard_s=60.0))


def test_closed_loop_replay_takes_as_long_as_the_original_call(tmp_path, monkeypatch):
    from vivid2.stack.reference.reasoners import ASSESSMENT_SCHEMA
    from vivid2.vlm import replay

    class Timed:
        model = "timed"

        def fingerprint(self):
            return {"model": self.model}

        def query(self, images, prompt, schema, system=None):
            return {"target_found": False}, UsageStats(latency_s=0.3, model=self.model)

    slept = []
    monkeypatch.setattr(replay.time, "sleep", slept.append)
    cache = tmp_path / "vlm.jsonl"
    img = np.ones((4, 4, 3), np.uint8)
    ReplayVLM(str(cache), Timed(), "record").query([img], "p", ASSESSMENT_SCHEMA)
    ReplayVLM(str(cache), Timed(), "replay", hold_latency=True).query([img], "p", ASSESSMENT_SCHEMA)
    ReplayVLM(str(cache), Timed(), "replay").query([img], "p", ASSESSMENT_SCHEMA)  # open loop: no wait
    assert slept == [0.3]




# -- the live runner context, with a fake simulator ---------------------------------------------

def _images(t_ns):
    from projectairsim.types import ImageType

    img = np.zeros((4, 4, 3), np.uint8)
    return {ImageType.SCENE: {"data": img.tobytes(), "height": 4, "width": 4, "time_stamp": t_ns, "pos_x": 0.0,
                              "pos_y": 0.0, "pos_z": -50.0, "rot_w": 1.0, "rot_x": 0.0, "rot_y": 0.0, "rot_z": 0.0,
                              "annotations": []}}


class _Socket:
    """A socket context of the fake server: answers a GetImages once the clock has stepped."""

    def __init__(self, world):
        self.world = world

    async def asend(self, data):
        self.world.calls.append("request")
        self.stamp = self.world.t  # the camera captures at the paused pose

    async def arecv_msg(self):
        while self.world.calls[-1] != "step":
            await asyncio.sleep(0.001)
        return {str(int(k)): v for k, v in _images(self.stamp).items()}

    def close(self):
        pass


class _ClockWorld:
    """Records the order of simulator calls. A running capture takes 1.2 s of sim time (the image
    transfer); ``lockstep=False`` gives a client that cannot do lockstep frames."""

    def __init__(self, lockstep=True, pause_fails=False):
        self.t = 5_000_000_000
        self.calls = []
        self.pause_fails = pause_fails
        client = SimpleNamespace(unsubscribe=lambda topic: None, preprocess_request=lambda r: r,
                                 postprocess_response=lambda m: m)
        if lockstep:
            client.socket_services = SimpleNamespace(new_context=lambda: _Socket(self))
        self.client = client

    def get_sim_time(self):
        return self.t

    def pause(self):
        self.calls.append("pause")
        if self.pause_fails:
            raise RuntimeError("real-time clock")
        return "paused"

    def resume(self):
        self.calls.append("resume")
        return "resumed"

    def continue_for_sim_time(self, ns, wait):
        self.t += ns
        self.calls.append("step")
        return self.t


class _Drone:
    sensors_topic = "/Sim/Scene/robots/Drone1/sensors"

    def __init__(self, world):
        self.world = world

    def get_images(self, camera, types):
        self.world.calls.append("capture")
        stamp = self.world.t + 30_000_000
        self.world.t += 1_200_000_000
        return _images(stamp)


def _context(blocks_run, tmp_path, world):
    from vivid2.runner import SimMissionContext

    rec = Recorder(tmp_path / "ctx")
    return SimMissionContext(blocks_run, world, _Drone(world), "Payload", rec, controller=None, fov_deg=80.0), rec


def test_a_lockstep_frame_costs_one_capture_step(blocks_run, tmp_path):
    pytest.importorskip("projectairsim")  # the Project AirSim client (not on PyPI, so not in CI)
    world = _ClockWorld()
    ctx, rec = _context(blocks_run, tmp_path, world)
    t_ask = world.t
    frame = asyncio.run(ctx.get_frame())
    rec.close()
    # paused, asked, ran one step, then recorded paused and resumed
    assert world.calls == ["pause", "request", "step", "resume"]
    assert frame.t_ns == t_ask and world.t - t_ask == ctx.capture_step_ns
    f = RunLog.load(tmp_path / "ctx")["frames"][0]
    assert f["capture_s"] == pytest.approx(0.25) and f["record_wall_s"] >= 0.0


def test_capture_step_follows_the_capture_interval():
    from vivid2.runner import capture_step_ns

    assert capture_step_ns({"capture-interval": 0.2}) == 250_000_000
    assert capture_step_ns({"capture-interval": 0.5}) == 550_000_000


def test_without_lockstep_the_simulator_is_paused_while_a_frame_is_recorded(blocks_run, tmp_path):
    pytest.importorskip("projectairsim")
    world = _ClockWorld(lockstep=False)
    ctx, rec = _context(blocks_run, tmp_path, world)
    assert asyncio.run(ctx.get_frame()) is not None and asyncio.run(ctx.get_frame()) is not None
    rec.close()
    # 1st frame: the lockstep attempt fails (logged once), the sim is resumed, then a running capture;
    # from then on only running captures, each recorded paused
    assert world.calls == ["pause", "resume", "capture", "pause", "resume", "capture", "pause", "resume"]
    log = RunLog.load(tmp_path / "ctx")
    assert [e["kind"] for e in log["events"]] == ["lockstep_unavailable"]
    assert [f["capture_s"] for f in log["frames"]] == [pytest.approx(1.2)] * 2  # the transfer took mission time


def test_a_clock_that_cannot_pause_is_logged_once_and_runs_on(blocks_run, tmp_path):
    pytest.importorskip("projectairsim")
    world = _ClockWorld(lockstep=False, pause_fails=True)
    ctx, rec = _context(blocks_run, tmp_path, world)
    ctx.lockstep = False
    assert asyncio.run(ctx.get_frame()) is not None and asyncio.run(ctx.get_frame()) is not None
    rec.close()
    assert world.calls == ["capture", "pause", "capture"]  # no resume, and no second try
    assert [e["kind"] for e in RunLog.load(tmp_path / "ctx")["events"]] == ["pause_unavailable"]


def test_async_requests_may_outlast_the_sockets_300_s(tmp_path):
    pynng = pytest.importorskip("pynng")
    from vivid2.runner import ASYNC_RECV_TIMEOUT_S, allow_long_async_requests

    sock = pynng.Req0(recv_timeout=300000)
    allow_long_async_requests(SimpleNamespace(socket_services=sock))
    ctx = sock.new_context()
    value = pynng.ffi.new("nng_duration *")
    pynng.lib.nng_ctx_get_ms(ctx._context[0], b"recv-timeout", value)
    assert value[0] == int(ASYNC_RECV_TIMEOUT_S * 1000) and sock.recv_timeout == 300000  # sync requests keep 300 s
    ctx.close()
    sock.close()
