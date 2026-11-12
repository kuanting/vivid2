"""The shared clock of a run, and how a team of drones takes photos on it.

Project AirSim has one clock for the whole scene: pausing it stops every drone. A single drone
takes each photo in lockstep (pause, ask for the images, let the simulator run one capture step,
receive and record them paused, resume), so a photo costs the same mission time on any machine.

A team does the same in batches. Every drone that wants a photo at the same moment joins one
batch: one pause, one GetImages per (drone, camera), one capture step (the longest of their
cameras'), all photos recorded paused, one resume. Live on an RTX 5090 laptop
(tools/probe_fleet.py, 2026-11-08), one batch of 4 payload photos took 1.9 s of wall time against
1.2 s for one, and every photo was stamped within the 0.25 s step. A single drone asking for
several cameras at once (``get_frames``) takes them the same way, with ``lockstep_fetch``.

While one drone's photo pauses the clock, the other drones' models may still be computing in
their threads. ``SimMissionContext.think`` charges that paused time back in mission time, from
the ledger of pauses kept here, so nobody thinks for free while the clock stands still.
"""
from __future__ import annotations

import asyncio
import contextlib
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from vivid2.stack.base import FleetContext, MissionContext, MissionSpec, TargetReport

# A drone whose sleep ends within this much mission time of a batch joins it (photos taken on a
# shared beat share one batch: about 2.5x less wall time for four drones).
FRAME_COALESCE_S = 0.05
# Event-loop turns given to drones woken for a batch to ask for their photo.
WAKE_TURNS = 20
# Bumped when the team runtime changes in a way that changes team results (part of a team run's
# config hash; single-drone runs don't use it). 2: simulator builds with the team weather fix
# (patches/projectairsim-team-weather.patch: before it, every camera saw every drone's fog).
FLEET_VERSION = 2


class ClockState:
    """What every drone of a run shares about the simulator's clock and camera pipeline."""

    def __init__(self, t0_ns: int):
        self.t0_ns = t0_ns
        self.last_t_ns = t0_ns  # last sim time seen, for logging after the connection drops
        self.depth_scale: Optional[float] = None  # metres per raw depth unit, detected in flight
        self.frame_id = 0  # unique across the team and its cameras: frame-bank file names and metric joins use it
        self.frames_by_camera: Dict[str, int] = {}
        self.lockstep = True  # off if a lockstep frame fails (logged once); frames then cost machine time
        self.pause_to_record = True  # off if the simulator's clock cannot pause (logged once)
        self.capture_step_ns = int(0.25e9)  # set from the camera's capture-interval by the runner
        self.stop = False


class PauseLedger:
    """Wall-clock intervals during which the simulator was paused."""

    def __init__(self):
        self.intervals: List[Tuple[float, float]] = []
        self._since: Optional[float] = None

    def paused(self) -> None:
        if self._since is None:
            self._since = time.monotonic()

    def resumed(self) -> None:
        if self._since is not None:
            self.intervals.append((self._since, time.monotonic()))
            self._since = None

    def overlap(self, a: float, b: float) -> float:
        """Paused wall seconds within [a, b]."""
        total = 0.0
        if self._since is not None:
            total += max(0.0, b - max(a, self._since))
        for s, e in reversed(self.intervals):
            if e <= a:
                break
            total += max(0.0, min(b, e) - max(a, s))
        return total


class LockstepFailed(Exception):
    """A batch failed; the photo is fetched with the simulator running instead."""


async def lockstep_fetch(world, requests: List[Tuple[str, List[int]]], step_ns: int, register_s: float,
                         timeout_s: float) -> Tuple[int, List[Dict[int, Any]]]:
    """Ask for several cameras' images, run one capture step, receive them all (simulator paused).

    ``requests`` are (camera topic, image types). Each request goes out on its own nng context
    (as ``request_async`` does, but without its done-callback, which disconnects the client when
    a request fails). The caller pauses before and resumes after. Returns the sim time after the
    step and the images of each request, by image type. Raises if any reply is missing.
    """
    client = world.client
    socks, replies = [], []
    try:
        for topic, types in requests:
            req = {"method": f"{topic}/GetImages", "params": {"image_type_ids": [int(t) for t in types]},
                   "version": 1.0}
            sock = client.socket_services.new_context()
            socks.append(sock)
            await sock.asend(client.preprocess_request(req))
            replies.append(asyncio.ensure_future(sock.arecv_msg()))
        await asyncio.sleep(register_s)
        t_fetched = int(await asyncio.to_thread(world.continue_for_sim_time, step_ns, True))
        done, pending = await asyncio.wait(replies, timeout=timeout_s)
        if pending:
            raise TimeoutError(f"{len(pending)} of {len(replies)} photos not received in {timeout_s} s")
        return t_fetched, [{int(k): v for k, v in client.postprocess_response(r.result()).items()} for r in replies]
    finally:
        for r in replies:
            if not r.done():
                r.cancel()
        for s in socks:
            s.close()


class FetchFailed(Exception):
    """The simulator did not deliver a batch's photos (the cause is chained)."""


class TeamRuntime(ClockState):
    """The clock, photo batches and pause ledger of a team run (one per run, shared by all drones).

    Drone contexts call ``get_frames``, ``sleep_until``, ``command_gate`` and read ``ledger``.
    """

    def __init__(self, world, recorder, drone_ids: List[str], register_s: float = 0.02,
                 reply_timeout_s: float = 15.0, reply_extra_s: float = 5.0):
        super().__init__(world.get_sim_time())
        self.world = world
        self.recorder = recorder
        self.order = {d: i for i, d in enumerate(drone_ids)}
        self.register_s = register_s
        self.reply_timeout_s = reply_timeout_s  # one photo; each more in a batch adds reply_extra_s
        self.reply_extra_s = reply_extra_s
        self.ledger = PauseLedger()
        self.frame_owner: Dict[int, str] = {}
        self.last_pos: Dict[str, Any] = {}  # each drone's newest position (trajectory poller)
        self.think_stats: Dict[str, Dict[str, float]] = {}
        self.batches = 0
        self.max_batch = 0
        self._pending: List[Tuple[Any, List[str], asyncio.Future]] = []  # (drone context, cameras, future)
        self._runner: Optional[asyncio.Future] = None
        self._sleepers: List[Tuple[float, asyncio.Future]] = []
        self._cmd_active = 0
        self._cmd_idle: Optional[asyncio.Event] = None
        self._batch_idle: Optional[asyncio.Event] = None
        self._fallback = None

    # -- clock --------------------------------------------------------------------------------
    def elapsed_s(self) -> float:
        return (self.world.get_sim_time() - self.t0_ns) / 1e9

    async def sleep_until(self, target_s: float, should_stop: Callable[[], bool]) -> None:
        """Wait until ``target_s`` of mission time; a batch starting just before may wake it early."""
        loop = asyncio.get_running_loop()
        while not should_stop():
            left = target_s - self.elapsed_s()
            if left <= 0.001:
                return
            fut = loop.create_future()
            entry = (target_s, fut)
            self._sleepers.append(entry)
            try:
                await asyncio.wait({fut}, timeout=min(left, 1.0))
            finally:
                self._sleepers.remove(entry)
            if fut.done():
                return  # woken to take its photo in the batch that is starting

    # -- photos ------------------------------------------------------------------------------
    def _events(self):
        if self._cmd_idle is None:
            self._cmd_idle, self._batch_idle = asyncio.Event(), asyncio.Event()
            self._cmd_idle.set()
            self._batch_idle.set()
            self._fallback = asyncio.Lock()

    async def get_frames(self, ctx, cameras: List[str]) -> Dict[str, Any]:
        """Photos from ``ctx``'s ``cameras``, in the next batch (or fetched running once lockstep failed)."""
        self._events()
        if self.lockstep:
            fut = asyncio.get_running_loop().create_future()
            self._pending.append((ctx, list(cameras), fut))
            self._kick()
            try:
                return await fut
            except LockstepFailed:
                pass
        return await self._running_frames(ctx, cameras)

    async def get_frame(self, ctx, camera: Optional[str] = None):
        cam = camera or ctx.camera_id
        return (await self.get_frames(ctx, [cam]))[cam]

    def _kick(self) -> None:
        if self._runner is None:
            self._batch_idle.clear()  # now, not when the task starts: a gimbal command must not slip in
            self._runner = asyncio.ensure_future(self._run_batches())

    async def _run_batches(self) -> None:
        paused = False
        results: List[Tuple[asyncio.Future, Any]] = []
        errors: List[Tuple[asyncio.Future, BaseException]] = []
        failed: List[Tuple[Any, List[str], asyncio.Future]] = []
        try:
            while self._cmd_active:  # a gimbal command needs the clock running
                await self._cmd_idle.wait()
            while self._pending:
                if not paused:
                    await asyncio.to_thread(self.world.pause)
                    self.ledger.paused()
                    paused = True
                await self._wake_due()
                batch, self._pending = self._pending, []
                batch.sort(key=lambda item: self.order.get(item[0].drone_id, 0))
                if not self.lockstep:
                    failed += batch
                    continue
                # One photo per (drone, camera), in drone then camera order: two requests for the
                # same camera of the same drone share it.
                items: List[Tuple[Any, str]] = []
                for ctx, cams, _ in batch:
                    for cam in ctx._ordered(cams):
                        if not any(c is ctx and k == cam for c, k in items):
                            items.append((ctx, cam))
                try:
                    shots = await self._one_batch(items)
                    by_item = {(id(c), k): shot for (c, k), shot in zip(items, shots)}
                    for ctx, cams, f in batch:
                        got = {k: by_item[(id(ctx), k)] for k in cams}
                        err = next((v for v in got.values() if isinstance(v, BaseException)), None)
                        if err is not None:
                            errors.append((f, err))
                        else:
                            results.append((f, got))
                except FetchFailed as e:  # the simulator: photos are fetched running from now on
                    self.lockstep = False
                    self.recorder.log("events", {"t": self.last_t_ns, "kind": "lockstep_unavailable",
                                                 "error": repr(e.__cause__ or e),
                                                 "batch": [c.drone_id for c, _, _ in batch]})
                    failed += batch
                except Exception as e:  # recording the photos: the drones' agents get the error, as alone
                    errors += [(f, e) for _, _, f in batch]
        except BaseException as e:
            for _, _, fut in self._pending + failed:
                if not fut.done():
                    fut.set_exception(e if isinstance(e, Exception) else LockstepFailed())
            self._pending, failed = [], []
            raise
        finally:
            if paused:
                with contextlib.suppress(Exception):
                    await asyncio.to_thread(self.world.resume)
                self.ledger.resumed()
            self._runner = None
            self._batch_idle.set()
            for fut, frames in results:  # photos are handed over once the clock runs again
                if not fut.done():
                    fut.set_result(frames)
            for fut, err in errors:
                if not fut.done():
                    fut.set_exception(err)
            for _, _, fut in failed:
                if not fut.done():
                    fut.set_exception(LockstepFailed())
            if self._pending:
                self._kick()

    async def _wake_due(self) -> None:
        """Wake every drone whose sleep ends by now (plus FRAME_COALESCE_S), so it joins this batch."""
        if not self._sleepers:
            return
        now = self.elapsed_s()
        woke = False
        for target, fut in list(self._sleepers):
            if target <= now + FRAME_COALESCE_S and not fut.done():
                fut.set_result(None)
                woke = True
        if woke:
            for _ in range(WAKE_TURNS):
                await asyncio.sleep(0)

    async def _one_batch(self, items: List[Tuple[Any, str]]) -> List[Any]:
        """One capture step for every (drone context, camera) in ``items`` (simulator paused); their frames."""
        from projectairsim.types import ImageType

        w = self.world
        try:
            t_request = int(await asyncio.to_thread(w.get_sim_time))
            step = max(ctx._step_ns(cam) for ctx, cam in items)
            timeout = self.reply_timeout_s + self.reply_extra_s * (len(items) - 1)
            t_fetched, images = await lockstep_fetch(
                w, [(ctx._camera_topic(cam), ctx._types(cam)) for ctx, cam in items], step, self.register_s, timeout)
        except Exception as e:
            raise FetchFailed(str(e)) from e
        self.last_t_ns = t_fetched
        wall0 = time.monotonic()
        ids: List[Optional[int]] = []
        for (ctx, cam), imgs in zip(items, images):  # ids in drone, then camera order
            if imgs.get(ImageType.SCENE):
                self.frame_id += 1
                ids.append(self.frame_id)
                self.frame_owner[self.frame_id] = ctx.drone_id
            else:
                ids.append(None)
        deferred: List[Tuple[int, Dict[str, Any]]] = []

        async def record(ctx, cam, imgs, fid):
            if fid is None:
                return None
            return await ctx._record_frame(imgs, imgs[ImageType.SCENE], t_request, t_fetched, wall0, frame_id=fid,
                                           deferred=deferred, batch=self.batches + 1, camera=cam)

        # one photo's failure to record stays its own: the others are delivered
        frames = await asyncio.gather(*(record(c, k, i, fid) for (c, k), i, fid in zip(items, images, ids)),
                                      return_exceptions=True)
        for _, rec in sorted(deferred, key=lambda x: x[0]):
            self.recorder.log("frames", rec)
        self.batches += 1
        self.max_batch = max(self.max_batch, len(items))
        return list(frames)

    async def _running_frames(self, ctx, cameras: List[str]):
        """After lockstep failed: one drone at a time, fetched running, recorded paused."""
        self._events()
        async with self._fallback:
            return await ctx._running_frames(cameras)

    @contextlib.asynccontextmanager
    async def command_gate(self):
        """Hold photo batches off while a command that needs the clock running is sent (a gimbal move).

        The camera renders its new pose only while the simulator runs.
        """
        self._events()
        while self._runner is not None:
            await self._batch_idle.wait()
            await asyncio.sleep(0)
        self._cmd_active += 1
        self._cmd_idle.clear()
        try:
            yield
        finally:
            self._cmd_active -= 1
            if not self._cmd_active:
                self._cmd_idle.set()

    def timing(self) -> Dict[str, Any]:
        return {"lockstep": self.lockstep, "capture_step_s": self.capture_step_ns / 1e9,
                "pause_to_record": self.pause_to_record, "batches": self.batches, "max_batch": self.max_batch,
                "frames": self.frame_id, "frames_by_camera": dict(self.frames_by_camera)}


class SimFleetContext(FleetContext):
    """The central team's view: every drone's context on the shared clock."""

    def __init__(self, mission: MissionSpec, drones: Dict[str, MissionContext], runtime: TeamRuntime):
        self.mission = mission
        self.drones = drones
        self._runtime = runtime
        self._first = next(iter(drones.values()))

    def should_stop(self) -> bool:
        return self._first.should_stop()

    def elapsed_s(self) -> float:
        return self._first.elapsed_s()

    def t_ns(self) -> int:
        return self._first.sim_time_or_last()

    async def sleep(self, seconds: float) -> None:
        await self._first.sleep(seconds)

    def log(self, stream: str, record: Dict[str, Any]) -> None:
        self._runtime.recorder.log(stream, record)

    def report_target(self, report: TargetReport) -> None:
        """Report through the drone whose photo the find was made on."""
        owner = self._runtime.frame_owner.get(report.frame_id)
        (self.drones[owner] if owner in self.drones else self._first).report_target(report)


class TeamAgentError(RuntimeError):
    def __init__(self, who: str, error: BaseException):
        super().__init__(f"{who}: {error!r}")
        self.who = who
        self.error = error


async def run_tasks(named: Dict[str, Awaitable[Any]], elapsed_s: Callable[[], float], hard_s: float,
                    wall_factor: float) -> bool:
    """Run every agent coroutine (one per drone, or one for the team); True if stopped at ``hard_s``.

    Mission time, like ``runner.run_agent``; a wall-clock limit (``wall_factor`` x) still catches a
    simulator that stops answering. The first agent error cancels the others and is raised as a
    TeamAgentError naming its drone.
    """
    tasks = {asyncio.ensure_future(c): who for who, c in named.items()}
    wall_end = time.monotonic() + wall_factor * hard_s
    try:
        while tasks:
            done, _ = await asyncio.wait(set(tasks), timeout=1.0, return_when=asyncio.FIRST_COMPLETED)
            for t in done:
                who = tasks.pop(t)
                if not t.cancelled() and t.exception() is not None:
                    raise TeamAgentError(who, t.exception()) from t.exception()
            if tasks and (elapsed_s() > hard_s or time.monotonic() > wall_end):
                return True
        return False
    finally:
        for t in tasks:
            t.cancel()
        for t in tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await t
