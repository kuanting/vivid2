"""The shared clock of a run, and how a team of drones takes photos on it.

Project AirSim has one clock for the whole scene: pausing it stops every drone. A single drone
takes each photo in lockstep (pause, ask for the images, let the simulator run one capture step,
receive and record them paused, resume), so a photo costs the same mission time on any machine.

A team does the same in batches. Every drone that wants a photo at the same moment joins one
batch: one pause, one GetImages per drone, one capture step, all photos recorded paused, one
resume. Live on an RTX 5090 laptop (tools/probe_fleet.py, 2026-11-08), one batch of 4 photos took
1.9 s of wall time against 1.2 s for one, and every photo was stamped within the 0.25 s step.

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
# config hash; single-drone runs don't use it).
FLEET_VERSION = 1


class ClockState:
    """What every drone of a run shares about the simulator's clock and camera pipeline."""

    def __init__(self, t0_ns: int):
        self.t0_ns = t0_ns
        self.last_t_ns = t0_ns  # last sim time seen, for logging after the connection drops
        self.depth_scale: Optional[float] = None  # metres per raw depth unit, detected in flight
        self.frame_id = 0  # unique across the team: frame-bank file names and metric joins use it
        self.nav_frames = 0
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


class FetchFailed(Exception):
    """The simulator did not deliver a batch's photos (the cause is chained)."""


class TeamRuntime(ClockState):
    """The clock, photo batches and pause ledger of a team run (one per run, shared by all drones).

    Drone contexts call ``get_frame``, ``sleep_until``, ``nav_gate`` and read ``ledger``.
    """

    def __init__(self, world, recorder, drone_ids: List[str], register_s: float = 0.02,
                 reply_timeout_s: float = 15.0):
        super().__init__(world.get_sim_time())
        self.world = world
        self.recorder = recorder
        self.order = {d: i for i, d in enumerate(drone_ids)}
        self.register_s = register_s
        self.reply_timeout_s = reply_timeout_s
        self.ledger = PauseLedger()
        self.frame_owner: Dict[int, str] = {}
        self.last_pos: Dict[str, Any] = {}  # each drone's newest position (trajectory poller)
        self.think_stats: Dict[str, Dict[str, float]] = {}
        self.batches = 0
        self.max_batch = 0
        self._pending: List[Tuple[Any, asyncio.Future]] = []
        self._runner: Optional[asyncio.Future] = None
        self._sleepers: List[Tuple[float, asyncio.Future]] = []
        self._nav_active = 0
        self._nav_idle: Optional[asyncio.Event] = None
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
        if self._nav_idle is None:
            self._nav_idle, self._batch_idle = asyncio.Event(), asyncio.Event()
            self._nav_idle.set()
            self._batch_idle.set()
            self._fallback = asyncio.Lock()

    async def get_frame(self, ctx):
        """A photo for ``ctx``'s drone, in the next batch (or fetched running once lockstep failed)."""
        self._events()
        if self.lockstep:
            fut = asyncio.get_running_loop().create_future()
            self._pending.append((ctx, fut))
            self._kick()
            try:
                return await fut
            except LockstepFailed:
                pass
        return await self._running_frame(ctx)

    def _kick(self) -> None:
        if self._runner is None:
            self._batch_idle.clear()  # now, not when the task starts: a navigation fetch must not slip in
            self._runner = asyncio.ensure_future(self._run_batches())

    async def _run_batches(self) -> None:
        paused = False
        results: List[Tuple[asyncio.Future, Any]] = []
        errors: List[Tuple[asyncio.Future, BaseException]] = []
        failed: List[Tuple[Any, asyncio.Future]] = []
        try:
            while self._nav_active:  # a navigation photo is being fetched with the clock running
                await self._nav_idle.wait()
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
                try:
                    frames = await self._one_batch([c for c, _ in batch])
                    for (_, f), frame in zip(batch, frames):
                        if isinstance(frame, BaseException):
                            errors.append((f, frame))
                        else:
                            results.append((f, frame))
                except FetchFailed as e:  # the simulator: photos are fetched running from now on
                    self.lockstep = False
                    self.recorder.log("events", {"t": self.last_t_ns, "kind": "lockstep_unavailable",
                                                 "error": repr(e.__cause__ or e), "batch": [c.drone_id for c, _ in batch]})
                    failed += batch
                except Exception as e:  # recording the photos: the drones' agents get the error, as alone
                    errors += [(f, e) for _, f in batch]
        except BaseException as e:
            for _, fut in self._pending + failed:
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
            for fut, frame in results:  # photos are handed over once the clock runs again
                if not fut.done():
                    fut.set_result(frame)
            for fut, err in errors:
                if not fut.done():
                    fut.set_exception(err)
            for _, fut in failed:
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

    async def _one_batch(self, ctxs: List[Any]) -> List[Any]:
        """One capture step for every drone in ``ctxs`` (the simulator is paused); their frames."""
        from projectairsim.types import ImageType

        w = self.world
        client = w.client
        socks, replies = [], []
        try:
            t_request = int(await asyncio.to_thread(w.get_sim_time))
            for ctx in ctxs:
                req = {"method": f"{ctx.drone.sensors_topic}/{ctx.camera_id}/GetImages",
                       "params": {"image_type_ids": [int(t) for t in ctx.image_types()]}, "version": 1.0}
                sock = client.socket_services.new_context()
                socks.append(sock)
                await sock.asend(client.preprocess_request(req))
                replies.append(asyncio.ensure_future(sock.arecv_msg()))
            await asyncio.sleep(self.register_s)
            t_fetched = int(await asyncio.to_thread(w.continue_for_sim_time, self.capture_step_ns, True))
            done, pending = await asyncio.wait(replies, timeout=self.reply_timeout_s)
            if pending:
                raise TimeoutError(f"{len(pending)} of {len(replies)} photos not received in {self.reply_timeout_s} s")
            images = [{int(k): v for k, v in client.postprocess_response(r.result()).items()} for r in replies]
        except Exception as e:
            raise FetchFailed(str(e)) from e
        finally:
            for r in replies:
                if not r.done():
                    r.cancel()
            for s in socks:
                s.close()
        self.last_t_ns = t_fetched
        wall0 = time.monotonic()
        ids: List[Optional[int]] = []
        for ctx, imgs in zip(ctxs, images):  # ids in drone order
            if imgs.get(ImageType.SCENE):
                self.frame_id += 1
                ids.append(self.frame_id)
                self.frame_owner[self.frame_id] = ctx.drone_id
            else:
                ids.append(None)
        deferred: List[Tuple[int, Dict[str, Any]]] = []

        async def record(ctx, imgs, fid):
            if fid is None:
                return None
            return await ctx._record_frame(imgs, imgs[ImageType.SCENE], t_request, t_fetched, wall0, frame_id=fid,
                                           deferred=deferred, batch=self.batches + 1)

        # one drone's failure to record stays its own: the others get their photos
        frames = await asyncio.gather(*(record(c, i, fid) for c, i, fid in zip(ctxs, images, ids)),
                                      return_exceptions=True)
        for _, rec in sorted(deferred, key=lambda x: x[0]):
            self.recorder.log("frames", rec)
        self.batches += 1
        self.max_batch = max(self.max_batch, len(ctxs))
        return list(frames)

    async def _running_frame(self, ctx):
        """After lockstep failed: one photo at a time, fetched running, recorded paused."""
        self._events()
        async with self._fallback:
            return await ctx._running_frame()

    @contextlib.asynccontextmanager
    async def nav_gate(self):
        """Hold photo batches off while a navigation photo is fetched with the clock running.

        A paused clock would hold its capture back, and Project AirSim gives up after 5 s.
        """
        self._events()
        while self._runner is not None:
            await self._batch_idle.wait()
            await asyncio.sleep(0)
        self._nav_active += 1
        self._nav_idle.clear()
        try:
            yield
        finally:
            self._nav_active -= 1
            if not self._nav_active:
                self._nav_idle.set()

    def timing(self) -> Dict[str, Any]:
        return {"lockstep": self.lockstep, "capture_step_s": self.capture_step_ns / 1e9,
                "pause_to_record": self.pause_to_record, "batches": self.batches, "max_batch": self.max_batch,
                "frames": self.frame_id}


class SimFleetContext(FleetContext):
    """The central team's view: every drone's context on the shared clock."""

    def __init__(self, mission: MissionSpec, drones: Dict[str, MissionContext], runtime: TeamRuntime):
        self.mission = mission
        self.drones = drones
        self.runtime = runtime
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
        self.runtime.recorder.log(stream, record)

    def report_target(self, report: TargetReport) -> None:
        """Report through the drone whose photo the find was made on."""
        owner = self.runtime.frame_owner.get(report.frame_id)
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
