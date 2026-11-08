"""The radio a team's drones talk over: range, delay, bandwidth and loss, in mission time.

In a distributed team (``team_mode`` "distributed"), each drone runs its own agent, and
``ctx.radio`` is the only way to share what it knows. The scenario's ``comms`` sets the link:

    range_m         messages to a drone farther away than this (3D, when sent) are lost
    latency_s       delay before a message arrives
    bandwidth_kbps  each drone sends one message at a time; a message of B bytes occupies its
                    transmitter for 8 B / (1000 kbps) seconds, and later messages queue behind it
    loss            chance that a message to a given drone is lost (deterministic per run seed,
                    sender, message number and receiver, so a re-run loses the same messages)

Payloads are JSON (dicts, lists, numbers, strings): they are serialised when sent, which counts
their bytes and keeps drones from sharing Python objects. Every delivery or loss is logged in
``comms.jsonl``. Reports of found people go to VIVID 2 over a separate, perfect channel
(``ctx.report_target``); the radio is only between drones.
"""
from __future__ import annotations

import hashlib
import heapq
import json
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence

import numpy as np

BROADCAST = "*"


@dataclass
class RadioProfile:
    range_m: Optional[float] = None  # None: no limit
    latency_s: float = 0.1
    bandwidth_kbps: Optional[float] = None  # None: no limit
    loss: float = 0.0

    @classmethod
    def from_spec(cls, comms: Optional[Dict[str, Any]]) -> "RadioProfile":
        c = comms or {}
        return cls(range_m=c.get("range_m"), latency_s=float(c.get("latency_s", 0.1)),
                   bandwidth_kbps=c.get("bandwidth_kbps"), loss=float(c.get("loss", 0.0)))

    @classmethod
    def ideal(cls) -> "RadioProfile":
        """No range limit, no delay, no bandwidth limit, no loss (a central team's in-memory link)."""
        return cls(range_m=None, latency_s=0.0, bandwidth_kbps=None, loss=0.0)

    def record(self) -> Dict[str, Any]:
        return {"range_m": self.range_m, "latency_s": self.latency_s, "bandwidth_kbps": self.bandwidth_kbps,
                "loss": self.loss}


@dataclass
class Message:
    sender: str
    payload: Any
    sent_s: float  # mission time
    arrives_s: float
    seq: int


class Radio:
    """The team's network. ``clock()`` is mission time (s), ``t_ns()`` sim time for the log,
    ``position(id)`` a drone's NED position (only needed with a range limit)."""

    def __init__(self, profile: RadioProfile, ids: Sequence[str], clock: Callable[[], float],
                 t_ns: Callable[[], int], position: Optional[Callable[[str], Any]] = None,
                 log: Optional[Callable[[str, Dict[str, Any]], None]] = None, seed: int = 0):
        self.profile = profile
        self.ids = list(ids)
        self.clock = clock
        self.t_ns = t_ns
        self.position = position
        self._log = log
        self.seed = seed
        self._inbox: Dict[str, List] = {i: [] for i in self.ids}
        self._seq: Dict[str, int] = {i: 0 for i in self.ids}
        self._tx_free: Dict[str, float] = {i: 0.0 for i in self.ids}

    def port(self, me: str, sleep: Callable[[float], Awaitable[None]]) -> "RadioPort":
        return RadioPort(self, me, sleep)

    def _lost(self, sender: str, seq: int, to: str) -> bool:
        if self.profile.loss <= 0:
            return False
        h = hashlib.sha1(f"{self.seed}:{sender}:{seq}:{to}".encode()).hexdigest()
        return int(h[:8], 16) / 2 ** 32 < self.profile.loss

    def send(self, sender: str, to: str, payload: Any) -> None:
        data = json.dumps(payload, separators=(",", ":"), sort_keys=True)
        nbytes = len(data.encode())
        now = self.clock()
        t_send = self.t_ns()
        self._seq[sender] += 1
        seq = self._seq[sender]
        p = self.profile
        start = max(now, self._tx_free[sender])
        tx = nbytes * 8.0 / (p.bandwidth_kbps * 1000.0) if p.bandwidth_kbps else 0.0
        self._tx_free[sender] = start + tx
        arrives = start + tx + p.latency_s
        receivers = [i for i in self.ids if i != sender] if to == BROADCAST else [to]
        here = None
        for r in receivers:
            if r not in self._inbox:
                raise ValueError(f"no drone {r!r} on the radio")
            reason = "ok"
            if p.range_m is not None and self.position is not None:
                here = np.asarray(self.position(sender), float) if here is None else here
                if float(np.linalg.norm(np.asarray(self.position(r), float) - here)) > p.range_m:
                    reason = "range"
            if reason == "ok" and self._lost(sender, seq, r):
                reason = "loss"
            if reason == "ok":
                heapq.heappush(self._inbox[r], (arrives, sender, seq, json.loads(data), now))
            if self._log is not None:
                self._log("comms", {"t_send": t_send, "t_deliver": t_send + int((arrives - now) * 1e9) if reason == "ok"
                                    else None, "from": sender, "to": r, "seq": seq, "bytes": nbytes,
                                    "delivered": reason == "ok", "reason": reason})

    def poll(self, me: str) -> List[Message]:
        now = self.clock()
        box = self._inbox[me]
        out = []
        while box and box[0][0] <= now + 1e-9:
            arrives, sender, seq, payload, sent = heapq.heappop(box)
            out.append(Message(sender, payload, sent, arrives, seq))
        return out


class RadioPort:
    """One drone's radio (``ctx.radio``)."""

    def __init__(self, radio: Radio, me: str, sleep: Callable[[float], Awaitable[None]]):
        self.radio = radio
        self.me = me
        self._sleep = sleep

    @property
    def teammates(self) -> List[str]:
        return [i for i in self.radio.ids if i != self.me]

    def send(self, to: str, payload: Any) -> None:
        """Send ``payload`` (JSON) to one teammate; it may arrive late or not at all."""
        self.radio.send(self.me, to, payload)

    def broadcast(self, payload: Any) -> None:
        """Send ``payload`` to every teammate."""
        self.radio.send(self.me, BROADCAST, payload)

    def poll(self) -> List[Message]:
        """Every message that has arrived, oldest first (without waiting)."""
        return self.radio.poll(self.me)

    async def recv(self, timeout_s: Optional[float] = None, step_s: float = 0.1) -> Optional[Message]:
        """The next message, waiting up to ``timeout_s`` of mission time (None: no limit)."""
        waited = 0.0
        while True:
            got = self.poll()
            if got:
                for m in got[1:]:  # keep the rest for the next call
                    heapq.heappush(self.radio._inbox[self.me], (m.arrives_s, m.sender, m.seq, m.payload, m.sent_s))
                return got[0]
            if timeout_s is not None and waited >= timeout_s:
                return None
            await self._sleep(step_s)
            waited += step_s
