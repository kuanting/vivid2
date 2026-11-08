"""The team radio: range, delay, bandwidth, deterministic loss, JSON copies and the comms log."""
import asyncio

import pytest

from vivid2.metrics.team import comms_metrics
from vivid2.radio import Radio, RadioProfile


class _Net:
    """A radio on a hand-moved clock, with drones at fixed positions."""

    def __init__(self, profile, positions=None, seed=0):
        self.now = 0.0
        self.positions = positions or {"A": [0, 0, -50], "B": [100, 0, -50], "C": [400, 0, -50]}
        self.log = []
        self.radio = Radio(profile, list(self.positions), clock=lambda: self.now, t_ns=lambda: int(self.now * 1e9),
                           position=lambda i: self.positions[i], log=lambda s, r: self.log.append(r), seed=seed)

    async def sleep(self, s):
        self.now += s

    def port(self, i):
        return self.radio.port(i, self.sleep)


def test_messages_arrive_after_the_latency_and_only_in_range():
    net = _Net(RadioProfile(range_m=300, latency_s=0.5))
    a, b, c = net.port("A"), net.port("B"), net.port("C")
    a.broadcast({"kind": "found", "ned": [1, 2, 0]})
    assert b.poll() == [] and c.poll() == []
    net.now = 0.5
    got = b.poll()
    assert [m.payload for m in got] == [{"kind": "found", "ned": [1, 2, 0]}] and got[0].sender == "A"
    assert c.poll() == []  # 400 m away
    assert [(r["to"], r["reason"]) for r in net.log] == [("B", "ok"), ("C", "range")]
    assert net.log[0]["t_deliver"] - net.log[0]["t_send"] == 500_000_000


def test_a_sender_queues_its_messages_behind_its_bandwidth():
    net = _Net(RadioProfile(latency_s=0.0, bandwidth_kbps=8.0))  # 1000 bytes per second
    a, b = net.port("A"), net.port("B")
    blob = {"x": "y" * 990}  # 1000 bytes of JSON
    a.send("B", blob)
    a.send("B", blob)
    net.now = 1.0
    assert len(b.poll()) == 1
    net.now = 2.0
    assert len(b.poll()) == 1


def test_loss_is_the_same_whatever_else_is_sent():
    def lost(extra):
        net = _Net(RadioProfile(latency_s=0.0, loss=0.5), seed=7)
        a, c = net.port("A"), net.port("C")
        for k in range(40):
            if extra:
                c.broadcast({"noise": k})  # another drone's traffic in between
            a.send("B", {"k": k})
        return [r["seq"] for r in net.log if r["from"] == "A" and r["reason"] == "loss"]

    first = lost(False)
    assert 8 < len(first) < 32 and lost(True) == first


def test_payloads_are_copied_as_json():
    net = _Net(RadioProfile(latency_s=0.0))
    a, b = net.port("A"), net.port("B")
    payload = {"spots": [[1, 2]]}
    a.send("B", payload)
    payload["spots"].append([3, 4])
    assert b.poll()[0].payload == {"spots": [[1, 2]]}
    with pytest.raises(TypeError):
        a.send("B", {"obj": object()})


def test_recv_waits_in_mission_time():
    net = _Net(RadioProfile(latency_s=0.3))
    a, b = net.port("A"), net.port("B")
    a.send("B", "hello")
    a.send("B", "again")
    m = asyncio.run(b.recv(timeout_s=1.0))
    assert m.payload == "hello" and net.now == pytest.approx(0.3)
    assert asyncio.run(b.recv(timeout_s=1.0)).payload == "again"
    assert asyncio.run(b.recv(timeout_s=0.5)) is None


def test_comms_metrics_count_messages_once_and_losses_by_reason():
    net = _Net(RadioProfile(range_m=300, latency_s=0.2))
    a, b = net.port("A"), net.port("B")
    a.broadcast({"pos": [0, 0]})
    b.send("A", {"ack": 1})
    m = comms_metrics(net.log, ["A", "B", "C"])
    assert m["messages"] == 2 and m["deliveries"] == 2 and m["lost"] == {"range": 1}
    assert m["mean_delay_s"] == pytest.approx(0.2) and m["per_drone"]["A"]["messages"] == 1
