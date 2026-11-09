# Teams of drones

A scenario can be flown by a team of up to four drones, so stacks can be compared on
cooperative search: who searches where, who looks closer at what, what to tell the others, and
how to keep out of each other's way. Teams are scored like single drones, on the same six
layers and weights, plus a team section and a scaling table on the scorecard.

## Running a team

```
vivid2 run --campaign campaigns/forest_team_v1.jsonc --agent coverage_gt --team central
vivid2 run --campaign campaigns/forest_team_v1.jsonc --agent coverage_gt --team dist
```

`--team` picks one of two modes. Both are listed in separate tables.

Teams need a simulator build with the team weather fix (VIVID2Env builds from 2026-11-09, or a
plugin with `patches/projectairsim-team-weather.patch`): without it every drone's camera also
sees the other drones' fog, so team photos come out hazier than a lone drone's.

| Mode | Who decides | What the drones share |
|---|---|---|
| `central` | One `FleetAgent` commands every drone. | Everything: like a ground station that sees every photo. |
| `dist` (distributed) | Each drone runs its own `Agent`. | Only what they send over the scenario's simulated radio (`ctx.radio`). |

Any reference stack (`coverage_gt`, `coverage_yolo`, `vlm`, `hybrid`, with `--verify` and
`--dip`) flies as a team. The mode is part of the agent's name (`coverage_gt-central`,
`coverage_gt-dist`), and the team size is part of the run.

## In a scenario

```jsonc
"fleet": {"size": 4, "spacing_m": 10},          // sweepable: "sweep": {"fleet.size": [1, 2, 4]}
"comms": {"range_m": 400, "latency_s": 0.2, "bandwidth_kbps": 250, "loss": 0.02},
"limits": {..., "min_agl_m": 0.5, "energy_budget_m": 3500, "min_separation_m": 5}
```

- **Launch pads:** a grid around `dut_start`, `spacing_m` apart: a row of two, or a 2 × 2
  square. Pads must be at least 10 m apart, so each drone's 5 m home-pad radius is its own.
  `fleet.drones` lists pads explicitly instead (the first `size` are used).
- **Drone names:** Drone1 to DroneN. All drones use the scenario's robot config and its
  payload camera. Teams fly simple_flight; PX4 SITL needs per-vehicle ports.
- **Unchanged single-drone runs:** a scenario without `fleet` is unchanged, and so are its runs
  and their config hashes. Sweeping `fleet.size` from a campaign
  (`"sweep": {"fleet.size": [1, 2, 4]}`) turns any scenario into a team scenario.

`forest_team_01` (`campaigns/forest_team_v1.jsonc`) has four hikers scattered over 650 × 550 m
of ForestSAR at dusk, three times the single-drone area. It is flown with 1, 2 and 4 drones,
3 seeds each. Its limits are set so that one drone cannot find everyone. Offline (every hiker
in view counts as seen), with the reference team:
- **one drone** surveys 70 % of the area before the 480 s timeout, finds 2 of 4 hikers, and
  runs past its battery;
- **two drones** find everyone in about 310 s;
- **four drones** find everyone in 150–230 s, depending on where the hikers lie (the strips
  holding two hikers take longest).

Live, dusk and fog hide some hikers from 60 m whatever the team size: finding them is part of
the challenge (closer looks, dips, lower flight).

## Writing a team stack

**Central:** subclass `vivid2.stack.base.FleetAgent` and implement `async run(fleet)`.
- `fleet.drones` maps each drone's id to its own `MissionContext`: `get_frame()`,
  `controller`, `pose()`, `get_nav_frame()` and so on, as in
  [adding_a_dut.md](adding_a_dut.md).
- `fleet.mission` is the team's mission: `start_ned` is the centre of the pads, and `team`
  lists every drone's id and pad.
- Report a find through the drone whose photo it was made on
  (`fleet.drones[id].report_target(...)`, or `fleet.report_target(...)`, which works that out).

**Distributed:** build a `vivid2.stack.base.DistributedTeam(name, [agent per drone])`.
- Each agent gets its own drone's context, with:
  - `ctx.drone_id`;
  - `ctx.mission.team`, every drone's id and launch pad (public knowledge);
  - `ctx.radio`.
- **Rule:** the agents share information only over the radio. Nothing in-process enforces this.
  Published results will check it.

In both modes, a drone's `ctx.mission.start_ned` is its own pad. `ground_down_m` is the ground
under the centre of the pads, one height datum for the whole team.

Return either object from your agent factory (`vivid2/registry.py` shows how the reference
team is built).

### The radio (`vivid2/radio.py`)

```python
ctx.radio.broadcast({"kind": "found", "ned": [120.0, -40.0, 0.0]})
ctx.radio.send("Drone3", {"kind": "take", "strip": 2})
for msg in ctx.radio.poll():           # everything that has arrived; never waits
    msg.sender, msg.payload, msg.sent_s, msg.arrives_s
msg = await ctx.radio.recv(timeout_s=5.0)   # waits in mission time
```

- **Payloads are JSON.** They are copied when sent, and their size counts.
- **Delivery:** a message arrives after `latency_s`, behind the sender's earlier messages at
  `bandwidth_kbps`.
- **Range:** a message is lost if the receiver is more than `range_m` away (3D) when it is sent.
- **Loss:** a fraction `loss` of messages is lost, the same ones on a re-run (seeded per run,
  sender, message number and receiver).
- **Logging:** every delivery and loss is logged in `comms.jsonl`.
- **Reports** of found people reach VIVID 2 over a separate, perfect channel; the radio is only
  between drones.

### Time

- **One mission clock for the whole team.**
  - A drone's photo costs 0.25 s of mission time, as for one drone.
  - Photos of drones that ask at the same moment are taken together, in one lockstep batch.
    Drones whose sleep ends within 0.05 s of a batch are woken to join it, so photos taken on a
    shared beat share a batch.
  - Measured on an RTX 5090 laptop: one batch of 4 photos takes 1.9 s of wall time, against 1.2 s
    for one photo (`tools/probe_fleet.py`).
- **Thinking takes mission time, as for one drone. Run models through `await ctx.think(fn, ...)`.**
  - While one drone's photo is recorded the clock stands still, but other drones' models keep
    computing in their threads.
  - `think` charges that paused time back, so no drone thinks for free.
  - Code that blocks the event loop stalls every drone: keep heavy work in `think`.
- **Shared detector (known limit):** the reference YOLO detector serves all drones from one
  GPU, one frame at a time. A drone waiting for it loses mission time, which is fair for a
  central ground station but a little harsh for a distributed team, whose drones would each
  have their own computer.

## Scoring

The six layers and weights are the same as for one drone. A team is scored as a whole:

- **Mission:** every target found by any drone. A second report of a found person is not a false
  alarm. Time to find is team mission time.
- **Planning:** coverage is the union of every drone's photos, and path length is the sum of
  the drones' paths.
- **Control:** each drone is measured against its own commands; the errors are pooled.
- **Safety:**
  - Each drone has its own battery: the budget is exceeded if any drone exceeds it, and
    `energy_used_m` is the highest drone's.
  - Every drone must end within 5 m of its own pad.
  - Geofence and low-altitude seconds add up over the drones.
  - A drone–drone collision is a collision (both drones report it; it counts once) and caps the
    overall score at 25.
  - Each pair-second two drones spend closer than `limits.min_separation_m` (3D) costs 3
    points, up to 30.

**The team section of `result.json`** (reported, not weighted):
- `time_to_find_all_s`: a target never found counts the whole timeout;
- duplicate reports;
- frames and reports per drone;
- drones used;
- coverage overlap between drones and workload balance;
- time to 90 % coverage;
- comms: messages, bytes, deliveries, losses by reason, mean delay.

**The scorecard's "Team scaling" table** compares each team stack across team sizes.
- **Speedup:** one drone's mean time to find everyone over N drones' mean, on the runs both
  flew (same scenario, settings and seed).
- **Efficiency:** speedup / N.

## The reference team

`vivid2/stack/reference/team.py` is the floor for team algorithms, safe and simple:
- **Strips at one height:** each drone flies the reference lawnmower over its own equal-width
  east–west strip at the survey height, so its photos are the same as a lone drone's.
- **Transit heights:** to reach its strip and to come home, a drone flies a layer
  (`min_separation_m` + 2 m) or more above the survey height. The drone whose strip starts
  farthest away flies highest, so a drone passing over a nearer strip is always above the drone
  climbing or descending there.
- **The pads' keep-out circle:** drones climb and land vertically over their own pads. Nobody
  flies across the circle around the pads (15 m beyond the outermost pad): each drone enters
  and leaves it straight along its own pad's direction, and survey lanes go around it.
- **Closer looks and dips** only in the drone's own strip; a candidate in a teammate's strip is
  left to that teammate.
- **Sharing:** over the radio, a drone tells its teammates:
  - every candidate it goes to look at closer (a claim),
  - every person it reports or dismisses,
  - its position, once a second.

  A spot a teammate claimed or reported is left alone. The search ends when the team has found
  everyone.
- **One beat:** photos are taken on the same beat, so they share batches.
- **Clear below:** before changing height (a closer look, a dip, or between survey and transit
  height), a drone waits until no teammate is within 30 m horizontally.

Dynamic re-allocation, probability maps and comms-aware planning are left to the stacks under
test.

## Records

Team runs add `drone` to every record (and `other_drone` to drone–drone collisions), `batch` to
frames, and write `comms.jsonl`.

`meta.json` adds:
- `fleet` (every drone's id and pad);
- `team` (mode, size, pad centre);
- `comms` (the radio used);
- `think` (per drone: thinking calls, wall seconds, paused seconds charged back);
- `frame_timing.batches` and `max_batch`.

`start_ned` is Drone1's pad.
