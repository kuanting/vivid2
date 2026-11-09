"""DUT stack interfaces.

VIVID 2 scores each layer of a drone AI stack separately, so a stack is
composed of pluggable layers:

    Perception  : Frame -> [Detection]
    Reasoner    : Frame (+ detections, mission brief) -> Assessment   (VLM layer)
    Planner     : MissionSpec -> waypoints; may accept replanning hints
    Controller  : waypoints / setpoints -> flight (simple_flight, PX4, ...)
    VLAPolicy   : Frame + brief -> velocity command (end-to-end alternative)

An Agent wires layers together and drives a mission through a MissionContext
supplied by the runner. Vendors can implement any subset; the reference
implementations live in ``vivid2.stack.reference``.

Teams of drones (``fleet.size`` > 1 in a scenario) run in one of two modes:

    central      a FleetAgent commands every drone through a FleetContext (a ground
                 station that sees every photo)
    distributed  a DistributedTeam runs one ordinary Agent per drone; the drones
                 share information only over ``ctx.radio``
"""
from __future__ import annotations

import abc
import asyncio
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from vivid2.terrain import TerrainModel


@dataclass
class GTAnnotation:
    object_id: str
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2 (pixels); sim box, ignores occlusion
    visible_px: Optional[int] = None  # pixels of the object in the segmentation image (None: not measured)
    bbox_vis: Optional[Tuple[float, float, float, float]] = None  # tight box around the visible pixels
    contrast: Optional[float] = None  # CIELAB Delta E against the surrounding ground (low in fog); None: not measured


@dataclass
class Frame:
    t_ns: int
    frame_id: int
    rgb: np.ndarray  # H x W x 3, uint8, BGR as delivered by the sim
    cam_pos: np.ndarray  # NED (m)
    cam_q: Tuple[float, float, float, float]  # w, x, y, z
    fov_deg: float
    depth: Optional[np.ndarray] = None  # H x W planar depth (m)
    gt: List[GTAnnotation] = field(default_factory=list)  # evaluation only; stacks must not read it
    jpeg: Optional[bytes] = None  # the frame as encoded once by the runner, at camera resolution
    # Set by an agent taking a closer look: the pixel of the candidate it came to inspect. A
    # reasoner may then look at a full-resolution crop around it (a digital zoom).
    focus: Optional[Tuple[float, float]] = None

    @property
    def width(self) -> int:
        return int(self.rgb.shape[1])

    @property
    def height(self) -> int:
        return int(self.rgb.shape[0])


@dataclass
class Detection:
    label: str
    bbox: Tuple[float, float, float, float]
    score: float

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.bbox[0] + self.bbox[2]) / 2.0, (self.bbox[1] + self.bbox[3]) / 2.0)


@dataclass
class Assessment:
    """Output of a Reasoner (VLM) for one frame."""

    target_found: bool
    confidence: float = 0.0
    pixel: Optional[Tuple[float, float]] = None
    description: str = ""
    hazards: List[str] = field(default_factory=list)
    next_action: str = "continue"  # continue | inspect | skip_area | return_home
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UsageStats:
    latency_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    model: str = ""  # the model that served the reply (may differ from the one requested)
    fallback: bool = False  # True when a server-side refusal fallback answered


@dataclass
class TargetReport:
    ned: np.ndarray
    confidence: float
    source: str
    frame_id: int
    t_ns: int
    note: str = ""


@dataclass
class MissionSpec:
    """What the DUT is allowed to know about the mission (no ground truth)."""

    mission: str
    brief: str
    search_polygon_ne: List[Sequence[float]]
    altitude_m: float
    ground_down_m: float
    start_ned: Sequence[float]
    max_speed_mps: float
    timeout_s: float
    expected_targets: int
    # The limits the stack is scored on, so it can respect them.
    min_agl_m: float = 0.5
    geofence_margin_m: float = 20.0
    energy_budget_m: Optional[float] = None  # path-length proxy for the battery
    # The payload camera, so a stack can plan before its first frame: {"width": px, "height": px, "fov_deg": deg}.
    camera: Optional[Dict[str, Any]] = None
    # Coarse bare-ground terrain, like the public DEM a real crew carries (20 m cells on maps
    # with relief, flat ground otherwise): ``terrain.down_at(n, e)`` is the NED down of the
    # ground. Never the simulator's 1 m model, and no trees.
    terrain: Optional[TerrainModel] = field(default=None, repr=False, compare=False)
    # Teams only (None for one drone). ``start_ned`` is then this drone's own launch pad, while
    # ``ground_down_m`` stays the ground under the centre of the pads, one height datum for all.
    drone_id: Optional[str] = None
    team: Optional[List[Dict[str, Any]]] = None  # every drone's {"id", "start_ned"}: public knowledge
    team_mode: Optional[str] = None  # "central" or "distributed"
    min_separation_m: Optional[float] = None  # drones closer than this (3D) cost safety score

    def record(self) -> Dict[str, Any]:
        """JSON-ready fields for meta.json; the terrain is summarised, not copied."""
        out = {f.name: getattr(self, f.name) for f in fields(self)
               if f.name != "terrain" and not (f.name in TEAM_FIELDS and getattr(self, f.name) is None)}
        t = self.terrain
        out["terrain_info"] = None if t is None else {"flat": t.is_flat, "cell_m": None if t.is_flat else t.cell_m,
                                                      "source": t.source}
        return out

    @classmethod
    def from_record(cls, rec: Dict[str, Any]) -> "MissionSpec":
        """Rebuild from ``record()`` output (without the terrain grid)."""
        names = {f.name for f in fields(cls)} - {"terrain"}
        return cls(**{k: v for k, v in rec.items() if k in names})


TEAM_FIELDS = ("drone_id", "team", "team_mode", "min_separation_m")


class Perception(abc.ABC):
    name = "perception"

    @abc.abstractmethod
    def detect(self, frame: Frame) -> List[Detection]:
        ...


class Reasoner(abc.ABC):
    name = "reasoner"

    @abc.abstractmethod
    def assess(
        self, frame: Frame, detections: List[Detection], mission: MissionSpec
    ) -> Tuple[Assessment, UsageStats]:
        ...


class Planner(abc.ABC):
    name = "planner"

    @abc.abstractmethod
    def plan(self, mission: MissionSpec, camera_fov_deg: float, image_wh: Tuple[int, int]) -> List[List[float]]:
        """Return NED waypoints covering the mission."""


class Controller(abc.ABC):
    name = "controller"

    @abc.abstractmethod
    async def takeoff(self, altitude_down_m: float) -> None: ...

    @abc.abstractmethod
    async def fly_path(self, waypoints: List[List[float]], speed_mps: float) -> None:
        """Start following the path; must return promptly (non-blocking)."""

    @abc.abstractmethod
    async def goto(self, ned: Sequence[float], speed_mps: float) -> None: ...

    @abc.abstractmethod
    def path_done(self) -> bool: ...

    @abc.abstractmethod
    async def hover(self) -> None: ...

    @abc.abstractmethod
    async def return_and_land(self, home_ned: Sequence[float], speed_mps: float) -> None: ...


class VLAPolicy(abc.ABC):
    name = "vla"

    @abc.abstractmethod
    def act(self, frame: Frame, mission: MissionSpec) -> Tuple[np.ndarray, UsageStats]:
        """Return body-frame velocity command [vx, vy, vz, yaw_rate]."""


class MissionContext(abc.ABC):
    """Runner-provided services available to an Agent during a run (one drone's view of it)."""

    mission: MissionSpec
    controller: Controller
    drone_id: str = "Drone1"
    radio: Optional[Any] = None  # distributed teams: this drone's RadioPort (vivid2.radio); else None

    @abc.abstractmethod
    async def get_frame(self) -> Optional[Frame]: ...

    @abc.abstractmethod
    def pose(self) -> Tuple[np.ndarray, Tuple[float, float, float, float]]: ...

    @abc.abstractmethod
    def report_target(self, report: TargetReport) -> None: ...

    @abc.abstractmethod
    def log(self, stream: str, record: Dict[str, Any]) -> None: ...

    @abc.abstractmethod
    def should_stop(self) -> bool: ...

    @abc.abstractmethod
    def elapsed_s(self) -> float: ...

    async def sleep(self, seconds: float) -> None:
        """Wait ``seconds`` of mission time (sim time, as ``elapsed_s``).

        Pace with this, not ``asyncio.sleep``: VIVID 2 pauses the simulator while
        it records each frame, so wall time runs ahead of mission time by an
        amount that depends on the machine.
        """
        target = self.elapsed_s() + seconds
        while not self.should_stop():
            left = target - self.elapsed_s()
            if left <= 0:
                return
            await asyncio.sleep(min(left, 0.05))

    async def think(self, fn, *args, **kwargs):
        """Run the stack's own computation (a model call, a detector) and return its result.

        ``fn`` is a function, run in a worker thread, or a coroutine function. Thinking takes
        mission time: the drone flies on meanwhile. In a team, another drone's photo pauses the
        simulator while it is recorded; ``think`` charges that pause back in mission time, so a
        model does not think for free while the clock stands still. Run models through it.
        """
        if asyncio.iscoroutinefunction(fn):
            return await fn(*args, **kwargs)
        return await asyncio.to_thread(fn, *args, **kwargs)

    async def get_nav_frame(self, camera: str = "Front") -> Optional[Frame]:
        """RGB + planar depth (m) from a navigation camera, for obstacle avoidance.

        The robot's ``Front`` camera looks forward, pitched 10 degrees down, 640x480,
        90 degree FOV. These frames are not scored and carry no ground truth. A
        context without navigation cameras returns None.
        """
        return None


class Agent(abc.ABC):
    name = "agent"

    def describe(self) -> Dict[str, Any]:
        return {"agent": self.name}

    @abc.abstractmethod
    async def run(self, ctx: MissionContext) -> None:
        ...


class FleetContext(abc.ABC):
    """A central team's view of the run: every drone's context, one shared clock.

    ``drones`` maps drone id to that drone's MissionContext (its own ``get_frame``,
    ``controller``, ``pose``...). Finds are reported per drone, through that drone's context.
    """

    mission: MissionSpec  # the team's mission: start_ned is the centre of the pads, team lists them
    drones: Dict[str, MissionContext]

    @abc.abstractmethod
    def should_stop(self) -> bool: ...

    @abc.abstractmethod
    def elapsed_s(self) -> float: ...

    def t_ns(self) -> int:
        """Sim time in ns, for records (default: mission time)."""
        return int(self.elapsed_s() * 1e9)

    async def sleep(self, seconds: float) -> None:
        await next(iter(self.drones.values())).sleep(seconds)

    def log(self, stream: str, record: Dict[str, Any]) -> None:
        next(iter(self.drones.values())).log(stream, record)


class FleetAgent(abc.ABC):
    """A central team stack: one agent that commands every drone (``team_mode`` "central")."""

    name = "fleet_agent"
    team_mode = "central"

    def describe(self) -> Dict[str, Any]:
        return {"agent": self.name, "team_mode": self.team_mode}

    @abc.abstractmethod
    async def run(self, fleet: FleetContext) -> None:
        ...


class DistributedTeam:
    """A distributed team stack: one Agent per drone, in drone order (``team_mode`` "distributed").

    The agents may share information only through their contexts' ``radio``.
    """

    team_mode = "distributed"

    def __init__(self, name: str, agents: Sequence[Agent]):
        self.name = name
        self.agents = list(agents)

    def describe(self) -> Dict[str, Any]:
        first = self.agents[0].describe()
        return {**first, "agent": self.name, "team_mode": self.team_mode}
