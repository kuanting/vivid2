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
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class GTAnnotation:
    object_id: str
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2 (pixels); sim box, ignores occlusion
    visible_px: Optional[int] = None  # pixels of the object in the segmentation image (None: not measured)
    bbox_vis: Optional[Tuple[float, float, float, float]] = None  # tight box around the visible pixels


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
    min_agl_m: float = 5.0
    geofence_margin_m: float = 20.0
    energy_budget_m: Optional[float] = None  # path-length proxy for the battery


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
    """Runner-provided services available to an Agent during a run."""

    mission: MissionSpec
    controller: Controller

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
