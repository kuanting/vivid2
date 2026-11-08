"""Live camera windows for watching a run (``vivid2 run --view``).

Project AirSim's counterpart of the old AirSim sub-windows is client-side
OpenCV windows (``projectairsim.image_utils.ImageDisplay``). VIVID 2 opens two:

* **Payload**: the payload camera's three images for every frame the agent
  fetches, like AirSim's RGB / depth / segmentation sub-windows:
  - RGB (left, full size): exactly what the agent is evaluated on, with the
    ground truth drawn on top. Each target's box is green when at least
    MIN_VISIBLE_PX pixels of it are visible (scaled to the image size), yellow when fewer, grey when
    hidden. The agent's detections are magenta, and its decision for the frame
    and any point it reported are shown too.
  - Depth (top right): greyscale, near = dark, 0-150 m.
  - Segmentation (bottom right): the image visibility is scored from; targets
    and distractors keep their palette colours and everything else is
    background.
* **Chase**: the chase camera's stream, for seeing what the drone is doing.

Before the first scored frame (scene loading, exposure warm-up, takeoff and the
climb to survey height), the Payload window shows a **live preview** instead:
the robot's ``PayloadPreview`` camera, which has the payload camera's mount and
lens at a quarter of its pixels and its own exposure. It is marked "not scored".
Unreal renders a camera only while someone subscribes to it, and the runner
unsubscribes from the preview at the first scored frame, so the preview never
touches the scored camera's images or exposure.

The viewer is for people only. It runs in its own thread, keeps only the newest
image per window and swallows its own errors, so it never changes a run.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Callable, Dict, Optional

import numpy as np

from vivid2.sim.segmentation import min_visible_px

DISPLAY_WIDTH = 960  # RGB panel width on screen; larger cameras are scaled down (insets keep full detail)
GREEN, YELLOW, GREY, MAGENTA, RED, WHITE = (60, 220, 60), (0, 220, 255), (150, 150, 150), (255, 0, 255), (0, 0, 255), (255, 255, 255)


TEAM_TILE = (320, 240)  # team window: one thumbnail per drone


class RunViewer:
    def __init__(self, chase_size=(420, 236), refresh_hz: float = 20.0):
        import cv2

        self.cv2 = cv2
        self.chase_size = chase_size
        self.period = 1.0 / refresh_hz
        self._lock = threading.Lock()
        self._payload: Optional[Dict[str, Any]] = None
        self._chase: Optional[np.ndarray] = None
        self._status = ""
        self._live = False  # True from start_run() until the run's first scored frame
        self._phase = ""
        self._progress = ""
        self._preview_msg = None
        self._preview: Optional[np.ndarray] = None
        self._has_preview: Optional[bool] = None  # None until the runner has looked for the camera
        self._team: Dict[str, Dict[str, Any]] = {}  # team runs: each drone's newest photo (thumbnail)
        self._dirty = False
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="vivid2-viewer", daemon=True)
        self._thread.start()

    # -- inputs (called from the runner) ---------------------------------------
    def set_status(self, text: str) -> None:
        with self._lock:
            self._status = text
            self._dirty = True

    def start_run(self, status: str) -> None:
        """A new run: back to the live preview until its first scored frame."""
        with self._lock:
            self._status = status
            self._payload = None
            self._team = {}
            self._live = True
            self._phase = "loading the scene"
            self._progress = ""
            self._preview_msg = None
            self._preview = None
            self._has_preview = None
            self._dirty = True

    def no_preview(self) -> None:
        """The robot has no preview camera: say so instead of waiting for one."""
        with self._lock:
            self._has_preview = False
            self._dirty = True

    def set_phase(self, text: str) -> None:
        """What the run is doing before the test starts (shown on the preview)."""
        with self._lock:
            self._phase = text
            self._dirty = True

    def set_progress(self, height_m: float, elapsed_s: float) -> None:
        """Drone height and mission time, shown on the preview."""
        with self._lock:
            if self._live:
                self._progress = f"height {height_m:.0f} m  t {elapsed_s:.0f} s"
                self._dirty = True

    def set_team_progress(self, heights: Dict[str, float], elapsed_s: float) -> None:
        """Every team drone's height and the mission time (preview status line, and the team window)."""
        with self._lock:
            self._progress = "  ".join(f"{d} {h:.0f} m" for d, h in heights.items()) + f"  t {elapsed_s:.0f} s"
            self._dirty = True

    @property
    def live(self) -> bool:
        return self._live

    def preview_callback(self) -> Callable[[Any, Dict[str, Any]], None]:
        with self._lock:
            self._has_preview = True

        def cb(_topic, msg):
            with self._lock:
                if self._live:
                    self._preview_msg = msg
                    self._dirty = True
        return cb

    def show_frame(self, frame, seg: Optional[np.ndarray] = None, extra: str = "", drone: Optional[str] = None) -> None:
        """A frame the agent fetched (``vivid2.stack.base.Frame``), plus its segmentation image.

        In a team (``drone`` given) the payload window shows the newest photo of any drone, and the
        team window a thumbnail of each drone's newest photo.
        """
        try:
            if drone is not None:
                thumb = self.cv2.resize(np.ascontiguousarray(frame.rgb), TEAM_TILE, interpolation=self.cv2.INTER_AREA)
                with self._lock:
                    self._team[drone] = {"image": thumb, "frame_id": frame.frame_id, "extra": extra}
            with self._lock:
                self._payload = {"frame_id": frame.frame_id, "image": frame.rgb, "depth": frame.depth, "seg": seg,
                                 "gt": list(frame.gt or []), "dets": [], "decision": None, "extra": extra,
                                 "t": frame.t_ns}
                self._live = False
                self._preview_msg = None
                self._dirty = True
        except Exception:
            pass

    def note(self, stream: str, record: Dict[str, Any]) -> None:
        """Agent output for the current frame (``detections``/``decisions``/``reports`` records)."""
        try:
            with self._lock:
                p = self._payload
                if p is None or record.get("frame_id") != p["frame_id"]:
                    if stream != "reports":
                        return
                if stream == "detections":
                    p["dets"] = record.get("dets") or []
                elif stream == "decisions":
                    p["decision"] = record.get("output") or {}
                elif stream == "reports" and p is not None:
                    p["reported"] = True
                self._dirty = True
        except Exception:
            pass

    def chase_callback(self) -> Callable[[Any, Dict[str, Any]], None]:
        # Keep the network callback trivial: the sim times out publishing to a client that
        # decodes ~30 HD frames a second in its receive thread. The display loop decodes the
        # newest message only.
        def cb(_topic, msg):
            with self._lock:
                self._chase_msg = msg
                self._dirty = True
        return cb

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=2.0)

    # -- drawing -----------------------------------------------------------------
    def _draw_payload(self, p: Dict[str, Any]) -> np.ndarray:
        """RGB panel (scaled to DISPLAY_WIDTH) with overlays, depth and segmentation beside it.

        Overlays are drawn in display coordinates; the zoom insets crop the
        full-resolution image, so a 1920 px camera shows its real detail there.
        """
        cv2 = self.cv2
        clean = np.ascontiguousarray(p["image"])
        h0, w0 = clean.shape[:2]
        self._min_px = min_visible_px(w0, h0)
        k = min(1.0, DISPLAY_WIDTH / w0)  # display pixels per camera pixel
        img = cv2.resize(clean, (round(w0 * k), round(h0 * k)), interpolation=cv2.INTER_AREA) if k < 1 else clean.copy()
        insets = [self._inset(clean, g, k) for g in p["gt"]]
        for g in p["gt"]:
            vis = getattr(g, "visible_px", None)
            box = getattr(g, "bbox_vis", None) or g.bbox
            colour = GREY if not vis else (GREEN if vis >= self._min_px else YELLOW)
            x0, y0, x1, y1 = (int(round(v * k)) for v in box)
            pad = 6  # targets are tiny from survey height; pad so the box doesn't hide them
            cv2.rectangle(img, (x0 - pad, y0 - pad), (x1 + pad, y1 + pad), colour, 2)
            label = f"{g.object_id} {vis if vis is not None else '?'}px"
            cv2.putText(img, label, (x0 - pad, max(12, y0 - pad - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)
        for d in p["dets"]:
            x0, y0, x1, y1 = (int(round(v * k)) for v in d["bbox"])
            cv2.rectangle(img, (x0, y0), (x1, y1), MAGENTA, 1)
        # magnified crops of the targets in view (bottom-left, above the status bar)
        x = 8
        for inset in insets:
            if inset is None:
                continue
            ih, iw = inset.shape[:2]
            y = img.shape[0] - 52 - ih
            if x + iw > img.shape[1] or y < 0:
                break
            img[y:y + ih, x:x + iw] = inset
            x += iw + 8
        dec = p.get("decision")
        line2 = ""
        if dec is not None:
            found = dec.get("target_found")
            line2 = f"decision: {'TARGET' if found else 'no target'}  conf {dec.get('confidence') or 0:.2f}"
            if dec.get("error"):
                line2 = f"decision: error ({str(dec['error'])[:40]})"
            if found and dec.get("pixel"):
                u, v = (int(round(c * k)) for c in dec["pixel"])
                cv2.drawMarker(img, (u, v), RED, cv2.MARKER_CROSS, 22, 2)
        if p.get("reported"):
            line2 += "  REPORTED"
        h, w = img.shape[:2]
        side = (w // 2, h // 2)
        depth = self._depth_panel(p.get("depth"), side)
        seg = self._panel(p.get("seg"), side, "no segmentation")
        img = np.hstack([img, np.vstack([depth, seg])])
        cv2.putText(img, f"RGB {w0}x{h0}", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1, cv2.LINE_AA)
        cv2.putText(img, "Depth", (w + 8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1, cv2.LINE_AA)
        cv2.putText(img, "Segmentation", (w + 8, side[1] + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1, cv2.LINE_AA)
        cv2.rectangle(img, (0, h - 44), (img.shape[1], h), (0, 0, 0), -1)
        cv2.putText(img, f"frame {p['frame_id']}  {self._status}  {p.get('extra', '')}", (8, h - 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
        cv2.putText(img, line2, (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
        return img

    def _draw_team(self, team: Dict[str, Dict[str, Any]], progress: str) -> np.ndarray:
        """Each drone's newest photo, two to a row, with the drones' heights below."""
        cv2 = self.cv2
        tiles = []
        for drone in sorted(team):
            t = team[drone]["image"].copy()
            cv2.rectangle(t, (0, 0), (t.shape[1], 24), (0, 0, 0), -1)
            cv2.putText(t, f"{drone}  frame {team[drone]['frame_id']}", (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1,
                        cv2.LINE_AA)
            tiles.append(t)
        if len(tiles) % 2:
            tiles.append(np.zeros_like(tiles[0]))
        grid = np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)])
        bar = np.zeros((28, grid.shape[1], 3), np.uint8)
        cv2.putText(bar, progress, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
        return np.vstack([grid, bar])

    def _draw_preview(self, img: Optional[np.ndarray], phase: str, progress: str,
                      has_preview: Optional[bool] = None) -> np.ndarray:
        """The live preview before the first scored frame, in the same layout as a scored frame."""
        cv2 = self.cv2
        w = DISPLAY_WIDTH
        if img is not None and getattr(img, "size", 0):
            h = round(img.shape[0] * w / img.shape[1])
            rgb = cv2.resize(np.ascontiguousarray(img[..., :3]), (w, h), interpolation=cv2.INTER_AREA)
        else:
            h = w * 3 // 4
            rgb = np.zeros((h, w, 3), np.uint8)
            msg = "this robot has no PayloadPreview camera" if has_preview is False else "waiting for the preview camera"
            cv2.putText(rgb, msg, (20, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREY, 1, cv2.LINE_AA)
        cv2.rectangle(rgb, (0, 0), (w, 36), (0, 0, 0), -1)
        cv2.putText(rgb, "LIVE PREVIEW - not scored", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, YELLOW, 2, cv2.LINE_AA)
        depth = self._panel(None, (w // 2, h // 2), "depth: from the first test frame")
        seg = self._panel(None, (w // 2, h - h // 2), "segmentation: from the first test frame")
        out = np.hstack([rgb, np.vstack([depth, seg])])
        cv2.rectangle(out, (0, h - 44), (out.shape[1], h), (0, 0, 0), -1)
        cv2.putText(out, f"{self._status}  {phase}  {progress}", (8, h - 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
        cv2.putText(out, "The test starts at survey height, with the payload camera's frame 1.", (8, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW, 1, cv2.LINE_AA)
        return out

    def _inset(self, img: np.ndarray, g, k: float = 1.0, size=(384, 288), zoom: float = 6.0) -> Optional[np.ndarray]:
        """A magnified crop of the full-resolution image around a target that is (partly) in frame.

        The crop covers the same ground at any camera resolution (``size / zoom``
        display pixels, i.e. ``size / (zoom * k)`` camera pixels).
        """
        cv2 = self.cv2
        if not getattr(g, "visible_px", None) and getattr(g, "visible_px", None) is not None:
            return None  # hidden (behind cover) - nothing to look at
        box = getattr(g, "bbox_vis", None) or g.bbox
        cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
        half = (round(size[0] / (2 * zoom * k)), round(size[1] / (2 * zoom * k)))
        h, w = img.shape[:2]
        x0 = int(np.clip(round(cx) - half[0], 0, max(0, w - 2 * half[0])))
        y0 = int(np.clip(round(cy) - half[1], 0, max(0, h - 2 * half[1])))
        crop = img[y0:y0 + 2 * half[1], x0:x0 + 2 * half[0]]
        if crop.size == 0:
            return None
        big = cv2.resize(crop, size, interpolation=cv2.INTER_NEAREST)
        vis = g.visible_px
        colour = GREY if not vis else (GREEN if vis >= self._min_px else YELLOW)
        cv2.rectangle(big, (0, 0), (big.shape[1] - 1, big.shape[0] - 1), colour, 2)
        cv2.putText(big, f"{g.object_id} x{size[0] / crop.shape[1]:.0f} ({crop.shape[1]}x{crop.shape[0]} px)", (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
        return big

    def _panel(self, img: Optional[np.ndarray], size, missing: str) -> np.ndarray:
        cv2 = self.cv2
        if img is None or not getattr(img, "size", 0):
            out = np.zeros((size[1], size[0], 3), np.uint8)
            cv2.putText(out, missing, (10, size[1] // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREY, 1, cv2.LINE_AA)
            return out
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return cv2.resize(np.ascontiguousarray(img[..., :3]), size, interpolation=cv2.INTER_NEAREST)

    def _depth_panel(self, depth: Optional[np.ndarray], size, max_m: float = 150.0) -> np.ndarray:
        if depth is None:
            return self._panel(None, size, "no depth yet (unit detected above 5 m)")
        d = np.asarray(depth, np.float32)
        grey = (np.clip(d / max_m, 0.0, 1.0) * 255).astype(np.uint8)
        grey[d <= 0] = 0  # no data
        return self._panel(grey, size, "")

    def _loop(self) -> None:
        cv2 = self.cv2
        placed = set()
        try:
            while self._running:
                with self._lock:
                    dirty, p, chase_msg = self._dirty, self._payload, getattr(self, "_chase_msg", None)
                    live, preview_msg, phase, progress = self._live, self._preview_msg, self._phase, self._progress
                    has_preview = self._has_preview
                    team = dict(self._team)
                    self._chase_msg = None
                    self._preview_msg = None
                    self._dirty = False
                chase = None
                if chase_msg is not None:
                    try:
                        from projectairsim.utils import unpack_image

                        chase = unpack_image(chase_msg)
                    except Exception:
                        chase = None
                if preview_msg is not None:
                    try:
                        from projectairsim.utils import unpack_image

                        self._preview = unpack_image(preview_msg)
                    except Exception:
                        pass
                if dirty:
                    if live:
                        cv2.imshow("VIVID2 Payload", self._draw_preview(self._preview, phase, progress, has_preview))
                        if "p" not in placed:
                            cv2.moveWindow("VIVID2 Payload", 40, 40)
                            placed.add("p")
                    elif p is not None:
                        cv2.imshow("VIVID2 Payload", self._draw_payload(p))
                        if "p" not in placed:
                            cv2.moveWindow("VIVID2 Payload", 40, 40)
                            placed.add("p")
                    if team:
                        cv2.imshow("VIVID2 Team", self._draw_team(team, progress))
                        if "t" not in placed:
                            cv2.moveWindow("VIVID2 Team", 1490, 320)  # below the chase window
                            placed.add("t")
                    if chase is not None and getattr(chase, "size", 0):
                        cv2.imshow("VIVID2 Chase", cv2.resize(chase, self.chase_size))
                        if "c" not in placed:
                            cv2.moveWindow("VIVID2 Chase", 1490, 40)  # right of the 1440 px payload window
                            placed.add("c")
                cv2.waitKey(1)
                time.sleep(self.period)
        except Exception:
            pass
        finally:
            try:
                cv2.destroyAllWindows()
                cv2.waitKey(1)
            except Exception:
                pass


def make_viewer(enabled: bool, log: Callable[[str], None] = print) -> Optional[RunViewer]:
    if not enabled:
        return None
    try:
        return RunViewer()
    except Exception as e:  # no OpenCV or no display
        log(f"  (no live view: {e!r})")
        return None
