"""Reference perception layers.

GroundTruthPerception: oracle upper bound built from simulator ground truth.
With segmentation pixel counts (ground truth v2) it reports exactly the
visible targets, boxed around their visible pixels; otherwise it falls back
to a depth-based occlusion test (sim boxes ignore occluders such as canopy).
It sees what the scoring counts as visible, so with contrast (ground truth v3)
it does not see through fog: it is perfect perception of the rendered image.

OnnxYoloDetector: a COCO-trained ONNX person detector (YOLOX, or YOLOv8/YOLO11),
run tiled at full resolution with onnxruntime. Stand-in for a vendor's onboard
detector; the default weights are YOLOX-s (see detectors.py).
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from vivid2.geo import world_to_pixel
from vivid2.sim.segmentation import visibility_state
from vivid2.stack.base import Detection, Frame, Perception


class GroundTruthPerception(Perception):
    name = "gt_perception"
    uses_ground_truth = True  # the oracle: reads the frames' ground truth (docs/rules.md)

    def __init__(self, targets: Sequence[Dict], label: str = "person", target_height_m: float = 1.7, occlusion_margin_m: float = 1.5):
        self.targets = {t["id"]: np.asarray(t["ned"], float) for t in targets}
        self.label = label
        self.half_h = target_height_m / 2.0
        self.margin = occlusion_margin_m

    def _visible(self, frame: Frame, object_id: str, bbox) -> bool:
        if frame.depth is None or object_id not in self.targets:
            return True
        centre = self.targets[object_id] + np.array([0.0, 0.0, -self.half_h])
        proj = world_to_pixel(centre, frame.cam_pos, frame.cam_q, frame.width, frame.height, frame.fov_deg)
        if proj is None:
            return False
        u, v, expected = proj
        x1, y1, x2, y2 = (int(round(c)) for c in bbox)
        patch = frame.depth[max(y1, 0):max(y2, y1 + 1), max(x1, 0):max(x2, x1 + 1)]
        if patch.size == 0:
            return False
        # Visible if a reasonable share of the box sees the target, not an occluder in front of it.
        return float(np.mean(patch >= expected - self.margin)) >= 0.25

    def detect(self, frame: Frame) -> List[Detection]:
        dets = []
        for a in frame.gt:
            if a.visible_px is not None:  # ground truth v2: segmentation says exactly what is visible
                state = visibility_state({"visible_px": a.visible_px, "contrast": a.contrast}, frame_wh=(frame.width, frame.height))
                if state == "visible" and a.bbox_vis is not None:
                    dets.append(Detection(self.label, tuple(a.bbox_vis), 1.0))
            elif self._visible(frame, a.object_id, a.bbox):
                dets.append(Detection(self.label, tuple(a.bbox), 1.0))
        return dets


YOLO_FORMATS = ("yolox", "yolov8")
YOLO_PREFERRED_PROVIDERS = ("CUDAExecutionProvider", "DmlExecutionProvider")


def detect_yolo_format(output_shape: Sequence) -> str:
    """'yolox' for a raw YOLOX head (1, N, 5 + classes); 'yolov8' for YOLOv8/YOLO11 (1, 4 + classes, N)."""
    dims = list(output_shape)[1:]
    if len(dims) == 2 and all(isinstance(d, int) for d in dims):
        return "yolox" if dims[0] > dims[1] else "yolov8"
    raise ValueError(f"cannot tell the detector's output format from shape {output_shape}; pass fmt=")


def decode_yolox(out: np.ndarray, size: int, strides: Sequence[int] = (8, 16, 32)) -> np.ndarray:
    """A raw YOLOX head output (N, 5 + classes) for a size x size input, with its boxes decoded to
    centre x, centre y, width, height in input pixels (objectness and class scores unchanged)."""
    grids, steps = [], []
    for s in strides:
        n = size // s
        yv, xv = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
        steps.append(np.full((n * n, 1), s))
    grid = np.concatenate(grids).astype(np.float32)
    stride = np.concatenate(steps).astype(np.float32)
    out = np.array(out, np.float32, copy=True)
    out[:, :2] = (out[:, :2] + grid) * stride
    out[:, 2:4] = np.exp(out[:, 2:4]) * stride
    return out


def tile_grid(w: int, h: int, tile: int = 640, overlap: float = 0.2) -> List[Tuple[int, int, int, int]]:
    """Tiles (x0, y0, x1, y1) covering a w x h image; neighbours overlap by at least ``overlap`` of a tile."""
    def starts(n: int) -> List[int]:
        if n <= tile:
            return [0]
        step = max(1, int(tile * (1.0 - overlap)))
        return sorted(set(list(range(0, n - tile, step)) + [n - tile]))

    return [(x, y, min(x + tile, w), min(y + tile, h)) for y in starts(h) for x in starts(w)]


def nms(boxes: np.ndarray, scores: np.ndarray, thr: float) -> List[int]:
    """Indices kept by greedy non-maximum suppression (boxes x1, y1, x2, y2)."""
    order = scores.argsort()[::-1]
    keep = []
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    while order.size:
        i = order[0]
        keep.append(int(i))
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (area[i] + area[rest] - inter + 1e-9)
        order = rest[iou < thr]
    return keep


class OnnxYoloDetector(Perception):
    """A COCO-format ONNX detector (YOLOX or YOLOv8/YOLO11), run tiled at full resolution.

    From 55 m a person is about 15 x 20 px of a 1920 x 1440 frame; shrunk to the detector's
    640 px input that would be 5 x 7 px. So the frame is cut into overlapping 640 px tiles
    (``tile_grid``: 4 x 3 for 1920 x 1440) and each is detected at full resolution, plus one
    pass over the whole frame for people seen up close; the boxes are merged with NMS.
    ``conf_thr`` is low: the reasoner's threshold decides what counts as a find.
    """

    name = "onnx_yolo"

    def __init__(self, model_path: str, input_size: int = 640, conf_thr: float = 0.05, iou_thr: float = 0.45,
                 class_ids: Sequence[int] = (0,), label: str = "person", fmt: Optional[str] = None,
                 tile: Optional[int] = 640, overlap: float = 0.2, whole_frame: bool = True,
                 providers: Optional[List[str]] = None, model_name: Optional[str] = None,
                 model_sha8: Optional[str] = None, session=None):
        if session is None:
            import onnxruntime as ort  # optional dependency: pip install vivid2[yolo]

            available = ort.get_available_providers()
            wanted = providers or [p for p in YOLO_PREFERRED_PROVIDERS if p in available] + ["CPUExecutionProvider"]
            opts = ort.SessionOptions()
            if "DmlExecutionProvider" in wanted:  # DirectML needs these off
                opts.enable_mem_pattern = False
                opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            session = ort.InferenceSession(model_path, sess_options=opts, providers=wanted)
        self.session = session
        self._run_lock = threading.Lock()  # one inference at a time (DirectML fails on concurrent runs)
        self.input_name = session.get_inputs()[0].name
        self.fmt = fmt or detect_yolo_format(session.get_outputs()[0].shape)
        if self.fmt not in YOLO_FORMATS:
            raise ValueError(f"fmt must be one of {YOLO_FORMATS}")
        get_providers = getattr(session, "get_providers", None)
        self.provider = get_providers()[0] if get_providers else "unknown"  # inference time is thinking time
        self.size = input_size
        self.conf_thr = conf_thr
        self.iou_thr = iou_thr
        self.class_ids = list(class_ids)
        self.label = label
        self.tile = tile
        self.overlap = overlap
        self.whole_frame = whole_frame
        self.model_name = model_name or Path(model_path).stem
        self.model_sha8 = model_sha8

    def _prepare(self, img: np.ndarray):
        """BGR image -> network input, scale and padding offset."""
        import cv2

        h, w = img.shape[:2]
        r = self.size / max(h, w)
        nw, nh = round(w * r), round(h * r)
        resized = img if (nw, nh) == (w, h) else cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.size, self.size, 3), 114, np.uint8)
        if self.fmt == "yolox":  # BGR, 0-255, padded at the bottom right
            left = top = 0
            canvas[:nh, :nw] = resized
            x = canvas.astype(np.float32)
        else:  # YOLOv8: RGB, 0-1, centred
            top, left = (self.size - nh) // 2, (self.size - nw) // 2
            canvas[top:top + nh, left:left + nw] = resized
            x = canvas[:, :, ::-1].astype(np.float32) / 255.0
        return np.ascontiguousarray(x.transpose(2, 0, 1)[None]), r, left, top

    def _detect_image(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Boxes (x1, y1, x2, y2 in ``img`` pixels) and scores of the wanted classes in one image."""
        x, r, left, top = self._prepare(img)
        with self._run_lock:
            out = self.session.run(None, {self.input_name: x})[0][0]
        if self.fmt == "yolox":
            p = decode_yolox(out, self.size)
            cls_scores = p[:, 5:] * p[:, 4:5]
        else:
            p = out.T
            cls_scores = p[:, 4:]
        cls = cls_scores.argmax(1)
        score = cls_scores.max(1)
        sel = (score >= self.conf_thr) & np.isin(cls, self.class_ids)
        if not sel.any():
            return np.zeros((0, 4), np.float32), np.zeros(0, np.float32)
        cx, cy, bw, bh = p[sel, 0], p[sel, 1], p[sel, 2], p[sel, 3]
        boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)
        return (boxes - [left, top, left, top]) / r, score[sel]

    def detect(self, frame: Frame) -> List[Detection]:
        img = frame.rgb
        h, w = img.shape[:2]
        all_boxes, all_scores = [], []
        tiles = tile_grid(w, h, self.tile, self.overlap) if self.tile and max(w, h) > self.tile else []
        if self.whole_frame or not tiles:
            tiles = [(0, 0, w, h)] + tiles
        for x0, y0, x1, y1 in tiles:
            b, s = self._detect_image(img[y0:y1, x0:x1])
            all_boxes.append(b + [x0, y0, x0, y0])
            all_scores.append(s)
        boxes = np.concatenate(all_boxes)
        scores = np.concatenate(all_scores)
        if not len(scores):
            return []
        keep = nms(boxes, scores, self.iou_thr)
        return [Detection(self.label, tuple(float(v) for v in boxes[i]), float(scores[i])) for i in keep]
