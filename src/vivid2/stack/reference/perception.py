"""Reference perception layers.

GroundTruthPerception: oracle upper bound built from simulator ground truth.
With segmentation pixel counts (ground truth v2) it reports exactly the
visible targets, boxed around their visible pixels; otherwise it falls back
to a depth-based occlusion test (sim boxes ignore occluders such as canopy).

OnnxYoloDetector: any YOLOv8/YOLO11-format ONNX person detector (COCO class 0),
run with onnxruntime. Stand-in for a vendor's onboard detector.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from vivid2.geo import world_to_pixel
from vivid2.sim.segmentation import MIN_VISIBLE_PX
from vivid2.stack.base import Detection, Frame, Perception


class GroundTruthPerception(Perception):
    name = "gt_perception"

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
                if a.visible_px >= MIN_VISIBLE_PX and a.bbox_vis is not None:
                    dets.append(Detection(self.label, tuple(a.bbox_vis), 1.0))
            elif self._visible(frame, a.object_id, a.bbox):
                dets.append(Detection(self.label, tuple(a.bbox), 1.0))
        return dets


class OnnxYoloDetector(Perception):
    name = "onnx_yolo"

    def __init__(self, model_path: str, input_size: int = 640, conf_thr: float = 0.25, iou_thr: float = 0.45,
                 class_ids: Sequence[int] = (0,), label: str = "person", providers: Optional[List[str]] = None):
        import onnxruntime as ort  # optional dependency

        self.session = ort.InferenceSession(model_path, providers=providers or ort.get_available_providers())
        self.input_name = self.session.get_inputs()[0].name
        self.size = input_size
        self.conf_thr = conf_thr
        self.iou_thr = iou_thr
        self.class_ids = list(class_ids)
        self.label = label

    def _letterbox(self, img: np.ndarray):
        from PIL import Image

        h, w = img.shape[:2]
        r = self.size / max(h, w)
        nw, nh = round(w * r), round(h * r)
        resized = np.asarray(Image.fromarray(img[:, :, ::-1]).resize((nw, nh), Image.BILINEAR))
        canvas = np.full((self.size, self.size, 3), 114, np.uint8)
        top, left = (self.size - nh) // 2, (self.size - nw) // 2
        canvas[top:top + nh, left:left + nw] = resized
        x = canvas.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        return x, r, left, top

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, thr: float) -> List[int]:
        order = scores.argsort()[::-1]
        keep = []
        while order.size:
            i = order[0]
            keep.append(int(i))
            xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
            yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
            xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
            yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
            inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
            area = lambda b: (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
            iou = inter / (area(boxes[i]) + area(boxes[order[1:]]) - inter + 1e-9)
            order = order[1:][iou < thr]
        return keep

    def detect(self, frame: Frame) -> List[Detection]:
        x, r, left, top = self._letterbox(frame.rgb)
        out = self.session.run(None, {self.input_name: x})[0][0]  # (4 + n_classes, N)
        out = out.T
        cls_scores = out[:, 4:]
        cls = cls_scores.argmax(1)
        score = cls_scores.max(1)
        sel = (score >= self.conf_thr) & np.isin(cls, self.class_ids)
        if not sel.any():
            return []
        cx, cy, bw, bh = out[sel, 0], out[sel, 1], out[sel, 2], out[sel, 3]
        boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)
        boxes = (boxes - [left, top, left, top]) / r
        keep = self._nms(boxes, score[sel], self.iou_thr)
        return [Detection(self.label, tuple(float(v) for v in boxes[i]), float(score[sel][i])) for i in keep]
