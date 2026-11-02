"""VLM provider interface: images + prompt + JSON schema -> JSON dict + usage.

Images are BGR uint8 arrays or already-encoded JPEG bytes. The runner encodes
each camera frame once at camera resolution (``FRAME_JPEG_MAX_SIDE``) and saves
those bytes in the frame bank. ``vlm_views`` derives what a model is sent from
them: the whole frame (at most ``VLM_MAX_SIDE``) or a grid of tiles. The
derivation is deterministic, so a live run and an offline evaluation of the
same frame send identical bytes.
"""
from __future__ import annotations

import abc
import base64
import hashlib
import io
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from vivid2.stack.base import UsageStats

JPEG_QUALITY = 85
JPEG_MAX_SIDE = 1024        # providers: largest side when they are given a raw array
FRAME_JPEG_MAX_SIDE = 4096  # frame bank: frames are saved at camera resolution
VLM_MAX_SIDE = 1568         # largest image side worth sending (Claude resizes anything bigger)

ImageInput = Union[np.ndarray, bytes]


def encode_jpeg_bytes(img_bgr: np.ndarray, max_side: int = JPEG_MAX_SIDE, quality: int = JPEG_QUALITY) -> bytes:
    """BGR uint8 array -> JPEG bytes, downscaled so the longer side is at most ``max_side``."""
    im = Image.fromarray(img_bgr[:, :, ::-1].copy())
    scale = min(1.0, max_side / max(im.size))
    if scale < 1.0:
        im = im.resize((round(im.width * scale), round(im.height * scale)), Image.BILINEAR)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def image_bytes(img: ImageInput, max_side: int = JPEG_MAX_SIDE) -> bytes:
    """JPEG bytes for an image input; already-encoded bytes pass through unchanged."""
    return bytes(img) if isinstance(img, (bytes, bytearray)) else encode_jpeg_bytes(img, max_side)


def encode_jpeg(img: ImageInput, max_side: int = JPEG_MAX_SIDE) -> Tuple[str, Tuple[int, int]]:
    """Image input -> (base64 JPEG, (width, height) as sent)."""
    data = image_bytes(img, max_side)
    size = Image.open(io.BytesIO(data)).size  # reads the header only
    return base64.standard_b64encode(data).decode("ascii"), size


def decode_jpeg(data: bytes) -> np.ndarray:
    """JPEG bytes -> BGR uint8 array."""
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))[:, :, ::-1].copy()


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class View:
    """One image sent to a VLM, cut from a frame: frame pixel = (x0 + u * sx, y0 + v * sy)."""

    __slots__ = ("jpeg", "width", "height", "x0", "y0", "sx", "sy")

    def __init__(self, jpeg: bytes, width: int, height: int, x0: float = 0.0, y0: float = 0.0,
                 sx: float = 1.0, sy: float = 1.0):
        self.jpeg, self.width, self.height = jpeg, width, height
        self.x0, self.y0, self.sx, self.sy = x0, y0, sx, sy

    def to_frame(self, u: float, v: float) -> Tuple[float, float]:
        return self.x0 + u * self.sx, self.y0 + v * self.sy


def parse_tiles(tiles: str) -> Tuple[int, int]:
    """'2x2' -> (2 columns, 2 rows)."""
    try:
        cols, rows = (int(v) for v in tiles.lower().split("x"))
    except Exception:
        raise ValueError(f"tiles must look like '1x1' or '2x2', got {tiles!r}") from None
    if not (1 <= cols <= 6 and 1 <= rows <= 6):
        raise ValueError(f"tiles {tiles!r} out of range (1x1 .. 6x6)")
    return cols, rows


def vlm_views(image: ImageInput, tiles: str = "1x1", max_side: int = VLM_MAX_SIDE) -> List[View]:
    """What a VLM is sent for one frame: the whole frame, or a grid of tiles at full detail.

    Deterministic, so a live run and an offline evaluation of the same banked
    JPEG send identical bytes. A frame that already fits (e.g. 960x720) is sent
    as the banked bytes unchanged.
    """
    cols, rows = parse_tiles(tiles)
    data = image_bytes(image, FRAME_JPEG_MAX_SIDE)
    w, h = Image.open(io.BytesIO(data)).size
    if cols == rows == 1 and max(w, h) <= max_side:
        return [View(data, w, h)]
    full = decode_jpeg(data)
    views = []
    xs = np.linspace(0, w, cols + 1).round().astype(int)
    ys = np.linspace(0, h, rows + 1).round().astype(int)
    for r in range(rows):
        for c in range(cols):
            crop = full[ys[r]:ys[r + 1], xs[c]:xs[c + 1]]
            ch, cw = crop.shape[:2]
            jpeg = encode_jpeg_bytes(crop, max_side)
            vw, vh = Image.open(io.BytesIO(jpeg)).size
            views.append(View(jpeg, vw, vh, float(xs[c]), float(ys[r]), cw / vw, ch / vh))
    return views


class VLMError(RuntimeError):
    pass


class VLMProvider(abc.ABC):
    """A vision-language model backend.

    Implementations must return a dict that conforms to ``schema`` or raise
    VLMError; the reasoner treats an error as an abstention for that frame.
    """

    model: str = ""

    def fingerprint(self) -> Dict[str, Any]:
        """Everything that changes the model's answer; keys the record/replay cache.

        Subclasses add their sampling and image settings.
        """
        return {"provider": type(self).__name__, "model": self.model}

    @abc.abstractmethod
    def query(
        self,
        images: List[ImageInput],
        prompt: str,
        schema: Dict[str, Any],
        system: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], UsageStats]:
        ...
