"""VLM provider interface: images + prompt + JSON schema -> JSON dict + usage.

Images are BGR uint8 arrays or already-encoded JPEG bytes. The runner encodes
each camera frame once (``encode_jpeg_bytes`` with the defaults below) and
saves those bytes in the frame bank, so a live run and an offline evaluation
of the same frame send the model identical bytes.
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
JPEG_MAX_SIDE = 1024

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
