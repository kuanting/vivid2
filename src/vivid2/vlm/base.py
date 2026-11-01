"""VLM provider interface: images + prompt + JSON schema -> JSON dict + usage."""
from __future__ import annotations

import abc
import base64
import io
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from vivid2.stack.base import UsageStats


def encode_jpeg(img_bgr: np.ndarray, max_side: int = 1024, quality: int = 85) -> Tuple[str, Tuple[int, int]]:
    """BGR uint8 array -> (base64 JPEG, (width, height) actually sent)."""
    im = Image.fromarray(img_bgr[:, :, ::-1].copy())
    scale = min(1.0, max_side / max(im.size))
    if scale < 1.0:
        im = im.resize((round(im.width * scale), round(im.height * scale)), Image.BILINEAR)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    return base64.standard_b64encode(buf.getvalue()).decode("ascii"), im.size


class VLMError(RuntimeError):
    pass


class VLMProvider(abc.ABC):
    """A vision-language model backend.

    Implementations must return a dict that conforms to ``schema`` or raise
    VLMError; the reasoner treats an error as "no decision" for that frame.
    """

    model: str = ""

    @abc.abstractmethod
    def query(
        self,
        images: List[np.ndarray],
        prompt: str,
        schema: Dict[str, Any],
        system: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], UsageStats]:
        ...
