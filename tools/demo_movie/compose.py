"""The edit: segments (rendered shots, data shots, cards) on a timeline, with captions and transitions.

A segment has ``n`` frames and ``frame(i) -> BGR 1920x1080``. A rendered shot that is not rendered yet
shows a labelled placeholder, so the cut can be watched as an animatic before any rendering.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from demo_movie import typo as ty

W, H = ty.W, ty.H


class Seq:
    """A Movie Render Queue shot: <root>/<shot>/<shot>.<frame:06d>.png (frames offset..offset+n-1)."""

    def __init__(self, root: Path, shot: str, n: int, offset: int = 0, label: str = "", hold_last: bool = True):
        self.root, self.shot, self.n, self.offset, self.label = Path(root), shot, n, offset, label
        self.hold_last = hold_last
        self.available = len(list((self.root / shot).glob(f"{shot}.*.png"))) if (self.root / shot).exists() else 0

    def path(self, k: int) -> Path:
        return self.root / self.shot / f"{self.shot}.{k:06d}.png"

    def frame(self, i: int) -> np.ndarray:
        k = self.offset + i
        p = self.path(k)
        if not p.exists() and self.hold_last and self.available:
            p = self.path(min(k, self.available - 1))
        img = cv2.imread(str(p)) if p.exists() else None
        if img is None:
            img = ty.card([(f"{self.shot}", 64, "display"), (self.label or "story shot", 34, "regular"),
                           ("not rendered yet", 26, "light")], bg=(38, 34, 30))
            cv2.rectangle(img, (0, H - 12), (int(W * (i + 1) / max(1, self.n)), H), ty.ACCENT, -1)
            return img
        if img.shape[1] != W or img.shape[0] != H:
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_CUBIC)
        return img


class Fn:
    def __init__(self, n: int, fn: Callable[[int], np.ndarray]):
        self.n, self.fn = n, fn

    def frame(self, i: int) -> np.ndarray:
        return self.fn(i)


class Concat:
    """Segments one after another; ``dissolve`` frames of overlap between consecutive parts."""

    def __init__(self, parts: Sequence[Any], dissolve: int = 0):
        self.parts, self.dissolve = list(parts), dissolve
        self.n = sum(p.n for p in self.parts) - dissolve * (len(self.parts) - 1)

    def frame(self, i: int) -> np.ndarray:
        start = 0
        for k, p in enumerate(self.parts):
            end = start + p.n
            if i < end or k == len(self.parts) - 1:
                j = i - start
                img = p.frame(min(max(0, j), p.n - 1))
                if self.dissolve and k + 1 < len(self.parts) and j >= p.n - self.dissolve:
                    nxt = self.parts[k + 1].frame(j - (p.n - self.dissolve))
                    img = ty.blend(img, nxt, (j - (p.n - self.dissolve) + 1) / (self.dissolve + 1))
                return img
            start = end - self.dissolve
        raise IndexError(i)


class PIP:
    """``inset`` (a segment, or a callable i -> image) drawn into ``rect`` over ``base`` from frame
    ``start`` to ``end``, sliding/fading in."""

    def __init__(self, base: Any, inset: Any, rect: Tuple[int, int, int, int], start: int = 0, end: Optional[int] = None,
                 title: str = ""):
        self.base, self.inset, self.rect, self.start = base, inset, rect, start
        self.end = end if end is not None else base.n
        self.n, self.title = base.n, title

    def frame(self, i: int) -> np.ndarray:
        img = self.base.frame(i)
        if not (self.start <= i < self.end):
            return img
        j = i - self.start
        a = ty.fade(j, self.end - self.start, 8, 8)
        x0, y0, x1, y1 = self.rect
        ins = self.inset.frame(j) if hasattr(self.inset, "frame") else self.inset(j)
        ins = cv2.resize(ins, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)
        ty.panel(img, x0 - 8, y0 - (48 if self.title else 8), x1 + 8, y1 + 8, alpha=0.55 * a)
        roi = img[y0:y1, x0:x1]
        img[y0:y1, x0:x1] = cv2.addWeighted(ins, a, roi, 1 - a, 0)
        if self.title:
            ty.text(img, (x0 + 4, y0 - 42), self.title, 24, "semibold", alpha=a)
        return img


class Over:
    """Draw on top of a segment: ``draw(img, i)`` in place."""

    def __init__(self, base: Any, draw: Callable[[np.ndarray, int], None]):
        self.base, self.draw, self.n = base, draw, base.n

    def frame(self, i: int) -> np.ndarray:
        img = self.base.frame(i).copy()
        self.draw(img, i)
        return img


@dataclass
class Caption:
    title: str
    sub: str = ""
    start: int = 12
    end: Optional[int] = None   # default: segment end - 6


@dataclass
class Item:
    id: str
    seg: Any
    captions: List[Caption] = field(default_factory=list)
    transition: str = "cut"     # cut | dissolve:<k> | fade:<k> (from black) | whip:<k>
    grade: bool = True

    @property
    def n(self) -> int:
        return self.seg.n


def whip(img: np.ndarray, strength: float) -> np.ndarray:
    k = max(1, int(strength * 120)) | 1
    if k <= 1:
        return img
    kernel = np.zeros((1, k), np.float32)
    kernel[0, :] = 1.0 / k
    return cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REFLECT)


def render(items: Sequence[Item], out, on_progress: Optional[Callable[[int, int], None]] = None,
           frames: Optional[range] = None) -> int:
    """Write the edit to ``out`` (an encode.VideoOut). Returns the number of frames written."""
    total = sum(it.n for it in items)
    want = frames or range(total)
    g = written = 0
    prev_last: Optional[np.ndarray] = None
    for it in items:
        kind, _, k = it.transition.partition(":")
        k = int(k or 0)
        for i in range(it.n):
            if g not in want:
                g += 1
                if i == it.n - 1:
                    prev_last = None
                continue
            img = it.seg.frame(i).copy()
            if kind == "dissolve" and prev_last is not None and i < k:
                img = ty.blend(prev_last, img, (i + 1) / (k + 1))
            elif kind == "fade" and i < k:
                img = (img.astype(np.float32) * ((i + 1) / (k + 1))).astype(np.uint8)
            elif kind == "whip" and i < k:
                img = whip(img, 1.0 - i / k)
            for c in it.captions:
                end = c.end if c.end is not None else it.n - 6
                if c.start <= i < end:
                    ty.lower_third(img, c.title, c.sub, alpha=ty.fade(i - c.start, end - c.start, 10, 10))
            if it.grade:
                img = ty.vignette(img, 0.22)
            out.write(img)
            written += 1
            if i == it.n - 1:
                prev_last = img
            g += 1
            if on_progress and g % 30 == 0:
                on_progress(g, total)
    return written
