"""Film typography and small drawing helpers on BGR numpy frames (PIL for text, OpenCV for shapes).

Fonts: Segoe UI / Bahnschrift on Windows; Pillow's own scalable font elsewhere.
"""
from __future__ import annotations

import functools
import math
from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 1920, 1080
FONT_DIRS = [Path("C:/Windows/Fonts"), Path("/usr/share/fonts/truetype/dejavu")]
FONTS = {
    "regular": ["segoeui.ttf", "DejaVuSans.ttf"],
    "light": ["segoeuil.ttf", "segoeui.ttf", "DejaVuSans.ttf"],
    "semibold": ["seguisb.ttf", "segoeuib.ttf", "DejaVuSans-Bold.ttf"],
    "bold": ["segoeuib.ttf", "DejaVuSans-Bold.ttf"],
    "display": ["bahnschrift.ttf", "segoeuib.ttf", "DejaVuSans-Bold.ttf"],
    "mono": ["consola.ttf", "DejaVuSansMono.ttf"],
}
ACCENT = (32, 150, 255)       # BGR safety orange
WHITE = (245, 245, 245)
INK = (18, 18, 20)
DRONE_COLORS = [(255, 191, 64), (90, 220, 120), (60, 170, 255), (220, 120, 240)]  # BGR per drone


@functools.lru_cache(maxsize=64)
def font(kind: str = "regular", size: int = 36) -> ImageFont.FreeTypeFont:
    for name in FONTS.get(kind, FONTS["regular"]):
        for d in FONT_DIRS:
            p = d / name
            if p.exists():
                return ImageFont.truetype(str(p), size)
    return ImageFont.load_default(size=size)


def ease(x: float) -> float:
    """Smoothstep 0..1."""
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def fade(i: int, n: int, fin: int = 10, fout: int = 10) -> float:
    """Opacity of an element shown over frames [0, n): fades in over fin, out over fout frames."""
    a = 1.0
    if fin > 0:
        a = min(a, ease(i / fin))
    if fout > 0:
        a = min(a, ease((n - 1 - i) / fout))
    return a


def text(img: np.ndarray, xy: Tuple[float, float], s: str, size: int = 36, kind: str = "regular",
         color=WHITE, alpha: float = 1.0, anchor: str = "la", shadow: bool = True, spacing: int = 6) -> Tuple[int, int, int, int]:
    """Draw text with PIL (antialiased) onto a BGR frame in place. Returns its bbox (x0, y0, x1, y1)."""
    if alpha <= 0.0 or not s:
        return (0, 0, 0, 0)
    f = font(kind, size)
    probe = ImageDraw.Draw(Image.new("L", (1, 1)))
    x0, y0, x1, y1 = probe.multiline_textbbox(xy, s, font=f, anchor=anchor, spacing=spacing)
    pad = 12
    bx0, by0 = max(0, int(x0) - pad), max(0, int(y0) - pad)
    bx1, by1 = min(img.shape[1], int(x1) + pad), min(img.shape[0], int(y1) + pad)
    if bx1 <= bx0 or by1 <= by0:
        return (0, 0, 0, 0)
    mask = Image.new("L", (bx1 - bx0, by1 - by0), 0)
    d = ImageDraw.Draw(mask)
    d.multiline_text((xy[0] - bx0, xy[1] - by0), s, font=f, fill=255, anchor=anchor, spacing=spacing)
    m = np.asarray(mask, np.float32) / 255.0
    roi = img[by0:by1, bx0:bx1].astype(np.float32)
    if shadow:
        sh = cv2.GaussianBlur(m, (0, 0), max(1.0, size / 14.0))
        sh = np.roll(np.roll(sh, 2, 0), 2, 1) * 0.65 * alpha
        roi *= (1.0 - sh[..., None])
    a = (m * alpha)[..., None]
    roi = roi * (1.0 - a) + np.asarray(color, np.float32)[None, None, :] * a
    img[by0:by1, bx0:bx1] = np.clip(roi, 0, 255).astype(np.uint8)
    return (int(x0), int(y0), int(x1), int(y1))


def text_size(s: str, size: int, kind: str = "regular") -> Tuple[int, int]:
    probe = ImageDraw.Draw(Image.new("L", (1, 1)))
    x0, y0, x1, y1 = probe.multiline_textbbox((0, 0), s, font=font(kind, size), spacing=6)
    return int(x1 - x0), int(y1 - y0)


def panel(img: np.ndarray, x0: int, y0: int, x1: int, y1: int, color=INK, alpha: float = 0.6, radius: int = 14) -> None:
    """A translucent rounded rectangle."""
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(img.shape[1], x1), min(img.shape[0], y1)
    if x1 <= x0 or y1 <= y0 or alpha <= 0:
        return
    mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
    r = min(radius, (x1 - x0) // 2, (y1 - y0) // 2)
    cv2.rectangle(mask, (r, 0), (x1 - x0 - r, y1 - y0), 255, -1)
    cv2.rectangle(mask, (0, r), (x1 - x0, y1 - y0 - r), 255, -1)
    for cx, cy in ((r, r), (x1 - x0 - r - 1, r), (r, y1 - y0 - r - 1), (x1 - x0 - r - 1, y1 - y0 - r - 1)):
        cv2.circle(mask, (cx, cy), r, 255, -1, cv2.LINE_AA)
    a = (mask.astype(np.float32) / 255.0 * alpha)[..., None]
    roi = img[y0:y1, x0:x1].astype(np.float32)
    img[y0:y1, x0:x1] = (roi * (1 - a) + np.asarray(color, np.float32) * a).astype(np.uint8)


def lower_third(img: np.ndarray, title: str, sub: str = "", alpha: float = 1.0, x: int = 96, y: int = 900) -> None:
    """Caption: an accent bar and one or two lines, bottom left."""
    if alpha <= 0 or not title:
        return
    tw, th = text_size(title, 46, "semibold")
    sw, sh = text_size(sub, 30, "regular") if sub else (0, 0)
    w = max(tw, sw) + 64
    h = th + (sh + 14 if sub else 0) + 40
    panel(img, x - 24, y - 20, x - 24 + w, y - 20 + h, alpha=0.45 * alpha)
    cv2.rectangle(img, (x - 24, y - 20), (x - 18, y - 20 + h), ACCENT, -1)
    text(img, (x, y), title, 46, "semibold", alpha=alpha)
    if sub:
        text(img, (x, y + th + 14), sub, 30, "regular", color=(215, 215, 215), alpha=alpha)


def label_box(img: np.ndarray, box: Sequence[float], label: str = "", color=ACCENT, thick: int = 3, alpha: float = 1.0,
              corner: bool = True) -> None:
    """A detection box (corner brackets) with a label tag."""
    if alpha <= 0:
        return
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    over = img.copy()
    if corner:
        L = max(8, int(0.28 * min(x1 - x0, y1 - y0)))
        for (cx, cy, dx, dy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
            cv2.line(over, (cx, cy), (cx + dx * L, cy), color, thick, cv2.LINE_AA)
            cv2.line(over, (cx, cy), (cx, cy + dy * L), color, thick, cv2.LINE_AA)
    else:
        cv2.rectangle(over, (x0, y0), (x1, y1), color, thick, cv2.LINE_AA)
    cv2.addWeighted(over, alpha, img, 1 - alpha, 0, img)
    if label:
        tw, th = text_size(label, 24, "semibold")
        panel(img, x0, y0 - th - 18, x0 + tw + 16, y0 - 4, color=color, alpha=0.85 * alpha, radius=6)
        text(img, (x0 + 8, y0 - th - 13), label, 24, "semibold", color=INK, alpha=alpha, shadow=False)


def cover(img: np.ndarray, w: int = W, h: int = H, center: Optional[Tuple[float, float]] = None, zoom: float = 1.0) -> np.ndarray:
    """Scale/crop an image to fill w x h, optionally zoomed in around a centre (pixel coords of img)."""
    ih, iw = img.shape[:2]
    s = max(w / iw, h / ih) * zoom
    cx, cy = center if center is not None else (iw / 2, ih / 2)
    cw, ch = w / s, h / s
    x0 = min(max(0.0, cx - cw / 2), iw - cw)
    y0 = min(max(0.0, cy - ch / 2), ih - ch)
    M = np.array([[s, 0, -x0 * s], [0, s, -y0 * s]], np.float32)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA, borderMode=cv2.BORDER_REFLECT)


def cover_map(iw: int, ih: int, w: int, h: int, center: Optional[Tuple[float, float]] = None, zoom: float = 1.0):
    """The pixel mapping ``cover`` applies: image (u, v) -> output (x, y)."""
    s = max(w / iw, h / ih) * zoom
    cx, cy = center if center is not None else (iw / 2, ih / 2)
    cw, ch = w / s, h / s
    x0 = min(max(0.0, cx - cw / 2), iw - cw)
    y0 = min(max(0.0, cy - ch / 2), ih - ch)
    return lambda u, v: ((u - x0) * s, (v - y0) * s)


def vignette(img: np.ndarray, strength: float = 0.28) -> np.ndarray:
    h, w = img.shape[:2]
    key = (h, w, round(strength, 3))
    m = _VIG.get(key)
    if m is None:
        y, x = np.ogrid[:h, :w]
        r = np.sqrt(((x - w / 2) / (w / 2)) ** 2 + ((y - h / 2) / (h / 2)) ** 2) / math.sqrt(2)
        m = (1.0 - strength * np.clip(r, 0, 1) ** 2.2).astype(np.float32)[..., None]
        _VIG[key] = m
    return np.clip(img.astype(np.float32) * m, 0, 255).astype(np.uint8)


_VIG: dict = {}


def blend(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    t = min(1.0, max(0.0, t))
    return cv2.addWeighted(a, 1.0 - t, b, t, 0)


def card(lines: Sequence[Tuple[str, int, str]], bg=(14, 14, 16), w: int = W, h: int = H) -> np.ndarray:
    """A centred title card: lines of (text, size, font kind)."""
    img = np.full((h, w, 3), bg, np.uint8)
    total = sum(text_size(s, sz, k)[1] + 18 for s, sz, k in lines)
    y = (h - total) // 2
    for s, sz, k in lines:
        text(img, (w // 2, y), s, sz, k, anchor="ma", shadow=False)
        y += text_size(s, sz, k)[1] + 18
    return img
