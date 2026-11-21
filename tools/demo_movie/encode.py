"""Write BGR frames to an MP4.

With ``imageio-ffmpeg`` (``pip install -e ".[demo]"``): H.264 (libx264, CRF 18, yuv420p, faststart)
and, given a music file, AAC audio faded out over the last seconds. Without it: OpenCV's H.264
writer (Windows Media Foundation), silent.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np


def ffmpeg_exe() -> Optional[str]:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        return shutil.which("ffmpeg")


class VideoOut:
    def __init__(self, path: Path, w: int, h: int, fps: int = 30, music: Optional[Path] = None,
                 duration_s: Optional[float] = None, crf: int = 18):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.w, self.h, self.fps = w, h, fps
        self.n = 0
        exe = ffmpeg_exe()
        self.proc = None
        self.cv = None
        if exe:
            cmd = [exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
                   "-r", str(fps), "-i", "-"]
            if music:
                cmd += ["-i", str(music)]
            cmd += ["-c:v", "libx264", "-preset", "slow", "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
            if music:
                fade = ""
                if duration_s:
                    fade = f",afade=t=out:st={max(0.0, duration_s - 2.5):.2f}:d=2.5"
                cmd += ["-c:a", "aac", "-b:a", "192k", "-af", f"afade=t=in:d=0.5{fade}", "-shortest"]
            cmd += [str(self.path)]
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            self.backend = "ffmpeg"
        else:
            import cv2
            self.cv = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"avc1"), fps, (w, h))
            if not self.cv.isOpened():
                self.cv = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            self.backend = "opencv (silent)"
            if music:
                print(f"  note: no ffmpeg (pip install imageio-ffmpeg): {Path(music).name} not mixed in")

    def write(self, frame: np.ndarray) -> None:
        if frame.shape[1] != self.w or frame.shape[0] != self.h:
            import cv2
            frame = cv2.resize(frame, (self.w, self.h), interpolation=cv2.INTER_AREA)
        if self.proc is not None:
            self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        else:
            self.cv.write(frame)
        self.n += 1

    def close(self) -> None:
        if self.proc is not None:
            self.proc.stdin.close()
            if self.proc.wait() != 0:
                raise RuntimeError(f"ffmpeg failed writing {self.path}")
        if self.cv is not None:
            self.cv.release()
