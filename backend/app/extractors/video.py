"""Video handling: probe a file, pick keyframes at scene cuts, grab frames, pull out the audio.

Keyframes are the middle frame of each scene PySceneDetect finds (its content detector, default
settings). A video with more than ``KEYFRAME_CAP`` scenes keeps its longest scenes, since they
make up most of what a viewer sees. A video with no cuts is sampled at even intervals: one frame
per ``EVEN_INTERVAL_S`` seconds, at the middle of each interval, also capped.

Everything here shells out to ffmpeg/ffprobe or uses OpenCV through PySceneDetect; no model is
loaded. Failures raise :class:`VideoError` with a message fit to show a user.
"""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path

KEYFRAME_CAP = 8
EVEN_INTERVAL_S = 7.5


class VideoError(ValueError):
    """The file cannot be used as a video; the message says why in plain words."""


@dataclass(frozen=True)
class VideoInfo:
    duration_s: float
    has_audio: bool
    width: int
    height: int


def probe(path: Path) -> VideoInfo:
    """Duration, size and whether there is an audio track. Raises VideoError if unreadable."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type,width,height",
             "-of", "json", str(path)],
            capture_output=True, text=True, timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise VideoError(f"The file could not be read as a video ({type(exc).__name__}).") from exc
    try:
        info = json.loads(out.stdout or "{}")
    except json.JSONDecodeError:
        info = {}
    streams = info.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    try:
        duration = float(info.get("format", {}).get("duration"))
    except (TypeError, ValueError):
        duration = float("nan")
    if out.returncode != 0 or video is None or not math.isfinite(duration) or duration <= 0:
        raise VideoError("The file could not be read as a video.")
    return VideoInfo(duration_s=duration, has_audio=any(s.get("codec_type") == "audio" for s in streams),
                     width=int(video.get("width") or 0), height=int(video.get("height") or 0))


def even_timestamps(duration_s: float, cap: int = KEYFRAME_CAP) -> list[float]:
    """Middle points of equal intervals, about one per EVEN_INTERVAL_S seconds, at most ``cap``."""
    n = min(cap, max(1, math.ceil(duration_s / EVEN_INTERVAL_S)))
    return [round((i + 0.5) * duration_s / n, 3) for i in range(n)]


def keyframes_from_scenes(scenes: list[tuple[float, float]], duration_s: float,
                          cap: int = KEYFRAME_CAP) -> tuple[list[float], str]:
    """Keyframe times from scene boundaries, and how they were chosen ("scene cuts" or "even").

    One scene or none means the detector found no cut, so the video is sampled evenly instead.
    """
    if len(scenes) <= 1:
        return even_timestamps(duration_s, cap), "even"
    longest = sorted(scenes, key=lambda s: (-(s[1] - s[0]), s[0]))[:cap]
    return sorted(round((a + b) / 2, 3) for a, b in longest), "scene cuts"


def sample_keyframes(path: Path, duration_s: float) -> tuple[list[float], str]:
    """Keyframe times for a video file: scene cuts by PySceneDetect, or even intervals."""
    from scenedetect import ContentDetector, detect

    try:
        found = detect(str(path), ContentDetector())
    except Exception as exc:  # noqa: BLE001 - any detector failure falls back to even sampling
        return even_timestamps(duration_s), f"even (scene detection failed: {type(exc).__name__})"
    scenes = [(a.get_seconds(), b.get_seconds()) for a, b in found]
    return keyframes_from_scenes(scenes, duration_s)


def grab_frame(path: Path, t: float) -> bytes:
    """The frame at ``t`` seconds as PNG bytes."""
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{max(0.0, t):.3f}", "-i", str(path), "-frames:v", "1",
         "-f", "image2pipe", "-vcodec", "png", "-"],
        capture_output=True, timeout=60,
    )
    if out.returncode != 0 or not out.stdout:
        raise VideoError(f"The frame at {t:.1f} s could not be read.")
    return out.stdout


def extract_audio(path: Path, wav: Path) -> None:
    """The audio track as 16 kHz mono WAV, the rate the speech recogniser expects."""
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(path), "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", str(wav)],
        capture_output=True, text=True, timeout=120,
    )
    if out.returncode != 0 or not wav.is_file() or wav.stat().st_size <= 44:
        raise VideoError("The audio track could not be read.")


def thumbnail(png: bytes, width: int = 160) -> str:
    """A small JPEG of a frame as a data URL, for the keyframe strip."""
    import base64
    from io import BytesIO

    from PIL import Image

    with Image.open(BytesIO(png)) as img:
        img = img.convert("RGB")
        img.thumbnail((width, width))
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=70)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def clock(seconds: float) -> str:
    """Seconds as m:ss, the way the interface and the flags cite moments."""
    total = int(round(seconds))
    return f"{total // 60}:{total % 60:02d}"
