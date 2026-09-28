"""Image edits used to test how well the perceptual hash survives re-posting.

Each one mimics something that happens when a picture is saved and re-posted (JPEG
re-encoding, resizing, cropping, borders, screenshots, flips, small rotations). All are
deterministic PIL functions, so runs are repeatable.
"""

from __future__ import annotations

from collections.abc import Callable
from io import BytesIO

from PIL import Image, ImageDraw, ImageOps


def _rgb(img: Image.Image) -> Image.Image:
    return img.convert("RGB")


def jpeg(quality: int) -> Callable[[Image.Image], Image.Image]:
    def _f(img: Image.Image) -> Image.Image:
        buf = BytesIO()
        _rgb(img).save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        out = Image.open(buf)
        out.load()
        return out

    return _f


def resize(percent: int) -> Callable[[Image.Image], Image.Image]:
    def _f(img: Image.Image) -> Image.Image:
        w, h = img.size
        size = (max(1, round(w * percent / 100)), max(1, round(h * percent / 100)))
        return _rgb(img).resize(size, Image.Resampling.LANCZOS)

    return _f


def crop(percent: int) -> Callable[[Image.Image], Image.Image]:
    # Cuts percent% off each dimension, half from each edge, so crop(20) keeps the central 80% x 80%.
    def _f(img: Image.Image) -> Image.Image:
        w, h = img.size
        dx, dy = round(w * percent / 200), round(h * percent / 200)
        return _rgb(img).crop((dx, dy, w - dx, h - dy))

    return _f


def border(img: Image.Image) -> Image.Image:
    # White border 10% of the shorter side on all four edges.
    pad = round(min(img.size) * 0.10)
    return ImageOps.expand(_rgb(img), border=pad, fill="white")


def screenshot(img: Image.Image) -> Image.Image:
    """Put the image inside a phone-style frame: status bar, header with avatar, footer icons."""
    src = _rgb(img)
    w, h = src.size
    inner_w = round(w * 0.90)
    inner = src.resize((inner_w, max(1, round(h * inner_w / w))), Image.Resampling.LANCZOS)
    status, header, footer = round(h * 0.06), round(h * 0.16), round(h * 0.14)
    canvas = Image.new("RGB", (w, status + header + inner.height + footer), "white")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, w, status], fill=(24, 24, 24))  # status bar
    r = round(header * 0.28)
    cx, cy = round(w * 0.05) + r, status + header // 2
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(170, 170, 170))  # avatar
    x0 = cx + r + round(w * 0.03)
    draw.rectangle([x0, cy - r, x0 + round(w * 0.35), cy - r // 3], fill=(60, 60, 60))  # name
    draw.rectangle([x0, cy + r // 3, x0 + round(w * 0.22), cy + r], fill=(150, 150, 150))
    canvas.paste(inner, ((w - inner_w) // 2, status + header))
    fy = status + header + inner.height + footer // 2
    for i in range(4):  # like / comment / share / save
        fx = round(w * (0.08 + 0.12 * i))
        draw.rectangle([fx, fy - r // 2, fx + r, fy + r // 2], fill=(90, 90, 90))
    return canvas


def flip(img: Image.Image) -> Image.Image:
    return ImageOps.mirror(_rgb(img))


def rotate(degrees: float) -> Callable[[Image.Image], Image.Image]:
    # Anticlockwise about the centre, same canvas size, corners filled white.
    def _f(img: Image.Image) -> Image.Image:
        return _rgb(img).rotate(
            degrees, resample=Image.Resampling.BICUBIC, expand=False, fillcolor="white"
        )

    return _f


# The names are used as column labels in the results tables.
TRANSFORMS: dict[str, Callable[[Image.Image], Image.Image]] = {
    "jpeg_q90": jpeg(90),
    "jpeg_q70": jpeg(70),
    "jpeg_q50": jpeg(50),
    "jpeg_q30": jpeg(30),
    "resize_75": resize(75),
    "resize_50": resize(50),
    "resize_25": resize(25),
    "crop_05": crop(5),
    "crop_10": crop(10),
    "crop_20": crop(20),
    "border": border,
    "screenshot": screenshot,
    "flip": flip,
    "rotate_2": rotate(2),
    "rotate_5": rotate(5),
}
