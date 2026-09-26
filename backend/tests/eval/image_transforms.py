"""The re-post transformations the hash robustness harness applies (FINAL_PLAN WP-1 step 4).

Each models something that happens to an image when it is saved, shared and re-posted. Every
transform is deterministic, so a harness run is exactly reproducible. Definitions are stated
here once, in the words the report uses:

  jpeg_qN        re-encode as JPEG at quality N (90, 70, 50, 30) and decode again
  resize_N       scale both sides to N% (75, 50, 25) with Lanczos resampling
  crop_N         cut N% (5, 10, 20) off each dimension, split evenly between the two edges,
                 keeping the centre — so crop_20 keeps the central 80% x 80% (64% of the area)
  border         a uniform white border 10% of the shorter side, on all four edges
  screenshot     the image scaled to 90% width inside a phone-style frame: dark status bar,
                 white header with an avatar and name lines, white footer with action icons
  flip           horizontal mirror
  rotate_N       rotate N degrees anticlockwise (2, 5) about the centre, same canvas size,
                 bicubic, exposed corners filled white

Pure image-in, image-out functions over PIL images. No I/O.
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
    def _f(img: Image.Image) -> Image.Image:
        w, h = img.size
        dx, dy = round(w * percent / 200), round(h * percent / 200)
        return _rgb(img).crop((dx, dy, w - dx, h - dy))

    return _f


def border(img: Image.Image) -> Image.Image:
    pad = round(min(img.size) * 0.10)
    return ImageOps.expand(_rgb(img), border=pad, fill="white")


def screenshot(img: Image.Image) -> Image.Image:
    """Frame the image the way a phone screenshot of a social post frames it."""
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
    def _f(img: Image.Image) -> Image.Image:
        return _rgb(img).rotate(
            degrees, resample=Image.Resampling.BICUBIC, expand=False, fillcolor="white"
        )

    return _f


# Ordered as the plan lists them; the names are the column labels in every results table.
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
