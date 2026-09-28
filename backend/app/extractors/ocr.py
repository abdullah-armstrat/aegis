"""OCR extractor: reads on-screen text from an image (the bundle's ``on_screen_text``).

Uses pytesseract, a wrapper around the Tesseract binary, picked over easyocr because it needs
no torch. The status separates "text found", "read but no text" (CLEAR) and "could not run"
(NOT_ASSESSED), so missing text is never treated as a passed check.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO

from app.config import get_settings
from app.models import FlagStatus


@dataclass
class OcrResult:
    """Result of an OCR attempt.

    ``lines`` holds one entry per non-empty line of text. ``status`` is FIRED when text was
    found, CLEAR when the image had no legible text and NOT_ASSESSED when OCR could not run.
    ``detail`` is a short note shown when the status is not FIRED.
    """

    lines: list[str] = field(default_factory=list)
    status: FlagStatus = FlagStatus.CLEAR
    detail: str = ""


def _configure_tesseract() -> None:
    """Point pytesseract at the configured Tesseract binary (it is not on PATH here)."""
    import pytesseract

    cmd = get_settings().tesseract_cmd
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd


def extract_on_screen_text(image_bytes: bytes) -> OcrResult:
    """Run OCR on raw image bytes and return an :class:`OcrResult`.

    Never raises: bad input or an engine failure gives a NOT_ASSESSED result.
    """
    # lazy imports, so the rest of the app still imports on a machine without OCR installed
    try:
        import pytesseract
        from PIL import Image, UnidentifiedImageError
    except ImportError as exc:  # pragma: no cover - environment guard
        return OcrResult(status=FlagStatus.NOT_ASSESSED, detail="The text in the picture could not be read.")

    try:
        image = Image.open(BytesIO(image_bytes))
        image.load()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        return OcrResult(status=FlagStatus.NOT_ASSESSED, detail="The picture could not be read.")

    try:
        _configure_tesseract()
        raw = pytesseract.image_to_string(image)
    except Exception:  # noqa: BLE001 - any engine failure is "could not assess"
        return OcrResult(status=FlagStatus.NOT_ASSESSED, detail="The text in the picture could not be read.")

    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if not lines:
        return OcrResult(status=FlagStatus.CLEAR, detail="No legible on-screen text found.")
    return OcrResult(lines=lines, status=FlagStatus.FIRED)


def text_box_share(image_bytes: bytes) -> float | None:
    """Share (0 to 1) of the image covered by the boxes of the words Tesseract reads.

    Words with text and a confidence of 0 or more count, and overlapping boxes are counted once.
    Returns None if the image or the engine could not be read.
    """
    try:
        import numpy as np
        import pytesseract
        from PIL import Image

        image = Image.open(BytesIO(image_bytes))
        image.load()
        _configure_tesseract()
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    except Exception:  # noqa: BLE001 - no measurement rather than a crash
        return None
    covered = np.zeros((image.height, image.width), dtype=bool)
    for text, conf, left, top, width, height in zip(
        data["text"], data["conf"], data["left"], data["top"], data["width"], data["height"]
    ):
        if str(text).strip() and float(conf) >= 0:
            covered[top:top + height, left:left + width] = True
    return float(covered.mean()) if covered.size else None
