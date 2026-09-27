"""OCR extractor — on-screen text from an image (the bundle's ``on_screen_text``).

Uses **pytesseract**, a thin wrapper over the Tesseract binary, deliberately chosen over
easyocr so the first extractor carries no torch dependency. An
easyocr-vs-pytesseract accuracy comparison is a planned later model-trial.

The extractor returns an :class:`OcrResult` whose ``status`` keeps three outcomes apart: it
distinguishes text that was read, an image legibly read but containing no text (``CLEAR``),
and an image that could not be processed at all (``NOT_ASSESSED``). Downstream fusion uses
that status so "no on-screen text" is never silently treated as "checked and consistent".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO

from app.config import get_settings
from app.models import FlagStatus


@dataclass
class OcrResult:
    """Outcome of an OCR attempt.

    ``lines`` is the extracted on-screen text (one entry per non-empty line). ``status`` is
    the honest outcome of the attempt:
      * ``FIRED``        — text was found and extracted.
      * ``CLEAR``        — the image was read but contained no legible text.
      * ``NOT_ASSESSED`` — OCR could not run (unreadable bytes, engine error).
    ``detail`` is a short human-readable note, surfaced when status is not FIRED.
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
    """Run OCR over raw image bytes and return an :class:`OcrResult`.

    Never raises for bad input or engine failure: those become a ``NOT_ASSESSED`` result so
    the caller can record honestly that the check could not be performed.
    """
    # Import lazily so the module (and tests that monkeypatch it) load without the heavy
    # imports, and so an OCR-less environment can still import the rest of the app.
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
    """The share of the image's area covered by the boxes of the words Tesseract reads in it.

    Every word with non-empty text and a confidence of 0 or more counts, and overlapping boxes are
    counted once. None when the image or the engine could not be read.
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
