"""OCR extractor — on-screen text from an image (SSOT §3.2 ``on_screen_text``).

Uses **pytesseract**, a thin wrapper over the Tesseract binary, deliberately chosen over
easyocr so the first extractor carries no torch dependency (DEVLOG 2026-05-31). An
easyocr-vs-pytesseract accuracy comparison is a planned later model-trial.

The extractor returns an :class:`OcrResult` whose ``status`` honours ADR-009: it
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
    the caller can record honestly that the check could not be performed (ADR-009).
    """
    # Import lazily so the module (and tests that monkeypatch it) load without the heavy
    # imports, and so an OCR-less environment can still import the rest of the app.
    try:
        import pytesseract
        from PIL import Image, UnidentifiedImageError
    except ImportError as exc:  # pragma: no cover - environment guard
        return OcrResult(status=FlagStatus.NOT_ASSESSED, detail=f"OCR deps missing: {exc}")

    try:
        image = Image.open(BytesIO(image_bytes))
        image.load()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        return OcrResult(
            status=FlagStatus.NOT_ASSESSED, detail=f"Image could not be read: {exc}"
        )

    try:
        _configure_tesseract()
        raw = pytesseract.image_to_string(image)
    except Exception as exc:  # noqa: BLE001 - any engine failure is "could not assess"
        return OcrResult(status=FlagStatus.NOT_ASSESSED, detail=f"OCR engine error: {exc}")

    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if not lines:
        return OcrResult(status=FlagStatus.CLEAR, detail="No legible on-screen text found.")
    return OcrResult(lines=lines, status=FlagStatus.FIRED)
