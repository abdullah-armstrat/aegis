"""Tests for the OCR extractor.

Test images are generated at runtime with PIL (no binary fixtures committed) and run
through the real Tesseract engine, so these double as a smoke test that Tesseract is wired
up. The key assertions cover all three honest outcomes (ADR-009): text found (FIRED), image
read but empty (CLEAR), and unreadable input (NOT_ASSESSED).
"""

from io import BytesIO

import pytest

from app.models import FlagStatus

pytesseract = pytest.importorskip("pytesseract")
PIL = pytest.importorskip("PIL")

from PIL import Image, ImageDraw  # noqa: E402

from app.extractors.ocr import extract_on_screen_text  # noqa: E402


def _png_bytes(image: Image.Image) -> bytes:
    buf = BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _text_image(text: str, size=(420, 120)) -> bytes:
    img = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(img)
    # Render large so the default bitmap font is legible to Tesseract.
    draw.text((10, 40), text, fill="black")
    return _png_bytes(img)


def _tesseract_available() -> bool:
    try:
        from app.extractors.ocr import _configure_tesseract

        _configure_tesseract()
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


requires_tesseract = pytest.mark.skipif(
    not _tesseract_available(), reason="Tesseract binary not available"
)


@requires_tesseract
def test_reads_on_screen_text():
    result = extract_on_screen_text(_text_image("BREAKING NEWS"))
    assert result.status == FlagStatus.FIRED
    joined = " ".join(result.lines).upper()
    # OCR is imperfect; assert on a robust substring rather than an exact match.
    assert "BREAKING" in joined or "NEWS" in joined


@requires_tesseract
def test_blank_image_is_clear_not_fired():
    """An image legibly read but containing no text is CLEAR — explicitly not FIRED and
    explicitly not NOT_ASSESSED (the load-bearing distinction, ADR-009)."""
    blank = Image.new("RGB", (200, 80), "white")
    result = extract_on_screen_text(_png_bytes(blank))
    assert result.status == FlagStatus.CLEAR
    assert result.lines == []


def test_unreadable_bytes_are_not_assessed():
    """Garbage input must not masquerade as 'clear' — it is NOT_ASSESSED (ADR-009).
    Needs no Tesseract: it fails at the image-decode step."""
    result = extract_on_screen_text(b"this is not an image")
    assert result.status == FlagStatus.NOT_ASSESSED
    assert result.detail
