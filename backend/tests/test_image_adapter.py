"""Tests for the image adapter: image + caption -> EvidenceBundle.

Verifies the adapter carries the caption through, populates on-screen text from OCR, and
records the OCR status in ``extractor_status`` so fusion can honour ADR-009.
"""

from io import BytesIO

import pytest

pytest.importorskip("PIL")

from PIL import Image  # noqa: E402

from app.adapters.image_adapter import build_bundle  # noqa: E402
from app.models import EvidenceBundle, FlagStatus, Modality  # noqa: E402


def _png_bytes(image: Image.Image) -> bytes:
    buf = BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def test_adapter_carries_caption_and_records_ocr_status():
    blank = _png_bytes(Image.new("RGB", (120, 60), "white"))
    bundle = build_bundle(blank, caption="A calm day", source_ref="t.png")

    assert isinstance(bundle, EvidenceBundle)
    assert bundle.caption == "A calm day"
    assert bundle.meta.modality == Modality.IMAGE
    assert bundle.meta.source_ref == "t.png"
    # OCR ran; status is recorded so an empty on_screen_text is not read as "consistent".
    assert "ocr" in bundle.extractor_status
    assert bundle.extractor_status["ocr"] in {FlagStatus.CLEAR, FlagStatus.FIRED}


def test_adapter_marks_ocr_not_assessed_on_bad_image():
    bundle = build_bundle(b"not an image", caption=None, source_ref="bad")
    assert bundle.extractor_status["ocr"] == FlagStatus.NOT_ASSESSED
    assert bundle.on_screen_text == []
