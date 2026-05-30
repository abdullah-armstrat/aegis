"""Tests for the BLIP scene-caption extractor.

The empty/unreadable path needs no model and always runs. The real captioning path loads
BLIP (~1GB) and is marked ``slow`` so the fast suite stays quick (`-m "not slow"`).
"""

import pytest

from app.extractors.captioner import describe_scene
from app.models import FlagStatus


def test_unreadable_bytes_are_not_assessed():
    """Garbage input must not masquerade as a caption — it is NOT_ASSESSED (ADR-009).
    Fails at the image-decode step, so no model is needed."""
    result = describe_scene(b"this is not an image")
    assert result.status == FlagStatus.NOT_ASSESSED
    assert result.detail
    assert result.scene_descriptions == []


@pytest.mark.slow
def test_captions_a_real_image():
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    from io import BytesIO

    from PIL import Image, ImageDraw

    img = Image.new("RGB", (384, 256), "skyblue")
    d = ImageDraw.Draw(img)
    d.rectangle([0, 180, 384, 256], fill="green")
    d.ellipse([300, 20, 360, 80], fill="yellow")
    buf = BytesIO()
    img.save(buf, format="PNG")

    result = describe_scene(buf.getvalue())
    if result.status == FlagStatus.NOT_ASSESSED:
        pytest.skip(f"captioner model unavailable: {result.detail}")
    assert result.status == FlagStatus.FIRED
    assert result.scene_descriptions
    assert result.scene_descriptions[0].text.strip()  # a non-empty caption
