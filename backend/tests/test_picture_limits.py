"""The picture check's limits (ADR-041): a nearly blank picture, or one that is mostly text, gives
CLIP no scene to compare with the caption, so the check is not assessed and says why."""

from io import BytesIO

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from app.adapters.image_adapter import measure_caption_fit
from app.config import get_settings
from app.extractors import caption_match
from app.extractors.caption_match import MOSTLY_TEXT_SHARE, NEARLY_BLANK_STD, picture_limit, picture_measures
from app.fusion.rules import caption_scene_mismatch_rule
from app.models import EvidenceBundle, FlagStatus, Meta, Modality


def _png(img: Image.Image) -> bytes:
    buf = BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def _grey_noise(std: float, seed: int = 1) -> bytes:
    rng = np.random.default_rng(seed)
    pixels = np.clip(rng.normal(128, std, (120, 160)), 0, 255).astype(np.uint8)
    return _png(Image.fromarray(pixels, "L").convert("RGB"))


def _text_page() -> bytes:
    """A screenshot of a text post: large black words filling a white picture."""
    img = Image.new("RGB", (900, 600), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=90)
    for i, line in enumerate(["BREAKING NEWS TODAY", "SHARE THIS POST NOW", "BEFORE THEY DELETE",
                              "EVERYONE MUST READ", "WHAT THEY HIDE", "FROM ALL OF YOU"]):
        draw.text((10, 8 + i * 98), line, fill="black", font=font)
    return _png(img)


@pytest.fixture
def no_clip(monkeypatch):
    """CLIP must not be loaded for a picture it cannot compare."""
    def refuse(*args, **kwargs):
        raise AssertionError("CLIP was run")

    monkeypatch.setattr(caption_match, "clip_image_vector", refuse)
    monkeypatch.setattr(caption_match, "clip_text_vector", refuse)


def test_nearly_blank_is_a_greyscale_spread_under_10():
    assert picture_measures(_grey_noise(4.0), [])[0] < NEARLY_BLANK_STD
    assert "nearly blank" in picture_limit(_grey_noise(4.0), [])
    assert picture_measures(_grey_noise(30.0), [])[0] > NEARLY_BLANK_STD
    assert picture_limit(_grey_noise(30.0), []) is None
    assert picture_measures(_png(Image.new("RGB", (64, 32), "white")), []) == (0.0, 0.0)


def test_text_is_measured_only_when_ocr_read_some(monkeypatch):
    from app.extractors import ocr

    calls = []
    monkeypatch.setattr(ocr, "text_box_share", lambda b: calls.append(1) or 0.9)
    assert picture_measures(_grey_noise(30.0), [])[1] == 0.0 and calls == []
    assert picture_measures(_grey_noise(30.0), ["SOME TEXT"])[1] == 0.9 and calls == [1]
    assert "mostly text (the words read in it cover 90% of it" in picture_limit(_grey_noise(30.0), ["SOME TEXT"])
    monkeypatch.setattr(ocr, "text_box_share", lambda b: MOSTLY_TEXT_SHARE)
    assert picture_limit(_grey_noise(30.0), ["SOME TEXT"]) is None  # 40% exactly is not "more than 40%"


def test_a_screenshot_of_a_text_post_is_mostly_text():
    from app.extractors.ocr import extract_on_screen_text, text_box_share

    page = _text_page()
    assert extract_on_screen_text(page).lines
    assert text_box_share(page) > MOSTLY_TEXT_SHARE
    assert "mostly text" in picture_limit(page, extract_on_screen_text(page).lines)


@pytest.mark.parametrize("method", ["image", "meaning"])
def test_the_picture_check_is_not_assessed_and_says_why_without_running_clip(no_clip, monkeypatch, method):
    monkeypatch.setenv("AEGIS_CAPTION_MATCH_METHOD", method)
    get_settings.cache_clear()
    match, status = measure_caption_fit(_png(Image.new("RGB", (64, 32), "white")), "A flooded street", [], [], method)
    assert status == FlagStatus.NOT_ASSESSED
    flag = caption_scene_mismatch_rule(EvidenceBundle(caption="A flooded street", caption_match=match,
                                                      meta=Meta(modality=Modality.IMAGE)))
    assert flag.status == FlagStatus.NOT_ASSESSED
    assert flag.evidence.startswith("Cannot compare: The picture is nearly blank")
    get_settings.cache_clear()


def test_without_a_caption_the_reason_stays_that_there_is_no_caption(no_clip):
    match, status = measure_caption_fit(_png(Image.new("RGB", (64, 32), "white")), "", [], [], "image")
    assert status == FlagStatus.NOT_ASSESSED and "no caption" in match.detail
