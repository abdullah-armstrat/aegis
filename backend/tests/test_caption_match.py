"""Tests for the caption-vs-picture similarity extractor.

The fast tests pin the honest outcomes without loading a model: missing or altered weights, and
a missing caption or scene text, are NOT_ASSESSED, and nothing is ever downloaded. The slow tests
load CLIP and spaCy for real on an openly licensed image (skimage's ``chelsea``, a cat, CC0).
"""

import urllib.request

import pytest

from app.extractors import caption_match
from app.extractors.caption_match import measure_caption_match, read_from_image
from app.models import FlagStatus


@pytest.fixture
def no_network(monkeypatch):
    def _refuse(*_a, **_k):
        raise AssertionError("the extractor must never download")

    monkeypatch.setattr(urllib.request, "urlopen", _refuse)


@pytest.fixture
def empty_clip_cache(monkeypatch, tmp_path):
    caption_match._clip.cache_clear()
    monkeypatch.setattr(caption_match, "CLIP_CACHE", tmp_path)
    yield tmp_path
    caption_match._clip.cache_clear()


def _measure(**kwargs):
    kwargs.setdefault("image_model", "ViT-B/32")
    kwargs.setdefault("text_method", "spacy")
    return measure_caption_match(b"not read", "A cat on a rug", ["a cat"], [], **kwargs)


def test_missing_clip_weights_are_not_assessed_and_not_downloaded(no_network, empty_clip_cache):
    result = _measure()
    assert result.status == FlagStatus.NOT_ASSESSED
    assert result.image_similarity is None
    assert "weights are not in" in result.detail


def test_altered_clip_weights_are_not_assessed_and_not_downloaded(no_network, empty_clip_cache):
    (empty_clip_cache / "ViT-B-32.pt").write_bytes(b"not the published checkpoint")
    result = _measure()
    assert result.status == FlagStatus.NOT_ASSESSED
    assert "do not match the published checksum" in result.detail


def test_no_caption_is_not_assessed():
    result = measure_caption_match(b"x", "  ", ["a cat"], [], image_model="ViT-B/32", text_method="spacy")
    assert result.status == FlagStatus.NOT_ASSESSED
    assert "no caption" in result.detail


def test_nothing_read_from_the_picture_leaves_the_text_score_unmeasured(monkeypatch):
    monkeypatch.setattr(caption_match, "spacy_similarity", lambda a, b: pytest.fail("nothing to compare"))
    result = measure_caption_match(None, "A cat on a rug", [], [], image_model=None, text_method="spacy")
    assert result.status == FlagStatus.NOT_ASSESSED
    assert result.text_similarity is None
    assert "no scene description or on-screen text" in result.detail


def test_read_from_image_joins_scene_and_on_screen_text():
    assert read_from_image(["a cat on a rug"], ["SALE", "50% off"]) == "a cat on a rug SALE 50% off"
    assert read_from_image([], []) == ""


# --- real models -----------------------------------------------------------------------------

def _cat_png() -> bytes:
    from io import BytesIO

    from PIL import Image
    from skimage import data

    buf = BytesIO()
    Image.fromarray(data.chelsea()).save(buf, format="PNG")
    return buf.getvalue()


@pytest.mark.slow
def test_clip_prefers_the_caption_that_describes_the_picture(no_network):
    cat = _cat_png()
    fits = measure_caption_match(cat, "A ginger cat resting indoors", ["a cat"], [],
                                 image_model="ViT-B/32", text_method="spacy")
    wrong = measure_caption_match(cat, "Crowds at a football stadium at night", ["a cat"], [],
                                  image_model="ViT-B/32", text_method="spacy")
    assert fits.status == wrong.status == FlagStatus.FIRED
    assert fits.image_similarity > wrong.image_similarity
    assert fits.text_similarity > wrong.text_similarity
    assert not fits.caption_truncated


@pytest.mark.slow
def test_a_caption_longer_than_clip_reads_is_cut_and_says_so(no_network):
    long_caption = "A ginger cat resting indoors on a soft rug near the window. " * 12
    result = measure_caption_match(_cat_png(), long_caption, ["a cat"], [],
                                   image_model="ViT-B/32", text_method="spacy")
    assert result.status == FlagStatus.FIRED
    assert result.caption_truncated
