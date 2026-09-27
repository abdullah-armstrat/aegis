"""Tests for the image adapter: image + caption -> EvidenceBundle.

These are hermetic unit tests of the adapter's *assembly* logic — the extractors are
monkeypatched so no real model/engine runs here (the extractors have their own tests). The
adapter's job is to call each extractor, carry the caption through, place outputs in the
right bundle fields, and record each extractor's honest status.
"""

import pytest

from app.adapters import image_adapter
from app.config import get_settings
from app.extractors.caption_match import CaptionMatchResult
from app.extractors.captioner import CaptionResult
from app.extractors.ocr import OcrResult
from app.extractors.reverse_image import ReverseImageResult
from app.models import EvidenceBundle, FlagStatus, Modality, SceneDescription, WebMatch


def _patch(monkeypatch, ocr, rev, cap=None):
    monkeypatch.setattr(image_adapter, "extract_on_screen_text", lambda _b: ocr)
    monkeypatch.setattr(image_adapter, "find_web_matches", lambda _ref: rev)
    if cap is not None:
        monkeypatch.setattr(image_adapter, "describe_scene", lambda _b: cap)
    # Captioner is enabled by default; ensure settings reflect that for these tests.
    get_settings.cache_clear()
    monkeypatch.setenv("AEGIS_USE_CAPTIONER", "true")
    get_settings.cache_clear()


def test_adapter_assembles_bundle_and_records_status(monkeypatch):
    ocr = OcrResult(lines=["BREAKING"], status=FlagStatus.FIRED)
    rev = ReverseImageResult(
        matches=[WebMatch(url="https://e.com/2019", published_date="2019-03-04")],
        status=FlagStatus.FIRED,
    )
    cap = CaptionResult(
        scene_descriptions=[SceneDescription(text="a dry empty street")], status=FlagStatus.FIRED
    )
    _patch(monkeypatch, ocr, rev, cap)

    bundle = image_adapter.build_bundle(b"img", caption="Floods hit!", source_ref="t.png")

    assert isinstance(bundle, EvidenceBundle)
    assert bundle.caption == "Floods hit!"
    assert bundle.on_screen_text == ["BREAKING"]
    assert len(bundle.web_matches) == 1
    assert [s.text for s in bundle.scene_descriptions] == ["a dry empty street"]
    assert bundle.meta.modality == Modality.IMAGE
    assert bundle.extractor_status == {
        "ocr": FlagStatus.FIRED,
        "reverse_image": FlagStatus.FIRED,
        "captioner": FlagStatus.FIRED,
    }
    get_settings.cache_clear()


def test_adapter_propagates_not_assessed(monkeypatch):
    """A failed extractor surfaces as NOT_ASSESSED in the bundle, not as a silent empty
    field read as 'consistent'."""
    ocr = OcrResult(status=FlagStatus.NOT_ASSESSED, detail="bad image")
    rev = ReverseImageResult(status=FlagStatus.NOT_ASSESSED, detail="not in cache")
    cap = CaptionResult(status=FlagStatus.NOT_ASSESSED, detail="bad image")
    _patch(monkeypatch, ocr, rev, cap)

    bundle = image_adapter.build_bundle(b"bad", caption=None, source_ref="bad")

    assert bundle.on_screen_text == []
    assert bundle.web_matches == []
    assert bundle.scene_descriptions == []
    assert bundle.extractor_status["ocr"] == FlagStatus.NOT_ASSESSED
    assert bundle.extractor_status["reverse_image"] == FlagStatus.NOT_ASSESSED
    assert bundle.extractor_status["captioner"] == FlagStatus.NOT_ASSESSED
    get_settings.cache_clear()


def test_adapter_skips_captioner_when_disabled(monkeypatch):
    """With AEGIS_USE_CAPTIONER=false the captioner is not called and caption↔scene is
    NOT_ASSESSED — used by the eval harness to measure the before/after of BLIP."""
    ocr = OcrResult(status=FlagStatus.CLEAR)
    rev = ReverseImageResult(status=FlagStatus.NOT_ASSESSED)
    # describe_scene must NOT be called; make it raise if it is.
    def _boom(_b):
        raise AssertionError("captioner should not run when disabled")
    monkeypatch.setattr(image_adapter, "extract_on_screen_text", lambda _b: ocr)
    monkeypatch.setattr(image_adapter, "find_web_matches", lambda _ref: rev)
    monkeypatch.setattr(image_adapter, "describe_scene", _boom)
    get_settings.cache_clear()
    monkeypatch.setenv("AEGIS_USE_CAPTIONER", "false")
    get_settings.cache_clear()

    bundle = image_adapter.build_bundle(b"x", caption="anything", source_ref="x")
    assert bundle.scene_descriptions == []
    assert bundle.extractor_status["captioner"] == FlagStatus.NOT_ASSESSED
    get_settings.cache_clear()


def _patch_basic(monkeypatch, method: str):
    monkeypatch.setattr(image_adapter, "extract_on_screen_text", lambda _b: OcrResult(lines=["SALE"], status=FlagStatus.FIRED))
    monkeypatch.setattr(image_adapter, "find_web_matches", lambda _ref: ReverseImageResult(status=FlagStatus.CLEAR))
    monkeypatch.setattr(image_adapter, "describe_scene", lambda _b: CaptionResult(
        scene_descriptions=[SceneDescription(text="a cat on a rug")], status=FlagStatus.FIRED))
    get_settings.cache_clear()
    monkeypatch.setenv("AEGIS_USE_CAPTIONER", "true")
    monkeypatch.setenv("AEGIS_CAPTION_MATCH_METHOD", method)
    get_settings.cache_clear()


def test_adapter_measures_caption_fit_by_meaning(monkeypatch):
    """With the meaning method, the similarities land in the bundle with their honest status,
    measured against the scene description plus the on-screen text."""
    seen = {}

    def _measure(image_bytes, caption, scenes, on_screen, *, image_model, text_method):
        seen.update(caption=caption, scenes=scenes, on_screen=on_screen)
        return CaptionMatchResult(image_similarity=0.21, text_similarity=0.64, status=FlagStatus.FIRED)

    _patch_basic(monkeypatch, "meaning")
    monkeypatch.setattr(image_adapter, "measure_caption_match", _measure)
    bundle = image_adapter.build_bundle(b"img", caption="A dog in a park", source_ref="t.png")

    assert seen == {"caption": "A dog in a park", "scenes": ["a cat on a rug"], "on_screen": ["SALE"]}
    assert bundle.caption_match.image_similarity == 0.21
    assert bundle.caption_match.text_similarity == 0.64
    assert bundle.caption_match.image_model == "CLIP ViT-B/32"
    assert bundle.extractor_status["caption_match"] == FlagStatus.FIRED
    get_settings.cache_clear()


def test_adapter_measures_only_the_picture_for_the_image_method(monkeypatch):
    """The image method asks for no text score, so spaCy is never loaded for it."""
    asked = {}

    def _measure(image_bytes, caption, scenes, on_screen, *, image_model, text_method):
        asked.update(image_model=image_model, text_method=text_method)
        return CaptionMatchResult(image_similarity=0.21, status=FlagStatus.FIRED)

    _patch_basic(monkeypatch, "image")
    monkeypatch.setattr(image_adapter, "measure_caption_match", _measure)
    bundle = image_adapter.build_bundle(b"img", caption="A dog in a park", source_ref="t.png")

    assert asked == {"image_model": "ViT-B/32", "text_method": None}
    assert bundle.caption_match.image_similarity == 0.21
    assert bundle.caption_match.text_similarity is None
    assert bundle.caption_match.text_model is None
    assert bundle.extractor_status["caption_match"] == FlagStatus.FIRED
    get_settings.cache_clear()


@pytest.mark.parametrize("method", ["overlap", "off"])
def test_adapter_skips_the_similarity_models_when_not_needed(monkeypatch, method):
    def _boom(*_a, **_k):
        raise AssertionError(f"similarity models should not run for {method}")

    _patch_basic(monkeypatch, method)
    monkeypatch.setattr(image_adapter, "measure_caption_match", _boom)
    bundle = image_adapter.build_bundle(b"img", caption="A dog in a park", source_ref="t.png")
    assert bundle.caption_match is None
    assert "caption_match" not in bundle.extractor_status
    get_settings.cache_clear()


def test_default_method_is_the_picture_only_check(monkeypatch):
    """Set by the fresh-set test: of the methods tried, only the picture-only check flagged few
    truthful captions while catching images used out of context. Tests run word overlap by
    default (conftest), so read the production default with that override removed."""
    from app.config import Settings

    monkeypatch.delenv("AEGIS_CAPTION_MATCH_METHOD", raising=False)
    assert Settings().caption_match_method == "image"


@pytest.mark.parametrize("method, use_llm, runs", [
    ("image", "false", False),
    ("off", "false", False),
    ("image", "true", True),
    ("meaning", "false", True),
    ("overlap", "false", True),
])
def test_captioner_runs_only_when_its_description_is_read(monkeypatch, method, use_llm, runs):
    """BLIP is loaded only for word overlap, the meaning check's text score or the LLM."""
    called = []
    _patch_basic(monkeypatch, method)
    monkeypatch.setattr(image_adapter, "describe_scene", lambda _b: called.append(1) or CaptionResult(
        scene_descriptions=[SceneDescription(text="a cat on a rug")], status=FlagStatus.FIRED))
    monkeypatch.setattr(image_adapter, "measure_caption_match",
                        lambda *a, **k: CaptionMatchResult(image_similarity=0.3, text_similarity=0.8,
                                                           status=FlagStatus.FIRED))
    monkeypatch.setenv("AEGIS_USE_LLM", use_llm)
    get_settings.cache_clear()
    bundle = image_adapter.build_bundle(b"img", caption="A dog in a park", source_ref="t.png")
    assert bool(called) is runs
    assert bundle.extractor_status["captioner"] == (FlagStatus.FIRED if runs else FlagStatus.NOT_ASSESSED)
    get_settings.cache_clear()
