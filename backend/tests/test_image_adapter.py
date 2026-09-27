"""Tests for the image adapter: image + caption -> EvidenceBundle.

These are hermetic unit tests of the adapter's *assembly* logic — the extractors are
monkeypatched so no real model/engine runs here (the extractors have their own tests). The
adapter's job is to call each extractor, carry the caption through, place outputs in the
right bundle fields, and record each extractor's honest status.
"""

from app.adapters import image_adapter
from app.config import get_settings
from app.extractors.caption_match import CaptionMatchResult
from app.extractors.captioner import CaptionResult
from app.extractors.ocr import OcrResult
from app.extractors.reverse_image import ReverseImageResult
from app.extractors.sentiment import SentimentResult
from app.models import EvidenceBundle, FlagStatus, Modality, SceneDescription, Sentiment, WebMatch


def _patch(monkeypatch, ocr, sent, rev, cap=None):
    monkeypatch.setattr(image_adapter, "extract_on_screen_text", lambda _b: ocr)
    monkeypatch.setattr(image_adapter, "analyse_sentiment", lambda _t, source: sent)
    monkeypatch.setattr(image_adapter, "find_web_matches", lambda _ref: rev)
    if cap is not None:
        monkeypatch.setattr(image_adapter, "describe_scene", lambda _b: cap)
    # Captioner is enabled by default; ensure settings reflect that for these tests.
    get_settings.cache_clear()
    monkeypatch.setenv("AEGIS_USE_CAPTIONER", "true")
    get_settings.cache_clear()


def test_adapter_assembles_bundle_and_records_status(monkeypatch):
    ocr = OcrResult(lines=["BREAKING"], status=FlagStatus.FIRED)
    sent = SentimentResult(
        Sentiment(label="negative", score=0.97, source="caption"), FlagStatus.FIRED
    )
    rev = ReverseImageResult(
        matches=[WebMatch(url="https://e.com/2019", published_date="2019-03-04")],
        status=FlagStatus.FIRED,
    )
    cap = CaptionResult(
        scene_descriptions=[SceneDescription(text="a dry empty street")], status=FlagStatus.FIRED
    )
    _patch(monkeypatch, ocr, sent, rev, cap)

    bundle = image_adapter.build_bundle(b"img", caption="Floods hit!", source_ref="t.png")

    assert isinstance(bundle, EvidenceBundle)
    assert bundle.caption == "Floods hit!"
    assert bundle.on_screen_text == ["BREAKING"]
    assert bundle.sentiment is not None and bundle.sentiment.label == "negative"
    assert len(bundle.web_matches) == 1
    assert [s.text for s in bundle.scene_descriptions] == ["a dry empty street"]
    assert bundle.meta.modality == Modality.IMAGE
    assert bundle.extractor_status == {
        "ocr": FlagStatus.FIRED,
        "sentiment": FlagStatus.FIRED,
        "reverse_image": FlagStatus.FIRED,
        "captioner": FlagStatus.FIRED,
    }
    get_settings.cache_clear()


def test_adapter_propagates_not_assessed(monkeypatch):
    """A failed extractor surfaces as NOT_ASSESSED in the bundle, not as a silent empty
    field read as 'consistent'."""
    ocr = OcrResult(status=FlagStatus.NOT_ASSESSED, detail="bad image")
    sent = SentimentResult(None, FlagStatus.CLEAR, "No text to assess.")
    rev = ReverseImageResult(status=FlagStatus.NOT_ASSESSED, detail="not in cache")
    cap = CaptionResult(status=FlagStatus.NOT_ASSESSED, detail="bad image")
    _patch(monkeypatch, ocr, sent, rev, cap)

    bundle = image_adapter.build_bundle(b"bad", caption=None, source_ref="bad")

    assert bundle.on_screen_text == []
    assert bundle.sentiment is None
    assert bundle.web_matches == []
    assert bundle.scene_descriptions == []
    assert bundle.extractor_status["ocr"] == FlagStatus.NOT_ASSESSED
    assert bundle.extractor_status["sentiment"] == FlagStatus.CLEAR
    assert bundle.extractor_status["reverse_image"] == FlagStatus.NOT_ASSESSED
    assert bundle.extractor_status["captioner"] == FlagStatus.NOT_ASSESSED
    get_settings.cache_clear()


def test_adapter_skips_captioner_when_disabled(monkeypatch):
    """With AEGIS_USE_CAPTIONER=false the captioner is not called and caption↔scene is
    NOT_ASSESSED — used by the eval harness to measure the before/after of BLIP."""
    ocr = OcrResult(status=FlagStatus.CLEAR)
    sent = SentimentResult(None, FlagStatus.CLEAR)
    rev = ReverseImageResult(status=FlagStatus.NOT_ASSESSED)
    # describe_scene must NOT be called; make it raise if it is.
    def _boom(_b):
        raise AssertionError("captioner should not run when disabled")
    monkeypatch.setattr(image_adapter, "extract_on_screen_text", lambda _b: ocr)
    monkeypatch.setattr(image_adapter, "analyse_sentiment", lambda _t, source: sent)
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
    monkeypatch.setattr(image_adapter, "analyse_sentiment", lambda _t, source: SentimentResult(None, FlagStatus.CLEAR))
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


def test_adapter_skips_the_similarity_models_for_word_overlap(monkeypatch):
    def _boom(*_a, **_k):
        raise AssertionError("similarity models should not run for word overlap")

    _patch_basic(monkeypatch, "overlap")
    monkeypatch.setattr(image_adapter, "measure_caption_match", _boom)
    bundle = image_adapter.build_bundle(b"img", caption="A dog in a park", source_ref="t.png")
    assert bundle.caption_match is None
    assert "caption_match" not in bundle.extractor_status
    get_settings.cache_clear()
