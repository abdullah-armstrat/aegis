"""Tests for the image adapter: image + caption -> EvidenceBundle.

These are hermetic unit tests of the adapter's *assembly* logic — the extractors are
monkeypatched so no real model/engine runs here (the extractors have their own tests). The
adapter's job is to call each extractor, carry the caption through, place outputs in the
right bundle fields, and record each extractor's honest status.
"""

from app.adapters import image_adapter
from app.config import get_settings
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
