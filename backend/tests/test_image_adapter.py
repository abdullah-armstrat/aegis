"""Tests for the image adapter: image + caption -> EvidenceBundle.

These are hermetic unit tests of the adapter's *assembly* logic — the extractors are
monkeypatched so no real model/engine runs here (the extractors have their own tests). The
adapter's job is to call each extractor, carry the caption through, place outputs in the
right bundle fields, and record each extractor's honest status (ADR-009).
"""

from app.adapters import image_adapter
from app.extractors.ocr import OcrResult
from app.extractors.sentiment import SentimentResult
from app.models import EvidenceBundle, FlagStatus, Modality, Sentiment


def _patch(monkeypatch, ocr: OcrResult, sent: SentimentResult):
    monkeypatch.setattr(image_adapter, "extract_on_screen_text", lambda _b: ocr)
    monkeypatch.setattr(image_adapter, "analyse_sentiment", lambda _t, source: sent)


def test_adapter_assembles_bundle_and_records_status(monkeypatch):
    ocr = OcrResult(lines=["BREAKING"], status=FlagStatus.FIRED)
    sent = SentimentResult(
        Sentiment(label="negative", score=0.97, source="caption"), FlagStatus.FIRED
    )
    _patch(monkeypatch, ocr, sent)

    bundle = image_adapter.build_bundle(b"img", caption="Floods hit!", source_ref="t.png")

    assert isinstance(bundle, EvidenceBundle)
    assert bundle.caption == "Floods hit!"
    assert bundle.on_screen_text == ["BREAKING"]
    assert bundle.sentiment is not None and bundle.sentiment.label == "negative"
    assert bundle.meta.modality == Modality.IMAGE
    assert bundle.extractor_status == {"ocr": FlagStatus.FIRED, "sentiment": FlagStatus.FIRED}


def test_adapter_propagates_not_assessed(monkeypatch):
    """A failed extractor surfaces as NOT_ASSESSED in the bundle, not as a silent empty
    field read as 'consistent' (ADR-009)."""
    ocr = OcrResult(status=FlagStatus.NOT_ASSESSED, detail="bad image")
    sent = SentimentResult(None, FlagStatus.CLEAR, "No text to assess.")
    _patch(monkeypatch, ocr, sent)

    bundle = image_adapter.build_bundle(b"bad", caption=None, source_ref="bad")

    assert bundle.on_screen_text == []
    assert bundle.sentiment is None
    assert bundle.extractor_status["ocr"] == FlagStatus.NOT_ASSESSED
    assert bundle.extractor_status["sentiment"] == FlagStatus.CLEAR
