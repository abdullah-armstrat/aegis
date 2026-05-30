"""Tests for the sentiment / emotional-intensity extractor.

The empty-text CLEAR path needs no model and always runs. The scoring path needs
transformers + a (cached) model download, so it is guarded with importorskip and skipped
cleanly in environments without it.
"""

import pytest

from app.extractors.sentiment import analyse_sentiment
from app.models import FlagStatus


def test_empty_text_is_clear_not_assessed_distinction():
    """Empty text is CLEAR (nothing to assess), never NOT_ASSESSED (ADR-009)."""
    r = analyse_sentiment("", source="caption")
    assert r.status == FlagStatus.CLEAR
    assert r.sentiment is None

    r2 = analyse_sentiment(None, source="caption")
    assert r2.status == FlagStatus.CLEAR


@pytest.mark.slow
def test_scores_emotional_text():
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    r = analyse_sentiment("This is an absolute outrage and a disgrace!", source="caption")
    # If the model is genuinely unavailable in this env, NOT_ASSESSED is acceptable;
    # otherwise it must FIRE with a populated, well-formed Sentiment.
    if r.status == FlagStatus.NOT_ASSESSED:
        pytest.skip(f"sentiment model unavailable: {r.detail}")
    assert r.status == FlagStatus.FIRED
    assert r.sentiment is not None
    assert r.sentiment.source == "caption"
    assert 0.0 <= r.sentiment.score <= 1.0
    assert r.sentiment.label in {"positive", "negative"}
