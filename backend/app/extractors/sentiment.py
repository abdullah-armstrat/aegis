"""Sentiment / emotional-intensity extractor (the bundle's ``sentiment`` field).

Written for the emotional-framing flag; since that flag moved to wording markers, no rule
reads this output. We deliberately read both a sentiment *label* and
an *intensity* (how far from neutral / how confident), since the media-literacy signal is
about intensity of framing, not which polarity.

Uses a small HuggingFace text-classification model via ``transformers``. The model is loaded
lazily and memoised so importing this module is cheap and the model downloads only on first
use. Like the OCR extractor, this never raises: if the model or its deps are unavailable, it
returns a ``NOT_ASSESSED`` result so fusion records honestly that the check could not run.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from app.models import FlagStatus, Sentiment

# A small, widely-available sentiment model. Kept here as the single source of truth so a
# model swap (or an accuracy trial vs an alternative) is a one-line change.
_MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"


@dataclass
class SentimentResult:
    """Outcome of a sentiment attempt; ``sentiment`` is None when NOT_ASSESSED."""

    sentiment: Sentiment | None
    status: FlagStatus
    detail: str = ""


@lru_cache(maxsize=1)
def _get_pipeline():
    """Lazily build and memoise the HF sentiment pipeline (CPU)."""
    from transformers import pipeline

    return pipeline("sentiment-analysis", model=_MODEL_NAME)


def analyse_sentiment(text: str | None, source: str) -> SentimentResult:
    """Score the emotional intensity of ``text``.

    ``source`` records which field was scored (e.g. "caption", "transcript"). Empty/whitespace
    text is a legitimate CLEAR (nothing to assess), distinct from a model failure
    (NOT_ASSESSED), so "nothing to check" never reads as "could not check".
    """
    if not text or not text.strip():
        return SentimentResult(None, FlagStatus.CLEAR, "No text to assess.")

    try:
        pipe = _get_pipeline()
        out = pipe(text[:512])[0]  # truncate to the model's comfortable input length
    except ImportError as exc:  # pragma: no cover - environment guard
        return SentimentResult(None, FlagStatus.NOT_ASSESSED, f"Sentiment deps missing: {exc}")
    except Exception as exc:  # noqa: BLE001 - any model failure is "could not assess"
        return SentimentResult(None, FlagStatus.NOT_ASSESSED, f"Sentiment model error: {exc}")

    sentiment = Sentiment(
        label=str(out["label"]).lower(),
        score=float(out["score"]),  # model confidence; used as the intensity signal
        source=source,
    )
    return SentimentResult(sentiment, FlagStatus.FIRED)
