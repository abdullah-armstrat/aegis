"""Image adapter — turns an image + caption into an :class:`EvidenceBundle`.

This is the image path's entry into the modality-agnostic core (ADR-003): it runs the
image extractors and assembles their outputs into the one normalised structure the fusion
core consumes. Extractors are wired in one at a time as they land; each records its outcome
in ``extractor_status`` so fusion can honour the fired/clear/not_assessed distinction
(ADR-009) rather than inferring meaning from an empty field.

Wired: OCR, sentiment, reverse-image (cached), and the BLIP captioner (ADR-013).
"""

from __future__ import annotations

from app.config import get_settings
from app.extractors.captioner import describe_scene
from app.extractors.ocr import extract_on_screen_text
from app.extractors.reverse_image import find_web_matches
from app.extractors.sentiment import analyse_sentiment
from app.models import EvidenceBundle, FlagStatus, Meta, Modality


def build_bundle(image_bytes: bytes, caption: str | None, source_ref: str | None = None) -> EvidenceBundle:
    """Run the image extractors and assemble an :class:`EvidenceBundle`.

    The user-supplied ``caption`` is carried through verbatim; the extractors populate the
    evidence fields. Each extractor's honest status is recorded in ``extractor_status``.

    Wired: OCR, sentiment (over the caption), reverse-image (cached), BLIP captioner. The
    captioner can be disabled via ``AEGIS_USE_CAPTIONER`` (e.g. for fast tests); when off, the
    caption↔scene check honestly reports NOT_ASSESSED rather than passing silently (ADR-009).
    """
    ocr = extract_on_screen_text(image_bytes)
    sentiment = analyse_sentiment(caption, source="caption")
    reverse = find_web_matches(source_ref)

    if get_settings().use_captioner:
        caption_result = describe_scene(image_bytes)
        scene_descriptions = caption_result.scene_descriptions
        caption_status = caption_result.status
    else:
        scene_descriptions = []
        caption_status = FlagStatus.NOT_ASSESSED

    return EvidenceBundle(
        scene_descriptions=scene_descriptions,
        on_screen_text=ocr.lines,
        caption=caption,
        sentiment=sentiment.sentiment,
        web_matches=reverse.matches,
        extractor_status={
            "ocr": ocr.status,
            "sentiment": sentiment.status,
            "reverse_image": reverse.status,
            "captioner": caption_status,
        },
        meta=Meta(modality=Modality.IMAGE, source_ref=source_ref),
    )
