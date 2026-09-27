"""Image adapter — turns an image + caption into an :class:`EvidenceBundle`.

This is the image path's entry into the modality-agnostic core: it runs the
image extractors and assembles their outputs into the one normalised structure the fusion
core consumes. Extractors are wired in one at a time as they land; each records its outcome
in ``extractor_status`` so fusion can tell a check that ran and found nothing from one that
could not run, rather than inferring meaning from an empty field.

Wired: OCR, sentiment, reverse-image (matched by content against the image history index),
the BLIP captioner, which runs locally on the CPU, and, when ``caption_match_method`` is
"image" or "meaning", the caption-vs-picture similarities that method needs (CLIP, and spaCy
for "meaning" only).
"""

from __future__ import annotations

from app.config import get_settings
from app.extractors.caption_match import IMAGE_MODEL, SPACY_MODEL, TEXT_METHOD, measure_caption_match
from app.extractors.captioner import describe_scene
from app.extractors.ocr import extract_on_screen_text
from app.extractors.reverse_image import find_web_matches
from app.extractors.sentiment import analyse_sentiment
from app.models import CaptionMatch, EvidenceBundle, FlagStatus, Meta, Modality


def measure_caption_fit(
    image_bytes: bytes,
    caption: str | None,
    scene_texts: list[str],
    on_screen_text: list[str],
    method: str = "meaning",
) -> tuple[CaptionMatch, FlagStatus]:
    """The caption-vs-picture similarities a method needs, and the measurement's status.

    "image" measures only the picture against the caption, so spaCy is never loaded for it.
    """
    text_method = TEXT_METHOD if method == "meaning" else None
    result = measure_caption_match(
        image_bytes, caption, scene_texts, on_screen_text,
        image_model=IMAGE_MODEL, text_method=text_method,
    )
    if text_method is None:
        text_model = None
    else:
        text_model = f"spaCy {SPACY_MODEL}" if text_method == "spacy" else f"CLIP {IMAGE_MODEL} text encoder"
    match = CaptionMatch(
        image_similarity=result.image_similarity,
        text_similarity=result.text_similarity,
        image_model=f"CLIP {IMAGE_MODEL}",
        text_model=text_model,
        caption_truncated=result.caption_truncated,
        detail=result.detail,
    )
    return match, result.status


def build_bundle(
    image_bytes: bytes,
    caption: str | None,
    source_ref: str | None = None,
    posted_date: str | None = None,
) -> EvidenceBundle:
    """Run the image extractors and assemble an :class:`EvidenceBundle`.

    The user-supplied ``caption`` is carried through verbatim; the extractors populate the
    evidence fields. Each extractor's honest status is recorded in ``extractor_status``.

    ``source_ref`` is a label only (the uploaded filename): the reverse-image lookup
    matches on the image's content, so renaming a file no longer changes the result.
    ``posted_date`` is the optional ISO date the post claims; the recycled-context rule compares
    earlier appearances against it.

    Wired: OCR, sentiment (over the caption), reverse-image (history index), BLIP captioner. The
    captioner can be disabled via ``AEGIS_USE_CAPTIONER`` (e.g. for fast tests); when off, the
    caption↔scene check honestly reports NOT_ASSESSED rather than passing silently. It also runs
    only when something reads the scene description: word overlap, the meaning check's text
    score, or the LLM. The picture-only check and a switched-off check do not, so BLIP is not
    loaded for them.
    """
    settings = get_settings()
    method = settings.caption_match_method
    ocr = extract_on_screen_text(image_bytes)
    sentiment = analyse_sentiment(caption, source="caption")
    reverse = find_web_matches(image_bytes)

    needs_scene = method in ("meaning", "overlap") or settings.use_llm
    if settings.use_captioner and needs_scene:
        caption_result = describe_scene(image_bytes)
        scene_descriptions = caption_result.scene_descriptions
        caption_status = caption_result.status
    else:
        scene_descriptions = []
        caption_status = FlagStatus.NOT_ASSESSED

    extractor_status = {
        "ocr": ocr.status,
        "sentiment": sentiment.status,
        "reverse_image": reverse.status,
        "captioner": caption_status,
    }
    caption_match = None
    if method in ("image", "meaning"):
        caption_match, extractor_status["caption_match"] = measure_caption_fit(
            image_bytes, caption, [s.text for s in scene_descriptions], ocr.lines, method
        )

    return EvidenceBundle(
        scene_descriptions=scene_descriptions,
        on_screen_text=ocr.lines,
        caption=caption,
        sentiment=sentiment.sentiment,
        web_matches=reverse.matches,
        caption_match=caption_match,
        extractor_status=extractor_status,
        meta=Meta(
            modality=Modality.IMAGE,
            source_ref=source_ref,
            posted_date=posted_date,
            image_phash=reverse.phash,
        ),
    )
