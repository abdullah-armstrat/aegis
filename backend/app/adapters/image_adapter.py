"""Image adapter: turns an image and its caption into an :class:`EvidenceBundle`.

Runs OCR, the reverse-image lookup, the BLIP captioner and (for the "image" and "meaning"
caption match methods) the CLIP/spaCy similarities. Each extractor's outcome goes into
``extractor_status`` so fusion can tell "ran and found nothing" apart from "could not run".
"""

from __future__ import annotations

from app.config import get_settings
from app.extractors.caption_match import (
    IMAGE_MODEL,
    SPACY_MODEL,
    TEXT_METHOD,
    measure_caption_match,
    picture_limit,
)
from app.extractors.captioner import describe_scene
from app.extractors.ocr import extract_on_screen_text
from app.extractors.reverse_image import find_web_matches
from app.models import CaptionMatch, EvidenceBundle, FlagStatus, Meta, Modality


def measure_caption_fit(
    image_bytes: bytes,
    caption: str | None,
    scene_texts: list[str],
    on_screen_text: list[str],
    method: str = "meaning",
) -> tuple[CaptionMatch, FlagStatus]:
    """Measure the caption-vs-picture similarities a method needs, and return them with a status.

    "image" only compares the picture with the caption, so spaCy is not loaded for it. A picture
    that is nearly blank or mostly text is NOT_ASSESSED (with the reason) and CLIP is skipped.
    """
    text_method = TEXT_METHOD if method == "meaning" else None
    if caption and caption.strip() and (limit := picture_limit(image_bytes, on_screen_text)):
        return CaptionMatch(image_model=f"CLIP {IMAGE_MODEL}", detail=limit), FlagStatus.NOT_ASSESSED
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
    search_web: bool = False,
) -> EvidenceBundle:
    """Run the image extractors and return the :class:`EvidenceBundle`.

    ``caption`` is passed through unchanged. ``source_ref`` is just a label (the filename), since
    the reverse-image lookup matches on content. ``posted_date`` is the optional ISO date the post
    claims, used by the recycled-context rule. The captioner can be turned off with
    ``AEGIS_USE_CAPTIONER``, in which case the caption-scene check reports NOT_ASSESSED.
    """
    settings = get_settings()
    method = settings.caption_match_method
    ocr = extract_on_screen_text(image_bytes)
    reverse = find_web_matches(image_bytes, search_web=search_web)

    # only run BLIP if something reads the scene description (overlap, meaning or the LLM)
    needs_scene = method in ("meaning", "overlap") or settings.use_llm
    extractor_detail = {}
    if settings.use_captioner and needs_scene:
        caption_result = describe_scene(image_bytes)
        scene_descriptions = caption_result.scene_descriptions
        caption_status = caption_result.status
        if caption_result.detail and caption_status != FlagStatus.FIRED:
            extractor_detail["captioner"] = caption_result.detail
    else:
        scene_descriptions = []
        caption_status = FlagStatus.NOT_ASSESSED
        if needs_scene:
            extractor_detail["captioner"] = "the model that describes the picture is switched off."

    if reverse.detail:
        extractor_detail["reverse_image"] = reverse.detail
    if reverse.web_searched:
        extractor_detail["web_search"] = reverse.web_detail
    extractor_status = {
        "ocr": ocr.status,
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
        web_matches=reverse.matches,
        caption_match=caption_match,
        extractor_status=extractor_status,
        extractor_detail=extractor_detail,
        meta=Meta(
            modality=Modality.IMAGE,
            source_ref=source_ref,
            posted_date=posted_date,
            image_phash=reverse.phash,
        ),
    )
