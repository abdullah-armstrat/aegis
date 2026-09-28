"""Similarity scores between the caption and the picture, used by the caption-match check.

Image score: the caption against the picture itself, with CLIP. Text score: the caption against
the BLIP scene description and on-screen text, with CLIP's text encoder or spaCy word vectors.
Both are cosine similarities, so they compare meaning rather than exact words. Models are only
loaded from local files unless ``allow_model_downloads`` is set; a missing one gives NOT_ASSESSED.
CLIP reads at most 77 tokens, so longer captions are cut and marked as truncated.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import numpy as np

from app.models import FlagStatus

CLIP_CACHE = Path(os.path.expanduser("~/.cache/clip"))
SPACY_MODEL = "en_core_web_md"

# Chosen on the VERITE sample: of the models that flagged at most 1 in 10 truthful captions, this
# pair caught the most out-of-context images. The default picture-only check uses IMAGE_MODEL
# alone, the meaning check adds TEXT_METHOD.
IMAGE_MODEL = "ViT-B/32"
TEXT_METHOD = "spacy"


@dataclass
class CaptionMatchResult:
    """Outcome of measuring the caption against the image; scores are None when not measured."""

    image_similarity: float | None = None
    text_similarity: float | None = None
    caption_truncated: bool = False
    status: FlagStatus = FlagStatus.NOT_ASSESSED
    detail: str = ""


# ------------------------------------------------------------------------------ CLIP
def clip_checkpoint(arch: str) -> Path:
    """Return the cached CLIP checkpoint after checking it against OpenAI's published SHA-256.

    Raises FileNotFoundError if the file is missing and ValueError if the checksum is wrong.
    """
    from clip.clip import _MODELS

    url = _MODELS[arch]
    path = CLIP_CACHE / os.path.basename(url)
    if not path.is_file():
        raise FileNotFoundError(f"CLIP {arch} weights are not in {CLIP_CACHE}")
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != url.split("/")[-2]:
        raise ValueError(f"CLIP {arch} weights at {path} do not match the published checksum")
    return path


@lru_cache(maxsize=2)
def _clip(arch: str):
    """Load CLIP from the checked local file (cached per architecture).

    A missing file is only downloaded from OpenAI when ``allow_model_downloads`` is set.
    """
    import clip

    from app.config import get_settings

    try:
        source = str(clip_checkpoint(arch))
    except FileNotFoundError:
        if not get_settings().allow_model_downloads:
            raise
        source = arch  # a model name makes the clip package download it into CLIP_CACHE
    model, preprocess = clip.load(source, device="cpu", download_root=str(CLIP_CACHE))
    model.eval()
    return model, preprocess


def clip_token_count(text: str) -> int:
    """Number of tokens CLIP needs for this text, including the start and end tokens."""
    from clip.clip import _tokenizer

    return len(_tokenizer.encode(text)) + 2


def clip_text_vector(text: str, arch: str) -> np.ndarray:
    """Unit-length CLIP vector for a text (cut to CLIP's token limit)."""
    import clip
    import torch

    model, _ = _clip(arch)
    with torch.no_grad():
        vec = model.encode_text(clip.tokenize([text], truncate=True))[0].float().numpy()
    return vec / np.linalg.norm(vec)


def clip_image_vector(image, arch: str) -> np.ndarray:
    """Unit-length CLIP vector for a PIL image."""
    import torch

    model, preprocess = _clip(arch)
    with torch.no_grad():
        vec = model.encode_image(preprocess(image.convert("RGB")).unsqueeze(0))[0].float().numpy()
    return vec / np.linalg.norm(vec)


def clip_context_length(arch: str) -> int:
    return _clip(arch)[0].context_length


# ------------------------------------------------------------------------------ spaCy
@lru_cache(maxsize=1)
def _spacy():
    import spacy

    return spacy.load(SPACY_MODEL)


def spacy_similarity(a: str, b: str) -> float | None:
    """Cosine similarity of the two texts' averaged word vectors; None if either has no vector."""
    nlp = _spacy()
    da, db = nlp(a), nlp(b)
    if not da.vector_norm or not db.vector_norm:
        return None
    return float(da.similarity(db))


# ------------------------------------------------------------------------------ the picture's limits
# CLIP needs a scene in the picture to compare with the caption, so a nearly blank or mostly-text
# picture is NOT_ASSESSED with the reason. Both limits were set before any results were measured.
NEARLY_BLANK_STD = 10.0   # greyscale standard deviation (0-255) under this: nearly blank
MOSTLY_TEXT_SHARE = 0.40  # OCR word boxes covering more than this share of it: mostly text


def picture_measures(image_bytes: bytes, on_screen_text: list[str]) -> tuple[float | None, float | None]:
    """Return (greyscale standard deviation, share of the picture covered by OCR text).

    The share is only measured when OCR read some text (otherwise 0.0). Either value is None if it
    could not be measured.
    """
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(BytesIO(image_bytes)) as img:
            img.load()
            spread = float(np.asarray(img.convert("L"), dtype=np.float32).std())
    except (UnidentifiedImageError, OSError, ValueError):
        return None, None
    if not on_screen_text:
        return spread, 0.0
    from app.extractors.ocr import text_box_share

    return spread, text_box_share(image_bytes)


def picture_limit(image_bytes: bytes, on_screen_text: list[str]) -> str | None:
    """Return why the picture has no scene for CLIP to compare, or None if it has one."""
    spread, share = picture_measures(image_bytes, on_screen_text)
    if spread is not None and spread < NEARLY_BLANK_STD:
        return (f"The picture is nearly blank (its brightness varies by only {spread:.1f} on a scale of 0 to "
                f"255; under {NEARLY_BLANK_STD:.0f} counts as nearly blank), so there is no scene in it to "
                "compare with the caption.")
    if share is not None and share > MOSTLY_TEXT_SHARE:
        return (f"The picture is mostly text (the words read in it cover {share:.0%} of it; over "
                f"{MOSTLY_TEXT_SHARE:.0%} counts as mostly text), so there is little scene in it to compare "
                "with the caption.")
    return None


# ------------------------------------------------------------------------------ the check's inputs
def read_from_image(scene_descriptions: list[str], on_screen_text: list[str]) -> str:
    """Join what the other extractors read from the picture into one text."""
    return " ".join([*scene_descriptions, *on_screen_text]).strip()


def measure_caption_match(
    image_bytes: bytes | None,
    caption: str | None,
    scene_descriptions: list[str],
    on_screen_text: list[str],
    *,
    image_model: str | None,
    text_method: str | None,
) -> CaptionMatchResult:
    """Measure the scores the configured check needs. Never raises.

    ``image_model`` is a CLIP architecture ("ViT-B/32", "RN50") or None to skip the image score.
    ``text_method`` is "clip" (the text encoder of ``image_model``, or ViT-B/32 if that is None),
    "spacy", or None to skip the text score. The status is FIRED once every requested score was
    measured, NOT_ASSESSED otherwise.
    """
    if not caption or not caption.strip():
        return CaptionMatchResult(detail="There is no caption to compare with the image.")
    result = CaptionMatchResult()
    try:
        if image_model is not None or text_method == "clip":
            arch = image_model or "ViT-B/32"
            result.caption_truncated = clip_token_count(caption) > clip_context_length(arch)
        if image_model is not None:
            if image_bytes is None:
                return CaptionMatchResult(detail="There is no image to compare with the caption.")
            from PIL import Image, UnidentifiedImageError

            try:
                with Image.open(BytesIO(image_bytes)) as img:
                    img.load()
                    picture = img.convert("RGB")
            except (UnidentifiedImageError, OSError, ValueError) as exc:
                return CaptionMatchResult(detail="The picture could not be read.")
            result.image_similarity = float(
                clip_image_vector(picture, image_model) @ clip_text_vector(caption, image_model)
            )
        if text_method is not None:
            seen = read_from_image(scene_descriptions, on_screen_text)
            if seen:
                if text_method == "clip":
                    arch = image_model or "ViT-B/32"
                    result.text_similarity = float(
                        clip_text_vector(seen, arch) @ clip_text_vector(caption, arch)
                    )
                elif text_method == "spacy":
                    result.text_similarity = spacy_similarity(caption, seen)
                else:
                    raise ValueError(f"unknown text method {text_method!r}")
    except (FileNotFoundError, OSError, ValueError, ImportError, RuntimeError):
        return CaptionMatchResult(detail="A model that compares the caption with the picture could not run.")

    measured = (image_model is None or result.image_similarity is not None) and (
        text_method is None or result.text_similarity is not None
    )
    result.status = FlagStatus.FIRED if measured else FlagStatus.NOT_ASSESSED
    if not measured:
        result.detail = "There was no scene description or on-screen text to compare with the caption."
    return result
