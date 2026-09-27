"""Scene-caption extractor: fills the bundle's ``scene_descriptions``.

Describes what an image actually shows, so the fusion core can compare that description with
the user's caption (the caption↔scene-mismatch check). Uses BLIP-base via ``transformers``,
loaded lazily and memoised so import stays cheap. It is read from the local Hugging Face cache
only; ``allow_model_downloads`` lets a missing copy be downloaded. Measured on the development laptop on 2026-05-31: first load 118.0s (including the
one-time model download), then warm captioning ~1.6s/image on CPU — well under the pre-set
20s/image threshold, so it runs locally rather than on a hosted service.

Like the other extractors, this never raises: if the model or its deps are unavailable it
returns a ``NOT_ASSESSED`` result so fusion records honestly that the check could not run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from io import BytesIO

from app.models import FlagStatus, SceneDescription

# Single source of truth for the model id, so a swap / accuracy trial is a one-line change.
_MODEL_NAME = "Salesforce/blip-image-captioning-base"
_MAX_NEW_TOKENS = 30


@dataclass
class CaptionResult:
    """Outcome of a captioning attempt; ``scene_descriptions`` is empty when not FIRED."""

    scene_descriptions: list[SceneDescription] = field(default_factory=list)
    status: FlagStatus = FlagStatus.NOT_ASSESSED
    detail: str = ""


@lru_cache(maxsize=1)
def _get_model():
    """Lazily build and memoise the BLIP processor+model (CPU), from local files unless downloads are allowed."""
    from transformers import BlipForConditionalGeneration, BlipProcessor

    from app.config import get_settings

    local_only = not get_settings().allow_model_downloads
    if local_only:
        # The cached folder itself, not the Hub name: given a name, transformers asks the Hub about
        # a safetensors conversion even with local_files_only set.
        from huggingface_hub import snapshot_download

        source = snapshot_download(_MODEL_NAME, local_files_only=True)
    else:
        source = _MODEL_NAME
    processor = BlipProcessor.from_pretrained(source, local_files_only=local_only)
    model = BlipForConditionalGeneration.from_pretrained(source, local_files_only=local_only)
    model.eval()
    return processor, model


def describe_scene(image_bytes: bytes) -> CaptionResult:
    """Caption the image and return a :class:`CaptionResult`.

    Never raises: unreadable bytes or a model/dep failure become a ``NOT_ASSESSED`` result so
    the caller can record honestly that the check could not be performed.
    """
    try:
        import torch
        from PIL import Image, UnidentifiedImageError
    except ImportError as exc:  # pragma: no cover - environment guard
        return CaptionResult(status=FlagStatus.NOT_ASSESSED, detail=f"Captioner deps missing: {exc}")

    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        image.load()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        return CaptionResult(
            status=FlagStatus.NOT_ASSESSED, detail=f"Image could not be read: {exc}"
        )

    try:
        processor, model = _get_model()
        inputs = processor(image, return_tensors="pt")
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=_MAX_NEW_TOKENS)
        text = processor.decode(output[0], skip_special_tokens=True).strip()
    except Exception as exc:  # noqa: BLE001 - any model failure is "could not assess"
        return CaptionResult(status=FlagStatus.NOT_ASSESSED, detail=f"Captioner error: {exc}")

    if not text:
        return CaptionResult(status=FlagStatus.CLEAR, detail="Model produced no caption.")
    return CaptionResult(
        scene_descriptions=[SceneDescription(text=text)],
        status=FlagStatus.FIRED,
    )
