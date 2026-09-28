"""Scene captioner: fills the bundle's ``scene_descriptions`` using BLIP-base.

The fusion core compares this description of the picture with the user's caption. BLIP is loaded
lazily from the local Hugging Face cache (``allow_model_downloads`` lets it download). On the dev
laptop the first load took 118 s including the download, then about 1.6 s per image on CPU, well
under the 20 s limit set beforehand, so it runs locally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from io import BytesIO

from app.models import FlagStatus, SceneDescription

# model id kept in one place so it is easy to swap
_MODEL_NAME = "Salesforce/blip-image-captioning-base"
_MAX_NEW_TOKENS = 30  # upper limit on caption length, in tokens


@dataclass
class CaptionResult:
    """Outcome of a captioning attempt; ``scene_descriptions`` is empty when not FIRED."""

    scene_descriptions: list[SceneDescription] = field(default_factory=list)
    status: FlagStatus = FlagStatus.NOT_ASSESSED
    detail: str = ""


@lru_cache(maxsize=1)
def _get_model():
    """Load the BLIP processor and model once (CPU), from local files unless downloads are allowed."""
    from transformers import BlipForConditionalGeneration, BlipProcessor

    from app.config import get_settings

    local_only = not get_settings().allow_model_downloads
    if local_only:
        # pass the cached folder, not the Hub name: given a name, transformers still asks the Hub
        # about a safetensors conversion even with local_files_only set
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

    Never raises: unreadable bytes or a model failure give NOT_ASSESSED, an empty caption CLEAR.
    """
    try:
        import torch
        from PIL import Image, UnidentifiedImageError
    except ImportError as exc:  # pragma: no cover - environment guard
        return CaptionResult(status=FlagStatus.NOT_ASSESSED, detail="the model that describes the picture could not run.")

    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        image.load()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        return CaptionResult(status=FlagStatus.NOT_ASSESSED, detail="the picture could not be read.")

    try:
        processor, model = _get_model()
        inputs = processor(image, return_tensors="pt")
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=_MAX_NEW_TOKENS)
        text = processor.decode(output[0], skip_special_tokens=True).strip()
    except Exception:  # noqa: BLE001 - any model failure is "could not assess"
        return CaptionResult(status=FlagStatus.NOT_ASSESSED, detail="the model that describes the picture could not run.")

    if not text:
        return CaptionResult(status=FlagStatus.CLEAR, detail="the model gave no description of the picture.")
    return CaptionResult(
        scene_descriptions=[SceneDescription(text=text)],
        status=FlagStatus.FIRED,
    )
