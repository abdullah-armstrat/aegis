"""Aegis FastAPI application — entry point and routes.

Routes: ``/health`` (liveness) and ``/analyze`` (image + caption -> explainable scorecard).
``/analyze`` runs the image adapter to build an Evidence Bundle, then the fusion core to
produce a :class:`Scorecard` — a list of typed, explained flags, never a trust verdict.
"""

from __future__ import annotations

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import __version__
from app.adapters.image_adapter import build_bundle
from app.config import get_settings
from app.fusion.scorecard import build_scorecard
from app.models import Scorecard

settings = get_settings()

# Guard against oversized uploads (the analysis is CPU-bound; large images add no value).
_MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB
_ALLOWED_PREFIXES = ("image/",)

app = FastAPI(
    title="Aegis",
    description=(
        "A multi-modal cross-consistency auditor. Surfaces cross-modal contradictions "
        "and recycled context in social-media content and explains them — it never "
        "outputs a trust verdict."
    ),
    version=__version__,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["meta"])
def health() -> dict:
    """Liveness probe. Reports version and the active feature-flag configuration."""
    return {
        "status": "ok",
        "service": "aegis",
        "version": __version__,
        "config": {
            "reverse_image_mode": settings.reverse_image_mode,
            "use_llm": settings.use_llm,
        },
    }


@app.post("/analyze", response_model=Scorecard, tags=["analysis"])
async def analyze(
    image: UploadFile = File(..., description="The image to audit."),
    caption: str = Form("", description="The post's caption accompanying the image."),
) -> Scorecard:
    """Audit an image + caption and return an explainable :class:`Scorecard`.

    Runs the image adapter (OCR, sentiment, reverse-image; captioner pending) to build an
    Evidence Bundle, then the fusion core (deterministic rules + optional LLM second opinion)
    to produce the scorecard. The result describes what was checked — never a trust verdict.
    """
    # The content-type guard runs before reading the body, where raising HTTPException
    # converts to a response cleanly.
    if image.content_type and not image.content_type.startswith(_ALLOWED_PREFIXES):
        raise HTTPException(status_code=415, detail=f"Unsupported file type: {image.content_type}")

    data = await image.read()
    # Post-read guards return a JSONResponse rather than raising: on this Starlette version,
    # raising an HTTPException after the multipart body is consumed does not convert to a
    # response cleanly (it surfaces as a RuntimeError). Returning a Response is robust.
    if not data:
        return JSONResponse(status_code=400, content={"detail": "Empty image upload."})
    if len(data) > _MAX_IMAGE_BYTES:
        return JSONResponse(status_code=413, content={"detail": "Image exceeds the 10 MB limit."})

    bundle = build_bundle(data, caption=caption or None, source_ref=image.filename)
    return build_scorecard(bundle)
