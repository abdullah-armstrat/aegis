"""Aegis FastAPI application — entry point and routes.

Routes: ``/health`` (liveness) and ``/analyze`` (image + caption -> explainable scorecard).
``/analyze`` runs the image adapter to build an Evidence Bundle, then the fusion core to
produce a :class:`Scorecard` — a list of typed, explained flags, never a trust verdict.
"""

from __future__ import annotations

import re
from datetime import date, timedelta

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
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _posted_date_error(value: str) -> str | None:
    """Why ``posted_date`` is unusable, or None if it is empty or a valid past ISO date.

    One day of slack past the server's date absorbs time-zone differences; anything later
    cannot be the date a post was published.
    """
    if not value:
        return None
    try:
        if not _ISO_DATE.match(value):
            raise ValueError
        parsed = date.fromisoformat(value)
    except ValueError:
        return "posted_date must be a date in the form YYYY-MM-DD."
    if parsed > date.today() + timedelta(days=1):
        return "posted_date cannot be in the future."
    return None

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
            "phash_match_threshold": settings.phash_match_threshold,
            "phash_mirror_lookup": settings.phash_mirror_lookup,
            "keypoint_matching": settings.keypoint_matching,
        },
    }


@app.post("/analyze", response_model=Scorecard, tags=["analysis"])
async def analyze(
    image: UploadFile = File(..., description="The image to audit."),
    caption: str = Form("", description="The post's caption accompanying the image."),
    posted_date: str = Form(
        "", description="Optional: the date the post says it was published, as YYYY-MM-DD."
    ),
) -> Scorecard:
    """Audit an image + caption and return an explainable :class:`Scorecard`.

    Runs the image adapter (OCR, sentiment, reverse-image by content, captioner) to build an
    Evidence Bundle, then the fusion core (deterministic rules + optional LLM second opinion)
    to produce the scorecard. The result describes what was checked — never a trust verdict.
    """
    # The content-type guard runs before reading the body, where raising HTTPException
    # converts to a response cleanly.
    if image.content_type and not image.content_type.startswith(_ALLOWED_PREFIXES):
        raise HTTPException(status_code=415, detail=f"Unsupported file type: {image.content_type}")

    posted_date = posted_date.strip()
    if (problem := _posted_date_error(posted_date)) is not None:
        return JSONResponse(status_code=400, content={"detail": problem})

    data = await image.read()
    # Post-read guards return a JSONResponse rather than raising: on this Starlette version,
    # raising an HTTPException after the multipart body is consumed does not convert to a
    # response cleanly (it surfaces as a RuntimeError). Returning a Response is robust.
    if not data:
        return JSONResponse(status_code=400, content={"detail": "Empty image upload."})
    if len(data) > _MAX_IMAGE_BYTES:
        return JSONResponse(status_code=413, content={"detail": "Image exceeds the 10 MB limit."})

    bundle = build_bundle(
        data,
        caption=caption or None,
        source_ref=image.filename,
        posted_date=posted_date or None,
    )
    return build_scorecard(bundle)
