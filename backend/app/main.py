"""Aegis FastAPI application — entry point and routes.

Week 0: ``/health`` only, with CORS for the Vite dev server. The ``/analyze`` endpoint
(image+caption -> scorecard) arrives in Week 2 once the adapters and fusion core exist.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.config import get_settings

settings = get_settings()

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
