"""Reverse-image-search extractor — the recycled-context signal (SSOT §1.4, §3.2).

Answers "has this exact image appeared elsewhere before?" — the highest-value, fully reliable
flag, because recency comes from a live web index rather than a model's memory. For the
Preliminary Report this reads from a **cached fixture** (ADR-007): no paid API, no signup, and
fully testable offline. The fixture's shape mirrors a live Google Vision / TinEye response, so
a real backend can drop in later behind the ``reverse_image_mode`` flag without touching fusion.

Status semantics (ADR-009) are deliberately three-way and matter here:
  * ``FIRED``        — the image is in the cache AND has web matches (possible recycled context).
  * ``CLEAR``        — the image is in the cache with an empty match list (checked, none found).
  * ``NOT_ASSESSED`` — the image is not in the cache / mode is unsupported (we could NOT check).
The CLEAR vs NOT_ASSESSED distinction is the whole point: an unknown image must never be reported
as "no recycled context found", which would be a false reassurance.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from app.config import get_settings
from app.models import FlagStatus, WebMatch

# Packaged default fixture location (overridable via settings.reverse_image_cache_path).
_DEFAULT_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "eval"
    / "fixtures"
    / "reverse_image_cache.json"
)


@dataclass
class ReverseImageResult:
    """Outcome of a reverse-image lookup."""

    matches: list[WebMatch] = field(default_factory=list)
    status: FlagStatus = FlagStatus.NOT_ASSESSED
    detail: str = ""


def _fixture_path() -> Path:
    configured = get_settings().reverse_image_cache_path
    return Path(configured) if configured else _DEFAULT_FIXTURE


@lru_cache(maxsize=8)
def _load_cache(path_str: str) -> dict:
    """Load and memoise the fixture's ``matches`` map. Memoised by path string."""
    data = json.loads(Path(path_str).read_text(encoding="utf-8"))
    matches = data.get("matches", {})
    return matches if isinstance(matches, dict) else {}


def find_web_matches(source_ref: str | None) -> ReverseImageResult:
    """Look up prior web appearances of the image identified by ``source_ref``.

    ``source_ref`` is the bundle's image identifier (e.g. the uploaded filename). Never raises:
    a missing fixture or unreadable file yields ``NOT_ASSESSED`` so fusion records honestly that
    the check could not run.
    """
    settings = get_settings()
    if settings.reverse_image_mode != "cache":
        # Live API not wired for the Prelim (ADR-007) — be explicit, don't pretend it's clear.
        return ReverseImageResult(
            status=FlagStatus.NOT_ASSESSED,
            detail=f"reverse_image_mode='{settings.reverse_image_mode}' not implemented; cache only.",
        )

    if not source_ref:
        return ReverseImageResult(
            status=FlagStatus.NOT_ASSESSED, detail="No source_ref to look up."
        )

    try:
        cache = _load_cache(str(_fixture_path()))
    except (OSError, json.JSONDecodeError) as exc:
        return ReverseImageResult(
            status=FlagStatus.NOT_ASSESSED, detail=f"Cache unreadable: {exc}"
        )

    if source_ref not in cache:
        # Unknown image — we genuinely could not check it. NOT 'clear'.
        return ReverseImageResult(
            status=FlagStatus.NOT_ASSESSED,
            detail=f"'{source_ref}' not in reverse-image cache.",
        )

    raw_matches = cache[source_ref] or []
    matches = [WebMatch(**m) for m in raw_matches]
    if matches:
        return ReverseImageResult(matches=matches, status=FlagStatus.FIRED)
    # Known image, genuinely no prior appearances found.
    return ReverseImageResult(status=FlagStatus.CLEAR, detail="No prior web appearances found.")
