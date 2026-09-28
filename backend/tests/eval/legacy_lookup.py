"""Fake reverse-image results for the 19-example regression set (test-only).

Every example uses the same blank image, so a real hash lookup can't tell them apart. Instead
each example's result is read from a small fixture by its ``source_ref``: listed with matches
gives FIRED, listed but empty gives CLEAR, not listed gives NOT_ASSESSED.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from app.models import EvidenceBundle, FlagStatus, WebMatch

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "reverse_image_cache.json"


@lru_cache(maxsize=1)
def _fixture() -> dict[str, list[dict]]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))["matches"]


def legacy_lookup(source_ref: str) -> tuple[list[WebMatch], FlagStatus]:
    """Return the fixture's matches and lookup status for this filename."""
    fixture = _fixture()
    if source_ref not in fixture:
        return [], FlagStatus.NOT_ASSESSED
    matches = [WebMatch(**m) for m in fixture[source_ref] or []]
    return matches, FlagStatus.FIRED if matches else FlagStatus.CLEAR


def inject_lookup(bundle: EvidenceBundle, source_ref: str) -> None:
    """Overwrite the bundle's lookup result with the fixture one (in place)."""
    matches, status = legacy_lookup(source_ref)
    bundle.web_matches = matches
    bundle.extractor_status["reverse_image"] = status
