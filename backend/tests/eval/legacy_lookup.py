"""Injected reverse-image results for the 19-example regression set (ADR-018).

The regression set scores the recycled-context *rule*, and every one of its examples uses the
same blank image. Since WP-1 the production lookup matches on image content, so it cannot tell
those examples apart: all of them would hash identically. Instead of hashing, the harness injects
each example's lookup result, exactly as it injects ``inject_scene`` for caption-vs-scene. The
results come from the pre-WP-1 filename-keyed fixture, looked up by the example's ``source_ref``,
with the same three outcomes the old extractor gave:

  source_ref in the fixture with matches  ->  matches, lookup status FIRED
  source_ref in the fixture, empty list   ->  no matches, lookup status CLEAR (searched, none)
  source_ref not in the fixture           ->  no matches, lookup status NOT_ASSESSED

That keeps the set's meaning, and its pinned numbers, independent of the hash threshold, which is
still provisional. Hash matching itself is measured by the robustness harness and unit tests.
This module is test-only: nothing in ``app/`` looks anything up by filename any more.
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
    """The lookup result the pre-WP-1 extractor returned for this filename."""
    fixture = _fixture()
    if source_ref not in fixture:
        return [], FlagStatus.NOT_ASSESSED
    matches = [WebMatch(**m) for m in fixture[source_ref] or []]
    return matches, FlagStatus.FIRED if matches else FlagStatus.CLEAR


def inject_lookup(bundle: EvidenceBundle, source_ref: str) -> None:
    """Replace the bundle's content-based lookup with the example's injected one, in place."""
    matches, status = legacy_lookup(source_ref)
    bundle.web_matches = matches
    bundle.extractor_status["reverse_image"] = status
