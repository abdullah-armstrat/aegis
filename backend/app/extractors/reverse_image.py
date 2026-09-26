"""Reverse-image lookup — the recycled-context signal, matched by image content (WP-1, ADR-017).

Answers "has this picture appeared before, and when?". The upload is fingerprinted with a 64-bit
perceptual hash (``phash.py``) and compared against an **image history index**: a JSON file in
which each entry holds a hash, the earliest known date the image appeared, and the pages it
appeared on. A match is a Hamming distance at or below ``phash_match_threshold``. Matching on
content means a renamed, recompressed or resized copy is still found — the filename lookup it
replaces (ADR-007) was defeated by any re-save.

The index runs fully offline with no key. The live web lookup (WP-4) will sit behind the same
function and the same ``ReverseImageResult``.

Status semantics (ADR-009). This extractor reports what the *lookup* found; whether that makes
the post look recycled depends on the posting date, which the rule layer decides.
  * ``FIRED``        — at least one index entry is within the threshold.
  * ``CLEAR``        — the lookup ran against a readable index and nothing is within the threshold.
  * ``NOT_ASSESSED`` — the lookup could not run: unreadable image, missing or invalid index, or an
                       unsupported mode. Never reported as CLEAR, which would be false reassurance.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache
from pathlib import Path

from app.config import get_settings
from app.extractors.phash import HASH_BITS, compute_phash, hamming
from app.models import FlagStatus, WebMatch

_DEFAULT_INDEX = Path(__file__).resolve().parents[1] / "data" / "image_history_index.json"
_OFFLINE_MODES = {"index", "cache"}  # "cache" = the pre-WP-1 name, accepted as an alias
_HEX = re.compile(rf"^[0-9a-f]{{{HASH_BITS // 4}}}$")


@dataclass(frozen=True)
class IndexEntry:
    """One known image: its hash, when it first appeared, and where."""

    id: str
    phash: str
    earliest_date: str
    sources: tuple[WebMatch, ...]


@dataclass
class ReverseImageResult:
    """Outcome of a lookup. ``phash`` is set whenever the image could be hashed."""

    matches: list[WebMatch] = field(default_factory=list)
    status: FlagStatus = FlagStatus.NOT_ASSESSED
    detail: str = ""
    phash: str | None = None
    best_distance: int | None = None


class HistoryIndexError(ValueError):
    """The history index file exists but does not satisfy the contract."""


def _index_path() -> Path:
    configured = get_settings().image_index_path
    return Path(configured) if configured else _DEFAULT_INDEX


def _parse_entry(raw: dict, position: int) -> IndexEntry:
    """Validate one index entry. Its earliest_date must be backed by a dated source."""
    where = f"entry {position} ({raw.get('id', '?')})"
    phash = str(raw.get("phash", "")).lower()
    if not _HEX.match(phash):
        raise HistoryIndexError(f"{where}: phash must be {HASH_BITS // 4} hex characters")
    try:
        earliest = date.fromisoformat(str(raw.get("earliest_date")))
    except ValueError as exc:
        raise HistoryIndexError(f"{where}: earliest_date is not an ISO date") from exc
    sources = tuple(WebMatch(**s) for s in raw.get("sources") or [])
    if not sources:
        raise HistoryIndexError(f"{where}: an entry needs at least one source page")
    dated = [date.fromisoformat(s.published_date) for s in sources if s.published_date]
    if not dated or min(dated) != earliest:
        raise HistoryIndexError(f"{where}: earliest_date must equal the earliest dated source")
    return IndexEntry(id=str(raw["id"]), phash=phash, earliest_date=earliest.isoformat(),
                      sources=sources)


@lru_cache(maxsize=8)
def load_index(path_str: str) -> tuple[IndexEntry, ...]:
    """Load and validate the history index. Memoised by path; raises on any contract breach."""
    data = json.loads(Path(path_str).read_text(encoding="utf-8"))
    entries = data.get("entries")
    if not isinstance(entries, list):
        raise HistoryIndexError("index has no 'entries' list")
    parsed = tuple(_parse_entry(e, i) for i, e in enumerate(entries))
    ids = [e.id for e in parsed]
    if len(ids) != len(set(ids)):
        raise HistoryIndexError("index entry ids are not unique")
    return parsed


def find_web_matches(image_bytes: bytes) -> ReverseImageResult:
    """Look up prior appearances of this image by its content. Never raises."""
    settings = get_settings()
    if settings.reverse_image_mode not in _OFFLINE_MODES:
        return ReverseImageResult(
            detail=f"reverse_image_mode='{settings.reverse_image_mode}' is not available; "
            "only the offline image history index is implemented."
        )

    hashed = compute_phash(image_bytes)
    if hashed.hash_hex is None:
        return ReverseImageResult(detail=hashed.detail)

    try:
        index = load_index(str(_index_path()))
    except (OSError, json.JSONDecodeError, ValueError, TypeError, KeyError) as exc:
        return ReverseImageResult(phash=hashed.hash_hex, detail=f"Image history index unusable: {exc}")

    threshold = settings.phash_match_threshold
    scored: list[tuple[int, IndexEntry]] = []
    for entry in index:
        d = hamming(hashed.hash_hex, entry.phash)
        if settings.phash_mirror_lookup and hashed.mirrored_hex:
            d = min(d, hamming(hashed.mirrored_hex, entry.phash))
        if d <= threshold:
            scored.append((d, entry))

    if not scored:
        return ReverseImageResult(
            status=FlagStatus.CLEAR, phash=hashed.hash_hex,
            detail=f"No image in the history index ({len(index)} entries) is within "
            f"{threshold} of {HASH_BITS} bits.",
        )

    scored.sort(key=lambda pair: (pair[0], pair[1].id))
    matches = [
        source.model_copy(update={"hash_distance": d})
        for d, entry in scored
        for source in entry.sources
    ]
    return ReverseImageResult(
        matches=matches, status=FlagStatus.FIRED, phash=hashed.hash_hex,
        best_distance=scored[0][0],
    )
