"""Reverse-image lookup: has this picture appeared before, and when?

The upload's pHash is compared with the offline image history index (known images with the
earliest date and the pages they appeared on). If the hash finds nothing, keypoint matching looks
for cropped or framed copies. The live web lookup (Google Cloud Vision) is added for images when
the mode is "live" or the upload asks for it. The rules decide whether a match means the post is
recycled.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache
from io import BytesIO
from pathlib import Path

from app.config import get_settings
from app.extractors.phash import HASH_BITS, compute_phash, hamming
from app.models import FlagStatus, WebMatch

_DEFAULT_INDEX = Path(__file__).resolve().parents[1] / "data" / "image_history_index.json"
_REPO_ROOT = Path(__file__).resolve().parents[3]  # index entries name their image relative to this
_LOCAL_MODES = {"local", "index", "cache"}  # "index" and "cache" are earlier names for "local"
_MODES = _LOCAL_MODES | {"live"}
OFF = "off"  # lookup turned off in the settings
_HEX = re.compile(rf"^[0-9a-f]{{{HASH_BITS // 4}}}$")


@dataclass(frozen=True)
class IndexEntry:
    """One known image: its hash, when it first appeared, and where."""

    id: str
    phash: str
    earliest_date: str
    sources: tuple[WebMatch, ...]
    image: str | None = None  # the known image itself, if available; keypoint matching needs it


@dataclass
class ReverseImageResult:
    """Result of a lookup. ``phash`` is set whenever the image could be hashed.

    ``status`` is FIRED when something matched, CLEAR when the lookup ran and found nothing, and
    NOT_ASSESSED when it could not run (unreadable image, bad index, unknown mode, or a failed web
    search with no local match).
    """

    matches: list[WebMatch] = field(default_factory=list)
    status: FlagStatus = FlagStatus.NOT_ASSESSED
    detail: str = ""
    phash: str | None = None
    best_distance: int | None = None
    method: str | None = None  # "hash" or "keypoints": which stage found the match
    web_searched: bool = False  # the live web search was asked for
    web_detail: str = ""        # what the web search found or why it could not run
    live_call: bool = False     # a Vision call was spent (not a cache replay)


class HistoryIndexError(ValueError):
    """The history index file exists but is not valid."""


def _index_path() -> Path:
    configured = get_settings().image_index_path
    return Path(configured) if configured else _DEFAULT_INDEX


def _parse_entry(raw: dict, position: int) -> IndexEntry:
    """Validate one index entry. Its earliest_date must equal its earliest dated source."""
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
    image = raw.get("image")
    return IndexEntry(id=str(raw["id"]), phash=phash, earliest_date=earliest.isoformat(),
                      sources=sources, image=str(image) if image else None)


@lru_cache(maxsize=8)
def load_index(path_str: str) -> tuple[IndexEntry, ...]:
    """Load and validate the history index (cached by path). Raises if it is not valid."""
    data = json.loads(Path(path_str).read_text(encoding="utf-8"))
    entries = data.get("entries")
    if not isinstance(entries, list):
        raise HistoryIndexError("index has no 'entries' list")
    parsed = tuple(_parse_entry(e, i) for i, e in enumerate(entries))
    ids = [e.id for e in parsed]
    if len(ids) != len(set(ids)):
        raise HistoryIndexError("index entry ids are not unique")
    return parsed


def find_web_matches(image_bytes: bytes, search_web: bool = False) -> ReverseImageResult:
    """Look up earlier copies of the image by content. Never raises.

    Searches the local index, and the web too when the mode is "live" or ``search_web`` is set.
    """
    settings = get_settings()
    if settings.reverse_image_mode == OFF:
        return ReverseImageResult(detail="The search for earlier copies of this image is switched off.")
    if settings.reverse_image_mode not in _MODES:
        return ReverseImageResult(
            detail="The search for earlier copies of this image is not set up correctly, so it did not run.")
    local = find_local_matches(image_bytes)
    if not (search_web or settings.reverse_image_mode == "live"):
        return local

    from app.extractors.web_lookup import web_lookup

    web = web_lookup(image_bytes)
    result = ReverseImageResult(
        matches=local.matches + web.matches, phash=local.phash, best_distance=local.best_distance,
        method=local.method, web_searched=True, web_detail=web.detail, live_call=web.live_call,
        detail=local.detail,
    )
    if result.matches:
        result.status = FlagStatus.FIRED
    elif local.status == FlagStatus.CLEAR and web.status == FlagStatus.CLEAR:
        result.status = FlagStatus.CLEAR
    else:
        # nothing found and at least one search could not run, so this cannot be CLEAR
        result.status = FlagStatus.NOT_ASSESSED
        result.detail = " ".join(d for d in (
            local.detail if local.status == FlagStatus.NOT_ASSESSED else "The local image history index holds no copy.",
            web.detail) if d)
    return result


def find_local_matches(image_bytes: bytes) -> ReverseImageResult:
    """Search the image history index: pHash first, then keypoints if the hash finds nothing."""
    settings = get_settings()

    hashed = compute_phash(image_bytes)
    if hashed.hash_hex is None:
        return ReverseImageResult(detail=hashed.detail)

    try:
        index = load_index(str(_index_path()))
    except (OSError, json.JSONDecodeError, ValueError, TypeError, KeyError) as exc:
        return ReverseImageResult(phash=hashed.hash_hex, detail=(
            "The image history index could not be read, so earlier copies of this image were not searched for."))

    threshold = settings.phash_match_threshold
    scored: list[tuple[int, IndexEntry]] = []
    for entry in index:
        d = hamming(hashed.hash_hex, entry.phash)
        if settings.phash_mirror_lookup and hashed.mirrored_hex:
            d = min(d, hamming(hashed.mirrored_hex, entry.phash))
        if d <= threshold:
            scored.append((d, entry))

    if scored:
        scored.sort(key=lambda pair: (pair[0], pair[1].id))
        matches = [
            source.model_copy(update={"hash_distance": d, "found_by": "index"})
            for d, entry in scored
            for source in entry.sources
        ]
        return ReverseImageResult(
            matches=matches, status=FlagStatus.FIRED, phash=hashed.hash_hex,
            best_distance=scored[0][0], method="hash",
        )

    no_hash_match = (f"No image in the history index ({len(index)} entries) is within "
                     f"{threshold} of {HASH_BITS} bits.")
    if not settings.keypoint_matching:
        return ReverseImageResult(status=FlagStatus.CLEAR, phash=hashed.hash_hex, detail=no_hash_match)

    # second stage: look for a cropped or framed copy by keypoints
    try:
        by_keypoints = _keypoint_matches(image_bytes, index, settings.keypoint_min_inliers)
    except Exception:  # noqa: BLE001 - the hash ran, but a copy it cannot see may have been missed
        return ReverseImageResult(
            status=FlagStatus.NOT_ASSESSED, phash=hashed.hash_hex,
            detail=("The search for cropped or framed copies of this image could not run, so an earlier "
                    "copy may have been missed."),
        )
    if not by_keypoints:
        return ReverseImageResult(
            status=FlagStatus.CLEAR, phash=hashed.hash_hex,
            detail=f"{no_hash_match} Keypoint matching found no cropped or framed copy either.",
        )
    matches = [
        source.model_copy(update={"keypoint_inliers": inliers, "found_by": "index"})
        for inliers, entry in by_keypoints
        for source in entry.sources
    ]
    return ReverseImageResult(
        matches=matches, status=FlagStatus.FIRED, phash=hashed.hash_hex, method="keypoints",
    )


def _entry_image(entry: IndexEntry) -> Path | None:
    """Path of the entry's image file, or None if it names none or the file is missing."""
    if not entry.image:
        return None
    path = Path(entry.image)
    path = path if path.is_absolute() else _REPO_ROOT / path
    return path if path.is_file() else None


def keypoints_path(index_path: Path) -> Path:
    """Where the index's keypoints are stored: ``<name>.keypoints.npz`` next to the index."""
    return index_path.with_name(index_path.stem + ".keypoints.npz")


def keypoints_fingerprint(index_path: Path, index) -> str:
    """Hash of the index, its images' sizes and times, and the ORB settings, to spot stale keypoints."""
    from app.extractors.keypoint_match import ORB_FEATURES, ORB_MAX_SIDE

    digest = hashlib.sha256(index_path.read_bytes())
    digest.update(f"orb {ORB_FEATURES} {ORB_MAX_SIDE}".encode())
    for entry in index:
        path = _entry_image(entry)
        stat = path.stat() if path else None
        digest.update(f"|{entry.id}|{entry.image}|{stat.st_size if stat else -1}|"
                      f"{stat.st_mtime_ns if stat else -1}".encode())
    return digest.hexdigest()


def compute_index_keypoints(index) -> dict[str, tuple]:
    """ORB features of every known image found on disk, keyed by entry id."""
    from PIL import Image

    from app.extractors.keypoint_match import orb_features

    features = {}
    for entry in index:
        path = _entry_image(entry)
        if path is not None:
            with Image.open(path) as img:
                features[entry.id] = orb_features(img.convert("RGB"))
    return features


@lru_cache(maxsize=4)
def index_keypoints(index_path_str: str, fingerprint: str) -> dict[str, tuple]:
    """The index's keypoints, loaded from the stored file if it is current, else computed and saved."""
    import numpy as np

    index_path = Path(index_path_str)
    stored = keypoints_path(index_path)
    try:
        with np.load(stored, allow_pickle=False) as data:
            if str(data["fingerprint"]) == fingerprint:
                return {str(i): (data[f"pts_{n}"], data[f"desc_{n}"]) for n, i in enumerate(data["ids"])}
    except (OSError, KeyError, ValueError):
        pass  # no stored keypoints yet, or unreadable: compute them
    features = compute_index_keypoints(load_index(index_path_str))
    arrays = {"fingerprint": np.array(fingerprint), "ids": np.array(list(features), dtype=str)}
    for n, (pts, desc) in enumerate(features.values()):
        arrays[f"pts_{n}"] = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
        arrays[f"desc_{n}"] = desc if desc is not None else np.zeros((0, 32), dtype=np.uint8)
    tmp = stored.with_name(stored.name + ".tmp")
    try:
        with open(tmp, "wb") as fh:
            np.savez(fh, **arrays)
        tmp.replace(stored)
    except OSError:
        pass  # a read-only folder: keep them in memory for this process
    return {i: (arrays[f"pts_{n}"], arrays[f"desc_{n}"]) for n, i in enumerate(features)}


def _aligned_distance(query, entry: IndexEntry, homography) -> int:
    """Hash distance between the upload aligned onto the entry's image and that image (64 if none)."""
    from PIL import Image

    from app.extractors.keypoint_match import aligned_distance
    from app.extractors.phash import HASH_BITS

    path = _entry_image(entry)
    if path is None:
        return HASH_BITS
    with Image.open(path) as img:
        return aligned_distance(query, img.convert("RGB"), homography)


def _keypoint_matches(image_bytes: bytes, index, min_inliers: int) -> list[tuple[int, IndexEntry]]:
    """Return (inliers, entry) for entries whose image lines up with the upload, best first.

    A match is only kept if the aligned upload is also within the hash threshold, so another photo
    of the same subject is not taken for a copy. Entries without an image file are skipped.
    """
    from PIL import Image

    from app.extractors.keypoint_match import orb_features, orb_homography

    index_path = _index_path()
    known = index_keypoints(str(index_path), keypoints_fingerprint(index_path, index))
    with Image.open(BytesIO(image_bytes)) as img:
        query_image = img.convert("RGB")
    query = orb_features(query_image)
    threshold = get_settings().phash_match_threshold
    found = []
    for entry in index:
        if entry.id not in known:
            continue
        inliers, homography = orb_homography(query, known[entry.id])
        if inliers >= min_inliers and homography is not None \
                and _aligned_distance(query_image, entry, homography) <= threshold:
            found.append((inliers, entry))
    return sorted(found, key=lambda pair: (-pair[0], pair[1].id))
