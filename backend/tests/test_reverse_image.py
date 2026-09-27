"""Tests for the content-matched reverse-image lookup.

Every lookup outcome is exercised against a temporary history index, so the tests do not depend
on the committed index's contents — except the last test, which exists to guard exactly that
committed index against drifting away from the committed illustrative images.

The load-bearing assertions carry over from the filename era: a lookup that could not
run is NOT_ASSESSED, never CLEAR. What is new is that matching follows the picture, not the name:
a recompressed, resized copy still matches, and an unrelated image does not.
"""

import json
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageOps

from app.config import get_settings
from app.extractors import reverse_image
from app.extractors.phash import hamming, phash_of_image
from app.extractors.reverse_image import find_web_matches
from app.models import FlagStatus

_REPO = Path(__file__).resolve().parents[2]
_SOURCE = {
    "url": "https://news.example.com/2019/story",
    "title": "Old story",
    "published_date": "2019-03-04",
    "context": "Original 2019 coverage.",
}


def _photo(seed: int, size=(320, 240)) -> Image.Image:
    """A synthetic photo-like image: smooth seeded texture plus a block, distinct per seed."""
    rng = np.random.default_rng(seed)
    small = (rng.random((size[1] // 20, size[0] // 20, 3)) * 255).astype(np.uint8)
    img = Image.fromarray(small).resize(size, Image.Resampling.BICUBIC)
    x, y = int(rng.integers(20, size[0] - 120)), int(rng.integers(20, size[1] - 100))
    img.paste((int(rng.integers(0, 255)),) * 3, (x, y, x + 100, y + 80))
    return img


def _bytes(img: Image.Image, fmt="PNG", **kw) -> bytes:
    buf = BytesIO()
    img.convert("RGB").save(buf, format=fmt, **kw)
    return buf.getvalue()


def _entry(entry_id: str, phash: str, **overrides) -> dict:
    entry = {"id": entry_id, "phash": phash, "earliest_date": "2019-03-04", "sources": [_SOURCE]}
    entry.update(overrides)
    return entry


def _flip_bits(hash_hex: str, n: int) -> str:
    """A hash exactly n bits away from ``hash_hex`` (flips the n lowest bits)."""
    return f"{int(hash_hex, 16) ^ ((1 << n) - 1):016x}"


def _reset():
    get_settings.cache_clear()
    reverse_image.load_index.cache_clear()


@pytest.fixture
def use_index(tmp_path, monkeypatch):
    """Point the extractor at a temporary index holding ``entries``; extra kwargs set env."""

    def _use(entries, **env):
        path = tmp_path / "index.json"
        path.write_text(json.dumps({"entries": entries}), encoding="utf-8")
        monkeypatch.setenv("AEGIS_REVERSE_IMAGE_MODE", "index")
        monkeypatch.setenv("AEGIS_IMAGE_INDEX_PATH", str(path))
        for key, value in env.items():
            monkeypatch.setenv(f"AEGIS_{key.upper()}", str(value))
        _reset()
        return path

    yield _use
    _reset()


# --- the three lookup outcomes ---------------------------------------------------------------


def test_recompressed_resized_copy_still_matches(use_index):
    """The point of content matching: a re-saved copy is found by content, where a filename lookup failed."""
    original = _photo(1)
    use_index([_entry("orig", phash_of_image(original))])
    copy = original.resize((160, 120), Image.Resampling.LANCZOS)
    result = find_web_matches(_bytes(copy, "JPEG", quality=70))
    assert result.status == FlagStatus.FIRED
    assert result.matches[0].published_date == "2019-03-04"
    assert result.matches[0].hash_distance is not None
    assert result.matches[0].hash_distance <= get_settings().phash_match_threshold
    assert result.phash is not None and len(result.phash) == 16


def test_unrelated_image_is_clear(use_index):
    use_index([_entry("orig", phash_of_image(_photo(1)))])
    result = find_web_matches(_bytes(_photo(2)))
    assert result.status == FlagStatus.CLEAR
    assert result.matches == []
    assert result.phash is not None  # the lookup ran: the image was hashed and compared


def test_unreadable_image_is_not_assessed(use_index):
    use_index([_entry("orig", phash_of_image(_photo(1)))])
    result = find_web_matches(b"definitely not an image")
    assert result.status == FlagStatus.NOT_ASSESSED
    assert result.phash is None


def test_missing_index_is_not_assessed_not_clear(tmp_path, monkeypatch):
    """No index means the lookup did not run — reporting CLEAR would be false reassurance."""
    monkeypatch.setenv("AEGIS_IMAGE_INDEX_PATH", str(tmp_path / "does_not_exist.json"))
    _reset()
    result = find_web_matches(_bytes(_photo(1)))
    assert result.status == FlagStatus.NOT_ASSESSED
    assert "unusable" in result.detail
    _reset()


@pytest.mark.parametrize(
    "bad_entry, reason",
    [
        (_entry("x", "not-a-hash"), "phash"),
        (_entry("x", "0" * 16, earliest_date="2001-01-01"), "earliest dated source"),
        (_entry("x", "0" * 16, sources=[]), "at least one source"),
        (_entry("x", "0" * 16, earliest_date="yesterday"), "ISO date"),
    ],
)
def test_invalid_index_is_not_assessed(use_index, bad_entry, reason):
    use_index([bad_entry])
    result = find_web_matches(_bytes(_photo(1)))
    assert result.status == FlagStatus.NOT_ASSESSED
    assert reason in result.detail


def test_duplicate_entry_ids_are_rejected(use_index):
    use_index([_entry("same", "0" * 16), _entry("same", "f" * 16)])
    assert find_web_matches(_bytes(_photo(1))).status == FlagStatus.NOT_ASSESSED


# --- matching behaviour ----------------------------------------------------------------------


def test_threshold_is_inclusive_and_exact(use_index):
    """Distance == threshold matches; threshold + 1 does not. Mirror lookup off to isolate it."""
    img = _photo(3)
    h = phash_of_image(img)
    threshold = 10
    use_index([_entry("at", _flip_bits(h, threshold))],
              phash_match_threshold=threshold, phash_mirror_lookup="false")
    at = find_web_matches(_bytes(img))
    assert at.status == FlagStatus.FIRED and at.best_distance == threshold

    use_index([_entry("past", _flip_bits(h, threshold + 1))],
              phash_match_threshold=threshold, phash_mirror_lookup="false")
    assert find_web_matches(_bytes(img)).status == FlagStatus.CLEAR


def test_mirror_lookup_finds_a_flipped_copy(use_index):
    original = _photo(4)
    flipped = ImageOps.mirror(original)
    h = phash_of_image(original)
    # Precondition: plain pHash really does lose the flip (else this test proves nothing).
    assert hamming(h, phash_of_image(flipped)) > 10

    use_index([_entry("orig", h)], phash_mirror_lookup="true")
    found = find_web_matches(_bytes(flipped))
    assert found.status == FlagStatus.FIRED and found.best_distance == 0

    use_index([_entry("orig", h)], phash_mirror_lookup="false")
    assert find_web_matches(_bytes(flipped)).status == FlagStatus.CLEAR


def test_every_matching_entry_is_reported_nearest_first(use_index):
    img = _photo(5)
    h = phash_of_image(img)
    later = {**_SOURCE, "url": "https://blog.example.net/2021/repost", "published_date": "2021-01-01"}
    use_index([
        _entry("near", _flip_bits(h, 3), earliest_date="2021-01-01", sources=[later]),
        _entry("exact", h),
    ], phash_mirror_lookup="false")
    result = find_web_matches(_bytes(img))
    assert [m.hash_distance for m in result.matches] == [0, 3]
    assert result.best_distance == 0


# --- modes -----------------------------------------------------------------------------------


def test_live_api_mode_is_not_assessed(use_index):
    """There is no live lookup yet, so 'api' mode must say NOT_ASSESSED, not pretend."""
    use_index([_entry("orig", phash_of_image(_photo(1)))], reverse_image_mode="api")
    assert find_web_matches(_bytes(_photo(1))).status == FlagStatus.NOT_ASSESSED


def test_legacy_cache_mode_is_an_alias_for_the_index(use_index):
    """Older .env files say 'cache'; they must keep working, not silently stop."""
    img = _photo(1)
    use_index([_entry("orig", phash_of_image(img))], reverse_image_mode="cache")
    assert find_web_matches(_bytes(img)).status == FlagStatus.FIRED


# --- the committed index ---------------------------------------------------------------------


def test_committed_index_matches_the_committed_illustrative_images():
    """Drift guard: the shipped index must still describe the shipped images.

    flood and protest are registered, so they match exactly; sunset and cat are not registered,
    so they demonstrate "lookup ran, no match". If a dependency changes how pHash is computed,
    this fails and the index must be rebuilt with scripts/build_image_index.py.
    """
    _reset()
    images = _REPO / "data" / "illustrative"
    index = reverse_image.load_index(str(reverse_image._DEFAULT_INDEX))
    by_id = {e.id: e for e in index}
    for filename, entry_id in [("flood_illustrative.png", "illustrative-flood"),
                               ("protest_illustrative.png", "illustrative-protest")]:
        data = (images / filename).read_bytes()
        assert by_id[entry_id].phash == phash_of_image(Image.open(BytesIO(data)))
        result = find_web_matches(data)
        assert result.status == FlagStatus.FIRED and result.best_distance == 0
    for filename in ("sunset_illustrative.png", "cat_illustrative.png"):
        assert find_web_matches((images / filename).read_bytes()).status == FlagStatus.CLEAR
    _reset()


def test_committed_index_holds_every_dataset_a_photo_with_its_date():
    """Every dataset A photo is an index entry carrying the manifest's date and source page.

    The photos themselves are not in the repository; where get_dataset_a.py has fetched them,
    each entry's hash must still describe its file.
    """
    import csv

    _reset()
    with open(_REPO / "data" / "labels" / "A_originals.csv", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 40
    by_id = {e.id: e for e in reverse_image.load_index(str(reverse_image._DEFAULT_INDEX))}
    for row in rows:
        entry = by_id[f"nasa-{row['nasa_id']}"]
        assert entry.earliest_date == row["earliest_date"]
        assert entry.sources[0].url == row["source_url"]
        assert entry.image == f"data/A_originals/{row['file']}"
        path = _REPO / entry.image
        if path.is_file():
            assert entry.phash == phash_of_image(Image.open(path))
    _reset()


# --- second stage: keypoint matching when the hash finds nothing ------------------------------

from app.extractors.keypoint_match import orb_features, orb_inliers  # noqa: E402
from tests.eval.image_transforms import border, screenshot  # noqa: E402


def _textured(seed: int, size=(320, 240)) -> Image.Image:
    """Sharp random blocks: plenty of corners for ORB, distinct per seed."""
    rng = np.random.default_rng(seed)
    blocks = (rng.random((size[1] // 8, size[0] // 8, 3)) * 255).astype(np.uint8)
    return Image.fromarray(blocks).resize(size, Image.Resampling.NEAREST)


def _entry_with_image(tmp_path, entry_id: str, img: Image.Image) -> dict:
    path = tmp_path / f"{entry_id}.png"
    img.save(path)
    return _entry(entry_id, phash_of_image(img), image=str(path))


def test_screenshot_copy_is_found_by_keypoints_when_the_hash_misses(use_index, tmp_path):
    """The point of the second stage: a screenshot frame defeats the hash, not the keypoints."""
    original = _textured(1)
    shot = screenshot(original)
    assert hamming(phash_of_image(shot), phash_of_image(original)) > 10  # the hash really misses
    use_index([_entry_with_image(tmp_path, "orig", original)])
    result = find_web_matches(_bytes(shot))
    assert result.status == FlagStatus.FIRED
    assert result.method == "keypoints"
    assert result.matches[0].keypoint_inliers >= get_settings().keypoint_min_inliers
    assert result.matches[0].hash_distance is None


def test_bordered_copy_is_found_by_keypoints(use_index, tmp_path):
    original = _textured(1)
    use_index([_entry_with_image(tmp_path, "orig", original)])
    assert find_web_matches(_bytes(border(original))).method == "keypoints"


def test_keypoint_matching_can_be_switched_off(use_index, tmp_path):
    original = _textured(1)
    use_index([_entry_with_image(tmp_path, "orig", original)], keypoint_matching="false")
    assert find_web_matches(_bytes(screenshot(original))).status == FlagStatus.CLEAR


def test_unrelated_framed_image_is_not_matched_by_keypoints(use_index, tmp_path):
    use_index([_entry_with_image(tmp_path, "orig", _textured(1))])
    result = find_web_matches(_bytes(screenshot(_textured(2))))
    assert result.status == FlagStatus.CLEAR
    assert "Keypoint matching found no" in result.detail


def test_entry_without_its_image_is_matched_by_hash_only(use_index):
    """No image to compare against means no keypoint match, and no crash."""
    original = _textured(1)
    use_index([_entry("orig", phash_of_image(original))])  # no "image" field
    assert find_web_matches(_bytes(screenshot(original))).status == FlagStatus.CLEAR


def test_keypoint_stage_failure_keeps_the_hash_result_and_says_why(use_index, tmp_path, monkeypatch):
    use_index([_entry_with_image(tmp_path, "orig", _textured(1))])

    def boom(*_args):
        raise RuntimeError("opencv unavailable")

    monkeypatch.setattr(reverse_image, "_keypoint_matches", boom)
    result = find_web_matches(_bytes(screenshot(_textured(1))))
    assert result.status == FlagStatus.CLEAR
    assert "Keypoint matching could not run" in result.detail


def test_keypoint_matches_are_one_to_one():
    """Regression: many query points must not pile onto the few keypoints of a low-texture image.

    `clock` (a motion-blurred photo, openly licensed in scikit-image) yields about 10 keypoints.
    Before matches were made one-to-one, an unrelated bordered `coins` image scored 32 inliers
    against it, above the threshold: a false match. One-to-one matching cannot exceed the known
    image's own keypoint count.
    """
    data = pytest.importorskip("skimage.data")
    clock = orb_features(Image.fromarray(data.clock()).convert("RGB"))
    coins = orb_features(border(Image.fromarray(data.coins()).convert("RGB")))
    inliers = orb_inliers(coins, clock)
    assert inliers <= len(clock[0])
    assert inliers < get_settings().keypoint_min_inliers


def test_screenshot_of_the_committed_flood_image_is_found_by_keypoints():
    """End-to-end on the shipped index: a screenshot of a registered image is still found."""
    _reset()
    data = (_REPO / "data" / "illustrative" / "flood_illustrative.png").read_bytes()
    result = find_web_matches(_bytes(screenshot(Image.open(BytesIO(data)))))
    assert result.status == FlagStatus.FIRED and result.method == "keypoints"
    assert result.matches[0].published_date == "2019-03-04"
    _reset()
