"""Tests for the reverse-image lookup, which matches on image content.

Most tests use a temporary index; a few check the committed index against the committed
images. A lookup that couldn't run must be NOT_ASSESSED, never CLEAR.
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
    """A made-up photo-like image (smooth texture plus a block), different for each seed."""
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
    reverse_image.index_keypoints.cache_clear()


@pytest.fixture
def use_index(tmp_path, monkeypatch):
    """Point the extractor at a temp index holding ``entries``; extra kwargs become AEGIS_ env vars."""

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
    """A smaller, re-saved JPEG copy is still found."""
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
    assert result.phash is not None  # the lookup did run


def test_unreadable_image_is_not_assessed(use_index):
    use_index([_entry("orig", phash_of_image(_photo(1)))])
    result = find_web_matches(b"definitely not an image")
    assert result.status == FlagStatus.NOT_ASSESSED
    assert result.phash is None


def test_missing_index_is_not_assessed_not_clear(tmp_path, monkeypatch):
    """No index means the lookup didn't run, so CLEAR would be misleading."""
    monkeypatch.setenv("AEGIS_IMAGE_INDEX_PATH", str(tmp_path / "does_not_exist.json"))
    _reset()
    result = find_web_matches(_bytes(_photo(1)))
    assert result.status == FlagStatus.NOT_ASSESSED
    assert "index could not be read" in result.detail
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
    path = use_index([bad_entry])
    result = find_web_matches(_bytes(_photo(1)))
    assert result.status == FlagStatus.NOT_ASSESSED
    assert "index could not be read" in result.detail  # plain message for the user
    with pytest.raises(reverse_image.HistoryIndexError, match=reason):  # exact problem from validation
        reverse_image.load_index(str(path))


def test_duplicate_entry_ids_are_rejected(use_index):
    use_index([_entry("same", "0" * 16), _entry("same", "f" * 16)])
    assert find_web_matches(_bytes(_photo(1))).status == FlagStatus.NOT_ASSESSED


# --- matching behaviour ----------------------------------------------------------------------


def test_threshold_is_inclusive_and_exact(use_index):
    """Distance equal to the threshold matches, one more does not (mirror lookup off)."""
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
    # Check plain pHash really misses the flip, otherwise this test shows nothing.
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


def test_unknown_mode_is_not_assessed(use_index):
    """An unknown mode (here the old name 'api') gives NOT_ASSESSED."""
    use_index([_entry("orig", phash_of_image(_photo(1)))], reverse_image_mode="api")
    assert find_web_matches(_bytes(_photo(1))).status == FlagStatus.NOT_ASSESSED


def test_legacy_cache_mode_is_an_alias_for_the_index(use_index):
    """Older .env files say 'cache', which should still work."""
    img = _photo(1)
    use_index([_entry("orig", phash_of_image(img))], reverse_image_mode="cache")
    assert find_web_matches(_bytes(img)).status == FlagStatus.FIRED


# --- the committed index ---------------------------------------------------------------------


def test_committed_index_matches_the_committed_illustrative_images():
    """The committed index still matches the committed images.

    flood and protest are in the index and match exactly; sunset and cat are not, so they come
    back clear. If pHash output changes, rebuild with scripts/build_image_index.py.
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
    """Every dataset A photo is in the index with the date and source from the manifest.

    The photos aren't in the repository; if get_dataset_a.py has fetched them, hashes are checked too.
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
    """Sharp random blocks, so ORB has lots of corners; different for each seed."""
    rng = np.random.default_rng(seed)
    blocks = (rng.random((size[1] // 8, size[0] // 8, 3)) * 255).astype(np.uint8)
    return Image.fromarray(blocks).resize(size, Image.Resampling.NEAREST)


def _entry_with_image(tmp_path, entry_id: str, img: Image.Image) -> dict:
    path = tmp_path / f"{entry_id}.png"
    img.save(path)
    return _entry(entry_id, phash_of_image(img), image=str(path))


def test_screenshot_copy_is_found_by_keypoints_when_the_hash_misses(use_index, tmp_path):
    """A screenshot frame beats the hash but keypoint matching still finds it."""
    original = _textured(1)
    shot = screenshot(original)
    assert hamming(phash_of_image(shot), phash_of_image(original)) > 10  # hash alone misses it
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
    """With no image file there is no keypoint match, and no crash."""
    original = _textured(1)
    use_index([_entry("orig", phash_of_image(original))])  # no "image" field
    assert find_web_matches(_bytes(screenshot(original))).status == FlagStatus.CLEAR


@pytest.mark.parametrize("failure", [TimeoutError("slow"), RuntimeError("opencv unavailable")])
def test_keypoint_stage_failure_is_not_assessed_never_clear(use_index, tmp_path, monkeypatch, failure):
    """If the keypoint stage fails, a framed copy may have been missed, so it can't say clear."""
    use_index([_entry_with_image(tmp_path, "orig", _textured(1))])

    def boom(*_args):
        raise failure

    monkeypatch.setattr(reverse_image, "_keypoint_matches", boom)
    result = find_web_matches(_bytes(screenshot(_textured(1))))
    assert result.status == FlagStatus.NOT_ASSESSED
    assert result.detail == ("The search for cropped or framed copies of this image could not run, so an earlier "
                             "copy may have been missed.")


@pytest.mark.parametrize("mode, reason", [
    ("off", "The search for earlier copies of this image is switched off."),
    ("sideways", "The search for earlier copies of this image is not set up correctly, so it did not run."),
])
def test_a_lookup_that_is_off_or_unavailable_says_so_in_plain_words(use_index, mode, reason):
    use_index([], reverse_image_mode=mode)
    result = find_web_matches(_bytes(_textured(1)))
    assert result.status == FlagStatus.NOT_ASSESSED and result.detail == reason


def test_an_unreadable_index_says_so_in_plain_words(tmp_path, monkeypatch):
    monkeypatch.setenv("AEGIS_IMAGE_INDEX_PATH", str(tmp_path / "missing.json"))
    _reset()
    result = find_web_matches(_bytes(_textured(1)))
    assert result.status == FlagStatus.NOT_ASSESSED
    assert result.detail.startswith("The image history index could not be read") and "missing.json" not in result.detail
    _reset()


def test_keypoint_matches_are_one_to_one():
    """Many query points can't all match the few keypoints of a low-texture image.

    skimage's blurry `clock` has about 10 keypoints. Before matching was one-to-one, an unrelated
    `coins` image with a border got 32 inliers against it, which was a false match.
    """
    data = pytest.importorskip("skimage.data")
    clock = orb_features(Image.fromarray(data.clock()).convert("RGB"))
    coins = orb_features(border(Image.fromarray(data.coins()).convert("RGB")))
    inliers = orb_inliers(coins, clock)
    assert inliers <= len(clock[0])
    assert inliers < get_settings().keypoint_min_inliers


def test_screenshot_of_the_committed_flood_image_is_found_by_keypoints():
    """With the committed index, a screenshot of the flood image is still found."""
    _reset()
    data = (_REPO / "data" / "illustrative" / "flood_illustrative.png").read_bytes()
    result = find_web_matches(_bytes(screenshot(Image.open(BytesIO(data)))))
    assert result.status == FlagStatus.FIRED and result.method == "keypoints"
    assert result.matches[0].published_date == "2019-03-04"
    _reset()


# --- the known images' keypoints, stored next to the index ------------------------------------


def _count_computations(monkeypatch) -> list:
    computed = []
    real = reverse_image.compute_index_keypoints
    monkeypatch.setattr(reverse_image, "compute_index_keypoints", lambda index: computed.append(1) or real(index))
    return computed


def test_keypoints_are_computed_once_and_then_read_from_the_stored_file(use_index, tmp_path, monkeypatch):
    original = _textured(1)
    index = use_index([_entry_with_image(tmp_path, "orig", original),
                       _entry_with_image(tmp_path, "other", _textured(2))])
    computed = _count_computations(monkeypatch)
    shot = _bytes(screenshot(original))
    first = find_web_matches(shot)
    stored = reverse_image.keypoints_path(index)
    assert stored == tmp_path / "index.keypoints.npz" and stored.is_file()
    assert computed == [1]
    reverse_image.index_keypoints.cache_clear()  # like a fresh process
    again = find_web_matches(shot)
    assert computed == [1]  # read from the file, not recomputed
    assert again.method == "keypoints"
    assert [m.keypoint_inliers for m in again.matches] == [m.keypoint_inliers for m in first.matches]


def test_the_stored_keypoints_are_exactly_those_of_the_image(use_index, tmp_path):
    original = _textured(3)
    use_index([_entry_with_image(tmp_path, "orig", original)])
    find_web_matches(_bytes(screenshot(original)))
    points, descriptors = orb_features(Image.open(tmp_path / "orig.png").convert("RGB"))
    with np.load(tmp_path / "index.keypoints.npz") as data:
        assert list(data["ids"]) == ["orig"]
        assert np.array_equal(data["pts_0"], points) and np.array_equal(data["desc_0"], descriptors)


def test_keypoints_are_computed_again_only_when_the_index_or_an_image_changes(use_index, tmp_path, monkeypatch):
    original = _textured(1)
    entries = [_entry_with_image(tmp_path, "orig", original)]
    use_index(entries)
    computed = _count_computations(monkeypatch)
    shot = _bytes(screenshot(original))
    find_web_matches(shot)
    assert computed == [1]

    reverse_image.index_keypoints.cache_clear()
    find_web_matches(shot)
    assert computed == [1]  # nothing changed

    use_index(entries + [_entry_with_image(tmp_path, "other", _textured(2))])  # the index changed
    find_web_matches(shot)
    assert computed == [1, 1]

    _textured(4).save(tmp_path / "other.png")  # an image it names changed
    reverse_image.index_keypoints.cache_clear()
    find_web_matches(shot)
    assert computed == [1, 1, 1]


def test_a_different_picture_sharing_a_region_is_not_taken_for_a_copy(use_index, tmp_path):
    """Two pictures sharing one region can match on keypoints, so a keypoint match is then
    checked by aligning the upload and hashing the covered region. A real copy still passes."""
    original = _textured(1)
    other = _textured(9)
    other.paste(original.crop((40, 30, 280, 210)), (40, 30))  # the shared region
    use_index([_entry_with_image(tmp_path, "orig", original)])
    query = orb_features(screenshot(other).convert("RGB"))
    known = orb_features(original.convert("RGB"))
    assert orb_inliers(query, known) >= get_settings().keypoint_min_inliers  # keypoints alone would match
    assert find_web_matches(_bytes(screenshot(other))).status == FlagStatus.CLEAR
    assert find_web_matches(_bytes(screenshot(original))).method == "keypoints"  # a real copy still matches
