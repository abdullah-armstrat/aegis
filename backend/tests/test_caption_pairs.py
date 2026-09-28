"""The default caption check (picture vs caption, CLIP ViT-B/32) on 16 dataset A photos.

Each photo gets its own NASA caption (expected clear) and one from another topic (expected to
fire). 13 of 16 come out as expected; the pins record what the check actually does. Skips when
dataset A is not downloaded."""

import json
import os
from pathlib import Path

import pytest

from app.config import get_settings

BACKEND = Path(__file__).resolve().parents[1]
PHOTOS = BACKEND.parent / "data" / "A_originals"
CASES = json.loads((BACKEND / "tests" / "eval" / "test_set" / "caption_pairs_A.json").read_text(encoding="utf-8"))["cases"]

# (status, CLIP similarity) per case, measured by backend/scripts/build_caption_pairs.py score.
PINNED = {
    "KSC-2009-1011-own": ("fired", 0.2648),
    "KSC-2009-1011-other": ("fired", 0.1585),
    "KSC-2011-2584-own": ("clear", 0.3786),
    "KSC-2011-2584-other": ("clear", 0.2787),
    "S77-28200-own": ("clear", 0.3317),
    "S77-28200-other": ("fired", 0.1987),
    "KSC-98PC-629-own": ("clear", 0.3964),
    "KSC-98PC-629-other": ("fired", 0.199),
    "KSC-07pd0148-own": ("clear", 0.3574),
    "KSC-07pd0148-other": ("fired", 0.1776),
    "KSC-2010-5875-own": ("clear", 0.3374),
    "KSC-2010-5875-other": ("fired", 0.1187),
    "sts059-219-065-own": ("fired", 0.2745),
    "sts059-219-065-other": ("fired", 0.1457),
    "sts054-152-189-own": ("clear", 0.327),
    "sts054-152-189-other": ("fired", 0.153),
}


def test_the_cases_follow_the_selection_rule():
    assert len(CASES) == 16 and {c["id"] for c in CASES} == set(PINNED)
    topics = [c["topic"] for c in CASES if c["expected"] == "clear"]
    assert {t: topics.count(t) for t in topics} == {"weather": 2, "transport": 2, "animals": 2, "landscapes": 2}
    for c in CASES:
        own = c["caption_from"] == c["nasa_id"]
        assert own == (c["expected"] == "clear")
        if not own:
            other = next(x for x in CASES if x["nasa_id"] == c["caption_from"])
            assert other["topic"] != c["topic"]


@pytest.mark.slow
@pytest.mark.skipif(not PHOTOS.is_dir(), reason="dataset A is not on this machine")
def test_the_default_caption_check_on_real_pictures_is_pinned(monkeypatch):
    from app.adapters.image_adapter import build_bundle
    from app.fusion.scorecard import build_scorecard

    for key in [k for k in os.environ if k.startswith("AEGIS_")]:
        monkeypatch.delenv(key)
    get_settings.cache_clear()
    assert get_settings().caption_match_method == "image"
    met = 0
    for case in CASES:
        bundle = build_bundle((PHOTOS / case["file"]).read_bytes(), caption=case["caption"], source_ref=case["file"])
        flag = next(f for f in build_scorecard(bundle).flags if f.type.value == "caption_content_mismatch")
        status, similarity = PINNED[case["id"]]
        assert flag.status.value == status, case["id"]
        assert bundle.caption_match.image_similarity == pytest.approx(similarity, abs=0.002), case["id"]
        met += flag.status.value == {"clear": "clear", "fire": "fired"}[case["expected"]]
    assert met == 13
    get_settings.cache_clear()
