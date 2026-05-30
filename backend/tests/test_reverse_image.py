"""Tests for the cached reverse-image extractor (ADR-007).

These exercise the three-way status that the recycled-context flag depends on (ADR-009),
using a small temporary fixture so the test is independent of the committed fixture's
contents. The all-important assertion is that an unknown image is NOT_ASSESSED, never CLEAR.
"""

import json

import pytest

from app.config import get_settings
from app.extractors import reverse_image
from app.extractors.reverse_image import find_web_matches
from app.models import FlagStatus


@pytest.fixture
def fixture_cache(tmp_path, monkeypatch):
    data = {
        "matches": {
            "recycled.jpg": [
                {
                    "url": "https://news.example.com/2019/story",
                    "title": "Old story",
                    "published_date": "2019-03-04",
                    "context": "Original 2019 coverage.",
                }
            ],
            "known_clean.jpg": [],
        }
    }
    path = tmp_path / "cache.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    get_settings.cache_clear()
    monkeypatch.setenv("AEGIS_REVERSE_IMAGE_MODE", "cache")
    monkeypatch.setenv("AEGIS_REVERSE_IMAGE_CACHE_PATH", str(path))
    get_settings.cache_clear()
    reverse_image._load_cache.cache_clear()
    yield
    get_settings.cache_clear()
    reverse_image._load_cache.cache_clear()


def test_image_with_matches_fires(fixture_cache):
    r = find_web_matches("recycled.jpg")
    assert r.status == FlagStatus.FIRED
    assert len(r.matches) == 1
    assert r.matches[0].published_date == "2019-03-04"


def test_known_image_no_matches_is_clear(fixture_cache):
    r = find_web_matches("known_clean.jpg")
    assert r.status == FlagStatus.CLEAR
    assert r.matches == []


def test_unknown_image_is_not_assessed_not_clear(fixture_cache):
    """The load-bearing distinction: an image absent from the cache could NOT be checked,
    so it is NOT_ASSESSED — never CLEAR, which would be a false 'nothing recycled' (ADR-009)."""
    r = find_web_matches("never_seen.jpg")
    assert r.status == FlagStatus.NOT_ASSESSED
    assert r.matches == []


def test_no_source_ref_is_not_assessed(fixture_cache):
    assert find_web_matches(None).status == FlagStatus.NOT_ASSESSED


def test_api_mode_is_not_assessed(tmp_path, monkeypatch):
    """Until a live API is wired (ADR-007), 'api' mode must say NOT_ASSESSED, not pretend."""
    get_settings.cache_clear()
    monkeypatch.setenv("AEGIS_REVERSE_IMAGE_MODE", "api")
    get_settings.cache_clear()
    r = find_web_matches("recycled.jpg")
    assert r.status == FlagStatus.NOT_ASSESSED
    get_settings.cache_clear()


def test_committed_fixture_is_loadable_and_matches_contract():
    """Smoke test the real committed fixture: it parses and its entries satisfy WebMatch."""
    get_settings.cache_clear()
    reverse_image._load_cache.cache_clear()
    # Default mode is cache; the packaged fixture has a known recycled entry.
    r = find_web_matches("flood_recycled_2019.jpg")
    assert r.status == FlagStatus.FIRED
    assert all(m.url for m in r.matches)
    # And a known-empty entry reads as CLEAR.
    assert find_web_matches("consistent_sunset.jpg").status == FlagStatus.CLEAR
    get_settings.cache_clear()
    reverse_image._load_cache.cache_clear()
