"""Tests for the dev input-hash cache helper."""

from app.extractors.cache import JsonCache


def test_roundtrip_and_miss(tmp_path):
    cache = JsonCache("t", root=tmp_path)
    key = cache.key("prompt text", "phi3:mini")

    assert cache.get(key) is None  # cold miss
    cache.set(key, {"answer": "same thing", "n": 1})
    assert cache.get(key) == {"answer": "same thing", "n": 1}


def test_key_is_input_sensitive(tmp_path):
    cache = JsonCache("t", root=tmp_path)
    assert cache.key("a", "m") == cache.key("a", "m")
    assert cache.key("a", "m") != cache.key("a", "m2")
    assert cache.key("a", "m") != cache.key("b", "m")


def test_corrupt_entry_is_a_miss(tmp_path):
    cache = JsonCache("t", root=tmp_path)
    key = cache.key("x")
    (tmp_path / "t" / f"{key}.json").write_text("{not json", encoding="utf-8")
    assert cache.get(key) is None
